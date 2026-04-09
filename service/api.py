"""Service API layer - write operations for ContextEngine.

This is the ONLY layer where RequestContext is mandatory and account_id is injected.
All external calls must provide a RequestContext for multi-tenant isolation.

Note: dev branch is WRITE-ONLY. Read operations (ReadAPI) are in phase1 branch.
See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

from dataclasses import dataclass
from typing import Optional

from core.logging_config import get_logger

logger = get_logger(__name__)
from core.interfaces import ContextFS, LLM, CandidateExtractor
from core.models import RequestContext, CandidateMemory, WritePlan
from commit import ContextWriter, CandidatePipeline, OutboxStore
from extraction import Extractor


class MemoryWriteAPI:
    """Public API for memory write operations.

    All methods enforce multi-tenant isolation via RequestContext.
    Writes are orchestrated through the commit pipeline:
    1. Extract candidates via CandidateExtractor
    2. Plan write actions via MergePolicy
    3. Build ContextNode via ArchiveBuilder
    4. Write to storage via ContextFS
    5. Register OutboxEvents for async indexing
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        outbox_store: Optional[OutboxStore] = None,
    ):
        """Initialize the API with required dependencies.

        Args:
            fs: ContextFS implementation for persisting nodes
            llm: LLM instance for extraction
            outbox_store: OutboxStore for registering index events (optional)
        """
        self._fs = fs
        self._llm = llm
        self._outbox_store = outbox_store

        # Initialize write components
        self._writer = ContextWriter(fs, llm=self._llm, outbox_store=outbox_store)
        self._pipeline = CandidatePipeline()
        self._pipeline.set_extractors(self._create_extractors())

    def _create_extractors(self) -> list[CandidateExtractor]:
        """Create default extractors for the pipeline.

        Returns:
            List of CandidateExtractor instances (single tool-use Extractor)
        """
        return [Extractor(self._llm)]

    def commit_session(
        self,
        messages: list[dict],
        ctx: RequestContext,
        confidence_threshold: float = 0.5,
        wait: bool = True,
    ) -> dict:
        """Commit a conversation session to memory.

        This is the main entry point for writing memories.
        Extracts candidates from messages, filters by confidence,
        and writes to storage.

        Args:
            messages: List of message dicts with "role" and "content"
                      Example: [{"role": "user", "content": "..."}, ...]
            ctx: RequestContext for this operation
            confidence_threshold: Minimum confidence for writing (default 0.5)
            wait: If True, block until extraction completes (default).
                  If False, return immediately with task_id for async processing.

        Returns:
            Dict with write results:
            {
                "candidates_extracted": int,
                "candidates_filtered": int,
                "writes_completed": int,
                "writes_skipped": int,
                "writes_failed": int,
                "plans": list[WritePlan dict],
                "task_id": str (only if wait=False),
                "status": "processing" (only if wait=False)
            }
        """
        # Step 1: Extract candidates
        candidates = self._pipeline.extract(messages, ctx)

        # Step 2: Filter by confidence
        filtered = self._pipeline.filter_by_confidence(candidates, confidence_threshold)

        # Step 3: Deduplicate
        deduplicated = self._pipeline.deduplicate(filtered)

        # Step 4: Write candidates (ContextWriter handles outbox registration internally)
        plans = self._writer.write_candidates(deduplicated, ctx)

        # Compile results
        writes_completed = sum(1 for p in plans if p.action != "skip")
        writes_skipped = sum(1 for p in plans if p.action == "skip")
        writes_failed = len(deduplicated) - writes_completed - writes_skipped

        result = {
            "candidates_extracted": len(candidates),
            "candidates_filtered": len(candidates) - len(filtered),
            "writes_completed": writes_completed,
            "writes_skipped": writes_skipped,
            "writes_failed": writes_failed,
            "plans": [
                {
                    "action": p.action,
                    "target_uri": p.target_uri,
                    "merged_fields": p.merged_fields,
                }
                for p in plans
            ],
        }

        # For async mode, return task_id
        if not wait:
            # In a full async implementation, this would return a real task_id
            # For now, we return a placeholder since we're doing synchronous writes
            import uuid
            result["task_id"] = str(uuid.uuid4())
            result["status"] = "completed"  # Already completed since we write synchronously

        return result

    def write_memory(
        self,
        candidate: CandidateMemory,
        ctx: RequestContext,
    ) -> dict:
        """Write a single candidate memory.

        Bypasses extraction - use when you already have a CandidateMemory.

        Args:
            candidate: CandidateMemory to write
            ctx: RequestContext for this operation

        Returns:
            Dict with write result:
            {
                "action": str,
                "target_uri": str,
                "merged_fields": dict,
            }
        """
        plan = self._writer.write_candidate(candidate, ctx)

        return {
            "action": plan.action,
            "target_uri": plan.target_uri,
            "merged_fields": plan.merged_fields,
        }

    def write_memories(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext,
        parallel: bool = True,
    ) -> list[dict]:
        """Write multiple candidate memories.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations
            parallel: If True, write in parallel (default True)

        Returns:
            List of write result dicts
        """
        # Deduplicate first
        deduplicated = self._pipeline.deduplicate(candidates)

        # Write (ContextWriter handles outbox registration internally)
        if parallel:
            plans = self._writer.write_candidates_parallel(deduplicated, ctx)
        else:
            plans = self._writer.write_candidates(deduplicated, ctx)

        return [
            {
                "action": p.action,
                "target_uri": p.target_uri,
                "merged_fields": p.merged_fields,
            }
            for p in plans
        ]

    def poll_task(self, task_id: str, ctx: RequestContext) -> "TaskStatus":
        """Poll the status of an async extraction task.

        Args:
            task_id: Task identifier returned from commit_session(wait=False)
            ctx: RequestContext for this operation

        Returns:
            TaskStatus object with current status
        """
        # For now, return completed status since we're doing synchronous writes
        # In a full async implementation, this would query the task store
        return TaskStatus(
            task_id=task_id,
            status="completed",
            memories_extracted=0,
        )


@dataclass
class TaskStatus:
    """Status of an async extraction task."""
    task_id: str
    status: str  # pending|processing|completed|failed
    memories_extracted: int = 0
    error: str | None = None


# Singleton instances for simple usage
# In production, use dependency injection
_default_write_api: Optional[MemoryWriteAPI] = None


def init_write_api(
    fs: ContextFS,
    llm: LLM,
    outbox_store: Optional[OutboxStore] = None,
) -> MemoryWriteAPI:
    """Initialize the global write API instance.

    Args:
        fs: ContextFS implementation
        llm: LLM instance for extraction
        outbox_store: Optional OutboxStore for async indexing

    Returns:
        Configured MemoryWriteAPI instance
    """
    global _default_write_api
    _default_write_api = MemoryWriteAPI(fs, llm, outbox_store)
    return _default_write_api


def get_write_api() -> Optional[MemoryWriteAPI]:
    """Get the global write API instance.

    Returns:
        MemoryWriteAPI if initialized, None otherwise
    """
    return _default_write_api


# ---------------------------------------------------------------------------
# Read / Search API
# ---------------------------------------------------------------------------

from core.errors import AccessDeniedError, ValidationError as CoreValidationError
from core.models import (
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
)
from retrieval.pipeline import RetrievalPipeline
from retrieval.context_reader import ContextReader


class ReadAPI:
    """Public API for memory search and read operations.

    Exposes two tools consumed by AI agents:
      - search_memory: semantic retrieval -> structured SearchMemoryResult
      - read_memory: URI-based read -> RetrievedBlock with full content
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        read_service: ContextReader | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._read_service = read_service
        self._cfg = config or RetrievalConfig()

    def search_memory(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int = 10,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        session_archive: dict | None = None,
        score_threshold: float | None = None,
        include_debug: bool = False,
        mode: str = RetrieverMode.QUICK,
        fill_content_for_top_k: int = 0,
    ) -> SearchMemoryResult:
        if not (query or "").strip():
            raise CoreValidationError("query", "query must not be empty")
        if top_k > self._cfg.max_top_k:
            raise CoreValidationError("top_k", f"top_k={top_k} exceeds max {self._cfg.max_top_k}")

        if target_uri:
            prefix = f"ctx://{ctx.account_id}/"
            if target_uri.startswith("ctx://") and not target_uri.startswith(prefix):
                raise AccessDeniedError(target_uri, ctx.account_id, "target_uri account mismatch")

        result = self._pipeline.run(
            query, ctx,
            top_k=top_k,
            categories=categories,
            target_uri=target_uri,
            session_archive=session_archive,
            score_threshold=score_threshold,
            mode=mode,
            fill_content_for_top_k=fill_content_for_top_k,
        )

        if not include_debug:
            result.trace = None

        return result

    def read_memory(
        self,
        uri: str,
        ctx: RequestContext,
    ) -> RetrievedBlock:
        """Read L2 md file content by URI.

        Since search_memory already returns abstract in results,
        read_memory only reads the actual md file content.
        """
        return self._read_service.read(uri, ctx=ctx)
