"""Service API layer - external interface for ContextEngine.

This is the ONLY layer where RequestContext is mandatory and account_id is injected.
All external calls must provide a RequestContext for multi-tenant isolation.

See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

from typing import Optional, Literal

from core.errors import AccessDeniedError, NodeNotFoundError
from core.logging_config import get_logger

logger = get_logger(__name__)
from core.interfaces import ContextFS, VectorIndex, LLM, CandidateExtractor
from core.models import RequestContext, SeedHit, CandidateMemory, WritePlan
from retrieval import Planner, L0Retriever, MemoryReadService, AssemblyService, RetrievedBlock, RetrievalResult
from commit import ContextWriter, CandidatePipeline, OutboxStore
from extraction import ProfileExtractor, PreferenceExtractor, EntityEventExtractor, ProcedureExtractor


Level = Literal["L0", "L1", "L2"]


class MemoryReadAPI:
    """Public API for memory search and read operations.

    All methods enforce multi-tenant isolation via RequestContext.
    account_id is injected from context, cannot be overridden by callers.
    """

    def __init__(self, fs: ContextFS, vector_index: VectorIndex):
        """Initialize the API with required dependencies.

        Args:
            fs: ContextFS implementation for reading nodes
            vector_index: VectorIndex implementation for similarity search
        """
        self._fs = fs
        self._vector_index = vector_index

        # Initialize retrieval components
        self._planner = Planner()
        self._l0_retriever = L0Retriever(vector_index)
        self._read_service = MemoryReadService(fs)
        self._assembly = AssemblyService(fs)

    def search_memory(
        self,
        query: str,
        ctx: RequestContext,
        top_k: int = 10,
        category: Optional[list[str]] = None,
        include_overview: bool = False,
    ) -> list[dict]:
        """Search memories using vector similarity.

        This is the L0 entry point for memory retrieval.
        account_id is MANDATORY and injected from RequestContext.

        Args:
            query: Natural language query text
            ctx: RequestContext containing account_id (injected, not from caller)
            top_k: Maximum number of results to return (default 10)
            category: Optional list of categories to filter by
            include_overview: If True, include L1 overview in results

        Returns:
            List of dicts with keys: uri, score, abstract, overview (optional)

        Raises:
            AccessDeniedError: If caller attempts to override account_id

        Note:
            account_id is ALWAYS taken from ctx.account_id.
            Callers cannot specify a different account_id (security requirement).
        """
        # Step 1: Use Planner to infer query type
        typed_query = self._planner.plan(query, ctx)

        # Step 2: Override account_id from context (MANDATORY - caller cannot override)
        # This is the critical security injection point
        typed_query.account_id = ctx.account_id

        # Apply category filter if specified
        if category is not None:
            typed_query.categories = category

        # Override top_k if specified
        if top_k is not None:
            typed_query.top_k = top_k

        # Step 3: L0 vector retrieval
        seed_hits = self._l0_retriever.retrieve(typed_query, ctx)

        # Step 4: Assemble results
        result = self._assembly.assemble_top_k(
            query=query,
            seed_hits=seed_hits,
            account_id=ctx.account_id,
            top_k=top_k,
            include_overview=include_overview,
            ctx=ctx,
        )

        # Convert to dict format for JSON serialization
        return [
            {
                "uri": block.uri,
                "score": block.score,
                "abstract": block.abstract,
                "overview": block.overview if include_overview else None,
                "has_overview": block.has_overview,
                "has_content": block.has_content,
            }
            for block in result.blocks
        ]

    def read_memory(
        self,
        uri: str,
        level: Level,
        ctx: RequestContext,
    ) -> str:
        """Read a memory node at the specified level.

        Args:
            uri: ContextNode URI (e.g., ctx://{account}/users/{user}/memories/profile)
            level: "L0" for abstract, "L1" for overview, "L2" for full content
            ctx: RequestContext for account validation

        Returns:
            Content at the requested level:
            - L0: abstract (≤100 chars summary)
            - L1: overview (structured summary)
            - L2: content (full original text)

        Raises:
            AccessDeniedError (403): If URI account prefix != ctx.account_id
            NodeNotFoundError (404): If node does not exist or is not ACTIVE
            ValueError: If level is not "L0", "L1", or "L2"

        Note:
            This method validates that the URI's account prefix matches
            the RequestContext.account_id. Cross-account access is ALWAYS denied.
        """
        # Extract and validate account from URI
        uri_account = self._extract_account_from_uri(uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                uri=uri,
                account_id=ctx.account_id,
                reason=f"URI account '{uri_account}' does not match context account '{ctx.account_id}'"
            )

        # L0 returns abstract from the node metadata
        if level == "L0":
            node = self._read_service.read_node(uri, ctx)
            return node.abstract

        # L1 and L2 are handled by MemoryReadService
        return self._read_service.read(uri, level, ctx)

    def search_and_read(
        self,
        query: str,
        ctx: RequestContext,
        top_k: int = 3,
        read_level: Level = "L1",
    ) -> list[dict]:
        """Search and read in one call - convenience method.

        Performs search_memory and then read_memory for each result.

        Args:
            query: Natural language query
            ctx: RequestContext
            top_k: Number of results to fetch and read
            read_level: Level to read (L0/L1/L2)

        Returns:
            List of dicts with uri, score, and content at requested level
        """
        # Search for URIs
        search_results = self.search_memory(query, ctx, top_k=top_k)

        # Read each result
        results = []
        for result in search_results:
            try:
                content = self.read_memory(result["uri"], read_level, ctx)
                results.append({
                    "uri": result["uri"],
                    "score": result["score"],
                    "content": content,
                })
            except (AccessDeniedError, NodeNotFoundError):
                # Skip results that can't be read
                continue

        return results

    def _extract_account_from_uri(self, uri: str) -> str:
        """Extract account ID from a ContextNode URI.

        URI format: ctx://{account}/users/{user}/memories/...
                    or ctx://{account}/agents/{agent}/...

        Args:
            uri: ContextNode URI

        Returns:
            Account ID string

        Raises:
            ValueError: If URI format is invalid
        """
        if not uri.startswith("ctx://"):
            raise ValueError(f"Invalid URI format: {uri}")

        path = uri[6:]  # Remove "ctx://"
        parts = path.split("/")

        if len(parts) < 2:
            raise ValueError(f"Invalid URI format: {uri}")

        return parts[0]


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
        self._writer = ContextWriter(fs, outbox_store=outbox_store)
        self._pipeline = CandidatePipeline()
        self._pipeline.set_extractors(self._create_extractors())

    def _create_extractors(self) -> list[CandidateExtractor]:
        """Create default extractors for the pipeline.

        Returns:
            List of CandidateExtractor instances
        """
        return [
            ProfileExtractor(self._llm),
            PreferenceExtractor(self._llm),
            EntityEventExtractor(self._llm),
            ProcedureExtractor(self._llm),
        ]

    def commit_session(
        self,
        messages: list[dict],
        ctx: RequestContext,
        confidence_threshold: float = 0.5,
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

        Returns:
            Dict with write results:
            {
                "candidates_extracted": int,
                "candidates_filtered": int,
                "writes_completed": int,
                "writes_skipped": int,
                "writes_failed": int,
                "plans": list[WritePlan dict]
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

        return {
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


# Singleton instances for simple usage
# In production, use dependency injection
_default_read_api: Optional[MemoryReadAPI] = None
_default_write_api: Optional[MemoryWriteAPI] = None


def init_read_api(fs: ContextFS, vector_index: VectorIndex) -> MemoryReadAPI:
    """Initialize the global read API instance.

    Args:
        fs: ContextFS implementation
        vector_index: VectorIndex implementation

    Returns:
        Configured MemoryReadAPI instance
    """
    global _default_read_api
    _default_read_api = MemoryReadAPI(fs, vector_index)
    return _default_read_api


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


def init_api(
    fs: ContextFS,
    vector_index: VectorIndex,
    llm: LLM,
    outbox_store: Optional[OutboxStore] = None,
) -> tuple[MemoryReadAPI, MemoryWriteAPI]:
    """Initialize both read and write API instances.

    Args:
        fs: ContextFS implementation
        vector_index: VectorIndex implementation
        llm: LLM instance for extraction
        outbox_store: Optional OutboxStore for async indexing

    Returns:
        Tuple of (MemoryReadAPI, MemoryWriteAPI)
    """
    read_api = init_read_api(fs, vector_index)
    write_api = init_write_api(fs, llm, outbox_store)
    return read_api, write_api


def get_read_api() -> Optional[MemoryReadAPI]:
    """Get the global read API instance.

    Returns:
        MemoryReadAPI if initialized, None otherwise
    """
    return _default_read_api


def get_write_api() -> Optional[MemoryWriteAPI]:
    """Get the global write API instance.

    Returns:
        MemoryWriteAPI if initialized, None otherwise
    """
    return _default_write_api


# Backwards compatibility
def get_api() -> Optional[MemoryReadAPI]:
    """Get the global read API instance (backwards compatible).

    Returns:
        MemoryReadAPI if initialized, None otherwise
    """
    return _default_read_api
