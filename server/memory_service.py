"""MemoryService — Core oG-Memory business logic, decoupled from transport.

Used by both:
  - server/app.py        (HTTP / Flask mode)
  - bridge/memory_api.py (CLI / subprocess mode)
"""

from __future__ import annotations

import logging
import os
import re as _re
import threading
from copy import deepcopy
from datetime import datetime as _dt
from datetime import timezone as _timezone
from uuid import uuid4

from core.models import ArchiveRef, ComposedContext, RequestContext, RetrievalConfig, SeedHit, TokenBudget, Role
from session.models import ArchiveEntry, SessionMessage, SessionWindowState
from session.session_manager import SessionManager
from session.rolling_compressor import RollingCompressor
from session.topic_buffer import SlotContent
from providers.config import ProviderConfig
from providers.llm import get_openai_llm
from providers.unified_config import OgMemConfig, get_config
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner, sanitize_query
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthorizationError, AuthService, ResolvedIdentity
from server.control_plane_store import ControlPlaneStore
from server.internal_tool_usage import InternalToolUsageStore, InternalToolUsageTracker
from server.tenant_admin import TenantAdminService

# Chunking support
from extraction.chunking import ConversationChunker

try:
    from pyagfs import AGFSClient

    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI, ReadAPI
    _HAS_AGFS = True
except ImportError:
    _HAS_AGFS = False

try:
    from commit.sql_outbox_store import SQLOutboxStore
    from fs.sql_adapter import SQLContextFS
    from providers.relation_store.sql_relation_store import SQLRelationStore
    _HAS_SQL = True
except ImportError:
    _HAS_SQL = False

logger = logging.getLogger("ogmem.service")

# Fallback archive-trim safety margin used only when incoming messages cannot
# be aligned with the session buffer.  It preserves a likely in-flight
# user/assistant turn that has not reached after_turn yet.
_ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN = 2


# ---------------------------------------------------------------------------
# Helpers (stateless, reusable)
# ---------------------------------------------------------------------------

def extract_content_text(content) -> str:
    """Extract text from message content (string or structured blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content)


def _message_diag(messages) -> list[dict]:
    """Return compact message fingerprints for extraction diagnostics."""
    import hashlib

    diag = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = extract_content_text(msg.get("content", ""))
            msg_id = msg.get("id", "")
            created_at = msg.get("created_at", "")
        else:
            role = getattr(msg, "role", "")
            content = extract_content_text(getattr(msg, "content", ""))
            msg_id = getattr(msg, "id", "")
            created_at = getattr(msg, "created_at", "")

        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        diag.append({
            "i": idx,
            "role": role,
            "chars": len(content),
            "sha": digest,
            "id": msg_id,
            "created_at": str(created_at) if created_at else "",
        })
    return diag


def extract_content_and_tool_calls(content) -> dict:
    """Extract text AND tool call metadata from message content.

    Returns:
        {"text": str, "tool_calls": list[dict]}
    """
    from extraction.tool_collector import parse_message_parts
    return parse_message_parts(content)


def extract_query(messages: list[dict]) -> str:
    """Build a search query from the last 3 user messages."""
    user_texts = []
    for msg in messages:
        if msg.get("role") == "user":
            text = extract_content_text(msg.get("content", ""))
            if text:
                user_texts.append(text)
    return "\n".join(user_texts[-3:]).strip()


def _message_role_content(msg) -> tuple[str, str]:
    if isinstance(msg, dict):
        role = str(msg.get("role", ""))
        content = extract_content_text(msg.get("content", ""))
    else:
        role = str(getattr(msg, "role", ""))
        content = extract_content_text(getattr(msg, "content", ""))
    return role, content


def _seed_hit_to_working_set_item(hit: SeedHit) -> dict:
    return {
        "uri": hit.uri,
        "category": hit.category or "memory",
        "abstract": hit.abstract or "",
        "overview": "",
        "content": "",
        "metadata": hit.metadata or {},
        "score": hit.score,
    }


def _retrieved_block_to_seed_hit(hit) -> SeedHit:
    return SeedHit(
        uri=getattr(hit, "uri", "") or "",
        score=float(getattr(hit, "score", 0.0) or 0.0),
        level=2,
        category=getattr(hit, "category", "") or "memory",
        abstract=(
            getattr(hit, "abstract", None)
            or getattr(hit, "overview", None)
            or getattr(hit, "content_excerpt", None)
            or ""
        ),
        metadata={"source": "prefetch"},
    )


def _find_last_message_sequence_start(messages: list, sequence: list) -> int | None:
    """Find the last contiguous occurrence of sequence inside messages."""
    if not messages or not sequence or len(sequence) > len(messages):
        return None

    haystack = [_message_role_content(msg) for msg in messages]
    needle = [_message_role_content(msg) for msg in sequence]
    last_start = len(haystack) - len(needle)
    for start in range(last_start, -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def _bounded_message_count(value, max_count: int) -> int:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        count = 0
    return max(0, min(count, max_count))


def _update_extraction_summary(existing_summary: str, write_result: dict, max_chars: int = 2000) -> str:
    """Append new candidate abstracts to the extraction summary.

    Simple concatenation strategy: keep [{category}] {abstract} lines,
    truncate to max_chars (oldest entries dropped when exceeded).

    Args:
        existing_summary: Current summary text from previous extractions
        write_result: Dict from commit_session() with "plans" list
        max_chars: Maximum summary length in chars (~500 tokens)

    Returns:
        Updated summary string
    """
    plans = write_result.get("plans", [])
    new_lines = []
    for plan in plans:
        if plan.get("action") == "skip":
            continue
        uri = plan.get("target_uri", "")
        # Extract category and routing_key from URI for context
        parts = uri.rstrip("/").split("/")
        category = parts[-2] if len(parts) >= 2 else ""
        slug = parts[-1] if parts else ""
        new_lines.append(f"[{category}] {slug}")

    if not new_lines:
        return existing_summary

    combined = existing_summary + "\n" + "\n".join(new_lines) if existing_summary else "\n".join(new_lines)

    # Truncate to max_chars, keeping newest entries
    if len(combined) > max_chars:
        combined = combined[-max_chars:]

    return combined


def _positive_int(value: object, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _summary_max_chars(config: OgMemConfig, params: dict | None = None, default: int = 2000) -> int:
    """Resolve request-level summary trim overrides before falling back to config."""
    config_value = _positive_int(getattr(config, "summary_max_chars", default), default)
    request_value = None if params is None else params.get("summaryMaxChars")
    if request_value is not None:
        return _positive_int(request_value, config_value)
    return config_value


def _trim_summary(summary: str, max_chars: int) -> str:
    """Trim summary text explicitly, keeping the newest tail when oversized."""
    if not summary:
        return ""
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:]


def _short_term_index_mode(params: dict | None = None) -> str:
    """Normalize compact short-term index mode for Phase 1 backend handling."""
    raw_value = None if params is None else params.get("shortTermIndexMode")
    if raw_value is None:
        return "sync"
    normalized = str(raw_value).strip().lower()
    if normalized in {"async", "off"}:
        return normalized
    return "sync"


def _parse_session_time(created_at: object) -> object | None:
    if not created_at:
        return None
    try:
        if isinstance(created_at, str):
            return _dt.fromisoformat(created_at.replace("Z", "+00:00"))
        if isinstance(created_at, (int, float)):
            return _dt.fromtimestamp(created_at)
        return created_at
    except Exception as exc:
        logger.warning("failed to parse session_time created_at=%s: %s", created_at, exc)
        return None


_DATE_FROM_CONTENT = _re.compile(
    r"\[group chat conversation:\s*(.+?)\]", _re.IGNORECASE
)


def _extract_date_from_content(content: str):
    """Parse date from message content like '[group chat conversation: 1:56 pm on 8 May, 2023]'."""
    m = _DATE_FROM_CONTENT.search(content or "")
    if not m:
        return None
    date_str = m.group(1).strip()
    if " on " in date_str:
        date_part = date_str.rsplit(" on ", 1)[-1]
    else:
        date_part = date_str
    try:
        return _dt.strptime(date_part.strip(), "%d %B, %Y")
    except ValueError:
        pass
    return None




# ---------------------------------------------------------------------------
# Archive helpers (stateless, testable)
# ---------------------------------------------------------------------------

def build_archive_refs(
    entries: list[ArchiveEntry],
    budget: TokenBudget,
) -> tuple[list[ArchiveRef], list[ArchiveRef]]:
    """Build distance-tiered archive references within token budget.

    Sorting: newest first (by created_at descending).
    Tier 1: Latest archive gets full overview.
    Tier 2: Remaining archives get abstract only, budget-truncated
            (oldest dropped first when budget exceeded).

    Args:
        entries: List of ArchiveEntry objects from SessionArchiveStore
        budget: Token budget for assembly

    Returns:
        Tuple of (latest_archives, pre_archives)
    """
    if not entries:
        return [], []

    # Sort by created_at descending (newest first)
    sorted_entries = sorted(
        entries,
        key=lambda e: e.created_at or "",
        reverse=True,
    )

    archive_budget = budget.archive_limit

    # Tier 1: Latest archive gets full overview
    latest = sorted_entries[0]
    overview_text = latest.overview or ""
    overview_tokens = len(overview_text) // 4

    latest_ref = ArchiveRef(
        archive_id=latest.archive_id,
        archive_uri=f"archive://{latest.session_id}/{latest.archive_id}",
        abstract=latest.abstract or "",
        overview=overview_text,
        tokens=overview_tokens,
    )
    latest_refs = [latest_ref]
    used_tokens = overview_tokens

    # Tier 2: Remaining archives get abstract only
    # Budget-truncated: iterate newest→oldest, stop when budget exceeded
    # (oldest entries are naturally dropped first)
    pre_refs: list[ArchiveRef] = []
    for entry in sorted_entries[1:]:
        abstract_text = entry.abstract or ""
        abstract_tokens = len(abstract_text) // 4

        if used_tokens + abstract_tokens > archive_budget:
            break

        pre_refs.append(ArchiveRef(
            archive_id=entry.archive_id,
            archive_uri=f"archive://{entry.session_id}/{entry.archive_id}",
            abstract=abstract_text,
            overview=None,
            tokens=abstract_tokens,
        ))
        used_tokens += abstract_tokens

    return latest_refs, pre_refs


# ---------------------------------------------------------------------------
# MemoryService — warm singleton holding initialized resources
# ---------------------------------------------------------------------------

class MemoryService:
    """Transport-agnostic oG-Memory service.

    Holds lazy-initialized LLM, ReadAPI, WriteAPI instances.
    Each handler method accepts a plain dict and returns a plain dict.
    """

    def __init__(self, config: OgMemConfig | None = None):
        cfg = config or get_config()
        self._cfg = cfg
        self._provider_cfg = ProviderConfig.from_ogmem_config(cfg)

        self._agfs_base_url = cfg.agfs_base_url
        self._mount_prefix = cfg.agfs_mount_prefix
        self._default_account_id = cfg.account_id
        self._default_user_id = cfg.user_id
        self._default_agent_id = cfg.agent_id

        self._llm = None
        self._write_api = None
        self._read_api = None
        self._session_mgr: SessionManager | None = None
        self._archive_store = None
        self._archive_store_failed = False
        self._sql_pool = None  # Shared pool for all SQL stores

        # Shared vector index + embedder (used by both read API and outbox worker)
        self._vector_index = None
        self._embedder = None
        self._outbox_thread: threading.Thread | None = None
        self._outbox_listener = None
        self._control_store = None
        self._key_manager = None
        self._auth = None
        self._audit = None
        self._tenant_admin = None
        self._internal_tool_usage = InternalToolUsageTracker(
            max_rounds_per_session=_positive_int(
                os.environ.get("OGMEM_TOOL_USAGE_MAX_ROUNDS"),
                1000,
            )
        )
        self._internal_tool_usage_store = None
        self._pending_tool_usage_snapshots: dict[str, dict] = {}
        self._pending_tool_usage_lock = threading.Lock()

    @property
    def _use_sql(self) -> bool:
        return self._cfg.storage_backend == "sql" and _HAS_SQL and self._cfg.sql_connection_string

    def _get_shared_sql_pool(self):
        """Create (once) and return the shared connection pool for all SQL stores."""
        if self._sql_pool is None and self._use_sql:
            from fs.sql_adapter.pool import SharedConnectionPool
            self._sql_pool = SharedConnectionPool(
                connection_string=self._cfg.sql_connection_string,
                pool_size=self._cfg.sql_pool_size,
            )
        return self._sql_pool

    def _get_context_fs(self):
        """Create the appropriate ContextFS implementation based on config."""
        if self._use_sql:
            return SQLContextFS(pool=self._get_shared_sql_pool())
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
        return None

    def _get_outbox_store(self, fs):
        """Create the appropriate OutboxStore implementation based on config."""
        if self._use_sql:
            return SQLOutboxStore(
                pool=self._get_shared_sql_pool(),
                fs=fs,
            )
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return OutboxStore(client=client, fs=fs, mount_prefix=self._mount_prefix)
        return None

    def _get_relation_store(self):
        """Create the appropriate RelationStore implementation based on config."""
        if self._use_sql:
            return SQLRelationStore(pool=self._get_shared_sql_pool())
        if _HAS_AGFS:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return AGFSRelationStore(client=client, mount_prefix=self._mount_prefix)
        return None

    def _get_chunker(self) -> ConversationChunker | None:
        """Create a ConversationChunker if chunking is enabled.

        Returns None if chunking is disabled or not configured.
        """
        if not getattr(self._cfg, "chunking_enabled", False):
            return None

        return ConversationChunker(
            llm=self.get_llm(),
            max_tokens=getattr(self._cfg, "chunk_max_segment_tokens", 8000),
            max_messages=getattr(self._cfg, "chunk_max_messages", 500),
            min_chunk_tokens=getattr(self._cfg, "chunk_min_tokens", 300),
            use_llm_boundary=getattr(self._cfg, "chunk_llm_boundary", True),
        )

    def _generate_and_write_summary(
        self,
        messages: list[dict],
        session_time,
        ctx,
        session_id: str,
        write_api,
    ) -> None:
        """Generate L0 structured summary for a session and write to storage."""
        try:
            from extraction.summary_generator import SessionSummaryGenerator

            # Format messages into text
            lines = []
            participants = set()
            for msg in messages:
                role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("text")
                    )
                if content:
                    lines.append(f"{role}: {content}")
                    if role not in ("system", "assistant"):
                        participants.add(role)

            messages_text = "\n".join(lines)
            if not messages_text.strip():
                return

            participants_str = ", ".join(participants) if participants else ""
            gen = SessionSummaryGenerator(llm=self.get_llm())
            summary = gen.generate(messages_text, session_time, participants_str)
            if not summary:
                return

            formatted = gen.format_summary(summary)
            candidate = gen.build_candidate(
                summary=summary,
                formatted_text=formatted,
                session_time=session_time,
                participants=participants_str,
                session_id=session_id,
            )
            write_api.write_memory(candidate, ctx)
            logger.info(
                "L0 summary written: session=%s len=%d",
                session_id[:8], len(formatted),
            )
        except Exception as exc:
            logger.warning("L0 summary generation failed: %s", exc, exc_info=True)

    def _chunk_and_extract(
        self,
        messages: list[dict],
        write_api,
        ctx,
        session_time,
        session_summary: str,
        tool_stats_text: str,
        extraction_run_id: str,
        session_id: str,
        archive_id: str | None = None,
    ) -> dict:
        """Extract memories from messages, optionally chunking first.

        When chunking is enabled, splits messages into chunks via
        ConversationChunker and calls write_api.commit_session() for each
        chunk independently.  Aggregates results from all chunks.

        When chunking is disabled (or produces a single chunk), falls back
        to a single commit_session() call — identical to the original path.
        """
        # Step 0: Generate L0 summary for the whole session
        self._generate_and_write_summary(
            messages, session_time, ctx, session_id, write_api,
        )

        chunker = self._get_chunker()

        if chunker is None:
            result = write_api.commit_session(
                messages=messages,
                ctx=ctx,
                confidence_threshold=0.5,
                wait=True,
                session_time=session_time,
                session_summary=session_summary,
                tool_stats_text=tool_stats_text,
                archive_id=archive_id,
            )
            # Store raw chunk text for retrieval fallback
            try:
                write_api.write_raw_chunk(
                    messages=messages, ctx=ctx, chunk_index=0,
                    session_id=session_id, session_time=session_time,
                )
            except Exception as exc:
                logger.debug("raw chunk write failed (no-chunker): %s", exc)
            return result

        chunking_result = chunker.chunk_messages(messages, flush=True)

        if not chunking_result.chunks:
            logger.info(
                "chunk_and_extract: chunker returned no chunks, run=%s session=%s",
                extraction_run_id, session_id[:8],
            )
            return {
                "candidates_extracted": 0,
                "candidates_filtered": 0,
                "writes_completed": 0,
                "writes_skipped": 0,
                "writes_failed": 0,
                "plans": [],
            }

        if len(chunking_result.chunks) == 1:
            # Single chunk — no need for multi-extract overhead
            logger.info(
                "chunk_and_extract: single chunk, run=%s session=%s msgs=%d",
                extraction_run_id, session_id[:8], len(chunking_result.chunks[0]),
            )
            result = write_api.commit_session(
                messages=chunking_result.chunks[0],
                ctx=ctx,
                confidence_threshold=0.5,
                wait=True,
                session_time=session_time,
                session_summary=session_summary,
                tool_stats_text=tool_stats_text,
                archive_id=archive_id,
            )
            # Store raw chunk text for retrieval fallback
            try:
                write_api.write_raw_chunk(
                    messages=chunking_result.chunks[0], ctx=ctx, chunk_index=0,
                    session_id=session_id, session_time=session_time,
                )
            except Exception as exc:
                logger.debug("raw chunk write failed (single chunk): %s", exc)
            return result

        # Multi-chunk extraction: each chunk gets its own commit_session()
        logger.info(
            "chunk_and_extract: multi-chunk extraction, run=%s session=%s "
            "chunks=%d total_msgs=%d",
            extraction_run_id, session_id[:8],
            len(chunking_result.chunks), len(messages),
        )

        aggregated = {
            "candidates_extracted": 0,
            "candidates_filtered": 0,
            "writes_completed": 0,
            "writes_skipped": 0,
            "writes_failed": 0,
            "plans": [],
        }

        for idx, chunk in enumerate(chunking_result.chunks):
            logger.info(
                "chunk_and_extract: processing chunk %d/%d, run=%s session=%s msgs=%d",
                idx + 1, len(chunking_result.chunks),
                extraction_run_id, session_id[:8], len(chunk),
            )
            try:
                chunk_result = write_api.commit_session(
                    messages=chunk,
                    ctx=ctx,
                    confidence_threshold=0.5,
                    wait=True,
                    session_time=session_time,
                    session_summary=session_summary,
                    tool_stats_text=tool_stats_text,
                    archive_id=archive_id,
                )
                for key in (
                    "candidates_extracted",
                    "candidates_filtered",
                    "writes_completed",
                    "writes_skipped",
                    "writes_failed",
                ):
                    aggregated[key] += chunk_result.get(key, 0)
                aggregated["plans"].extend(chunk_result.get("plans", []))
                # Store raw chunk text for retrieval fallback
                try:
                    write_api.write_raw_chunk(
                        messages=chunk, ctx=ctx, chunk_index=idx,
                        session_id=session_id, session_time=session_time,
                    )
                except Exception as raw_exc:
                    logger.debug("raw chunk write failed (chunk %d): %s", idx, raw_exc)
            except Exception as exc:
                logger.warning(
                    "chunk_and_extract: chunk %d/%d failed, run=%s session=%s: %s",
                    idx + 1, len(chunking_result.chunks),
                    extraction_run_id, session_id[:8], exc,
                )
                aggregated["writes_failed"] += 1

        logger.info(
            "chunk_and_extract: all chunks done, run=%s session=%s "
            "chunks=%d extracted=%d writes=%d failed=%d",
            extraction_run_id, session_id[:8],
            len(chunking_result.chunks),
            aggregated["candidates_extracted"],
            aggregated["writes_completed"],
            aggregated["writes_failed"],
        )
        return aggregated

    @staticmethod
    def _supports_batch_claim(outbox_store) -> bool:
        """Safely detect SQL outbox implementations with batch-claim support."""
        class_flag = getattr(type(outbox_store), "supports_batch_claim", False) is True
        instance_flag = (
            getattr(outbox_store, "__dict__", {}).get("supports_batch_claim", False)
            is True
        )
        return class_flag or instance_flag

    # -- RequestContext factory ------------------------------------------------

    def build_context(
        self,
        params: dict,
        identity: ResolvedIdentity | None = None,
    ) -> RequestContext:
        """Build RequestContext, allowing per-request overrides.

        Empty strings from the plugin are treated as absent, so defaults apply.
        """
        session_id = params.get("sessionId") or "unknown"
        explicit_agent_id = params.get("agentId")
        user_id = params.get("userId") or params.get("user_id")
        if not user_id:
            # Inherit from session meta when caller doesn't supply userId
            mgr = self.get_session_manager()
            if mgr.has_session(session_id):
                user_id = mgr.get_or_create(session_id).meta.user_id
            if not user_id:
                user_id = self._default_user_id
        account_id = params.get("accountId") or params.get("account_id") or self._default_account_id
        logger.debug("build_context userId=%s (from params: %s)", user_id, params.get("userId"))

        if identity is not None:
            requested_agent_id = explicit_agent_id
            ctx = self.get_auth_service().build_request_context(
                identity,
                account_id=account_id,
                user_id=user_id,
                agent_id=requested_agent_id or "",
                session_id=session_id,
            )
            if ctx.role == Role.ROOT:
                visible = [ctx.user_space_name()]
                if requested_agent_id:
                    visible.append(f"agent:{requested_agent_id}")
                return RequestContext(
                    account_id=ctx.account_id,
                    user_id=ctx.user_id,
                    agent_id=ctx.agent_id,
                    session_id=ctx.session_id,
                    trace_id=ctx.trace_id,
                    role=ctx.role,
                    visible_owner_spaces=tuple(dict.fromkeys(v for v in visible if v)),
                )
            visible = [ctx.user_space_name()]
            visible_agent_ids = self.get_tenant_admin_service().list_visible_agent_ids(ctx.account_id, ctx.user_id, self._cfg)
            visible.extend(f"agent:{agent_id}" for agent_id in visible_agent_ids)
            if requested_agent_id and requested_agent_id not in set(visible_agent_ids):
                raise PermissionError(f"agent access denied: {requested_agent_id}")
            return RequestContext(
                account_id=ctx.account_id,
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                session_id=ctx.session_id,
                trace_id=ctx.trace_id,
                role=ctx.role,
                visible_owner_spaces=tuple(dict.fromkeys(v for v in visible if v)),
            )

        requested_agent_id = explicit_agent_id or self._default_agent_id
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=requested_agent_id,
            session_id=session_id,
            trace_id=str(uuid4()),
            role=Role.ROOT,
            visible_owner_spaces=(),
        )

    def get_control_plane_store(self) -> ControlPlaneStore:
        if self._control_store is None:
            if self._use_sql:
                from server.sql_control_plane_store import SQLControlPlaneStore
                self._control_store = SQLControlPlaneStore(
                    pool=self._get_shared_sql_pool(),
                )
            elif _HAS_AGFS:
                client = AGFSClient(api_base_url=self._agfs_base_url)
                self._control_store = ControlPlaneStore(
                    mount_prefix=self._mount_prefix,
                    client=client,
                )
            else:
                self._control_store = ControlPlaneStore(
                    mount_prefix=self._mount_prefix,
                    local_root=os.path.join(os.getcwd(), ".ogmem_control"),
                )
        return self._control_store

    def get_internal_tool_usage_store(self) -> InternalToolUsageStore:
        if self._internal_tool_usage_store is None:
            self._internal_tool_usage_store = InternalToolUsageStore(self.get_control_plane_store())
        return self._internal_tool_usage_store

    def get_key_manager(self) -> APIKeyManager:
        if self._key_manager is None:
            self._key_manager = APIKeyManager(self.get_control_plane_store())
        return self._key_manager

    def get_auth_service(self) -> AuthService:
        if self._auth is None:
            self._auth = AuthService(self._cfg, self.get_key_manager())
        return self._auth

    def get_audit_service(self) -> AuditService:
        if self._audit is None:
            self._audit = AuditService(self.get_control_plane_store())
        return self._audit

    def get_tenant_admin_service(self) -> TenantAdminService:
        if self._tenant_admin is None:
            self._tenant_admin = TenantAdminService(
                self.get_key_manager(),
                self.get_control_plane_store(),
                self.get_audit_service(),
            )
        return self._tenant_admin

    def _read_profile(self, ctx: RequestContext) -> str:
        """Read user profile L2 content directly from storage.

        Profile may be stored as a single node or as field-level child nodes
        (e.g. profile/name, profile/location, profile/identity). This method
        handles both: if the profile node has content, use it; otherwise
        enumerate children and concatenate their content.
        """
        try:
            fs = self._get_context_fs()
            if fs is None:
                return ""
            profile_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            # Try reading as a single node first
            if fs.exists(profile_uri, ctx):
                node = fs.read_node(profile_uri, ctx)
                if node.content:
                    return node.content

            # Field-level split: enumerate children and read each
            try:
                children = fs.list_children(profile_uri, ctx)
            except Exception as exc:
                logger.error("list_children failed in _read_profile: %s", exc)
                raise

            if not children:
                return ""

            parts = []
            for child_uri in children:
                if fs.exists(child_uri, ctx):
                    child_node = fs.read_node(child_uri, ctx)
                    if child_node.content:
                        # Extract field name from URI for labeling
                        field_name = child_uri.rstrip("/").split("/")[-1]
                        parts.append(f"- {field_name}: {child_node.content.strip()}")

            return "\n".join(parts) if parts else ""
        except Exception as exc:
            logger.error("_read_profile failed for %s: %s", ctx.user_id, exc, exc_info=True)
            raise

    def _collect_archives(self, ctx: RequestContext, token_budget: TokenBudget) -> tuple[list[ArchiveRef], list[ArchiveRef]]:
        """Collect session archives with distance-based graduated compression.

        Tier 1 (latest): full overview.  Tier 2 (older): abstract only.
        """
        try:
            store = self._get_archive_store()
            if store is None:
                return [], []

            entries = store.list_archives(session_id=ctx.session_id, ctx=ctx)
            return build_archive_refs(entries, token_budget)
        except Exception as exc:
            logger.warning("_collect_archives failed: %s", exc, exc_info=True)
            return [], []


    def _search_working_set(self, query: str, ctx: RequestContext) -> tuple[list[dict], list[dict], dict]:
        """Two-round vector search for memory nodes.

        Round 1: All categories → structured items for retrievedEvidence.
        Round 2: session_summary only, top_k=1 → L0 summary for sessionContext.

        Returns (structured_items, summary_items, stats).
        """
        items = []
        summary_items = []
        stats: dict = {}
        try:
            read_api = self.get_read_api()

            # Round 1: structured nodes (all categories)
            result = read_api.search_memory(
                query=query, ctx=ctx, top_k=15,
                fill_content_for_top_k=5, mode="QUICK",
            )
            if result and result.hits:
                if result.trace and hasattr(result.trace, "stages"):
                    for stage in result.trace.stages:
                        if stage.stage == "seed_retrieval":
                            stats["seed_output"] = stage.output_count
                            break
                if result.trace and hasattr(result.trace, "level_histogram"):
                    stats["level_histogram"] = result.trace.level_histogram
                stats["hit_count"] = len(result.hits)
                stats["hit_categories"] = list({h.category for h in result.hits if h.category})

                items = [
                    {
                        "uri": h.uri,
                        "abstract": h.abstract or "",
                        "overview": h.overview or "",
                        "content": h.content_excerpt or "",
                        "score": h.score,
                        "category": h.category,
                    }
                    for h in result.hits
                ]
                items = self._truncate_by_score_gap(items)

            # Round 2: L0 session summary for sessionContext injection
            try:
                summary_result = read_api.search_memory(
                    query=query, ctx=ctx, top_k=1,
                    categories=["session_summary"],
                    fill_content_for_top_k=1, mode="QUICK",
                )
                if summary_result and summary_result.hits:
                    summary_items = [
                        {
                            "uri": h.uri,
                            "abstract": h.abstract or "",
                            "overview": h.overview or "",
                            "content": h.content_excerpt or "",
                            "score": h.score,
                            "category": h.category,
                        }
                        for h in summary_result.hits
                    ]
                    logger.info(
                        "L0 summary search: found=%d score=%.3f len=%d",
                        len(summary_items),
                        summary_items[0]["score"] if summary_items else 0,
                        len(summary_items[0]["content"]) if summary_items else 0,
                    )
            except Exception as exc:
                logger.debug("L0 summary search failed: %s", exc)

            return items, summary_items, stats
        except Exception as exc:
            logger.warning("Working set search failed: %s", exc)
        return [], [], {}

    @staticmethod
    def _truncate_by_score_gap(items: list[dict], min_keep: int = 3, gap_ratio: float = 0.15) -> list[dict]:
        """Truncate items at the largest score gap.

        Finds the biggest relative score drop between consecutive items.
        Keeps at least min_keep items. If no significant gap, returns all.
        """
        if len(items) <= min_keep:
            return items

        scores = [it["score"] for it in items]
        max_gap_idx = min_keep - 1
        max_gap = 0.0

        for i in range(min_keep - 1, len(scores) - 1):
            if scores[i] > 0:
                gap = (scores[i] - scores[i + 1]) / scores[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i

        if max_gap >= gap_ratio:
            return items[: max_gap_idx + 1]

        return items


    # -- System prompt builder (KV-cache friendly) ----------------------------

    def _format_profile(
        self,
        profile: str,
        budget: dict[str, int],
    ) -> str:
        """Build stable identity context (Profile only) → systemPromptAddition.

        This is the MOST stable layer — changes only when user profile updates.
        Fully KV-cacheable across all turns and sessions.
        """
        if not profile:
            return ""

        max_tokens = budget.get("identity", 5000)
        text = f"## Profile\n{profile}"
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _format_archives(
        self,
        latest_archives: list[ArchiveRef],
        pre_archives: list[ArchiveRef],
        budget: dict[str, int],
    ) -> str:
        """Build episodic history context (Archives only) → systemPromptSuffix prefix.

        Goes into systemPromptSuffix BEFORE session state. Semi-stable:
        only changes when sessions are archived (not every turn).
        """
        max_tokens = budget.get("archive", 40000)

        archive_parts: list[str] = []
        if latest_archives:
            latest = latest_archives[0]
            if latest.overview:
                archive_parts.append(f"### Latest Session\n{latest.overview}")
        if pre_archives:
            lines = ["### Previous Sessions"]
            for ref in pre_archives:
                lines.append(f"- {ref.archive_id}: {ref.abstract}")
            archive_parts.append("\n".join(lines))

        if not archive_parts:
            return ""

        text = "## Archive History\n" + "\n".join(archive_parts)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _format_working_set(self, working_set: list[dict], budget: dict[str, int]) -> str:
        """Build DYNAMIC working set as user message content.

        This goes into memoryUserMessage and changes every turn.
        Placed in user message so KV cache for stable system prompt
        prefix is preserved across turns.

        Args:
            working_set: List of retrieved memory items
            budget: dict allocation from allocate()
        """
        if not working_set:
            return ""
        ws_lines = [
            "## Retrieved Memories",
            "The following memories were retrieved from the user's long-term memory. "
            "Use them to answer the user's question. If the answer is found in these memories, respond based on them.",
            "",
        ]
        for item in working_set:
            cat = item.get("category", "memory")
            abstract = item.get("abstract", "")
            overview = item.get("overview", "")
            content = item.get("content", "")
            text_to_show = overview or content or abstract

            # Add temporal context if available
            when = (item.get("metadata") or {}).get("when")
            if when:
                text_to_show = f"[{when}] {text_to_show}"

            ws_lines.append(f"- [{cat}] {text_to_show}")
        ws_text = "\n".join(ws_lines)
        ws_tokens = len(ws_text) // 4

        working_set_budget = budget.get("working_set", 20000)

        if ws_tokens > working_set_budget and working_set_budget > 100:
            ws_text = ws_text[: working_set_budget * 4]
        return ws_text

    def _get_session_state(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> SessionWindowState:
        """Get or compress session window state (Layer 2)."""
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        mgr.load_session_state(session_id, ctx)
        window_state = buf.window_state
        window_state.last_accessed_at = _dt.now(_timezone.utc).isoformat()
        self._sync_session_state_bridge_if_due(session_id, buf, window_state)
        mgr.update_window_state(session_id, window_state)

        if not getattr(self._cfg, "rolling_compress_enabled", False):
            return window_state

        if not buf.should_compress():
            return window_state

        # Need to compress — use RollingCompressor
        try:
            compressor = RollingCompressor(
                llm=self.get_llm(),
                fallback_enabled=getattr(
                    self._cfg,
                    "rolling_compress_fallback_enabled",
                    False,
                ),
            )
            window_state = compressor.compress(
                buf.messages,
                window_state,
                session_state=(
                    mgr.get_session_state()
                    if getattr(self._cfg, "session_state_bridge_enabled", True)
                    else None
                ),
                session_id=session_id,
            )
            mgr.update_window_state(session_id, window_state)
            logger.info(
                "Session window compressed: session=%s turns=%d tokens=%d",
                session_id,
                window_state.turn_count_at_last_compress,
                window_state.token_count_at_last_compress,
            )
        except Exception as exc:
            logger.warning("Session window compression failed: %s", exc)

        return window_state

    def _sync_session_state_bridge_if_due(
        self,
        session_id: str,
        buf,
        window_state: SessionWindowState,
    ) -> bool:
        if not getattr(self._cfg, "session_state_bridge_enabled", True):
            return False
        try:
            session_state = self.get_session_manager().get_session_state()
            version = session_state.get_version(session_id)
            if version == window_state.session_state_version:
                return False

            interval = _positive_int(
                getattr(self._cfg, "session_state_sync_interval_turns", 1),
                1,
            )
            synced_turn_count = getattr(window_state, "session_state_sync_turn_count", 0)
            if synced_turn_count > 0 and (buf.turn_count - synced_turn_count) < interval:
                return False

            from session.session_state_bridge import apply_session_state_bridge

            return apply_session_state_bridge(
                window_state,
                session_state,
                session_id,
                turn_count=buf.turn_count,
            )
        except Exception as exc:
            logger.warning("SessionState bridge sync failed: %s", exc)
            return False

    def _format_session_state(
        self,
        window_state: SessionWindowState,
        budget: dict[str, int],
    ) -> str:
        """Build Layer 2 session state suffix for system prompt.

        Goes into systemPromptSuffix — updated atomically every N turns.
        Placed at end of system prompt so KV cache for stable prefix
        (Layer 1) is preserved; only this suffix chunk changes.

        Args:
            window_state: Current session window state
            budget: dict allocation from allocate()
        """
        sections: list[str] = []

        if window_state.active_task:
            sections.append(f"## Active Task\n{window_state.active_task}")

        if window_state.skills_text:
            sections.append(f"## Skills Summary\n{window_state.skills_text}")

        if window_state.confirmed_constraints:
            items = "\n".join(f"- {c}" for c in window_state.confirmed_constraints[:5])
            sections.append(f"## Confirmed Constraints\n{items}")

        if window_state.recent_decisions:
            items = "\n".join(f"- {d}" for d in window_state.recent_decisions[:5])
            sections.append(f"## Recent Decisions\n{items}")

        if window_state.open_loops:
            items = "\n".join(f"- {loop}" for loop in window_state.open_loops[:5])
            sections.append(f"## Open Loops\n{items}")

        if window_state.compressed_text:
            sections.append(f"## Recent Session Summary\n{window_state.compressed_text}")

        if not sections:
            return ""

        combined = "\n\n".join(sections)

        session_state_budget = budget.get("session_state", 10000)

        # Truncate to budget
        max_chars = session_state_budget * 4
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        return combined

    # -- Token estimation -----------------------------------------------------

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a list of messages.

        Uses CJK-aware estimation: ~1.5 chars/token for CJK, ~4 chars/token for English.
        """
        total_chars = 0
        cjk_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
                cjk_chars += sum(1 for c in content if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
            elif isinstance(content, list):
                for block in content:
                    text = block.get("text", "") if isinstance(block, dict) else (block if isinstance(block, str) else "")
                    total_chars += len(text)
                    cjk_chars += sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        # CJK: ~1.5 chars/token, English: ~4 chars/token
        cjk_tokens = int(cjk_chars / 1.5)
        en_tokens = (total_chars - cjk_chars) // 4
        return cjk_tokens + en_tokens

    # -- Lazy resource getters ------------------------------------------------

    def _get_shared_vector_index(self):
        """Lazy-init shared vector index (same instance for read + outbox worker)."""
        if self._vector_index is None:
            self._vector_index = self._provider_cfg.create_vector_index()
            logger.info("Shared vector_index ready: %s", type(self._vector_index).__name__)
        return self._vector_index

    def _get_shared_embedder(self):
        """Lazy-init shared embedder (same instance for read + outbox worker)."""
        if self._embedder is None:
            self._embedder = self._provider_cfg.create_embedder()
            logger.info("Shared embedder ready: %s", type(self._embedder).__name__)
        return self._embedder


    def _start_outbox_worker(self):
        """Start OutboxWorker background thread (once, idempotent)."""
        if self._outbox_thread is not None:
            return
        if not _HAS_AGFS and not _HAS_SQL:
            logger.info("OutboxWorker skipped: no storage backend available")
            return
        try:
            from index.outbox_worker import OutboxWorker

            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()

            fs = self._get_context_fs()
            if fs is None:
                return
            outbox_store = self._get_outbox_store(fs)
            if outbox_store is None:
                return

            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=fs,
                llm=self.get_llm(),
                directory_summary_enabled=getattr(self._cfg, 'directory_summary_enabled', False),
            )

            # SQL backend: use notify-based listener for real-time processing
            if self._supports_batch_claim(outbox_store):
                from index.sql_notify_listener import SQLNotifyListener

                worker_id = f"sql-listener-{uuid4().hex[:8]}"
                listener = SQLNotifyListener(
                    outbox_store=outbox_store,
                    worker=worker,
                    worker_id=worker_id,
                )
                self._outbox_listener = listener
                self._outbox_thread = threading.Thread(
                    target=listener.run_forever,
                    daemon=True,
                    name="outbox-listener",
                )
                self._outbox_thread.start()
                logger.info("OutboxWorker SQL listener thread launched")
                return

            # AGFS backend: polling loop with multi-account support
            def _get_account_ids_to_scan():
                if self._cfg.role_control_enabled:
                    try:
                        accounts = self.get_key_manager().get_accounts()
                        return [a["account_id"] for a in accounts if a.get("account_id")]
                    except Exception as exc:
                        logger.warning("Failed to list accounts for outbox scan: %s", exc)
                return [self._default_account_id]

            def _outbox_loop():
                import time
                logger.info("OutboxWorker thread started (polling every 5s)")
                while True:
                    try:
                        account_ids = _get_account_ids_to_scan()
                        worker.run_once(
                            outbox_store=outbox_store,
                            account_ids=account_ids,
                        )
                    except Exception as exc:
                        logger.warning("OutboxWorker poll error: %s", exc)
                    time.sleep(5)

            self._outbox_thread = threading.Thread(
                target=_outbox_loop, daemon=True, name="outbox-worker"
            )
            self._outbox_thread.start()
            logger.info("OutboxWorker background thread launched")
        except Exception as exc:
            logger.warning("Failed to start OutboxWorker: %s", exc, exc_info=True)

    def _async_drain(self, account_id: str | None = None):
        """Background thread wrapper for outbox drain with logging."""
        try:
            drain_stats = self.drain_outbox_sync(account_id=account_id)
            logger.info(
                "outbox_drain processed=%d succeeded=%d failed=%d",
                drain_stats.get("processed", 0),
                drain_stats.get("succeeded", 0),
                drain_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.error("outbox_drain failed in background: %s", exc, exc_info=True)

    def drain_outbox_sync(self, account_id: str | None = None) -> dict:
        """Synchronously process pending OutboxEvents: embed → upsert to vector index.

        In subprocess mode the background OutboxWorker thread dies with the process,
        so we must drain inline before returning to ensure data is indexed.
        ChromaDB persists to disk, so the next subprocess call will find the data.
        """
        if not _HAS_AGFS and not _HAS_SQL:
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        try:
            from index.outbox_worker import OutboxWorker

            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()

            fs = self._get_context_fs()
            if fs is None:
                return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
            outbox_store = self._get_outbox_store(fs)
            if outbox_store is None:
                return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=fs,
                llm=self.get_llm(),
                directory_summary_enabled=getattr(self._cfg, 'directory_summary_enabled', False),
            )

            account_ids = []
            if not self._supports_batch_claim(outbox_store):
                effective_account_id = account_id or self._default_account_id
                account_ids = [effective_account_id]

            stats = worker.run_once(
                outbox_store=outbox_store,
                account_ids=account_ids,
            )
            logger.info("drain_outbox_sync: %s", stats)
            return stats
        except Exception as exc:
            logger.warning("drain_outbox_sync failed: %s", exc)
            return {"processed": 0, "succeeded": 0, "failed": 1, "error": str(exc)}

    def shutdown(self, join_timeout: float = 1.0) -> None:
        """Best-effort shutdown for background SQL listener threads."""
        if self._outbox_listener is not None:
            try:
                self._outbox_listener.stop()
            except Exception as exc:
                logger.warning("Failed to stop outbox listener: %s", exc)

        thread = self._outbox_thread
        if (
            thread is not None
            and hasattr(thread, "is_alive")
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            try:
                thread.join(timeout=join_timeout)
            except Exception as exc:
                logger.warning("Failed to join outbox thread: %s", exc)

        self._outbox_listener = None
        self._outbox_thread = None

    def get_llm(self):
        if self._llm is None:
            cfg = self._provider_cfg
            if cfg.provider == "mock" and cfg.openai_api_key:
                cfg = ProviderConfig(
                    provider="openai",
                    openai_api_key=cfg.openai_api_key,
                    openai_base_url=cfg.openai_base_url,
                    openai_llm_model=cfg.openai_llm_model,
                    llm_temperature=cfg.llm_temperature,
                    llm_max_tokens=cfg.llm_max_tokens,
                )
            if cfg.provider == "mock":
                from providers.llm import MockLLM
                self._llm = MockLLM()
            else:
                openai_llm_cls, _ = get_openai_llm()
                self._llm = openai_llm_cls(
                    api_key=cfg.effective_openai_api_key(),
                    base_url=cfg.openai_base_url,
                    model=cfg.openai_llm_model,
                )
            logger.info("LLM ready: %s", type(self._llm).__name__)
        return self._llm

    def get_write_api(self):
        if self._write_api is None:
            fs = self._get_context_fs()
            if fs is None:
                return None
            outbox = self._get_outbox_store(fs)
            if outbox is None:
                return None
            # Create SchemaRegistry for dynamic tool generation
            schema_registry = None
            try:
                from extraction.schemas.registry import SchemaRegistry
                schema_registry = SchemaRegistry()
                logger.info("SchemaRegistry initialized for dynamic tool generation")
            except Exception as exc:
                logger.warning("SchemaRegistry initialization failed; write API unavailable: %s", exc)
                return None

            # Create URIResolver for prefetch
            uri_resolver = None
            try:
                from core.uri_resolver import URIResolver
                if schema_registry:
                    uri_resolver = URIResolver(schema_registry)
                    logger.info("URIResolver initialized for prefetch")
            except Exception as exc:
                logger.warning("URIResolver initialization failed, prefetch disabled: %s", exc)

            # Get shared vector_index and embedder for prefetch
            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()

            self._write_api = MemoryWriteAPI(
                fs=fs,
                llm=self.get_llm(),
                outbox_store=outbox,
                schema_registry=schema_registry,
                vector_index=vector_index,
                embedder=embedder,
                uri_resolver=uri_resolver,
internal_tool_usage_tracker=self._internal_tool_usage,
                extraction_mode=self._cfg.extraction_mode,
            )
            logger.info("WriteAPI ready (backend=%s, extraction_mode=%s)", self._cfg.storage_backend, self._cfg.extraction_mode)
            # Start outbox worker so written memories get indexed
            self._start_outbox_worker()
        return self._write_api

    def forget_memory(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        external_uri: str | None = None,
        reason: str | None = None,
        client_request_id: str | None = None,
    ) -> dict:
        """Forget a single memory via soft archive and index outbox sync."""

        fs = self._get_context_fs()
        if fs is None:
            raise RuntimeError("ContextFS unavailable for forget_memory")

        node = fs.read_node(uri, ctx)
        if node.category == "profile" or uri.rstrip("/").endswith("/memories/profile"):
            raise AuthorizationError("profile memory cannot be forgotten via MCP")

        role = ctx.role if isinstance(ctx.role, Role) else Role(ctx.role)
        agent_scoped = node.owner_space == ctx.agent_space_name() or "/agents/" in uri
        if agent_scoped and role not in (Role.ADMIN, Role.ROOT):
            raise AuthorizationError("agent-scoped memory forget requires ADMIN or ROOT")

        fs.archive_node(uri, ctx)

        outbox_event_id = None
        outbox = self._get_outbox_store(fs)
        if outbox is not None:
            event = outbox.register_archive(uri, ctx)
            outbox_event_id = getattr(event, "event_id", None)

        self.get_audit_service().record(
            ctx.account_id,
            actor=ctx.user_id,
            target=external_uri or uri,
            action="mcp.delete_memory",
            result="success",
            trace_id=ctx.trace_id,
            details={
                "mode": "archive",
                "internal_uri": uri,
                "external_uri": external_uri,
                "category": node.category,
                "owner_space": node.owner_space,
                "role": str(ctx.role),
                "agent_id": ctx.agent_id,
                "session_id": ctx.session_id,
                "reason": reason,
                "client_request_id": client_request_id,
                "outbox_event_id": outbox_event_id,
            },
        )

        return {
            "ok": True,
            "deleted": True,
            "mode": "archive",
            "uri": external_uri or uri,
            "category": node.category,
            "outbox_event_id": outbox_event_id,
        }

    def get_read_api(self):
        if self._read_api is None:
            embedder = self._get_shared_embedder()
            vector_index = self._get_shared_vector_index()

            cfg = RetrievalConfig()

            # Create relation_store
            relation_store = None
            try:
                relation_store = self._get_relation_store()
            except Exception:
                pass

            # Create context_reader for content filling (shared by pipeline and read_service)
            context_reader = None
            try:
                fs = self._get_context_fs()
                if fs is not None:
                    context_reader = ContextReader(fs=fs, storage_backend=self._cfg.storage_backend)
            except Exception as exc:
                logger.warning("ContextFS unavailable for read_memory: %s", exc)

            pipeline = RetrievalPipeline(
                planner=QueryPlanner(cfg),
                seed_retriever=SeedRetriever(vector_index, embedder, cfg),
                hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
                assembly=ResultRanker(cfg, relation_store=relation_store),
                config=cfg,
                context_reader=context_reader,
            )

            self._read_api = ReadAPI(
                pipeline=pipeline,
                read_service=context_reader,
                config=cfg,
            )
            logger.info(
                "ReadAPI ready: embedder=%s, index=%s, backend=%s",
                type(embedder).__name__,
                type(vector_index).__name__,
                self._cfg.storage_backend,
            )
            # Start outbox worker so pending events from ingest get indexed
            self._start_outbox_worker()
        return self._read_api

    # -- Lifecycle handlers ---------------------------------------------------

    def get_session_manager(self) -> SessionManager:
        """Lazy-init SessionManager with current service dependencies."""
        if self._session_mgr is None:
            session_state_context_fs = (
                self._get_context_fs
                if getattr(self._cfg, "session_state_bridge_enabled", True)
                else None
            )
            self._session_mgr = SessionManager(
                get_llm=self.get_llm,
                get_write_api=self.get_write_api,
                get_agfs=self._get_context_fs,
                get_archive_store=self._get_archive_store,
                archive_store_required=self._use_sql,
                get_context_fs=session_state_context_fs,
                archive_max_count=self._cfg.archive_max_count,
                archive_merge_threshold=self._cfg.archive_merge_threshold,
                compression_quality_enabled=self._cfg.compression_quality_enabled,
                compression_quality_persist_metadata=self._cfg.compression_quality_persist_metadata,
            )
            logger.info("SessionManager ready")
        return self._session_mgr

    def _get_archive_store(self):
        """Return an archive store instance.

        SQL mode: cached for service lifetime (shared connection pool).
        AGFS mode: fresh instance each call (self-healing on transient failures).
        """
        if self._use_sql:
            if self._archive_store is not None:
                return self._archive_store
            if self._archive_store_failed:
                return None
            try:
                from session.sql_archive_store import SQLSessionArchiveStore
                self._archive_store = SQLSessionArchiveStore(
                    pool=self._get_shared_sql_pool(),
                )
                return self._archive_store
            except Exception as exc:
                logger.error("Failed to create SQL archive store: %s", exc)
                self._archive_store_failed = True
                return None
        if _HAS_AGFS:
            try:
                from session import SessionArchiveStore
                client = AGFSClient(api_base_url=self._agfs_base_url)
                agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
                return SessionArchiveStore(fs=agfs)
            except Exception as exc:
                logger.error("Failed to create AGFS archive store: %s", exc)
                return None
        return None

    def _get_agfs_fs(self):
        """Create a fresh ContextFS instance for session operations."""
        try:
            return self._get_context_fs()
        except Exception as exc:
            logger.warning("_get_agfs_fs failed: %s", exc)
            return None

    def _require_global_session_admin(self, params: dict | None) -> None:
        ctx = (params or {}).get("_ctx")
        auth = self.get_auth_service()
        if not isinstance(ctx, RequestContext):
            if auth.role_control_active():
                raise PermissionError("authenticated context required")
            return
        auth.require_role(ctx, Role.ROOT, Role.ADMIN)

    def prefetch(self, params: dict) -> dict:
        """Best-effort pre-compose retrieval staged into the session topic buffer."""
        if not getattr(self._cfg, "prefetch_enabled", False):
            return {"ok": True, "prefetched": 0, "reason": "disabled"}
        try:
            ctx = params.get("_ctx") or self.build_context(params)
            session_id = params.get("sessionId") or ctx.session_id
            messages = params.get("messages", [])
            raw = params.get("prompt", "") or params.get("query", "") or extract_query(messages)
            query = sanitize_query(raw)
            if not query:
                return {"ok": True, "prefetched": 0, "reason": "empty_query"}

            top_k = _positive_int(
                params.get("prefetchTopK", getattr(self._cfg, "prefetch_top_k", 5)),
                5,
            )
            result = self.get_read_api().search_memory(
                query=query,
                ctx=ctx,
                top_k=top_k,
                fill_content_for_top_k=0,
                mode="QUICK",
            )
            hits = [
                _retrieved_block_to_seed_hit(hit)
                for hit in (getattr(result, "hits", []) or [])[:top_k]
                if getattr(hit, "uri", "")
            ]
            topic_buffer = self.get_session_manager().get_topic_buffer(session_id)
            topic_buffer.set_pending_injection(hits)
            return {"ok": True, "prefetched": len(hits)}
        except Exception as exc:
            logger.warning("prefetch failed: %s", exc, exc_info=True)
            return {"ok": False, "prefetched": 0, "error": str(exc)}

    def session_working_set(self, params: dict | None = None) -> dict:
        params = params or {}
        self._require_global_session_admin(params)
        return {
            "ok": True,
            "sessions": self.get_session_manager().list_session_working_set(),
        }

    def evict_idle_sessions(self, params: dict) -> dict:
        self._require_global_session_admin(params)
        max_idle_seconds = _positive_int(params.get("maxIdleSeconds"), 0)
        evicted = self.get_session_manager().evict_idle_sessions(
            max_idle_seconds=max_idle_seconds,
            now_iso=params.get("nowIso"),
            ctx=params.get("_ctx"),
        )
        return {"ok": True, "evicted": evicted}

    def compose(self, params: dict) -> dict:
        """Assemble memory context for the current turn.

        Pipeline with 3-layer architecture:
          1. Extract query + budget
          2. Read profile          (Layer 1 — stable identity, rarely changes)
          3. Collect archives       (Layer 1b — episodic history, semi-stable)
          4. Search working set     (Layer 3 — dynamic, changes every turn)
          4b. Get/compress session state (Layer 2 — updates every N turns)
          5. Build stable identity  (Layer 1 output → system message)
          5b. Build episodic + session state (→ system messages)
          6. Return (all context injected into messages as system messages)

        Memory context is injected into messages as system messages,
        ordered by stability (less dynamic first, more dynamic last):

          Before original messages:
            ## Profile            ← changes only on profile update
            ## Archive History    ← changes when sessions end
            ## Session State      ← updates every N turns
          After original messages:
            ## Working Set        ← changes every turn

        systemPromptAddition is empty for KV-cache efficiency.

        Args:
            params: Dict with optional keys:
                - messages: list[dict] - Current conversation messages
                - prompt: str - Direct query prompt (optional)
                - tokenBudget: int - Token budget (default: 128_000)

        Returns:
            Dict with:
                - messages: list[dict] - Messages with injected system messages
                - systemPromptAddition: str - Empty (content in messages)
                - systemPromptSuffix: str - Empty (content in messages)
                - memoryUserMessage: str - Empty (content in messages)
                - estimatedTokens: int - Total estimated tokens
                - archiveCount: int - Number of archives found
                - archiveIncluded: bool - Whether archives were injected
                - identityContext: str - Semantic slot: profile only
                - episodicContext: str - Semantic slot: archive history
                - sessionContext: str - Semantic slot: structured session state
        """
        messages = params.get("messages", [])
        prompt = params.get("prompt", "") or params.get("query", "")

        # Strip previous synthetic memory messages to prevent cumulative duplication.
        # Synthetic messages are tagged with _ogmem=True by _to_response().
        messages = [
            m for m in messages
            if not (isinstance(m, dict) and m.get("_ogmem"))
        ]
        logger.info(
            "assemble entry: msgs=%d prompt_len=%d keys=%s",
            len(messages), len(prompt), sorted(params.keys()),
        )
        raw = prompt.strip() if prompt else extract_query(messages)
        query = sanitize_query(raw)

        # Step 1: Budget
        token_budget_value = params.get("tokenBudget", 128_000)
        token_budget = TokenBudget(total=token_budget_value)
        budget_allocation = token_budget.allocate()

        # Empty query: return messages unchanged
        if not query:
            result = ComposedContext(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

        try:
            ctx = params.get("_ctx") or self.build_context(params)
        except Exception as exc:
            logger.warning("assemble build_context failed: %s", exc, exc_info=True)
            result = ComposedContext(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

        # Each step degrades independently — a failure in one doesn't kill the rest
        session_id = params.get("sessionId") or ctx.session_id
        mgr = self.get_session_manager()
        topic_buffer = None
        try:
            if session_id:
                topic_buffer = mgr.get_topic_buffer(session_id)
        except Exception as exc:
            logger.warning("assemble topic buffer unavailable: %s", exc)

        try:
            if (
                topic_buffer is not None
                and session_id
                and getattr(self._cfg, "topic_detection_enabled", False)
            ):
                from session.topic_detector import TopicDetector

                buf = mgr.get_or_create(session_id, ctx=ctx)
                topic_messages = list(buf.messages)
                for idx, msg in enumerate(messages[-8:]):
                    role, content = _message_role_content(msg)
                    if role and content:
                        topic_messages.append(
                            SessionMessage(
                                id=f"compose_topic_{idx}",
                                role=role,
                                content=content,
                            )
                        )
                embedder = None
                try:
                    embedder = self._get_shared_embedder()
                except Exception as exc:
                    logger.debug("topic detection embedder unavailable: %s", exc)
                topic_changed = topic_buffer.update_topic(
                    TopicDetector(embedder=embedder).detect(
                        topic_messages,
                        previous=topic_buffer.get_current_topic(),
                    )
                )
                if topic_changed:
                    topic_buffer.clear_injected_tracking()
        except Exception as exc:
            logger.warning("assemble topic detection failed: %s", exc)

        profile = ""
        identity_context = ""
        try:
            identity_slot = topic_buffer.get_cached_slot("identity") if topic_buffer else None
            if identity_slot is not None:
                identity_context = identity_slot.content
            else:
                profile = self._read_profile(ctx)
        except Exception as exc:
            logger.warning("assemble profile read failed: %s", exc)

        latest_archives, pre_archives = [], []
        episodic_context = ""
        archive_count_from_cache = None
        try:
            archive_slot = topic_buffer.get_cached_slot("archive_history") if topic_buffer else None
            if archive_slot is not None:
                episodic_context = archive_slot.content
                archive_count_from_cache = len(archive_slot.uris) or (
                    1 if episodic_context else 0
                )
            else:
                latest_archives, pre_archives = self._collect_archives(ctx, token_budget)
        except Exception as exc:
            logger.warning("assemble archive collection failed: %s", exc)

        working_set = []
        summary_items = []
        search_stats: dict = {}
        try:
            working_set, summary_items, search_stats = self._search_working_set(query, ctx)
        except Exception as exc:
            logger.warning("assemble working set search failed: %s", exc)

        try:
            if topic_buffer is not None:
                pending_hits = topic_buffer.get_pending_injection() or []
                if pending_hits:
                    existing_uris = {
                        item.get("uri") for item in working_set
                        if isinstance(item, dict) and item.get("uri")
                    }
                    injected_uris: list[str] = []
                    for hit in pending_hits:
                        if not hit.uri or hit.uri in existing_uris or topic_buffer.was_injected(hit.uri):
                            continue
                        working_set.append(_seed_hit_to_working_set_item(hit))
                        existing_uris.add(hit.uri)
                        injected_uris.append(hit.uri)
                    if injected_uris:
                        topic_buffer.mark_injected(injected_uris)
                    topic_buffer.clear_pending_injection()
        except Exception as exc:
            logger.warning("assemble pending injection merge failed: %s", exc)

        # Each step degrades independently — a failure in one doesn't kill the rest

        # Step 5a: Build stable identity (Profile only) → systemPromptAddition
        try:
            if not identity_context:
                identity_context = self._format_profile(
                    profile=profile,
                    budget=budget_allocation,
                )
                if topic_buffer is not None and identity_context:
                    topic_buffer.set_cached_slot(
                        "identity",
                        SlotContent(
                            content=identity_context,
                            uris=[f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"],
                            tokens=len(identity_context) // 4,
                        ),
                    )
        except Exception as exc:
            logger.warning("assemble identity_context build failed: %s", exc)

        # Step 5b: Build episodic history (Archives only) → systemPromptSuffix prefix
        try:
            if not episodic_context:
                episodic_context = self._format_archives(
                    latest_archives=latest_archives,
                    pre_archives=pre_archives,
                    budget=budget_allocation,
                )
                if topic_buffer is not None and episodic_context:
                    topic_buffer.set_cached_slot(
                        "archive_history",
                        SlotContent(
                            content=episodic_context,
                            uris=[
                                ref.archive_uri
                                for ref in [*latest_archives, *pre_archives]
                                if ref.archive_uri
                            ],
                            tokens=len(episodic_context) // 4,
                        ),
                    )
        except Exception as exc:
            logger.warning("assemble episodic_context build failed: %s", exc)

        # Step 5c: Build dynamic working set → user message (preserves KV cache)
        working_set_msg = ""
        try:
            working_set_msg = self._format_working_set(working_set, budget_allocation)
        except Exception as exc:
            logger.warning("assemble working_set_msg build failed: %s", exc)

        # Step 4b: Get/compress session state → Layer 2 (updates every N turns)
        session_state_suffix = ""
        window_state = SessionWindowState()  # Default empty state
        try:
            window_state = self._get_session_state(session_id, ctx)
            skills_slot = topic_buffer.get_cached_slot("skills_summary") if topic_buffer else None
            if skills_slot is not None and skills_slot.content and not window_state.skills_text:
                window_state.skills_text = skills_slot.content
            elif topic_buffer is not None and window_state.skills_text and skills_slot is None:
                topic_buffer.set_cached_slot(
                    "skills_summary",
                    SlotContent(
                        content=window_state.skills_text,
                        tokens=len(window_state.skills_text) // 4,
                    ),
                )
            session_state_suffix = self._format_session_state(
                window_state, budget_allocation,
            )
        except Exception as exc:
            logger.warning("assemble session_state_suffix build failed: %s", exc)

        # Step 4c: Inject L0 summary into sessionContext
        summary_context = ""
        try:
            if summary_items:
                summary_content = summary_items[0].get("content", "")
                if summary_content:
                    summary_context = f"## Session Summary\n{summary_content}"
                    logger.info(
                        "L0 summary injected into sessionContext: len=%d score=%.3f",
                        len(summary_context),
                        summary_items[0].get("score", 0),
                    )
        except Exception as exc:
            logger.warning("assemble L0 summary injection failed: %s", exc)

        # Combine session state + L0 summary into session_context
        session_context = "\n\n".join(p for p in [session_state_suffix, summary_context] if p)

        # Step 5d: Trim archived messages from the message list.
        # If archives exist, the archived content is already represented in
        # episodic_context (position 2). Only keep unarchived messages
        # (session buffer) plus the current turn's new messages.
        trimmed_messages = messages
        try:
            session_id_for_buf = params.get("sessionId") or ctx.session_id
            if episodic_context and session_id_for_buf:
                mgr = self.get_session_manager()
                buf = mgr.get_or_create(session_id_for_buf)
                buf_count = len(buf.messages)

                # buf.messages are the unarchived messages recorded by
                # after_turn.  Prefer finding that exact suffix in the caller's
                # message list so current-turn messages after the buffer are
                # also preserved.  If matching fails, fall back to the named
                # tail margin above so this heuristic is explicit.
                pre_prompt_count = _bounded_message_count(
                    params.get("prePromptMessageCount", 0),
                    len(messages),
                )
                pre_prompt_messages = messages[:pre_prompt_count]
                conversation_messages = messages[pre_prompt_count:]

                if buf_count == 0:
                    keep_from = 0
                    trim_reason = "empty_buffer"
                    if conversation_messages:
                        logger.warning(
                            "compose archive trim skipped because session buffer "
                            "is empty; preserving messages: session=%s trace=%s "
                            "total=%d pre_prompt=%d",
                            getattr(ctx, "session_id", ""),
                            getattr(ctx, "trace_id", ""),
                            len(messages),
                            pre_prompt_count,
                        )
                else:
                    keep_from = _find_last_message_sequence_start(
                        conversation_messages,
                        buf.messages,
                    )
                    trim_reason = "matched_buffer"
                    if keep_from is None:
                        if "prePromptMessageCount" not in params:
                            logger.warning(
                                "compose archive trim using tail margin without "
                                "prePromptMessageCount: session=%s trace=%s "
                                "total=%d buf=%d margin=%d",
                                getattr(ctx, "session_id", ""),
                                getattr(ctx, "trace_id", ""),
                                len(messages),
                                buf_count,
                                _ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN,
                            )
                        keep_count = min(
                            len(conversation_messages),
                            buf_count + _ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN,
                        )
                        keep_from = len(conversation_messages) - keep_count
                        trim_reason = "tail_margin"

                if keep_from > 0:
                    kept_messages = conversation_messages[keep_from:]
                    trimmed_messages = pre_prompt_messages + kept_messages
                    logger.info(
                        "compose trimmed archived messages: total=%d buf=%d "
                        "pre_prompt=%d dropped=%d kept=%d reason=%s",
                        len(messages), buf_count, pre_prompt_count, keep_from,
                        len(trimmed_messages), trim_reason,
                    )
        except Exception as exc:
            logger.warning("compose archive trim failed, using full messages: %s", exc)

        # Also trim messages compressed by RollingCompressor if session_state exists.
        # turn_count_at_last_compress is a user-turn count, so derive the exact
        # compressed message span from the runtime buffer and token watermark.
        if session_state_suffix and window_state.turn_count_at_last_compress > 0:
            pre_prompt_count = _bounded_message_count(
                params.get("prePromptMessageCount", 0),
                len(trimmed_messages),
            )
            compressed_turns = window_state.turn_count_at_last_compress
            compressed_tokens = window_state.token_count_at_last_compress
            conversation_count = len(trimmed_messages) - pre_prompt_count

            if pre_prompt_count < len(trimmed_messages) and compressed_tokens > 0:
                conversation_messages = trimmed_messages[pre_prompt_count:]
                compressed_message_count = 0
                try:
                    session_id_for_buf = params.get("sessionId") or ctx.session_id
                    if session_id_for_buf:
                        buf = self.get_session_manager().get_or_create(
                            session_id_for_buf,
                        )
                        user_turns = 0
                        estimated_tokens = 0
                        for idx, msg in enumerate(buf.messages):
                            if msg.role == "user":
                                user_turns += 1
                            estimated_tokens += msg.estimated_tokens
                            if (
                                user_turns == compressed_turns
                                and estimated_tokens == compressed_tokens
                            ):
                                compressed_message_count = idx + 1
                                break
                            if (
                                user_turns > compressed_turns
                                or estimated_tokens > compressed_tokens
                            ):
                                break
                except Exception as exc:
                    logger.debug(
                        "compose compressed trim buffer span check failed; "
                        "preserving caller messages: %s",
                        exc,
                    )

                compression_covers_messages = False
                if (
                    compressed_message_count > 0
                    and len(conversation_messages) >= compressed_message_count
                ):
                    buffer_prefix = buf.messages[:compressed_message_count]
                    caller_prefix = conversation_messages[:compressed_message_count]
                    compression_covers_messages = [
                        _message_role_content(msg)
                        for msg in buffer_prefix
                    ] == [
                        _message_role_content(msg)
                        for msg in caller_prefix
                    ]

                if compression_covers_messages:
                    trimmed_messages = (
                        trimmed_messages[:pre_prompt_count]
                        + conversation_messages[compressed_message_count:]
                    )
                    logger.info(
                        "compose trimmed verified compressed messages "
                        "(compressed_turns=%d, compressed_messages=%d, "
                        "pre_prompt=%d, kept=%d)",
                        compressed_turns,
                        compressed_message_count,
                        pre_prompt_count,
                        len(trimmed_messages),
                    )
                else:
                    logger.debug(
                        "compose skipped compression trim because caller messages "
                        "are not confirmed in verified compressed buffer span "
                        "(compressed_turns=%d, compressed_tokens=%d, "
                        "compressed_messages=%d, pre_prompt=%d, conversation=%d)",
                        compressed_turns,
                        compressed_tokens,
                        compressed_message_count,
                        pre_prompt_count,
                        conversation_count,
                    )

        # Step 6: Assemble final output
        try:
            system_prompt_suffix = "\n\n".join(
                p for p in [episodic_context, session_state_suffix] if p
            )

            archive_count = (
                archive_count_from_cache
                if archive_count_from_cache is not None
                else len(latest_archives) + len(pre_archives)
            )
            prompt_tokens = (
                len(identity_context) // 4
                + len(system_prompt_suffix) // 4
                + len(working_set_msg) // 4
            )
            msg_tokens = self._estimate_tokens(trimmed_messages)

            # Extract open_loops and uncertainties from window_state
            open_loops = window_state.open_loops if window_state else []
            uncertainties = window_state.uncertainties if window_state else []

            # Build budget metadata by slot
            budget_used_by_slot = {
                "identity": len(identity_context) // 4,
                "episodic": len(episodic_context) // 4,
                "session": len(session_context) // 4,
                "retrieved": len(working_set_msg) // 4,
            }

            # Create ComposedContext with all semantic slots populated
            result = ComposedContext(
                # Semantic slots
                identity_context=identity_context,
                episodic_context=episodic_context,
                session_context=session_context,
                task_context="",
                retrieved_evidence=working_set_msg,
                open_loops=open_loops,
                uncertainties=uncertainties,
                # Budget metadata
                estimated_tokens=msg_tokens + prompt_tokens,
                budget_used_by_slot=budget_used_by_slot,
                # Stats for analysis
                stats=search_stats,
                # Legacy compat
                messages=trimmed_messages,
                archive_count=archive_count,
                archive_included=archive_count > 0,
                system_prompt_suffix=system_prompt_suffix,
            )

            return self._to_response(result)
        except Exception as exc:
            logger.warning("assemble final result construction failed: %s", exc, exc_info=True)
            result = ComposedContext(
                messages=trimmed_messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._to_response(result)

    def _to_response(self, result: ComposedContext) -> dict:
        """Convert ComposedContext to dict for the OpenClaw plugin.

        Message injection order (all go into messages as user role for KV-cache):
          1. Profile (identity_context) — stable prefix
          2. Episodic (archive history) — rarely changes
          3. Working Set (retrieved_evidence) — before question to avoid confirmation bias
          4. Original messages — grows each turn
          5. Session state — dynamic suffix

        systemPromptAddition/systemPromptSuffix/memoryUserMessage are empty —
        all context is in messages.  The plugin only passes messages through.
        """
        injected_messages = []

        if result.identity_context:
            injected_messages.append({"role": "user", "content": result.identity_context, "_ogmem": True})

        if result.episodic_context:
            injected_messages.append({"role": "user", "content": result.episodic_context, "_ogmem": True})

        injected_messages.extend(result.messages)

        if result.session_context:
            injected_messages.append({"role": "user", "content": result.session_context, "_ogmem": True})

        if result.retrieved_evidence:
            injected_messages.append({"role": "user", "content": result.retrieved_evidence, "_ogmem": True})

        return {
            "messages": injected_messages,
            "systemPromptAddition": "",
            "systemPromptSuffix": "",
            "memoryUserMessage": "",
            "estimatedTokens": result.estimated_tokens,
            "archiveCount": result.archive_count,
            "archiveIncluded": result.archive_included,
            "stats": result.stats,

            "identityContext": result.identity_context,
            "episodicContext": result.episodic_context or "",
            "sessionContext": result.session_context or "",
            "taskContext": result.task_context,
            "retrievedEvidence": result.retrieved_evidence,
            "openLoops": result.open_loops,
            "uncertainties": result.uncertainties,
            "budgetUsedBySlot": result.budget_used_by_slot,
        }

    def after_turn(self, params: dict) -> dict:
        """Unified write entry point: accumulate session buffer + extract memories.

        Step 1 (every turn): add new messages to session buffer (lightweight).
        Step 2 (threshold reached): extract memories via LLM → write to AGFS →
              drain outbox (embed + upsert vector index) → commit session archive.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        pre_prompt_count = params.get("prePromptMessageCount", 0)
        if not messages:
            return {"ok": True}

        ctx = params.get("_ctx") or self.build_context(params)

        # Extract new messages only (skip pre-prompt system messages)
        new_messages = messages[pre_prompt_count:]
        if not new_messages:
            return {"ok": True, "status": "no_new_messages"}

        request_session_time = None
        for msg in new_messages:
            created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
            request_session_time = _parse_session_time(created_at)
            if request_session_time is not None:
                break

        # Threshold comes from unified config first; env fallback is handled by OgMemConfig.
        threshold = _positive_int(getattr(self._cfg, "after_turn_threshold", 200), 200)

        # Step 1: Accumulate in session buffer (lightweight, every turn)
        # Dedup: skip messages already written by ingest()
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        from extraction.tool_collector import update_buffer_stats
        # Snapshot buffer tail once to avoid TOCTOU race with concurrent ingest()
        buf_snapshot = list(buf.messages[-len(new_messages):]) if buf.messages else []
        for msg in new_messages:
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")
            parsed = extract_content_and_tool_calls(content_raw)
            content = parsed["text"]

            # Dedup against ingest-written messages (match role+content in snapshot)
            if content and buf_snapshot:
                for existing in buf_snapshot:
                    if existing.role == role and existing.content == content:
                        content = None
                        break

            if content:
                created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
                mgr.add_message(session_id, role, content, ctx, created_at=created_at)
            # Accumulate tool call metadata into buffer stats
            if parsed["tool_calls"]:
                update_buffer_stats(buf.usage_stats, parsed["tool_calls"], session_id=session_id)

        # Check pending_tokens
        session = mgr.get_session(session_id, ctx)
        pending_tokens = session.get("pending_tokens", 0)

        if pending_tokens < threshold:
            return {"ok": True, "status": "accumulating", "pending_tokens": pending_tokens}

        # Step 2: Threshold reached → extract memories + write + index + archive
        write_api = self.get_write_api()
        if write_api is None:
            logger.warning("after_turn: write API unavailable, session commit only")
            result = mgr.commit(session_id, ctx, wait=False)
            return {"ok": True, **result}

        # Extract candidates → filter → write to AGFS → create OutboxEvents
        # Use INCREMENTAL messages (after watermark) + extraction summary
        buf = mgr.get_or_create(session_id, ctx=ctx)
        extraction_run_id = uuid4().hex[:12]
        extraction_in_progress_before = buf.extraction_in_progress
        extraction_state = self._build_incremental_extraction_state(buf)
        incremental = extraction_state["incremental"]
        archive_snapshot = list(incremental)
        archive_snapshot_ids = {m.id for m in archive_snapshot}
        extraction_messages = extraction_state["messages"]
        original_extraction_watermark = buf.extraction_watermark

        # Pre-generate archive_id so extraction and commit_snapshot share the
        # same provenance link. commit_session auto-generates when archive_id
        # is None, but after_turn must pass it explicitly across both calls.
        from session.session_manager import generate_archive_id
        archive_id = generate_archive_id()

        def _rewind_failed_archive_watermark() -> bool:
            rewound = buf.rewind_watermark_to_ids(archive_snapshot_ids)
            if not rewound:
                buf.extraction_watermark = min(original_extraction_watermark, len(buf.messages))
            return rewound

        logger.info(
            "after_turn extraction scheduled: run=%s session=%s "
            "pending_tokens=%d buffer_len=%d watermark=%d incremental=%d "
            "extractable=%d active_extractions=%d",
            extraction_run_id,
            session_id[:8],
            pending_tokens,
            len(buf.messages),
            buf.extraction_watermark,
            len(incremental),
            len(extraction_messages),
            getattr(buf, "extraction_active_count", 0),
        )
        if extraction_in_progress_before:
            logger.warning(
                "after_turn overlapping extraction detected: run=%s session=%s "
                "buffer_len=%d watermark=%d active_extractions=%d",
                extraction_run_id,
                session_id[:8],
                len(buf.messages),
                buf.extraction_watermark,
                getattr(buf, "extraction_active_count", 0),
            )
        if not extraction_messages:
            buf.extraction_watermark = len(buf.messages)
            drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
            commit_result = mgr.commit_snapshot(session_id, archive_snapshot, ctx, wait=True, archive_id=archive_id)
            removed = 0
            if commit_result.get("archived"):
                removed = buf.remove_messages_by_id(archive_snapshot_ids)
                if removed:
                    mgr.save_session_state(session_id, ctx)
            logger.info(
                "after_turn no-extractable snapshot commit: run=%s session=%s "
                "archived=%s removed=%d reason=%s",
                extraction_run_id,
                session_id[:8],
                commit_result.get("archived"),
                removed,
                commit_result.get("reason", ""),
            )
            if commit_result.get("status") == "failed":
                rewound = _rewind_failed_archive_watermark()
                logger.error(
                    "after_turn no-extractable snapshot commit failed: "
                    "run=%s session=%s rewound=%s error=%s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    commit_result.get("error", "archive commit failed"),
                )
                return {
                    "ok": False,
                    "status": "failed",
                    "error": commit_result.get("error", "archive commit failed"),
                    "drain": drain_stats,
                    "commit": commit_result,
                    "extraction_run_id": extraction_run_id,
                }
            return {
                "ok": True,
                "status": "no_extractable_messages",
                "drain": drain_stats,
                "commit": commit_result,
                "extraction_run_id": extraction_run_id,
            }

        # Keep the upstream async after_turn behavior, but source extraction
        # inputs from the shared incremental-state helper so after_turn and
        # prepare_compaction operate on the same delta semantics.
        tool_stats_text = extraction_state["tool_stats_text"]
        session_time = extraction_state["session_time"] or request_session_time
        extraction_watermark = original_extraction_watermark
        extraction_summary = buf.extraction_summary
        buf.extraction_watermark = len(buf.messages)

        import time as _time
        buf.begin_extraction()

        def _background_extract_write():
            start_time = _time.monotonic()
            try:
                write_result = self._chunk_and_extract(
                    messages=extraction_messages,
                    write_api=write_api,
                    ctx=ctx,
                    session_time=session_time,
                    session_summary=extraction_summary,
                    tool_stats_text=tool_stats_text,
                    extraction_run_id=extraction_run_id,
                    session_id=session_id,
                    archive_id=archive_id,
                )
                if self._write_result_has_failures(write_result):
                    raise RuntimeError("memory extraction write failed")
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                elapsed_ms = int((_time.monotonic() - start_time) * 1000)
                logger.info(
                    "after_turn background extract done: run=%s session=%s "
                    "extracted=%d filtered=%d writes=%d skipped=%d failed=%d "
                    "watermark=%d->%d buffer_len_now=%d elapsed_ms=%d",
                    extraction_run_id,
                    session_id[:8],
                    write_result.get("candidates_extracted", 0),
                    write_result.get("candidates_filtered", 0),
                    write_result.get("writes_completed", 0),
                    write_result.get("writes_skipped", 0),
                    write_result.get("writes_failed", 0),
                    extraction_watermark,
                    buf.extraction_watermark,
                    len(buf.messages),
                    elapsed_ms,
                )
                if write_result.get("candidates_extracted", 0) == 0:
                    logger.warning(
                        "after_turn zero-candidate diagnostic: run=%s session=%s "
                        "extractable=%d extraction_diag=%s write_result_keys=%s",
                        extraction_run_id,
                        session_id[:8],
                        len(extraction_messages),
                        _message_diag(extraction_messages),
                        sorted(write_result.keys()),
                    )

                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(
                        buf.extraction_summary, write_result
                    )

                try:
                    self._async_drain()
                except Exception as exc:
                    logger.warning("outbox_drain failed after background extract: %s", exc)

                commit_result = mgr.commit_snapshot(session_id, archive_snapshot, ctx, wait=True, archive_id=archive_id)
                removed = 0
                if commit_result.get("status") == "failed":
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn background snapshot commit failed: "
                        "run=%s session=%s rewound=%s error=%s",
                        extraction_run_id,
                        session_id[:8],
                        rewound,
                        commit_result.get("error", "archive commit failed"),
                    )
                elif commit_result.get("archived"):
                    removed = buf.remove_messages_by_id(archive_snapshot_ids)
                    if removed:
                        mgr.save_session_state(session_id, ctx)
                logger.info(
                    "after_turn background snapshot commit done: run=%s session=%s "
                    "archived=%s reason=%s removed=%d buffer_len=%d watermark=%d",
                    extraction_run_id,
                    session_id[:8],
                    commit_result.get("archived"),
                    commit_result.get("reason", ""),
                    removed,
                    len(buf.messages),
                    buf.extraction_watermark,
                )
            except Exception as exc:
                rewound = buf.rewind_watermark_to_ids(archive_snapshot_ids)
                logger.error(
                    "after_turn background extract failed for run=%s session=%s "
                    "rewound=%s: %s",
                    extraction_run_id, session_id[:8], rewound, exc, exc_info=True,
                )
            finally:
                buf.end_extraction()

        try:
            t = threading.Thread(
                target=_background_extract_write, daemon=True,
                name=f"extract-{session_id[:8]}",
            )
            t.start()
        except Exception as exc:
            # Thread spawn failed — fall back to synchronous (better than losing data)
            buf.end_extraction()
            logger.error(
                "Failed to spawn background extract thread: run=%s session=%s %s",
                extraction_run_id,
                session_id[:8],
                exc,
                exc_info=True,
            )
            # Sync fallback: use chunking-aware extraction
            write_result = self._chunk_and_extract(
                messages=extraction_messages,
                write_api=write_api,
                ctx=ctx,
                session_time=session_time,
                session_summary=extraction_summary,
                tool_stats_text=tool_stats_text,
                extraction_run_id=extraction_run_id,
                session_id=session_id,
                archive_id=archive_id,
            )
            if self._write_result_has_failures(write_result):
                rewound = _rewind_failed_archive_watermark()
                logger.error(
                    "after_turn sync extraction write failed: "
                    "run=%s session=%s rewound=%s write_result=%s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    write_result,
                )
                return {
                    **write_result,
                    "ok": False,
                    "status": "failed",
                    "error": "memory extraction write failed",
                    "commit": None,
                    "extraction_run_id": extraction_run_id,
                }
            buf.extraction_watermark = len(buf.messages)
            self._invalidate_identity_cache_if_profile_written(session_id, write_result)
            if write_result.get("candidates_extracted", 0) > 0:
                buf.extraction_summary = _update_extraction_summary(
                    buf.extraction_summary, write_result
                )
            try:
                self._async_drain()
            except Exception:
                pass
            commit_result = mgr.commit_snapshot(session_id, archive_snapshot, ctx, wait=True, archive_id=archive_id)
            removed = 0
            if commit_result.get("status") == "failed":
                rewound = _rewind_failed_archive_watermark()
                logger.error(
                    "after_turn sync snapshot commit failed: "
                    "run=%s session=%s rewound=%s error=%s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    commit_result.get("error", "archive commit failed"),
                )
                return {
                    **write_result,
                    "ok": False,
                    "status": "failed",
                    "error": commit_result.get("error", "archive commit failed"),
                    "commit": commit_result,
                    "extraction_run_id": extraction_run_id,
                }
            if commit_result.get("archived"):
                removed = buf.remove_messages_by_id(archive_snapshot_ids)
                if removed:
                    mgr.save_session_state(session_id, ctx)
            logger.info(
                "after_turn sync snapshot commit done: run=%s session=%s "
                "archived=%s removed=%d reason=%s",
                extraction_run_id,
                session_id[:8],
                commit_result.get("archived"),
                removed,
                commit_result.get("reason", ""),
            )
            return {
                "ok": True,
                "status": "completed",
                "extraction_run_id": extraction_run_id,
                **write_result,
            }

        return {"ok": True, "status": "processing", "extraction_run_id": extraction_run_id}

    def ingest(self, params: dict) -> dict:
        """Ingest messages into the session buffer.

        Writes messages to the session buffer so that compact() can find data
        even when called before after_turn().  after_turn() will deduplicate
        messages already written here.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if not messages:
            return {"ingested": True}

        ctx = params.get("_ctx") or self.build_context(params)
        mgr = self.get_session_manager()

        count = 0
        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content_raw = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            content = extract_content_and_tool_calls(content_raw).get("text", "")
            created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
            if content:
                mgr.add_message(session_id, role, content, ctx, created_at=created_at)
                count += 1

        return {"ingested": True, "count": count}

    def ingest_batch(self, params: dict) -> dict:
        """Batch ingest — same as ingest() for multiple messages."""
        return self.ingest(params)

    def compact(self, params: dict) -> dict:
        """Compact session: commit synchronously, return compressed context."""
        prepare_token = params.get("prepareToken")
        prepared_flag = bool(params.get("prepared"))
        if prepare_token:
            prepared = self._consume_prepared_compaction_state(params, str(prepare_token))
        elif prepared_flag:
            raise ValueError("prepare token required when prepared=true")
        else:
            prepared = self.prepare_compaction(params)
        session_id = prepared["session_id"]
        ctx = prepared["ctx"]
        token_budget = params.get("tokenBudget", 128_000)
        mgr = prepared["mgr"]

        # Estimate tokensBefore
        session = mgr.get_session(session_id, ctx)
        tokens_before = session.get("pending_tokens", 0)

        # Commit synchronously (wait=true)
        commit_result = mgr.commit(
            session_id,
            ctx,
            wait=True,
            archive_id=prepared.get("archive_id"),
        )

        if not commit_result.get("archived"):
            if commit_result.get("status") == "failed":
                return {
                    "ok": False,
                    "compacted": False,
                    "status": "failed",
                    "error": commit_result.get("error", "archive commit failed"),
                    "commit": commit_result,
                    "result": {
                        "summary": "",
                        "tokensBefore": tokens_before,
                    },
                }
            return {
                "ok": True,
                "compacted": False,
                "reason": commit_result.get("reason", "no_archive"),
                "result": {
                    "summary": "",
                    "tokensBefore": tokens_before,
                },
            }

        # Get compressed context (archive overview as summary)
        context = mgr.get_context(session_id, token_budget, ctx)
        summary = _trim_summary(
            context.get("latest_archive_overview", ""),
            _summary_max_chars(self._cfg, params),
        )
        tokens_after = context.get("estimatedTokens", 0)
        self._apply_compact_short_term_index_mode(params)

        result = {
            "ok": True,
            "compacted": True,
            "result": {
                "summary": summary,
                "tokensBefore": tokens_before,
                "tokensAfter": tokens_after,
            },
        }
        first_kept_entry_id = (
            commit_result.get("firstKeptEntryId")
            or commit_result.get("first_kept_entry_id")
        )
        if first_kept_entry_id:
            result["result"]["firstKeptEntryId"] = first_kept_entry_id
        return result

    def _apply_compact_short_term_index_mode(self, params: dict) -> None:
        """Apply Phase 1 short-term index semantics without blocking compact success."""
        mode = _short_term_index_mode(params)
        if mode == "off":
            logger.info("compact short-term index update skipped: mode=off")
            return
        if mode == "async":
            logger.info("compact short-term index update deferred: mode=async")
            return

        try:
            drain_stats = self.drain_outbox_sync()
            logger.info(
                "compact short-term index sync processed=%d succeeded=%d failed=%d",
                drain_stats.get("processed", 0),
                drain_stats.get("succeeded", 0),
                drain_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.warning("compact short-term index sync skipped: %s", exc)

    def _build_compaction_state(self, params: dict) -> dict:
        """Build transport-agnostic compact state without mutating extraction progress."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        extraction_state = self._build_incremental_extraction_state(buf)

        result = {
            "session_id": session_id,
            "ctx": ctx,
            "mgr": mgr,
            "buffer": buf,
            **extraction_state,
        }

        return result

    def _build_incremental_extraction_state(self, buf) -> dict:
        incremental = buf.messages[buf.extraction_watermark:]
        extraction_messages = [
            {"role": message.role, "content": message.content, "id": message.id}
            for message in incremental
            if message.role != "assistant"
        ]
        from extraction.tool_collector import format_tool_stats_text

        session_time = None
        time_source_messages = [
            message for message in incremental
            if message.role != "assistant"
        ] or incremental or buf.messages
        for message in time_source_messages:
            session_time = _parse_session_time(getattr(message, "created_at", None))
            if session_time is not None:
                break

        # Fallback: if created_at is missing or very recent (<24h, likely auto-generated),
        # try to extract date from message content like '[group chat conversation: ...]'.
        now = _dt.now(session_time.tzinfo) if session_time and session_time.tzinfo else _dt.now()
        if session_time is None or abs((now - session_time).total_seconds()) < 86400:
            for message in time_source_messages:
                content = getattr(message, "content", "")
                extracted = _extract_date_from_content(content)
                if extracted is not None:
                    session_time = extracted
                    break

        return {
            "incremental": incremental,
            "messages": extraction_messages,
            "session_time": session_time,
            "tool_stats_text": format_tool_stats_text(buf.usage_stats),
        }

    def _consume_prepared_compaction_state(self, params: dict, prepare_token: str) -> dict:
        result = self._build_compaction_state(params)
        ttl_seconds = _positive_int(getattr(self._cfg, "compact_prepare_token_ttl", 300), 300)
        archive_id = result["mgr"].get_compaction_prepare_archive_id(result["session_id"])
        if not result["mgr"].consume_compaction_prepare_token(
            result["session_id"],
            prepare_token,
            ttl_seconds=ttl_seconds,
        ):
            raise ValueError("invalid prepare token")
        result["prepareToken"] = prepare_token
        result["archive_id"] = archive_id or None
        return result

    @staticmethod
    def _write_result_has_failures(write_result: dict | None) -> bool:
        if not isinstance(write_result, dict):
            return True
        if write_result.get("ok") is False or write_result.get("status") == "failed":
            return True
        try:
            return int(write_result.get("writes_failed", 0) or 0) > 0
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _plan_field(plan, name: str, default=None):
        if isinstance(plan, dict):
            return plan.get(name, default)
        return getattr(plan, name, default)

    @classmethod
    def _write_result_updates_profile(cls, write_result: dict | None) -> bool:
        if not isinstance(write_result, dict):
            return False

        for plan in write_result.get("plans") or []:
            if cls._plan_field(plan, "action") == "skip":
                continue
            category = str(cls._plan_field(plan, "category", "") or "").lower()
            if category == "profile":
                return True

            uri = str(
                cls._plan_field(plan, "target_uri", None)
                or cls._plan_field(plan, "uri", "")
                or ""
            ).rstrip("/")
            if uri.endswith("/memories/profile") or "/memories/profile/" in uri:
                return True

        return False

    def _invalidate_identity_cache_if_profile_written(
        self,
        session_id: str,
        write_result: dict | None,
    ) -> None:
        if self._write_result_updates_profile(write_result):
            self.get_session_manager().invalidate_topic_slot(session_id, "identity")

    def prepare_compaction(self, params: dict) -> dict:
        """Prepare incremental extraction state before compact commit."""
        result = self._build_compaction_state(params)
        result["mgr"].load_session_state(result["session_id"], result["ctx"])
        buf = result["buffer"]
        incremental = result["incremental"]
        ttl_seconds = _positive_int(getattr(self._cfg, "compact_prepare_token_ttl", 300), 300)

        existing_prepare = result["mgr"].get_compaction_prepare_state(
            result["session_id"],
            ttl_seconds=ttl_seconds,
        )
        if existing_prepare is not None:
            result["prepareToken"] = existing_prepare["prepareToken"]
            result["archive_id"] = existing_prepare.get("archive_id") or None
            return result

        from session.session_manager import generate_archive_id
        archive_id = generate_archive_id()
        result["archive_id"] = archive_id

        if not incremental:
            result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
                result["session_id"],
                archive_id=archive_id,
            )
            return result

        if not result["messages"]:
            buf.extraction_watermark = len(buf.messages)
            result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
                result["session_id"],
                archive_id=archive_id,
            )
            return result

        write_api = self.get_write_api()
        if write_api is None:
            raise RuntimeError("write API unavailable for compaction")

        # Apply chunking if enabled
        write_result = self._chunk_and_extract(
            messages=result["messages"],
            write_api=write_api,
            ctx=result["ctx"],
            session_time=result["session_time"],
            session_summary=buf.extraction_summary,
            tool_stats_text=result["tool_stats_text"],
            extraction_run_id="prepare_compaction",
            session_id=result["session_id"],
            archive_id=archive_id,
        )
        if self._write_result_has_failures(write_result):
            raise RuntimeError("compaction extraction write failed")
        self._invalidate_identity_cache_if_profile_written(result["session_id"], write_result)
        buf.extraction_watermark = len(buf.messages)
        if write_result.get("candidates_extracted", 0) > 0:
            buf.extraction_summary = _update_extraction_summary(
                buf.extraction_summary,
                write_result,
            )
        result["prepareToken"] = result["mgr"].issue_compaction_prepare_token(
            result["session_id"],
            archive_id=archive_id,
        )
        result["write_result"] = write_result
        return result

    def bootstrap(self, params: dict) -> dict:
        return {"bootstrapped": True}

    def prepare_subagent_spawn(self, params: dict) -> dict:
        return {"prepared": True}

    def on_subagent_ended(self, params: dict) -> dict:
        return {"cleaned": True}

    def dispose(self, params: dict | None = None) -> dict:
        """Session end: dispatch background flush for pending memories."""
        params = params or {}
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"disposed": True, "flushed": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        if not mgr.has_session(session_id):
            return {"disposed": True, "flushed": False, "reason": "session_not_found"}

        buf = mgr.get_or_create(session_id)
        ctx = params.get("_ctx")

        # Wait for any in-progress background extraction to finish before
        # touching the session.  This prevents the race where after_turn()
        # has optimistically advanced extraction_watermark but the background
        # thread hasn't committed yet — removing the session here would cause
        # the background commit to fail and lose data.
        if buf.extraction_in_progress and buf.extraction_done_event is not None:
            logger.info("dispose: waiting for background extraction on session=%s", session_id[:8])
            completed = buf.extraction_done_event.wait(timeout=30)
            if not completed and buf.extraction_in_progress:
                logger.warning(
                    "dispose: background extraction still in progress after timeout; "
                    "keeping session=%s active_extractions=%d watermark=%d total=%d",
                    session_id[:8],
                    getattr(buf, "extraction_active_count", 0),
                    buf.extraction_watermark,
                    len(buf.messages),
                )
                return {
                    "disposed": True,
                    "flushed": False,
                    "reason": "extraction_in_progress_timeout",
                    "extraction_in_progress": True,
                    "active_extractions": getattr(buf, "extraction_active_count", 0),
                }

        unextracted = buf.messages[buf.extraction_watermark:]
        if not unextracted:
            self._save_session_state_best_effort(mgr, session_id, params)
            self._flush_session_tool_usage_async(session_id, ctx)
            mgr.remove_session(session_id)
            return {"disposed": True, "flushed": False, "reason": "no_pending_messages"}

        pending_tokens = sum(len(m.content) // 4 for m in unextracted)
        _MIN_FLUSH_TOKENS = 200
        if pending_tokens < _MIN_FLUSH_TOKENS:
            logger.info(
                "dispose flush skipped: session=%s pending_tokens=%d < %d",
                session_id[:8], pending_tokens, _MIN_FLUSH_TOKENS,
            )
            self._save_session_state_best_effort(mgr, session_id, params)
            self._flush_session_tool_usage_async(session_id, ctx)
            mgr.remove_session(session_id)
            return {"disposed": True, "flushed": False, "reason": "below_token_threshold",
                    "pending_tokens": pending_tokens}

        extraction_messages = [
            {"role": m.role, "content": m.content}
            for m in unextracted if m.role != "assistant"
        ]
        if not extraction_messages:
            self._save_session_state_best_effort(mgr, session_id, params)
            self._flush_session_tool_usage_async(session_id, ctx)
            mgr.remove_session(session_id)
            return {"disposed": True, "flushed": False, "reason": "no_user_messages"}

        # Check write_api BEFORE removing the session so that we can bail
        # out safely without losing pending data.
        write_api = self.get_write_api()
        if write_api is None:
            logger.warning("dispose: write API unavailable, keeping session for retry session=%s", session_id[:8])
            # Do NOT remove the session — a subsequent dispose or flush can retry.
            return {"disposed": True, "flushed": False, "reason": "no_write_api"}

        ctx = ctx or self.build_context(params)
        extraction_summary = buf.extraction_summary
        tool_stats_text = ""
        try:
            from extraction.tool_collector import format_tool_stats_text
            tool_stats_text = format_tool_stats_text(buf.usage_stats)
        except Exception:
            pass

        buf.extraction_watermark = len(buf.messages)
        self._save_session_state_best_effort(mgr, session_id, params)
        mgr.remove_session(session_id)

        logger.info(
            "dispose: dispatching background flush for session=%s msgs=%d tokens=%d",
            session_id[:8], len(extraction_messages), pending_tokens,
        )

        svc = self

        def _background_flush():
            try:
                write_result = svc._chunk_and_extract(
                    messages=extraction_messages,
                    write_api=write_api,
                    ctx=ctx,
                    session_time=None,
                    session_summary=extraction_summary,
                    tool_stats_text=tool_stats_text,
                    extraction_run_id="dispose",
                    session_id=session_id,
                )
                logger.info(
                    "dispose background flush done: session=%s extracted=%d writes=%d",
                    session_id[:8],
                    write_result.get("candidates_extracted", 0),
                    write_result.get("writes_completed", 0),
                )
                try:
                    svc.drain_outbox_sync()
                except Exception:
                    pass
                svc._flush_session_tool_usage_async(session_id, ctx)
            except Exception:
                logger.exception("dispose background flush failed: session=%s", session_id[:8])
                svc._flush_session_tool_usage_async(session_id, ctx)

        import threading
        threading.Thread(target=_background_flush, daemon=True).start()
        return {"disposed": True, "flushed": True, "status": "background_flush_dispatched",
                "pending_tokens": pending_tokens}

    def wait_until_idle(self, params: dict | None = None) -> dict:
        """Wait for process-local extraction/indexing to settle without removing the session."""
        params = params or {}
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"idle": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        ctx = params.get("_ctx") or self.build_context(params)
        drain_outbox = self._param_bool(params, "drainOutbox", "drain_outbox", default=True)
        wait_outbox = self._param_bool(params, "waitOutbox", "wait_outbox", default=not drain_outbox)
        if not mgr.has_session(session_id):
            if drain_outbox:
                drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
                return {
                    "idle": True,
                    "reason": "session_not_found",
                    "extraction_in_progress": False,
                    "drain": drain_stats,
                }
            outbox_wait = (
                self._wait_for_outbox_idle(ctx, self._idle_timeout_seconds(params))
                if wait_outbox
                else {"idle": True, "outbox": {"skipped": True}}
            )
            if not outbox_wait.get("idle", True):
                return {
                    "idle": False,
                    "reason": "outbox_in_progress_timeout",
                    "extraction_in_progress": False,
                    "outbox": outbox_wait.get("outbox", {}),
                }
            return {
                "idle": True,
                "reason": "session_not_found",
                "extraction_in_progress": False,
                "outbox": outbox_wait.get("outbox", {}),
            }

        buf = mgr.get_or_create(session_id)
        timeout_seconds = self._idle_timeout_seconds(params)

        if buf.extraction_in_progress:
            event = buf.extraction_done_event
            completed = bool(event.wait(timeout=timeout_seconds)) if event is not None else False
            if not completed and buf.extraction_in_progress:
                return {
                    "idle": False,
                    "reason": "extraction_in_progress_timeout",
                    "extraction_in_progress": True,
                    "active_extractions": getattr(buf, "extraction_active_count", 0),
                }

        if drain_outbox:
            drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
            return {
                "idle": True,
                "reason": "idle",
                "extraction_in_progress": False,
                "drain": drain_stats,
            }

        outbox_wait = (
            self._wait_for_outbox_idle(ctx, timeout_seconds)
            if wait_outbox
            else {"idle": True, "outbox": {"skipped": True}}
        )
        if not outbox_wait.get("idle", True):
            return {
                "idle": False,
                "reason": "outbox_in_progress_timeout",
                "extraction_in_progress": False,
                "outbox": outbox_wait.get("outbox", {}),
            }
        return {
            "idle": True,
            "reason": "idle",
            "extraction_in_progress": False,
            "outbox": outbox_wait.get("outbox", {}),
        }

    @staticmethod
    def _param_bool(params: dict, camel_name: str, snake_name: str, *, default: bool) -> bool:
        value = params.get(camel_name, params.get(snake_name, default))
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _idle_timeout_seconds(params: dict) -> float:
        raw_timeout = params.get("timeoutSeconds")
        if raw_timeout is None:
            raw_timeout = params.get("timeout_seconds")
        if raw_timeout is None:
            raw_timeout = 30
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_seconds = 30.0
        return max(0.0, timeout_seconds)

    def _wait_for_outbox_idle(self, ctx: RequestContext, timeout_seconds: float) -> dict:
        """Poll outbox state without claiming events from the background worker."""
        import time as _time

        deadline = _time.monotonic() + timeout_seconds
        last_counts = self._count_outbox_unfinished(ctx)
        while True:
            total = int(last_counts.get("total") or 0)
            if total <= 0 or last_counts.get("supported") is False:
                return {"idle": True, "outbox": last_counts}
            if _time.monotonic() >= deadline:
                return {"idle": False, "outbox": last_counts}
            _time.sleep(min(1.0, max(0.0, deadline - _time.monotonic())))
            last_counts = self._count_outbox_unfinished(ctx)

    def _count_outbox_unfinished(self, ctx: RequestContext) -> dict:
        try:
            fs = self._get_context_fs()
            outbox_store = self._get_outbox_store(fs) if fs is not None else None
            if outbox_store is not None and hasattr(outbox_store, "count_unfinished"):
                return outbox_store.count_unfinished(account_id=ctx.account_id)
            return {"pending": 0, "processing": 0, "total": 0, "supported": False}
        except Exception as exc:
            logger.warning("outbox idle status unavailable: %s", exc)
            return {"pending": 0, "processing": 0, "total": 0, "supported": False, "error": str(exc)}

    def _save_session_state_best_effort(
        self,
        mgr: SessionManager,
        session_id: str,
        params: dict,
    ) -> None:
        try:
            ctx = params.get("_ctx") or self.build_context(params)
            mgr.save_session_state(session_id, ctx)
        except Exception as exc:
            logger.warning("dispose session state save skipped session=%s: %s", session_id, exc)

    def get_cumulative_token_usage(self, reset: bool = False) -> dict:
        result: dict = {}
        internal_tool_counts = self._internal_tool_usage.count_snapshot()
        llm = self._llm
        if llm and hasattr(llm, "token_tracker"):
            tracker = llm.token_tracker
            snap = tracker.snapshot_and_reset() if reset else tracker.snapshot()
            result["llm"] = {
                "input_tokens": snap.input_tokens,
                "output_tokens": snap.output_tokens,
                "cache_read": snap.cache_read,
                "cache_write": snap.cache_write,
                "total_tokens": snap.input_tokens + snap.output_tokens,
                "calls": snap.llm_calls,
            }
        if self._embedder and hasattr(self._embedder, "token_tracker"):
            tracker = self._embedder.token_tracker
            snap = tracker.snapshot_and_reset() if reset else tracker.snapshot()
            result["embedding"] = {"total_tokens": snap.embed_tokens, "calls": snap.embed_calls}
        if internal_tool_counts["rounds"] or internal_tool_counts["tool_calls"]:
            result["internal_tool_rounds"] = internal_tool_counts
        if reset:
            self._internal_tool_usage.reset()
        if result:
            result["total_tokens"] = (
                result.get("llm", {}).get("total_tokens", 0)
                + result.get("embedding", {}).get("total_tokens", 0)
            )
        return result

    def _flush_session_tool_usage_async(
        self,
        session_id: str,
        ctx: RequestContext | None = None,
    ) -> None:
        snapshot = self._internal_tool_usage.session_snapshot(session_id)
        if not snapshot:
            self._internal_tool_usage.clear_session(session_id)
            return
        if not snapshot.get("account_id"):
            snapshot["account_id"] = ctx.account_id if ctx is not None else self._default_account_id
        with self._pending_tool_usage_lock:
            self._pending_tool_usage_snapshots[session_id] = snapshot

        def _write_snapshot():
            try:
                self.get_internal_tool_usage_store().write_session(snapshot)
                with self._pending_tool_usage_lock:
                    self._pending_tool_usage_snapshots.pop(session_id, None)
            except Exception as exc:
                logger.warning("tool usage AGFS write failed: session=%s error=%s", session_id[:8], exc)
                with self._pending_tool_usage_lock:
                    self._pending_tool_usage_snapshots[session_id] = snapshot
            finally:
                self._internal_tool_usage.clear_session(session_id)

        threading.Thread(target=_write_snapshot, daemon=True, name=f"tool-usage-{session_id[:8]}").start()

    def get_tool_usage_stats(self, params: dict | None = None) -> dict:
        params = params or {}
        session_id = params.get("session_id") or params.get("sessionId")
        user_id = params.get("user_id") or params.get("userId")
        start_time = params.get("start_time") or params.get("startTime")
        end_time = params.get("end_time") or params.get("endTime")
        pipeline_name = params.get("pipeline")
        include_rounds = str(params.get("include_rounds") or params.get("includeRounds") or "").lower() == "true"
        stats = self._internal_tool_usage.get_stats(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            pipeline_name=pipeline_name,
            include_rounds=include_rounds,
        )
        if session_id and not stats["summary"]["llm_tool_rounds"] and not stats["summary"]["tool_calls"]:
            with self._pending_tool_usage_lock:
                pending = deepcopy(self._pending_tool_usage_snapshots.get(session_id))
            if pending:
                if not include_rounds:
                    pending.pop("rounds", None)
                    pending.pop("rounds_limit", None)
                    pending.pop("rounds_truncated", None)
                return pending
            try:
                account_id = params.get("accountId") or params.get("account_id") or self._default_account_id
                return self.get_internal_tool_usage_store().read_session(
                    account_id,
                    session_id,
                    include_rounds=include_rounds,
                )
            except Exception:
                logger.debug("tool_usage: AGFS session stats unavailable", exc_info=True)
        return stats

    def health(self) -> dict:
        agfs_required = self._cfg.storage_backend == "agfs"
        sql_required = self._cfg.storage_backend == "sql"
        info: dict = {
            "backend": "og-memory",
            "storage_backend": self._cfg.storage_backend,
            "agfs": agfs_required and _HAS_AGFS,
            "sql": None,
        }
        try:
            if agfs_required:
                if not _HAS_AGFS:
                    raise RuntimeError(
                        "AGFS backend configured but AGFS client is unavailable"
                    )
                AGFSClient(api_base_url=self._agfs_base_url).ls("/")
                info["agfs_url"] = self._agfs_base_url
            elif sql_required:
                dsn = self._cfg.sql_connection_string
                if not dsn:
                    raise RuntimeError(
                        "SQL backend configured but sql_connection_string is empty"
                    )
                pool = self._get_shared_sql_pool()
                conn = pool.get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    conn.rollback()
                finally:
                    pool.return_connection(conn)
                info["sql"] = "connected"
            llm = self.get_llm()
            info["llm"] = getattr(llm, "model", "unknown")
            info["status"] = "ok"
        except Exception as exc:
            info["status"] = "error"
            info["error"] = str(exc)
        return info
