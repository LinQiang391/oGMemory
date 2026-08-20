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
import hashlib
import hmac
import json
import time
from copy import deepcopy
from dataclasses import replace
from datetime import UTC
from datetime import datetime as _dt
from functools import wraps
from uuid import uuid4

from core.models import (
    ArchiveRef,
    ComposedContext,
    ContextNode,
    RequestContext,
    RetrievalConfig,
    Role,
    SeedHit,
    TokenBudget,
)

# Chunking support
from extraction.chunking import ConversationChunker
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
from session.models import ArchiveEntry, SessionMessage, SessionWindowState
from session.rolling_compressor import RollingCompressor
from session.session_manager import SessionManager
from session.topic_buffer import SlotContent

try:
    from pyagfs import AGFSClient

    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter import AGFSContextFS, AGFSTraceEventStore
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

_TOOL_RESULT_GOVERNANCE_FALLBACK_STATE = "tool_result_governance_raw_fallbacks"
_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS = 8_192

# Fallback archive-trim safety margin used only when incoming messages cannot
# be aligned with the session buffer.  It preserves a likely in-flight
# user/assistant turn that has not reached after_turn yet.
_ARCHIVE_TRIM_UNMATCHED_TAIL_MARGIN = 2
# User-accepted replay-protection window. Sessions exceeding this many
# completed semantic turns may replay a tombstone that has aged out.
_SEMANTIC_TURN_COMPLETED_KEY_LIMIT = 1_024


# ---------------------------------------------------------------------------
# Helpers (stateless, reusable)
# ---------------------------------------------------------------------------

def _serialize_lazy_initialization(attribute: str):
    """Protect only the first initialization of a process-wide resource."""
    def _decorate(method):
        @wraps(method)
        def _wrapped(self, *args, **kwargs):
            if getattr(self, attribute) is not None:
                return method(self, *args, **kwargs)
            with self._init_lock:
                return method(self, *args, **kwargs)

        return _wrapped

    return _decorate


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

    def __init__(self, config: OgMemConfig | None = None, tool_result_store=None):
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
        self._trace_event_store = None
        self._skill_sedimentation_service = None
        self._skill_sedimentation_worker = None
        self._skill_sedimentation_thread: threading.Thread | None = None

        # Shared vector index + embedder (used by both read API and outbox worker)
        self._vector_index = None
        self._projection_index = None
        self._lexical_index = None
        self._lexical_backfill = None
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
        self._init_lock = threading.RLock()
        self._tool_result_store = None
        self._tool_result_externalizer = None
        self._tool_result_turn_ledger = None
        self._tool_result_io_runner = None
        self._tool_result_source_attestation_key = os.urandom(32)
        if getattr(self._cfg, "tool_result_externalization_enabled", False):
            self._initialize_tool_result_externalization(tool_result_store)
        if self._cfg.skill_sedimentation_enabled:
            self._start_skill_sedimentation_worker()

    def _initialize_tool_result_externalization(self, store=None) -> None:
        """Build the optional component graph without changing the service factory."""
        from features.tool_result_externalization.externalizer import (
            BoundedDaemonIORunner,
            ToolResultExternalizer,
        )
        from features.tool_result_externalization.normalizer import ToolResultNormalizer
        from features.tool_result_externalization.policy import ToolResultPolicy
        from features.tool_result_externalization.store_base import (
            UnavailableToolResultArtifactStore,
        )
        from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder

        if store is None:
            try:
                if self._use_sql:
                    from features.tool_result_externalization.store_sql import (
                        SQLToolResultArtifactStore,
                    )

                    store = SQLToolResultArtifactStore(pool=self._get_shared_sql_pool())
                elif _HAS_AGFS:
                    from features.tool_result_externalization.store_agfs import (
                        AGFSToolResultArtifactStore,
                    )

                    store = AGFSToolResultArtifactStore(
                        AGFSClient(api_base_url=self._agfs_base_url),
                        mount_prefix=self._mount_prefix,
                    )
                else:
                    raise RuntimeError("no configured SQL or AGFS artifact backend")
            except Exception as exc:
                logger.warning("tool-result artifact store initialization failed: %s", exc)
                store = UnavailableToolResultArtifactStore(exc)
        self._tool_result_store = store
        self._tool_result_io_runner = BoundedDaemonIORunner()
        self._tool_result_externalizer = ToolResultExternalizer(
            ToolResultNormalizer(),
            ToolResultPolicy(self._cfg.tool_result_single_chars),
            ToolResultSynopsisBuilder(self._cfg.tool_result_synopsis_max_chars),
            store,
            retry_attempts=self._cfg.tool_result_retry_attempts,
            write_timeout_ms=self._cfg.tool_result_write_timeout_ms,
            summary_per_artifact_chars=self._cfg.tool_result_synopsis_max_chars,
            summary_per_turn_chars=self._cfg.tool_result_stub_summary_turn_chars,
            source_attestation_key=self._tool_result_source_attestation_key,
            io_runner=self._tool_result_io_runner,
        )
        from features.tool_result_externalization.turn_ledger import (
            ToolResultTurnLedger,
        )

        self._tool_result_turn_ledger = ToolResultTurnLedger(
            self._tool_result_source_attestation_key,
        )

    @property
    def _use_sql(self) -> bool:
        return self._cfg.storage_backend == "sql" and _HAS_SQL and self._cfg.sql_connection_string

    @_serialize_lazy_initialization("_sql_pool")
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

    @_serialize_lazy_initialization("_trace_event_store")
    def _get_trace_event_store(self):
        """Return the lifecycle-trace store for the configured backend."""
        if self._trace_event_store is None and self._use_sql:
            try:
                from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

                self._trace_event_store = SQLTraceEventStore(
                    pool=self._get_shared_sql_pool(),
                    max_turn_events=self._cfg.trace_max_turn_events,
                )
            except Exception as exc:
                logger.warning("SQLTraceEventStore initialization failed: %s", exc)
                return None
        elif self._trace_event_store is None and self._cfg.storage_backend == "agfs" and _HAS_AGFS:
            try:
                self._trace_event_store = AGFSTraceEventStore(
                    client=AGFSClient(api_base_url=self._agfs_base_url),
                    mount_prefix=self._mount_prefix,
                    max_turn_events=self._cfg.trace_max_turn_events,
                )
            except Exception as exc:
                logger.warning("AGFSTraceEventStore initialization failed: %s", exc)
                return None
        return self._trace_event_store

    def _get_skill_sedimentation_service(self):
        if not self._cfg.skill_sedimentation_enabled:
            return None
        if self._skill_sedimentation_service is None:
            from skills.sedimentation import SedimentationSettings, SkillSedimentationService

            trace_store = self._get_trace_event_store()
            context_fs = self._get_context_fs()
            if trace_store is None or context_fs is None:
                raise RuntimeError("skill sedimentation stores are unavailable")
            assessor_cfg = replace(self._provider_cfg, llm_temperature=0.0)
            reviewer_cfg = replace(
                self._provider_cfg,
                provider=self._cfg.skill_reviewer_provider or self._provider_cfg.provider,
                openai_llm_model=(
                    self._cfg.skill_reviewer_model or self._provider_cfg.openai_llm_model
                ),
                llm_temperature=0.0,
            )
            try:
                embedder = self._get_shared_embedder()
            except Exception as exc:
                raise RuntimeError(
                    "skill sedimentation requires semantic embeddings for safe deduplication"
                ) from exc
            self._skill_sedimentation_service = SkillSedimentationService(
                trace_store=trace_store,
                context_fs=context_fs,
                assessor_llm=assessor_cfg.create_llm(),
                reviewer_llm=reviewer_cfg.create_llm(),
                settings=SedimentationSettings.from_config(self._cfg),
                embedder=embedder,
            )
        return self._skill_sedimentation_service

    def _start_skill_sedimentation_worker(self) -> None:
        if self._skill_sedimentation_thread is not None:
            if self._skill_sedimentation_thread.is_alive():
                return
            self._skill_sedimentation_thread = None
            self._skill_sedimentation_worker = None
        from skills.sedimentation import SkillSedimentationWorker

        skill_service = self._get_skill_sedimentation_service()
        if skill_service is None:
            return
        self._skill_sedimentation_worker = SkillSedimentationWorker(
            skill_service,
            self._skill_account_ids,
            interval_seconds=self._cfg.skill_scan_interval_seconds,
            worker_id=f"skill-{uuid4().hex[:16]}",
            batch_size=self._cfg.skill_batch_size,
            account_workers=self._cfg.skill_account_workers,
        )
        self._skill_sedimentation_thread = threading.Thread(
            target=self._skill_sedimentation_worker.run,
            daemon=True,
            name="ogmemory-skill-sedimentation",
        )
        self._skill_sedimentation_thread.start()

    def _skill_account_ids(self) -> list[str]:
        try:
            trace_store = self._get_trace_event_store()
            if trace_store is None:
                return []
            return trace_store.list_pending_skill_accounts(
                limit=max(1, self._cfg.skill_account_workers * 2)
            )
        except Exception:
            logger.warning("Unable to discover accounts with pending skill work", exc_info=True)
            raise

    def wake_skill_sedimentation(self) -> None:
        """Wake the asynchronous worker; never run generation on the request thread."""

        if self._skill_sedimentation_worker is not None:
            self._skill_sedimentation_worker.wake()

    def stop_skill_sedimentation(self, timeout: float = 5.0) -> bool:
        if self._skill_sedimentation_worker is not None:
            self._skill_sedimentation_worker.stop()
        if self._skill_sedimentation_thread is not None:
            self._skill_sedimentation_thread.join(timeout=timeout)
            if self._skill_sedimentation_thread.is_alive():
                logger.warning("Skill sedimentation worker did not stop within %.1fs", timeout)
                return False
        self._skill_sedimentation_worker = None
        self._skill_sedimentation_thread = None
        return True

    def list_skill_candidates(
        self,
        ctx: RequestContext,
        *,
        state: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        if not ctx.agent_id:
            raise ValueError("agent_id is required to list skill candidates")
        store = self._get_trace_event_store()
        if store is None:
            raise RuntimeError("trace store unavailable")
        return store.list_skill_candidates(
            account_id=ctx.account_id,
            agent_id=ctx.agent_id,
            state=state,
            limit=limit,
        )

    def list_published_skills(
        self,
        ctx: RequestContext,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict:
        """Return one stable page from generated and managed Skill stores."""

        from skills.artifact import is_governed_skill, is_managed_skill, validate_skill_name
        from skills.management import validate_managed_skill_name

        if not ctx.agent_id:
            raise ValueError("agent_id is required to list skills")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        cursor_name = None
        cursor_source = None
        if cursor:
            if "|" in cursor:
                cursor_name, cursor_source = cursor.rsplit("|", 1)
                if cursor_source == "imported":
                    validate_managed_skill_name(cursor_name)
                elif cursor_source == "generated":
                    validate_skill_name(cursor_name)
                else:
                    raise ValueError("invalid skill cursor")
            else:
                cursor_name, cursor_source = validate_skill_name(cursor), "generated"
        context_fs = self._get_context_fs()
        rows: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()
        roots = (
            (
                "generated",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/generated",
                is_governed_skill,
            ),
            (
                "imported",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/imported",
                is_managed_skill,
            ),
            (
                "generated",
                f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills",
                is_governed_skill,
            ),
        )
        for source, base_uri, published in roots:
            root_rows: list[dict] = []
            target = limit + 1

            def append_node(node) -> None:
                if not published(node) or len(root_rows) >= target:
                    return
                name = node.uri.rsplit("/", 1)[-1]
                row_key = (name, source)
                if row_key in seen_keys or (
                    cursor_name and row_key <= (cursor_name, cursor_source)
                ):
                    return
                seen_keys.add(row_key)
                root_rows.append({
                    "name": name,
                    "description": node.metadata.get(
                        "package_description", node.abstract
                    ),
                    "source": source,
                    "updated_at": node.metadata.get("updated_at", ""),
                })

            after_uri = f"{base_uri}/{cursor_name}" if cursor_name else None
            if cursor_name and source == "imported" and cursor_source == "generated":
                same_name_uri = f"{base_uri}/{cursor_name}"
                if context_fs.exists(same_name_uri, ctx):
                    append_node(context_fs.read_node(same_name_uri, ctx))

            if hasattr(context_fs, "list_node_summaries_page"):
                while len(root_rows) < target:
                    batch_size = target - len(root_rows)
                    nodes = context_fs.list_node_summaries_page(
                        base_uri, ctx, limit=batch_size, after_uri=after_uri,
                    )
                    if not nodes:
                        break
                    for node in nodes:
                        append_node(node)
                    after_uri = nodes[-1].uri
                    if len(nodes) < batch_size:
                        break
            else:
                uris = sorted(context_fs.list_children(base_uri, ctx))
                for uri in uris:
                    if after_uri and uri <= after_uri:
                        continue
                    append_node(context_fs.read_node(uri, ctx))
                    if len(root_rows) >= target:
                        break
            rows.extend(root_rows)
        rows.sort(key=lambda row: (row["name"], row["source"]))
        page = rows[:limit]
        truncated = len(rows) > limit
        payload = {"skills": page, "truncated": truncated}
        if truncated and page:
            payload["next_cursor"] = f"{page[-1]['name']}|{page[-1]['source']}"
        return payload

    @property
    def default_agent_id(self) -> str:
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self) -> int:
        return int(self._cfg.skill_search_scan_limit)

    def read_published_skill(
        self, ctx: RequestContext, name: str, source: str | None = None
    ):
        """Read one published Skill selected by name and optional source."""

        from core.errors import NodeNotFoundError
        from skills.artifact import (
            internal_managed_skill_uri,
            internal_skill_uri,
            is_governed_skill,
            is_managed_skill,
            legacy_internal_skill_uri,
        )

        if source not in {None, "generated", "imported"}:
            raise ValueError("source must be generated or imported")
        candidates = []
        if source in {None, "generated"}:
            try:
                generated_uris = (
                    internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                    legacy_internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                )
                candidates.extend(
                    ("generated", uri, is_governed_skill) for uri in generated_uris
                )
            except ValueError:
                if source == "generated":
                    raise
        if source in {None, "imported"}:
            candidates.append((
                "imported",
                internal_managed_skill_uri(ctx.account_id, ctx.agent_id, name),
                is_managed_skill,
            ))
        found = []
        found_sources = set()
        context_fs = self._get_context_fs()
        for candidate_source, uri, published in candidates:
            try:
                node = context_fs.read_node(uri, ctx)
            except (FileNotFoundError, NodeNotFoundError):
                continue
            if published(node) and candidate_source not in found_sources:
                if candidate_source == "imported":
                    node.content = node.metadata.get("package_skill_md", node.content)
                    node.abstract = node.metadata.get("package_description", node.abstract)
                found.append(node)
                found_sources.add(candidate_source)
        if not found:
            raise FileNotFoundError("published skill not found")
        if len(found) > 1:
            raise ValueError("skill name is ambiguous; source is required")
        return found[0]

    def validate_managed_skill(
        self,
        data,
        *,
        expected_name: str | None = None,
        strict: bool = False,
        skill_dir_name: str | None = None,
        source_path: str | None = None,
    ) -> dict:
        """Validate supported skill input without writing it."""

        from skills.management import (
            package_file_manifest,
            parse_managed_skill,
            validate_skill_format,
        )

        result = validate_skill_format(
            data,
            strict=strict,
            skill_dir_name=skill_dir_name,
            source_path=source_path,
        )
        if expected_name is not None and result.get("name") != expected_name:
            raise ValueError("skill content name must match the requested name")
        if not result["valid"]:
            return result
        try:
            skill = parse_managed_skill(data)
        except ValueError as exc:
            result["valid"] = False
            result["errors"].append({
                "rule": "installable",
                "message": str(exc),
                "field": "data",
            })
            return result
        result.update({
            "content_bytes": len(skill.content.encode("utf-8")),
            "package_bytes": skill.total_bytes,
            "auxiliary_files": skill.auxiliary_files,
            "files": package_file_manifest(skill, source="imported"),
        })
        return result

    def add_managed_skill(self, ctx: RequestContext, data) -> dict:
        """Install one explicit complete Skill package."""

        from core.errors import ConcurrentModificationError
        from skills.artifact import artifact_body_hash, internal_managed_skill_uri
        from skills.management import parse_managed_skill

        self._require_skill_management_role(ctx)
        skill = parse_managed_skill(data)
        uri = internal_managed_skill_uri(ctx.account_id, ctx.agent_id, skill.name)
        fs = self._require_skill_fs()
        now = _dt.now(UTC).isoformat()
        node = ContextNode(
            uri=uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space=ctx.agent_space_name(),
            abstract=skill.description[:200],
            overview=f"Agent skill: {skill.name}",
            content=skill.content,
            metadata={
                "ogmemory_managed": True,
                "publication_status": "published",
                "content_hash": artifact_body_hash(skill.content),
                "package_skill_md": skill.content,
                "package_description": skill.description,
                "package_files": skill.serialized_files(),
                "package_bytes": skill.total_bytes,
                "created_at": now,
                "updated_at": now,
                "_expected_version": 0,
            },
        )
        try:
            fs.write_node(node, ctx)
        except ConcurrentModificationError as exc:
            raise FileExistsError(f"skill already exists: {skill.name}") from exc
        return self._managed_skill_result(node, action="create", version=1)

    def update_managed_skill(self, ctx: RequestContext, name: str, data) -> dict:
        """Replace one imported Skill package without touching generated skills."""

        from skills.artifact import artifact_body_hash
        from skills.management import parse_managed_skill, validate_managed_skill_name

        self._require_skill_management_role(ctx)
        name = validate_managed_skill_name(name)
        skill = parse_managed_skill(data)
        if skill.name != name:
            raise ValueError("skill content name must match the requested name")

        existing = self.read_published_skill(ctx, name, source="imported")
        version = existing.metadata.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError("imported skill has an invalid version")

        now = _dt.now(UTC).isoformat()
        node = ContextNode(
            uri=existing.uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space=ctx.agent_space_name(),
            abstract=skill.description[:200],
            overview=f"Agent skill: {skill.name}",
            content=skill.content,
            metadata={
                "ogmemory_managed": True,
                "publication_status": "published",
                "content_hash": artifact_body_hash(skill.content),
                "package_skill_md": skill.content,
                "package_description": skill.description,
                "package_files": skill.serialized_files(),
                "package_bytes": skill.total_bytes,
                "created_at": existing.metadata.get("created_at", now),
                "updated_at": now,
                "_expected_version": version,
            },
        )
        self._require_skill_fs().write_node(node, ctx)
        return self._managed_skill_result(
            node, action="update", version=version + 1
        )

    def delete_published_skill(
        self, ctx: RequestContext, name: str, source: str | None
    ) -> dict:
        """Delete one published Skill from the selected source."""

        from core.errors import NodeNotFoundError
        from skills.artifact import (
            canonical_skill_uri,
            internal_skill_uri,
            is_governed_skill,
            legacy_internal_skill_uri,
        )

        if source not in {"generated", "imported"}:
            raise ValueError("source is required and must be generated or imported")
        self._require_skill_management_role(ctx)
        node = self.read_published_skill(ctx, name, source=source)
        fs = self._require_skill_fs()
        nodes = [node]
        if source == "generated":
            for uri in (
                internal_skill_uri(ctx.account_id, ctx.agent_id, name),
                legacy_internal_skill_uri(ctx.account_id, ctx.agent_id, name),
            ):
                if uri == node.uri:
                    continue
                try:
                    candidate = fs.read_node(uri, ctx)
                except (FileNotFoundError, NodeNotFoundError):
                    continue
                if is_governed_skill(candidate):
                    nodes.append(candidate)

        deleted = False
        for candidate in nodes:
            try:
                fs.delete_node(candidate.uri, ctx)
                deleted = True
            except (FileNotFoundError, NodeNotFoundError):
                continue
        if not deleted:
            raise FileNotFoundError("published skill not found")
        return {
            "name": name,
            "uri": canonical_skill_uri(name, source),
            "source": source,
            "action": "delete",
        }

    def read_managed_skill_file(
        self, ctx: RequestContext, name: str, path: str
    ) -> dict:
        """Read one sanitized file from an installed managed Skill package."""

        import base64

        from skills.management import (
            PackageFile,
            deserialize_package_file,
            package_file_uri,
        )

        node = self.read_published_skill(ctx, name, source="imported")
        if path == "SKILL.md":
            item = PackageFile("SKILL.md", node.content.encode("utf-8"), "text/markdown")
        else:
            item = None
            for record in node.metadata.get("package_files") or []:
                candidate = deserialize_package_file(record)
                if candidate.path == path:
                    item = candidate
                    break
            if item is None:
                raise FileNotFoundError(f"skill package file not found: {path}")
        result = {
            "name": item.path.rsplit("/", 1)[-1],
            "path": item.path,
            "uri": package_file_uri(name, item.path),
            "source": "imported",
            "media_type": item.media_type,
            "size": len(item.data),
            "sha256": item.sha256,
        }
        if b"\x00" not in item.data:
            try:
                result["content"] = item.data.decode("utf-8")
                result["encoding"] = "utf-8"
                return result
            except UnicodeDecodeError:
                pass
        result["content_base64"] = base64.b64encode(item.data).decode("ascii")
        result["encoding"] = "base64"
        return result

    def _require_skill_fs(self):
        fs = self._get_context_fs()
        if fs is None:
            raise RuntimeError("ContextFS unavailable for skill management")
        return fs

    def _require_skill_management_role(self, ctx: RequestContext) -> None:
        if not ctx.agent_id:
            raise ValueError("agent_id is required to manage skills")
        self.get_auth_service().require_role(ctx, Role.ADMIN, Role.ROOT)

    @staticmethod
    def _managed_skill_result(node: ContextNode, *, action: str, version: int) -> dict:
        name = node.uri.rsplit("/", 1)[-1]
        uri = f"ogmemory://skills/imported/{name}"
        return {
            "status": "success",
            "name": name,
            "description": node.metadata.get("package_description", node.abstract),
            "root_uri": uri,
            "uri": uri,
            "source": "imported",
            "action": action,
            "version": version,
            "auxiliary_files": len(node.metadata.get("package_files") or []),
            "package_bytes": node.metadata.get(
                "package_bytes", len(node.content.encode("utf-8"))
            ),
        }

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

    @_serialize_lazy_initialization("_control_store")
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

    @_serialize_lazy_initialization("_internal_tool_usage_store")
    def get_internal_tool_usage_store(self) -> InternalToolUsageStore:
        if self._internal_tool_usage_store is None:
            self._internal_tool_usage_store = InternalToolUsageStore(self.get_control_plane_store())
        return self._internal_tool_usage_store

    @_serialize_lazy_initialization("_key_manager")
    def get_key_manager(self) -> APIKeyManager:
        if self._key_manager is None:
            self._key_manager = APIKeyManager(self.get_control_plane_store())
        return self._key_manager

    @_serialize_lazy_initialization("_auth")
    def get_auth_service(self) -> AuthService:
        if self._auth is None:
            self._auth = AuthService(self._cfg, self.get_key_manager())
        return self._auth

    @_serialize_lazy_initialization("_audit")
    def get_audit_service(self) -> AuditService:
        if self._audit is None:
            self._audit = AuditService(self.get_control_plane_store())
        return self._audit

    @_serialize_lazy_initialization("_tenant_admin")
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
            if self._cfg.skill_sedimentation_enabled:
                result = self._search_without_generated_skills(
                    read_api, query=query, ctx=ctx, top_k=15,
                    fill_content_for_top_k=5,
                )
            else:
                result = read_api.search_memory(
                    query=query, ctx=ctx, top_k=15,
                    fill_content_for_top_k=5, mode="QUICK",
                )
            if result and result.hits:
                ordinary_hits = list(result.hits)
                stats["hit_count"] = len(ordinary_hits)
                stats["hit_categories"] = list(
                    {h.category for h in ordinary_hits if h.category}
                )

                items = [
                    {
                        "uri": h.uri,
                        "abstract": h.abstract or "",
                        "overview": h.overview or "",
                        "content": h.content_excerpt or "",
                        "score": h.score,
                        "category": h.category,
                    }
                    for h in ordinary_hits
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
        except Exception as exc:
            logger.warning("Working set search failed: %s", exc)
            return [], [], {}
        return items, summary_items, stats

    def _search_without_generated_skills(
        self,
        read_api,
        *,
        query: str,
        ctx: RequestContext,
        top_k: int,
        fill_content_for_top_k: int,
    ):
        """Search ordinary memories and legacy skills before applying top-k.

        Generated skills are consumed through the explicit skill catalog. Legacy
        or user-authored skill memories retain their pre-feature compose behavior.
        """

        search_top_k = min(
            int(getattr(self._cfg, "retrieval_max_top_k", top_k)),
            max(
                top_k,
                top_k * int(getattr(self._cfg, "retrieval_candidate_pool_multiplier", 3)),
                int(getattr(self._cfg, "retrieval_min_candidate_pool_size", top_k)),
            ),
        )
        result = read_api.search_memory(
            query=query,
            ctx=ctx,
            top_k=search_top_k,
            fill_content_for_top_k=fill_content_for_top_k,
            mode="QUICK",
        )
        if result is None:
            return None
        candidates = [
            hit
            for hit in (getattr(result, "hits", []) or [])
            if (getattr(hit, "metadata", {}) or {}).get("ogmemory_generated") is not True
        ]
        deduped = {hit.uri: hit for hit in candidates}
        result.hits = sorted(
            deduped.values(),
            key=lambda hit: float(
                getattr(hit, "ranking_score", None)
                if getattr(hit, "ranking_score", None) is not None
                else getattr(hit, "score", 0.0)
            ),
            reverse=True,
        )[:top_k]
        return result

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
        window_state.last_accessed_at = _dt.now(UTC).isoformat()
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

    @_serialize_lazy_initialization("_vector_index")
    def _get_shared_vector_index(self):
        """Lazy-init shared vector index (same instance for read + outbox worker)."""
        if self._vector_index is None:
            vector_index = self._provider_cfg.create_vector_index()
            projection_index = vector_index
            lexical_index = None
            if getattr(self._cfg, "retrieval_hybrid_enabled", False):
                try:
                    from core.lexical import HybridLexicalIndex
                    from providers.composite_index import CompositeProjectionWriter
                    from providers.lexical_index import create_lexical_index

                    lexical_index = create_lexical_index(self._cfg)
                    if not isinstance(lexical_index, HybridLexicalIndex):
                        raise TypeError(
                            "hybrid lexical provider lacks durable checkpoint capabilities"
                        )
                    projection_index = CompositeProjectionWriter(
                        vector_index,
                        lexical_index,
                    )
                    from core.lexical import lexical_scopes_from_index_records

                    projection_index.set_failure_callbacks(
                        records_failed=lambda records: lexical_index.mark_scopes_dirty(
                            lexical_scopes_from_index_records(records)
                        ),
                        scope_unknown_failed=lexical_index.mark_all_scopes_dirty,
                    )
                except Exception:
                    for resource in (lexical_index, vector_index):
                        close = getattr(resource, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                logger.warning(
                                    "Resource cleanup failed after lexical initialization error",
                                    exc_info=True,
                                )
                    raise
            # Publish the resource tree only after required lexical
            # initialization succeeds, so transient failures remain retryable.
            self._vector_index = vector_index
            self._projection_index = projection_index
            self._lexical_index = lexical_index
            logger.info("Shared vector_index ready: %s", type(self._vector_index).__name__)
        return self._vector_index

    def _get_shared_projection_index(self):
        """Return the outbox sink that keeps vector and lexical projections aligned."""
        self._get_shared_vector_index()
        return self._projection_index

    @_serialize_lazy_initialization("_lexical_backfill")
    def _get_lexical_backfill(self):
        """Return the lifecycle-owned lexical backfill coordinator."""
        if not getattr(self._cfg, "retrieval_hybrid_enabled", False):
            return None
        if self._lexical_backfill is None:
            self._get_shared_vector_index()
            if self._lexical_index is None:
                raise RuntimeError("hybrid retrieval requires a lexical index")
            from fs.sql_adapter.node_scanner import SQLCanonicalNodeScanner
            from providers.lexical_backfill import LexicalBackfill

            self._lexical_backfill = LexicalBackfill(
                SQLCanonicalNodeScanner(self._get_shared_sql_pool()),
                self._lexical_index,
                page_size=getattr(self._cfg, "retrieval_backfill_page_size", 500),
            )
        return self._lexical_backfill

    @_serialize_lazy_initialization("_embedder")
    def _get_shared_embedder(self):
        """Lazy-init shared embedder (same instance for read + outbox worker)."""
        if self._embedder is None:
            self._embedder = self._provider_cfg.create_embedder()
            logger.info("Shared embedder ready: %s", type(self._embedder).__name__)
        return self._embedder


    @_serialize_lazy_initialization("_outbox_thread")
    def _start_outbox_worker(self):
        """Start OutboxWorker background thread (once, idempotent)."""
        if self._outbox_thread is not None:
            return
        if not _HAS_AGFS and not _HAS_SQL:
            logger.info("OutboxWorker skipped: no storage backend available")
            return
        try:
            from index.outbox_worker import OutboxWorker

            vector_index = self._get_shared_projection_index()
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

            vector_index = self._get_shared_projection_index()
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
        self.stop_skill_sedimentation(timeout=join_timeout)
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
        ):
            if thread is not threading.current_thread():
                try:
                    thread.join(timeout=join_timeout)
                except Exception as exc:
                    logger.warning("Failed to join outbox thread: %s", exc)
            if thread.is_alive():
                logger.warning("Outbox worker did not stop before shutdown deadline")
                return

        self._outbox_listener = None
        self._outbox_thread = None
        if self._lexical_backfill is not None:
            if not self._lexical_backfill.close(timeout=join_timeout):
                logger.warning("Lexical backfill did not stop before shutdown deadline")
                return
        if self._lexical_index is not None and self._projection_index is not None:
            try:
                if not self._projection_index.close(timeout=join_timeout):
                    logger.warning("Hybrid retrieval indexes did not close before deadline")
                    return
            except Exception as exc:
                logger.warning("Failed to close hybrid retrieval indexes: %s", exc)
                return
            self._projection_index = None
            self._lexical_index = None
            self._lexical_backfill = None
            self._vector_index = None

    @_serialize_lazy_initialization("_llm")
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
                    llm_json_mode=cfg.llm_json_mode,
                    llm_reasoning_effort=cfg.llm_reasoning_effort,
                    llm_is_reasoning_model=cfg.llm_is_reasoning_model,
                    llm_extra_body=dict(cfg.llm_extra_body),
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
                    temperature=cfg.llm_temperature,
                    max_tokens=cfg.llm_max_tokens,
                    json_mode=cfg.llm_json_mode,
                    reasoning_effort=cfg.llm_reasoning_effort,
                    is_reasoning_model=cfg.llm_is_reasoning_model,
                    extra_body=cfg.llm_extra_body,
                )
            logger.info("LLM ready: %s", type(self._llm).__name__)
        return self._llm

    @_serialize_lazy_initialization("_write_api")
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
                if self._cfg.skill_sedimentation_enabled:
                    schema_registry.unregister("skill")
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

        original_node = deepcopy(node)
        fs.archive_node(uri, ctx)

        outbox_event_id = None
        outbox = self._get_outbox_store(fs)
        if outbox is not None:
            try:
                event = outbox.register_archive(uri, ctx)
                outbox_event_id = getattr(event, "event_id", None)
            except Exception:
                try:
                    fs.write_node(original_node, ctx)
                except Exception as rollback_exc:
                    logger.error(
                        "forget_memory rollback failed after outbox registration error uri=%s: %s",
                        uri,
                        rollback_exc,
                        exc_info=True,
                    )
                raise

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

    @_serialize_lazy_initialization("_read_api")
    def get_read_api(self):
        if self._read_api is None:
            embedder = self._get_shared_embedder()
            vector_index = self._get_shared_vector_index()

            cfg = RetrievalConfig(
                max_top_k=getattr(self._cfg, "retrieval_max_top_k", 50),
                default_top_k=getattr(self._cfg, "retrieval_default_top_k", 10),
                global_search_topk=getattr(self._cfg, "retrieval_global_search_topk", 30),
                candidate_pool_enabled=getattr(self._cfg, "retrieval_candidate_pool_enabled", False),
                candidate_pool_multiplier=getattr(self._cfg, "retrieval_candidate_pool_multiplier", 3),
                min_candidate_pool_size=getattr(self._cfg, "retrieval_min_candidate_pool_size", 15),
                result_ranker_strategy=getattr(self._cfg, "retrieval_result_ranker_strategy", "semantic"),
                global_result_rerank_enabled=getattr(
                    self._cfg, "retrieval_global_result_rerank_enabled", False,
                ),
                owner_scope_filter_enabled=getattr(self._cfg, "retrieval_owner_scope_filter_enabled", False),
                hybrid_enabled=getattr(self._cfg, "retrieval_hybrid_enabled", False),
                vector_top_k=getattr(self._cfg, "retrieval_vector_top_k", 100),
                lexical_top_k=getattr(self._cfg, "retrieval_lexical_top_k", 50),
                rrf_k=getattr(self._cfg, "retrieval_rrf_k", 60),
                semantic_weight=getattr(self._cfg, "retrieval_semantic_weight", 1.0),
                lexical_weight=getattr(self._cfg, "retrieval_lexical_weight", 1.0),
            )

            # Create relation_store
            relation_store = None
            try:
                relation_store = self._get_relation_store()
            except Exception:
                pass

            # Create context_reader for content filling (shared by pipeline and read_service)
            context_reader = None
            fs = None
            try:
                fs = self._get_context_fs()
                if fs is not None:
                    context_reader = ContextReader(fs=fs, storage_backend=self._cfg.storage_backend)
            except Exception as exc:
                logger.warning("ContextFS unavailable for read_memory: %s", exc)

            lexical_backfill = None
            if cfg.hybrid_enabled:
                if fs is None or self._lexical_index is None:
                    raise RuntimeError(
                        "hybrid retrieval requires canonical storage and lexical index"
                    )
                lexical_backfill = self._get_lexical_backfill()

            pipeline = RetrievalPipeline(
                planner=QueryPlanner(cfg),
                seed_retriever=SeedRetriever(
                    vector_index,
                    embedder,
                    cfg,
                    self._lexical_index,
                    lexical_backfill,
                ),
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

    @_serialize_lazy_initialization("_session_mgr")
    def get_session_manager(self) -> SessionManager:
        """Lazy-init SessionManager with current service dependencies."""
        if self._session_mgr is None:
            # Ordinary SessionState bridge injection and durable runtime state
            # are separate concerns. Tool-result externalization needs the
            # ContextFS-backed runtime_extensions for its Strong Identity Turn
            # Ledger and TTL state even when the ordinary bridge is disabled.
            # With externalization disabled, the historical bridge-off path
            # remains a strict no-op and does not initialize ContextFS.
            durable_runtime_state_enabled = bool(
                getattr(self._cfg, "session_state_bridge_enabled", True)
                or self._tool_result_externalizer is not None
            )
            session_state_context_fs = (
                self._get_context_fs
                if durable_runtime_state_enabled
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
                session_state_bridge_enabled=getattr(
                    self._cfg,
                    "session_state_bridge_enabled",
                    True,
                ),
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
            read_api = self.get_read_api()
            if self._cfg.skill_sedimentation_enabled:
                result = self._search_without_generated_skills(
                    read_api, query=query, ctx=ctx, top_k=top_k,
                    fill_content_for_top_k=0,
                )
            else:
                result = read_api.search_memory(
                    query=query,
                    ctx=ctx,
                    top_k=top_k,
                    fill_content_for_top_k=0,
                    mode="QUICK",
                )
            ordinary_hits = list(getattr(result, "hits", []) or [])
            hits = [
                _retrieved_block_to_seed_hit(hit)
                for hit in ordinary_hits[:top_k]
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

    def _govern_request_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        persistence: bool,
        ranking_prompt: str = "",
        fallback_original: bool = False,
    ):
        """Apply the single provider-neutral governance path to a request copy."""
        if self._tool_result_externalizer is None:
            return None
        # Binding the session through SessionManager provides the same complete
        # account/user/agent/session visibility check as the ordinary buffer.
        self.get_session_manager().get_or_create(session_id, ctx=ctx)
        if fallback_original:
            return self._tool_result_externalizer.govern_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=persistence,
                ranking_prompt=ranking_prompt,
                fallback_original=True,
            )
        if persistence and hasattr(self._tool_result_store, "cleanup_stale_writes"):
            try:
                self._tool_result_store.cleanup_stale_writes(
                    session_id,
                    ctx,
                    self._cfg.tool_result_artifact_temp_grace_seconds,
                )
            except Exception as exc:
                logger.debug("tool-result stale marker cleanup skipped: %s", exc)
        return self._tool_result_externalizer.govern_messages(
            messages,
            ctx=ctx,
            session_id=session_id,
            persistence=persistence,
            ranking_prompt=ranking_prompt,
        )

    def _tool_result_semantic_mode(self, params: dict) -> str:
        """Resolve the configured tool-result semantic Turn lifecycle."""
        configured = getattr(self._cfg, "tool_result_semantic_turn_mode", "auto")
        if configured in {"user_message", "model_request"}:
            return configured
        requested = params.get("semanticTurnMode")
        if requested in {"user_message", "model_request"}:
            return requested
        if self._semantic_turn_identity(params) is not None:
            return "model_request"
        return "auto"

    def _tool_result_governance_fallback_payload(
        self,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bytes:
        """Bind one raw-persistence fallback to the exact failed checkpoint."""
        payload = {
            "schema_version": 1,
            "owner": {
                "account_id": ctx.account_id,
                "user_id": ctx.user_id,
                "agent_id": ctx.agent_id,
                "session_id": session_id,
            },
            "messages": params.get("messages", []),
            "pre_prompt_message_count": params.get("prePromptMessageCount", 0),
            "semantic_turn_mode": self._tool_result_semantic_mode(params),
            "semantic_turn_id": params.get("semanticTurnId"),
            "semantic_turn_start_message_count": params.get(
                "semanticTurnStartMessageCount"
            ),
            "semantic_turn_source_namespace": params.get(
                "semanticTurnSourceNamespace"
            ),
            "semantic_turn_append_only_verified": params.get(
                "semanticTurnAppendOnlyVerified"
            ) is True,
            "semantic_turn_complete": params.get("semanticTurnComplete") is True,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _issue_tool_result_governance_fallback_token(
        self,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> str:
        return hmac.new(
            self._tool_result_source_attestation_key,
            b"tool-result-governance-fallback-v1\0"
            + self._tool_result_governance_fallback_payload(params, ctx, session_id),
            hashlib.sha256,
        ).hexdigest()

    def _validate_tool_result_governance_fallback_token(
        self,
        token,
        params: dict,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        if not isinstance(token, str) or _re.fullmatch(r"[a-f0-9]{64}", token) is None:
            return False
        expected = self._issue_tool_result_governance_fallback_token(
            params,
            ctx,
            session_id,
        )
        return hmac.compare_digest(token, expected)

    def _register_user_message_tool_turn(
        self,
        messages: list,
        *,
        start_boundary: int,
        ctx: RequestContext,
        session_id: str,
    ):
        """Persist one complete user-message Turn before issuing strong ids."""
        ledger = self._tool_result_turn_ledger
        if ledger is None:
            return None
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            previous = deepcopy(buf.runtime_extensions.get(ledger.STATE_KEY))
            registration = ledger.register_user_message_turn(
                previous,
                messages,
                start_boundary=start_boundary,
                ctx=ctx,
                session_id=session_id,
            )
            if registration is None:
                return None
            if not registration.changed:
                return registration
            buf.runtime_extensions[ledger.STATE_KEY] = registration.state
            if mgr.save_session_state(session_id, ctx):
                return registration
            if previous is None:
                buf.runtime_extensions.pop(ledger.STATE_KEY, None)
            else:
                buf.runtime_extensions[ledger.STATE_KEY] = previous
            logger.warning(
                "tool-result Turn Ledger unavailable; using safe fallback identity "
                "session=%s",
                session_id[:8],
            )
            return None

    def _record_user_message_turn_artifacts(
        self,
        registration,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> None:
        """Record verified Artifact ids in the durable Turn Ledger."""
        ledger = self._tool_result_turn_ledger
        if ledger is None or registration is None or self._tool_result_store is None:
            return
        from features.tool_result_externalization.externalizer import (
            ToolResultExternalizer,
        )
        from features.tool_result_externalization.normalizer import (
            canonical_tool_call_id,
        )

        entry = registration.entry
        artifacts: dict[tuple[int, str, int], tuple[str, str]] = {}
        deadline = time.monotonic() + self._tool_result_externalizer.write_timeout_ms / 1000
        for result in entry.get("results", []):
            artifact_id = ToolResultExternalizer.strong_artifact_id(
                ctx.account_id,
                ctx.user_id,
                ctx.agent_id,
                session_id,
                entry["semantic_turn_id"],
                result["source_message_id"],
                result.get("tool_call_key", ""),
                result["part_path"],
                str(result["occurrence_index"]),
            )
            try:
                metadata, output = self._tool_result_io_runner.run(
                    lambda artifact_id=artifact_id: self._tool_result_store.read_all(
                        artifact_id,
                        session_id,
                        ctx,
                    ),
                    deadline=deadline,
                    operation_key=session_id,
                )
            except Exception:
                continue
            if not (
                metadata.identity_strength == "strong"
                and metadata.turn_id == entry["semantic_turn_id"]
                and metadata.source_message_id == result["source_message_id"]
                and canonical_tool_call_id(metadata.tool_call_id)
                == result.get("tool_call_key", "")
                and metadata.part_path == result["part_path"]
                and metadata.occurrence_index == result["occurrence_index"]
                and metadata.original_chars == result["original_chars"]
                and metadata.sha256 == result["content_sha256"]
                and hashlib.sha256(output.encode("utf-8")).hexdigest()
                == result["content_sha256"]
            ):
                continue
            artifacts[(
                result["message_ordinal"],
                result["part_path"],
                result["occurrence_index"],
            )] = (artifact_id, "")
        if not artifacts:
            return

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            previous_ledger = deepcopy(
                buf.runtime_extensions.get(ledger.STATE_KEY)
            )
            updated_ledger = ledger.update_entry_results(
                previous_ledger,
                turn_key=entry["turn_key"],
                artifacts=artifacts,
                ctx=ctx,
                session_id=session_id,
            )
            buf.runtime_extensions[ledger.STATE_KEY] = updated_ledger
            if mgr.save_session_state(session_id, ctx):
                return
            if previous_ledger is None:
                buf.runtime_extensions.pop(ledger.STATE_KEY, None)
            else:
                buf.runtime_extensions[ledger.STATE_KEY] = previous_ledger
            logger.warning(
                "tool-result Turn Ledger Artifact update was not durable; "
                "compose will re-verify from the store session=%s",
                session_id[:8],
            )

    def _mark_user_message_turn_governed(
        self,
        registration,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> bool:
        """Durably close the exact-retry ingest boundary for one Ledger Turn."""
        ledger = self._tool_result_turn_ledger
        if ledger is None or registration is None:
            return False
        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        with buf.semantic_turn_lock:
            previous = deepcopy(buf.runtime_extensions.get(ledger.STATE_KEY))
            updated, changed = ledger.mark_governance_complete(
                previous,
                turn_key=registration.entry["turn_key"],
                ctx=ctx,
                session_id=session_id,
            )
            entry = next(
                (
                    row for row in updated.get("completed_turns", [])
                    if row.get("turn_key") == registration.entry["turn_key"]
                ),
                None,
            )
            if entry is None or entry.get("governance_complete") is not True:
                return False
            if not changed:
                return True
            buf.runtime_extensions[ledger.STATE_KEY] = updated
            if mgr.save_session_state(session_id, ctx):
                return True
            if previous is None:
                buf.runtime_extensions.pop(ledger.STATE_KEY, None)
            else:
                buf.runtime_extensions[ledger.STATE_KEY] = previous
            logger.warning(
                "tool-result Turn Ledger governance marker was not durable; "
                "afterTurn remains retryable session=%s",
                session_id[:8],
            )
            return False

    @staticmethod
    def _semantic_turn_identity(params: dict) -> tuple[str, int, str] | None:
        """Return adapter-attested semantic-turn identity, never an inferred one."""
        semantic_turn_id = params.get("semanticTurnId")
        source_namespace = params.get("semanticTurnSourceNamespace")
        start = params.get("semanticTurnStartMessageCount")
        if (
            params.get("semanticTurnAppendOnlyVerified") is not True
            or not isinstance(semantic_turn_id, str)
            or not semantic_turn_id.strip()
            or not isinstance(source_namespace, str)
            or not source_namespace.strip()
            or len(semantic_turn_id) > 512
            or len(source_namespace) > 512
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
        ):
            return None
        return semantic_turn_id.strip(), start, source_namespace.strip()

    @staticmethod
    def _message_semantic_source(
        message,
        semantic_turn_id: str,
        source_namespace: str,
        semantic_index: int,
    ) -> bool:
        if not isinstance(message, dict):
            return False
        source = message.get("_ogmem_source")
        if not isinstance(source, dict):
            return False
        turn_id = source.get("turn_id") or source.get("turnId")
        message_id = source.get("source_message_id") or source.get("sourceMessageId")
        return bool(
            turn_id == semantic_turn_id
            and message_id == f"{source_namespace}-{semantic_index}"
            and source.get("semantic_turn_attested") is True
        )

    def _semantic_turn_messages(
        self,
        messages: list,
        params: dict,
        *,
        checkpoint_boundary: int | None = None,
        original_indices: tuple[int, ...] | None = None,
        original_semantic_start: int | None = None,
    ) -> list | None:
        """Validate and return the append-only semantic suffix for persistence."""
        identity = self._semantic_turn_identity(params)
        if identity is None:
            return None
        semantic_turn_id, start, source_namespace = identity
        if start > len(messages):
            return None
        if checkpoint_boundary is not None and start > checkpoint_boundary:
            return None
        suffix = messages[start:]
        if original_indices is not None and len(original_indices) != len(messages):
            return None
        source_positions = (
            [
                original_indices[index] - original_semantic_start
                for index in range(start, len(messages))
            ]
            if original_indices is not None
            and original_semantic_start is not None
            else list(range(len(suffix)))
        )
        if not suffix or not all(
            self._message_semantic_source(
                message,
                semantic_turn_id,
                source_namespace,
                source_positions[semantic_index],
            )
            for semantic_index, message in enumerate(suffix)
        ) or any(position < 0 for position in source_positions):
            return None
        return suffix

    @staticmethod
    def _strip_unverified_semantic_sources(messages: list) -> list:
        """Drop every caller-asserted private identity before fallback governance.

        A normalizer treats a source/turn pair as stable, so retaining even an
        envelope without ``semantic_turn_attested`` would let an arbitrary HTTP
        caller manufacture a strong Artifact identity.  Trusted model-request
        suffixes and Turn-Ledger matches bypass this helper and are reattached
        only after their complete server-side validation.
        """
        cleaned = deepcopy(messages)
        for message in cleaned:
            if not isinstance(message, dict):
                continue
            message.pop("_ogmem_source", None)
            message.pop("_ogmem_semantic_source_validated", None)
        return cleaned

    @staticmethod
    def _attach_server_fallback_sources(
        messages: list,
        *,
        reuse_from: list | None = None,
    ) -> list:
        """Bind fallback persistence to fresh server-issued Buffer ids.

        These ids do not make an Artifact strong and are never accepted from a
        caller. They only prove to later internal safety scans that a mixed
        ``stub + inline`` projection was generated for this exact Buffer
        message, avoiding an Artifact-of-Artifact while keeping TTL disabled.
        """
        from features.tool_result_externalization.externalizer import (
            render_safe_message_content,
        )

        bound = deepcopy(messages)
        candidates = list(reuse_from or [])
        consumed: set[str] = set()
        for message in bound:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            original_content = render_safe_message_content(message)
            matched_id = ""
            for existing in candidates:
                existing_id = str(getattr(existing, "id", "") or "")
                if (
                    existing_id
                    and existing_id not in consumed
                    and getattr(existing, "role", None) == role
                    and getattr(existing, "content", None) == original_content
                ):
                    matched_id = existing_id
                    consumed.add(existing_id)
                    break
            message["_ogmem_source"] = {
                "source_message_id": matched_id or f"msg_{uuid4().hex[:12]}",
                "identity_strength": "fallback",
                "server_fallback_allocated": True,
            }
            message.pop("_ogmem_semantic_source_validated", None)
        return bound

    @staticmethod
    def _normalized_tool_result_governance_fallback_ids(raw_state) -> list[str]:
        if not isinstance(raw_state, dict) or raw_state.get("schema_version") != 1:
            return []
        source_ids = raw_state.get("source_message_ids")
        if not isinstance(source_ids, list):
            return []
        normalized = []
        seen = set()
        for value in source_ids:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or value in seen
            ):
                continue
            normalized.append(value)
            seen.add(value)
        return normalized[-_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS:]

    def _record_tool_result_governance_fallback(
        self,
        buf,
        source_message_ids,
    ) -> None:
        """Exclude only one failed checkpoint from future governance attempts."""
        new_ids = [
            value for value in source_message_ids
            if isinstance(value, str) and value and len(value) <= 512
        ]
        if not new_ids:
            return
        with buf.extraction_lock:
            existing = self._normalized_tool_result_governance_fallback_ids(
                buf.runtime_extensions.get(
                    _TOOL_RESULT_GOVERNANCE_FALLBACK_STATE
                )
            )
            combined = list(dict.fromkeys([*existing, *new_ids]))
            buf.runtime_extensions[_TOOL_RESULT_GOVERNANCE_FALLBACK_STATE] = {
                "schema_version": 1,
                "source_message_ids": combined[
                    -_MAX_TOOL_RESULT_GOVERNANCE_FALLBACK_SOURCE_IDS:
                ],
            }

    def _tool_result_governance_fallback_source_ids(
        self,
        session_id: str,
        ctx: RequestContext,
    ) -> frozenset[str]:
        buf = self.get_session_manager().get_or_create(session_id, ctx=ctx)
        with buf.extraction_lock:
            return frozenset(
                self._normalized_tool_result_governance_fallback_ids(
                    buf.runtime_extensions.get(
                        _TOOL_RESULT_GOVERNANCE_FALLBACK_STATE
                    )
                )
            )

    @staticmethod
    def _mark_validated_semantic_sources(messages: list) -> list:
        marked = deepcopy(messages)
        for message in marked:
            if isinstance(message, dict):
                message["_ogmem_semantic_source_validated"] = True
        return marked

    @staticmethod
    def _semantic_turn_ledger(raw_state) -> list[dict]:
        """Normalize legacy single-turn state into a bounded ledger projection."""
        if not isinstance(raw_state, dict):
            return []
        rows = raw_state.get("turns")
        if not isinstance(rows, list):
            rows = [raw_state] if raw_state.get("turn_id") else []
        ledger = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            turn_id = row.get("turn_id")
            namespace = row.get("source_namespace")
            if not isinstance(turn_id, str) or not turn_id:
                continue
            if not isinstance(namespace, str) or not namespace:
                continue
            try:
                max_count = max(0, int(row.get("max_message_count", 0) or 0))
            except (TypeError, ValueError):
                max_count = 0
            try:
                pending_count = max(
                    max_count,
                    int(row.get("pending_message_count", max_count) or max_count),
                )
            except (TypeError, ValueError):
                pending_count = max_count
            ledger.append({
                "turn_id": turn_id,
                "source_namespace": namespace,
                "max_message_count": max_count,
                "complete": row.get("complete") is True,
                "governance_pending": row.get("governance_pending") is True,
                "pending_message_count": (
                    pending_count
                    if row.get("governance_pending") is True
                    else max_count
                ),
            })
        return ledger[-32:]

    @staticmethod
    def _bounded_semantic_turn_completed_keys(rows) -> list[str]:
        """Keep last-occurrence order and the newest bounded tombstones."""
        if not isinstance(rows, list):
            return []
        seen = set()
        newest_first = []
        for row in reversed(rows):
            if not isinstance(row, str) or not row or row in seen:
                continue
            seen.add(row)
            newest_first.append(row)
            if len(newest_first) >= _SEMANTIC_TURN_COMPLETED_KEY_LIMIT:
                break
        newest_first.reverse()
        return newest_first

    @classmethod
    def _semantic_turn_completed_keys(cls, raw_state) -> list[str]:
        if not isinstance(raw_state, dict):
            return []
        return cls._bounded_semantic_turn_completed_keys(
            raw_state.get("completed_turn_keys")
        )

    @staticmethod
    def _semantic_turn_key(turn_id: str, source_namespace: str) -> str:
        return hashlib.sha256(
            f"{turn_id}\0{source_namespace}".encode("utf-8")
        ).hexdigest()

    @classmethod
    def _semantic_turn_ledger_state(
        cls,
        ledger: list[dict],
        completed_turn_keys: list[str] | None = None,
    ) -> dict:
        return {
            "schema_version": 3,
            "turns": deepcopy(ledger[-32:]),
            # This finite, user-selected window bounds session state growth.
            # A turn older than 1024 completions can be accepted again.
            "completed_turn_keys": cls._bounded_semantic_turn_completed_keys(
                completed_turn_keys or []
            ),
        }

    @staticmethod
    def _map_projected_message_boundary(
        retained_prefix_counts: tuple[int, ...],
        boundary: int,
    ) -> int:
        if not isinstance(boundary, int):
            raise TypeError("message boundary must be an integer")
        original_count = max(0, len(retained_prefix_counts) - 1)
        clamped = min(max(0, boundary), original_count)
        return retained_prefix_counts[clamped]

    @staticmethod
    def _canonical_internal_tool_name(value) -> str:
        return str(value or "").rsplit("__", 1)[-1].casefold()

    def _strip_internal_artifact_list_roundtrips(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> tuple[list, tuple[int, ...], tuple[int, ...], bool]:
        """Drop only uniquely paired, attested list navigation after terminal output."""
        identity = tuple(range(len(messages)))
        unchanged_prefix = tuple(range(len(messages) + 1))
        if self._tool_result_externalizer is None or not messages:
            return messages, identity, unchanged_prefix, False

        from features.tool_result_externalization.normalizer import (
            canonical_tool_call_id,
        )
        from features.tool_result_externalization.turn_ledger import (
            _terminal_assistant,
        )

        if (
            not isinstance(messages[-1], dict)
            or not _terminal_assistant(messages[-1])
        ):
            return messages, identity, unchanged_prefix, False

        copied = deepcopy(messages)
        all_calls: dict[str, list[tuple[int, str, int]]] = {}
        list_calls: dict[str, list[tuple[int, str, int]]] = {}
        call_types = {"tool_call", "toolcall", "tool_use", "tooluse"}

        def record_call(
            message_index: int,
            field: str,
            item_index: int,
            call_id,
            tool_name,
        ) -> None:
            key = canonical_tool_call_id(call_id)
            if not key:
                return
            location = (message_index, field, item_index)
            all_calls.setdefault(key, []).append(location)
            if self._canonical_internal_tool_name(tool_name) == (
                "list_tool_result_artifacts"
            ):
                list_calls.setdefault(key, []).append(location)

        for message_index, message in enumerate(copied):
            if not isinstance(message, dict) or str(
                message.get("role", "")
            ).casefold() != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item_index, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type", "")).casefold() not in call_types:
                        continue
                    call_id = next((
                        block.get(key)
                        for key in (
                            "id", "tool_call_id", "toolCallId",
                            "tool_use_id", "toolUseId",
                        )
                        if block.get(key) not in (None, "")
                    ), None)
                    tool_name = next((
                        block.get(key)
                        for key in ("name", "tool_name", "toolName")
                        if block.get(key) not in (None, "")
                    ), None)
                    record_call(
                        message_index, "content", item_index, call_id, tool_name,
                    )
            for field in ("tool_calls", "toolCalls"):
                calls = message.get(field)
                if not isinstance(calls, list):
                    continue
                for item_index, call in enumerate(calls):
                    if not isinstance(call, dict):
                        continue
                    function = (
                        call.get("function")
                        if isinstance(call.get("function"), dict)
                        else {}
                    )
                    call_id = next((
                        call.get(key)
                        for key in (
                            "id", "tool_call_id", "toolCallId",
                            "tool_use_id", "toolUseId",
                        )
                        if call.get(key) not in (None, "")
                    ), None)
                    tool_name = next((
                        call.get(key)
                        for key in ("name", "tool_name", "toolName")
                        if call.get(key) not in (None, "")
                    ), None) or function.get("name")
                    record_call(
                        message_index, field, item_index, call_id, tool_name,
                    )

        if not list_calls:
            return messages, identity, unchanged_prefix, False

        normalized = self._tool_result_externalizer.normalizer.normalize(copied)
        all_results: dict[str, list[tuple[int, object]]] = {}
        trusted_list_results: dict[str, list[tuple[int, object]]] = {}
        for normalized_message in normalized:
            for part in normalized_message.tool_results:
                key = canonical_tool_call_id(part.tool_call_id)
                if not key:
                    continue
                occurrence = (normalized_message.message_index, part)
                all_results.setdefault(key, []).append(occurrence)
                if self._tool_result_externalizer.validates_artifact_list_result(
                    part,
                    ctx,
                    session_id,
                ):
                    trusted_list_results.setdefault(key, []).append(occurrence)

        call_removals: dict[int, dict[str, set[int]]] = {}
        result_block_removals: dict[int, set[int]] = {}
        whole_message_removals: set[int] = set()
        terminal_index = len(copied) - 1
        for key, candidate_calls in list_calls.items():
            candidates = trusted_list_results.get(key, [])
            if (
                len(candidate_calls) != 1
                or len(all_calls.get(key, [])) != 1
                or len(candidates) != 1
                or len(all_results.get(key, [])) != 1
            ):
                continue
            call_location = candidate_calls[0]
            result_message_index, result_part = candidates[0]
            if not (
                call_location[0] < result_message_index < terminal_index
            ):
                continue
            result_message = copied[result_message_index]
            role = str(result_message.get("role", "")).casefold()
            remove_whole_result = role in {
                "tool", "toolresult", "tool_result",
            }
            result_block_index = None
            payload_path = tuple(getattr(result_part, "payload_path", ()))
            if (
                not remove_whole_result
                and len(payload_path) >= 2
                and payload_path[0] == "content"
                and isinstance(payload_path[1], int)
            ):
                result_block_index = payload_path[1]
            if not remove_whole_result and result_block_index is None:
                continue
            call_removals.setdefault(call_location[0], {}).setdefault(
                call_location[1], set(),
            ).add(call_location[2])
            if remove_whole_result:
                whole_message_removals.add(result_message_index)
                continue
            result_block_removals.setdefault(
                result_message_index, set(),
            ).add(result_block_index)

        if not call_removals:
            return messages, identity, unchanged_prefix, False

        retained_messages = []
        retained_indices: list[int] = []
        retained_prefix_counts = [0]
        touched_indices = set(call_removals) | set(result_block_removals)
        for message_index, message in enumerate(copied):
            if message_index in whole_message_removals:
                retained_prefix_counts.append(len(retained_messages))
                continue
            for field, item_indices in call_removals.get(
                message_index, {},
            ).items():
                items = message.get(field)
                if isinstance(items, list):
                    for item_index in sorted(item_indices, reverse=True):
                        if 0 <= item_index < len(items):
                            items.pop(item_index)
                    if not items and field != "content":
                        message.pop(field, None)
            content = message.get("content")
            if isinstance(content, list):
                for item_index in sorted(
                    result_block_removals.get(message_index, set()),
                    reverse=True,
                ):
                    if 0 <= item_index < len(content):
                        content.pop(item_index)
            if message_index in touched_indices:
                content = message.get("content")
                has_content = (
                    isinstance(content, str) and bool(content.strip())
                ) or (isinstance(content, list) and bool(content))
                has_calls = any(
                    isinstance(message.get(field), list) and message[field]
                    for field in ("tool_calls", "toolCalls")
                )
                if not has_content and not has_calls:
                    retained_prefix_counts.append(len(retained_messages))
                    continue
            retained_messages.append(message)
            retained_indices.append(message_index)
            retained_prefix_counts.append(len(retained_messages))

        return (
            retained_messages,
            tuple(retained_indices),
            tuple(retained_prefix_counts),
            True,
        )

    def _upsert_governed_messages(
        self,
        session_id: str,
        governed,
        ctx: RequestContext,
        *,
        tail_fallback: bool,
    ) -> int:
        """Write governed flat projections using stable ids or one-to-one tail matching."""
        from features.tool_result_externalization.externalizer import (
            render_safe_message_content,
        )

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)
        fallback_original = bool(
            getattr(governed, "fallback_original", False)
        )
        validation_deadline = (
            time.monotonic()
            + self._tool_result_externalizer.write_timeout_ms / 1000
        )
        preserved_ids = set(getattr(governed, "preserved_message_ids", ()) or ())
        for message_id in preserved_ids:
            existing = next(
                (message for message in buf.messages if message.id == message_id),
                None,
            )
            if existing is None:
                continue
            existing_status = (
                self._tool_result_externalizer.embedded_stub_projection_status(
                    existing.content,
                    ctx,
                    session_id,
                    deadline=validation_deadline,
                    expected_source_message_id=existing.id,
                )
            )
            if existing_status in {"valid", "unavailable"}:
                continue
            # Only a verified last-known-good stub is preservable. An older raw
            # value that newly became eligible must still be removed fail closed.
            mgr.remove_message(session_id, message_id, ctx)
        for message_id in governed.removed_message_ids:
            if message_id in preserved_ids:
                continue
            mgr.remove_message(session_id, message_id, ctx)

        fallback_candidates = []
        if tail_fallback:
            fallback_count = len(governed.messages) + len(
                governed.removed_fallback_messages
            )
            start = max(
                buf.extraction_watermark,
                len(buf.messages) - fallback_count,
            )
            fallback_candidates = list(buf.messages[start:])
        consumed_tail: set[str] = set()

        for role, original_content in governed.removed_fallback_messages:
            for existing in fallback_candidates:
                if existing.id in consumed_tail:
                    continue
                if existing.role == role and existing.content == original_content:
                    consumed_tail.add(existing.id)
                    mgr.remove_message(session_id, existing.id, ctx)
                    break

        count = 0
        for message, source_message_id, original_content in zip(
            governed.messages,
            governed.source_message_ids,
            governed.fallback_original_contents,
            strict=True,
        ):
            role = str(message.get("role", "user"))
            content = render_safe_message_content(message)
            if not content:
                continue
            created_at = message.get("created_at")
            source = message.get("_ogmem_source")
            semantic_attested = bool(
                isinstance(source, dict)
                and source.get("semantic_turn_attested") is True
            )
            semantic_validated = bool(
                message.get("_ogmem_semantic_source_validated") is True
            )
            if semantic_attested and not semantic_validated:
                source_message_id = None
            if source_message_id and semantic_attested:
                existing = next(
                    (
                        item for item in buf.messages
                        if item.id == source_message_id
                    ),
                    None,
                )
                existing_status = (
                    self._tool_result_externalizer.stub_sequence_status(
                        existing.content,
                        ctx,
                        session_id,
                        deadline=validation_deadline,
                    )
                    if existing is not None
                    else "invalid"
                )
                incoming_status = self._tool_result_externalizer.stub_sequence_status(
                    content,
                    ctx,
                    session_id,
                    deadline=validation_deadline,
                )
                if (
                    existing is not None
                    and existing_status in {"valid", "unavailable"}
                    and incoming_status != "valid"
                    and not fallback_original
                ):
                    # A late, shorter checkpoint may no longer exceed the turn
                    # budget. Never let it downgrade an already verified stub
                    # for the same adapter-attested append-only source message.
                    continue
            if not source_message_id and tail_fallback:
                matched = None
                for existing in fallback_candidates:
                    if existing.id in consumed_tail:
                        continue
                    if existing.role == role and existing.content == original_content:
                        matched = existing
                        consumed_tail.add(existing.id)
                        break
                if matched is not None:
                    if matched.content != content:
                        mgr.add_message(
                            session_id,
                            role,
                            content,
                            ctx,
                            created_at=created_at,
                            message_id=matched.id,
                        )
                        count += 1
                    continue
            mgr.add_message(
                session_id,
                role,
                content,
                ctx,
                created_at=created_at,
                message_id=source_message_id,
            )
            count += 1
        return count

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

    def add_session_trace(self, params: dict) -> dict:
        """Persist one bounded agent event directly to SQL."""
        ctx = params.get("_ctx")
        if ctx is None:
            return {"ok": False, "error": "missing context", "retryable": False}

        event = params.get("event")
        if not isinstance(event, dict) or not event:
            return {
                "ok": False,
                "error": "missing event payload",
                "retryable": False,
            }

        event_id = str(params.get("eventId") or "").strip()
        session_id = str(params.get("sessionId") or "").strip()
        hook_type = str(params.get("hookType") or event.get("hook_type") or "unknown")
        if not event_id:
            return {"ok": False, "error": "missing event_id", "retryable": False}
        if len(event_id) > 64:
            return {
                "ok": False,
                "error": "event_id exceeds 64 characters",
                "retryable": False,
            }
        if not session_id or session_id == "unknown":
            return {"ok": False, "error": "missing session_id", "retryable": False}
        if len(session_id) > 128:
            return {
                "ok": False,
                "error": "session_id exceeds 128 characters",
                "retryable": False,
            }
        if len(hook_type) > 64:
            return {
                "ok": False,
                "error": "hook_type exceeds 64 characters",
                "retryable": False,
            }

        event_time = params.get("eventTime")
        if isinstance(event_time, str):
            try:
                event_time = _dt.fromisoformat(event_time.replace("Z", "+00:00"))
            except ValueError:
                return {
                    "ok": False,
                    "error": "invalid event_time",
                    "retryable": False,
                }
        if event_time is not None and not isinstance(event_time, _dt):
            return {
                "ok": False,
                "error": "invalid event_time",
                "retryable": False,
            }
        if isinstance(event_time, _dt) and event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)

        store = self._get_trace_event_store()
        if store is None:
            return {
                "ok": False,
                "error": "trace store unavailable",
                "retryable": True,
            }

        try:
            return store.insert(
                event_id=event_id,
                account_id=ctx.account_id,
                owner_space=ctx.user_space_name(),
                agent_id=ctx.agent_id,
                session_id=session_id,
                hook_type=hook_type,
                content=event,
                event_time=event_time,
            )
        except Exception as exc:
            logger.warning(
                "add_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "trace persistence unavailable",
                "error_code": "trace_persistence_unavailable",
                "retryable": True,
            }

    def finalize_session_trace(self, params: dict) -> dict:
        """Finalize the turn-end-bounded turn containing ``stopEventIndex``."""
        ctx = params.get("_ctx")
        if ctx is None:
            return {"ok": False, "error": "missing context", "retryable": False}
        session_id = str(params.get("sessionId") or "").strip()
        try:
            stop_event_index = int(params.get("stopEventIndex") or 0)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "invalid stop_event_index",
                "retryable": False,
            }
        if not session_id or session_id == "unknown":
            return {"ok": False, "error": "missing session_id", "retryable": False}
        if stop_event_index <= 0:
            return {
                "ok": False,
                "error": "invalid stop_event_index",
                "retryable": False,
            }

        store = self._get_trace_event_store()
        if store is None:
            return {
                "ok": False,
                "error": "trace store unavailable",
                "retryable": True,
            }
        try:
            finalized = store.finalize_session(
                account_id=ctx.account_id,
                owner_space=ctx.user_space_name(),
                agent_id=ctx.agent_id,
                session_id=session_id,
                stop_event_index=stop_event_index,
            )
            if self._cfg.skill_sedimentation_enabled:
                queued = store.enqueue_skill_candidate_from_trace(
                    account_id=ctx.account_id,
                    agent_id=ctx.agent_id,
                    session_id=session_id,
                    turn_id=str(finalized["turn_id"]),
                    trace_uri=str(finalized["uri"]),
                )
                finalized["skill_candidate_id"] = queued["candidate_id"]
            return finalized
        except ValueError as exc:
            logger.warning(
                "finalize_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "invalid trace boundary",
                "error_code": "invalid_trace_boundary",
                "retryable": False,
            }
        except Exception as exc:
            logger.warning(
                "finalize_session_trace failed session=%s: %s",
                session_id[:8],
                exc,
            )
            return {
                "ok": False,
                "error": "trace finalization unavailable",
                "error_code": "trace_finalization_unavailable",
                "retryable": True,
            }

    def after_turn(self, params: dict) -> dict:
        messages = params.get("messages", [])
        if not messages:
            return {"ok": True}

        client_request_id = str(
            params.get("clientRequestId") or params.get("client_request_id") or ""
        ).strip()
        if not client_request_id:
            return self._after_turn(params)
        if len(client_request_id) > 128:
            raise ValueError("client_request_id must be at most 128 characters")

        ctx = params.get("_ctx") or self.build_context(params)
        buf = self.get_session_manager().get_or_create(
            params.get("sessionId", "unknown"),
            ctx=ctx,
        )
        claim_status = buf.claim_request(client_request_id)
        if claim_status == "completed":
            return self._after_turn(params)
        if claim_status == "processing":
            return {
                "ok": False,
                "status": "processing",
                "retryable": True,
                "client_request_id": client_request_id,
            }

        try:
            result = self._after_turn(params)
        except Exception:
            buf.release_request(client_request_id)
            raise
        if result.get("ok", True) is False:
            buf.release_request(client_request_id)
        return result

    def _after_turn(self, params: dict) -> dict:
        """Keep governance, thresholding, and extraction dispatch cleanup-safe."""
        if self._tool_result_externalizer is None:
            return self._after_turn_impl(params)
        messages = params.get("messages", [])
        if not messages:
            return self._after_turn_impl(params)
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        buf = self.get_session_manager().begin_managed_operation(session_id, ctx)
        operation_guarded = self._tool_result_externalizer is not None
        try:
            if params.get("_ctx") is None:
                params = {**params, "_ctx": ctx}
            return self._after_turn_impl(params)
        finally:
            buf.end_operation()

    def _after_turn_impl(self, params: dict) -> dict:
        """Unified write entry point: accumulate session buffer + extract memories.

        Step 1 (every turn): add new messages to session buffer (lightweight).
        Step 2 (threshold reached): extract memories via LLM → write to AGFS →
              drain outbox (embed + upsert vector index) → commit session archive.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if not messages:
            return {"ok": True}

        ctx = params.get("_ctx") or self.build_context(params)
        client_request_id = str(
            params.get("clientRequestId") or params.get("client_request_id") or ""
        ).strip()
        if len(client_request_id) > 128:
            raise ValueError("client_request_id must be at most 128 characters")
        pre_prompt_count = _bounded_message_count(
            params.get("prePromptMessageCount", 0),
            len(messages),
        )
        if (
            self._tool_result_externalizer is None
            and not client_request_id
            and not messages[pre_prompt_count:]
        ):
            return {"ok": True, "status": "no_new_messages"}

        mgr = self.get_session_manager()
        buf = mgr.get_or_create(session_id, ctx=ctx)

        def _save_idempotency_state() -> bool:
            if (
                self._tool_result_externalizer is None
                and not getattr(self._cfg, "session_state_bridge_enabled", True)
            ):
                return True
            return mgr.save_session_state(session_id, ctx)

        def _complete_request(result: dict) -> dict:
            if client_request_id and result.get("ok", True):
                buf.mark_request_completed(client_request_id)
                result["client_request_id"] = client_request_id
                if not _save_idempotency_state():
                    return {
                        "ok": False,
                        "status": "failed",
                        "error": "idempotency state unavailable",
                        "retryable": True,
                        "client_request_id": client_request_id,
                    }
                buf.release_request(client_request_id)
            return result
        governance_fallback_token = params.get(
            "toolResultGovernanceFallbackToken"
        )
        governance_fallback_original = bool(governance_fallback_token)
        if governance_fallback_original and (
            self._tool_result_externalizer is None
            or not self._validate_tool_result_governance_fallback_token(
                governance_fallback_token,
                params,
                ctx,
                session_id,
            )
        ):
            raise ValueError("invalid tool-result governance fallback token")

        if client_request_id and buf.request_completed(client_request_id):
            if not _save_idempotency_state():
                return {
                    "ok": False,
                    "status": "failed",
                    "error": "idempotency state unavailable",
                    "retryable": True,
                    "client_request_id": client_request_id,
                }
            buf.release_request(client_request_id)
            return {
                "ok": True,
                "status": "duplicate",
                "client_request_id": client_request_id,
            }

        # Keep the checkpoint delta for usage/extraction accounting. Tool-result
        # governance may separately replay the full adapter-attested semantic turn.
        original_semantic_identity = self._semantic_turn_identity(params)
        original_semantic_start = (
            original_semantic_identity[1]
            if original_semantic_identity is not None
            else None
        )
        stripped_messages = messages
        projected_original_indices = tuple(range(len(messages)))
        (
            list_stripped_messages,
            retained_list_indices,
            retained_list_prefix_counts,
            list_projection_changed,
        ) = self._strip_internal_artifact_list_roundtrips(
            stripped_messages,
            ctx=ctx,
            session_id=session_id,
        )
        if list_projection_changed:
            pre_prompt_count = self._map_projected_message_boundary(
                retained_list_prefix_counts,
                pre_prompt_count,
            )
            projected_boundaries = {"prePromptMessageCount": pre_prompt_count}
            if original_semantic_identity is not None:
                projected_boundaries["semanticTurnStartMessageCount"] = (
                    self._map_projected_message_boundary(
                        retained_list_prefix_counts,
                        original_semantic_start,
                    )
                )
            params = {**params, **projected_boundaries}
            stripped_messages = list_stripped_messages
            projected_original_indices = tuple(
                projected_original_indices[index]
                for index in retained_list_indices
            )
        new_messages = stripped_messages[pre_prompt_count:]

        if self._tool_result_externalizer is None and not new_messages:
            return _complete_request({"ok": True, "status": "no_new_messages"})

        # Step 1: Accumulate in session buffer (lightweight, every turn)
        # Dedup: skip messages already written by ingest()
        semantic_messages = None
        semantic_highwater_rollback = None
        semantic_checkpoint = None
        semantic_turn_lock_held = False
        semantic_turn_complete = None
        semantic_completion_transition = False
        semantic_governance_failed = False
        semantic_governance_pending = False
        ledger_registration = None
        if self._tool_result_externalizer is not None:
            semantic_mode = self._tool_result_semantic_mode(params)
            identity = self._semantic_turn_identity(params)
            if semantic_mode == "user_message" or (
                semantic_mode == "auto" and identity is None
            ):
                ledger_registration = self._register_user_message_tool_turn(
                    new_messages,
                    start_boundary=pre_prompt_count,
                    ctx=ctx,
                    session_id=session_id,
                )
                if ledger_registration is not None:
                    semantic_messages = ledger_registration.messages
                    new_messages = ledger_registration.messages
                    semantic_turn_complete = True
                    if (
                        not ledger_registration.changed
                        and ledger_registration.entry.get("governance_complete") is True
                    ):
                        # The complete Turn was already persisted and upserted.
                        # An exact transport retry must not refill a buffer that
                        # may already have been archived and drained.
                        new_messages = []
            if ledger_registration is None and semantic_mode != "user_message":
                semantic_messages = self._semantic_turn_messages(
                    stripped_messages,
                    params,
                    checkpoint_boundary=pre_prompt_count,
                    original_indices=projected_original_indices,
                    original_semantic_start=original_semantic_start,
                )
            else:
                identity = None
            if semantic_messages is not None and identity is not None:
                buf.semantic_turn_lock.acquire()
                semantic_turn_lock_held = True
                semantic_turn_id, semantic_start, source_namespace = identity
                checkpoint_count = len(semantic_messages)
                requested_delta_start = max(0, pre_prompt_count - semantic_start)
                semantic_turn_complete = params.get("semanticTurnComplete") is True
                state_key = "tool_result_semantic_turn"
                with buf.extraction_lock:
                    previous = deepcopy(buf.runtime_extensions.get(state_key))
                    ledger = self._semantic_turn_ledger(previous)
                    completed_turn_keys = self._semantic_turn_completed_keys(previous)
                    semantic_turn_key = self._semantic_turn_key(
                        semantic_turn_id,
                        source_namespace,
                    )
                    previous_entry = next(
                        (
                            row for row in ledger
                            if row["turn_id"] == semantic_turn_id
                            and row["source_namespace"] == source_namespace
                        ),
                        None,
                    )
                    previous_count = (
                        previous_entry["max_message_count"]
                        if previous_entry is not None
                        else 0
                    )
                    previously_complete = bool(
                        (
                            previous_entry is not None
                            and previous_entry["complete"]
                        )
                        or semantic_turn_key in completed_turn_keys
                    )
                    if previous_entry is None and previously_complete:
                        previous_count = checkpoint_count
                    previously_pending = bool(
                        previous_entry is not None
                        and previous_entry.get("governance_pending") is True
                    )
                    previous_pending_count = (
                        previous_entry.get("pending_message_count", previous_count)
                        if previous_entry is not None
                        else previous_count
                    )
                    semantic_governance_pending = previously_pending
                    effective_delta_start = max(
                        requested_delta_start,
                        min(previous_count, checkpoint_count),
                    )
                    if previously_pending:
                        # A prior store failure did not advance stats. Retry from
                        # the last successfully governed high-water even if the
                        # adapter's checkpoint delta starts later.
                        effective_delta_start = min(previous_count, checkpoint_count)
                    if previously_complete and not previously_pending:
                        # A completed semantic turn is immutable. Retried or
                        # out-of-order checkpoints cannot create fresh stats or
                        # rewrite its stable messages.
                        effective_delta_start = checkpoint_count
                    new_messages = stripped_messages[
                        semantic_start + effective_delta_start:
                    ]
                    semantic_completion_transition = bool(
                        semantic_turn_complete and not previously_complete
                    )
                    should_govern_checkpoint = bool(
                        new_messages
                        or semantic_completion_transition
                        or previously_pending
                    )
                    if should_govern_checkpoint:
                        # This is a provisional claim, not a successful final.
                        # It may be persisted concurrently, so it must remain
                        # retryable even if the process exits during Store I/O.
                        updated_entry = {
                            "turn_id": semantic_turn_id,
                            "source_namespace": source_namespace,
                            # Successful governance/stats high-water. Never
                            # advance it until Store I/O and upsert complete;
                            # a persisted provisional snapshot must retry the
                            # entire unaccounted delta after process restart.
                            "max_message_count": previous_count,
                            "complete": previously_complete,
                            "governance_pending": True,
                            "pending_message_count": max(
                                previous_pending_count,
                                checkpoint_count,
                            ),
                        }
                        ledger = [
                            row for row in ledger
                            if not (
                                row["turn_id"] == semantic_turn_id
                                and row["source_namespace"] == source_namespace
                            )
                        ]
                        ledger.append(updated_entry)
                        claimed_state = self._semantic_turn_ledger_state(
                            ledger,
                            completed_turn_keys,
                        )
                        buf.runtime_extensions[state_key] = claimed_state
                        semantic_highwater_rollback = (
                            state_key,
                            previous,
                            deepcopy(claimed_state),
                        )
                        semantic_checkpoint = (
                            state_key,
                            semantic_turn_id,
                            source_namespace,
                            checkpoint_count,
                            updated_entry["max_message_count"],
                            updated_entry["pending_message_count"],
                            updated_entry["complete"],
                            semantic_turn_complete,
                            previous_count,
                            previously_complete,
                            previous_pending_count,
                            semantic_turn_key,
                            completed_turn_keys,
                        )
        if (
            not new_messages
            and not semantic_completion_transition
            and not semantic_governance_pending
        ):
            if semantic_turn_lock_held:
                buf.semantic_turn_lock.release()
            status = (
                "semantic_turn_accumulating"
                if semantic_turn_complete is False
                else "no_new_messages"
            )
            return _complete_request({"ok": True, "status": status})

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
        from extraction.tool_collector import update_buffer_stats
        operation_guarded = self._tool_result_externalizer is not None
        if operation_guarded:
            buf.begin_operation()
        try:
            fallback_reuse = []
            if semantic_messages is None:
                with buf.extraction_lock:
                    fallback_reuse = list(
                        buf.messages[buf.extraction_watermark:]
                    )
            governance_messages = (
                self._mark_validated_semantic_sources(semantic_messages)
                if semantic_messages is not None
                else self._attach_server_fallback_sources(
                    self._strip_unverified_semantic_sources(new_messages),
                    reuse_from=fallback_reuse,
                )
            )
            governed = self._govern_request_messages(
                governance_messages,
                ctx=ctx,
                session_id=session_id,
                persistence=True,
                fallback_original=governance_fallback_original,
            )
            if governed is not None:
                failed_count = int(getattr(governed, "failed_count", 0) or 0)
                fallback_applied = bool(
                    getattr(governed, "fallback_original", False)
                )
                if semantic_checkpoint is None:
                    self._upsert_governed_messages(
                        session_id, governed, ctx, tail_fallback=True,
                    )
                    if fallback_applied:
                        self._record_tool_result_governance_fallback(
                            buf,
                            governed.source_message_ids,
                        )
                    semantic_governance_failed = failed_count > 0
                    if ledger_registration is not None and not failed_count:
                        if self._mark_user_message_turn_governed(
                            ledger_registration,
                            ctx=ctx,
                            session_id=session_id,
                        ):
                            if not fallback_applied:
                                self._record_user_message_turn_artifacts(
                                    ledger_registration,
                                    ctx=ctx,
                                    session_id=session_id,
                                )
                        else:
                            semantic_governance_failed = not fallback_applied
                else:
                    (
                        state_key,
                        turn_id,
                        namespace,
                        observed_checkpoint_count,
                        claimed_message_count,
                        claimed_pending_count,
                        claimed_complete,
                        requested_complete,
                        previous_count,
                        previously_complete,
                        previous_pending_count,
                        semantic_turn_key,
                        completed_turn_keys,
                    ) = semantic_checkpoint
                    with buf.extraction_lock:
                        current = buf.runtime_extensions.get(state_key)
                        current_entry = next(
                            (
                                row for row in self._semantic_turn_ledger(current)
                                if row["turn_id"] == turn_id
                                and row["source_namespace"] == namespace
                            ),
                            None,
                        )
                        checkpoint_is_current = bool(
                            current_entry is not None
                            and current_entry["max_message_count"] == claimed_message_count
                            and current_entry["pending_message_count"] == claimed_pending_count
                            and current_entry["governance_pending"] is True
                            and current_entry["complete"] == claimed_complete
                        )
                    if checkpoint_is_current:
                        # semantic_turn_lock serializes all checkpoint claims;
                        # never call SessionManager while holding extraction_lock,
                        # whose global lock order is the inverse cleanup order.
                        self._upsert_governed_messages(
                            session_id, governed, ctx, tail_fallback=True,
                        )
                        if fallback_applied:
                            self._record_tool_result_governance_fallback(
                                buf,
                                governed.source_message_ids,
                            )
                        pending_count = max(
                            previous_pending_count,
                            observed_checkpoint_count,
                        )
                        pending_after_success = bool(
                            previous_pending_count > observed_checkpoint_count
                        )
                        finalized_entry = {
                            "turn_id": turn_id,
                            "source_namespace": namespace,
                            "max_message_count": (
                                previous_count
                                if failed_count
                                else max(previous_count, observed_checkpoint_count)
                            ),
                            "complete": bool(
                                previously_complete
                                or (
                                    requested_complete
                                    and not failed_count
                                    and not pending_after_success
                                )
                            ),
                            "governance_pending": bool(
                                failed_count or pending_after_success
                            ),
                            "pending_message_count": (
                                pending_count
                                if failed_count or pending_after_success
                                else max(previous_count, observed_checkpoint_count)
                            ),
                        }
                        finalized_completed_turn_keys = list(completed_turn_keys)
                        if finalized_entry["complete"]:
                            finalized_completed_turn_keys.append(semantic_turn_key)
                        with buf.extraction_lock:
                            current = buf.runtime_extensions.get(state_key)
                            current_ledger = self._semantic_turn_ledger(current)
                            current_entry = next(
                                (
                                    row for row in current_ledger
                                    if row["turn_id"] == turn_id
                                    and row["source_namespace"] == namespace
                                ),
                                None,
                            )
                            if (
                                current_entry is not None
                                and current_entry["max_message_count"] == claimed_message_count
                                and current_entry["pending_message_count"] == claimed_pending_count
                                and current_entry["governance_pending"] is True
                                and current_entry["complete"] == claimed_complete
                            ):
                                current_ledger = [
                                    row for row in current_ledger
                                    if not (
                                        row["turn_id"] == turn_id
                                        and row["source_namespace"] == namespace
                                    )
                                ]
                                current_ledger.append(finalized_entry)
                                buf.runtime_extensions[state_key] = (
                                    self._semantic_turn_ledger_state(
                                        current_ledger,
                                        finalized_completed_turn_keys,
                                    )
                                )
                        semantic_governance_failed = failed_count > 0
                        semantic_governance_pending = finalized_entry[
                            "governance_pending"
                        ]
                if not semantic_governance_failed:
                    for msg in new_messages:
                        parsed = extract_content_and_tool_calls(msg.get("content", ""))
                        if parsed["tool_calls"]:
                            update_buffer_stats(buf.usage_stats, parsed["tool_calls"], session_id=session_id)
            else:
                # Snapshot buffer tail once to avoid TOCTOU race with concurrent ingest()
                buf_snapshot = list(buf.messages[-len(new_messages):]) if buf.messages else []
                for msg in new_messages:
                    role = msg.get("role", "user")
                    content_raw = msg.get("content", "")
                    parsed = extract_content_and_tool_calls(content_raw)
                    content = parsed["text"]

                    # Preserve the pre-feature ingest/after_turn fallback: one
                    # matching tail entry may satisfy repeated incoming copies.
                    if content and buf_snapshot:
                        for existing in buf_snapshot:
                            if existing.role == role and existing.content == content:
                                content = None
                                break

                    if content:
                        created_at = msg.get("created_at") if isinstance(msg, dict) else getattr(msg, "created_at", None)
                        mgr.add_message(session_id, role, content, ctx, created_at=created_at)
                    if parsed["tool_calls"]:
                        update_buffer_stats(buf.usage_stats, parsed["tool_calls"], session_id=session_id)
        except Exception:
            if semantic_highwater_rollback is not None:
                state_key, previous_state, claimed_state = semantic_highwater_rollback
                with buf.extraction_lock:
                    if buf.runtime_extensions.get(state_key) == claimed_state:
                        if previous_state is None:
                            buf.runtime_extensions.pop(state_key, None)
                        else:
                            buf.runtime_extensions[state_key] = previous_state
            raise
        finally:
            if operation_guarded:
                buf.end_operation()
            if semantic_turn_lock_held:
                buf.semantic_turn_lock.release()

        if semantic_governance_failed or semantic_governance_pending:
            return {
                "ok": False,
                "status": "tool_result_governance_pending",
                "retryable": True,
                "toolResultGovernanceFallbackToken": (
                    self._issue_tool_result_governance_fallback_token(
                        params,
                        ctx,
                        session_id,
                    )
                ),
            }

        # Check pending_tokens
        session = mgr.get_session(session_id, ctx)
        pending_tokens = session.get("pending_tokens", 0)
        force_extract = self._param_bool(
            params, "forceExtract", "force_extract", default=False
        )

        if semantic_turn_complete is False:
            return _complete_request({
                "ok": True,
                "status": "semantic_turn_accumulating",
                "pending_tokens": pending_tokens,
            })

        if pending_tokens < threshold and not force_extract:
            return _complete_request(
                {"ok": True, "status": "accumulating", "pending_tokens": pending_tokens}
            )

        # Step 2: Threshold reached → extract memories + write + index + archive
        write_api = self.get_write_api()
        if write_api is None:
            logger.warning("after_turn: write API unavailable, session commit only")
            result = mgr.commit(session_id, ctx, wait=bool(client_request_id))
            return _complete_request({"ok": True, **result})

        # Extract candidates → filter → write to AGFS → create OutboxEvents
        # Use INCREMENTAL messages (after watermark) + extraction summary
        buf = mgr.get_or_create(session_id, ctx=ctx)
        extraction_run_id = uuid4().hex[:12]
        extraction_claim = buf.claim_incremental_extraction()
        extraction_in_progress_before = extraction_claim["extraction_in_progress_before"]
        incremental = extraction_claim["incremental"]
        if not incremental:
            if extraction_in_progress_before and client_request_id:
                return {
                    "ok": False,
                    "status": "processing",
                    "retryable": True,
                    "pending_tokens": pending_tokens,
                    "client_request_id": client_request_id,
                }
            return _complete_request(
                {
                    "ok": True,
                    "status": "processing" if extraction_in_progress_before else "no_new_messages",
                    "pending_tokens": pending_tokens,
                }
            )

        try:
            extraction_state = self._build_incremental_extraction_state(
                buf,
                incremental=incremental,
            )
        except Exception:
            buf.rewind_watermark_to_ids({message.id for message in incremental})
            if extraction_claim["extraction_started"]:
                buf.end_extraction()
            raise
        archive_snapshot = list(incremental)
        archive_snapshot_ids = {m.id for m in archive_snapshot}
        extraction_messages = extraction_state["messages"]
        original_extraction_watermark = extraction_claim["original_watermark"]

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
            original_extraction_watermark,
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
            return _complete_request(
                {
                    "ok": True,
                    "status": "no_extractable_messages",
                    "drain": drain_stats,
                    "commit": commit_result,
                    "extraction_run_id": extraction_run_id,
                }
            )

        # Keep the upstream async after_turn behavior, but source extraction
        # inputs from the shared incremental-state helper so after_turn and
        # prepare_compaction operate on the same delta semantics.
        tool_stats_text = extraction_state["tool_stats_text"]
        session_time = extraction_state["session_time"] or request_session_time
        extraction_watermark = original_extraction_watermark
        extraction_summary = extraction_claim["extraction_summary"]

        import time as _time

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
                if commit_result.get("status") != "failed":
                    _complete_request({"ok": True, "status": "completed"})
            except Exception as exc:
                rewound = buf.rewind_watermark_to_ids(archive_snapshot_ids)
                logger.error(
                    "after_turn background extract failed for run=%s session=%s "
                    "rewound=%s: %s",
                    extraction_run_id, session_id[:8], rewound, exc, exc_info=True,
                )
            finally:
                buf.end_extraction()

        if client_request_id or self._param_bool(params, "wait", "wait", default=False):
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
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn wait extraction write failed: "
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
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(
                        buf.extraction_summary, write_result
                    )
                drain_stats = self.drain_outbox_sync(account_id=ctx.account_id)
                commit_result = mgr.commit_snapshot(
                    session_id,
                    archive_snapshot,
                    ctx,
                    wait=True,
                    archive_id=archive_id,
                )
                removed = 0
                if commit_result.get("status") == "failed":
                    rewound = _rewind_failed_archive_watermark()
                    logger.error(
                        "after_turn wait snapshot commit failed: "
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
                        "drain": drain_stats,
                        "commit": commit_result,
                        "extraction_run_id": extraction_run_id,
                    }
                if commit_result.get("archived"):
                    removed = buf.remove_messages_by_id(archive_snapshot_ids)
                    if removed:
                        mgr.save_session_state(session_id, ctx)
                return _complete_request(
                    {
                        "ok": True,
                        **write_result,
                        "status": "completed",
                        "extraction_run_id": extraction_run_id,
                        "drain": drain_stats,
                        "commit": commit_result,
                    }
                )
            except Exception as exc:
                rewound = _rewind_failed_archive_watermark()
                logger.error(
                    "after_turn wait extraction failed for run=%s session=%s "
                    "rewound=%s: %s",
                    extraction_run_id,
                    session_id[:8],
                    rewound,
                    exc,
                    exc_info=True,
                )
                return {
                    "ok": False,
                    "status": "failed",
                    "error": str(exc),
                    "extraction_run_id": extraction_run_id,
                }
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
            return _complete_request(
                {
                    "ok": True,
                    "status": "completed",
                    "extraction_run_id": extraction_run_id,
                    **write_result,
                }
            )

        return {"ok": True, "status": "processing", "extraction_run_id": extraction_run_id}

    def ingest(self, params: dict) -> dict:
        """Ingest messages into the session buffer.

        Writes messages to the session buffer so that compact() can find data
        even when called before after_turn().  after_turn() will deduplicate
        messages already written here.
        """
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if (
            self._tool_result_externalizer is not None
            and not messages
            and isinstance(params.get("message"), dict)
        ):
            messages = [params["message"]]
        if not messages:
            return {"ingested": True}
        ctx = params.get("_ctx") or self.build_context(params)
        mgr = self.get_session_manager()
        if (
            self._tool_result_externalizer is not None
            and self._tool_result_semantic_mode(params) == "user_message"
        ):
            # In a user-message lifecycle, ingest can run before the model has
            # finished consuming the current tool output. Persisting a fallback
            # Artifact here would both risk first-view replacement and leave a
            # duplicate beside the strong Artifact created from the complete
            # terminal afterTurn. Tool payloads must not be persisted raw while
            # governance is deferred: a failed/missing afterTurn or unavailable
            # Artifact store would otherwise leave the large result eligible for
            # later archive/compaction. Ordinary user/assistant messages retain
            # the pre-feature ingest behavior.
            normalized = self._tool_result_externalizer.normalizer.normalize(messages)
            deferred_indexes = {
                item.message_index for item in normalized if item.tool_results
            }
            count = 0
            for message_index, msg in enumerate(messages):
                if message_index in deferred_indexes:
                    continue
                role = (
                    msg.get("role", "user")
                    if isinstance(msg, dict)
                    else getattr(msg, "role", "user")
                )
                content_raw = (
                    msg.get("content", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                )
                content = extract_content_and_tool_calls(content_raw).get("text", "")
                created_at = (
                    msg.get("created_at")
                    if isinstance(msg, dict)
                    else getattr(msg, "created_at", None)
                )
                if content:
                    mgr.add_message(
                        session_id,
                        role,
                        content,
                        ctx,
                        created_at=created_at,
                    )
                    count += 1
            return {
                "ingested": True,
                "count": count,
                "externalized": 0,
                "governance_deferred": True,
                "deferred_tool_result_messages": len(deferred_indexes),
            }
        if self._tool_result_externalizer is None:
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
        buf = mgr.begin_managed_operation(session_id, ctx)
        try:
            # Ingest is a generic persistence entry point, not a trusted
            # lifecycle identity boundary. Ignore caller-owned private fields;
            # afterTurn will later reattach a validated semantic identity when
            # the adapter supplies one or the server registers a complete Turn.
            messages = self._attach_server_fallback_sources(
                self._strip_unverified_semantic_sources(messages)
            )
            governed = self._govern_request_messages(
                messages,
                ctx=ctx,
                session_id=session_id,
                persistence=True,
            )
            if governed is not None:
                count = self._upsert_governed_messages(
                    session_id, governed, ctx, tail_fallback=False,
                )
                if int(getattr(governed, "failed_count", 0) or 0) > 0:
                    return {
                        "ingested": False,
                        "ok": False,
                        "count": count,
                        "externalized": governed.externalized_count,
                        "status": "tool_result_governance_pending",
                        "retryable": True,
                    }
                return {
                    "ingested": True,
                    "count": count,
                    "externalized": governed.externalized_count,
                }

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
        finally:
            buf.end_operation()

    def ingest_batch(self, params: dict) -> dict:
        """Batch ingest through the standard externalization path."""
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

    def _build_incremental_extraction_state(self, buf, incremental=None) -> dict:
        if incremental is None:
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

    def _run_dispose_background(self, mgr, session_id: str, params: dict, ctx, buf) -> None:
        snapshot_ids: set[str] = set()
        extraction_started = False
        succeeded = False
        effective_ctx = ctx
        try:
            with buf.extraction_lock:
                event = buf.extraction_done_event if buf.extraction_in_progress else None
            if event is not None:
                event.wait(timeout=600)
                with buf.extraction_lock:
                    if buf.extraction_in_progress:
                        logger.warning(
                            "dispose: extraction still active after timeout; keeping session=%s",
                            session_id[:8],
                        )
                        return

            with buf.extraction_lock:
                original_watermark = buf.extraction_watermark
                snapshot = list(buf.messages[original_watermark:])
                snapshot_ids = {message.id for message in snapshot}
                pending_tokens = sum(len(message.content) // 4 for message in snapshot)
                extraction_messages = [
                    {"role": message.role, "content": message.content}
                    for message in snapshot
                    if message.role != "assistant"
                ]
                if snapshot:
                    buf.extraction_watermark = len(buf.messages)
                if pending_tokens >= 200 and extraction_messages:
                    buf.begin_extraction()
                    extraction_started = True
                    extraction_summary = buf.extraction_summary

            if not snapshot:
                succeeded = True
            elif pending_tokens < 200 or not extraction_messages:
                buf.remove_messages_by_id(snapshot_ids)
                succeeded = True
            else:
                write_api = self.get_write_api()
                if write_api is None:
                    raise RuntimeError("dispose write API unavailable")
                effective_ctx = ctx or self.build_context(params)
                tool_stats_text = ""
                try:
                    from extraction.tool_collector import format_tool_stats_text

                    tool_stats_text = format_tool_stats_text(buf.usage_stats)
                except Exception:
                    pass
                run_id = f"dispose-{uuid4().hex[:12]}"
                write_result = self._chunk_and_extract(
                    messages=extraction_messages,
                    write_api=write_api,
                    ctx=effective_ctx,
                    session_time=None,
                    session_summary=extraction_summary,
                    tool_stats_text=tool_stats_text,
                    extraction_run_id=run_id,
                    session_id=session_id,
                )
                if self._write_result_has_failures(write_result):
                    raise RuntimeError("dispose extraction write failed")
                self._invalidate_identity_cache_if_profile_written(session_id, write_result)
                self.drain_outbox_sync(account_id=effective_ctx.account_id)
                removed = buf.remove_messages_by_id(snapshot_ids)
                if removed:
                    mgr.save_session_state(session_id, effective_ctx)
                succeeded = True
                logger.info(
                    "dispose background flush done: run=%s session=%s extracted=%d writes=%d removed=%d",
                    run_id,
                    session_id[:8],
                    write_result.get("candidates_extracted", 0),
                    write_result.get("writes_completed", 0),
                    removed,
                )
        except Exception:
            if snapshot_ids:
                buf.rewind_watermark_to_ids(snapshot_ids)
            logger.exception("dispose background flush failed: session=%s", session_id[:8])
        finally:
            if extraction_started:
                buf.end_extraction()
            try:
                if succeeded:
                    self._save_session_state_best_effort(mgr, session_id, params)
                    removed_session = mgr.remove_session_if_empty(
                        session_id,
                        buf,
                        ctx=effective_ctx,
                    )
                    if removed_session:
                        self._flush_session_tool_usage_async(session_id, effective_ctx)
            finally:
                with buf.extraction_lock:
                    buf.dispose_in_progress = False

    def dispose(self, params: dict | None = None) -> dict:
        """Session end: safely dispatch a non-blocking pending-memory flush."""
        params = params or {}
        self.wake_skill_sedimentation()
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"disposed": True, "flushed": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        ctx = params.get("_ctx")
        if not mgr.has_session(session_id, ctx=ctx):
            return {"disposed": True, "flushed": False, "reason": "session_not_found"}

        buf = mgr.get_or_create(session_id, ctx=ctx)
        fast_path_reason = None
        with buf.extraction_lock:
            if buf.dispose_in_progress:
                return {
                    "disposed": True,
                    "flushed": False,
                    "status": "background_flush_dispatched",
                    "reason": "dispose_in_progress",
                }
            buf.dispose_in_progress = True
            extraction_active = buf.extraction_in_progress
            reason = "extraction_in_progress" if extraction_active else "background_flush_dispatched"
            pending_tokens = None
            if not extraction_active:
                unextracted = list(buf.messages[buf.extraction_watermark:])
                pending_tokens = sum(len(message.content) // 4 for message in unextracted)
                if not unextracted:
                    reason = "no_pending_messages"
                elif pending_tokens < 200:
                    reason = "below_token_threshold"
                elif not any(message.role != "assistant" for message in unextracted):
                    reason = "no_user_messages"

                if reason in {
                    "no_pending_messages",
                    "below_token_threshold",
                    "no_user_messages",
                }:
                    fast_path_reason = reason
                    if unextracted:
                        snapshot_ids = {message.id for message in unextracted}
                        buf.extraction_watermark = len(buf.messages)
                        buf.remove_messages_by_id(snapshot_ids)

        if fast_path_reason is not None:
            try:
                self._save_session_state_best_effort(mgr, session_id, params)
                removed_session = mgr.remove_session_if_empty(
                    session_id,
                    buf,
                    ctx=ctx,
                )
                if removed_session:
                    self._flush_session_tool_usage_async(session_id, ctx)
            finally:
                with buf.extraction_lock:
                    buf.dispose_in_progress = False

            result = {
                "disposed": True,
                "flushed": False,
                "reason": fast_path_reason,
            }
            if pending_tokens is not None:
                result["pending_tokens"] = pending_tokens
            return result

        try:
            threading.Thread(
                target=self._run_dispose_background,
                args=(mgr, session_id, params, ctx, buf),
                daemon=True,
                name=f"dispose-{session_id[:8]}",
            ).start()
        except Exception as exc:
            with buf.extraction_lock:
                buf.dispose_in_progress = False
            logger.error(
                "dispose background thread unavailable: session=%s: %s",
                session_id[:8],
                exc,
                exc_info=True,
            )
            return {"disposed": True, "flushed": False, "reason": "thread_unavailable"}

        result = {
            "disposed": True,
            "flushed": False,
            "status": "dispose_deferred" if extraction_active else "background_flush_dispatched",
            "reason": reason,
        }
        if pending_tokens is not None:
            result["pending_tokens"] = pending_tokens
        return result

    def wait_until_idle(self, params: dict | None = None) -> dict:
        """Wait for process-local dispose/extraction/indexing to settle."""
        params = params or {}
        session_id = params.get("sessionId") or params.get("session_id") or ""
        if not session_id:
            return {"idle": False, "reason": "no_session_id"}

        mgr = self.get_session_manager()
        ctx = params.get("_ctx") or self.build_context(params)
        drain_outbox = self._param_bool(params, "drainOutbox", "drain_outbox", default=True)
        wait_outbox = self._param_bool(params, "waitOutbox", "wait_outbox", default=not drain_outbox)
        if not mgr.has_session(session_id, ctx=ctx):
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

        buf = mgr.get_or_create(session_id, ctx=ctx)
        timeout_seconds = self._idle_timeout_seconds(params)
        import time as _time

        deadline = _time.monotonic() + timeout_seconds
        while True:
            with buf.extraction_lock:
                extraction_in_progress = bool(buf.extraction_in_progress)
                dispose_in_progress = bool(buf.dispose_in_progress)
                extraction_event = buf.extraction_done_event
                active_extractions = getattr(buf, "extraction_active_count", 0)

            if not extraction_in_progress and not dispose_in_progress:
                break

            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return {
                    "idle": False,
                    "reason": (
                        "dispose_in_progress_timeout"
                        if dispose_in_progress
                        else "extraction_in_progress_timeout"
                    ),
                    "extraction_in_progress": extraction_in_progress,
                    "dispose_in_progress": dispose_in_progress,
                    "active_extractions": active_extractions,
                }

            # Extraction has an event; dispose itself does not, so poll briefly
            # while sharing the same overall timeout budget for both states.
            if (
                extraction_in_progress
                and extraction_event is not None
                and not extraction_event.is_set()
            ):
                extraction_event.wait(timeout=remaining)
            else:
                _time.sleep(min(0.01, remaining))

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
            "status": "ok",
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
                if self._cfg.retrieval_hybrid_enabled:
                    self._get_shared_vector_index()
                    lexical_health = self._lexical_index.health()
                    info["lexical"] = lexical_health
                    if not lexical_health.get("available", False):
                        raise RuntimeError("required openGauss BM25 projection is unavailable")
                    backfill_health = self._get_lexical_backfill().health()
                    info["lexical_backfill"] = backfill_health
                    if (
                        backfill_health.get("queued")
                        or backfill_health.get("running")
                        or backfill_health.get("at_capacity")
                        or backfill_health.get("cooldown_failures")
                        or backfill_health.get("last_errors")
                        or backfill_health.get("stopping")
                    ):
                        info["status"] = "degraded"
                        info["error"] = "lexical historical projection is not ready"
            llm = self.get_llm()
            info["llm"] = getattr(llm, "model", "unknown")
            if self._cfg.skill_sedimentation_enabled:
                worker_status = (
                    self._skill_sedimentation_worker.status()
                    if self._skill_sedimentation_worker is not None
                    else {"running": False, "last_error": "worker not initialized"}
                )
                worker_status["thread_alive"] = bool(
                    self._skill_sedimentation_thread
                    and self._skill_sedimentation_thread.is_alive()
                )
                trace_store = self._get_trace_event_store()
                worker_status["scheduler_backlog"] = (
                    trace_store.count_skill_work_accounts() if trace_store else None
                )
                info["skill_sedimentation"] = worker_status
                if not worker_status["thread_alive"]:
                    info["status"] = "error"
                    info["error"] = "skill sedimentation worker is not running"
                elif worker_status.get("last_error"):
                    info["status"] = "error"
                    info["error"] = str(worker_status["last_error"])
        except Exception as exc:
            info["status"] = "error"
            info["error"] = str(exc)
        return info
