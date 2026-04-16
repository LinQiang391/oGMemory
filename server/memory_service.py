"""MemoryService — Core oG-Memory business logic, decoupled from transport.

Used by both:
  - server/app.py        (HTTP / Flask mode)
  - bridge/memory_api.py (CLI / subprocess mode)
"""

from __future__ import annotations

import logging
import os
import threading
from uuid import uuid4

from core.models import RequestContext, RetrievalConfig, TokenBudget, ArchiveRef, ComposedContext, Role
from session.models import ArchiveEntry, SessionWindowState
from session.session_manager import SessionManager
from session.rolling_compressor import RollingCompressor
from providers.config import ProviderConfig
from providers.llm import get_openai_llm
from providers.unified_config import OgMemConfig, get_config
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner, sanitize_query
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.result_ranker import ResultRanker
from retrieval.context_reader import ContextReader
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService, ResolvedIdentity
from server.control_plane_store import ControlPlaneStore
from server.tenant_admin import TenantAdminService

try:
    from pyagfs import AGFSClient
    from fs.agfs_adapter import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI, ReadAPI
    from commit.outbox_store import OutboxStore
    _HAS_AGFS = True
except ImportError:
    _HAS_AGFS = False

logger = logging.getLogger("ogmem.service")


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

        # Shared vector index + embedder (used by both read API and outbox worker)
        self._vector_index = None
        self._embedder = None
        self._outbox_thread: threading.Thread | None = None
        self._control_store = None
        self._key_manager = None
        self._auth = None
        self._audit = None
        self._tenant_admin = None

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

        if identity is not None:
            requested_agent_id = explicit_agent_id
            ctx = self.get_auth_service().build_request_context(
                identity,
                account_id=self._default_account_id,
                user_id=self._default_user_id,
                agent_id=requested_agent_id or "",
                session_id=session_id,
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
        user_id = params.get("userId") or self._default_user_id
        account_id = params.get("accountId") or self._default_account_id
        logger.debug("build_context userId=%s (from params: %s)", user_id, params.get("userId"))
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
            if _HAS_AGFS:
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
        """Read user profile content directly from AGFS.

        Profile may be stored as a single node or as field-level child nodes
        (e.g. profile/name, profile/location, profile/identity). This method
        handles both: if the profile node has content, use it; otherwise
        enumerate children and concatenate their content.
        """
        if not _HAS_AGFS:
            return ""
        try:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            profile_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"

            # Try reading as a single node first
            if agfs.exists(profile_uri, ctx):
                node = agfs.read_node(profile_uri, ctx)
                if node.content:
                    return node.content

            # Field-level split: enumerate children and read each
            try:
                children = agfs.list_children(profile_uri, ctx)
            except Exception as exc:
                logger.error("list_children failed in _read_profile: %s", exc)
                raise

            if not children:
                return ""

            parts = []
            for child_uri in children:
                if agfs.exists(child_uri, ctx):
                    child_node = agfs.read_node(child_uri, ctx)
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
        if not _HAS_AGFS:
            return [], []

        try:
            from session import SessionArchiveStore

            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            store = SessionArchiveStore(fs=agfs)

            entries = store.list_archives(session_id=ctx.session_id, ctx=ctx)
            return build_archive_refs(entries, token_budget)
        except Exception as exc:
            logger.warning("_collect_archives failed: %s", exc, exc_info=True)
            return [], []


    def _search_working_set(self, query: str, ctx: RequestContext) -> list[dict]:
        """Vector search for relevant memories (top_k=50, score >= 0.6)."""
        try:
            result = self.get_read_api().search_memory(
                query=query, ctx=ctx, top_k=50, fill_content_for_top_k=3, mode="QUICK"
            )
            if not result or not result.hits:
                return []
            return [
                {
                    "uri": h.uri,
                    "abstract": h.abstract or "",
                    "content": h.content_excerpt or "",
                    "score": h.score,
                    "category": h.category,
                }
                for h in result.hits
                if h.score >= 0.3
            ][:15]
        except Exception as exc:
            logger.warning("Working set search failed: %s", exc)
        return []


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
            "Use them to answer the question below. If the answer is found in these memories, respond based on them.",
            "",
        ]
        for item in working_set:
            cat = item.get("category", "memory")
            abstract = item.get("abstract", "")
            content = item.get("content", "")
            text_to_show = content if content else abstract
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
        buf = mgr.get_or_create(session_id)
        window_state = buf.window_state

        if not buf.should_compress():
            return window_state

        # Need to compress — use RollingCompressor
        try:
            compressor = RollingCompressor(llm=self.get_llm())
            window_state = compressor.compress(buf.messages, window_state)
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

        if window_state.confirmed_constraints:
            items = "\n".join(f"- {c}" for c in window_state.confirmed_constraints[:5])
            sections.append(f"## Confirmed Constraints\n{items}")

        if window_state.recent_decisions:
            items = "\n".join(f"- {d}" for d in window_state.recent_decisions[:5])
            sections.append(f"## Recent Decisions\n{items}")

        if window_state.open_loops:
            items = "\n".join(f"- {l}" for l in window_state.open_loops[:5])
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
        if not _HAS_AGFS:
            logger.info("OutboxWorker skipped: AGFS not available")
            return
        try:
            from index.outbox_worker import OutboxWorker
            from commit.outbox_store import OutboxStore

            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()

            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            outbox_store = OutboxStore(client=client, fs=agfs, mount_prefix=self._mount_prefix)

            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=agfs,
                llm=self.get_llm(),
            )

            def _outbox_loop():
                import time
                logger.info("OutboxWorker thread started (polling every 5s)")
                while True:
                    try:
                        worker.run_once(
                            outbox_store=outbox_store,
                            account_ids=[self._default_account_id],
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

    def drain_outbox_sync(self) -> dict:
        """Synchronously process pending OutboxEvents: embed → upsert to vector index.

        In subprocess mode the background OutboxWorker thread dies with the process,
        so we must drain inline before returning to ensure data is indexed.
        ChromaDB persists to disk, so the next subprocess call will find the data.
        """
        if not _HAS_AGFS:
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}
        try:
            from index.outbox_worker import OutboxWorker
            from commit.outbox_store import OutboxStore

            vector_index = self._get_shared_vector_index()
            embedder = self._get_shared_embedder()

            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            outbox_store = OutboxStore(client=client, fs=agfs, mount_prefix=self._mount_prefix)

            worker = OutboxWorker(
                vector_index=vector_index,
                embedder=embedder,
                fs=agfs,
                llm=self.get_llm(),
            )

            stats = worker.run_once(
                outbox_store=outbox_store,
                account_ids=[self._default_account_id],
            )
            logger.info("drain_outbox_sync: %s", stats)
            return stats
        except Exception as exc:
            logger.warning("drain_outbox_sync failed: %s", exc)
            return {"processed": 0, "succeeded": 0, "failed": 1, "error": str(exc)}

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
                OpenAILLM, _ = get_openai_llm()
                self._llm = OpenAILLM(
                    api_key=cfg.openai_api_key,
                    base_url=cfg.openai_base_url,
                    model=cfg.openai_llm_model,
                )
            logger.info("LLM ready: %s", type(self._llm).__name__)
        return self._llm

    def get_write_api(self):
        if self._write_api is None:
            if not _HAS_AGFS:
                return None
            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            outbox = OutboxStore(client=client, fs=agfs, mount_prefix=self._mount_prefix)

            # Create SchemaRegistry for dynamic tool generation
            schema_registry = None
            try:
                from extraction.schemas.registry import SchemaRegistry
                schema_registry = SchemaRegistry()
                logger.info("SchemaRegistry initialized for dynamic tool generation")
            except Exception as exc:
                logger.warning("SchemaRegistry initialization failed, using legacy tools: %s", exc)

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
                fs=agfs,
                llm=self.get_llm(),
                outbox_store=outbox,
                schema_registry=schema_registry,
                vector_index=vector_index,
                embedder=embedder,
                uri_resolver=uri_resolver,
            )
            logger.info("WriteAPI ready")
            # Start outbox worker so written memories get indexed
            self._start_outbox_worker()
        return self._write_api

    def get_read_api(self):
        if self._read_api is None:
            embedder = self._get_shared_embedder()
            vector_index = self._get_shared_vector_index()

            cfg = RetrievalConfig()

            # Create relation_store if AGFS is available
            relation_store = None
            if _HAS_AGFS:
                try:
                    rs_client = AGFSClient(api_base_url=self._agfs_base_url)
                    relation_store = AGFSRelationStore(client=rs_client, mount_prefix=self._mount_prefix)
                except Exception:
                    pass

            # Create context_reader for content filling (shared by pipeline and read_service)
            context_reader = None
            if _HAS_AGFS:
                try:
                    client = AGFSClient(api_base_url=self._agfs_base_url)
                    agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
                    context_reader = ContextReader(fs=agfs)
                except Exception as exc:
                    logger.warning("AGFS unavailable for read_memory: %s", exc)

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
                "ReadAPI ready: embedder=%s, index=%s, agfs=%s",
                type(embedder).__name__,
                type(vector_index).__name__,
                "yes" if context_reader else "no",
            )
            # Start outbox worker so pending events from ingest get indexed
            self._start_outbox_worker()
        return self._read_api

    # -- Lifecycle handlers ---------------------------------------------------

    def get_session_manager(self) -> SessionManager:
        """Lazy-init SessionManager with current service dependencies."""
        if self._session_mgr is None:
            self._session_mgr = SessionManager(
                get_llm=self.get_llm,
                get_write_api=self.get_write_api,
                get_agfs=self._get_agfs_fs,
            )
            logger.info("SessionManager ready")
        return self._session_mgr

    def _get_agfs_fs(self):
        """Create a fresh AGFSContextFS instance for session operations."""
        if not _HAS_AGFS:
            return None
        try:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            return AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
        except Exception as exc:
            logger.warning("_get_agfs_fs failed: %s", exc)
            return None

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
        prompt = params.get("prompt", "")
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
        profile = ""
        try:
            profile = self._read_profile(ctx)
        except Exception as exc:
            logger.warning("assemble profile read failed: %s", exc)

        latest_archives, pre_archives = [], []
        try:
            latest_archives, pre_archives = self._collect_archives(ctx, token_budget)
        except Exception as exc:
            logger.warning("assemble archive collection failed: %s", exc)

        working_set = []
        try:
            working_set = self._search_working_set(query, ctx)
        except Exception as exc:
            logger.warning("assemble working set search failed: %s", exc)

        # Each step degrades independently — a failure in one doesn't kill the rest

        # Step 5a: Build stable identity (Profile only) → systemPromptAddition
        identity_context = ""
        try:
            identity_context = self._format_profile(
                profile=profile,
                budget=budget_allocation,
            )
        except Exception as exc:
            logger.warning("assemble identity_context build failed: %s", exc)

        # Step 5b: Build episodic history (Archives only) → systemPromptSuffix prefix
        episodic_context = ""
        try:
            episodic_context = self._format_archives(
                latest_archives=latest_archives,
                pre_archives=pre_archives,
                budget=budget_allocation,
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
            session_id = params.get("sessionId") or ctx.session_id
            window_state = self._get_session_state(session_id, ctx)
            session_state_suffix = self._format_session_state(
                window_state, budget_allocation,
            )
        except Exception as exc:
            logger.warning("assemble session_state_suffix build failed: %s", exc)

        # Step 5d: Trim messages already compressed by RollingCompressor
        # If session_state exists, compressed messages are redundant — skip them
        trimmed_messages = messages
        if session_state_suffix and window_state.turn_count_at_last_compress > 0:
            pre_prompt_count = params.get("prePromptMessageCount", 0)
            compressed = window_state.turn_count_at_last_compress
            end_of_compressed = pre_prompt_count + compressed
            if end_of_compressed < len(messages):
                trimmed_messages = messages[:pre_prompt_count] + messages[end_of_compressed:]
                logger.info(
                    "compose trimmed %d compressed messages (pre_prompt=%d, kept=%d)",
                    compressed, pre_prompt_count, len(trimmed_messages),
                )
            elif pre_prompt_count < len(messages):
                # compressed count exceeds available messages — keep at least pre_prompt
                trimmed_messages = messages[:pre_prompt_count]
                logger.info(
                    "compose trimmed to pre_prompt only (pre_prompt=%d, compressed=%d)",
                    pre_prompt_count, compressed,
                )

        # Step 6: Assemble final output
        try:
            system_prompt_suffix = "\n\n".join(
                p for p in [episodic_context, session_state_suffix] if p
            )

            archive_count = len(latest_archives) + len(pre_archives)
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
                "session": len(session_state_suffix) // 4,
                "retrieved": len(working_set_msg) // 4,
            }

            # Create ComposedContext with all semantic slots populated
            result = ComposedContext(
                # Semantic slots
                identity_context=identity_context,
                episodic_context=episodic_context,
                session_context=session_state_suffix,
                task_context="",
                retrieved_evidence=working_set_msg,
                open_loops=open_loops,
                uncertainties=uncertainties,
                # Budget metadata
                estimated_tokens=msg_tokens + prompt_tokens,
                budget_used_by_slot=budget_used_by_slot,
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

        Message injection order:
          1. Profile (identity_context) — before original messages
          2. Original messages
          3. Working Set (retrieved_evidence) — after original messages

        Archive history and session state are omitted from the final output.
        systemPromptAddition is always empty — all context goes into messages.
        """
        injected_messages = []

        if result.identity_context:
            injected_messages.append({"role": "user", "content": result.identity_context})

        injected_messages.extend(result.messages)

        if result.retrieved_evidence:
            injected_messages.append({"role": "user", "content": result.retrieved_evidence})

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
            "episodicContext": "",
            "sessionContext": "",
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
        logger.info(
            "after_turn entry: session=%s msgs=%d pre_prompt=%d threshold=%s keys=%s",
            session_id, len(messages), pre_prompt_count,
            params.get("commitTokenThreshold", "default"),
            sorted(params.keys()),
        )
        if not messages:
            return {"ok": True}

        ctx = params.get("_ctx") or self.build_context(params)

        # Extract new messages only (skip pre-prompt system messages)
        new_messages = messages[pre_prompt_count:]
        if not new_messages:
            return {"ok": True, "status": "no_new_messages"}

        # Derive session_time from first message's created_at (matches OpenViking behavior)
        session_time = None
        first_msg = new_messages[0] if new_messages else None
        if first_msg:
            created_at = first_msg.get("created_at") if isinstance(first_msg, dict) else getattr(first_msg, "created_at", None)
            if created_at:
                try:
                    from datetime import datetime as _dt
                    if isinstance(created_at, str):
                        session_time = _dt.fromisoformat(created_at.replace("Z", "+00:00"))
                    elif isinstance(created_at, (int, float)):
                        session_time = _dt.fromtimestamp(created_at)
                    else:
                        session_time = created_at
                    logger.info("after_turn session_time=%s (from first msg created_at)", session_time)
                except Exception as exc:
                    logger.warning("after_turn failed to parse created_at=%s: %s", created_at, exc)

        # Threshold controlled by env var OGMEM_AFTER_TURN_THRESHOLD.
        # Default 5000 tokens (~5-10 turns). Set to 1 for ingest (always extract), 999999 for QA (never extract).
        import os as _os
        threshold = int(_os.environ.get("OGMEM_AFTER_TURN_THRESHOLD", "200"))

        # Step 1: Accumulate in session buffer (lightweight, every turn)
        mgr = self.get_session_manager()
        from extraction.tool_collector import update_buffer_stats
        for msg in new_messages:
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")
            parsed = extract_content_and_tool_calls(content_raw)
            content = parsed["text"]
            if content:
                mgr.add_message(session_id, role, content, ctx)
            # Accumulate tool call metadata into buffer stats
            if parsed["tool_calls"]:
                buf = mgr.get_or_create(session_id)
                update_buffer_stats(buf.tool_usage_stats, parsed["tool_calls"])

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
        buf = mgr.get_or_create(session_id)
        incremental = buf.messages[buf.extraction_watermark:]
        incremental_messages = [{"role": m.role, "content": m.content} for m in incremental]
        # Filter out assistant messages for extraction — they don't contain user
        # information and dilute the prompt (e.g., "Already saved", "OK").
        extraction_messages = [
            m for m in incremental_messages
            if m.get("role") != "assistant"
        ]
        if not extraction_messages:
            # No user messages to extract from — skip extraction, advance
            # watermark, and proceed to drain + archive.
            buf.extraction_watermark = len(buf.messages)
            drain_stats = self.drain_outbox_sync()
            mgr.commit(session_id, ctx, wait=True)
            return {"ok": True, "status": "no_extractable_messages", "drain": drain_stats}

        # Format tool usage statistics for injection into extraction prompt
        from extraction.tool_collector import format_tool_stats_text
        tool_stats_text = format_tool_stats_text(buf.tool_usage_stats)

        write_result = write_api.commit_session(
            messages=extraction_messages,
            ctx=ctx,
            confidence_threshold=0.5,
            wait=True,
            session_time=session_time,
            session_summary=buf.extraction_summary,
            tool_stats_text=tool_stats_text,
        )
        logger.info(
            "after_turn extract done: extracted=%d writes=%d watermark=%d→%d",
            write_result.get("candidates_extracted", 0),
            write_result.get("writes_completed", 0),
            buf.extraction_watermark,
            len(buf.messages),
        )

        # Update watermark and summary for next incremental extraction
        buf.extraction_watermark = len(buf.messages)
        if write_result.get("candidates_extracted", 0) > 0:
            buf.extraction_summary = _update_extraction_summary(
                buf.extraction_summary, write_result
            )

        # Synchronously drain outbox: embed → upsert to vector index
        try:
            drain_stats = self.drain_outbox_sync()
            logger.info(
                "outbox_drain processed=%d succeeded=%d failed=%d",
                drain_stats.get("processed", 0),
                drain_stats.get("succeeded", 0),
                drain_stats.get("failed", 0),
            )
        except Exception as exc:
            logger.warning("outbox_drain skipped: %s", exc)

        # Commit session archive
        mgr.commit(session_id, ctx, wait=True)

        return {"ok": True, "status": "completed", **write_result}

    def ingest(self, params: dict) -> dict:
        """Lightweight ack — OpenClaw calls this every turn, but actual write
        logic lives in after_turn().  Intentionally a no-op to avoid
        double-processing since OpenClaw also calls afterTurn().
        """
        return {"ingested": True}

    def ingest_batch(self, params: dict) -> dict:
        return {"ingested": True, "count": len(params.get("messages", []))}

    def compact(self, params: dict) -> dict:
        """Compact session: commit synchronously, return compressed context."""
        session_id = params.get("sessionId", "unknown")
        ctx = params.get("_ctx") or self.build_context(params)
        token_budget = params.get("tokenBudget", 128_000)

        mgr = self.get_session_manager()

        # Estimate tokensBefore
        session = mgr.get_session(session_id, ctx)
        tokens_before = session.get("pending_tokens", 0)

        # Before commit, extract from buffer messages (compact doesn't go
        # through after_turn, so this is the only extraction entry point).
        write_api = self.get_write_api()
        buf = mgr.get_or_create(session_id)
        if write_api and buf.messages:
            msgs = [
                {"role": m.role, "content": m.content}
                for m in buf.messages
                if m.role != "assistant"
            ]
            if msgs:
                write_result = write_api.commit_session(
                    msgs, ctx,
                    session_summary=buf.extraction_summary,
                    confidence_threshold=0.5,
                    wait=True,
                )
                buf.extraction_watermark = len(buf.messages)
                # Update extraction_summary so subsequent extractions
                # know what was already extracted
                if write_result.get("candidates_extracted", 0) > 0:
                    buf.extraction_summary = _update_extraction_summary(
                        buf.extraction_summary, write_result,
                    )

        # Commit synchronously (wait=true)
        commit_result = mgr.commit(session_id, ctx, wait=True)

        if not commit_result.get("archived"):
            return {
                "ok": True,
                "compacted": False,
                "reason": commit_result.get("reason", "no_archive"),
                "result": {"summary": "", "tokensBefore": tokens_before},
            }

        # Get compressed context (archive overview as summary)
        context = mgr.get_context(session_id, token_budget, ctx)
        summary = context.get("latest_archive_overview", "")
        tokens_after = context.get("estimatedTokens", 0)

        return {
            "ok": True,
            "compacted": True,
            "result": {
                "summary": summary,
                "firstKeptEntryId": commit_result.get("archive_id", ""),
                "tokensBefore": tokens_before,
                "tokensAfter": tokens_after,
            },
        }

    def bootstrap(self, params: dict) -> dict:
        return {"bootstrapped": True}

    def prepare_subagent_spawn(self, params: dict) -> dict:
        return {"prepared": True}

    def on_subagent_ended(self, params: dict) -> dict:
        return {"cleaned": True}

    def dispose(self, params: dict | None = None) -> dict:
        # In HTTP mode, keep all state alive across turns.
        # _read_api holds the vector index (InMemoryVectorIndex loses data if reset).
        # _session_mgr holds session buffers.
        # _llm and _write_api are cheap to recreate if needed.
        # Only clear on explicit shutdown.
        return {"disposed": True}

    def health(self) -> dict:
        info: dict = {"backend": "og-memory", "agfs": _HAS_AGFS}
        try:
            if _HAS_AGFS:
                AGFSClient(api_base_url=self._agfs_base_url).ls("/")
                info["agfs_url"] = self._agfs_base_url
            llm = self.get_llm()
            info["llm"] = getattr(llm, "model", "unknown")
            info["status"] = "ok"
        except Exception as exc:
            info["status"] = "error"
            info["error"] = str(exc)
        return info
