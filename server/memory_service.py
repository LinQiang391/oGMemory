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

from core.models import RequestContext, RetrievalConfig, TokenBudget, ArchiveRef, AssembleResult
from session.models import ArchiveEntry, SessionWindowState
from session.session_manager import SessionManager
from session.rolling_compressor import RollingCompressor
from providers.config import ProviderConfig
from providers.llm import get_openai_llm
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner, sanitize_query
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.assembly_service import AssemblyService
from retrieval.memory_read_service import MemoryReadService

try:
    from pyagfs import AGFSClient
    from fs.agfs_adapter import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI, MemoryReadAPI
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


def extract_query(messages: list[dict]) -> str:
    """Build a search query from the last 3 user messages."""
    user_texts = []
    for msg in messages:
        if msg.get("role") == "user":
            text = extract_content_text(msg.get("content", ""))
            if text:
                user_texts.append(text)
    return "\n".join(user_texts[-3:]).strip()


def format_profile_section(profile: str) -> str:
    """Format user profile for system prompt injection.

    Returns empty string if no profile — caller should not inject anything.
    """
    if not profile:
        return ""
    return f"[oG-Memory 用户档案]\n{profile}"


def format_memory_section(hits: list) -> str:
    """Format memory search hits for user-message injection.

    Returns empty string if no hits — caller should not inject anything.
    """
    if not hits:
        return ""

    lines = [
        "[oG-Memory 长期记忆] 以下是从记忆库中语义检索到的用户相关记忆。"
        "你不需要再使用工具搜索这些信息。"
        "如果用户询问你是否记得某些事，"
        "请明确说明这些信息来自 oG-Memory 长期记忆：",
    ]
    for hit in hits[:3]:
        cat = hit.category or "memory"
        abstract = hit.abstract or "无摘要"
        score_pct = int(hit.score * 100) if hit.score > 0 else 0
        entry = f"  - [{cat}] {abstract} (相关度: {score_pct}%)"

        # Relation info
        if hasattr(hit, "relations") and hit.relations:
            rel_items = []
            for rel in hit.relations[:3]:
                rel_type = rel.relation_type
                target = rel.to_uri.split("/")[-1]
                rel_items.append(f"{rel_type}:{target}")
            if rel_items:
                entry += f" → 关联: {', '.join(rel_items)}"

        lines.append(entry)

    lines.append("  (如需详细内容，可使用 read_memory 工具读取具体记忆)")
    return "\n".join(lines)


# Backward-compatible alias: merges profile + memory into one string
def format_memory_addition(hits: list, profile: str = "") -> str:
    """Format profile + memory hits into a single snippet (legacy).

    Prefer format_profile_section() + format_memory_section() for new code.
    """
    parts = []
    ps = format_profile_section(profile)
    if ps:
        parts.append(ps)
    ms = format_memory_section(hits)
    if ms:
        parts.append(ms)
    if not parts:
        return "[oG-Memory] No relevant memories found."
    return "\n\n".join(parts)


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

    def __init__(
        self,
        agfs_base_url: str | None = None,
        agfs_mount_prefix: str = "/local/plugin",
        default_account_id: str | None = None,
        default_user_id: str | None = None,
        default_agent_id: str = "main",
    ):
        self._agfs_base_url = agfs_base_url or os.environ.get(
            "AGFS_BASE_URL", "http://localhost:1833"
        )
        self._mount_prefix = agfs_mount_prefix
        self._default_account_id = default_account_id or os.environ.get(
            "OG_ACCOUNT_ID", "acct-demo"
        )
        self._default_user_id = default_user_id or os.environ.get(
            "OG_USER_ID", "u-alice"
        )
        self._default_agent_id = default_agent_id

        self._llm = None
        self._write_api = None
        self._read_api = None
        self._session_mgr: SessionManager | None = None

        # Shared vector index + embedder (used by both read API and outbox worker)
        self._vector_index = None
        self._embedder = None
        self._outbox_thread: threading.Thread | None = None

    # -- RequestContext factory ------------------------------------------------

    def build_context(self, params: dict) -> RequestContext:
        """Build RequestContext, allowing per-request overrides.

        Empty strings from the plugin are treated as absent, so defaults apply.
        """
        user_id = params.get("userId") or self._default_user_id
        logger.debug("build_context userId=%s (from params: %s)", user_id, params.get("userId"))
        return RequestContext(
            account_id=params.get("accountId") or self._default_account_id,
            user_id=user_id,
            agent_id=params.get("agentId") or self._default_agent_id,
            session_id=params.get("sessionId") or "unknown",
            trace_id=str(uuid4()),
        )

    def _read_profile(self, ctx: RequestContext) -> str:
        """Read user profile L2 content directly from AGFS."""
        if not _HAS_AGFS:
            return ""
        try:
            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
            profile_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            if agfs.exists(profile_uri, ctx):
                node = agfs.read_node(profile_uri, ctx)
                return node.content or ""
        except Exception as exc:
            logger.warning("_read_profile failed for %s: %s", ctx.user_id, exc)
        return ""

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
        """Vector search for relevant memories (dynamic, changes every turn)."""
        try:
            result = self.get_read_api().search_memory(
                query=query, ctx=ctx, top_k=10, mode="QUICK"
            )
            if result and result.hits:
                return [
                    {
                        "uri": hit.uri,
                        "abstract": hit.abstract or "",
                        "score": hit.score,
                        "category": hit.category,
                    }
                    for hit in result.hits[:5]
                ]
        except Exception as exc:
            logger.warning("Working set search failed: %s", exc)
        return []

    # -- System prompt builder (KV-cache friendly) ----------------------------

    def _build_stable_identity(
        self,
        profile: str,
        budget: TokenBudget | dict[str, int],
    ) -> str:
        """Build stable identity context (Profile only) → systemPromptAddition.

        This is the MOST stable layer — changes only when user profile updates.
        Fully KV-cacheable across all turns and sessions.
        """
        if not profile:
            return ""

        if isinstance(budget, dict):
            max_tokens = budget.get("identity", 5000)
        else:
            max_tokens = budget.total

        text = f"## Profile\n{profile}"
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _build_episodic_history(
        self,
        latest_archives: list[ArchiveRef],
        pre_archives: list[ArchiveRef],
        budget: TokenBudget | dict[str, int],
    ) -> str:
        """Build episodic history context (Archives only) → systemPromptSuffix prefix.

        Goes into systemPromptSuffix BEFORE session state. Semi-stable:
        only changes when sessions are archived (not every turn).
        """
        if isinstance(budget, dict):
            max_tokens = budget.get("archive", 40000)
        else:
            max_tokens = budget.total

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

    def _build_working_set_message(self, working_set: list[dict], budget: TokenBudget | dict[str, int]) -> str:
        """Build DYNAMIC working set as user message content.

        This goes into memoryUserMessage and changes every turn.
        Placed in user message so KV cache for stable system prompt
        prefix is preserved across turns.

        Args:
            working_set: List of retrieved memory items
            budget: TokenBudget or dict allocation from allocate()
        """
        if not working_set:
            return ""
        ws_lines = ["## Working Set (relevant memories)"]
        for item in working_set:
            cat = item.get("category", "memory")
            abstract = item.get("abstract", "")
            score = item.get("score", 0)
            score_pct = int(score * 100) if score > 0 else 0
            ws_lines.append(f"- [{cat}] {abstract} (相关度: {score_pct}%)")
        ws_text = "\n".join(ws_lines)
        ws_tokens = len(ws_text) // 4

        # Handle both TokenBudget and dict allocation
        if isinstance(budget, dict):
            working_set_budget = budget.get("working_set", 20000)
        else:
            working_set_budget = budget.total // 2  # use at most 50% of budget

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

    def _build_session_state_suffix(
        self,
        window_state: SessionWindowState,
        budget: TokenBudget | dict[str, int],
    ) -> str:
        """Build Layer 2 session state suffix for system prompt.

        Goes into systemPromptSuffix — updated atomically every N turns.
        Placed at end of system prompt so KV cache for stable prefix
        (Layer 1) is preserved; only this suffix chunk changes.

        Args:
            window_state: Current session window state
            budget: TokenBudget or dict allocation from allocate()
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

        # Handle both TokenBudget and dict allocation
        if isinstance(budget, dict):
            session_state_budget = budget.get("session_state", 10000)
        else:
            session_state_budget = budget.session_state_limit

        # Truncate to budget
        max_chars = session_state_budget * 4
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        return combined

    # -- Token estimation -----------------------------------------------------

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a list of messages (~4 chars per token)."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(block.get("text", ""))
                    elif isinstance(block, str):
                        total_chars += len(block)
        return total_chars // 4

    # -- Lazy resource getters ------------------------------------------------

    def _get_shared_vector_index(self):
        """Lazy-init shared vector index (same instance for read + outbox worker)."""
        if self._vector_index is None:
            config = ProviderConfig.from_env()
            self._vector_index = config.create_vector_index()
            logger.info("Shared vector_index ready: %s", type(self._vector_index).__name__)
        return self._vector_index

    def _get_shared_embedder(self):
        """Lazy-init shared embedder (same instance for read + outbox worker)."""
        if self._embedder is None:
            config = ProviderConfig.from_env()
            self._embedder = config.create_embedder()
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

            def _worker_loop():
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
                target=_worker_loop, daemon=True, name="outbox-worker"
            )
            self._outbox_thread.start()
            logger.info("OutboxWorker background thread launched")
        except Exception as exc:
            logger.warning("Failed to start OutboxWorker: %s", exc, exc_info=True)

    def get_llm(self):
        if self._llm is None:
            config = ProviderConfig.from_env()
            if config.provider == "mock" and os.environ.get("OPENAI_API_KEY"):
                config.provider = "openai"
            if config.provider == "mock":
                from providers.llm import MockLLM
                self._llm = MockLLM()
            else:
                OpenAILLM, _ = get_openai_llm()
                self._llm = OpenAILLM(
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                    model=config.openai_llm_model,
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
            self._write_api = MemoryWriteAPI(
                fs=agfs, llm=self.get_llm(), outbox_store=outbox,
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

            pipeline = RetrievalPipeline(
                planner=QueryPlanner(cfg),
                seed_retriever=SeedRetriever(vector_index, embedder, cfg),
                hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
                assembly=AssemblyService(cfg, relation_store=relation_store),
                config=cfg,
            )

            read_service = None
            if _HAS_AGFS:
                try:
                    client = AGFSClient(api_base_url=self._agfs_base_url)
                    agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)
                    read_service = MemoryReadService(fs=agfs)
                except Exception as exc:
                    logger.warning("AGFS unavailable for read_memory: %s", exc)

            self._read_api = MemoryReadAPI(
                pipeline=pipeline,
                read_service=read_service,
                config=cfg,
            )
            logger.info(
                "ReadAPI ready: embedder=%s, index=%s, agfs=%s",
                type(embedder).__name__,
                type(vector_index).__name__,
                "yes" if read_service else "no",
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

    def assemble(self, params: dict) -> dict:
        """Assemble memory context for the current turn.

        Pipeline with 3-layer architecture:
          1. Extract query + budget
          2. Read profile          (Layer 1 — stable identity, rarely changes)
          3. Collect archives       (Layer 1b — episodic history, semi-stable)
          4. Search working set     (Layer 3 — dynamic, changes every turn)
          4b. Get/compress session state (Layer 2 — updates every N turns)
          5. Build stable identity  (Layer 1 output → systemPromptAddition)
          5b. Build episodic + session state (→ systemPromptSuffix)
          6. Return                 (original messages untouched)

        Layer 1: systemPromptAddition (stable identity, fully KV-cacheable)
          ## Profile            ← changes only on profile update

        Layer 1b+2: systemPromptSuffix (episodic + session state)
          ## Archive History    ← semi-stable, grows when sessions end
          ## Active Task        ← from structured session state
          ## Recent Session     ← rolling compressed window

        Layer 3: memoryUserMessage (dynamic, per-turn)
          ## Working Set        ← changes every turn

        Args:
            params: Dict with optional keys:
                - messages: list[dict] - Current conversation messages
                - prompt: str - Direct query prompt (optional)
                - tokenBudget: int - Token budget (default: 128_000)

        Returns:
            Dict with:
                - messages: list[dict] - Original messages, untouched
                - systemPromptAddition: str - Layer 1: stable identity (Profile only)
                - systemPromptSuffix: str - Layer 1b+2: episodic (Archives) + session state
                - memoryUserMessage: str - Layer 3: dynamic working set
                - estimatedTokens: int - Total estimated tokens
                - archiveCount: int - Number of archives found
                - archiveIncluded: bool - Whether archives were injected
                - identityContext: str - Semantic slot: profile only
                - episodicContext: str - Semantic slot: archive history
                - sessionContext: str - Semantic slot: structured session state
        """
        messages = params.get("messages", [])
        prompt = params.get("prompt", "")
        raw = prompt.strip() if prompt else extract_query(messages)
        query = sanitize_query(raw)

        # Step 1: Budget
        token_budget_value = params.get("tokenBudget", 128_000)
        token_budget = TokenBudget(total=token_budget_value)
        budget_allocation = token_budget.allocate()

        # Empty query: return messages unchanged
        if not query:
            result = AssembleResult(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._assemble_result_to_dict(result)

        try:
            ctx = self.build_context(params)
        except Exception as exc:
            logger.warning("assemble build_context failed: %s", exc, exc_info=True)
            result = AssembleResult(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._assemble_result_to_dict(result)

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

        try:
            # Step 5a: Build stable identity (Profile only) → systemPromptAddition
            identity_context = self._build_stable_identity(
                profile=profile,
                budget=budget_allocation,
            )

            # Step 5b: Build episodic history (Archives only) → systemPromptSuffix prefix
            episodic_context = self._build_episodic_history(
                latest_archives=latest_archives,
                pre_archives=pre_archives,
                budget=budget_allocation,
            )

            # Step 5c: Build dynamic working set → user message (preserves KV cache)
            working_set_msg = self._build_working_set_message(working_set, budget_allocation)

            # Step 4b: Get/compress session state → Layer 2 (updates every N turns)
            session_state_suffix = ""
            window_state = SessionWindowState()  # Default empty state
            try:
                session_id = params.get("sessionId") or ctx.session_id
                window_state = self._get_session_state(session_id, ctx)
                session_state_suffix = self._build_session_state_suffix(
                    window_state, budget_allocation,
                )
            except Exception as exc:
                logger.warning("assemble session state failed: %s", exc)

            # Step 6: Assemble final output
            # systemPromptSuffix = episodic history + session state
            # (episodic is semi-stable, session state updates every N turns)
            system_prompt_suffix = "\n\n".join(
                p for p in [episodic_context, session_state_suffix] if p
            )

            archive_count = len(latest_archives) + len(pre_archives)
            prompt_tokens = (
                len(identity_context) // 4
                + len(system_prompt_suffix) // 4
                + len(working_set_msg) // 4
            )
            msg_tokens = self._estimate_tokens(messages)

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

            # Create AssembleResult with all semantic slots populated
            result = AssembleResult(
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
                messages=messages,
                archive_count=archive_count,
                archive_included=archive_count > 0,
                system_prompt_suffix=system_prompt_suffix,
            )

            return self._assemble_result_to_dict(result)

        except Exception as exc:
            logger.warning("assemble failed: %s", exc, exc_info=True)
            result = AssembleResult(
                messages=messages,
                estimated_tokens=self._estimate_tokens(messages),
                archive_count=0,
                archive_included=False,
            )
            return self._assemble_result_to_dict(result)

    def _assemble_result_to_dict(self, result: AssembleResult) -> dict:
        """Convert AssembleResult to legacy dict format for backward compatibility.

        The AssembleResult dataclass is the source of truth.
        This method derives the legacy camelCase fields from the semantic slots.
        """
        return {
            # Legacy fields (derived from semantic slots for backward compat)
            "messages": result.messages,
            "systemPromptAddition": result.identity_context,
            "systemPromptSuffix": result.system_prompt_suffix,
            "memoryUserMessage": result.retrieved_evidence,
            "estimatedTokens": result.estimated_tokens,
            "archiveCount": result.archive_count,
            "archiveIncluded": result.archive_included,
            "stats": result.stats,

            # NEW semantic slots (cognitive contract)
            "identityContext": result.identity_context,
            "episodicContext": result.episodic_context,
            "sessionContext": result.session_context,
            "taskContext": result.task_context,
            "retrievedEvidence": result.retrieved_evidence,
            "openLoops": result.open_loops,
            "uncertainties": result.uncertainties,
            "budgetUsedBySlot": result.budget_used_by_slot,
        }

    def after_turn(self, params: dict) -> dict:
        """Accumulate messages in session buffer; commit when threshold reached."""
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        pre_prompt_count = params.get("prePromptMessageCount", 0)
        threshold = params.get("commitTokenThreshold", 20_000)

        if not messages:
            return {"ok": True}

        ctx = self.build_context(params)
        mgr = self.get_session_manager()

        # Extract new messages only (skip pre-prompt system messages)
        new_messages = messages[pre_prompt_count:]

        # Add each new message to session buffer
        for msg in new_messages:
            role = msg.get("role", "user")
            content = extract_content_text(msg.get("content", ""))
            if content:
                mgr.add_message(session_id, role, content, ctx)

        # Check pending_tokens
        session = mgr.get_session(session_id, ctx)
        pending_tokens = session.get("pending_tokens", 0)

        if pending_tokens < threshold:
            return {"ok": True, "status": "accumulating", "pending_tokens": pending_tokens}

        # Threshold reached → commit
        result = mgr.commit(session_id, ctx, wait=False)
        return {"ok": True, **result}

    def poll_extraction_status(self, params: dict) -> dict:
        """Poll the status of an async extraction task.

        Args:
            params: Dict with "taskId" key

        Returns:
            Dict with task status:
            {
                "ok": True,
                "status": "pending|processing|completed|failed",
                "memories_extracted": int
            }
        """
        task_id = params.get("taskId")
        if not task_id:
            return {"ok": False, "error": "taskId required"}

        ctx = self.build_context(params)
        write_api = self.get_write_api()

        if write_api is None:
            logger.warning("poll_extraction_status: write API unavailable")
            return {"ok": False, "error": "write API unavailable"}

        try:
            # For now, return completed status since we're doing synchronous writes
            # In a full async implementation, this would query the task store
            status = write_api.poll_task(task_id, ctx)
            return {
                "ok": True,
                "status": status.status,
                "memories_extracted": status.memories_extracted,
            }
        except Exception as exc:
            logger.error("poll_extraction_status failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    def commit_session(self, params: dict) -> dict:
        """Archive current session + trigger memory extraction.

        This is called when a session ends. It:
        1. Compresses session messages into an archive
        2. Stores the archive in AGFS
        3. Triggers async memory extraction (Phase2)

        Args:
            params: Dict with "sessionId" and "messages" keys

        Returns:
            Dict with archive and extraction info:
            {
                "ok": True,
                "archived": bool,
                "archive_id": str,
                "task_id": str,
                "memories_extracted": int
            }
        """
        ctx = self.build_context(params)
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])

        try:
            # Step 1: Compress session messages into archive
            if not _HAS_AGFS:
                logger.warning("commit_session: AGFS not available, skipping archival")
                return {"ok": False, "error": "AGFS not available"}

            from session import SessionCompressor, SessionArchiveStore

            client = AGFSClient(api_base_url=self._agfs_base_url)
            agfs = AGFSContextFS(client=client, mount_prefix=self._mount_prefix)

            # Create compressor and archive store
            compressor = SessionCompressor(llm=self.get_llm())
            store = SessionArchiveStore(fs=agfs)

            # Compress session
            overview, abstract = compressor.compress(messages)

            # Write archive
            archive_result = store.write_archive(
                session_id=session_id,
                overview=overview,
                abstract=abstract,
                messages=messages,
                ctx=ctx,
            )

            # Step 2: Trigger async memory extraction
            write_api = self.get_write_api()
            extraction = write_api.commit_session(messages, ctx, wait=False)

            logger.info(
                "commit_session session=%s archive_id=%s task_id=%s",
                session_id,
                archive_result.archive_id,
                extraction.get("task_id", "unknown"),
            )

            return {
                "ok": True,
                "archived": archive_result.success,
                "archive_id": archive_result.archive_id,
                "task_id": extraction.get("task_id"),
                "memories_extracted": extraction.get("writes_completed", 0),
            }

        except Exception as exc:
            logger.error("commit_session failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    def ingest(self, params: dict) -> dict:
        return {"ingested": True}

    def ingest_batch(self, params: dict) -> dict:
        return {"ingested": True, "count": len(params.get("messages", []))}

    def compact(self, params: dict) -> dict:
        """Compact session: commit synchronously, return compressed context."""
        session_id = params.get("sessionId", "unknown")
        ctx = self.build_context(params)
        token_budget = params.get("tokenBudget", 128_000)

        mgr = self.get_session_manager()

        # Estimate tokensBefore
        session = mgr.get_session(session_id, ctx)
        tokens_before = session.get("pending_tokens", 0)

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
