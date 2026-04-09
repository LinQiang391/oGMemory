"""MemoryService — Core oG-Memory business logic, decoupled from transport.

Used by both:
  - server/app.py        (HTTP / Flask mode)
  - bridge/memory_api.py (CLI / subprocess mode)
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from core.models import RequestContext, RetrievalConfig, TokenBudget, ArchiveRef, AssembleResult
from session.models import ArchiveEntry
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

    # -- RequestContext factory ------------------------------------------------

    def build_context(self, params: dict) -> RequestContext:
        """Build RequestContext, allowing per-request overrides."""
        return RequestContext(
            account_id=params.get("accountId", self._default_account_id),
            user_id=params.get("userId", self._default_user_id),
            agent_id=params.get("agentId", self._default_agent_id),
            session_id=params.get("sessionId", "unknown"),
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
        except Exception:
            pass
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

    def _build_system_prompt(
        self,
        profile: str,
        latest_archives: list[ArchiveRef],
        pre_archives: list[ArchiveRef],
        working_set: list[dict],
        budget: TokenBudget,
    ) -> str:
        """Build system prompt with stable prefix + dynamic suffix.

        Order matters for KV cache:
          1. Profile       — most stable (rarely changes)
          2. Archive History — semi-stable (grows when sessions end)
          3. Working Set   — dynamic (changes every turn)

        The caller should place this at the START of the system prompt so
        the stable prefix can be cached across turns.
        """
        sections: list[str] = []
        used_tokens = 0
        budget_tokens = budget.total

        # -- Stable: Profile --
        if profile:
            text = f"## Profile\n{profile}"
            sections.append(text)
            used_tokens += len(text) // 4

        # -- Stable: Archive History --
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

        if archive_parts:
            archive_text = "## Archive History\n" + "\n".join(archive_parts)
            archive_tokens = len(archive_text) // 4
            if used_tokens + archive_tokens <= budget_tokens:
                sections.append(archive_text)
                used_tokens += archive_tokens

        # -- Dynamic: Working Set (last position) --
        if working_set:
            ws_lines = ["## Working Set"]
            for item in working_set:
                cat = item.get("category", "memory")
                abstract = item.get("abstract", "")
                score = item.get("score", 0)
                score_pct = int(score * 100) if score > 0 else 0
                ws_lines.append(f"- [{cat}] {abstract} (相关度: {score_pct}%)")
            ws_text = "\n".join(ws_lines)
            ws_tokens = len(ws_text) // 4
            remaining = budget_tokens - used_tokens
            if remaining > 100 and ws_tokens > 0:
                if ws_tokens > remaining:
                    ws_text = ws_text[: remaining * 4]
                sections.append(ws_text)

        return "\n\n".join(sections)

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
        return self._write_api

    def get_read_api(self):
        if self._read_api is None:
            config = ProviderConfig.from_env()
            embedder = config.create_embedder()
            vector_index = config.create_vector_index()

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
        return self._read_api

    # -- Lifecycle handlers ---------------------------------------------------

    def assemble(self, params: dict) -> dict:
        """Assemble memory context for the current turn.

        6-step pipeline, structured for KV cache:
          1. Extract query + budget
          2. Read profile          (stable — rarely changes)
          3. Collect archives       (stable — grows when sessions end)
          4. Search working set     (dynamic — changes every turn)
          5. Build system prompt    (stable prefix + dynamic suffix)
          6. Return                 (original messages untouched)

        systemPromptAddition layout:
          ## Profile            ← stable
          ## Archive History    ← stable
          ## Working Set        ← dynamic (last = cache breakpoint)

        Args:
            params: Dict with optional keys:
                - messages: list[dict] - Current conversation messages
                - prompt: str - Direct query prompt (optional)
                - tokenBudget: int - Token budget (default: 128_000)

        Returns:
            Dict with:
                - messages: list[dict] - Original messages, untouched
                - systemPromptAddition: str - Context for system prompt
                - estimatedTokens: int - Total estimated tokens
                - archiveCount: int - Number of archives found
                - archiveIncluded: bool - Whether archives were injected
        """
        messages = params.get("messages", [])
        prompt = params.get("prompt", "")
        raw = prompt.strip() if prompt else extract_query(messages)
        query = sanitize_query(raw)

        # Step 1: Budget
        token_budget_value = params.get("tokenBudget", 128_000)
        token_budget = TokenBudget(total=token_budget_value)

        # Empty query: return messages unchanged
        if not query:
            return {
                "messages": messages,
                "systemPromptAddition": "",
                "estimatedTokens": self._estimate_tokens(messages),
            }

        try:
            ctx = self.build_context(params)
        except Exception as exc:
            logger.warning("assemble build_context failed: %s", exc, exc_info=True)
            return {
                "messages": messages,
                "systemPromptAddition": "",
                "estimatedTokens": self._estimate_tokens(messages),
            }

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

            # Step 5: Build system prompt — stable prefix + dynamic suffix
            system_prompt = self._build_system_prompt(
                profile=profile,
                latest_archives=latest_archives,
                pre_archives=pre_archives,
                working_set=working_set,
                budget=token_budget,
            )

            # Step 6: Return — original messages untouched
            archive_count = len(latest_archives) + len(pre_archives)
            prompt_tokens = len(system_prompt) // 4
            msg_tokens = self._estimate_tokens(messages)

            return {
                "messages": messages,
                "systemPromptAddition": system_prompt,
                "estimatedTokens": msg_tokens + prompt_tokens,
                "archiveCount": archive_count,
                "archiveIncluded": archive_count > 0,
            }

        except Exception as exc:
            logger.warning("assemble failed: %s", exc, exc_info=True)
            return {
                "messages": messages,
                "systemPromptAddition": "",
                "estimatedTokens": self._estimate_tokens(messages),
            }

    def after_turn(self, params: dict) -> dict:
        session_id = params.get("sessionId", "unknown")
        messages = params.get("messages", [])
        if not messages:
            return {"ok": True}

        ctx = self.build_context(params)
        write_api = self.get_write_api()
        if write_api is None:
            logger.warning("after_turn: write API unavailable")
            return {"ok": True, "skipped": True}

        try:
            # Use async extraction (wait=False)
            result = write_api.commit_session(messages, ctx, confidence_threshold=0.5, wait=False)

            # Log async extraction initiated
            task_id = result.get("task_id", "unknown")
            logger.info(
                "after_turn session=%s task_id=%s status=%s",
                session_id,
                task_id,
                result.get("status", "unknown"),
            )

            # Return async result
            return {"ok": True, "task_id": task_id, "status": result.get("status", "processing")}
        except Exception as exc:
            logger.error("after_turn failed: %s", exc, exc_info=True)
            return {"ok": True, "error": str(exc)}

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
        return {"ok": True, "compacted": True, "method": "og-memory-noop"}

    def bootstrap(self, params: dict) -> dict:
        return {"bootstrapped": True}

    def prepare_subagent_spawn(self, params: dict) -> dict:
        return {"prepared": True}

    def on_subagent_ended(self, params: dict) -> dict:
        return {"cleaned": True}

    def dispose(self, params: dict | None = None) -> dict:
        self._llm = None
        self._write_api = None
        self._read_api = None
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
