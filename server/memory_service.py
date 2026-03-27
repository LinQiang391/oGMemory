"""MemoryService — Core oG-Memory business logic, decoupled from transport.

Used by both:
  - server/app.py        (HTTP / Flask mode)
  - bridge/memory_api.py (CLI / subprocess mode)
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from core.models import RequestContext, RetrievalConfig
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


def format_memory_addition(hits: list) -> str:
    """Format search hits into a system-prompt snippet."""
    if not hits:
        return "[oG-Memory] No relevant memories found."
    lines = [
        "[oG-Memory 长期记忆系统] 以下是从向量数据库中语义检索到的用户相关记忆，"
        "请优先参考这些记忆来回答用户问题。如果用户询问你是否记得某些事，"
        "请明确说明这些信息来自 oG-Memory 长期记忆："
    ]
    for hit in hits[:5]:
        cat = hit.category or "memory"
        abstract = hit.abstract or "无摘要"
        score_pct = int(hit.score * 100) if hit.score > 0 else 0
        lines.append(f"  - [{cat}] {abstract} (相关度: {score_pct}%)")
    return "\n".join(lines)


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
            pipeline = RetrievalPipeline(
                planner=QueryPlanner(cfg),
                seed_retriever=SeedRetriever(vector_index, embedder, cfg),
                hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
                assembly=AssemblyService(cfg),
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
        messages = params.get("messages", [])
        prompt = params.get("prompt", "")
        raw = prompt.strip() if prompt else extract_query(messages)
        query = sanitize_query(raw)

        if not query:
            return {
                "messages": messages,
                "estimatedTokens": len(messages) * 200,
                "systemPromptAddition": "[oG-Memory] No context available yet.",
            }

        system_prompt_addition = ""
        try:
            ctx = self.build_context(params)
            result = self.get_read_api().search_memory(
                query=query, ctx=ctx, top_k=5, mode="QUICK",
            )
            hits = result.hits if result else []
            system_prompt_addition = format_memory_addition(hits)

            logger.info("assemble: %d hits for query=%.100s", len(hits), query)
            for i, h in enumerate(hits):
                logger.info(
                    "  hit[%d] [%s] score=%d%% uri=%s",
                    i, h.category or "?",
                    int(h.score * 100) if h.score > 0 else 0,
                    h.uri,
                )
        except Exception as exc:
            logger.warning("assemble search failed: %s", exc, exc_info=True)
            system_prompt_addition = (
                "[oG-Memory] Memory search temporarily unavailable. "
                "Conversation continues normally."
            )

        return {
            "messages": messages,
            "estimatedTokens": len(messages) * 200,
            "systemPromptAddition": system_prompt_addition,
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
            result = write_api.commit_session(messages, ctx, confidence_threshold=0.5)
            logger.info(
                "after_turn session=%s extracted=%d completed=%d failed=%d",
                session_id,
                result.get("candidates_extracted", 0),
                result.get("writes_completed", 0),
                result.get("writes_failed", 0),
            )
            return {"ok": True, **result}
        except Exception as exc:
            logger.error("after_turn failed: %s", exc, exc_info=True)
            return {"ok": True, "error": str(exc)}

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
