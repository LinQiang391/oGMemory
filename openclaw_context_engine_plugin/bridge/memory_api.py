"""
bridge/memory_api.py — OpenClaw Plugin Bridge to oG-Memory

This file serves as:
1. CLI entry point (receives method + json_params from index.js, dispatches calls)
2. OpenClaw lifecycle adapter (after_turn / assemble / bootstrap / compact / dispose)
3. Caller to oG-Memory service/api.py

Architecture:
    index.js (Node.js) → execFileSync → bridge/memory_api.py → service/api.py → oG-Memory internals
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# ============================================================================
# oG-Memory Imports (using relative path - no hardcoding)
# ============================================================================

# Add oG-Memory project root to Python path (relative to this file)
_file_dir = Path(__file__).parent.parent  # openclaw_context_engine_plugin/
project_root = _file_dir.parent  # oG-Memory/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Disable proxy for localhost access (WSL environment issue)
os.environ['no_proxy'] = "localhost,127.0.0.1"

from core.models import RequestContext, RetrievalConfig
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import get_openai_llm
from providers.config import ProviderConfig
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from service.api import MemoryWriteAPI, MemoryReadAPI
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.assembly_service import AssemblyService
from retrieval.memory_read_service import MemoryReadService

# ============================================================================
# Configuration
# ============================================================================

AGFS_BASE_URL = os.environ.get("AGFS_BASE_URL", "http://localhost:1833")
AGFS_MOUNT_PREFIX = "/local/plugin"
MOCK_ENGINE_LOG_LEVEL = os.environ.get("MOCK_ENGINE_LOG_LEVEL", "debug")

# Static RequestContext configuration (will be dynamic in production)
DEFAULT_ACCOUNT_ID = "default"
DEFAULT_USER_ID = "default-user"
DEFAULT_AGENT_ID = "main"

# ============================================================================
# Logging Helpers
# ============================================================================

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _log(level: str, msg: str) -> None:
    """Print a timestamped log line to stderr for visibility in gateway logs."""
    ts = _ts()
    print(f"[memory-api {ts}] [{level.upper()}] {msg}", file=sys.stderr, flush=True)


def _dump_params(method: str, params: dict) -> None:
    """Pretty-print the incoming params dict."""
    _log("debug", f"──── {method}() called ────")
    _log("debug", f"params = {json.dumps(params, indent=2, default=str)}")


def _respond(data: dict) -> None:
    """Print the JSON return value as the last line of stdout."""
    print(json.dumps(data, default=str), flush=True)


# ============================================================================
# oG-Memory Integration
# ============================================================================

def _build_request_context(session_id: str) -> RequestContext:
    """Build RequestContext from session_id (using static config for now)."""
    return RequestContext(
        account_id=DEFAULT_ACCOUNT_ID,
        user_id=DEFAULT_USER_ID,
        agent_id=DEFAULT_AGENT_ID,
        session_id=session_id,
        trace_id=str(uuid4()),
    )


def _get_llm():
    """Get or create the LLM instance (singleton pattern)."""
    if not hasattr(_get_llm, "_instance"):
        # Use ProviderConfig to create LLM (reads from environment)
        config = ProviderConfig.from_env()

        # Force openai provider if OPENAI_API_KEY is set (even if CONTEXTENGINE_PROVIDER is mock)
        if config.provider == "mock" and os.environ.get("OPENAI_API_KEY"):
            config.provider = "openai"
            _log("info", "Detected OPENAI_API_KEY, using openai provider instead of mock")

        if config.provider == "mock":
            # Use MockLLM
            from providers.llm import MockLLM
            _get_llm._instance = MockLLM()
            _log("info", f"LLM initialized: mock")
        else:
            # Use OpenAI LLM (lazy load)
            OpenAILLM, _ = get_openai_llm()
            _get_llm._instance = OpenAILLM(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                model=config.openai_llm_model,
            )
            _log("info", f"LLM initialized: openai model={config.openai_llm_model}")

    return _get_llm._instance


def _get_write_api() -> MemoryWriteAPI:
    """Get or create the MemoryWriteAPI instance (singleton pattern)."""
    if not hasattr(_get_write_api, "_instance"):
        # Initialize AGFS client
        agfs_client = AGFSClient(api_base_url=AGFS_BASE_URL)
        agfs = AGFSContextFS(client=agfs_client, mount_prefix=AGFS_MOUNT_PREFIX)

        # Initialize LLM (OpenAI)
        llm = _get_llm()

        # Create MemoryWriteAPI
        _get_write_api._instance = MemoryWriteAPI(fs=agfs, llm=llm)

    return _get_write_api._instance


def _get_read_api() -> MemoryReadAPI:
    """Get or create the MemoryReadAPI instance (singleton pattern)."""
    if not hasattr(_get_read_api, "_instance"):
        # Initialize AGFS client (reuse same instance)
        agfs_client = AGFSClient(api_base_url=AGFS_BASE_URL)
        agfs = AGFSContextFS(client=agfs_client, mount_prefix=AGFS_MOUNT_PREFIX)

        # Initialize retrieval components
        cfg = RetrievalConfig()
        embedder = MockEmbedder()
        vector_index = InMemoryVectorIndex(dimension=384)

        planner = QueryPlanner(cfg)
        seed_retriever = SeedRetriever(vector_index, embedder, cfg)
        hierarchical_searcher = HierarchicalSearcher(vector_index, cfg)
        assembly = AssemblyService(cfg)

        pipeline = RetrievalPipeline(
            planner=planner,
            seed_retriever=seed_retriever,
            hierarchical_searcher=hierarchical_searcher,
            assembly=assembly,
            config=cfg,
        )

        read_service = MemoryReadService(fs=agfs)

        # Create MemoryReadAPI
        _get_read_api._instance = MemoryReadAPI(
            pipeline=pipeline,
            read_service=read_service,
            config=cfg,
        )
        _log("info", "MemoryReadAPI initialized with MockEmbedder + InMemoryVectorIndex")

    return _get_read_api._instance


def commit_messages_to_memory(messages: list[dict], session_id: str) -> dict:
    """
    Commit OpenClaw messages to oG-Memory.

    This is the main integration point between OpenClaw and oG-Memory.

    Args:
        messages: OpenClaw format messages [{"role": "user|assistant", "content": "...", ...}, ...]
        session_id: OpenClaw session ID

    Returns:
        Write result statistics {"candidates_extracted": int, "writes_completed": int, ...}
    """
    ctx = _build_request_context(session_id)

    # Get write API and commit (real LLM extraction)
    write_api = _get_write_api()
    result = write_api.commit_session(messages, ctx, confidence_threshold=0.5)

    _log("info", f"commit_messages_to_memory "
         f"session={session_id} "
         f"extracted={result.get('candidates_extracted', 0)} "
         f"completed={result.get('writes_completed', 0)} "
         f"failed={result.get('writes_failed', 0)}")

    return result


# ============================================================================
# OpenClaw Lifecycle Methods
# ============================================================================

def ingest(params: dict) -> dict:
    """Called when a new message is added to the session."""
    _dump_params("ingest", params)

    session_id = params.get("sessionId", "unknown")
    message = params.get("message", {})
    is_heartbeat = params.get("isHeartbeat", False)

    _log("info", f"ingest  sessionId={session_id}  role={message.get('role', '?')}  heartbeat={is_heartbeat}")

    # Note: We don't write to oG-Memory here - that happens in after_turn
    return {"ingested": True}


def _extract_query(messages: list[dict]) -> str:
    """Extract search query from the last 3 user messages.

    Args:
        messages: OpenClaw format messages list

    Returns:
        Concatenated text from recent user messages (up to last 3)
    """
    # Filter user messages and get last 3
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    recent_user_messages = user_messages[-3:] if len(user_messages) > 3 else user_messages

    # Join with newlines
    query = "\n".join(recent_user_messages).strip()
    return query


def _format_memory_addition(hits: list) -> str:
    """Format memory search results into system prompt addition.

    Args:
        hits: List of RetrievedBlock from search_memory

    Returns:
        Formatted string for system prompt
    """
    if not hits:
        return "[oG-Memory] No relevant memories found."

    lines = ["[oG-Memory] 相关记忆："]
    for hit in hits[:5]:  # Limit to top 5 results
        category = hit.category or "memory"
        abstract = hit.abstract or "无摘要"
        score_pct = int(hit.score * 100) if hit.score > 0 else 0
        lines.append(f"  - [{category}] {abstract} (相关度: {score_pct}%)")

    lines.append("\n如需详细信息，可使用 search_memory / read_memory 工具。")
    return "\n".join(lines)


def assemble(params: dict) -> dict:
    """Called before each model run. Return messages + token estimate."""
    _dump_params("assemble", params)

    session_id = params.get("sessionId", "unknown")
    messages = params.get("messages", [])
    token_budget = params.get("tokenBudget", 128_000)

    _log("info", f"assemble  sessionId={session_id}  input_msgs={len(messages)}  budget={token_budget}")

    # Extract query from recent user messages
    query = _extract_query(messages)

    if not query:
        _log("info", "assemble  no user messages found, skipping memory search")
        return {
            "messages": messages,
            "estimatedTokens": len(messages) * 200,
            "systemPromptAddition": "[oG-Memory] No context available yet.",
        }

    # Search memory with graceful degradation
    system_prompt_addition = ""
    try:
        ctx = _build_request_context(session_id)
        read_api = _get_read_api()

        result = read_api.search_memory(
            query=query,
            ctx=ctx,
            top_k=5,
            mode="QUICK",
        )

        hits = result.hits if result else []
        system_prompt_addition = _format_memory_addition(hits)

        _log("info", f"assemble  memory search returned {len(hits)} results")

    except Exception as e:
        # Graceful degradation: search failure should not block conversation
        _log("warn", f"assemble  memory search failed, degrading gracefully: {e}")
        system_prompt_addition = (
            "[oG-Memory] Memory search temporarily unavailable. "
            "Conversation continues normally."
        )

    return {
        "messages": messages,
        "estimatedTokens": len(messages) * 200,
        "systemPromptAddition": system_prompt_addition,
    }


def compact(params: dict) -> dict:
    """Called when context window is full or user runs /compact."""
    _dump_params("compact", params)

    session_id = params.get("sessionId", "unknown")
    force = params.get("force", False)

    _log("info", f"compact  sessionId={session_id}  force={force}")
    _log("info", "compact  memory compaction not yet implemented, returning noop")

    return {"ok": True, "compacted": True, "method": "og-memory-noop"}


def bootstrap(params: dict) -> dict:
    """Initialize engine state for a session (called once per session)."""
    _dump_params("bootstrap", params)
    _log("info", f"bootstrap  sessionId={params.get('sessionId', 'unknown')}")
    return {"bootstrapped": True}


def ingest_batch(params: dict) -> dict:
    """Ingest a completed turn as a batch."""
    _dump_params("ingest_batch", params)

    messages = params.get("messages", [])
    _log("info", f"ingest_batch  count={len(messages)}")

    # Note: Batch ingest is handled by after_turn, not here
    return {"ingested": True, "count": len(messages)}


def after_turn(params: dict) -> dict:
    """
    Post-run lifecycle work.

    This is THE MAIN ENTRY POINT for writing conversation data to oG-Memory.
    Called after each agent turn completes.
    """
    _dump_params("after_turn", params)

    session_id = params.get("sessionId", "unknown")
    _log("info", f"after_turn  sessionId={session_id}")

    # Get messages from params
    messages = params.get("messages", [])

    if messages:
        _log("info", f"after_turn  committing {len(messages)} messages to oG-Memory")

        try:
            result = commit_messages_to_memory(messages, session_id)

            writes_completed = result.get("writes_completed", 0)
            _log("info", f"after_turn  successfully committed {writes_completed} writes to oG-Memory")
        except Exception as e:
            _log("error", f"after_turn  failed to commit to oG-Memory: {e}")
            # Don't fail the turn - oG-Memory errors should be non-blocking
    else:
        _log("warn", "after_turn  no messages in params, skipping oG-Memory commit")

    return {"ok": True}


def prepare_subagent_spawn(params: dict) -> dict:
    """Set up shared state for a child session (future use)."""
    _dump_params("prepare_subagent_spawn", params)
    parent_session = params.get('parentSessionId', '?')
    _log("info", f"prepare_subagent_spawn  parentSession={parent_session}")
    _log("info", "prepare_subagent_spawn  not yet implemented, returning noop")
    return {"prepared": True}


def on_subagent_ended(params: dict) -> dict:
    """Clean up after a subagent ends."""
    _dump_params("on_subagent_ended", params)

    child_session = params.get("childSessionId", "unknown")
    _log("info", f"on_subagent_ended  childSession={child_session}")
    _log("info", "on_subagent_ended  not yet implemented, returning noop")

    return {"cleaned": True}


def dispose(params: dict) -> dict:
    """Release resources during gateway shutdown."""
    _dump_params("dispose", params)
    _log("info", "dispose  releasing resources")

    # Cleanup write API instance
    if hasattr(_get_write_api, "_instance"):
        _get_write_api._instance = None

    # Cleanup read API instance
    if hasattr(_get_read_api, "_instance"):
        _get_read_api._instance = None

    # Cleanup LLM instance
    if hasattr(_get_llm, "_instance"):
        _get_llm._instance = None

    return {"disposed": True}


# ============================================================================
# Health Check
# ============================================================================

def memory_health() -> dict:
    """Check if the memory backend is reachable."""
    _log("info", "memory_health")

    try:
        # Test AGFS connection
        client = AGFSClient(api_base_url=AGFS_BASE_URL)
        client.ls("/")

        # Check LLM availability
        llm = _get_llm()

        return {
            "status": "ok",
            "backend": "og-memory",
            "agfs_url": AGFS_BASE_URL,
            "llm": llm.model if hasattr(llm, 'model') else "unknown",
        }
    except Exception as e:
        return {
            "status": "error",
            "backend": "og-memory",
            "error": str(e),
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

METHODS = {
    "ingest": ingest,
    "assemble": assemble,
    "compact": compact,
    "bootstrap": bootstrap,
    "ingest_batch": ingest_batch,
    "after_turn": after_turn,
    "prepare_subagent_spawn": prepare_subagent_spawn,
    "on_subagent_ended": on_subagent_ended,
    "dispose": dispose,
    "memory_health": lambda params: memory_health(),
}


def main() -> None:
    """
    CLI entry point called by index.js.

    Usage:
        python3 bridge/memory_api.py <method> '<json_params>'

    All stdout before the last line is treated as debug logging.
    The last line must be valid JSON (the return value).
    """
    if len(sys.argv) < 3:
        print("Usage: memory_api.py <method> '<json_params>'", file=sys.stderr)
        sys.exit(1)

    method_name = sys.argv[1]
    raw_params = sys.argv[2]

    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        _log("error", f"Invalid JSON params: {exc}")
        _respond({"error": str(exc)})
        sys.exit(1)

    handler = METHODS.get(method_name)
    if handler is None:
        _log("error", f"Unknown method: {method_name}")
        _respond({"error": f"unknown method: {method_name}"})
        sys.exit(1)

    try:
        result = handler(params)
        _respond(result)
    except Exception as e:
        _log("error", f"Handler {method_name} failed: {e}")
        _respond({"error": str(e), "method": method_name})
        sys.exit(1)


if __name__ == "__main__":
    main()
