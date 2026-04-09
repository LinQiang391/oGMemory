"""
bridge/memory_api.py — OpenClaw Plugin Bridge to oG-Memory

CLI entry point called by index.js (execFileSync).
Delegates all business logic to server.memory_service.MemoryService.

Architecture (all-in-one mode):
    index.js (Node.js) → execFileSync → bridge/memory_api.py → MemoryService → oG-Memory internals

Architecture (standalone mode):
    index.js (Node.js) → HTTP fetch → server/app.py → MemoryService → oG-Memory internals
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Python path setup (needed when called as subprocess by index.js)
# ---------------------------------------------------------------------------
_file_dir = Path(__file__).parent.parent  # openclaw_context_engine_plugin/
_project_root = _file_dir.parent          # oG-Memory/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ['no_proxy'] = "localhost,127.0.0.1"

from server.memory_service import MemoryService

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_service: MemoryService | None = None


def _get_service() -> MemoryService:
    global _service
    if _service is None:
        _service = MemoryService()
    return _service


# ---------------------------------------------------------------------------
# Logging / CLI helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _log(level: str, msg: str) -> None:
    print(f"[memory-api {_ts()}] [{level.upper()}] {msg}", file=sys.stderr, flush=True)


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

        # Try to use real embedder, fallback to mock
        config = ProviderConfig.from_env()
        if config.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                from providers.embedder import OpenAIEmbedder
                embedder = OpenAIEmbedder(
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                    model=config.openai_embedding_model,
                )
                dimension = 1536  # text-embedding-ada-002
                _log("info", f"Using OpenAIEmbedder: {config.openai_embedding_model}")
            except Exception as e:
                _log("warn", f"Failed to init OpenAIEmbedder, using mock: {e}")
                embedder = MockEmbedder()
                dimension = 384
        else:
            embedder = MockEmbedder()
            dimension = 384
            _log("info", "Using MockEmbedder (set OPENAI_API_KEY for real embeddings)")

        vector_index = InMemoryVectorIndex(dimension=dimension)

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
        _log("info", f"MemoryReadAPI initialized (embedder: {embedder.__class__.__name__}, dim: {dimension})")

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
    _log("info", f"ingest  sessionId={params.get('sessionId', '?')}")
    return _get_service().ingest(params)


def assemble(params: dict) -> dict:
    session_id = params.get("sessionId", "unknown")
    _log("info", f"assemble  sessionId={session_id}  msgs={len(params.get('messages', []))}")
    result = _get_service().assemble(params)
    hits_count = result.get("systemPromptAddition", "").count("相关度:")
    _log("info", f"assemble  memory search returned {hits_count} results")
    return result


def compact(params: dict) -> dict:
    _log("info", f"compact  sessionId={params.get('sessionId', '?')}")
    return _get_service().compact(params)


def bootstrap(params: dict) -> dict:
    _log("info", f"bootstrap  sessionId={params.get('sessionId', '?')}")
    return _get_service().bootstrap(params)


def ingest_batch(params: dict) -> dict:
    _log("info", f"ingest_batch  count={len(params.get('messages', []))}")
    return _get_service().ingest_batch(params)


def after_turn(params: dict) -> dict:
    session_id = params.get("sessionId", "unknown")
    messages = params.get("messages", [])
    _log("info", f"after_turn  sessionId={session_id}  msgs={len(messages)}")
    result = _get_service().after_turn(params)
    _log("info", f"after_turn  completed={result.get('writes_completed', 0)}")
    return result


def prepare_subagent_spawn(params: dict) -> dict:
    _log("info", f"prepare_subagent_spawn  parent={params.get('parentSessionId', '?')}")
    return _get_service().prepare_subagent_spawn(params)


def on_subagent_ended(params: dict) -> dict:
    _log("info", f"on_subagent_ended  child={params.get('childSessionId', '?')}")
    return _get_service().on_subagent_ended(params)


def dispose(params: dict) -> dict:
    _log("info", "dispose  releasing resources")
    return _get_service().dispose(params)


def memory_health() -> dict:
    return _get_service().health()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

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
    """CLI entry point called by index.js via execFileSync."""
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
        import traceback
        traceback.print_exc(file=sys.stderr)
        _respond({"error": str(e), "method": method_name})
        sys.exit(1)


if __name__ == "__main__":
    main()
