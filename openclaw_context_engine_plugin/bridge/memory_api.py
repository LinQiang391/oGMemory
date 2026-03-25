"""
bridge/memory_api.py — Mock Memory API for the context engine plugin.

In production, these functions would call the actual memory backend
(vector DB, SQLite, Redis, OpenClaw memory plugin, etc.).

In this mock, every function prints its arguments and returns a
plausible stub response.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

MEMORY_API_BASE_URL = os.environ.get("MOCK_MEMORY_API_BASE_URL", "http://localhost:9200")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _log(func_name: str, **kwargs: Any) -> None:
    print(
        f"[memory-api {_ts()}] {func_name}() "
        f"params={json.dumps(kwargs, indent=2, default=str)}",
        file=sys.stderr,
        flush=True,
    )


# ── CRUD ─────────────────────────────────────────────────────────────

def memory_store(key: str, value: Any, *, namespace: str = "default", ttl: int | None = None) -> dict:
    """Store a key-value pair in the memory backend."""
    _log("memory_store", key=key, value=value, namespace=namespace, ttl=ttl)
    return {"stored": True, "key": key, "namespace": namespace}


def memory_get(key: str, *, namespace: str = "default") -> dict | None:
    """Retrieve a value by key."""
    _log("memory_get", key=key, namespace=namespace)
    # Mock: always return None (key not found)
    return None


def memory_delete(key: str, *, namespace: str = "default") -> dict:
    """Delete a key (supports trailing wildcard *)."""
    _log("memory_delete", key=key, namespace=namespace)
    return {"deleted": True, "key": key}


def memory_list(prefix: str = "", *, namespace: str = "default", limit: int = 100) -> list[str]:
    """List keys matching a prefix."""
    _log("memory_list", prefix=prefix, namespace=namespace, limit=limit)
    return []


# ── Search / Retrieval ───────────────────────────────────────────────

def memory_search(
    query: str,
    *,
    top_k: int = 5,
    namespace: str = "default",
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Semantic search over stored memories.
    Returns a list of {key, value, score} dicts.
    """
    _log(
        "memory_search",
        query=query,
        top_k=top_k,
        namespace=namespace,
        score_threshold=score_threshold,
    )
    # Mock: return empty results
    return []


# ── Batch operations ─────────────────────────────────────────────────

def memory_store_batch(items: list[dict], *, namespace: str = "default") -> dict:
    """
    Store multiple key-value pairs at once.
    Each item: {"key": str, "value": Any}
    """
    _log("memory_store_batch", count=len(items), namespace=namespace)
    return {"stored": True, "count": len(items)}


def memory_clear(namespace: str = "default") -> dict:
    """Drop all keys in a namespace."""
    _log("memory_clear", namespace=namespace)
    return {"cleared": True, "namespace": namespace}


# ── Embedding (optional) ─────────────────────────────────────────────

def memory_embed(text: str, *, model: str = "default") -> list[float]:
    """
    Return an embedding vector for the given text.
    Mock: return a zero vector of dimension 8.
    """
    _log("memory_embed", text=text[:80], model=model)
    return [0.0] * 8


# ── Health ───────────────────────────────────────────────────────────

def memory_health() -> dict:
    """Check if the memory backend is reachable."""
    _log("memory_health")
    return {
        "status": "ok",
        "backend": "mock",
        "base_url": MEMORY_API_BASE_URL,
    }


# ── ContextEngine Integration ───────────────────────────────────────────

def commit_session(
    messages: list[dict],
    *,
    session_id: str = "",
    account_id: str = "default",
    user_id: str = "default",
    agent_id: str = "default",
) -> dict:
    """Call oG-Memory write pipeline to extract and store memories.

    Args:
        messages: Conversation messages [{"role": "user|assistant", "content": "..."}]
        session_id: Session identifier
        account_id: Account ID for multi-tenant isolation
        user_id: User ID
        agent_id: Agent ID

    Returns:
        {"ok": bool, "candidates_extracted": int, "writes_completed": int}
    """
    import os
    import sys

    _log("commit_session", message_count=len(messages), session_id=session_id,
         account_id=account_id, user_id=user_id, agent_id=agent_id)

    project_root = os.environ.get("OG_MEMORY_ROOT", "")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from core.models import RequestContext
        from fs.agfs_adapter import AGFSContextFS
        from pyagfs import AGFSClient
        from service.api import MemoryWriteAPI
        from providers.config import ProviderConfig
        import uuid

        agfs_base_url = os.environ.get("AGFS_BASE_URL", "http://localhost:8080")
        client = AGFSClient(api_base_url=agfs_base_url)
        fs = AGFSContextFS(client, mount_prefix="/local")

        config = ProviderConfig.from_env()
        llm = config.create_llm()
        api = MemoryWriteAPI(fs=fs, llm=llm)

        ctx = RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id or str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        result = api.commit_session(messages, ctx)
        return {"ok": True, **result}

    except Exception as e:
        _log("commit_session_error", error=str(e))
        return {"ok": False, "error": str(e)}


def search_memory(
    query: str,
    *,
    top_k: int = 5,
    account_id: str = "default",
    user_id: str = "default",
    agent_id: str = "default",
) -> list[dict]:
    """Call oG-Memory retrieval pipeline to search stored memories.

    Args:
        query: Search query text
        top_k: Maximum number of results
        account_id: Account ID for multi-tenant isolation
        user_id: User ID
        agent_id: Agent ID

    Returns:
        [{"uri": str, "abstract": str, "score": float}, ...]
    """
    import os
    import sys

    _log("search_memory", query=query[:80], top_k=top_k,
         account_id=account_id, user_id=user_id, agent_id=agent_id)

    project_root = os.environ.get("OG_MEMORY_ROOT", "")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from core.models import RequestContext
        from fs.agfs_adapter import AGFSContextFS
        from pyagfs import AGFSClient
        from providers.config import ProviderConfig
        from retrieval.pipeline import RetrievalPipeline
        from retrieval.query_planner import QueryPlanner
        from retrieval.seed_retriever import SeedRetriever
        from retrieval.hierarchical_searcher import HierarchicalSearcher
        from retrieval.assembly_service import AssemblyService
        from retrieval.memory_read_service import MemoryReadService
        from service.api import MemoryReadAPI
        import uuid

        agfs_base_url = os.environ.get("AGFS_BASE_URL", "http://localhost:8080")
        client = AGFSClient(api_base_url=agfs_base_url)
        fs = AGFSContextFS(client, mount_prefix="/local")

        config = ProviderConfig.from_env()
        llm = config.create_llm()
        embedder = config.create_embedder()
        vector_index = config.create_vector_index()

        ctx = RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        pipeline = RetrievalPipeline(
            planner=QueryPlanner(llm=llm),
            seed_retriever=SeedRetriever(vector_index=vector_index, embedder=embedder),
            hierarchical_searcher=HierarchicalSearcher(vector_index=vector_index, embedder=embedder),
            assembly=AssemblyService(),
        )
        read_api = MemoryReadAPI(
            pipeline=pipeline,
            read_service=MemoryReadService(fs=fs),
        )

        result = read_api.search_memory(query, ctx, top_k=top_k)
        return [
            {"uri": h.uri, "abstract": h.abstract, "score": h.score}
            for h in result.hits
        ]

    except Exception as e:
        _log("search_memory_error", error=str(e))
        return []
