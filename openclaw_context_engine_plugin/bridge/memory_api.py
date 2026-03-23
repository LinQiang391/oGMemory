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
