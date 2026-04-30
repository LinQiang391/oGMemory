"""Bounded in-process cache backends for embedder / LLM providers.

VECTORIZATION_PLAN §0 specifies the following cache-key discipline for every
provider-level cache:

    key = sha256(normalize_text(text))[:32] + ":" + model_name + ":" + dim

and §10 calls for a ``CacheBackend`` Protocol so an optional Redis / disk
backend can plug in later without rewriting the providers.

This module supplies:

* ``CacheBackend`` — the Protocol (``get`` / ``put`` / ``invalidate`` / ``stats``).
* ``LRUCacheBackend`` — the default size-bounded LRU implementation.
* ``make_embed_key`` / ``make_llm_key`` — helpers that construct canonical keys
  so every provider agrees on the exact input → key function. Providers MUST
  use these helpers and MUST NOT hash raw text / prompts directly.

The backend holds ``list[float]`` for embeddings or ``dict`` for LLM JSON; no
policy about what is cached is embedded here — that belongs in the providers.

Multi-tenant safety (CLAUDE.md §8): embedding cache keys are
account-agnostic (a raw string has the same vector regardless of tenant).
Query-level and search-result caches live elsewhere and are account-scoped by
construction. Do NOT put account-scoped keys through this backend without
including the account_id in the key string.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from core.models import normalize_text

__all__ = [
    "CacheBackend",
    "LRUCacheBackend",
    "make_embed_key",
    "make_llm_key",
    "CacheStats",
]


class CacheStats(dict):
    """Plain dict subclass so ``TokenTracker`` can merge stats easily."""

    @property
    def hit_rate(self) -> float:
        hits = self.get("hits", 0)
        misses = self.get("misses", 0)
        total = hits + misses
        return hits / total if total else 0.0


@runtime_checkable
class CacheBackend(Protocol):
    """Generic bounded cache used by embedder and LLM providers.

    Implementations MUST be thread-safe — providers call them from both
    request threads and the outbox worker thread.
    """

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if not present.

        Implementations MUST NOT raise for a missing key; a cache miss is a
        normal control-flow signal, not an error (this is the single
        documented exception to CLAUDE.md §"禁止 Fallback 路径": a cache
        miss is not a fallback path, it is the cache's defined absence
        signal — the caller then performs the real work).
        """
        ...

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``. Must be idempotent."""
        ...

    def invalidate(self, key: str) -> None:
        """Remove a key. No-op if absent."""
        ...

    def clear(self) -> None:
        """Drop all entries."""
        ...

    def stats(self) -> CacheStats:
        """Return a snapshot dict ``{"size", "hits", "misses", "evictions"}``."""
        ...


class LRUCacheBackend(CacheBackend):
    """Simple thread-safe LRU implementation built on ``OrderedDict``.

    We do NOT pull in ``cachetools`` so this module keeps the providers
    package free of optional dependencies. ``OrderedDict.move_to_end`` gives
    O(1) LRU without external libraries.

    Instantiated with ``max_size`` entries (value-type-agnostic; size is a
    raw entry count, not a byte count). Eviction is strict LRU.
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            raise ValueError(
                f"LRUCacheBackend max_size must be positive, got {max_size}"
            )
        self._max_size = max_size
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            if len(self._store) > self._max_size:
                # popitem(last=False) pops the *least recently used* entry.
                self._store.popitem(last=False)
                self._evictions += 1

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                size=len(self._store),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                max_size=self._max_size,
            )

    # Convenience accessors used by tests and older code paths ---------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Canonical cache-key builders
# ---------------------------------------------------------------------------


def make_embed_key(text: str, *, model: str, dimension: int) -> str:
    """Build the canonical embedder cache key for ``text``.

    ``model`` SHOULD include a provider prefix (e.g. ``"openai/text-embedding-3-small"``
    or ``"volcengine/doubao-embedding-vision-250615"``) so different providers
    that happen to use the same local model name cannot collide.

    ``dimension`` is included explicitly because dimension changes at the
    same ``model`` name (rare but possible during model upgrades) would
    otherwise silently return vectors the vector index will reject.
    """
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:32]
    return f"{digest}:{model}:{dimension}"


def make_llm_key(
    prompt: str,
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build the canonical LLM cache key.

    ``extra`` may include ``{"schema": ..., "tools": ..., "tool_choice": ...,
    "temperature": ..., "max_tokens": ...}``. It is serialized with
    ``sort_keys=True`` so the key is stable regardless of dict ordering.

    Cache keys MUST capture every parameter that affects the response —
    omitting one (e.g. ``temperature``) yields correctness bugs (same prompt
    + schema but different temperature returning a cached answer from the
    other temperature). Callers are expected to pass every such parameter in
    ``extra``.
    """
    payload = {
        "prompt": normalize_text(prompt),
        "model": model,
    }
    if extra:
        payload["extra"] = extra
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()