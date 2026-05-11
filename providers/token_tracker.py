"""Thread-safe token usage tracker for LLM and Embedding providers.

Providers SHOULD feed the raw ``response.usage`` block returned by their
SDK directly into ``record_llm_from_usage`` / ``record_embed_from_usage``
instead of estimating from prompt length. Both OpenAI-shape
(``prompt_tokens`` / ``completion_tokens``) and Anthropic-shape
(``input_tokens`` / ``output_tokens``) responses are accepted.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    # Local (in-process) cache accounting — counts requests that were served
    # entirely from the provider's ``CacheBackend`` without hitting the
    # external API. "saved" approximates the tokens we did NOT spend by
    # assuming a cache hit replaces an average miss; providers record this
    # via ``record_local_cache_hit(tokens_saved=...)``.
    local_cache_hits: int = 0
    local_cache_misses: int = 0
    local_cache_saved_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embed_tokens

    @property
    def local_cache_hit_rate(self) -> float:
        total = self.local_cache_hits + self.local_cache_misses
        return (self.local_cache_hits / total) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "llm": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read": self.cache_read,
                "cache_write": self.cache_write,
                "total_tokens": self.input_tokens + self.output_tokens,
                "calls": self.llm_calls,
            },
            "embedding": {
                "total_tokens": self.embed_tokens,
                "calls": self.embed_calls,
            },
            "local_cache": {
                "hits": self.local_cache_hits,
                "misses": self.local_cache_misses,
                "saved_tokens": self.local_cache_saved_tokens,
                "hit_rate": round(self.local_cache_hit_rate * 100, 2),
            },
            "total_tokens": self.total_tokens,
        }


class TokenTracker:
    """Accumulates token usage from API responses.

    Call ``record_llm`` / ``record_embed`` from provider implementations,
    and ``snapshot_and_reset`` from the service layer to get per-request stats.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._llm_calls = 0
        self._embed_tokens = 0
        self._embed_calls = 0
        self._local_cache_hits = 0
        self._local_cache_misses = 0
        self._local_cache_saved_tokens = 0

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        with self._lock:
            self._input += input_tokens
            self._output += output_tokens
            self._cache_read += cache_read
            self._cache_write += cache_write
            self._llm_calls += 1

    def record_llm_from_usage(self, usage: Any) -> bool:
        """Record an LLM call from a SDK usage object/dict.

        Returns True if a usage block was successfully parsed and recorded;
        callers can use the return value to detect mocks/responses that did
        not surface usage at all.
        """
        parsed = _parse_llm_usage(usage)
        if parsed is None:
            return False
        input_tokens, output_tokens, cache_read, cache_write = parsed
        self.record_llm(input_tokens, output_tokens, cache_read, cache_write)
        return True

    def record_embed(self, total_tokens: int) -> None:
        with self._lock:
            self._embed_tokens += total_tokens
            self._embed_calls += 1

    def record_embed_from_usage(self, usage: Any) -> bool:
        """Record an embedding call from a SDK usage object/dict."""
        total = _parse_embed_usage(usage)
        if total is None:
            return False
        self.record_embed(total)
        return True

    def record_local_cache_hit(self, tokens_saved: int = 0) -> None:
        """Record that a provider served a request from its in-process cache.

        ``tokens_saved`` is a best-effort estimate of the tokens we did NOT
        spend (providers count the cached payload length). Counters are
        separate from the "real" usage counters so dashboards can compute a
        true hit-rate without double-counting.
        """
        with self._lock:
            self._local_cache_hits += 1
            if tokens_saved > 0:
                self._local_cache_saved_tokens += tokens_saved

    def record_local_cache_miss(self) -> None:
        with self._lock:
            self._local_cache_misses += 1

    def snapshot(self) -> TokenUsageSnapshot:
        with self._lock:
            return TokenUsageSnapshot(
                input_tokens=self._input,
                output_tokens=self._output,
                cache_read=self._cache_read,
                cache_write=self._cache_write,
                llm_calls=self._llm_calls,
                embed_tokens=self._embed_tokens,
                embed_calls=self._embed_calls,
                local_cache_hits=self._local_cache_hits,
                local_cache_misses=self._local_cache_misses,
                local_cache_saved_tokens=self._local_cache_saved_tokens,
            )

    def snapshot_and_reset(self) -> TokenUsageSnapshot:
        with self._lock:
            snap = TokenUsageSnapshot(
                input_tokens=self._input,
                output_tokens=self._output,
                cache_read=self._cache_read,
                cache_write=self._cache_write,
                llm_calls=self._llm_calls,
                embed_tokens=self._embed_tokens,
                embed_calls=self._embed_calls,
                local_cache_hits=self._local_cache_hits,
                local_cache_misses=self._local_cache_misses,
                local_cache_saved_tokens=self._local_cache_saved_tokens,
            )
            self._reset_unlocked()
            return snap

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._llm_calls = 0
        self._embed_tokens = 0
        self._embed_calls = 0
        self._local_cache_hits = 0
        self._local_cache_misses = 0
        self._local_cache_saved_tokens = 0


def _get(usage: Any, *keys: str) -> Any:
    """Read a key from either a dict-shaped or attribute-shaped usage block."""
    for key in keys:
        if isinstance(usage, dict):
            if key in usage and usage[key] is not None:
                return usage[key]
        else:
            val = getattr(usage, key, None)
            if val is not None:
                return val
    return None


def _parse_llm_usage(usage: Any) -> tuple[int, int, int, int] | None:
    """Normalise an LLM usage block to (input, output, cache_read, cache_write).

    Accepts both OpenAI-shape (``prompt_tokens``/``completion_tokens``) and
    Anthropic-shape (``input_tokens``/``output_tokens``) responses, plus the
    nested ``prompt_tokens_details.cached_tokens`` cache field surfaced by
    several providers.
    """
    if usage is None:
        return None
    input_tokens = _get(usage, "prompt_tokens", "input_tokens") or 0
    output_tokens = _get(usage, "completion_tokens", "output_tokens") or 0
    cache_read = 0
    cache_write = 0
    details = _get(usage, "prompt_tokens_details")
    if details is not None:
        cache_read = _get(details, "cached_tokens") or 0
    cache_write = _get(usage, "cache_creation_input_tokens") or 0
    try:
        return int(input_tokens), int(output_tokens), int(cache_read), int(cache_write)
    except (TypeError, ValueError):
        return None


def _parse_embed_usage(usage: Any) -> int | None:
    """Normalise an embedding usage block to a total token count."""
    if usage is None:
        return None
    total = _get(usage, "total_tokens", "prompt_tokens", "input_tokens")
    if total is None:
        return None
    try:
        return int(total)
    except (TypeError, ValueError):
        return None