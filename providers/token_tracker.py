"""Thread-safe token usage tracker for LLM and Embedding providers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class TokenUsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embed_tokens

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

    def record_embed(self, total_tokens: int) -> None:
        with self._lock:
            self._embed_tokens += total_tokens
            self._embed_calls += 1

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
