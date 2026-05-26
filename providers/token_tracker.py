"""Thread-safe token usage tracker for LLM and Embedding providers.

Providers SHOULD feed the raw ``response.usage`` block returned by their
SDK directly into ``record_llm`` / ``record_embed`` from provider implementations.

Per-context buckets
-------------------
Beyond the process-level cumulative counters, this module maintains a
``ContextVar``-based **stack** of :class:`TokenBucket` objects. Every
``record_*`` call simultaneously updates the global tracker AND every
bucket on the active context's stack. Each thread / asyncio task / perf
span can push its own bucket and read isolated, accurate per-context
totals — eliminating the cross-thread pollution that snapshot-diff has
under concurrency.

Use :func:`push_bucket` to start a new attribution scope and
:func:`pop_bucket` (with the returned token) to end it. Buckets nest
naturally: when nested, every ``record_*`` is added to the inner bucket
*and* all enclosing buckets, so parent spans still see the full total.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class TokenUsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    tool_stats: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embed_tokens

    def to_dict(self) -> dict:
        data = {
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
        if self.tool_stats:
            data["tools"] = self.tool_stats
        return data


# ---------------------------------------------------------------------------
# Tool/session tracking helpers
# ---------------------------------------------------------------------------


def _empty_tool_counters() -> dict:
    return {
        "call_count": 0,
        "success_count": 0,
        "fail_count": 0,
        "total_duration_ms": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "llm_call_ids": [],
    }


def _empty_session_counters() -> dict:
    return {
        "llm_calls": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_tokens": 0,
    }


def _copy_tool_stats(stats: dict) -> dict:
    copied: dict = {}
    for group_name, group in stats.items():
        if not isinstance(group, dict):
            continue
        copied[group_name] = {}
        for key, value in group.items():
            if isinstance(value, dict):
                copied[group_name][key] = {
                    inner_key: list(inner_value) if isinstance(inner_value, list) else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                copied[group_name][key] = value
    return copied


# ---------------------------------------------------------------------------
# Per-context bucket — used by perf module to attribute tokens to a span
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Per-context accumulator written to by every active provider call.

    A bucket is pushed onto :data:`_active_buckets` (a ``ContextVar``
    stack) at the start of a perf span and popped at the end. Because
    the stack is a ``ContextVar``, each thread / asyncio task gets its
    own independent stack — concurrent extractions in background
    threads will each see only their own LLM/embed calls.

    Buckets are mutable but guarded by a per-bucket lock; updates from
    the network thread that issued an LLM request and from a
    surrounding perf span run on the same thread, but we keep the lock
    cheap so the design also supports providers that fan out work to
    background threads.
    """

    llm_input: int = 0
    llm_output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    llm_model: str | None = None
    embed_model: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_write: int,
        model: str | None,
    ) -> None:
        with self._lock:
            self.llm_input += int(input_tokens or 0)
            self.llm_output += int(output_tokens or 0)
            self.cache_read += int(cache_read or 0)
            self.cache_write += int(cache_write or 0)
            self.llm_calls += 1
            if model and not self.llm_model:
                self.llm_model = model

    def add_embed(self, total_tokens: int, model: str | None) -> None:
        with self._lock:
            self.embed_tokens += int(total_tokens or 0)
            self.embed_calls += 1
            if model and not self.embed_model:
                self.embed_model = model

    def to_attribution_dict(self) -> dict[str, Any]:
        """Render in the shape used by perf SpanEvent.tokens."""
        with self._lock:
            return {
                "llm": {
                    "input_tokens": self.llm_input,
                    "output_tokens": self.llm_output,
                    "cache_read": self.cache_read,
                    "cache_write": self.cache_write,
                    "llm_calls": self.llm_calls,
                    "embed_tokens": 0,
                    "embed_calls": 0,
                },
                "embed": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "llm_calls": 0,
                    "embed_tokens": self.embed_tokens,
                    "embed_calls": self.embed_calls,
                },
                "llm_model": self.llm_model,
                "embed_model": self.embed_model,
            }


# Module-level stack of active buckets.
_active_buckets: contextvars.ContextVar[tuple[TokenBucket, ...]] = contextvars.ContextVar(
    "ogmem_token_buckets", default=()
)


def push_bucket() -> tuple[TokenBucket, contextvars.Token]:
    """Push a fresh bucket onto the current context's stack."""
    bucket = TokenBucket()
    current = _active_buckets.get()
    token = _active_buckets.set(current + (bucket,))
    return bucket, token


def pop_bucket(token: contextvars.Token) -> None:
    """Restore the bucket stack to its state before the matching ``push_bucket``."""
    _active_buckets.reset(token)


def current_buckets() -> tuple[TokenBucket, ...]:
    """Return the active bucket stack for the current context (may be empty)."""
    return _active_buckets.get()


# ---------------------------------------------------------------------------
# UsageTracker — tracks tool calls and session-level usage
# ---------------------------------------------------------------------------


class UsageTracker:
    """Thread-safe provider and tool usage tracker.

    This is the unified tracker used by both provider-level token accounting
    and session-level tool usage accounting. ``TokenTracker`` remains as a
    backward-compatible name for provider code.
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
        self._tool_by_name: dict[str, dict] = {}
        self._tool_by_category: dict[str, dict] = {}
        self._usage_by_session: dict[str, dict] = {}

    @property
    def tool_stats(self) -> dict[str, dict]:
        """Per-tool aggregate used by extraction prompts."""
        with self._lock:
            return _copy_tool_stats({"by_tool": self._tool_by_name}).get("by_tool", {})

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
        *,
        session_id: str | None = None,
        call_id: str | None = None,
    ) -> str:
        call_id = call_id or f"llm_{uuid4().hex[:12]}"
        with self._lock:
            self._input += input_tokens
            self._output += output_tokens
            self._cache_read += cache_read
            self._cache_write += cache_write
            self._llm_calls += 1
            if session_id:
                session_stats = self._usage_by_session.setdefault(session_id, _empty_session_counters())
                session_stats["llm_calls"] += 1
                session_stats["input_tokens"] += input_tokens
                session_stats["output_tokens"] += output_tokens
        return call_id

    def record_embed(self, total_tokens: int) -> None:
        with self._lock:
            self._embed_tokens += total_tokens
            self._embed_calls += 1

    def record_tool_call(
        self,
        *,
        tool_name: str,
        category: str = "",
        session_id: str = "",
        status: str = "",
        duration_ms: float | None = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        llm_call_id: str | None = None,
    ) -> None:
        if not tool_name:
            return

        success = status in {"completed", "success"}
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        with self._lock:
            self._add_tool_call(
                self._tool_by_name.setdefault(tool_name, _empty_tool_counters()),
                success=success,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                llm_call_id=llm_call_id,
            )
            if category:
                self._add_tool_call(
                    self._tool_by_category.setdefault(category, _empty_tool_counters()),
                    success=success,
                    duration_ms=duration_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    llm_call_id=llm_call_id,
                )
            if session_id:
                session_stats = self._usage_by_session.setdefault(session_id, _empty_session_counters())
                session_stats["tool_calls"] += 1
                session_stats["tool_tokens"] += total_tokens

    def snapshot(self) -> TokenUsageSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def snapshot_and_reset(self) -> TokenUsageSnapshot:
        with self._lock:
            snap = self._snapshot_unlocked()
            self._reset_unlocked()
            return snap

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def replace_tool_stats(self, stats: dict[str, dict]) -> None:
        """Replace per-tool stats from legacy ``SessionBuffer.tool_usage_stats``."""
        with self._lock:
            self._tool_by_name = {}
            for name, raw in (stats or {}).items():
                counters = _empty_tool_counters()
                counters.update({
                    "call_count": int(raw.get("call_count", 0) or 0),
                    "success_count": int(raw.get("success_count", 0) or 0),
                    "fail_count": int(raw.get("fail_count", 0) or 0),
                    "total_duration_ms": float(raw.get("total_duration_ms", 0.0) or 0.0),
                    "total_prompt_tokens": int(raw.get("total_prompt_tokens", 0) or 0),
                    "total_completion_tokens": int(raw.get("total_completion_tokens", 0) or 0),
                    "total_tokens": int(raw.get("total_tokens", 0) or 0),
                    "llm_call_ids": list(raw.get("llm_call_ids") or []),
                })
                self._tool_by_name[name] = counters

    def merge_snapshot(self, snapshot: TokenUsageSnapshot) -> None:
        with self._lock:
            self._input += snapshot.input_tokens
            self._output += snapshot.output_tokens
            self._cache_read += snapshot.cache_read
            self._cache_write += snapshot.cache_write
            self._llm_calls += snapshot.llm_calls
            self._embed_tokens += snapshot.embed_tokens
            self._embed_calls += snapshot.embed_calls
            for name, stats in snapshot.tool_stats.get("by_tool", {}).items():
                self._merge_tool_counters(
                    self._tool_by_name.setdefault(name, _empty_tool_counters()),
                    stats,
                )
            for category, stats in snapshot.tool_stats.get("by_category", {}).items():
                self._merge_tool_counters(
                    self._tool_by_category.setdefault(category, _empty_tool_counters()),
                    stats,
                )
            for session_id, stats in snapshot.tool_stats.get("by_session", {}).items():
                target = self._usage_by_session.setdefault(session_id, _empty_session_counters())
                for key in target:
                    target[key] += stats.get(key, 0) or 0

    def _snapshot_unlocked(self) -> TokenUsageSnapshot:
        return TokenUsageSnapshot(
            input_tokens=self._input,
            output_tokens=self._output,
            cache_read=self._cache_read,
            cache_write=self._cache_write,
            llm_calls=self._llm_calls,
            embed_tokens=self._embed_tokens,
            embed_calls=self._embed_calls,
            tool_stats={
                "by_tool": _copy_tool_stats({"by_tool": self._tool_by_name}).get("by_tool", {}),
                "by_category": _copy_tool_stats({"by_category": self._tool_by_category}).get("by_category", {}),
                "by_session": _copy_tool_stats({"by_session": self._usage_by_session}).get("by_session", {}),
            },
        )

    def _reset_unlocked(self) -> None:
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._llm_calls = 0
        self._embed_tokens = 0
        self._embed_calls = 0
        self._tool_by_name = {}
        self._tool_by_category = {}
        self._usage_by_session = {}

    @staticmethod
    def _add_tool_call(
        counters: dict,
        *,
        success: bool,
        duration_ms: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        llm_call_id: str | None,
    ) -> None:
        counters["call_count"] += 1
        if success:
            counters["success_count"] += 1
        else:
            counters["fail_count"] += 1
        counters["total_duration_ms"] += duration_ms or 0
        counters["total_prompt_tokens"] += prompt_tokens or 0
        counters["total_completion_tokens"] += completion_tokens or 0
        counters["total_tokens"] += total_tokens
        if llm_call_id and llm_call_id not in counters["llm_call_ids"]:
            counters["llm_call_ids"].append(llm_call_id)

    @staticmethod
    def _merge_tool_counters(target: dict, source: dict) -> None:
        for key in (
            "call_count",
            "success_count",
            "fail_count",
            "total_duration_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
        ):
            target[key] += source.get(key, 0) or 0
        for call_id in source.get("llm_call_ids", []) or []:
            if call_id not in target["llm_call_ids"]:
                target["llm_call_ids"].append(call_id)


# ---------------------------------------------------------------------------
# TokenTracker — extends UsageTracker with bucket integration for perf
# ---------------------------------------------------------------------------


class TokenTracker(UsageTracker):
    """Accumulates token usage from API responses.

    Extends UsageTracker with per-context bucket support for perf span
    token attribution. Call ``record_llm`` / ``record_embed`` from provider
    implementations, and ``snapshot_and_reset`` from the service layer.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> str | None:
        """Model identifier used for per-bucket attribution (may be None)."""
        return self._model

    def set_model(self, model: str | None) -> None:
        """Late-bind a provider model name (used by providers that resolve model after init)."""
        self._model = model

    def record_llm(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> str:
        call_id = super().record_llm(input_tokens, output_tokens, cache_read, cache_write)
        for bucket in _active_buckets.get():
            bucket.add_llm(input_tokens, output_tokens, cache_read, cache_write, self._model)
        return call_id

    def record_embed(self, total_tokens: int) -> None:
        super().record_embed(total_tokens)
        for bucket in _active_buckets.get():
            bucket.add_embed(total_tokens, self._model)
