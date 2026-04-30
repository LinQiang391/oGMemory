"""Lifecycle performance recorder for ContextEngine.

Captures per-stage wall time, token usage, model names, and $-cost across
the six lifecycle stages defined in README.md (① message_received → ②
bootstrap+ingest+assemble → ③ before_tool_call+tool_result_persist → ④
afterTurn → ⑤ before_compaction+compact → ⑥ session_end+dispose).

Entry points:
    from perf import get_recorder, record_stage, span, is_enabled

    @record_stage("after_turn")
    def after_turn(self, params): ...

    with span("extract_llm", num_messages=len(msgs)) as s:
        candidates = extractor.extract(...)
        s.meta["num_candidates"] = len(candidates)

The recorder is opt-in. It is a no-op (zero overhead) unless
``OGMEM_PERF_ENABLED=1`` is set in the environment OR a driver explicitly
calls ``get_recorder().enable(...)``.
"""

from __future__ import annotations

from perf.recorder import Recorder, SpanEvent, get_recorder, is_enabled
from perf.decorators import record_stage, span

__all__ = [
    "Recorder",
    "SpanEvent",
    "get_recorder",
    "is_enabled",
    "record_stage",
    "span",
]
