"""Lightweight structured trace for the retrieval pipeline.

The trace is the only instrumentation surface that lives inside the
production container. Performance analysis (cost attribution, rate-card
pricing, latency reporting) is performed offline by the ``perf/`` package,
which consumes trace events emitted via the optional sink interface below.

Configuration is purely env-var driven so the docker image has no hard
dependency on ``perf/``:

    OGMEM_TRACE_SINK=noop                 # default
    OGMEM_TRACE_SINK=jsonl:/tmp/trace.jsonl
    OGMEM_TRACE_SINK=http://collector/api/v1/trace/events
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class StageTrace:
    stage: str
    elapsed_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


@dataclass
class RetrievalTrace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stages: list[StageTrace] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: StageTrace) -> None:
        self.stages.append(stage)
        self.warnings.extend(stage.warnings)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stages": [s.to_dict() for s in self.stages],
            "total_ms": round(self.total_ms, 2),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }

    def emit(self, sink: "TraceSink | None" = None) -> None:
        """Forward this trace to the configured sink (best-effort, never raises)."""
        target = sink or get_default_sink()
        try:
            target.write(self.to_dict())
        except Exception as exc:
            logger.debug("trace sink emit failed: %s", exc)


class TraceSink(Protocol):
    def write(self, event: dict) -> None: ...


class NoopSink:
    def write(self, event: dict) -> None:
        return None


class JsonlSink:
    def __init__(self, path: str) -> None:
        from pathlib import Path

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        with self._path.open("a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")


class HttpSink:
    def __init__(self, url: str, timeout: float = 2.0) -> None:
        self._url = url
        self._timeout = timeout

    def write(self, event: dict) -> None:
        import urllib.request

        body = json.dumps(event, default=str, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._timeout).close()


_DEFAULT_SINK: TraceSink | None = None


def build_sink(spec: str | None) -> TraceSink:
    """Build a sink from an ``OGMEM_TRACE_SINK`` style spec string."""
    if not spec or spec == "noop":
        return NoopSink()
    if spec.startswith("jsonl:"):
        return JsonlSink(spec[len("jsonl:"):])
    if spec.startswith("http://") or spec.startswith("https://"):
        return HttpSink(spec)
    logger.warning("unrecognised OGMEM_TRACE_SINK=%r — using noop", spec)
    return NoopSink()


def get_default_sink() -> TraceSink:
    global _DEFAULT_SINK
    if _DEFAULT_SINK is None:
        _DEFAULT_SINK = build_sink(os.environ.get("OGMEM_TRACE_SINK"))
    return _DEFAULT_SINK


def reset_default_sink() -> None:
    """Drop the cached sink (used by tests after mutating the env var)."""
    global _DEFAULT_SINK
    _DEFAULT_SINK = None


class TraceTimer:
    def __init__(self, stage_name: str) -> None:
        self.trace = StageTrace(stage=stage_name)
        self._t0 = 0.0

    def __enter__(self) -> StageTrace:
        self._t0 = time.monotonic()
        return self.trace

    def __exit__(self, *_exc: object) -> None:
        self.trace.elapsed_ms = (time.monotonic() - self._t0) * 1000
