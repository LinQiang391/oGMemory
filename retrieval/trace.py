"""Lightweight structured trace for the retrieval pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
_F = TypeVar("_F", bound=Callable[..., Any])

_DEFAULT_HTTP_TIMEOUT_SECONDS = 0.5
_DEFAULT_HTTP_QUEUE_SIZE = 1024
_DEFAULT_HTTP_EVENT_TTL_SECONDS = 30.0
_DEFAULT_HTTP_CIRCUIT_SECONDS = 5.0
_HTTP_FAILURE_THRESHOLD = 3
_HTTP_RETRY_ATTEMPTS = 3
_PRIVATE_TRACE_KEYS = {"account_id", "user_id", "agent_id", "trace_id"}


class _NoTraceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_trace_http_opener = urllib.request.build_opener(_NoTraceRedirect())


def trace_capable(func: _F) -> _F:
    """Mark a concrete method implementation as accepting trace injection."""
    setattr(func, "__trace_capable__", True)
    return func


def accepts_trace(method: Callable[..., Any]) -> bool:
    return bool(getattr(method, "__trace_capable__", False))


@dataclass
class StageTrace:
    stage: str
    elapsed_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    children: list[StageTrace] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def add_child(self, stage: StageTrace) -> None:
        if any(existing is stage for existing in self.children):
            return
        self.children.append(stage)
        self.warnings.extend(stage.warnings)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "metrics": dict(self.metrics),
            "children": [child.to_dict() for child in self.children],
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass
class RetrievalTrace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stages: list[StageTrace] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    level_histogram: dict[str, int] = field(default_factory=dict)

    def add_stage(self, stage: StageTrace) -> None:
        if any(existing is stage for existing in self.stages):
            return
        self.stages.append(stage)
        self.warnings.extend(stage.warnings)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stages": [s.to_dict() for s in self.stages],
            "total_ms": round(self.total_ms, 2),
            "warnings": list(self.warnings),
            "level_histogram": self.level_histogram,
        }


_active_trace_stage: ContextVar[StageTrace | None] = ContextVar(
    "active_retrieval_trace_stage", default=None
)


class TraceTimer:
    def __init__(
        self,
        stage_name: str,
        parent: StageTrace | None = None,
        *,
        activate_children: bool = True,
    ) -> None:
        self.trace = StageTrace(stage=stage_name)
        self._t0 = 0.0
        self._explicit_parent = parent
        self._activate_children = activate_children
        self._parent: StageTrace | None = None
        self._token: Token[StageTrace | None] | None = None

    def __enter__(self) -> StageTrace:
        self._t0 = time.perf_counter()
        self._parent = self._explicit_parent or _active_trace_stage.get()
        if self._activate_children:
            self._token = _active_trace_stage.set(self.trace)
        return self.trace

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        self.trace.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        if exc_type is not None:
            self.trace.error = getattr(exc_type, "__name__", str(exc_type))
        if self._token is not None:
            _active_trace_stage.reset(self._token)
        if self._parent is not None:
            self._parent.add_child(self.trace)


def _configured_trace_sink() -> str:
    configured = os.environ.get("OGMEM_TRACE_SINK")
    if configured is not None:
        return configured.strip() or "noop"

    legacy_enabled = os.environ.get("OGMEM_RETRIEVAL_TRACE", "").strip().lower()
    if legacy_enabled in {"1", "true", "yes", "on"}:
        legacy_path = os.environ.get("OGMEM_RETRIEVAL_TRACE_JSONL", "").strip()
        if legacy_path:
            return f"jsonl:{legacy_path}"
    return "noop"


def retrieval_trace_enabled() -> bool:
    return _configured_trace_sink().lower() not in {"", "noop", "none", "off"}


def _include_trace_identifiers() -> bool:
    return os.environ.get("OGMEM_TRACE_INCLUDE_IDENTIFIERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _export_trace_payload(value: Any) -> Any:
    include_identifiers = _include_trace_identifiers()
    if isinstance(value, dict):
        exported = {}
        for key, item in value.items():
            if key in _PRIVATE_TRACE_KEYS and item and not include_identifiers:
                digest = hashlib.sha256(str(item).encode("utf-8")).hexdigest()[:20]
                exported[key] = f"sha256:{digest}"
            else:
                exported[key] = _export_trace_payload(item)
        return exported
    if isinstance(value, list):
        return [_export_trace_payload(item) for item in value]
    return value


def write_retrieval_trace_jsonl(payload: dict[str, Any], path: str | None = None) -> bool:
    path = path or os.environ.get("OGMEM_RETRIEVAL_TRACE_JSONL")
    if not path:
        return False
    try:
        target = Path(path)
        parent_existed = target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt" and not parent_existed:
            target.parent.chmod(0o700)
        descriptor = os.open(target, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        if os.name != "nt":
            os.chmod(target, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    _export_trace_payload(payload),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("retrieval trace JSONL write failed: %s", exc)
        return False


def _write_retrieval_trace_http(payload: dict[str, Any], url: str) -> bool:
    if not _safe_trace_http_url(url):
        logger.warning("refusing insecure remote trace HTTP sink")
        return False
    try:
        timeout = float(
            os.environ.get(
                "OGMEM_TRACE_HTTP_TIMEOUT_SECONDS",
                str(_DEFAULT_HTTP_TIMEOUT_SECONDS),
            )
        )
    except ValueError:
        timeout = _DEFAULT_HTTP_TIMEOUT_SECONDS

    try:
        headers = {"Content-Type": "application/json"}
        if bearer := os.environ.get("OGMEM_TRACE_HTTP_BEARER_TOKEN", "").strip():
            headers["Authorization"] = f"Bearer {bearer}"
        if _include_trace_identifiers():
            if account_id := payload.get("account_id"):
                headers["X-Account-ID"] = str(account_id)
            if user_id := payload.get("user_id"):
                headers["X-User-ID"] = str(user_id)

        request = urllib.request.Request(
            url,
            data=json.dumps(
                _export_trace_payload(payload), ensure_ascii=False, default=str
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _trace_http_opener.open(request, timeout=max(timeout, 0.1)) as response:
            return 200 <= response.status < 300
    except (OSError, TypeError, urllib.error.URLError, ValueError) as exc:
        logger.warning("retrieval trace HTTP write failed: %s", exc)
        return False


def _safe_trace_http_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "https":
            return bool(parsed.hostname)
        return parsed.scheme == "http" and parsed.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
    except ValueError:
        return False


_http_trace_queue: queue.Queue[tuple[dict[str, Any], str, float] | None] | None = None
_http_trace_worker_started = False
_http_trace_worker_lock = threading.Lock()
_http_sink_states: dict[str, tuple[int, float]] = {}


def _http_queue_size() -> int:
    try:
        return max(
            1,
            int(os.environ.get("OGMEM_TRACE_HTTP_QUEUE_SIZE", str(_DEFAULT_HTTP_QUEUE_SIZE))),
        )
    except ValueError:
        return _DEFAULT_HTTP_QUEUE_SIZE


def _positive_float_env(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _http_trace_worker() -> None:
    assert _http_trace_queue is not None
    while True:
        item = _http_trace_queue.get()
        if item is None:
            _http_trace_queue.task_done()
            return
        payload, url, enqueued_at = item
        try:
            now = time.perf_counter()
            ttl = _positive_float_env(
                "OGMEM_TRACE_HTTP_EVENT_TTL_SECONDS",
                _DEFAULT_HTTP_EVENT_TTL_SECONDS,
            )
            failure_count, circuit_until = _http_sink_states.get(url, (0, 0.0))
            if now - enqueued_at > ttl or now < circuit_until:
                continue
            sent = False
            for attempt in range(_HTTP_RETRY_ATTEMPTS):
                if _write_retrieval_trace_http(payload, url):
                    sent = True
                    break
                if attempt + 1 < _HTTP_RETRY_ATTEMPTS:
                    time.sleep(0.1 * (2**attempt))
            if sent:
                _http_sink_states[url] = (0, 0.0)
            else:
                failure_count += 1
                if failure_count >= _HTTP_FAILURE_THRESHOLD:
                    failure_count = 0
                    circuit_until = now + _positive_float_env(
                        "OGMEM_TRACE_HTTP_CIRCUIT_SECONDS",
                        _DEFAULT_HTTP_CIRCUIT_SECONDS,
                    )
                _http_sink_states[url] = (failure_count, circuit_until)
        except Exception as exc:
            logger.warning("retrieval trace HTTP worker failed: %s", exc)
        finally:
            _http_trace_queue.task_done()


def _enqueue_retrieval_trace_http(payload: dict[str, Any], url: str) -> bool:
    global _http_trace_queue, _http_trace_worker_started
    if not _http_trace_worker_started:
        with _http_trace_worker_lock:
            if not _http_trace_worker_started:
                _http_trace_queue = queue.Queue(maxsize=_http_queue_size())
                threading.Thread(
                    target=_http_trace_worker,
                    name="retrieval-trace-http",
                    daemon=True,
                ).start()
                _http_trace_worker_started = True
    assert _http_trace_queue is not None
    try:
        _http_trace_queue.put_nowait((dict(payload), url, time.perf_counter()))
        return True
    except queue.Full:
        logger.warning("retrieval trace HTTP queue is full; dropping event")
        return False


def emit_retrieval_trace(payload: dict[str, Any]) -> bool:
    """Write one trace event to the configured sink without affecting retrieval."""
    try:
        sink = _configured_trace_sink()
        normalized = sink.lower()
        if normalized in {"", "noop", "none", "off"}:
            return False
        if normalized.startswith("jsonl:"):
            return write_retrieval_trace_jsonl(payload, sink[len("jsonl:") :])
        if normalized.startswith(("http://", "https://")):
            if not _safe_trace_http_url(sink):
                logger.warning("refusing insecure remote trace HTTP sink")
                return False
            return _enqueue_retrieval_trace_http(payload, sink)
        logger.warning("unsupported retrieval trace sink: %s", sink)
        return False
    except Exception as exc:
        logger.warning("retrieval trace emission failed: %s", exc)
        return False
