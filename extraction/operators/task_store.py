"""Async task tracking for operator pipeline execution.

Long-running pipelines (e.g. multi-step analysis of a video) are
executed on a background thread, and the caller polls
``get_task_status(task_id, ctx)`` for the account-scoped result.

Two backends are provided:
- InMemoryTaskStore: dict-backed, suitable for single-process dev
- The interface (TaskStore Protocol) leaves room for a SQL/Redis
  backend in production.

Mirrors the pattern of the existing ``MemoryWriteAPI.commit_session_async``
which keeps tasks in a thread-safe dict. Differences:
- Task records here are richer (typed) and live in their own module.
- The interface allows plug-in replacement with a durable backend.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    """Snapshot of a pipeline execution task."""
    task_id: str
    status: str = "processing"        # "processing" | "completed" | "failed"
    raw_node_id: str = ""
    operator_service: str = ""        # for observability
    submitted_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    trace_id: str = ""
    account_id: str = ""              # unowned legacy records are not queryable

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class TaskStore(Protocol):
    """Backend for tracking async pipeline tasks."""

    def submit(
        self,
        *,
        raw_node_id: str,
        operator_service: str,
        trace_id: str = "",
        account_id: str = "",
    ) -> TaskRecord: ...

    def mark_running(self, task_id: str) -> TaskRecord | None: ...

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> TaskRecord | None: ...

    def mark_failed(self, task_id: str, error: str) -> TaskRecord | None: ...

    def get(self, task_id: str) -> TaskRecord | None: ...


class InMemoryTaskStore:
    """Thread-safe in-memory task store. Suitable for single-process deployments."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        *,
        raw_node_id: str,
        operator_service: str,
        trace_id: str = "",
        account_id: str = "",
    ) -> TaskRecord:
        rec = TaskRecord(
            task_id=uuid.uuid4().hex,
            raw_node_id=raw_node_id,
            operator_service=operator_service,
            trace_id=trace_id,
            account_id=account_id,
        )
        with self._lock:
            self._tasks[rec.task_id] = rec
        return rec

    def mark_running(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            rec.started_at = _now_iso()
            return rec

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> TaskRecord | None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            rec.status = "completed"
            rec.finished_at = _now_iso()
            rec.result = result
            return rec

    def mark_failed(self, task_id: str, error: str) -> TaskRecord | None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            rec.status = "failed"
            rec.finished_at = _now_iso()
            rec.error = error
            return rec

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec

    def prune(self, older_than_seconds: int = 24 * 3600) -> int:
        """Remove terminal tasks older than the cutoff. Returns count removed."""
        cutoff = time.time() - older_than_seconds
        with self._lock:
            stale = [
                tid for tid, rec in self._tasks.items()
                if rec.status in ("completed", "failed") and rec.finished_at
                and _iso_to_ts(rec.finished_at) < cutoff
            ]
            for tid in stale:
                del self._tasks[tid]
        return len(stale)


def _iso_to_ts(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0
