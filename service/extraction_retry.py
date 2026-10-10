"""Bounded source retries backed by the existing durable extraction journal.

No credentials or caller-supplied payload are persisted here. After a restart,
an authenticated visit to the owning session wakes its saved task. SQL's session
fence protects claims across workers; uncertain writes are never re-extracted.
"""

import logging
import threading
import time
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from queue import Empty, Queue

from fs.sql_adapter.session_mutation import SessionMutationBusyError
from service.extraction_checkpoint import (
    EXTENSION_KEY,
    ExtractionBatchJournal,
    RecoveryNeedsConfirmationError,
    recovery_owner,
)

logger = logging.getLogger(__name__)
_running = ContextVar("extraction_retry_running", default=None)
_format_feedback = ContextVar("memory_card_format_feedback", default="")
MAX_ATTEMPTS = 2
ATTEMPT_SECONDS = 1200


def is_frozen_source_retry(ctx, batch):
    return _running.get() == (recovery_owner(ctx), batch.get("archive_id"))


def background_retry_running():
    return _running.get() is not None


def card_format_feedback():
    return _format_feedback.get()


def retry_failed_source(function):
    @wraps(function)
    def wrapped(self, params):
        requested = params.get("retryInBackground", False)
        if type(requested) is not bool:
            raise ValueError("retryInBackground must be a boolean")
        client_request_id = str(params.get("clientRequestId") or params.get("client_request_id") or "").strip()
        if requested and not client_request_id:
            raise ValueError("retryInBackground requires a stable clientRequestId")
        result = function(self, params)
        if (requested and not _running.get() and result.get("ok") is False
                and result.get("status") == "failed" and result.get("retryable") is True
                and result.get("client_request_id") == client_request_id):
            ctx = params.get("_ctx") or self.build_context(params)
            try:
                task = self._extraction_retry.register(ctx, result)
                if task:
                    return {**result, "recovery_task": task}
            except Exception as exc:
                logger.exception("Cannot persist extraction retry")
                return {**result, "retryable": False, "status": "needs_confirmation",
                        "error": "retry checkpoint not acknowledged: " + str(exc)}
        return result
    return wrapped


class ExtractionRetryWorker:
    def __init__(self, service):
        self.service = service
        self._queue = Queue()
        self._lock = threading.Lock()
        self._queued = set()
        self._thread = None
        self._stop = threading.Event()

    def _journal(self, ctx, archive_id=None):
        service = self.service
        service._load_extraction_recovery_before_accumulation(ctx.session_id, ctx)
        manager = service.get_session_manager()
        buf = manager.get_or_create(ctx.session_id, ctx=ctx)
        with buf.session_state_save_lock:
            batch = buf.runtime_extensions.get(EXTENSION_KEY, {}).get("batch")
            if batch is None:
                return None
            if batch.get("owner") != recovery_owner(ctx) or "lifecycle" in batch:
                raise RecoveryNeedsConfirmationError("retry source belongs to another owner or lifecycle")
            if archive_id is not None and batch.get("archive_id") != archive_id:
                return None
            return ExtractionBatchJournal.resume(manager, buf, ctx)

    def register(self, ctx, failure):
        archive_id = failure.get("extraction_archive_id")
        client_request_id = failure.get("client_request_id")
        if not archive_id or not client_request_id:
            return None
        with self.service._session_mutation_scope(ctx):
            journal = self._journal(ctx, archive_id)
            if journal is None:
                return None
            with journal.buffer.session_state_save_lock:
                if journal.batch.get("client_request_id") != client_request_id:
                    return None
                task = journal.batch.get("retry_task")
                if task is None:
                    # A new task never replaces accepted candidates or their receipts.
                    task = {"id": journal.batch["archive_id"], "status": "queued", "attempts": 0,
                            "max_attempts": MAX_ATTEMPTS, "next_attempt_at": time.time() + 2,
                            "last_error": str(failure.get("error", ""))[:1000]}
                    if failure.get("failure_kind") == "memory_card_format":
                        task["failure_kind"] = "memory_card_format"
                    journal.mutate(lambda batch: batch.update(retry_task=task))
                task = deepcopy(task)
        self.wake(ctx, task)
        return task

    def wake(self, ctx, task):
        if task.get("status") not in {"queued", "running"} or self._stop.is_set():
            return
        identity = (*recovery_owner(ctx).values(), task["id"])
        with self._lock:
            if identity in self._queued:
                return
            self._queued.add(identity)
            self._queue.put((identity, ctx, task["id"]))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="extraction-retry", daemon=True)
                self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                identity, ctx, archive_id = self._queue.get(timeout=.2)
            except Empty:
                continue
            requeued = False
            try:
                delay = self.run_once(ctx, archive_id)
                if delay is not None and not self._stop.is_set():
                    # Yield to other sources between attempts instead of holding
                    # the single worker through this source's whole retry budget.
                    self._queue.put((identity, ctx, archive_id))
                    requeued = True
                    self._stop.wait(min(delay, .2))
                    continue
            except Exception:
                # Durable queued/running record remains visible for authenticated
                # recovery; do not turn storage/fencing failures into success.
                logger.exception("Extraction retry worker stopped a source task")
            finally:
                if not requeued:
                    with self._lock:
                        self._queued.discard(identity)
                self._queue.task_done()

    def run_once(self, ctx, archive_id):
        """One fenced attempt; return backoff seconds or None for a terminal task."""
        try:
            delay, attempted = self._run_fenced(ctx, archive_id)
        except SessionMutationBusyError:
            return 1
        # Nested after_turn correctly defers indexing until the outer SQL fence
        # commits; launch it only after that transaction has actually finished.
        if attempted:
            self.service._async_drain(account_id=ctx.account_id)
        return delay

    def _run_fenced(self, ctx, archive_id):
        with self.service._session_mutation_scope(ctx):
            journal = self._journal(ctx, archive_id)
            if journal is None:
                return None, False
            with journal.buffer.session_state_save_lock:
                task = deepcopy(journal.batch.get("retry_task", {}))
                if task.get("status") not in {"queued", "running"}:
                    return None, False
                if task.get("attempts", 0) >= MAX_ATTEMPTS:
                    task.update(status="failed", last_error="retry attempts exhausted")
                    journal.mutate(lambda batch: batch.update(retry_task=task))
                    return None, False
                delay = task.get("next_attempt_at", 0) - time.time()
                if delay > 0:
                    return min(delay, 20), False
                task.update(status="running", attempts=task.get("attempts", 0) + 1)
                journal.mutate(lambda batch: batch.update(retry_task=task))
                batch = journal.batch
                if not batch.get("client_request_id"):
                    raise RecoveryNeedsConfirmationError("background retry source lacks a stable request identity")
                params = {"_ctx": ctx, "sessionId": ctx.session_id, "wait": True,
                          "forceExtract": True, "timeoutSeconds": ATTEMPT_SECONDS,
                          "clientRequestId": batch.get("client_request_id", ""),
                          "messages": deepcopy(batch["request_messages"])}
            token = _running.set((recovery_owner(ctx), archive_id))
            feedback_token = _format_feedback.set(
                task.get("last_error", "") if task.get("failure_kind") == "memory_card_format" else "")
            try:
                result = self.service.after_turn(params)
            finally:
                _format_feedback.reset(feedback_token)
                _running.reset(token)
            journal = self._journal(ctx, archive_id)
            if journal is None:
                return None, True  # finish() acknowledged this exact source.
            with journal.buffer.session_state_save_lock:
                current = journal.batch.get("retry_task", {})
                if current.get("status") != "running" or current.get("attempts") != task["attempts"]:
                    return None, True
                processing = result.get("status") == "processing"
                retryable = result.get("retryable") is True and result.get("status") != "needs_confirmation"
                if processing:
                    task.update(status="queued", attempts=task["attempts"] - 1)
                elif retryable:
                    task["status"] = "queued" if task["attempts"] < MAX_ATTEMPTS else "failed"
                else:
                    task["status"] = "needs_confirmation"
                task.update(last_error=str(result.get("error", result.get("status", "pending")))[:1000],
                            next_attempt_at=time.time() + 10)
                journal.mutate(lambda batch: batch.update(retry_task=task))
                return (10 if task["status"] == "queued" else None), not processing

    def close(self, timeout=1):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
