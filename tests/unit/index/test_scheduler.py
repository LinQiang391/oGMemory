"""Tests for index.scheduler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

import index.scheduler as scheduler_module


class FakeBackgroundScheduler:
    def __init__(self) -> None:
        self.jobs = []
        self.listeners = []
        self.started = False
        self.shutdown_calls = []

    def add_job(self, **kwargs):
        self.jobs.append(kwargs)

    def add_listener(self, callback, mask):
        self.listeners.append((callback, mask))

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


def test_start_and_stop_manage_scheduler_state(monkeypatch):
    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeBackgroundScheduler)
    worker = Mock()
    scheduler = scheduler_module.OutboxScheduler(
        worker=worker,
        outbox_store=Mock(),
        get_account_ids=lambda: ["acct-1"],
        interval_seconds=15,
        misfire_grace_time=4,
    )

    scheduler.start()
    scheduler.stop(wait=False)

    assert scheduler.worker_id.startswith("worker-")
    assert scheduler.is_running is False
    assert scheduler.metrics.is_running is False
    assert scheduler._scheduler.jobs[0]["seconds"] == 15
    assert scheduler._scheduler.jobs[0]["misfire_grace_time"] == 4
    assert scheduler._scheduler.shutdown_calls == [False]


def test_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeBackgroundScheduler)
    warning = Mock()
    monkeypatch.setattr(scheduler_module.logger, "warning", warning)
    scheduler = scheduler_module.OutboxScheduler(
        worker=Mock(),
        outbox_store=Mock(),
        get_account_ids=lambda: [],
        worker_id="worker-1",
    )

    scheduler.start()
    scheduler.start()

    warning.assert_called_once()
    assert len(scheduler._scheduler.jobs) == 1


def test_run_once_wrapped_updates_metrics_on_success(monkeypatch):
    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeBackgroundScheduler)
    worker = Mock()
    worker.run_once.return_value = {
        "processed": 3,
        "succeeded": 2,
        "failed": 1,
        "moved_to_dlq": 1,
    }
    scheduler = scheduler_module.OutboxScheduler(
        worker=worker,
        outbox_store=Mock(),
        get_account_ids=lambda: ["acct-1", "acct-2"],
        worker_id="worker-1",
    )

    stats = scheduler._run_once_wrapped()

    assert stats["processed"] == 3
    assert scheduler.metrics.total_processed == 3
    assert scheduler.metrics.total_succeeded == 2
    assert scheduler.metrics.total_failed == 1
    assert scheduler.metrics.total_dlq == 1
    assert scheduler.metrics.consecutive_errors == 0
    worker.run_once.assert_called_once_with(
        outbox_store=scheduler._store,
        account_ids=["acct-1", "acct-2"],
        worker_id="worker-1",
    )


def test_run_once_wrapped_tracks_consecutive_errors(monkeypatch):
    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeBackgroundScheduler)
    worker = Mock()
    worker.run_once.side_effect = RuntimeError("boom")
    scheduler = scheduler_module.OutboxScheduler(
        worker=worker,
        outbox_store=Mock(),
        get_account_ids=lambda: ["acct-1"],
        worker_id="worker-1",
    )

    try:
        scheduler._run_once_wrapped()
    except RuntimeError:
        pass

    assert scheduler.metrics.consecutive_errors == 1


def test_on_job_error_logs_failures_and_misses(monkeypatch):
    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeBackgroundScheduler)
    error_log = Mock()
    warning_log = Mock()
    monkeypatch.setattr(scheduler_module.logger, "error", error_log)
    monkeypatch.setattr(scheduler_module.logger, "warning", warning_log)
    scheduler = scheduler_module.OutboxScheduler(
        worker=Mock(),
        outbox_store=Mock(),
        get_account_ids=lambda: [],
        worker_id="worker-1",
    )

    scheduler._on_job_error(SimpleNamespace(job_id="job-1", exception=RuntimeError("boom"), code=EVENT_JOB_ERROR))
    scheduler._on_job_error(SimpleNamespace(job_id="job-2", exception=None, code=EVENT_JOB_MISSED))

    assert error_log.call_count == 1
    warning_log.assert_called_once()


def test_setup_signal_handlers_stops_all_schedulers(monkeypatch):
    handlers = {}
    exits = []
    monkeypatch.setattr(scheduler_module.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(scheduler_module.sys, "exit", lambda code: exits.append(code))
    schedulers = [Mock(), Mock()]

    scheduler_module.setup_signal_handlers(schedulers)
    handlers[scheduler_module.signal.SIGTERM](15, None)

    schedulers[0].stop.assert_called_once_with(wait=True)
    schedulers[1].stop.assert_called_once_with(wait=True)
    assert exits == [0]
