from __future__ import annotations

from unittest.mock import Mock

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _sql_config(**overrides):
    values = {
        "storage_backend": "sql",
        "sql_connection_string": "postgresql://unused",
    }
    values.update(overrides)
    return OgMemConfig(**values)


def test_worker_starts_only_when_sedimentation_is_enabled(monkeypatch):
    started = []
    monkeypatch.setattr(
        MemoryService,
        "_start_skill_sedimentation_worker",
        lambda self: started.append(self._cfg.skill_sedimentation_enabled),
    )

    MemoryService(config=_sql_config(skill_sedimentation_enabled=False))
    MemoryService(config=_sql_config(skill_sedimentation_enabled=True))

    assert started == [True]


def test_agfs_backend_builds_agfs_trace_store(monkeypatch):
    service = MemoryService(config=OgMemConfig(storage_backend="agfs"))
    client = Mock()
    store = object()
    monkeypatch.setattr("server.memory_service.AGFSClient", Mock(return_value=client))
    constructor = Mock(return_value=store)
    monkeypatch.setattr("server.memory_service.AGFSTraceEventStore", constructor)

    assert service._get_trace_event_store() is store
    constructor.assert_called_once_with(
        client=client,
        mount_prefix=service._mount_prefix,
        max_turn_events=service._cfg.trace_max_turn_events,
    )


def test_session_end_wakes_worker_even_without_buffer(monkeypatch):
    service = MemoryService(config=OgMemConfig(storage_backend="agfs"))
    woke = []
    monkeypatch.setattr(service, "wake_skill_sedimentation", lambda: woke.append(True))

    result = service.dispose({})

    assert result["reason"] == "no_session_id"
    assert woke == [True]


def test_candidate_listing_delegates_without_exposing_trace(monkeypatch):
    service = MemoryService(config=_sql_config())
    captured = {}

    class _Store:
        def list_skill_candidates(self, **kwargs):
            captured.update(kwargs)
            return [{"capability_key": "review-code", "state": "accumulating"}]

    monkeypatch.setattr(service, "_get_trace_event_store", lambda: _Store())
    ctx = RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id="bot",
        session_id="s1",
        trace_id="t1",
    )

    result = service.list_skill_candidates(ctx, state="accumulating")

    assert result == [{"capability_key": "review-code", "state": "accumulating"}]
    assert captured == {
        "account_id": "acct-1",
        "agent_id": "bot",
        "state": "accumulating",
        "limit": 50,
    }


def test_shutdown_stops_and_clears_skill_worker():
    service = MemoryService(config=_sql_config())

    class _Worker:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class _Thread:
        def __init__(self):
            self.joined = False

        def join(self, timeout):
            self.joined = True

        def is_alive(self):
            return False

    worker = _Worker()
    thread = _Thread()
    service._skill_sedimentation_worker = worker
    service._skill_sedimentation_thread = thread

    service.shutdown(join_timeout=.01)

    assert worker.stopped is True
    assert thread.joined is True
    assert service._skill_sedimentation_worker is None
    assert service._skill_sedimentation_thread is None


def test_stop_timeout_retains_worker_references_and_blocks_duplicate_start(monkeypatch):
    service = MemoryService(config=_sql_config())

    worker = Mock()
    thread = Mock()
    thread.is_alive.return_value = True
    service._skill_sedimentation_worker = worker
    service._skill_sedimentation_thread = thread
    constructor = Mock(side_effect=AssertionError("duplicate worker constructed"))
    monkeypatch.setattr("skills.sedimentation.SkillSedimentationWorker", constructor)

    assert service.stop_skill_sedimentation(timeout=.01) is False
    assert service._skill_sedimentation_worker is worker
    assert service._skill_sedimentation_thread is thread

    service._start_skill_sedimentation_worker()
    constructor.assert_not_called()

    thread.is_alive.return_value = False
    assert service.stop_skill_sedimentation(timeout=.01) is True
    assert service._skill_sedimentation_worker is None
    assert service._skill_sedimentation_thread is None


def test_account_discovery_requests_only_a_bounded_actionable_page(monkeypatch):
    service = MemoryService(config=_sql_config(skill_account_workers=3))
    captured = {}

    class _Store:
        def list_pending_skill_accounts(self, **kwargs):
            captured.update(kwargs)
            return ["oldest", "next"]

    monkeypatch.setattr(service, "_get_trace_event_store", lambda: _Store())

    assert service._skill_account_ids() == ["oldest", "next"]
    assert captured == {"limit": 6}


def test_health_exposes_worker_liveness_error_and_scheduler_backlog(monkeypatch):
    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    service = MemoryService(config=_sql_config(
        skill_sedimentation_enabled=True, provider="mock", vector_db_type="memory",
    ))
    mock_pool = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=mock_pool))

    class _Store:
        def count_skill_work_accounts(self):
            return 7

    monkeypatch.setattr(service, "_get_trace_event_store", lambda: _Store())
    service._skill_sedimentation_worker = Mock()
    service._skill_sedimentation_worker.status.return_value = {
        "running": False, "last_error": "database unavailable",
    }
    service._skill_sedimentation_thread = Mock()
    service._skill_sedimentation_thread.is_alive.return_value = False

    health = service.health()

    assert health["status"] == "error"
    assert health["skill_sedimentation"]["thread_alive"] is False
    assert health["skill_sedimentation"]["scheduler_backlog"] == 7
    assert "not running" in health["error"]


def test_health_is_error_when_live_worker_reports_processing_failure(monkeypatch):
    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    service = MemoryService(config=_sql_config(
        skill_sedimentation_enabled=True, provider="mock", vector_db_type="memory",
    ))
    mock_pool = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=mock_pool))
    store = Mock()
    store.count_skill_work_accounts.return_value = 2
    monkeypatch.setattr(service, "_get_trace_event_store", lambda: store)
    service._skill_sedimentation_worker = Mock()
    service._skill_sedimentation_worker.status.return_value = {
        "running": True,
        "last_error": "TimeoutError: provider unavailable",
        "last_batch": {"retrying": 1, "errors": ["TimeoutError: provider unavailable"]},
    }
    service._skill_sedimentation_thread = Mock()
    service._skill_sedimentation_thread.is_alive.return_value = True

    health = service.health()

    assert health["status"] == "error"
    assert "provider unavailable" in health["error"]
