from __future__ import annotations

import pytest

from core.models import RequestContext
from session.models import SessionMessage
from session.session_manager import SessionBuffer, SessionManager, SessionOperationInProgress


def test_stable_id_insert_noop_replace_remove_and_prepare_invalidation():
    buffer = SessionBuffer(session_id="s")
    first = buffer.add("tool", "inline", message_id="runtime-1")
    assert buffer.add("tool", "inline", message_id="runtime-1") is first

    buffer.compaction_prepare_token = "prepared"
    replacement = buffer.add("tool", "stub", message_id="runtime-1")
    assert replacement.content == "stub"
    assert len(buffer.messages) == 1
    assert buffer.compaction_prepare_token == ""

    buffer.compaction_prepare_token = "prepared-again"
    assert buffer.remove_unextracted("runtime-1") is True
    assert buffer.messages == []
    assert buffer.compaction_prepare_token == ""


def test_stable_id_cannot_rewrite_or_remove_extracted_snapshot():
    buffer = SessionBuffer(session_id="s")
    buffer.add("tool", "inline", message_id="runtime-1")
    buffer.extraction_watermark = 1
    with pytest.raises(ValueError):
        buffer.add("tool", "stub", message_id="runtime-1")
    with pytest.raises(ValueError):
        buffer.remove_unextracted("runtime-1")


def test_extraction_updates_shared_operation_counter():
    buffer = SessionBuffer(session_id="s")
    buffer.begin_extraction()
    assert buffer.active_operation_count == 1
    buffer.end_extraction()
    assert buffer.active_operation_count == 0


def test_commit_rejected_by_cleaning_does_not_leave_commit_flag():
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    buffer = manager.get_or_create("s", ctx=ctx)
    buffer.add("user", "pending")
    buffer.cleaning = True
    with pytest.raises(SessionOperationInProgress) as captured:
        manager.commit("s", ctx, wait=True)
    assert captured.value.conflict_type == "session_deleting"
    assert captured.value.retryable is False
    assert buffer.commit_in_progress is False


def test_begin_managed_operation_rejects_a_detached_buffer(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    detached = manager.get_or_create("s", ctx=ctx)
    manager.remove_session("s", ctx=ctx)
    monkeypatch.setattr(manager, "get_or_create", lambda *_args, **_kwargs: detached)

    with pytest.raises(SessionOperationInProgress) as captured:
        manager.begin_managed_operation("s", ctx)
    assert captured.value.conflict_type == "session_replaced"
    assert captured.value.retryable is True
    assert detached.active_operation_count == 0


def test_begin_managed_operation_does_not_retry_a_cleanup_deleted_buffer(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    deleted = manager.get_or_create("s", ctx=ctx)
    manager.finish_session_cleanup("s", deleted, remove=True)
    monkeypatch.setattr(manager, "get_or_create", lambda *_args, **_kwargs: deleted)

    with pytest.raises(SessionOperationInProgress) as captured:
        manager.begin_managed_operation("s", ctx)
    assert captured.value.conflict_type == "session_deleting"
    assert captured.value.retryable is False
    assert deleted.active_operation_count == 0


def _mark_commit_success(_snapshot, _session_id, _archive_id, task_info, _ctx, **_kwargs):
    task_info.update(status="completed", archived=True)


def test_commit_save_exception_releases_active_operation(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    manager.add_message("s", "user", "pending", ctx)
    buffer = manager.get_or_create("s", ctx=ctx)
    monkeypatch.setattr(manager, "_process_snapshot", _mark_commit_success)
    monkeypatch.setattr(
        manager,
        "save_session_state",
        lambda *_args: (_ for _ in ()).throw(OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        manager.commit("s", ctx, wait=True)

    assert buffer.active_operation_count == 0
    assert buffer.commit_in_progress is False


def test_commit_phase1_exception_restores_buffer_and_releases_operation(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    manager.add_message("s", "user", "pending", ctx)
    buffer = manager.get_or_create("s", ctx=ctx)
    original_ids = [message.id for message in buffer.messages]
    monkeypatch.setattr(
        manager,
        "_sync_runtime_state_to_session_state",
        lambda *_args: (_ for _ in ()).throw(OSError("phase1 failed")),
    )

    with pytest.raises(OSError, match="phase1 failed"):
        manager.commit("s", ctx, wait=True)

    assert [message.id for message in buffer.messages] == original_ids
    assert buffer.active_operation_count == 0
    assert buffer.commit_in_progress is False


def test_commit_snapshot_save_exception_releases_active_operation(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    buffer = manager.get_or_create("s", ctx=ctx)
    monkeypatch.setattr(manager, "_process_snapshot", _mark_commit_success)
    monkeypatch.setattr(
        manager,
        "save_session_state",
        lambda *_args: (_ for _ in ()).throw(OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        manager.commit_snapshot(
            "s", [SessionMessage(id="m1", role="user", content="pending")], ctx,
        )

    assert buffer.active_operation_count == 0


def test_commit_thread_start_failure_restores_buffer_and_releases_operation(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    manager.add_message("s", "user", "pending", ctx)
    buffer = manager.get_or_create("s", ctx=ctx)
    original_ids = [message.id for message in buffer.messages]
    monkeypatch.setattr(
        "session.session_manager.threading.Thread.start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        manager.commit("s", ctx, wait=False)

    assert [message.id for message in buffer.messages] == original_ids
    assert buffer.active_operation_count == 0
    assert buffer.commit_in_progress is False


def test_commit_snapshot_thread_start_failure_releases_operation(monkeypatch):
    manager = SessionManager()
    ctx = RequestContext("acct", "user", "agent", "s", "trace")
    buffer = manager.get_or_create("s", ctx=ctx)
    monkeypatch.setattr(
        "session.session_manager.threading.Thread.start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        manager.commit_snapshot(
            "s",
            [SessionMessage(id="m1", role="user", content="pending")],
            ctx,
            wait=False,
        )

    assert buffer.active_operation_count == 0
