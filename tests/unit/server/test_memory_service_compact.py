from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.models import RequestContext
from providers import unified_config
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.topic_buffer import SlotContent


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="sess-compact",
        trace_id="trace-1",
    )


@pytest.fixture
def service() -> MemoryService:
    return MemoryService(config=OgMemConfig())


def _seed_buffer(service: MemoryService, ctx: RequestContext, session_id: str = "sess-compact"):
    mgr = service.get_session_manager()
    mgr.add_message(session_id, "user", "first user fact", ctx)
    mgr.add_message(session_id, "assistant", "assistant filler", ctx)
    mgr.add_message(session_id, "user", "second user fact", ctx)
    return mgr.get_or_create(session_id)


def test_prepare_compaction_uses_incremental_non_assistant_messages(service: MemoryService, ctx: RequestContext):
    buf = _seed_buffer(service, ctx)
    buf.extraction_watermark = 1
    buf.extraction_summary = "existing-summary"
    buf.tool_usage_stats = {
        "search_web": {
            "call_count": 2,
            "success_count": 2,
            "fail_count": 0,
            "total_duration_ms": 14.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }
    }
    buf.messages[2].created_at = "2026-04-19T01:02:03+00:00"

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    }
    service._write_api = write_api

    result = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["buffer"] is buf
    write_api.commit_session.assert_called_once()
    call_messages = write_api.commit_session.call_args.kwargs["messages"]
    assert len(call_messages) == 1
    assert call_messages[0]["role"] == "user"
    assert call_messages[0]["content"] == "second user fact"
    assert "id" in call_messages[0]
    assert write_api.commit_session.call_args.kwargs["session_summary"] == "existing-summary"
    assert write_api.commit_session.call_args.kwargs["session_time"].isoformat() == "2026-04-19T01:02:03+00:00"
    assert "search_web" in write_api.commit_session.call_args.kwargs["tool_stats_text"]
    assert buf.extraction_watermark == 3
    assert "task-1" in buf.extraction_summary


def test_prepare_compaction_invalidates_cached_identity_when_profile_is_written(
    service: MemoryService,
    ctx: RequestContext,
):
    buf = _seed_buffer(service, ctx)
    topic = service.get_session_manager().get_topic_buffer("sess-compact")
    topic.set_cached_slot(
        "identity",
        SlotContent(
            content="stale identity",
            uris=["ctx://acct-test/users/user-test/memories/profile"],
            tokens=4,
        ),
    )

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [
            {
                "action": "merge",
                "target_uri": "ctx://acct-test/users/user-test/memories/profile",
            }
        ],
    }
    service._write_api = write_api

    service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert buf.extraction_watermark == 3
    assert topic.get_cached_slot("identity") is None


def test_prepare_compaction_raises_when_write_api_unavailable(service: MemoryService, ctx: RequestContext):
    _seed_buffer(service, ctx)
    service.get_write_api = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="write API unavailable"):
        service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})


def test_prepare_compaction_advances_watermark_when_only_assistant_messages(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "assistant", "only assistant", ctx)
    buf = mgr.get_or_create("sess-compact")

    write_api = Mock()
    service._write_api = write_api

    result = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["messages"] == []
    write_api.commit_session.assert_not_called()
    assert buf.extraction_watermark == 1
    assert buf.extraction_summary == ""


def test_prepare_compaction_does_not_advance_watermark_or_issue_token_when_write_fails(service: MemoryService, ctx: RequestContext):
    buf = _seed_buffer(service, ctx)
    buf.extraction_watermark = 1
    buf.extraction_summary = "existing-summary"

    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 1,
        "plans": [],
    }
    service._write_api = write_api

    with pytest.raises(RuntimeError, match="compaction extraction write failed"):
        service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert buf.extraction_watermark == 1
    assert buf.extraction_summary == "existing-summary"
    assert buf.compaction_prepare_token == ""
    assert buf.compaction_prepare_token_created_at == ""


def test_prepare_compaction_reuses_existing_prepare_token_and_archive_id(
    service: MemoryService,
    ctx: RequestContext,
):
    _seed_buffer(service, ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {
        "candidates_extracted": 0,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [],
    }
    service._write_api = write_api

    first = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    second = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})

    assert second["prepareToken"] == first["prepareToken"]
    assert second["archive_id"] == first["archive_id"]
    write_api.commit_session.assert_called_once()


def test_after_turn_uses_configured_threshold_before_env(
    ctx: RequestContext,
    monkeypatch,
):
    monkeypatch.setenv("OGMEM_AFTER_TURN_THRESHOLD", "1")
    service = MemoryService(config=OgMemConfig(after_turn_threshold=999999))
    service.get_write_api = Mock(
        side_effect=AssertionError("after_turn should still be accumulating")
    )

    result = service.after_turn({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "messages": [
            {"role": "user", "content": "x" * 80},
            {"role": "assistant", "content": "ack"},
        ],
    })

    assert result["status"] == "accumulating"
    assert result["pending_tokens"] < 999999


def test_compact_returns_required_not_compacted_result_shape_when_commit_has_no_archive(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": False, "reason": "empty_buffer"})

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx})

    assert result == {
        "ok": True,
        "compacted": False,
        "reason": "empty_buffer",
        "result": {
            "summary": "",
            "tokensBefore": 5,
        },
    }


def test_compact_surfaces_failed_archive_commit_and_keeps_buffer(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx})

    assert result["ok"] is False
    assert result["compacted"] is False
    assert result["status"] == "failed"
    assert "archive store unavailable" in result["error"]
    session = mgr.get_session("sess-compact", ctx)
    assert session["message_count"] == 1


def test_after_turn_no_extractable_archive_failure_rewinds_watermark(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service.drain_outbox_sync = Mock(return_value={"processed": 0, "succeeded": 0, "failed": 0})
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    result = service.after_turn({
        "sessionId": "sess-compact",
        "messages": [{"role": "assistant", "content": "assistant only content"}],
        "_ctx": ctx,
    })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive store unavailable"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_after_turn_background_archive_failure_rewinds_watermark(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "remember this failure path"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is True
    assert result["status"] == "processing"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_after_turn_background_invalidates_cached_identity_when_profile_is_written(
    service: MemoryService,
    ctx: RequestContext,
    monkeypatch,
):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [
            {
                "action": "merge",
                "target_uri": "ctx://acct-test/users/user-test/memories/profile",
            }
        ],
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": False})
    topic = mgr.get_topic_buffer("sess-compact")
    topic.set_cached_slot(
        "identity",
        SlotContent(
            content="stale identity",
            uris=["ctx://acct-test/users/user-test/memories/profile"],
            tokens=4,
        ),
    )

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "my profile changed"}],
            "_ctx": ctx,
        })

    assert result["ok"] is True
    assert result["status"] == "processing"
    assert topic.get_cached_slot("identity") is None


def test_after_turn_wait_runs_extraction_synchronously(service: MemoryService, ctx: RequestContext):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
        "plans": [{"action": "create", "target_uri": "ctx://acct/users/u/memories/case/task-1"}],
    })
    service._async_drain = Mock(return_value={"processed": 1, "succeeded": 1, "failed": 0})
    mgr = service.get_session_manager()
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": False})

    thread_attempts = []

    class RecordingThread:
        def __init__(self, *args, **kwargs):
            thread_attempts.append({"args": args, "kwargs": kwargs})
            raise AssertionError("wait=True should not start a background thread")

    with patch("server.memory_service.threading.Thread", RecordingThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "remember this immediately"}],
            "_ctx": ctx,
            "wait": True,
        })

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["writes_completed"] == 1
    assert result["drain"] == {"processed": 1, "succeeded": 1, "failed": 0}
    assert thread_attempts == []
    service._chunk_and_extract.assert_called_once()
    service._async_drain.assert_called_once_with(account_id="acct-test")
    mgr.commit_snapshot.assert_called_once()


def test_after_turn_background_write_failure_rewinds_watermark_and_skips_archive_commit(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 0,
        "writes_skipped": 0,
        "writes_failed": 1,
        "plans": [],
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": True,
        "uri": "ctx://acct-test/sessions/sess-compact/history/arc-ok",
        "archive_id": "arc-ok",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))
    mgr.commit_snapshot = Mock(return_value={"ok": True, "archived": True})

    class InlineThread:
        def __init__(self, target, daemon=None, name=None):
            self.target = target

        def start(self):
            self.target()

    with patch("server.memory_service.threading.Thread", InlineThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "trigger write failure"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is True
    assert result["status"] == "processing"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1
    mgr.commit_snapshot.assert_not_called()


def test_after_turn_sync_fallback_archive_failure_rewinds_watermark_and_fails(service: MemoryService, ctx: RequestContext, monkeypatch):
    service._cfg.after_turn_threshold = 1
    service._write_api = Mock()
    service._chunk_and_extract = Mock(return_value={
        "candidates_extracted": 1,
        "candidates_filtered": 0,
        "writes_completed": 1,
        "writes_skipped": 0,
        "writes_failed": 0,
    })
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr._write_archive = Mock(return_value={
        "success": False,
        "uri": "",
        "archive_id": "arc-failed",
        "error": "archive store unavailable",
    })
    mgr._compress = Mock(return_value=("overview", "abstract"))

    class FailingThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("thread unavailable")

    with patch("server.memory_service.threading.Thread", FailingThread):
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "remember sync fallback failure"}],
            "_ctx": ctx,
        })

    buf = mgr.get_or_create("sess-compact")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["error"] == "archive store unavailable"
    assert buf.extraction_watermark == 0
    assert len(buf.messages) == 1


def test_compact_rejects_prepared_flag_without_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(side_effect=AssertionError("prepare_compaction should not run"))

    with pytest.raises(ValueError, match="prepare token required"):
        service.compact({"sessionId": "sess-compact", "_ctx": ctx, "prepared": True})


def test_compact_rejects_invalid_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(side_effect=AssertionError("prepare_compaction should not run"))

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": "bad-token",
        })


def test_compact_invalid_prepare_token_does_not_invalidate_valid_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": "bad-token",
        })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


def test_compact_accepts_one_time_verified_prepare_token(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"
    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": prepared["prepareToken"],
        })


def test_compact_reuses_prepare_archive_id_for_archive_commit(
    service: MemoryService,
    ctx: RequestContext,
):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)

    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    archive_id = prepared["archive_id"]

    mgr.commit = Mock(return_value={"ok": True, "archived": True, "archive_id": archive_id})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert write_api.commit_session.call_args.kwargs["archive_id"] == archive_id
    mgr.commit.assert_called_once_with("sess-compact", ctx, wait=True, archive_id=archive_id)


def test_compact_invalid_prepare_token_ttl_falls_back_to_default(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    service._cfg.compact_prepare_token_ttl = "bad"
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 9,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "prepareToken": prepared["prepareToken"],
    })

    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


def test_compact_rejects_expired_prepare_token(service: MemoryService, ctx: RequestContext):
    service._cfg.compact_prepare_token_ttl = 300
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    write_api = Mock()
    write_api.commit_session.return_value = {"candidates_extracted": 0, "plans": []}
    service._write_api = write_api
    prepared = service.prepare_compaction({"sessionId": "sess-compact", "_ctx": ctx})
    buf = mgr.get_or_create("sess-compact")
    buf.compaction_prepare_token_created_at = "2000-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="invalid prepare token"):
        service.compact({
            "sessionId": "sess-compact",
            "_ctx": ctx,
            "prepareToken": prepared["prepareToken"],
        })

    assert buf.compaction_prepare_token == ""
    assert buf.compaction_prepare_token_created_at == ""


def test_compact_trims_summary_and_does_not_fabricate_first_kept_entry_id(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True, "archive_id": "archive-123"})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({"sessionId": "sess-compact", "_ctx": ctx, "tokenBudget": 128_000})

    assert result["ok"] is True
    assert result["compacted"] is True
    assert result["result"]["summary"] == "90ABCDEFGHIJ"
    assert result["result"]["tokensAfter"] == 42
    assert "firstKeptEntryId" not in result["result"]


def test_compact_request_summary_max_chars_overrides_config(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "summaryMaxChars": 8,
    })

    assert result["result"]["summary"] == "CDEFGHIJ"


def test_compact_invalid_request_summary_max_chars_falls_back_to_config(service: MemoryService, ctx: RequestContext):
    service._cfg.summary_max_chars = 12
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "1234567890ABCDEFGHIJ",
        "estimatedTokens": 42,
    })

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "summaryMaxChars": 0,
    })

    assert result["result"]["summary"] == "90ABCDEFGHIJ"


def test_compact_sync_short_term_index_mode_drains_outbox_best_effort(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 42,
    })
    service.drain_outbox_sync = Mock(side_effect=RuntimeError("index unavailable"))

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "shortTermIndexMode": "sync",
    })

    service.drain_outbox_sync.assert_called_once()
    assert result["ok"] is True
    assert result["compacted"] is True
    assert result["result"]["summary"] == "summary"


@pytest.mark.parametrize("mode", ["async", "off"])
def test_compact_async_and_off_short_term_index_modes_skip_drain(service: MemoryService, ctx: RequestContext, mode: str):
    mgr = service.get_session_manager()
    mgr.add_message("sess-compact", "user", "pending user content", ctx)
    service.prepare_compaction = Mock(return_value={"ctx": ctx, "mgr": mgr, "session_id": "sess-compact"})
    mgr.commit = Mock(return_value={"ok": True, "archived": True})
    mgr.get_context = Mock(return_value={
        "latest_archive_overview": "summary",
        "estimatedTokens": 42,
    })
    service.drain_outbox_sync = Mock()

    result = service.compact({
        "sessionId": "sess-compact",
        "_ctx": ctx,
        "shortTermIndexMode": mode,
    })

    service.drain_outbox_sync.assert_not_called()
    assert result["ok"] is True
    assert result["compacted"] is True


def test_dispose_no_pending_messages_removes_session_when_context_build_fails(service: MemoryService, ctx: RequestContext):
    mgr = service.get_session_manager()
    mgr.get_or_create("sess-compact", ctx=ctx)
    service.build_context = Mock(side_effect=RuntimeError("bad ctx"))

    result = service.dispose({"sessionId": "sess-compact"})

    assert result == {"disposed": True, "flushed": False, "reason": "no_pending_messages"}
    assert not mgr.has_session("sess-compact")


def test_ogmem_config_uses_default_summary_max_chars():
    cfg = OgMemConfig()

    assert cfg.summary_max_chars == 4000


def test_ogmem_config_load_reads_summary_max_chars_from_yaml_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OGMEM_SUMMARY_MAX_CHARS", "222")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"summary_max_chars": 123}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.summary_max_chars == 123

    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.summary_max_chars == 222


# ---------------------------------------------------------------------------
# ingest() buffer writing + after_turn() dedup
# ---------------------------------------------------------------------------

class TestIngestWritesToBuffer:
    """ingest() should write messages to the session buffer."""

    def test_ingest_writes_user_message_to_buffer(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "hello from ingest"}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 1

        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1
        assert buf.messages[0].role == "user"
        assert buf.messages[0].content == "hello from ingest"

    def test_ingest_empty_messages_returns_early(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({"sessionId": "sess-compact", "messages": [], "_ctx": ctx})

        assert result["ingested"] is True

    def test_ingest_skips_empty_content(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": ""}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 0

    def test_ingest_multiple_messages(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
            ],
            "_ctx": ctx,
        })

        assert result["count"] == 2
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2

    def test_ingest_batch_uses_same_logic(self, service: MemoryService, ctx: RequestContext):
        result = service.ingest_batch({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "batch msg"}],
            "_ctx": ctx,
        })

        assert result["ingested"] is True
        assert result["count"] == 1
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1


class TestAfterTurnDedup:
    """after_turn() should skip messages already written by ingest()."""

    def test_after_turn_skips_ingest_written_user_message(self, service: MemoryService, ctx: RequestContext):
        # Simulate OperatorAgent flow: ingest → after_turn
        service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "hello from user"}],
            "_ctx": ctx,
        })

        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 1  # ingest wrote 1 message

        # after_turn sends user + assistant (same user message as ingest)
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "hello from user"},
                {"role": "assistant", "content": "hi from assistant"},
            ],
            "_ctx": ctx,
        })

        assert result["ok"] is True
        # Should NOT duplicate user message — only assistant is new
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2
        assert buf.messages[0].role == "user"
        assert buf.messages[0].content == "hello from user"
        assert buf.messages[1].role == "assistant"
        assert buf.messages[1].content == "hi from assistant"

    def test_after_turn_no_dedup_when_ingest_not_called(self, service: MemoryService, ctx: RequestContext):
        # Direct after_turn without prior ingest — no dedup needed
        result = service.after_turn({
            "sessionId": "sess-compact",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "_ctx": ctx,
        })

        assert result["ok"] is True
        mgr = service.get_session_manager()
        buf = mgr.get_or_create("sess-compact")
        assert len(buf.messages) == 2

    def test_ingest_then_compact_finds_buffer(self, service: MemoryService, ctx: RequestContext):
        """compact() should find messages written by ingest() — the fix for empty_buffer."""
        service.ingest({
            "sessionId": "sess-compact",
            "messages": [{"role": "user", "content": "important data"}],
            "_ctx": ctx,
        })

        mgr = service.get_session_manager()

        # compact's commit should find buffer non-empty
        result = mgr.commit("sess-compact", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is True
        assert result.get("reason") != "empty_buffer"


def test_openclaw_plugin_manifest_exposes_compaction_config_schema():
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "openclaw_context_engine_plugin" / "openclaw.plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    props = manifest["configSchema"]["properties"]

    assert props["compactTakeoverEnabled"]["type"] == "boolean"
    assert props["compactTakeoverEnabled"]["default"] is True
    assert props["summaryMaxChars"]["type"] == "integer"
    assert props["summaryMaxChars"]["default"] == 4000
    assert props["shortTermIndexMode"]["enum"] == ["sync", "async", "off"]
    assert props["shortTermIndexMode"]["default"] == "sync"
