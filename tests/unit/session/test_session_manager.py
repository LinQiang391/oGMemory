"""Unit tests for SessionManager."""

from unittest.mock import Mock, patch

import pytest

from core.models import RequestContext
from session.models import ArchiveEntry, SessionMessage, SessionMeta
from session.session_manager import SessionBuffer, SessionManager


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct-test",
        user_id="u-test",
        agent_id="agent-test",
        session_id="sess-1",
        trace_id="trace-1",
    )


@pytest.fixture
def mgr():
    return SessionManager()


@pytest.fixture
def mgr_with_deps():
    mock_llm = Mock()
    mock_write_api = Mock()
    mock_write_api.commit_session.return_value = {
        "candidates_extracted": 1,
        "writes_completed": 1,
        "task_id": "t-1",
        "status": "completed",
    }
    return SessionManager(
        get_llm=lambda: mock_llm,
        get_write_api=lambda: mock_write_api,
        get_agfs=lambda: None,  # No AGFS in unit tests
    )


# ---------------------------------------------------------------------------
# SessionBuffer tests
# ---------------------------------------------------------------------------


class TestSessionBuffer:
    def test_add_message(self):
        buf = SessionBuffer(session_id="s1")
        msg = buf.add("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert len(buf.messages) == 1
        assert buf.meta.message_count == 1

    def test_pending_tokens(self):
        buf = SessionBuffer(session_id="s1")
        buf.add("user", "a" * 100)  # ~25 tokens
        buf.add("assistant", "b" * 200)  # ~50 tokens
        assert buf.pending_tokens == 75

    def test_tool_usage_stats_setter_merges_existing_usage(self):
        buf = SessionBuffer(session_id="s1")
        buf.usage_stats.record_tool_call(
            tool_name="new_tool",
            status="success",
            prompt_tokens=7,
            completion_tokens=3,
        )

        buf.tool_usage_stats = {
            "legacy_tool": {
                "call_count": 1,
                "success_count": 1,
                "fail_count": 0,
                "total_prompt_tokens": 2,
                "total_completion_tokens": 1,
            }
        }

        stats = buf.tool_usage_stats
        assert stats["new_tool"]["call_count"] == 1
        assert stats["new_tool"]["total_tokens"] == 10
        assert stats["legacy_tool"]["call_count"] == 1
        assert stats["legacy_tool"]["total_tokens"] == 3

    def test_snapshot_and_clear(self):
        buf = SessionBuffer(session_id="s1")
        buf.add("user", "hello")
        buf.add("assistant", "world")
        snap = buf.snapshot_and_clear()
        assert len(snap) == 2
        assert len(buf.messages) == 0
        assert buf.pending_tokens == 0
        assert buf.meta.message_count == 0

    def test_remove_messages_by_id_adjusts_watermark(self):
        buf = SessionBuffer(session_id="s1")
        first = buf.add("user", "first")
        second = buf.add("user", "second")
        third = buf.add("user", "third")
        buf.extraction_watermark = 3

        removed = buf.remove_messages_by_id({second.id})

        assert removed == 1
        assert [m.id for m in buf.messages] == [first.id, third.id]
        assert buf.extraction_watermark == 2

    def test_remove_messages_by_id_recomputes_turn_count(self):
        buf = SessionBuffer(session_id="s1")
        archived = [buf.add("user", f"archived {idx}") for idx in range(10)]

        assert buf.should_compress()

        removed = buf.remove_messages_by_id({msg.id for msg in archived})

        assert removed == 10
        assert len(buf.messages) == 0
        assert buf.turn_count == 0
        assert not buf.should_compress()

        buf.add("user", "new message")

        assert buf.turn_count == 1
        assert not buf.should_compress()


# ---------------------------------------------------------------------------
# SessionManager tests
# ---------------------------------------------------------------------------


class TestSessionManager:
    def test_get_or_create(self, mgr, ctx):
        buf = mgr.get_or_create("s1")
        assert isinstance(buf, SessionBuffer)
        assert buf.session_id == "s1"

    def test_get_or_create_idempotent(self, mgr, ctx):
        buf1 = mgr.get_or_create("s1")
        buf1.add("user", "hello")
        buf2 = mgr.get_or_create("s1")
        assert len(buf2.messages) == 1  # Same buffer

    def test_add_message(self, mgr, ctx):
        result = mgr.add_message("s1", "user", "hello", ctx)
        assert result["ok"] is True
        assert "message_id" in result
        assert result["pending_tokens"] > 0

    def test_add_message_increments_tokens(self, mgr, ctx):
        mgr.add_message("s1", "user", "a" * 100, ctx)
        r1 = mgr.get_session("s1", ctx)
        mgr.add_message("s1", "assistant", "b" * 200, ctx)
        r2 = mgr.get_session("s1", ctx)
        assert r2["pending_tokens"] > r1["pending_tokens"]

    def test_get_session(self, mgr, ctx):
        mgr.add_message("s1", "user", "hello", ctx)
        result = mgr.get_session("s1", ctx)
        assert result["ok"] is True
        assert result["session_id"] == "s1"
        assert result["message_count"] == 1
        assert result["pending_tokens"] > 0

    def test_commit_empty_buffer(self, mgr, ctx):
        result = mgr.commit("s1", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is False
        assert result["reason"] == "empty_buffer"

    def test_commit_sync(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello world", ctx)
        mgr.add_message("s1", "assistant", "hi there", ctx)
        result = mgr.commit("s1", ctx, wait=True)
        assert result["ok"] is True
        assert result["archived"] is True
        assert "archive_id" in result
        assert result["status"] == "completed"

        # Buffer should be cleared
        session = mgr.get_session("s1", ctx)
        assert session["pending_tokens"] == 0

    def test_commit_async(self, mgr_with_deps, ctx):
        import time
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello world", ctx)
        result = mgr.commit("s1", ctx, wait=False)
        assert result["ok"] is True
        assert result["status"] == "processing"
        assert "task_id" in result

        # Wait for background thread
        time.sleep(0.5)

        # Check task completed
        task = mgr.get_task(result["task_id"])
        assert task is not None
        assert task["status"] == "completed"

    def test_commit_in_progress_rejects_second(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello", ctx)

        # Manually set commit_in_progress
        buf = mgr.get_or_create("s1")
        buf.commit_in_progress = True

        result = mgr.commit("s1", ctx, wait=True)
        assert result["reason"] == "commit_in_progress"

    def test_commit_updates_meta(self, mgr_with_deps, ctx):
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "hello", ctx)
        mgr.commit("s1", ctx, wait=True)

        session = mgr.get_session("s1", ctx)
        assert session["commit_count"] == 1
        assert session["last_commit_at"] != ""

    def test_commit_snapshot_does_not_clear_live_buffer(self, mgr_with_deps, ctx):
        from session.session_manager import generate_archive_id
        mgr = mgr_with_deps
        mgr.add_message("s1", "user", "first", ctx)
        mgr.add_message("s1", "user", "second", ctx)
        buf = mgr.get_or_create("s1")
        snapshot = [buf.messages[0]]

        result = mgr.commit_snapshot("s1", snapshot, ctx, wait=True, archive_id=generate_archive_id())

        assert result["ok"] is True
        assert result["archived"] is True
        assert [m.content for m in buf.messages] == ["first", "second"]

    def test_get_context_empty_session(self, mgr, ctx):
        result = mgr.get_context("s1", 128_000, ctx)
        assert result["ok"] is True
        assert result["pending_tokens"] == 0
        assert result["archive_count"] == 0
        assert result["active_message_count"] == 0

    def test_get_context_with_active_messages(self, mgr, ctx):
        mgr.add_message("s1", "user", "a" * 100, ctx)
        result = mgr.get_context("s1", 128_000, ctx)
        assert result["active_message_count"] == 1
        assert result["pending_tokens"] > 0

    def test_remove_session(self, mgr, ctx):
        mgr.add_message("s1", "user", "hello", ctx)
        mgr.remove_session("s1")
        # get_or_create should create fresh
        buf = mgr.get_or_create("s1")
        assert len(buf.messages) == 0

    def test_get_task_unknown(self, mgr):
        assert mgr.get_task("nonexistent") is None

    def test_commit_fails_when_required_archive_store_unavailable(self, ctx):
        mock_llm = Mock()
        mock_write_api = Mock()
        mgr = SessionManager(
            get_llm=lambda: mock_llm,
            get_write_api=lambda: mock_write_api,
            get_archive_store=lambda: None,
            archive_store_required=True,
        )

        mgr.add_message("s1", "user", "hello world", ctx)
        result = mgr.commit("s1", ctx, wait=True)

        assert result["ok"] is True
        assert result["archived"] is False
        assert result["status"] == "failed"
        assert "archive store unavailable" in result["error"]
        mock_write_api.commit_session.assert_not_called()

    def test_get_latest_archive_context_prefers_archive_store(self, ctx):
        sql_store = Mock()
        sql_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-old",
                session_id="s1",
                overview="old overview",
                abstract="old abstract",
                messages=[],
                created_at="2026-01-01T00:00:00+00:00",
            ),
            ArchiveEntry(
                archive_id="a-new",
                session_id="s1",
                overview="new overview",
                abstract="new abstract",
                messages=[],
                created_at="2026-01-02T00:00:00+00:00",
            ),
        ]
        mgr = SessionManager(
            get_archive_store=lambda: sql_store,
            get_agfs=lambda: Mock(),
        )

        with patch("session.archive_store.SessionArchiveStore") as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "new overview"
        assert abstract == "new abstract"
        sql_store.list_archives.assert_called_once_with("s1", ctx)
        agfs_store_cls.assert_not_called()

    def test_get_latest_archive_context_falls_back_to_agfs_when_store_missing(self, ctx):
        agfs_store = Mock()
        agfs_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-only",
                session_id="s1",
                overview="agfs overview",
                abstract="agfs abstract",
                messages=[],
                created_at="2026-01-03T00:00:00+00:00",
            ),
        ]
        agfs_fs = Mock()
        mgr = SessionManager(
            get_archive_store=lambda: None,
            get_agfs=lambda: agfs_fs,
        )

        with patch("session.archive_store.SessionArchiveStore", return_value=agfs_store) as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "agfs overview"
        assert abstract == "agfs abstract"
        agfs_store_cls.assert_called_once_with(fs=agfs_fs)
        agfs_store.list_archives.assert_called_once_with("s1", ctx)

    def test_get_latest_archive_context_falls_back_to_agfs_when_store_errors(self, ctx):
        broken_store = Mock()
        broken_store.list_archives.side_effect = RuntimeError("db unavailable")
        agfs_store = Mock()
        agfs_store.list_archives.return_value = [
            ArchiveEntry(
                archive_id="a-fallback",
                session_id="s1",
                overview="fallback overview",
                abstract="fallback abstract",
                messages=[],
                created_at="2026-01-04T00:00:00+00:00",
            ),
        ]
        agfs_fs = Mock()
        mgr = SessionManager(
            get_archive_store=lambda: broken_store,
            get_agfs=lambda: agfs_fs,
        )

        with patch("session.archive_store.SessionArchiveStore", return_value=agfs_store) as agfs_store_cls:
            overview, abstract = mgr._get_latest_archive_context("s1", ctx)

        assert overview == "fallback overview"
        assert abstract == "fallback abstract"
        broken_store.list_archives.assert_called_once_with("s1", ctx)
        agfs_store_cls.assert_called_once_with(fs=agfs_fs)


# ---------------------------------------------------------------------------
# SessionMessage model tests
# ---------------------------------------------------------------------------


class TestSessionMessage:
    def test_estimated_tokens(self):
        msg = SessionMessage(id="m1", role="user", content="a" * 100)
        assert msg.estimated_tokens == 25

    def test_estimated_tokens_minimum(self):
        msg = SessionMessage(id="m1", role="user", content="")
        assert msg.estimated_tokens == 1  # max(1, 0)


# ---------------------------------------------------------------------------
# SessionMeta model tests
# ---------------------------------------------------------------------------


class TestSessionMeta:
    def test_defaults(self):
        meta = SessionMeta()
        assert meta.session_id == ""
        assert meta.message_count == 0
        assert meta.commit_count == 0

    def test_with_session_id(self):
        meta = SessionMeta(session_id="s1")
        assert meta.session_id == "s1"
