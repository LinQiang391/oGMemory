from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from providers import unified_config
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


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
    assert write_api.commit_session.call_args.kwargs["messages"] == [
        {"role": "user", "content": "second user fact"},
    ]
    assert write_api.commit_session.call_args.kwargs["session_summary"] == "existing-summary"
    assert write_api.commit_session.call_args.kwargs["session_time"].isoformat() == "2026-04-19T01:02:03+00:00"
    assert "search_web" in write_api.commit_session.call_args.kwargs["tool_stats_text"]
    assert buf.extraction_watermark == 3
    assert "task-1" in buf.extraction_summary


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
