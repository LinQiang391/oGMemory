from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from core.models import ArchiveRef, RequestContext, Role, SeedHit
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.models import SessionWindowState
from session.session_state import Commitment, TaskState
from session.topic_buffer import SlotContent
from session.topic_detector import TopicDetection


def _ctx(session_id: str = "sess-archive") -> RequestContext:
    return RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id=session_id,
        trace_id="trace-1",
    )


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _disable_session_persistence(service: MemoryService) -> None:
    service._get_context_fs = Mock(return_value=None)


class _StructuredStateLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, schema):
        self.calls += 1
        return {
            "active_task": "Stabilize compact mechanism",
            "confirmed_constraints": ["Use unit tests"],
            "recent_decisions": ["Enable rolling after tests"],
            "open_loops": ["Add prepare token TTL"],
            "uncertainties": ["SessionState persistence URI"],
            "summary": "The user is stabilizing compact behavior.",
        }


class _PrefetchHit:
    def __init__(self, uri, score, category, abstract, overview="", content_excerpt=""):
        self.uri = uri
        self.score = score
        self.category = category
        self.abstract = abstract
        self.overview = overview
        self.content_excerpt = content_excerpt


class _PrefetchResult:
    def __init__(self, hits):
        self.hits = hits
        self.trace = None


def test_get_session_state_runs_rolling_compressor_when_gate_enabled():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    ctx = _ctx("sess-rolling")
    llm = _StructuredStateLLM()
    service.get_llm = Mock(return_value=llm)

    mgr = service.get_session_manager()
    for idx in range(10):
        mgr.add_message("sess-rolling", "user", f"buffered request {idx}", ctx)

    state = service._get_session_state("sess-rolling", ctx)

    assert llm.calls == 1
    assert state.active_task == "Stabilize compact mechanism"
    assert state.open_loops == ["Add prepare token TTL"]
    assert state.compressed_text == "The user is stabilizing compact behavior."
    assert state.turn_count_at_last_compress == 10
    assert state.token_count_at_last_compress > 0


def test_prefetch_populates_topic_buffer_pending_injection():
    service = MemoryService(config=OgMemConfig(prefetch_enabled=True, prefetch_top_k=2))
    ctx = _ctx("sess-prefetch")
    read_api = Mock()
    read_api.search_memory.return_value = _PrefetchResult([
        _PrefetchHit(
            "ctx://acct-test/users/user-test/memories/preferences/coding_style",
            0.91,
            "preference",
            "Use terse engineering prose",
        ),
        _PrefetchHit(
            "ctx://acct-test/users/user-test/memories/profile",
            0.88,
            "profile",
            "User profile summary",
        ),
    ])
    service.get_read_api = Mock(return_value=read_api)

    result = service.prefetch({
        "sessionId": "sess-prefetch",
        "_ctx": ctx,
        "messages": [_msg("user", "How should I write this patch?")],
    })

    assert result == {"ok": True, "prefetched": 2}
    topic = service.get_session_manager().get_topic_buffer("sess-prefetch")
    pending = topic.get_pending_injection()
    assert [hit.uri for hit in pending] == [
        "ctx://acct-test/users/user-test/memories/preferences/coding_style",
        "ctx://acct-test/users/user-test/memories/profile",
    ]
    assert pending[0].abstract == "Use terse engineering prose"
    read_api.search_memory.assert_called_once()


def test_prefetch_noops_when_disabled():
    service = MemoryService(config=OgMemConfig(prefetch_enabled=False))
    ctx = _ctx("sess-prefetch-off")
    service.get_read_api = Mock(side_effect=AssertionError("prefetch should be disabled"))

    result = service.prefetch({
        "sessionId": "sess-prefetch-off",
        "_ctx": ctx,
        "messages": [_msg("user", "debug this")],
    })

    assert result == {"ok": True, "prefetched": 0, "reason": "disabled"}


def test_session_working_set_service_method_lists_access_stats():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-ws")
    buf = service.get_session_manager().get_or_create("sess-ws", ctx)
    buf.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"

    result = service.session_working_set({})

    assert result["ok"] is True
    assert result["sessions"][0]["session_id"] == "sess-ws"
    assert result["sessions"][0]["last_accessed_at"] == "2026-05-20T02:00:00+00:00"


def test_session_working_set_requires_admin_when_role_control_enabled():
    service = MemoryService(config=OgMemConfig(role_control_enabled=True, root_api_key="root-key"))
    ctx = RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="sess-ws",
        trace_id="trace-1",
        role=Role.MEMBER,
    )

    with pytest.raises(Exception, match="Operation requires role"):
        service.session_working_set({"_ctx": ctx})


def test_session_working_set_requires_authenticated_context_when_role_control_enabled():
    service = MemoryService(config=OgMemConfig(role_control_enabled=True, root_api_key="root-key"))

    with pytest.raises(PermissionError, match="authenticated context required"):
        service.session_working_set({})


def test_evict_idle_sessions_service_method_uses_last_accessed_at():
    service = MemoryService(config=OgMemConfig())
    mgr = service.get_session_manager()
    ctx = _ctx("active")
    active = mgr.get_or_create("active", ctx)
    idle = mgr.get_or_create("idle", ctx)
    active.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"

    result = service.evict_idle_sessions({
        "maxIdleSeconds": 1800,
        "nowIso": "2026-05-20T02:00:01+00:00",
    })

    assert result == {"ok": True, "evicted": ["idle"]}
    assert mgr.has_session("active") is True
    assert mgr.has_session("idle") is False


def test_evict_idle_sessions_requires_admin_when_role_control_enabled():
    service = MemoryService(config=OgMemConfig(role_control_enabled=True, root_api_key="root-key"))
    ctx = RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="idle",
        trace_id="trace-1",
        role=Role.MEMBER,
    )

    with pytest.raises(Exception, match="Operation requires role"):
        service.evict_idle_sessions({
            "_ctx": ctx,
            "maxIdleSeconds": 1,
            "nowIso": "2026-05-20T02:00:01+00:00",
        })


def test_evict_idle_sessions_requires_authenticated_context_when_role_control_enabled():
    service = MemoryService(config=OgMemConfig(role_control_enabled=True, root_api_key="root-key"))

    with pytest.raises(PermissionError, match="authenticated context required"):
        service.evict_idle_sessions({"maxIdleSeconds": 1})


def test_get_session_state_passes_durable_session_state_into_rolling_compressor():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    ctx = _ctx("sess-bridge")
    llm = _StructuredStateLLM()
    service.get_llm = Mock(return_value=llm)
    mgr = service.get_session_manager()
    mgr.get_session_state().update_task_state(
        "sess-bridge",
        TaskState(objective="Bridge objective"),
    )
    mgr.get_session_state().add_commitment(
        "sess-bridge",
        Commitment(content="Bridge loop"),
    )
    for idx in range(10):
        mgr.add_message("sess-bridge", "user", f"msg {idx}", ctx)

    state = service._get_session_state("sess-bridge", ctx)

    assert state.active_task == "Bridge objective"
    assert state.open_loops == ["Bridge loop"]


def test_get_session_state_binds_identity_for_fresh_session():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-fresh-identity")

    service._get_session_state("sess-fresh-identity", ctx)

    buf = service.get_session_manager().get_or_create("sess-fresh-identity")
    assert buf.meta.account_id == "acct-test"
    assert buf.meta.user_id == "user-test"
    assert buf.meta.agent_id == "agent-test"


def test_get_session_state_updates_last_accessed_at_on_each_access():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=False))
    ctx = _ctx("sess-access")

    first = service._get_session_state("sess-access", ctx)
    first_accessed_at = first.last_accessed_at
    second = service._get_session_state("sess-access", ctx)
    second_accessed_at = second.last_accessed_at

    assert first_accessed_at
    assert second_accessed_at
    assert second_accessed_at >= first_accessed_at


def test_get_session_state_syncs_durable_state_before_rolling_threshold():
    service = MemoryService(
        config=OgMemConfig(
            rolling_compress_enabled=True,
            session_state_bridge_enabled=True,
            session_state_sync_interval_turns=1,
        )
    )
    ctx = _ctx("sess-sync")
    mgr = service.get_session_manager()
    mgr.get_session_state().update_task_state(
        "sess-sync",
        TaskState(objective="Sync without rolling"),
    )
    mgr.get_session_state().add_commitment(
        "sess-sync",
        Commitment(content="No rolling threshold yet"),
    )
    mgr.add_message("sess-sync", "user", "only one turn", ctx)
    service.get_llm = Mock(side_effect=AssertionError("rolling compression should not run"))

    state = service._get_session_state("sess-sync", ctx)

    assert state.active_task == "Sync without rolling"
    assert state.open_loops == ["No rolling threshold yet"]
    assert state.turn_count_at_last_compress == 0


def test_get_session_state_respects_bridge_sync_interval_turns():
    service = MemoryService(
        config=OgMemConfig(
            rolling_compress_enabled=False,
            session_state_bridge_enabled=True,
            session_state_sync_interval_turns=3,
        )
    )
    ctx = _ctx("sess-sync-interval")
    mgr = service.get_session_manager()
    mgr.get_session_state().update_task_state(
        "sess-sync-interval",
        TaskState(objective="Initial objective"),
    )
    mgr.add_message("sess-sync-interval", "user", "turn one", ctx)
    first = service._get_session_state("sess-sync-interval", ctx)
    assert first.active_task == "Initial objective"

    mgr.get_session_state().update_task_state(
        "sess-sync-interval",
        TaskState(objective="Changed too soon"),
    )
    mgr.add_message("sess-sync-interval", "user", "turn two", ctx)
    second = service._get_session_state("sess-sync-interval", ctx)
    assert second.active_task == "Initial objective"

    mgr.add_message("sess-sync-interval", "user", "turn three", ctx)
    third = service._get_session_state("sess-sync-interval", ctx)
    assert third.active_task == "Initial objective"

    mgr.add_message("sess-sync-interval", "user", "turn four", ctx)
    fourth = service._get_session_state("sess-sync-interval", ctx)
    assert fourth.active_task == "Changed too soon"
    assert fourth.session_state_sync_turn_count == 4


def test_get_session_state_respects_bridge_disabled_during_rolling_compression():
    service = MemoryService(
        config=OgMemConfig(
            rolling_compress_enabled=True,
            session_state_bridge_enabled=False,
        )
    )
    ctx = _ctx("sess-bridge-disabled")
    llm = _StructuredStateLLM()
    service.get_llm = Mock(return_value=llm)
    mgr = service.get_session_manager()
    mgr.get_session_state().update_task_state(
        "sess-bridge-disabled",
        TaskState(objective="Durable task should not override"),
    )
    for idx in range(10):
        mgr.add_message("sess-bridge-disabled", "user", f"buffered request {idx}", ctx)

    state = service._get_session_state("sess-bridge-disabled", ctx)

    assert llm.calls == 1
    assert state.active_task == "Stabilize compact mechanism"
    assert state.session_state_version == 0


def test_session_state_bridge_disabled_does_not_load_context_fs_on_session_create():
    service = MemoryService(config=OgMemConfig(session_state_bridge_enabled=False))
    ctx = _ctx("sess-no-bridge-load")
    service._get_context_fs = Mock(side_effect=AssertionError("ContextFS should not load"))

    mgr = service.get_session_manager()
    mgr.add_message("sess-no-bridge-load", "user", "short write", ctx)

    service._get_context_fs.assert_not_called()


def test_get_session_state_uses_rolling_fallback_when_llm_missing_and_enabled():
    service = MemoryService(
        config=OgMemConfig(
            rolling_compress_enabled=True,
            rolling_compress_fallback_enabled=True,
        )
    )
    ctx = _ctx("sess-fallback")
    service.get_llm = Mock(return_value=None)
    mgr = service.get_session_manager()
    for idx in range(10):
        mgr.add_message("sess-fallback", "user", f"fallback request {idx}", ctx)

    state = service._get_session_state("sess-fallback", ctx)

    assert state.compressed_text
    assert state.turn_count_at_last_compress == 10


def test_compose_injects_rolling_session_context_and_trims_compressed_messages():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-rolling")
    llm = _StructuredStateLLM()
    service.get_llm = Mock(return_value=llm)
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))

    mgr = service.get_session_manager()
    buffered_messages = []
    for idx in range(10):
        content = f"buffered request {idx}"
        mgr.add_message("sess-compose-rolling", "user", content, ctx)
        buffered_messages.append(_msg("user", content))

    result = service.compose({
        "sessionId": "sess-compose-rolling",
        "_ctx": ctx,
        "messages": buffered_messages + [_msg("user", "current question")],
    })

    assert "## Active Task\nStabilize compact mechanism" in result["sessionContext"]
    assert "## Open Loops\n- Add prepare token TTL" in result["sessionContext"]
    assert (
        "## Recent Session Summary\nThe user is stabilizing compact behavior."
        in result["sessionContext"]
    )
    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["current question"]


def test_compose_trims_all_conversation_messages_when_rolling_compressed_count_covers_window():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-rolling-covered")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    mgr = service.get_session_manager()
    caller_messages = []
    for idx in range(1, 11):
        content = f"already compressed user {idx}"
        mgr.add_message(
            "sess-compose-rolling-covered",
            "user",
            content,
            ctx,
        )
        caller_messages.append(_msg("user", content))
    buf = mgr.get_or_create("sess-compose-rolling-covered")

    window = SessionWindowState(
        compressed_text="All visible caller conversation messages are already summarized.",
        turn_count_at_last_compress=10,
        token_count_at_last_compress=sum(m.estimated_tokens for m in buf.messages),
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-rolling-covered",
        "_ctx": ctx,
        "prePromptMessageCount": 1,
        "messages": [_msg("system", "caller pre-prompt")] + caller_messages,
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["caller pre-prompt"]
    assert (
        "All visible caller conversation messages are already summarized."
        in result["sessionContext"]
    )


def test_compose_trims_verified_compressed_message_span_for_interleaved_history():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-interleaved")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    mgr = service.get_session_manager()
    caller_messages = []
    for idx in range(10):
        user_content = f"interleaved user {idx}"
        assistant_content = f"interleaved assistant {idx}"
        mgr.add_message("sess-compose-interleaved", "user", user_content, ctx)
        mgr.add_message(
            "sess-compose-interleaved",
            "assistant",
            assistant_content,
            ctx,
        )
        caller_messages.append(_msg("user", user_content))
        caller_messages.append(_msg("assistant", assistant_content))
    buf = mgr.get_or_create("sess-compose-interleaved")

    window = SessionWindowState(
        compressed_text="Interleaved history is summarized.",
        turn_count_at_last_compress=10,
        token_count_at_last_compress=sum(m.estimated_tokens for m in buf.messages),
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-interleaved",
        "_ctx": ctx,
        "messages": caller_messages + [_msg("user", "current question")],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["current question"]
    assert "Interleaved history is summarized." in result["sessionContext"]


def test_compose_preserves_short_current_message_that_repeats_compressed_prefix():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-short-repeat")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    mgr = service.get_session_manager()
    for idx in range(10):
        mgr.add_message(
            "sess-compose-short-repeat",
            "user",
            "repeated current" if idx == 0 else f"compressed user {idx}",
            ctx,
        )
        mgr.add_message(
            "sess-compose-short-repeat",
            "assistant",
            f"compressed assistant {idx}",
            ctx,
        )
    buf = mgr.get_or_create("sess-compose-short-repeat")

    window = SessionWindowState(
        compressed_text="Compressed history starts with the repeated current text.",
        turn_count_at_last_compress=10,
        token_count_at_last_compress=sum(m.estimated_tokens for m in buf.messages),
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-short-repeat",
        "_ctx": ctx,
        "messages": [_msg("user", "repeated current")],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["repeated current"]
    assert (
        "Compressed history starts with the repeated current text."
        in result["sessionContext"]
    )


def test_compose_preserves_current_message_when_compressed_state_has_empty_runtime_buffer():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-empty-buffer")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))

    window = SessionWindowState(
        compressed_text="Persisted compressed session state from before restart.",
        turn_count_at_last_compress=10,
        token_count_at_last_compress=100,
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-empty-buffer",
        "_ctx": ctx,
        "messages": [_msg("user", "current after restart")],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["current after restart"]
    assert "Persisted compressed session state from before restart." in result["sessionContext"]


def test_compose_preserves_current_message_when_runtime_buffer_does_not_cover_compressed_window():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-restored-short-buffer")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    mgr = service.get_session_manager()
    mgr.add_message(
        "sess-compose-restored-short-buffer",
        "user",
        "new current after restored state",
        ctx,
    )

    window = SessionWindowState(
        compressed_text="Persisted compressed session state from before restart.",
        turn_count_at_last_compress=10,
        token_count_at_last_compress=100,
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-restored-short-buffer",
        "_ctx": ctx,
        "messages": [_msg("user", "new current after restored state")],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "new current after restored state"
    ]
    assert "Persisted compressed session state from before restart." in result["sessionContext"]


def test_compose_trims_covered_prefix_even_when_messages_repeat_later_in_buffer():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-compose-repeated-prefix")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    mgr = service.get_session_manager()
    mgr.add_message("sess-compose-repeated-prefix", "user", "repeat user", ctx)
    mgr.add_message("sess-compose-repeated-prefix", "assistant", "repeat assistant", ctx)
    for idx in range(2, 10):
        mgr.add_message(
            "sess-compose-repeated-prefix",
            "user",
            f"compressed filler user {idx}",
            ctx,
        )
    mgr.add_message("sess-compose-repeated-prefix", "user", "repeat user", ctx)
    mgr.add_message("sess-compose-repeated-prefix", "assistant", "repeat assistant", ctx)
    buf = mgr.get_or_create("sess-compose-repeated-prefix")
    repeated_prefix = buf.messages[:2]

    window = SessionWindowState(
        compressed_text="Repeated prefix is covered by compressed history.",
        turn_count_at_last_compress=1,
        token_count_at_last_compress=sum(m.estimated_tokens for m in repeated_prefix),
    )
    service._get_session_state = Mock(return_value=window)

    result = service.compose({
        "sessionId": "sess-compose-repeated-prefix",
        "_ctx": ctx,
        "prePromptMessageCount": 1,
        "messages": [
            _msg("system", "caller pre-prompt"),
            _msg("user", "repeat user"),
            _msg("assistant", "repeat assistant"),
        ],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["caller pre-prompt"]
    assert "Repeated prefix is covered by compressed history." in result["sessionContext"]


def test_compose_does_not_drop_current_short_window_after_rolling_compression():
    service = MemoryService(config=OgMemConfig(rolling_compress_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-short-window")
    service.get_llm = Mock(return_value=_StructuredStateLLM())
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))

    mgr = service.get_session_manager()
    for idx in range(10):
        mgr.add_message("sess-short-window", "user", f"old buffered {idx}", ctx)

    service.compose({
        "sessionId": "sess-short-window",
        "_ctx": ctx,
        "messages": [_msg("user", f"old buffered {idx}") for idx in range(10)]
        + [_msg("user", "first current")],
    })

    result = service.compose({
        "sessionId": "sess-short-window",
        "_ctx": ctx,
        "messages": [_msg("user", "new current question")],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == ["new current question"]


def test_compose_reuses_cached_identity_archive_and_skills_slots():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-topic")
    mgr = service.get_session_manager()
    topic = mgr.get_topic_buffer("sess-topic")
    topic.set_cached_slot(
        "identity",
        SlotContent(
            content="cached identity",
            uris=["ctx://acct-test/users/user-test/memories/profile"],
            tokens=12,
        ),
    )
    topic.set_cached_slot(
        "archive_history",
        SlotContent(content="cached archive history", tokens=20),
    )
    buf = mgr.get_or_create("sess-topic", ctx)
    buf.window_state.skills_text = "cached skills summary"

    service._read_profile = Mock(side_effect=AssertionError("profile should be cached"))
    service._collect_archives = Mock(side_effect=AssertionError("archives should be cached"))
    service._search_working_set = Mock(return_value=([], [], {}))

    result = service.compose({
        "sessionId": "sess-topic",
        "_ctx": ctx,
        "messages": [_msg("user", "current question")],
    })

    assert "cached identity" in result["identityContext"]
    assert "cached archive history" in result["episodicContext"]
    assert "## Skills Summary\ncached skills summary" in result["sessionContext"]


def test_compose_merges_pending_injection_into_working_set():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-topic")
    mgr = service.get_session_manager()
    topic = mgr.get_topic_buffer("sess-topic")
    topic.set_pending_injection([
        SeedHit(
            uri="ctx://acct-test/users/user-test/memories/preferences/coding_style",
            score=0.91,
            level=0,
            category="preference",
            abstract="prefetched memory",
        )
    ])

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))

    result = service.compose({
        "sessionId": "sess-topic",
        "_ctx": ctx,
        "messages": [_msg("user", "current question")],
    })

    assert "prefetched memory" in result["retrievedEvidence"]
    assert topic.get_pending_injection() is None


def test_compose_skips_pending_injection_uri_already_in_working_set():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-topic-dedupe")
    mgr = service.get_session_manager()
    topic = mgr.get_topic_buffer("sess-topic-dedupe")
    duplicate_uri = "ctx://acct-test/users/user-test/memories/preferences/coding_style"
    topic.set_pending_injection([
        SeedHit(
            uri=duplicate_uri,
            score=0.91,
            level=0,
            category="preference",
            abstract="prefetched duplicate",
        )
    ])

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([
        {
            "uri": duplicate_uri,
            "category": "preference",
            "abstract": "existing memory",
            "overview": "",
            "content": "",
            "score": 0.95,
        }
    ], [], {}))

    result = service.compose({
        "sessionId": "sess-topic-dedupe",
        "_ctx": ctx,
        "messages": [_msg("user", "current question")],
    })

    assert result["retrievedEvidence"].count("existing memory") == 1
    assert "prefetched duplicate" not in result["retrievedEvidence"]
    assert topic.get_pending_injection() is None


def test_compose_allows_prefetch_reinjection_after_topic_change():
    service = MemoryService(config=OgMemConfig(topic_detection_enabled=True))
    _disable_session_persistence(service)
    ctx = _ctx("sess-topic-reinject")
    mgr = service.get_session_manager()
    topic = mgr.get_topic_buffer("sess-topic-reinject")
    reused_uri = "ctx://acct-test/users/user-test/memories/preferences/coding_style"
    topic.update_topic(TopicDetection(label="planning", confidence=0.9, message_count=1))
    topic.mark_injected([reused_uri])
    topic.set_pending_injection([
        SeedHit(
            uri=reused_uri,
            score=0.91,
            level=0,
            category="preference",
            abstract="prefetched memory for the new topic",
        )
    ])

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service._get_shared_embedder = Mock(return_value=None)

    result = service.compose({
        "sessionId": "sess-topic-reinject",
        "_ctx": ctx,
        "messages": [_msg("user", "Please debug this failing pytest traceback")],
    })

    assert "prefetched memory for the new topic" in result["retrievedEvidence"]
    assert topic.get_current_topic().label == "debugging"
    assert topic.get_pending_injection() is None


def test_compose_updates_session_topic_buffer_from_current_messages():
    service = MemoryService(config=OgMemConfig(topic_detection_enabled=True))
    ctx = _ctx("sess-topic-detect")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service._get_shared_embedder = Mock(return_value=None)

    result = service.compose({
        "sessionId": "sess-topic-detect",
        "_ctx": ctx,
        "messages": [_msg("user", "Please debug this failing pytest traceback")],
    })

    topic = service.get_session_manager().get_topic_buffer(
        "sess-topic-detect"
    ).get_current_topic()
    assert result["estimatedTokens"] > 0
    assert topic.label == "debugging"
    assert topic.confidence > 0


def test_compose_skips_topic_detection_when_gate_disabled():
    service = MemoryService(config=OgMemConfig(topic_detection_enabled=False))
    ctx = _ctx("sess-topic-off")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service._get_shared_embedder = Mock(side_effect=AssertionError("topic detection disabled"))

    service.compose({
        "sessionId": "sess-topic-off",
        "_ctx": ctx,
        "messages": [_msg("user", "Please debug this traceback")],
    })

    topic = service.get_session_manager().get_topic_buffer(
        "sess-topic-off"
    ).get_current_topic()
    assert topic is None


def test_compose_runs_topic_detection_when_gate_enabled():
    service = MemoryService(config=OgMemConfig(topic_detection_enabled=True))
    ctx = _ctx("sess-topic-on")
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._search_working_set = Mock(return_value=([], [], {}))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service._get_shared_embedder = Mock(return_value=None)

    service.compose({
        "sessionId": "sess-topic-on",
        "_ctx": ctx,
        "messages": [_msg("user", "Please debug this traceback")],
    })

    topic = service.get_session_manager().get_topic_buffer(
        "sess-topic-on"
    ).get_current_topic()
    assert topic.label == "debugging"


def test_compose_trims_archived_messages_when_pre_prompt_count_defaults_to_zero():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx()

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([
        ArchiveRef(
            archive_id="archive-1",
            archive_uri="archive://sess-archive/archive-1",
            abstract="old conversation summary",
            overview="old conversation overview",
        )
    ], []))
    service._search_working_set = Mock(return_value=[])
    service._get_session_state = Mock(return_value=SessionWindowState())

    mgr = service.get_session_manager()
    mgr.add_message("sess-archive", "user", "live user", ctx)
    mgr.add_message("sess-archive", "assistant", "live assistant", ctx)

    result = service.compose({
        "sessionId": "sess-archive",
        "_ctx": ctx,
        "messages": [
            _msg("user", "archived user 1"),
            _msg("assistant", "archived assistant 1"),
            _msg("user", "archived user 2"),
            _msg("assistant", "archived assistant 2"),
            _msg("user", "live user"),
            _msg("assistant", "live assistant"),
            _msg("user", "current user asks question"),
        ],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "live user",
        "live assistant",
        "current user asks question",
    ]
    assert result["archiveIncluded"] is True
    assert result["archiveCount"] == 1
    assert "## Archive History" in result["messages"][0]["content"]


def test_compose_preserves_pre_prompt_messages_when_trimming_archived_history():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-prefix")

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([
        ArchiveRef(
            archive_id="archive-1",
            archive_uri="archive://sess-prefix/archive-1",
            abstract="old conversation summary",
            overview="old conversation overview",
        )
    ], []))
    service._search_working_set = Mock(return_value=[])
    service._get_session_state = Mock(return_value=SessionWindowState())

    mgr = service.get_session_manager()
    mgr.add_message("sess-prefix", "user", "live user", ctx)
    mgr.add_message("sess-prefix", "assistant", "live assistant", ctx)

    result = service.compose({
        "sessionId": "sess-prefix",
        "_ctx": ctx,
        "prePromptMessageCount": 1,
        "messages": [
            _msg("system", "caller pre-prompt"),
            _msg("user", "archived user"),
            _msg("assistant", "archived assistant"),
            _msg("user", "live user"),
            _msg("assistant", "live assistant"),
            _msg("user", "current user asks question"),
        ],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "caller pre-prompt",
        "live user",
        "live assistant",
        "current user asks question",
    ]


def test_compose_archive_trim_requires_contiguous_buffer_match():
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-contiguous")

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([
        ArchiveRef(
            archive_id="archive-1",
            archive_uri="archive://sess-contiguous/archive-1",
            abstract="old conversation summary",
            overview="old conversation overview",
        )
    ], []))
    service._search_working_set = Mock(return_value=[])
    service._get_session_state = Mock(return_value=SessionWindowState())

    mgr = service.get_session_manager()
    mgr.add_message("sess-contiguous", "user", "repeated live user", ctx)
    mgr.add_message("sess-contiguous", "assistant", "repeated live assistant", ctx)

    result = service.compose({
        "sessionId": "sess-contiguous",
        "_ctx": ctx,
        "messages": [
            _msg("user", "archived user"),
            _msg("user", "repeated live user"),
            _msg("assistant", "repeated live assistant"),
            _msg("user", "repeated live user"),
            _msg("user", "current turn inserted between repeats"),
            _msg("assistant", "repeated live assistant"),
            _msg("user", "current user asks question"),
        ],
    })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "repeated live user",
        "repeated live assistant",
        "repeated live user",
        "current turn inserted between repeats",
        "repeated live assistant",
        "current user asks question",
    ]


def test_compose_warns_when_tail_margin_fallback_has_no_pre_prompt_count(caplog):
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-tail-margin")

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([
        ArchiveRef(
            archive_id="archive-1",
            archive_uri="archive://sess-tail-margin/archive-1",
            abstract="old conversation summary",
            overview="old conversation overview",
        )
    ], []))
    service._search_working_set = Mock(return_value=[])
    service._get_session_state = Mock(return_value=SessionWindowState())

    mgr = service.get_session_manager()
    mgr.add_message("sess-tail-margin", "user", "buffer message missing from caller", ctx)

    with caplog.at_level(logging.WARNING, logger="ogmem.service"):
        result = service.compose({
            "sessionId": "sess-tail-margin",
            "_ctx": ctx,
            "messages": [
                _msg("user", "archived user"),
                _msg("assistant", "archived assistant"),
                _msg("user", "current user asks question"),
                _msg("assistant", "current assistant answer"),
            ],
        })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "archived assistant",
        "current user asks question",
        "current assistant answer",
    ]
    assert "compose archive trim using tail margin without prePromptMessageCount" in caplog.text


def test_compose_preserves_messages_when_archive_exists_but_buffer_is_empty(caplog):
    service = MemoryService(config=OgMemConfig())
    ctx = _ctx("sess-empty-buffer")

    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([
        ArchiveRef(
            archive_id="archive-1",
            archive_uri="archive://sess-empty-buffer/archive-1",
            abstract="old conversation summary",
            overview="old conversation overview",
        )
    ], []))
    service._search_working_set = Mock(return_value=[])
    service._get_session_state = Mock(return_value=SessionWindowState())

    with caplog.at_level(logging.WARNING, logger="ogmem.service"):
        result = service.compose({
            "sessionId": "sess-empty-buffer",
            "_ctx": ctx,
            "messages": [
                _msg("user", "live user 1"),
                _msg("assistant", "live assistant 1"),
                _msg("user", "live user 2"),
                _msg("assistant", "live assistant 2"),
            ],
        })

    original_messages = [msg for msg in result["messages"] if not msg.get("_ogmem")]
    assert [msg["content"] for msg in original_messages] == [
        "live user 1",
        "live assistant 1",
        "live user 2",
        "live assistant 2",
    ]
    assert "compose archive trim skipped because session buffer is empty" in caplog.text
