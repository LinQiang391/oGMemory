from __future__ import annotations

import logging
from unittest.mock import Mock

from core.models import ArchiveRef, RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.models import SessionWindowState


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
