from __future__ import annotations

from copy import deepcopy

from core.models import RequestContext
from features.tool_result_externalization.store_local import (
    LocalToolResultArtifactStore,
)
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _params(tool_text: str, **overrides) -> dict:
    values = {
        "sessionId": "session",
        "adapterId": "openclaw",
        "turnId": "turn-1",
        "toolCallId": "call-1",
        "toolName": "fixture",
        "content": [{"type": "text", "text": tool_text}],
        "rankingPrompt": "Find the exact failing request id",
        "_ctx": _ctx(),
    }
    values.update(overrides)
    return values


def _service(tmp_path, **overrides) -> MemoryService:
    values = {
        "tool_result_externalization_enabled": True,
        "tool_result_first_view_enabled": True,
        "tool_result_single_chars": 20,
        "tool_result_semantic_turn_mode": "user_message",
        "after_turn_threshold": 999_999,
    }
    values.update(overrides)
    service = MemoryService(
        OgMemConfig(**values),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    )
    service._get_archive_store = lambda: None
    manager = service.get_session_manager()
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _artifacts(service: MemoryService) -> list[dict]:
    return service._tool_result_store.list("session", _ctx())["artifacts"]


def test_first_view_externalizes_before_buffer_and_reuses_artifact(tmp_path):
    service = _service(tmp_path)
    payload = "ERROR request_id=req-123 failed\n" * 4

    first = service.govern_first_view_tool_result(_params(payload))
    second = service.govern_first_view_tool_result(_params(payload))

    assert first["externalized"] == 1
    assert first["failed"] == 0
    assert first["identityStrength"] == "strong"
    assert first["content"] == second["content"]
    assert "[ogmemory-tool-result v=2]" in first["content"][0]["text"]
    assert payload not in first["content"][0]["text"]
    assert service.get_session_manager().get_or_create(
        "session", ctx=_ctx(),
    ).messages == []
    assert len(_artifacts(service)) == 1


def test_first_view_stub_is_idempotent_at_terminal_after_turn(tmp_path):
    service = _service(tmp_path)
    payload = "X" * 80
    governed = service.govern_first_view_tool_result(_params(payload))
    turn = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "call-1",
                "name": "fixture",
                "arguments": {},
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "fixture",
            "content": deepcopy(governed["content"]),
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]

    result = service.after_turn({
        "sessionId": "session",
        "messages": turn,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })

    assert result["ok"] is True
    assert len(_artifacts(service)) == 1
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert all(payload not in message.content for message in buffer.messages)


def test_first_view_keeps_below_threshold_result_inline(tmp_path):
    service = _service(tmp_path, tool_result_single_chars=100)
    payload = "medium-result" * 2

    result = service.govern_first_view_tool_result(_params(payload))

    assert result["externalized"] == 0
    assert result["content"] == [{"type": "text", "text": payload}]
    assert _artifacts(service) == []


def test_first_view_disabled_is_exact_fail_open(tmp_path):
    service = _service(tmp_path, tool_result_first_view_enabled=False)
    content = [{"type": "text", "text": "X" * 80}]

    result = service.govern_first_view_tool_result(
        _params("unused", content=content)
    )

    assert result == {
        "ok": True,
        "externalized": 0,
        "failed": 0,
        "content": content,
        "reason": "first_view_disabled",
    }
    assert not service.get_session_manager().has_session("session", ctx=_ctx())
