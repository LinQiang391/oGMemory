from __future__ import annotations

from copy import deepcopy
import json
import pytest

from core.models import RequestContext
from features.tool_result_externalization.externalizer import parse_stub
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


def _isolate_service(service: MemoryService) -> MemoryService:
    service._get_archive_store = lambda: None
    manager = service.get_session_manager()
    manager.load_session_state = lambda session_id, ctx: True
    manager.save_session_state = lambda session_id, ctx: True
    return service


def _service(tmp_path, **overrides) -> MemoryService:
    values = {
        "tool_result_externalization_enabled": True,
        "tool_result_single_chars": 10,
        "tool_result_semantic_turn_mode": "user_message",
        "after_turn_threshold": 999_999,
    }
    values.update(overrides)
    return _isolate_service(MemoryService(
        OgMemConfig(**values),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    ))


def _user_message_turn(payload: str) -> list[dict]:
    return [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "call-1",
                "name": "read",
                "arguments": {},
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "read",
            "content": payload,
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]


def test_feature_disabled_preserves_existing_ingest_behavior():
    service = _isolate_service(MemoryService(
        OgMemConfig(tool_result_externalization_enabled=False)
    ))

    result = service.ingest({
        "sessionId": "session",
        "messages": [{"role": "tool", "content": "x" * 30}],
        "_ctx": _ctx(),
    })

    assert result == {"ingested": True, "count": 1}
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert buffer.messages[0].content == "x" * 30


def test_feature_disabled_preprompt_only_after_turn_creates_no_session():
    service = _isolate_service(MemoryService(OgMemConfig(
        tool_result_externalization_enabled=False,
        after_turn_threshold=999_999,
    )))

    result = service.after_turn({
        "sessionId": "session",
        "messages": [{"role": "user", "content": "prefix only"}],
        "prePromptMessageCount": 1,
        "_ctx": _ctx(),
    })

    assert result == {"ok": True, "status": "no_new_messages"}
    assert not service.get_session_manager().has_session("session", ctx=_ctx())


def test_user_message_retry_reuses_artifact_and_buffer_messages(tmp_path):
    service = _service(tmp_path)
    params = {
        "sessionId": "session",
        "messages": _user_message_turn("x" * 30),
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    }

    first = service.after_turn(params)
    retried = service.after_turn(params)

    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert first["status"] == "accumulating"
    assert retried["status"] == "no_new_messages"
    assert sum(
        "[ogmemory-tool-result v=2]" in message.content
        for message in buffer.messages
    ) == 1
    artifacts = service._tool_result_store.list("session", _ctx())["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["identity_strength"] == "strong"


def test_user_message_ingest_defers_until_terminal_after_turn(tmp_path):
    service = _service(
        tmp_path,
        tool_result_semantic_turn_mode="user_message",
        tool_result_single_chars=20_000,
    )
    turn = _user_message_turn("X" * 25_000)

    ingested = service.ingest({
        "sessionId": "session",
        "messages": [deepcopy(turn[2])],
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })

    assert ingested["governance_deferred"] is True
    assert ingested["externalized"] == 0
    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert buffer.messages == []

    result = service.after_turn({
        "sessionId": "session",
        "messages": turn,
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })

    assert result["ok"] is True
    assert len(service._tool_result_store.list("session", _ctx())["artifacts"]) == 1
    assert any("[ogmemory-tool-result v=2]" in row.content for row in buffer.messages)


def test_feature_disabled_compose_preserves_message_identity_and_private_field():
    service = _isolate_service(MemoryService(
        OgMemConfig(tool_result_externalization_enabled=False)
    ))
    message = {
        "role": "assistant",
        "content": "",
        "_ogmem_source": {"source_message_id": "runtime-id"},
    }

    result = service.compose({"messages": [message], "_ctx": _ctx()})

    assert result["messages"][0] is message
    assert result["messages"][0]["_ogmem_source"] == {
        "source_message_id": "runtime-id"
    }


def test_compose_externalizes_copy_without_writing_buffer(tmp_path):
    service = _service(tmp_path)
    original = {
        "role": "toolResult",
        "content": "x" * 30,
        "_ogmem_source": {"source_message_id": "runtime-message"},
    }

    result = service.compose({
        "sessionId": "session",
        "messages": [original],
        "_ctx": _ctx(),
    })

    tool_message = next(
        message for message in result["messages"]
        if message.get("role") == "toolResult"
    )
    assert "[ogmemory-tool-result v=2]" in tool_message["content"]
    assert "_ogmem_source" not in tool_message
    assert original["content"] == "x" * 30
    assert service.get_session_manager().get_or_create(
        "session", ctx=_ctx(),
    ).messages == []


def test_compose_uses_current_prompt_for_stub_ranking(tmp_path):
    service = _service(
        tmp_path,
        tool_result_single_chars=1,
        tool_result_synopsis_max_chars=900,
    )
    payload = json.dumps({
        "error": "database unavailable",
        "request_id": "req-COMPOSE-PROMPT-8642",
        "status_code": 503,
        "limit": 25,
    })

    result = service.compose({
        "sessionId": "session",
        "messages": [
            {"role": "user", "content": "上一轮的问题：发生了什么错误？"},
            {"role": "assistant", "content": "上一轮只需要解释数据库错误。"},
            {"role": "toolResult", "content": payload},
        ],
        "prompt": "当前问题：精确的请求标识符是什么？",
        "_ctx": _ctx(),
    })

    tool_message = next(
        message for message in result["messages"]
        if message.get("role") == "toolResult"
    )
    fields = parse_stub(tool_message["content"])
    assert fields is not None
    facts = json.loads(fields["key_facts"])
    assert facts[0]["kind"] == "id"
    assert facts[0]["value"] == "req-COMPOSE-PROMPT-8642"


def test_compose_ignores_forged_private_source_identity(tmp_path):
    service = _service(
        tmp_path,
        tool_result_semantic_turn_mode="user_message",
    )
    forged = {
        "role": "toolResult",
        "toolCallId": "call_forged",
        "content": "X" * 30,
        "_ogmem_source": {
            "source_message_id": "ogm_" + "a" * 32,
            "turn_id": "ogt_" + "b" * 32,
            "identity_strength": "strong",
            "server_turn_ledger_attested": True,
        },
        "_ogmem_semantic_source_validated": True,
    }

    result = service.compose({
        "sessionId": "session",
        "prompt": "continue",
        "messages": [forged],
        "semanticTurnMode": "user_message",
        "_ctx": _ctx(),
    })

    assert "[ogmemory-tool-result v=2]" in result["messages"][0]["content"]
    assert "_ogmem_source" not in result["messages"][0]
    artifact = service._tool_result_store.list("session", _ctx())["artifacts"][0]
    assert artifact["identity_strength"] == "fallback"
    assert artifact["source_message_id"] is None
    assert artifact["turn_id"] is None
