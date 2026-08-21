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
        "tool_result_semantic_turn_mode": "model_request",
        "after_turn_threshold": 999_999,
    }
    values.update(overrides)
    return _isolate_service(MemoryService(
        OgMemConfig(**values),
        tool_result_store=LocalToolResultArtifactStore(tmp_path),
    ))


def _attested_params(
    messages: list[dict],
    *,
    pre_prompt: int,
    complete: bool = False,
    turn_id: str = "turn-1",
) -> dict:
    namespace = f"ogmem-openclaw-test-{turn_id}"
    copied = deepcopy(messages)
    for index, message in enumerate(copied):
        message["_ogmem_source"] = {
            "source_message_id": f"{namespace}-{index}",
            "turn_id": turn_id,
            "semantic_turn_attested": True,
        }
    return {
        "sessionId": "session",
        "messages": copied,
        "prePromptMessageCount": pre_prompt,
        "semanticTurnId": turn_id,
        "semanticTurnStartMessageCount": 0,
        "semanticTurnSourceNamespace": namespace,
        "semanticTurnAppendOnlyVerified": True,
        "semanticTurnComplete": complete,
        "semanticTurnMode": "model_request",
        "_ctx": _ctx(),
    }


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


def test_attested_model_request_retry_reuses_artifact_and_buffer_message(tmp_path):
    service = _service(tmp_path)
    messages = [{
        "role": "toolResult",
        "content": "x" * 30,
        "toolCallId": "call-retry",
    }]
    params = _attested_params(
        messages,
        pre_prompt=0,
        complete=True,
        turn_id="retry-turn",
    )

    first = service.after_turn(params)
    retried = service.after_turn(params)

    buffer = service.get_session_manager().get_or_create("session", ctx=_ctx())
    assert first["status"] == "accumulating"
    assert retried["status"] == "no_new_messages"
    assert len(buffer.messages) == 1
    assert buffer.messages[0].id == "ogmem-openclaw-test-retry-turn-0"
    assert "[ogmemory-tool-result v=2]" in buffer.messages[0].content
    assert len(service._tool_result_store.list("session", _ctx())["artifacts"]) == 1
