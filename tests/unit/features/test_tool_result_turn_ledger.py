from __future__ import annotations

from copy import deepcopy
import json

from core.models import RequestContext
from features.tool_result_externalization.turn_ledger import ToolResultTurnLedger


def _ctx(**overrides) -> RequestContext:
    values = {
        "account_id": "acct",
        "user_id": "user",
        "agent_id": "agent",
        "session_id": "session",
        "trace_id": "trace",
    }
    values.update(overrides)
    return RequestContext(**values)


def _result_turn(
    output: str,
    *,
    prompt: str = "inspect",
    call_id: str = "call_ledger_1",
) -> list[dict]:
    return [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": call_id,
                "name": "read",
                "arguments": {"path": "/tmp/fixture"},
            }],
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "read",
            "content": output,
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]


def test_ledger_retry_reuses_sequence_and_normalized_call_id():
    ledger = ToolResultTurnLedger(b"x" * 32)
    first_turn = _result_turn("payload", call_id="call_abc_1")
    first = ledger.register_user_message_turn(
        None,
        first_turn,
        start_boundary=7,
        ctx=_ctx(),
        session_id="session",
    )
    retried_turn = deepcopy(first_turn)
    retried_turn[1]["content"][0]["id"] = "callabc1"
    retried_turn[2]["toolCallId"] = "callabc1"
    second = ledger.register_user_message_turn(
        first.state,
        retried_turn,
        start_boundary=7,
        ctx=_ctx(),
        session_id="session",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.entry["turn_sequence"] == second.entry["turn_sequence"] == 1
    assert first.entry["semantic_turn_id"] == second.entry["semantic_turn_id"]
    assert first.entry["message_source_ids"] == second.entry["message_source_ids"]


def test_ledger_excludes_recall_collection_from_original_results():
    ledger = ToolResultTurnLedger(b"x" * 32)
    source_collection = json.dumps({
        "results": [{
            "_ogmemory_artifact_source": True,
            "artifact_ref": "ogmemory://tool-results/tr_" + "a" * 32,
            "schema_version": 1,
            "source_tool": "search_tool_result_artifacts",
        }],
    })
    turn = _result_turn("R" * 25_000, call_id="call_original")[:-1] + [
        {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": "call_recall",
                "name": "search_tool_result_artifacts",
            }],
        },
        {
            "role": "toolResult",
            "toolCallId": "call_recall",
            "toolName": "search_tool_result_artifacts",
            "content": source_collection,
        },
        {"role": "assistant", "content": "done", "stopReason": "stop"},
    ]

    registered = ledger.register_user_message_turn(
        None,
        turn,
        start_boundary=0,
        ctx=_ctx(),
        session_id="session",
    )

    assert len(registered.entry["results"]) == 1
    assert registered.entry["results"][0]["tool_call_key"] == "calloriginal"


def test_owner_mismatch_and_ambiguous_projection_fail_open():
    ledger = ToolResultTurnLedger(b"x" * 32)
    turn = _result_turn("same")
    first = ledger.register_user_message_turn(
        None, turn, start_boundary=0, ctx=_ctx(), session_id="session",
    )
    second = ledger.register_user_message_turn(
        first.state, turn, start_boundary=4, ctx=_ctx(), session_id="session",
    )

    ambiguous = ledger.restore_completed_turns(
        second.state,
        turn + turn,
        ctx=_ctx(),
        session_id="session",
    )
    hidden = ledger.restore_completed_turns(
        second.state,
        turn,
        ctx=_ctx(account_id="other"),
        session_id="session",
    )

    assert ambiguous.source_groups == ()
    assert all("_ogmem_source" not in message for message in ambiguous.messages)
    assert hidden.source_groups == ()


def test_completed_turns_are_bounded_to_latest_1024():
    ledger = ToolResultTurnLedger(b"x" * 32)
    state = None
    for sequence in range(1, 1_026):
        registration = ledger.register_user_message_turn(
            state,
            _result_turn(
                "payload",
                prompt=f"turn {sequence}",
                call_id=f"call_{sequence}",
            ),
            start_boundary=(sequence - 1) * 4,
            ctx=_ctx(),
            session_id="session",
        )
        state = registration.state

    assert state["next_turn_sequence"] == 1_026
    assert len(state["completed_turns"]) == 1_024
    assert state["completed_turns"][0]["turn_sequence"] == 2
    assert state["completed_turns"][-1]["turn_sequence"] == 1_025
