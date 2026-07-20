"""Business-level tests for SQL lifecycle trace persistence."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from fs.sql_adapter.sql_trace_store import SQLTraceEventStore, _turn_id


@pytest.fixture
def store_and_db():
    store = SQLTraceEventStore.__new__(SQLTraceEventStore)
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda _: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    store._get_connection = lambda: conn
    store._return_connection = lambda _: None
    store.bind_tenant = MagicMock()
    store._max_turn_events = 128
    return store, conn, cursor


def _event_row(event_id="evt-1", event_index=17):
    return (
        event_id,
        f"ctx://acct/traces/sess/events/{event_id}",
        "acct",
        "user/alice",
        "agent",
        "sess",
        "PreToolUse",
        event_index,
        datetime(2026, 7, 15, tzinfo=UTC),
        {"event_type": "tool_start"},
        "[PreToolUse] Read",
        "[PreToolUse] Read",
    )


def test_insert_is_idempotent_by_stable_event_id(store_and_db):
    store, conn, cursor = store_and_db
    cursor.fetchone.side_effect = [None, _event_row()]

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.insert(
            event_id="evt-1",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="PreToolUse",
            content={"event_type": "tool_start", "tool_name": "Read"},
        )

    assert result == {
        "ok": True,
        "event_id": "evt-1",
        "uri": "ctx://acct/traces/sess/events/evt-1",
        "event_index": 17,
        "inserted": False,
    }
    assert "ON CONFLICT (account_id, event_id) DO NOTHING" in cursor.execute.call_args_list[0].args[0]
    assert "WHERE account_id = %s AND event_id = %s" in cursor.execute.call_args_list[1].args[0]
    assert cursor.execute.call_args_list[1].args[1] == ("acct", "evt-1")
    store.bind_tenant.assert_called_once_with(conn, "acct")
    conn.commit.assert_called_once()


def test_insert_rolls_back_database_errors(store_and_db):
    store, conn, cursor = store_and_db
    cursor.execute.side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        store.insert(
            event_id="evt-1",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="Stop",
            content={"event_type": "assistant_message"},
        )

    conn.rollback.assert_called_once()


def test_insert_persists_canonical_turn_end_type(store_and_db):
    store, _, cursor = store_and_db
    cursor.fetchone.return_value = _event_row(event_id="evt-turn-end", event_index=18)

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.insert(
            event_id="evt-turn-end",
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            hook_type="turn_end",
            content={"event_type": "turn_end"},
        )

    assert result["inserted"] is True
    insert_params = cursor.execute.call_args_list[0].args[1]
    assert insert_params[6] == "turn_end"
    assert insert_params[8] == {"event_type": "turn_end"}


@pytest.mark.parametrize("boundary_type", ["turn_end", "Stop"])
def test_finalize_aggregates_only_the_turn_end_bounded_turn(
    store_and_db, boundary_type
):
    store, conn, cursor = store_and_db
    started = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    ended = datetime(2026, 7, 15, 8, 1, tzinfo=UTC)
    cursor.fetchone.side_effect = [
        (ended,),
        (9,),
        (3, started, ended),
        ("ctx://acct/traces/sess/turns/t",),
    ]
    cursor.fetchall.return_value = [
        (10, "UserPromptSubmit", started, {"event_type": "user_message", "prompt": "hello"}, "hello"),
        (11, "PreToolUse", started, {"event_type": "tool_start", "tool_name": "Read"}, "[PreToolUse] Read"),
        (12, boundary_type, ended, {"event_type": "turn_end", "assistant_message": "done"}, "done"),
    ]

    with patch("fs.sql_adapter.sql_trace_store._json_value", side_effect=lambda value: value):
        result = store.finalize_session(
            account_id="acct",
            owner_space="user/alice",
            agent_id="agent",
            session_id="sess",
            stop_event_index=12,
        )

    assert result["event_count"] == 3
    assert result["prev_stop_event_index"] == 9
    boundary_query = cursor.execute.call_args_list[0].args[0]
    previous_boundary_query = cursor.execute.call_args_list[1].args[0]
    assert "hook_type IN ('turn_end', 'Stop')" in boundary_query
    assert "hook_type IN ('turn_end', 'Stop')" in previous_boundary_query
    range_query = cursor.execute.call_args_list[3]
    assert range_query.args[1][:-1] == (
        "acct",
        "sess",
        "user/alice",
        "agent",
        9,
        12,
    )
    upsert = cursor.execute.call_args_list[4]
    assert "INSERT INTO trace_sessions" in upsert.args[0]
    assert "ON CONFLICT (account_id, turn_id) DO UPDATE" in upsert.args[0]
    assert "hello" in upsert.args[1][9]
    assert "done" in upsert.args[1][9]
    assert upsert.args[1][10] == upsert.args[1][9]
    conn.commit.assert_called_once()


def test_fetch_turn_events_caps_stored_projection(store_and_db):
    store, _, cursor = store_and_db
    started = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    ended = datetime(2026, 7, 15, 8, 5, tzinfo=UTC)
    cursor.fetchone.side_effect = [
        (200, started, ended),
        (1, "UserPromptSubmit", started, {"event_type": "user_message"}, "hello"),
    ]
    cursor.fetchall.return_value = [
        (index, "PreToolUse", ended, {"event_type": "tool_start"}, "tool")
        for index in range(200, 73, -1)
    ]

    events, total, actual_start, actual_end = store._fetch_turn_events(
        cursor, "acct", "user:alice", "agent", "sess", 0, 200
    )

    assert total == 200
    assert len(events) == 128
    assert events[0]["event_index"] == 1
    assert events[1]["event_index"] == 74
    assert events[-1]["event_index"] == 200
    assert (actual_start, actual_end) == (started, ended)
    assert cursor.execute.call_args_list[2].args[1][-1] == 127

    metadata = store._session_metadata(events, total)
    assert metadata["truncated"] is True
    assert metadata["stored_event_count"] == 128


def test_truncated_metadata_labels_retained_sample_counts():
    events = [
        {
            "hook_type": "UserPromptSubmit",
            "content": {"event_type": "user_message"},
            "abstract": "hello",
        },
        {
            "hook_type": "Stop",
            "content": {"event_type": "assistant_message"},
            "abstract": "done",
        },
    ]

    metadata = SQLTraceEventStore._session_metadata(events, total_event_count=4)

    assert metadata["truncated"] is True
    assert metadata["stored_tool_event_count"] == 0
    assert metadata["stored_tools_used"] == []
    assert metadata["stored_hook_event_counts"] == {
        "UserPromptSubmit": 1,
        "Stop": 1,
    }
    assert "tool_call_count" not in metadata


def test_turn_id_includes_full_tenant_and_owner_identity():
    first = _turn_id("acct-a", "user:alice", "agent", "session-full", 7)
    other_tenant = _turn_id("acct-b", "user:alice", "agent", "session-full", 7)
    other_owner = _turn_id("acct-a", "user:bob", "agent", "session-full", 7)

    assert first.startswith("turn_")
    assert len(first) == 37
    assert len({first, other_tenant, other_owner}) == 3


@pytest.mark.parametrize("invalid", [1, 1025])
def test_rejects_invalid_turn_projection_configuration(invalid):
    with pytest.raises(ValueError, match="between 2 and 1024"):
        SQLTraceEventStore(connection_string="unused", max_turn_events=invalid)
