"""Offline SQL row-decoding regressions found by semantic consolidation."""

from unittest.mock import MagicMock
from uuid import UUID

import pytest

from core.graph_enums import GraphEdgeType
from core.models import RequestContext
from providers.graph_store.sql_graph_store import SQLGraphStore


SOURCE = "00000000-0000-0000-0000-000000000001"
TARGET = "00000000-0000-0000-0000-000000000002"


def memory_row(supports):
    return (
        TARGET, "memory", "active", None, "content", "summary",
        {}, {}, {}, None, None, None, None, None, supports,
        "semantic:video", ["m3"], None, 0.0, "account", None, None,
    )


@pytest.mark.parametrize("value,expected", [
    (None, []),
    ("{}", []),
    ("{" + SOURCE + "}", [SOURCE]),
    ("{" + SOURCE + "," + TARGET + "}", [SOURCE, TARGET]),
    ([SOURCE], [SOURCE]),
    ([UUID(SOURCE)], [SOURCE]),
])
def test_uuid_array_is_always_a_list_of_strings(value, expected):
    store = SQLGraphStore.__new__(SQLGraphStore)
    node = store._row_to_node(memory_row(value))
    assert node.supporting_node_ids == expected
    assert node.serialize()["supporting_node_ids"] == expected


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_neighbors_include_edge_endpoints(direction):
    store = SQLGraphStore.__new__(SQLGraphStore)
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        memory_row([]) + ("edge-id", "evolves", 1.0, {}, "account", None),
    ]
    store._get_connection = lambda: conn
    store._return_connection = lambda c: None
    store.bind_tenant = lambda *a: None
    ctx = RequestContext(
        account_id="account", user_id="u", agent_id="a", session_id="s", trace_id="t",
    )
    [(node, edge, actual_direction)] = store.get_neighbors(
        SOURCE, ctx, edge_types=[GraphEdgeType.EVOLVES], direction=direction,
    )
    assert actual_direction == direction
    assert node.node_id == TARGET
    expected = (SOURCE, TARGET) if direction == "outgoing" else (TARGET, SOURCE)
    assert (edge.source_id, edge.target_id) == expected
    conn.rollback.assert_called_once()
