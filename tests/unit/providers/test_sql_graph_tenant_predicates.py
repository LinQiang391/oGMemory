"""Offline guards: every SQL graph query binds an explicit tenant predicate."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from providers.graph_store.sql_graph_store import SQLGraphStore


@pytest.mark.parametrize("operation", [
    "get_node", "update_node", "delete_node", "get_edge", "delete_edge",
    "get_node_pagerank", "compute_pagerank", "compute_selected_pagerank",
    "search_memories", "search_raw_nodes", "get_graph_stats", "get_neighbors",
    "list_nodes", "search_by_embedding",
])
def test_every_query_binds_tenant_and_has_balanced_parameters(operation):
    store = SQLGraphStore.__new__(SQLGraphStore)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    cur.rowcount = 1
    store._get_connection = lambda: conn
    store._return_connection = lambda c: None
    store.bind_tenant = lambda *args: None  # A session setting alone is not enough.
    ctx = RequestContext(account_id="tenant-a", user_id="u", agent_id="a", session_id="s", trace_id="t")
    node_id = str(uuid4())
    if operation == "update_node":
        node = create_raw_node(RawDataType.PROMPT, "data", ctx.account_id, node_id=node_id)
        store.update_node(node.serialize(), ctx)
    elif operation == "compute_selected_pagerank":
        store.compute_pagerank(ctx, [node_id])
    elif operation == "compute_pagerank":
        store.compute_pagerank(ctx)
    elif operation == "search_memories":
        store.search_memories("secret", ctx)
    elif operation == "search_raw_nodes":
        store.search_raw_nodes(None, ctx, query="secret")
    elif operation == "get_graph_stats":
        cur.fetchone.side_effect = [(0, 0, 0, 0, 0.0), (0,)]
        store.get_graph_stats(ctx)
    elif operation == "list_nodes":
        store.list_nodes(None, ctx)
    elif operation == "search_by_embedding":
        store.search_by_embedding([1.0, 0.0], ctx)
    else:
        getattr(store, operation)(node_id, ctx)
    assert cur.execute.call_count
    for call in cur.execute.call_args_list:
        sql, params = call.args
        assert "account_id = %s" in sql
        assert ctx.account_id in params
        assert sql.count("%s") == len(params)
