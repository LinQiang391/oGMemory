"""The same tenant contract for memory and real SQL, without depending on RLS.

SQL cases run when OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN are supplied.
Use an isolated test database; the SQL fixture creates the graph schema.
"""

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.graph_enums import GraphEdgeType, RawDataType
from core.graph_models import create_graph_edge
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import GraphService
from service.operator_service import OperatorService, RawNodeNotFoundError


@pytest.fixture(params=["memory", "sql"])
def tenant_graph(request):
    if request.param == "sql":
        dsn = os.environ.get("OGMEM_TEST_DSN") or os.environ.get("GRAPH_DSN")
        if os.environ.get("OGMEM_SQL_INTEGRATION") != "1" or not dsn:
            pytest.skip("set OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN")
        from providers.graph_store.sql_graph_store import SQLGraphStore
        store = SQLGraphStore(connection_string=dsn)
    else:
        store = MemoryGraphStore()
    graph = GraphService(store)
    contexts = [
        RequestContext(account_id=uuid4().hex, user_id="u", agent_id="a",
                       session_id="s", trace_id="t")
        for _ in range(2)
    ]
    records = []
    for ctx in contexts:
        raw = graph.create_raw_node(RawDataType.PROMPT, "private needle", ctx, node_id=str(uuid4()))
        extracted = graph.create_extracted_node(
            raw.node_id, "test", "schema", ctx,
            extracted_content={"embedding": [1.0, 0.0]}, node_id=str(uuid4()),
        )
        memory = graph.create_memory_node(
            "private needle", ctx, supporting_node_ids=[extracted.node_id, raw.node_id],
            node_id=str(uuid4()),
        )
        records.append((raw, extracted, memory))
    yield store, graph, contexts, records
    for ctx in contexts:
        for node in store.list_nodes(None, ctx, limit=None):
            store.delete_node(node.node_id, ctx)
    if request.param == "sql":
        for conn in store._pool:
            conn.close()


def test_foreign_ids_are_not_readable(tenant_graph):
    store, graph, (owner, other), (nodes, _) = tenant_graph
    raw, extracted, memory = nodes
    for node in nodes:
        assert store.get_node(node.node_id, other) is None
        assert store.get_neighbors(node.node_id, other) == []
        for depth in (0, 3):
            subgraph = store.traverse(node.node_id, depth, other)
            assert subgraph.nodes == [] and subgraph.edges == []
        assert store.get_node_pagerank(node.node_id, other) == 0
    assert graph.get_raw_node(raw.node_id, other) is None
    with pytest.raises(NodeNotFoundError):
        store.get_supporting_chain(memory.node_id, other)
    edge = store.get_neighbors(extracted.node_id, owner, direction="outgoing")[0][1]
    assert store.get_edge(edge.edge_id, other) is None
    store.delete_edge(edge.edge_id, other)
    assert store.get_edge(edge.edge_id, owner) is not None


def test_search_statistics_and_pagerank_are_scoped(tenant_graph):
    store, _, (owner, other), (nodes, other_nodes) = tenant_graph
    raw, extracted, memory = nodes
    assert {n.node_id for n in store.list_nodes(None, owner)} == {n.node_id for n in nodes}
    assert [n.node_id for n in store.search_raw_nodes(None, owner, "needle")] == [raw.node_id]
    assert [n.node_id for n in store.search_memories("needle", owner)] == [memory.node_id]
    assert store.search_by_embedding([1.0, 0.0], owner) == [(str(extracted.node_id), 1.0)]
    stats = store.get_graph_stats(owner)
    assert (stats.total_nodes, stats.total_edges) == (3, 3)
    assert {row["node_id"] for row in stats.top_memories} == {memory.node_id}
    before = {n.node_id: store.get_node_pagerank(n.node_id, other) for n in other_nodes}
    assert store.compute_pagerank(owner, [n.node_id for n in other_nodes]) == {}
    assert set(store.compute_pagerank(owner)) == {n.node_id for n in nodes}
    assert before == {n.node_id: store.get_node_pagerank(n.node_id, other) for n in other_nodes}


def test_node_writes_cannot_spoof_or_reassign_account(tenant_graph):
    store, _, (owner, other), (nodes, _) = tenant_graph
    raw = nodes[0]
    payload = raw.serialize()
    payload["content"] = "overwritten"
    with pytest.raises(AccessDeniedError):
        store.update_node(payload, other)
    # Forging the payload owner still cannot update the existing foreign row.
    payload["account_id"] = other.account_id
    with pytest.raises((AccessDeniedError, NodeNotFoundError)):
        store.update_node(payload, other)
    with pytest.raises(AccessDeniedError):
        store.update_node(payload, owner)
    with pytest.raises(AccessDeniedError):
        store.create_node(payload, owner)
    try:
        store.delete_node(raw.node_id, other)
    except AccessDeniedError:
        pass  # Memory backend's existing denial convention.
    assert store.get_node(raw.node_id, owner).content == "private needle"
    assert len(store.get_neighbors(raw.node_id, owner)) == 2


def test_edge_creation_rejects_foreign_endpoints_and_owner(tenant_graph):
    store, _, (owner, other), (nodes, other_nodes) = tenant_graph
    edge = create_graph_edge(
        nodes[0].node_id, other_nodes[0].node_id, GraphEdgeType.REFERENCES,
        account_id=owner.account_id,
    )
    with pytest.raises(NodeNotFoundError):
        store.create_edge(edge, owner)
    with pytest.raises(AccessDeniedError):
        store.create_edge(replace(edge, target_id=nodes[1].node_id, account_id=other.account_id), owner)
    assert store.get_graph_stats(owner).total_edges == 3
    assert store.get_graph_stats(other).total_edges == 3


def test_supporting_chain_and_traversal_include_real_edge_endpoints(tenant_graph):
    store, _, (ctx, _), (nodes, _) = tenant_graph
    raw, extracted, memory = nodes
    chain = store.get_supporting_chain(memory.node_id, ctx)
    assert chain.memory.node_id == memory.node_id
    assert [n.node_id for n in chain.raw_nodes] == [raw.node_id]
    assert [n.node_id for n in chain.extracted_nodes] == [extracted.node_id]
    assert {(e.source_id, e.target_id, e.edge_type) for e in chain.edges} == {
        (raw.node_id, memory.node_id, GraphEdgeType.SUPPORTS),
        (extracted.node_id, memory.node_id, GraphEdgeType.SUPPORTS),
        (extracted.node_id, raw.node_id, GraphEdgeType.EXTRACTED_FROM),
    }
    subgraph = store.traverse(extracted.node_id, 2, ctx)
    assert {n.node_id for n in subgraph.nodes} == {n.node_id for n in nodes}
    assert len(subgraph.edges) == 3


def test_supporting_chain_follows_extraction_to_raw_without_direct_support(tenant_graph):
    store, graph, (ctx, _), (nodes, _) = tenant_graph
    raw, extracted, _ = nodes
    memory = graph.create_memory_node(
        "indirect evidence", ctx, supporting_node_ids=[extracted.node_id],
        node_id=str(uuid4()),
    )
    chain = store.get_supporting_chain(memory.node_id, ctx)
    assert [n.node_id for n in chain.raw_nodes] == [raw.node_id]
    assert [n.node_id for n in chain.extracted_nodes] == [extracted.node_id]
    assert {(e.source_id, e.target_id) for e in chain.edges} == {
        (extracted.node_id, memory.node_id), (extracted.node_id, raw.node_id),
    }


def test_operator_cannot_process_another_accounts_raw_node(tenant_graph):
    store, graph, (_, other), (nodes, _) = tenant_graph
    service = OperatorService(graph)
    with pytest.raises(RawNodeNotFoundError):
        service.run_operator(nodes[0].node_id, "document_structure", other)
    with pytest.raises(RawNodeNotFoundError):
        service.run_pipeline(
            nodes[0].node_id, {"steps": [{"operator": "document_structure"}]}, other,
        )
    assert store.get_graph_stats(other).total_nodes == 3


def test_operator_http_rejects_foreign_raw_id(tenant_graph, monkeypatch):
    import server.app as app_module

    store, graph, (_, other), (nodes, _) = tenant_graph
    monkeypatch.setattr(app_module, "_operator_service", OperatorService(graph))
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: other)
    client = app_module.app.test_client()
    response = client.post(
        "/api/v1/operators/document_structure/run", json={"raw_node_id": nodes[0].node_id},
    )
    assert response.status_code == 404
    assert "private needle" not in response.get_data(as_text=True)
    assert store.get_graph_stats(other).total_nodes == 3


def test_sql_helpers_fail_closed_without_the_matching_tenant(tenant_graph):
    store, _, (owner, other), (nodes, _) = tenant_graph
    if isinstance(store, MemoryGraphStore):
        pytest.skip("SQL schema helper functions only")
    raw, _, memory = nodes
    conn = store._get_connection()
    try:
        for ctx in (None, other, owner):
            if ctx is not None:
                store.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM get_supporting_chain(%s)", (memory.node_id,))
                assert bool(cur.fetchall()) == (ctx is owner)
                cur.execute("SELECT * FROM get_node_neighbors(%s)", (raw.node_id,))
                assert bool(cur.fetchall()) == (ctx is owner)
                cur.execute("SELECT calculate_pagerank_incremental(%s)", (raw.node_id,))
                assert (cur.fetchone()[0] > 0) == (ctx is owner)
            conn.rollback()
    finally:
        conn.rollback()
        store._return_connection(conn)


def test_memory_rejects_id_collision_and_hides_corrupt_cross_account_edges():
    store = MemoryGraphStore()
    a = RequestContext(account_id="a", user_id="u", agent_id="a", session_id="s", trace_id="t")
    b = replace(a, account_id="b")
    graph = GraphService(store)
    own = graph.create_raw_node(RawDataType.PROMPT, "a", a)
    foreign = graph.create_raw_node(RawDataType.PROMPT, "b", b)
    payload = foreign.serialize()
    payload["account_id"] = a.account_id
    with pytest.raises(AccessDeniedError):
        store.create_node(payload, a)
    # Simulate an invalid edge left by an older version: both edge ownership
    # and endpoint ownership must be checked, not merely one of them.
    bad = create_graph_edge(own.node_id, foreign.node_id, GraphEdgeType.REFERENCES, a.account_id)
    store._edges[bad.edge_id] = bad
    store._edges_by_source[own.node_id] = [bad.edge_id]
    store._edges_by_target[foreign.node_id] = [bad.edge_id]
    assert store.get_edge(bad.edge_id, a) is None
    assert store.get_neighbors(own.node_id, a) == []
    assert store.traverse(own.node_id, 2, a).edges == []
    assert store.get_graph_stats(a).total_edges == 0
