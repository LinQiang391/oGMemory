"""Shared memory/PostgreSQL regression contract for provenance and tag filters."""
import os
from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from core.errors import NodeNotFoundError
from core.graph_enums import RawDataType, GraphEdgeType
from core.graph_models import create_graph_edge
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore
from providers.graph_store.sql_graph_store import SQLGraphStore
from service.graph_service import GraphService


@pytest.fixture(params=['memory', 'sql'])
def store(request):
    if request.param == 'memory':
        return MemoryGraphStore()
    dsn = os.getenv('OGMEM_TEST_DSN')
    if os.getenv('OGMEM_SQL_INTEGRATION') != '1' or not dsn:
        pytest.skip('Set OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN')
    return SQLGraphStore(connection_string=dsn)


@pytest.fixture
def ctx():
    return RequestContext(account_id=uuid4().hex, user_id='u', agent_id='a', session_id='s', trace_id='t')


def test_provenance_rolls_back_all_edges(store, ctx, monkeypatch):
    graph = GraphService(store)
    supports = [graph.create_raw_node(RawDataType.PROMPT, 'evidence', ctx) for _ in range(2)]
    before = [store.get_node(n.node_id, ctx).pagerank for n in supports]
    nid = str(uuid4())
    # Force failure after the first edge has actually been inserted.
    method = '_insert_edge' if isinstance(store, SQLGraphStore) else 'create_edge'
    original = getattr(store, method)
    calls = []
    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('injected edge failure')
        return original(*args)
    monkeypatch.setattr(store, method, fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        graph.create_memory_node('memory', ctx, supporting_node_ids=[n.node_id for n in supports], node_id=nid)
    assert store.get_node(nid, ctx) is None
    assert [store.get_node(n.node_id, ctx).pagerank for n in supports] == before
    for node in supports:
        assert store.get_neighbors(node.node_id, ctx) == []


@pytest.mark.parametrize('kind', ['memory', 'extracted'])
def test_missing_provenance_leaves_no_node(store, ctx, kind):
    graph = GraphService(store)
    nid, missing = str(uuid4()), str(uuid4())
    with pytest.raises(NodeNotFoundError):
        if kind == 'memory':
            graph.create_memory_node('memory', ctx, supporting_node_ids=[missing], node_id=nid)
        else:
            graph.create_extracted_node(missing, 'test', 'v1', ctx, node_id=nid)
    assert store.get_node(nid, ctx) is None


@pytest.mark.parametrize('tags,expected', [(None, 3), ([], 3), (['a'], 2), (['a','b'], 1), (['a','a'], 2), (['A'], 0), (['missing'], 0)])
def test_tags_filter_contract(store, ctx, tags, expected):
    graph = GraphService(store)
    for labels in (['a','b'], ['a'], []):
        graph.create_memory_node('needle', ctx, tags=labels, topic='test')
    foreign = RequestContext(account_id=uuid4().hex, user_id='u', agent_id='a', session_id='s', trace_id='t')
    graph.create_memory_node('needle', foreign, tags=['a','b'], topic='test')
    graph.create_memory_node('needle', ctx, tags=['a','b'], topic='other')
    result = store.search_memories('needle', ctx, tags=tags, topic='test', top_k=10)
    assert len(result) == expected
    assert all(n.account_id == ctx.account_id for n in result)
    assert len(store.search_memories('needle', ctx, tags=tags, topic='test', top_k=1)) == min(1, expected)


def test_sql_edge_query_count_and_idempotent_rank_delta(ctx):
    store = SQLGraphStore.__new__(SQLGraphStore)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    store._get_connection = lambda: conn
    store._return_connection = lambda c: None
    store.bind_tenant = lambda *args: None
    target = str(uuid4())
    for i in range(100):
        source = str(uuid4())
        cur.fetchall.return_value = [(source,), (target,)]
        cur.fetchone.return_value = (1.0,) if i % 2 else None
        edge = create_graph_edge(source, target, GraphEdgeType.SUPPORTS, ctx.account_id, weight=1.0)
        cur.reset_mock()
        store.create_edge(edge, ctx)
        assert cur.execute.call_count == 4
        sql, params = cur.execute.call_args.args
        assert 'pagerank = pagerank +' in sql
        assert params == (0.0 if i % 2 else 1.0, target, ctx.account_id)


def test_foreign_provenance_leaves_no_node(store, ctx):
    graph = GraphService(store)
    other = RequestContext(account_id=uuid4().hex, user_id='u', agent_id='a', session_id='s', trace_id='t')
    foreign = graph.create_raw_node(RawDataType.PROMPT, 'private', other)
    nid = str(uuid4())
    with pytest.raises(NodeNotFoundError):
        graph.create_memory_node('memory', ctx, supporting_node_ids=[foreign.node_id], node_id=nid)
    assert store.get_node(nid, ctx) is None
    assert store.get_node(foreign.node_id, other).content == 'private'


def test_duplicate_node_does_not_overwrite_existing(store, ctx):
    from psycopg2 import IntegrityError
    graph = GraphService(store)
    existing = graph.create_memory_node('original', ctx)
    with pytest.raises((FileExistsError, IntegrityError)):
        graph.create_memory_node('replacement', ctx, node_id=existing.node_id)
    assert store.get_node(existing.node_id, ctx).summary == 'original'


def test_sql_fan_in_rank_is_idempotent(store, ctx):
    if not isinstance(store, SQLGraphStore):
        pytest.skip('SQL edge upsert semantics')
    graph = GraphService(store)
    target = graph.create_raw_node(RawDataType.AUDIO, 'audio', ctx)
    children = [graph.create_extracted_node(target.node_id, 'asr', 'v1', ctx) for _ in range(100)]
    assert store.get_node(target.node_id, ctx).pagerank == 100
    edge = create_graph_edge(children[0].node_id, target.node_id, GraphEdgeType.EXTRACTED_FROM, ctx.account_id)
    store.create_edge(edge, ctx)
    assert store.get_node(target.node_id, ctx).pagerank == 100
    edge.weight = 2.0
    store.create_edge(edge, ctx)
    assert store.get_node(target.node_id, ctx).pagerank == 101
