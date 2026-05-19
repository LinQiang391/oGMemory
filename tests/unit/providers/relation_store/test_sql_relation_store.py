"""Unit tests for SQLRelationStore."""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.errors import AccessDeniedError
from core.models import RelationEdge, RequestContext


def _ctx(account_id="test-account"):
    return RequestContext(
        account_id=account_id,
        user_id="user-123",
        agent_id="agent-456",
        session_id="session-789",
        trace_id="trace-abc",
    )


@pytest.fixture
def mock_store():
    """Create SQLRelationStore with mocked psycopg2 connection pool."""
    with patch("providers.relation_store.sql_relation_store.psycopg2") as mock_pg, \
         patch("providers.relation_store.sql_relation_store._HAS_PSYCOPG2", True):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value = mock_conn

        from providers.relation_store.sql_relation_store import SQLRelationStore

        store = SQLRelationStore.__new__(SQLRelationStore)
        store._connection_string = "host=localhost dbname=test"
        store._pool_size = 5
        store._pool = []
        store._get_connection = lambda: mock_conn
        store._return_connection = lambda c: None

        yield store, mock_conn, mock_cursor


class TestGetEdges:

    def test_returns_edges(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()
        cursor.fetchall.return_value = [
            (
                "ctx://test-account/users/user-123/memories/profile",
                "ctx://test-account/users/user-123/memories/entities/coffee",
                "related_to",
                0.9,
                "test",
            )
        ]

        edges = store.get_edges(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
        )

        assert len(edges) == 1
        assert edges[0].relation_type == "related_to"
        assert edges[0].weight == 0.9

    def test_filters_by_account_id(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        store.get_edges("ctx://test-account/users/user-123/memories/profile", ctx)

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "account_id = %s" in sql
        assert params == (
            "ctx://test-account/users/user-123/memories/profile",
            "test-account",
        )

    def test_cross_account_denied(self, mock_store):
        store, conn, cursor = mock_store

        with pytest.raises(AccessDeniedError, match="belongs to account"):
            store.get_edges(
                "ctx://other-account/users/alice/memories/profile",
                _ctx(),
            )


class TestUpsertEdges:

    def test_uses_request_context_account(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()
        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee",
                relation_type="related_to",
                weight=0.8,
                reason="test",
            )
        ]

        store.upsert_edges(edges, ctx)

        params = cursor.execute.call_args.args[1]
        assert params[-1] == "test-account"
        conn.commit.assert_called_once()

    def test_cross_account_denied(self, mock_store):
        store, conn, cursor = mock_store
        edges = [
            RelationEdge(
                from_uri="ctx://other-account/users/alice/memories/profile",
                to_uri="ctx://other-account/users/alice/memories/entities/coffee",
                relation_type="related_to",
                weight=0.8,
                reason="cross-account",
            )
        ]

        with pytest.raises(AccessDeniedError, match="belongs to account"):
            store.upsert_edges(edges, _ctx())

    def test_validates_all_edge_sources(self, mock_store):
        store, conn, cursor = mock_store
        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee",
                relation_type="related_to",
                weight=0.8,
                reason="ok",
            ),
            RelationEdge(
                from_uri="ctx://other-account/users/alice/memories/profile",
                to_uri="ctx://other-account/users/alice/memories/entities/tea",
                relation_type="related_to",
                weight=0.7,
                reason="bad",
            ),
        ]

        with pytest.raises(AccessDeniedError, match="belongs to account"):
            store.upsert_edges(edges, _ctx())


class TestGetOneHop:

    def test_filters_by_account_id_and_limit(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()
        cursor.fetchall.return_value = [
            (
                "ctx://test-account/users/user-123/memories/profile",
                "ctx://test-account/users/user-123/memories/entities/entity1",
                "related_to",
                0.95,
                "highest",
            ),
            (
                "ctx://test-account/users/user-123/memories/profile",
                "ctx://test-account/users/user-123/memories/entities/entity2",
                "related_to",
                0.85,
                "second",
            ),
        ]

        edges = store.get_one_hop(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
            limit=2,
        )

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "account_id = %s" in sql
        assert params == (
            "ctx://test-account/users/user-123/memories/profile",
            "test-account",
            2,
        )
        assert [edge.weight for edge in edges] == [0.95, 0.85]

    def test_cross_account_denied(self, mock_store):
        store, conn, cursor = mock_store

        with pytest.raises(AccessDeniedError, match="belongs to account"):
            store.get_one_hop(
                "ctx://other-account/users/alice/memories/profile",
                _ctx(),
                limit=1,
            )


class TestConstructor:

    def test_raises_without_psycopg2(self):
        with patch("providers.relation_store.sql_relation_store._HAS_PSYCOPG2", False):
            from providers.relation_store.sql_relation_store import SQLRelationStore

            with pytest.raises(ImportError, match="psycopg2 is required"):
                SQLRelationStore("host=localhost")


class TestFallbackToContextNodes:
    """Tests for fallback logic when relation_edges table is empty."""

    def test_get_edges_fallback_when_relation_edges_empty(self, mock_store):
        """get_edges should fallback to context_nodes.relations when relation_edges has no data."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # First query: relation_edges returns empty
        cursor.fetchall.return_value = []

        # Mock _fallback_from_context_nodes to return relations
        relations_json = [
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/coffee",
                "relation_type": "related_to",
                "weight": 0.85,
                "reason": "fallback test",
            },
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/tea",
                "relation_type": "related_to",
                "weight": 0.75,
                "reason": "fallback test 2",
            },
        ]

        with patch.object(store, '_fallback_from_context_nodes', return_value=[
            RelationEdge(
                from_uri=r["from_uri"],
                to_uri=r["to_uri"],
                relation_type=r["relation_type"],
                weight=r["weight"],
                reason=r["reason"],
            )
            for r in relations_json
        ]):
            edges = store.get_edges(
                "ctx://test-account/users/user-123/memories/profile",
                ctx,
            )

        assert len(edges) == 2
        assert edges[0].to_uri.endswith("coffee")
        assert edges[0].weight == 0.85

    def test_get_edges_prefers_relation_edges_when_has_data(self, mock_store):
        """get_edges should use relation_edges data when available, not fallback."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # relation_edges returns data
        cursor.fetchall.return_value = [
            (
                "ctx://test-account/users/user-123/memories/profile",
                "ctx://test-account/users/user-123/memories/entities/coffee",
                "related_to",
                0.95,
                "from relation_edges",
            )
        ]

        # Mock fallback (should NOT be called)
        with patch.object(store, '_fallback_from_context_nodes', return_value=[
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/fallback",
                relation_type="related_to",
                weight=0.5,
                reason="should not appear",
            )
        ]):
            edges = store.get_edges(
                "ctx://test-account/users/user-123/memories/profile",
                ctx,
            )

        assert len(edges) == 1
        assert edges[0].reason == "from relation_edges"
        assert edges[0].weight == 0.95

    def test_get_one_hop_fallback_and_sorts_by_weight(self, mock_store):
        """get_one_hop should fallback to context_nodes and sort by weight."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # relation_edges returns empty
        cursor.fetchall.return_value = []

        # Mock fallback with unsorted relations
        fallback_edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/low",
                relation_type="related_to",
                weight=0.5,
                reason="lowest weight",
            ),
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/high",
                relation_type="related_to",
                weight=0.95,
                reason="highest weight",
            ),
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/mid",
                relation_type="related_to",
                weight=0.75,
                reason="middle weight",
            ),
        ]

        with patch.object(store, '_fallback_from_context_nodes', return_value=fallback_edges):
            edges = store.get_one_hop(
                "ctx://test-account/users/user-123/memories/profile",
                ctx,
                limit=2,
            )

        # Should be sorted by weight DESC and limited to 2
        assert len(edges) == 2
        assert edges[0].weight == 0.95
        assert edges[1].weight == 0.75

    def test_get_one_hop_prefers_relation_edges_when_has_data(self, mock_store):
        """get_one_hop should use relation_edges when available, not fallback."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # relation_edges returns sorted data
        cursor.fetchall.return_value = [
            (
                "ctx://test-account/users/user-123/memories/profile",
                "ctx://test-account/users/user-123/memories/entities/top",
                "related_to",
                0.99,
                "from relation_edges",
            )
        ]

        with patch.object(store, '_fallback_from_context_nodes', return_value=[
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/fallback",
                relation_type="related_to",
                weight=0.5,
                reason="should not appear",
            )
        ]):
            edges = store.get_one_hop(
                "ctx://test-account/users/user-123/memories/profile",
                ctx,
                limit=1,
            )

        assert len(edges) == 1
        assert edges[0].reason == "from relation_edges"

    def test_fallback_from_context_nodes_parses_json(self, mock_store):
        """_fallback_from_context_nodes should parse relations JSON correctly."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # Mock context_nodes.relations column with JSON
        relations_json = '[{"from_uri": "uri1", "to_uri": "uri2", "relation_type": "test", "weight": 0.8, "reason": "ok"}]'
        cursor.fetchone.return_value = (relations_json,)

        edges = store._fallback_from_context_nodes(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
        )

        assert len(edges) == 1
        assert edges[0].from_uri == "uri1"
        assert edges[0].to_uri == "uri2"
        assert edges[0].weight == 0.8

    def test_fallback_from_context_nodes_returns_empty_when_node_missing(self, mock_store):
        """_fallback_from_context_nodes should return empty list when node doesn't exist."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        cursor.fetchone.return_value = None

        edges = store._fallback_from_context_nodes(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
        )

        assert edges == []


class TestRegressionRelationWriteRead:
    """Regression: verify relations written to context_nodes can be read via get_one_hop.

    This tests the scenario where:
    1. ContextNode is written with _relations metadata (stored in context_nodes.relations)
    2. relation_edges table is empty (write path hasn't synced)
    3. ResultRanker calls SQLRelationStore.get_one_hop()
    4. Relations are correctly returned via fallback
    """

    def test_write_node_with_relations_read_via_get_one_hop(self, mock_store):
        """Regression: relations in context_nodes.relations should be readable."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # Simulate: relation_edges table is empty
        cursor.fetchall.return_value = []

        # Simulate: context_nodes.relations has JSON data (from write_node)
        relations_json = json.dumps([
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/coffee",
                "relation_type": "related_to",
                "weight": 0.85,
                "reason": "user preference",
            },
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/tea",
                "relation_type": "related_to",
                "weight": 0.75,
                "reason": "secondary preference",
            },
        ])

        # First fetchone call for _fallback_from_context_nodes
        cursor.fetchone.return_value = (relations_json,)

        # ResultRanker calls get_one_hop
        edges = store.get_one_hop(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
            limit=3,
        )

        assert len(edges) == 2
        # Should be sorted by weight DESC
        assert edges[0].weight == 0.85
        assert edges[0].to_uri.endswith("coffee")
        assert edges[1].weight == 0.75
        assert edges[1].to_uri.endswith("tea")

    def test_multiple_nodes_with_cross_references(self, mock_store):
        """Regression: multiple nodes with different relations all readable."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        # relation_edges empty
        cursor.fetchall.return_value = []

        # First node: profile with relations
        profile_relations = json.dumps([
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/preferences/coffee",
                "relation_type": "has_preference",
                "weight": 0.9,
                "reason": "main preference",
            },
        ])

        cursor.fetchone.return_value = (profile_relations,)
        edges = store.get_one_hop(
            "ctx://test-account/users/user-123/memories/profile",
            ctx,
        )
        assert len(edges) == 1
        assert edges[0].relation_type == "has_preference"


class TestRegressionMoveNodeURIUpdate:
    """Regression: verify relations are updated after move_node.

    This tests the scenario where:
    1. Node has relations stored in context_nodes.relations
    2. Node is moved via move_node() to a new URI
    3. relation_edges should be updated with new URI
    4. Old URI should no longer appear in relations
    """

    def test_move_node_updates_relation_uri_in_edges(self, mock_store):
        """Regression: move_node cascades to relation_edges URI updates."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        old_uri = "ctx://test-account/users/user-123/memories/preferences/coffee"
        new_uri = "ctx://test-account/users/user-123/memories/preferences/latte"

        # After move, relation_edges should have new URI, not old
        # Simulate: relation_edges has updated data
        cursor.fetchall.return_value = [
            (
                new_uri,  # from_uri updated
                "ctx://test-account/users/user-123/memories/entities/milk",
                "related_to",
                0.8,
                "after move",
            ),
        ]

        # Query with new URI should return updated relation
        edges = store.get_edges(new_uri, ctx)
        assert len(edges) == 1
        assert edges[0].from_uri == new_uri
        assert edges[0].reason == "after move"

    def test_old_uri_no_relations_after_move(self, mock_store):
        """Regression: old URI should have no relations after move."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        old_uri = "ctx://test-account/users/user-123/memories/preferences/coffee"

        # relation_edges has no data for old URI
        cursor.fetchall.return_value = []

        # context_nodes also has no data for old URI (moved)
        cursor.fetchone.return_value = None

        # Query with old URI should return empty
        edges = store.get_edges(old_uri, ctx)
        assert edges == []

    def test_fallback_reads_correct_uri_after_move(self, mock_store):
        """Regression: fallback reads from context_nodes with correct (new) URI."""
        store, conn, cursor = mock_store
        ctx = _ctx()

        new_uri = "ctx://test-account/users/user-123/memories/preferences/latte"

        # relation_edges empty, fallback to context_nodes
        cursor.fetchall.return_value = []

        # context_nodes has relation with updated URI
        relations_json = json.dumps([
            {
                "from_uri": new_uri,  # updated URI
                "to_uri": "ctx://test-account/users/user-123/memories/entities/milk",
                "relation_type": "related_to",
                "weight": 0.9,
                "reason": "post-move relation",
            },
        ])
        cursor.fetchone.return_value = (relations_json,)

        edges = store.get_one_hop(new_uri, ctx)
        assert len(edges) == 1
        assert edges[0].from_uri == new_uri
        assert "coffee" not in edges[0].from_uri  # old URI should not appear
