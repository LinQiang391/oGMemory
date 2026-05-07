"""Unit tests for SQLRelationStore."""

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
