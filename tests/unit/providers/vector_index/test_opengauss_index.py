"""Unit tests for OpenGaussIndex.

Uses mock database connection for isolated testing.
For integration tests with real database, see scripts/test_opengauss_index.py
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any

from core.models import IndexRecord, TypedQuery, SeedHit


class MockCursor:
    """Mock database cursor for testing."""

    def __init__(self):
        self.executed_sql: list[str] = []
        self.executed_params: list[tuple] = []
        self.results: list[tuple] = []
        self.rowcount = 0

    def execute(self, sql: str, params: tuple = None):
        self.executed_sql.append(sql)
        if params:
            self.executed_params.append(params)

    def fetchone(self):
        if self.results:
            return self.results.pop(0)
        return None

    def fetchall(self):
        results = self.results
        self.results = []
        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, cursor: MockCursor = None):
        self._cursor = cursor or MockCursor()
        self.committed = False
        self._closed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, value):
        self._closed = value


class TestVectorToString:
    """Tests for _vector_to_string method."""

    def test_simple_vector(self):
        """Test conversion of simple vector."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        with patch('providers.vector_index.opengauss_index.psycopg2'):
            index = OpenGaussIndex.__new__(OpenGaussIndex)
            result = index._vector_to_string([1.0, 2.0, 3.0])
            assert result == "[1.0,2.0,3.0]"

    def test_negative_values(self):
        """Test conversion with negative values."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        with patch('providers.vector_index.opengauss_index.psycopg2'):
            index = OpenGaussIndex.__new__(OpenGaussIndex)
            result = index._vector_to_string([-1.5, 0.0, 2.5])
            assert result == "[-1.5,0.0,2.5]"

    def test_empty_vector(self):
        """Test conversion of empty vector."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        with patch('providers.vector_index.opengauss_index.psycopg2'):
            index = OpenGaussIndex.__new__(OpenGaussIndex)
            result = index._vector_to_string([])
            assert result == "[]"

    def test_float_precision(self):
        """Test that float values are preserved."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        with patch('providers.vector_index.opengauss_index.psycopg2'):
            index = OpenGaussIndex.__new__(OpenGaussIndex)
            result = index._vector_to_string([0.123456789, 1.0])
            assert "0.123456789" in result


class TestUpsert:
    """Tests for upsert operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_single_record(self, mock_psycopg2):
        """Test upserting a single record."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists check
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        record = IndexRecord(
            id="test_id_1",
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="Test abstract",
            filters={"account_id": "test", "owner_space": "users/u1"},
            metadata={"_embedding": [0.1] * 1536},
        )

        index.upsert([record])

        assert len(mock_cursor.executed_sql) == 1
        assert "INSERT INTO" in mock_cursor.executed_sql[0]
        assert "ON DUPLICATE KEY UPDATE" in mock_cursor.executed_sql[0]
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_multiple_records(self, mock_psycopg2):
        """Test upserting multiple records."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        records = [
            IndexRecord(
                id=f"test_id_{i}",
                uri="ctx://test/users/u1/memories/profile",
                level=i,
                text=f"Text {i}",
                filters={"account_id": "test", "owner_space": "users/u1"},
                metadata={"_embedding": [0.1] * 1536},
            )
            for i in range(3)
        ]

        index.upsert(records)

        assert len(mock_cursor.executed_sql) == 3
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_empty_list(self, mock_psycopg2):
        """Test upserting empty list does nothing."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.upsert([])

        assert not mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_upsert_missing_embedding_raises_error(self, mock_psycopg2):
        """Test that missing embedding raises ValueError."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        record = IndexRecord(
            id="test_id_1",
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="Test abstract",
            filters={"account_id": "test", "owner_space": "users/u1"},
            metadata={},  # Missing _embedding
        )

        with pytest.raises(ValueError, match="Missing '_embedding'"):
            index.upsert([record])


class TestDelete:
    """Tests for delete operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_single_id(self, mock_psycopg2):
        """Test deleting a single record by ID."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 1
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete(["id1"])

        assert len(mock_cursor.executed_sql) == 1
        assert "DELETE FROM" in mock_cursor.executed_sql[0]
        assert "ANY(%s)" in mock_cursor.executed_sql[0]

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_multiple_ids(self, mock_psycopg2):
        """Test deleting multiple records."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 3
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete(["id1", "id2", "id3"])

        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_delete_empty_list(self, mock_psycopg2):
        """Test deleting empty list does nothing."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        index.delete([])

        assert not mock_conn.committed


class TestSearch:
    """Tests for search operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_returns_results(self, mock_psycopg2):
        """Test search returns SeedHit results."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [
            (True,),  # Table exists check
            ("ctx://test/users/u1/memories/profile", 0.85, "Test abstract", True, True),
            ("ctx://test/users/u1/memories/pref", 0.72, "Pref abstract", False, True),
        ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=["profile"],
            account_id="test",
            owner_space="users/u1",
            top_k=10,
            embedding=[0.1] * 1536,
        )

        results = index.search(query)

        assert len(results) == 2
        assert isinstance(results[0], SeedHit)
        assert results[0].uri == "ctx://test/users/u1/memories/profile"
        assert results[0].score == 0.85

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_missing_embedding_raises_error(self, mock_psycopg2):
        """Test that search without embedding raises ValueError."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=[],
            account_id="test",
            top_k=10,
            embedding=None,  # Missing embedding
        )

        with pytest.raises(ValueError, match="embedding"):
            index.search(query)

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_with_category_filter(self, mock_psycopg2):
        """Test search applies category filter."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=["profile", "preference"],
            account_id="test",
            top_k=10,
            embedding=[0.1] * 1536,
        )

        index.search(query)

        sql = mock_cursor.executed_sql[0]
        assert "metadata->>'category' IN" in sql

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_search_with_owner_space_filter(self, mock_psycopg2):
        """Test search applies owner_space filter."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), ]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        query = TypedQuery(
            text="test query",
            context_type="MEMORY",
            categories=[],
            account_id="test",
            owner_space="users/alice",
            top_k=10,
            embedding=[0.1] * 1536,
        )

        index.search(query)

        sql = mock_cursor.executed_sql[0]
        assert "filters->>'owner_space'" in sql


class TestCount:
    """Tests for count operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_count_all_records(self, mock_psycopg2):
        """Test counting all records."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), (42,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        count = index.count()

        assert count == 42
        assert "SELECT COUNT(*)" in mock_cursor.executed_sql[0]

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_count_by_account(self, mock_psycopg2):
        """Test counting records for specific account."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,), (10,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        count = index.count(account_id="test_account")

        assert count == 10
        assert "filters->>'account_id'" in mock_cursor.executed_sql[0]


class TestClear:
    """Tests for clear operation."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_clear_all_records(self, mock_psycopg2):
        """Test clearing all records."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 100
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        deleted = index.clear()

        assert deleted == 100
        assert mock_conn.committed

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_clear_by_account(self, mock_psycopg2):
        """Test clearing records for specific account."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.rowcount = 25
        mock_cursor.results = [(True,)]
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        deleted = index.clear(account_id="test_account")

        assert deleted == 25
        assert "filters->>'account_id'" in mock_cursor.executed_sql[0]


class TestConnectionPool:
    """Tests for connection pool behavior."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_connection_reuse(self, mock_psycopg2):
        """Test that connections are reused from pool."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn1 = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn1

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 2
        index._pool = []
        index._initialized = True

        conn1 = index._get_connection()
        index._return_connection(conn1)
        conn2 = index._get_connection()

        assert conn1 is conn2
        assert len(index._pool) == 0

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_close_all_connections(self, mock_psycopg2):
        """Test that close() closes all pooled connections."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        conn = index._get_connection()
        index._return_connection(conn)

        index.close()

        assert len(index._pool) == 0


class TestContextManager:
    """Tests for context manager protocol."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_context_manager_closes_on_exit(self, mock_psycopg2):
        """Test that context manager closes connections on exit."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_conn = MockConnection()
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex.__new__(OpenGaussIndex)
        index._connection_string = "postgres://test"
        index._dimension = 1536
        index._table_name = "vector_index"
        index._pool_size = 5
        index._pool = []
        index._initialized = True

        with index:
            conn = index._get_connection()
            index._return_connection(conn)

        assert len(index._pool) == 0


class TestAutoInit:
    """Tests for auto-initialization."""

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_extension(self, mock_psycopg2):
        """Test that auto_init creates pgvector extension."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_table_with_dimension(self, mock_psycopg2):
        """Test that auto_init creates table with correct dimension."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test", dimension=768)

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "vector(768)" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_skips_if_table_exists(self, mock_psycopg2):
        """Test that auto_init skips table creation if table already exists."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "CREATE TABLE" not in sql_commands
        assert "CREATE EXTENSION" in sql_commands  # Extension is still created

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_hnsw_index(self, mock_psycopg2):
        """Test that auto_init creates HNSW index."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "USING hnsw" in sql_commands
        assert "vector_cosine_ops" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_filter_indexes(self, mock_psycopg2):
        """Test that auto_init creates filter indexes."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "filters->>'account_id'" in sql_commands
        assert "filters->>'owner_space'" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_custom_table_name(self, mock_psycopg2):
        """Test that auto_init uses custom table name."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test", table_name="custom_vectors")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "custom_vectors" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_auto_init_creates_trigger(self, mock_psycopg2):
        """Test that auto_init creates auto-update trigger."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(False,)]  # Table does not exist
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        OpenGaussIndex("postgres://test")

        sql_commands = " ".join(mock_cursor.executed_sql)
        assert "CREATE TRIGGER" in sql_commands
        assert "updated_at" in sql_commands

    @patch('providers.vector_index.opengauss_index.psycopg2')
    def test_ensure_table_exists_is_idempotent(self, mock_psycopg2):
        """Test that _ensure_table_exists is idempotent."""
        from providers.vector_index.opengauss_index import OpenGaussIndex

        mock_cursor = MockCursor()
        mock_cursor.results = [(True,)]  # Table exists
        mock_conn = MockConnection(mock_cursor)
        mock_psycopg2.connect.return_value = mock_conn

        index = OpenGaussIndex("postgres://test")
        first_count = len(mock_cursor.executed_sql)

        index._ensure_table_exists()
        second_count = len(mock_cursor.executed_sql)

        assert first_count == second_count
