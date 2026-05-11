"""Unit tests for SQLSessionArchiveStore.

Tests use mocked psycopg2 connections to verify behavior without
requiring a running PostgreSQL instance.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import RequestContext
from session.models import ArchiveEntry, ArchiveWriteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(account_id="test_account", user_id="test_user"):
    return RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )


SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Doing well!"},
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create SQLSessionArchiveStore with mocked psycopg2 connection pool."""
    mock_json = lambda x: x

    with patch("session.sql_archive_store.psycopg2") as mock_pg, \
         patch("session.sql_archive_store.Json", mock_json), \
         patch("session.sql_archive_store._HAS_PSYCOPG2", True):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.connect.return_value = mock_conn

        from session.sql_archive_store import SQLSessionArchiveStore
        store = SQLSessionArchiveStore.__new__(SQLSessionArchiveStore)
        store._connection_string = "host=localhost dbname=test"
        store._pool_size = 5
        store._pool = []

        # Override pool to use mock
        store._get_connection = lambda: mock_conn
        store._return_connection = lambda c: None

        yield store, mock_conn, mock_cursor


# ===========================================================================
# _ensure_table
# ===========================================================================

class TestEnsureTable:

    def test_creates_table_with_composite_pk(self, mock_store):
        """_ensure_table issues CREATE TABLE with composite primary key."""
        store, conn, cursor = mock_store
        store._ensure_table()

        sqls = [call.args[0] for call in cursor.execute.call_args_list]
        assert any("PRIMARY KEY (account_id, session_id, archive_id)" in s for s in sqls)

    def test_raises_on_failure(self, mock_store):
        """_ensure_table raises RuntimeError when DDL fails."""
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("connection lost")

        with pytest.raises(RuntimeError, match="Failed to ensure schema"):
            store._ensure_table()
        conn.rollback.assert_called()


# ===========================================================================
# write_archive
# ===========================================================================

class TestWriteArchive:

    def test_success(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="Test overview",
            abstract="Test abstract",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
            archive_id="arc001",
        )

        assert result.success is True
        assert result.archive_id == "arc001"
        assert result.session_id == "sess1"
        assert "ctx://test_account/sessions/sess1/history/arc001" in result.uri
        assert result.error is None
        conn.commit.assert_called_once()

    def test_auto_generates_archive_id(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="Overview",
            abstract="Abstract",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
        )

        assert result.success is True
        assert result.archive_id is not None
        assert len(result.archive_id) > 0
        # Format: YYYYMMDD_HHMMSS_{8hex}
        parts = result.archive_id.split("_")
        assert len(parts) == 3

    def test_passes_correct_params(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        store.write_archive(
            session_id="sess1",
            overview="Overview text",
            abstract="Abstract text",
            messages=SAMPLE_MESSAGES,
            ctx=ctx,
            archive_id="arc001",
        )

        # First call is SET LOCAL (bind_tenant for RLS), second is the INSERT.
        insert_call = cursor.execute.call_args_list[-1]
        sql = insert_call.args[0]
        params = insert_call.args[1]

        assert "INSERT INTO session_archives" in sql
        assert "ON CONFLICT (account_id, session_id, archive_id)" in sql
        # params order: archive_id, session_id, account_id, abstract, overview, messages, metadata, created_at
        assert params[0] == "arc001"
        assert params[1] == "sess1"
        assert params[2] == "test_account"
        assert params[3] == "Abstract text"
        assert params[4] == "Overview text"
        assert params[5] == SAMPLE_MESSAGES  # Json wrapper is identity in mock
        assert params[6]["message_count"] == 4

    def test_returns_failure_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("disk full")
        ctx = _ctx()

        result = store.write_archive(
            session_id="sess1",
            overview="O",
            abstract="A",
            messages=[],
            ctx=ctx,
            archive_id="arc001",
        )

        assert result.success is False
        assert "disk full" in result.error
        conn.rollback.assert_called()

    def test_empty_strings_for_none_abstract_overview(self, mock_store):
        store, conn, cursor = mock_store
        ctx = _ctx()

        store.write_archive(
            session_id="sess1",
            overview=None,
            abstract=None,
            messages=[],
            ctx=ctx,
            archive_id="arc001",
        )

        params = cursor.execute.call_args.args[1]
        assert params[3] == ""  # abstract
        assert params[4] == ""  # overview


# ===========================================================================
# list_archives
# ===========================================================================

class TestListArchives:

    def test_empty(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = []
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)

        assert entries == []
        # Verify query filters by account_id + session_id
        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "WHERE account_id = %s AND session_id = %s" in sql
        assert params == ("test_account", "sess1")

    def test_returns_entries_newest_first(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = [
            ("arc2", "sess1", "Abstract 2", "Overview 2",
             [], {"archive_id": "arc2"}, "2025-04-15T12:00:00"),
            ("arc1", "sess1", "Abstract 1", "Overview 1",
             [{"role": "user", "content": "hi"}], {"archive_id": "arc1"}, "2025-04-14T12:00:00"),
        ]
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)

        assert len(entries) == 2
        assert entries[0].archive_id == "arc2"
        assert entries[0].overview == "Overview 2"
        assert entries[0].messages == []  # list never includes messages
        assert entries[1].archive_id == "arc1"
        sql = cursor.execute.call_args.args[0]
        assert "ORDER BY created_at DESC" in sql

    def test_returns_empty_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        entries = store.list_archives("sess1", ctx)
        assert entries == []

    def test_isolates_by_account(self, mock_store):
        """list_archives filters by account_id — cross-account data invisible."""
        store, conn, cursor = mock_store
        cursor.fetchall.return_value = []
        ctx_other = _ctx(account_id="other_account")

        store.list_archives("sess1", ctx_other)

        params = cursor.execute.call_args.args[1]
        assert params[0] == "other_account"


# ===========================================================================
# read_archive
# ===========================================================================

class TestReadArchive:

    def test_not_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        entry = store.read_archive("sess1", "missing", ctx)

        assert entry is None

    def test_found(self, mock_store):
        store, conn, cursor = mock_store
        test_messages = [{"role": "user", "content": "Hello"}]
        cursor.fetchone.return_value = (
            "arc001", "sess1", "Test abstract", "Test overview",
            json.dumps(test_messages), {"key": "val"}, "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)

        assert entry is not None
        assert entry.archive_id == "arc001"
        assert entry.session_id == "sess1"
        assert entry.abstract == "Test abstract"
        assert entry.overview == "Test overview"
        assert entry.messages == test_messages
        assert entry.metadata == {"key": "val"}

    def test_filters_by_session_id(self, mock_store):
        """read_archive includes session_id in WHERE — prevents cross-session read."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        store.read_archive("sess1", "arc001", ctx)

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "session_id = %s" in sql
        assert params == ("arc001", "sess1", "test_account")

    def test_returns_none_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry is None

    def test_handles_string_metadata(self, mock_store):
        """Metadata returned as string from psycopg2 is parsed correctly."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "arc001", "sess1", "abs", "ovr",
            '[]', '{"archive_id": "arc001"}', "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry.metadata == {"archive_id": "arc001"}
        assert entry.messages == []

    def test_handles_null_messages(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = (
            "arc001", "sess1", "abs", "ovr",
            None, {}, "2025-04-15T10:00:00",
        )
        ctx = _ctx()

        entry = store.read_archive("sess1", "arc001", ctx)
        assert entry.messages == []

    def test_cross_session_read_returns_none(self, mock_store):
        """Reading with wrong session_id returns None even if archive_id exists for another session."""
        store, conn, cursor = mock_store
        # Simulate: archive arc001 belongs to sess1, not sess2
        cursor.fetchone.return_value = None
        ctx = _ctx()

        result = store.read_archive("sess2", "arc001", ctx)
        assert result is None

        # Verify the query includes session_id filter
        params = cursor.execute.call_args.args[1]
        assert params[1] == "sess2"


# ===========================================================================
# read_archive_abstract
# ===========================================================================

class TestReadArchiveAbstract:

    def test_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = ("Test abstract",)
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)

        assert abstract == "Test abstract"

    def test_not_found(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "missing", ctx)
        assert abstract is None

    def test_empty_string_returns_none(self, mock_store):
        """Empty abstract string should return None, not empty string."""
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = ("",)
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)
        assert abstract is None

    def test_filters_by_session_id(self, mock_store):
        store, conn, cursor = mock_store
        cursor.fetchone.return_value = None
        ctx = _ctx()

        store.read_archive_abstract("sess1", "arc001", ctx)

        sql = cursor.execute.call_args.args[0]
        params = cursor.execute.call_args.args[1]
        assert "session_id = %s" in sql
        assert params == ("arc001", "sess1", "test_account")

    def test_returns_none_on_db_error(self, mock_store):
        store, conn, cursor = mock_store
        cursor.execute.side_effect = Exception("timeout")
        ctx = _ctx()

        abstract = store.read_archive_abstract("sess1", "arc001", ctx)
        assert abstract is None


# ===========================================================================
# Constructor guard
# ===========================================================================

class TestConstructor:

    def test_raises_without_psycopg2(self):
        with patch("session.sql_archive_store._HAS_PSYCOPG2", False):
            from session.sql_archive_store import SQLSessionArchiveStore
            with pytest.raises(ImportError, match="psycopg2 is required"):
                SQLSessionArchiveStore("host=localhost")
