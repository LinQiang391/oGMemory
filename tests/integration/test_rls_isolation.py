"""Row-Level Security acceptance tests.

Verifies that PostgreSQL RLS policies block cross-tenant access even
when application-level WHERE clauses are bypassed via raw SQL.

These tests require a running PostgreSQL instance with the context_nodes,
relation_edges, and session_archives tables (created by the stores'
_ensure_table methods which include RLS DDL).
"""

import os
import uuid

import pytest

from providers.unified_config import OgMemConfig

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sql_dsn():
    cfg = OgMemConfig.load()
    dsn = (
        os.environ.get("TEST_SQL_CONNECTION_STRING")
        or os.environ.get("SQL_CONNECTION_STRING")
        or cfg.sql_connection_string
    )
    if not dsn:
        pytest.skip(
            "No SQL DSN configured in TEST_SQL_CONNECTION_STRING, "
            "SQL_CONNECTION_STRING, or ogmem.yaml"
        )
    if psycopg2 is None:
        pytest.skip("psycopg2 is not installed")
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    return dsn


@pytest.fixture
def tenant_a():
    return f"rls-test-a-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def tenant_b():
    return f"rls-test-b-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_context_nodes(sql_dsn, tenant_a):
    """Remove test rows after each test."""
    yield
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM context_nodes WHERE account_id LIKE 'rls-test-%'"
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def cleanup_relation_edges(sql_dsn):
    yield
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM relation_edges WHERE account_id LIKE 'rls-test-%'"
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def cleanup_session_archives(sql_dsn):
    yield
    conn = psycopg2.connect(sql_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM session_archives WHERE account_id LIKE 'rls-test-%'"
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def cleanup_trace_tables(sql_dsn, tenant_a, tenant_b):
    yield
    conn = psycopg2.connect(sql_dsn)
    try:
        for tenant in (tenant_a, tenant_b):
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant,))
                cur.execute("DELETE FROM trace_sessions WHERE account_id = %s", (tenant,))
                cur.execute("DELETE FROM trace_events WHERE account_id = %s", (tenant,))
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# lifecycle trace RLS and tenant-scoped idempotency
# ---------------------------------------------------------------------------


class TestTraceEventsRLS:
    def test_reference_schema_upgrade_backfills_under_forced_rls(self, sql_dsn):
        from fs.sql_adapter.schema import _SCHEMA_SQL

        schema_name = f"trace_upgrade_{uuid.uuid4().hex[:12]}"
        tenant = f"upgrade-{uuid.uuid4().hex[:8]}"
        other_tenant = f"upgrade-{uuid.uuid4().hex[:8]}"
        conn = psycopg2.connect(sql_dsn)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA "{schema_name}"')
                cur.execute(f'SET search_path TO "{schema_name}", public')
                cur.execute(
                    """
                    CREATE TABLE trace_events (
                        uri VARCHAR(512) PRIMARY KEY,
                        account_id VARCHAR(128) NOT NULL,
                        owner_space VARCHAR(128) NOT NULL,
                        agent_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        hook_type VARCHAR(64) NOT NULL,
                        event_index INTEGER NOT NULL,
                        event_time TIMESTAMPTZ NOT NULL,
                        content JSONB NOT NULL DEFAULT '{}'::jsonb,
                        abstract TEXT NOT NULL DEFAULT '',
                        overview TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "CREATE POLICY tenant_isolation ON trace_events "
                    "USING (account_id = current_setting('app.account_id', true)) "
                    "WITH CHECK (account_id = current_setting('app.account_id', true))"
                )
                cur.execute("ALTER TABLE trace_events ENABLE ROW LEVEL SECURITY")
                cur.execute("SET app.account_id = %s", (tenant,))
                cur.execute(
                    "INSERT INTO trace_events "
                    "(uri, account_id, owner_space, agent_id, session_id, "
                    "hook_type, event_index, event_time) "
                    "VALUES (%s, %s, 'user:alice', 'agent', 'sess', "
                    "'PreToolUse', 7, NOW())",
                    (f"ctx://{tenant}/legacy/event-7", tenant),
                )
                cur.execute("ALTER TABLE trace_events FORCE ROW LEVEL SECURITY")
                cur.execute("RESET app.account_id")
                cur.execute(
                    """
                    CREATE TABLE trace_sessions (
                        uri VARCHAR(512) PRIMARY KEY,
                        account_id VARCHAR(128) NOT NULL,
                        owner_space VARCHAR(128) NOT NULL,
                        agent_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL,
                        ended_at TIMESTAMPTZ,
                        event_count INTEGER NOT NULL DEFAULT 0,
                        abstract TEXT NOT NULL DEFAULT '',
                        overview TEXT NOT NULL DEFAULT '',
                        content JSONB NOT NULL DEFAULT '[]'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        status VARCHAR(16) NOT NULL DEFAULT 'finalized',
                        turn_id VARCHAR(64) NOT NULL UNIQUE,
                        stop_event_index INTEGER NOT NULL,
                        prev_stop_event_index INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "CREATE POLICY tenant_isolation ON trace_sessions "
                    "USING (account_id = current_setting('app.account_id', true)) "
                    "WITH CHECK (account_id = current_setting('app.account_id', true))"
                )
                cur.execute("ALTER TABLE trace_sessions ENABLE ROW LEVEL SECURITY")
                cur.execute("ALTER TABLE trace_sessions FORCE ROW LEVEL SECURITY")

                cur.execute(_SCHEMA_SQL)
                cur.execute("SET app.account_id = %s", (tenant,))
                cur.execute(
                    "SELECT event_id, event_index FROM trace_events WHERE account_id = %s",
                    (tenant,),
                )
                event_id, event_index = cur.fetchone()
                cur.execute(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE oid = 'trace_events'::regclass"
                )
                rls_enabled, rls_forced = cur.fetchone()
                cur.execute(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE oid = 'trace_sessions'::regclass"
                )
                sessions_rls_enabled, sessions_rls_forced = cur.fetchone()
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = 'trace_sessions' "
                    "AND column_name IN ('stop_event_index', 'prev_stop_event_index')",
                    (schema_name,),
                )
                stop_index_types = dict(cur.fetchall())
                for account in (tenant, other_tenant):
                    cur.execute("SET app.account_id = %s", (account,))
                    cur.execute(
                        "INSERT INTO trace_sessions "
                        "(turn_id, uri, account_id, owner_space, agent_id, "
                        "session_id, started_at, ended_at, stop_event_index) "
                        "VALUES ('legacy-collision', %s, %s, 'user:alice', "
                        "'agent', 'same-session', NOW(), NOW(), 7)",
                        (f"ctx://{account}/traces/legacy-turn", account),
                    )
                cur.execute("SET app.account_id = %s", (other_tenant,))
                cur.execute(
                    "SELECT COUNT(*) FROM trace_sessions WHERE account_id = %s",
                    (tenant,),
                )
                hidden_session_count = cur.fetchone()[0]
                cur.execute(
                    "UPDATE trace_sessions SET status = 'tampered' "
                    "WHERE account_id = %s",
                    (tenant,),
                )
                hidden_update_count = cur.rowcount

            assert len(event_id) == 32
            assert event_index == 7
            assert rls_enabled is True
            assert rls_forced is True
            assert sessions_rls_enabled is True
            assert sessions_rls_forced is True
            assert hidden_session_count == 0
            assert hidden_update_count == 0
            assert stop_index_types == {
                "prev_stop_event_index": "bigint",
                "stop_event_index": "bigint",
            }
        finally:
            with conn.cursor() as cur:
                cur.execute("RESET search_path")
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
            conn.close()

    def test_same_event_id_is_independent_across_tenants(
        self, sql_dsn, tenant_a, tenant_b, cleanup_trace_tables
    ):
        from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

        store = SQLTraceEventStore(connection_string=sql_dsn)
        common = {
            "event_id": "shared-id",
            "owner_space": "user:alice",
            "agent_id": "claude-code",
            "session_id": "session-1",
            "hook_type": "PreToolUse",
            "content": {"event_type": "tool_start"},
        }

        first = store.insert(account_id=tenant_a, **common)
        second = store.insert(account_id=tenant_b, **common)

        assert first["inserted"] is True
        assert second["inserted"] is True

    def test_retry_and_stop_finalization_are_idempotent(
        self, sql_dsn, tenant_a, cleanup_trace_tables
    ):
        from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

        store = SQLTraceEventStore(connection_string=sql_dsn)
        common = {
            "account_id": tenant_a,
            "owner_space": "user:alice",
            "agent_id": "claude-code",
            "session_id": "session-1",
        }
        first = store.insert(
            event_id="user-1",
            hook_type="UserPromptSubmit",
            content={"event_type": "user_message"},
            **common,
        )
        duplicate = store.insert(
            event_id="user-1",
            hook_type="UserPromptSubmit",
            content={"event_type": "user_message"},
            **common,
        )
        stop = store.insert(
            event_id="stop-1",
            hook_type="Stop",
            content={"event_type": "assistant_message"},
            **common,
        )
        finalized = store.finalize_session(
            stop_event_index=stop["event_index"], **common
        )
        replayed = store.finalize_session(
            stop_event_index=stop["event_index"], **common
        )

        assert duplicate["inserted"] is False
        assert duplicate["event_index"] == first["event_index"]
        assert replayed["turn_id"] == finalized["turn_id"]
        assert replayed["event_count"] == 2

    def test_consecutive_turns_and_same_session_owners_stay_separate(
        self, sql_dsn, tenant_a, cleanup_trace_tables
    ):
        from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

        store = SQLTraceEventStore(connection_string=sql_dsn)

        def write_turn(owner, suffix):
            common = {
                "account_id": tenant_a,
                "owner_space": owner,
                "agent_id": "claude-code",
                "session_id": "shared-session",
            }
            store.insert(
                event_id=f"user-{suffix}",
                hook_type="UserPromptSubmit",
                content={"event_type": "user_message"},
                **common,
            )
            stop = store.insert(
                event_id=f"stop-{suffix}",
                hook_type="Stop",
                content={"event_type": "assistant_message"},
                **common,
            )
            return store.finalize_session(
                stop_event_index=stop["event_index"], **common
            )

        alice_first = write_turn("user:alice", "alice-1")
        alice_second = write_turn("user:alice", "alice-2")
        bob = write_turn("user:bob", "bob-1")

        assert alice_second["prev_stop_event_index"] == alice_first["stop_event_index"]
        assert alice_second["event_count"] == 2
        assert bob["event_count"] == 2
        assert len({alice_first["turn_id"], alice_second["turn_id"], bob["turn_id"]}) == 3

    def test_trace_insert_does_not_write_memory_or_outbox_tables(
        self, sql_dsn, tenant_a, cleanup_trace_tables
    ):
        from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

        def counts():
            conn = psycopg2.connect(sql_dsn)
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                    cur.execute(
                        "SELECT COUNT(*) FROM context_nodes WHERE account_id = %s",
                        (tenant_a,),
                    )
                    nodes = cur.fetchone()[0]
                    cur.execute(
                        "SELECT COUNT(*) FROM outbox_events WHERE account_id = %s",
                        (tenant_a,),
                    )
                    outbox = cur.fetchone()[0]
                return nodes, outbox
            finally:
                conn.rollback()
                conn.close()

        before = counts()
        store = SQLTraceEventStore(connection_string=sql_dsn)
        store.insert(
            event_id="isolated-trace",
            account_id=tenant_a,
            owner_space="user:alice",
            agent_id="claude-code",
            session_id="session-1",
            hook_type="PreToolUse",
            content={"event_type": "tool_start"},
        )

        assert counts() == before

    def test_cross_tenant_select_returns_zero(
        self, sql_dsn, tenant_a, tenant_b, cleanup_trace_tables
    ):
        from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

        store = SQLTraceEventStore(connection_string=sql_dsn)
        store.insert(
            event_id="tenant-a-event",
            account_id=tenant_a,
            owner_space="user:alice",
            agent_id="claude-code",
            session_id="session-1",
            hook_type="PreToolUse",
            content={"event_type": "tool_start"},
        )

        conn = psycopg2.connect(sql_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_b,))
                cur.execute(
                    "SELECT COUNT(*) FROM trace_events WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
        finally:
            conn.rollback()
            conn.close()

        assert count == 0


# ---------------------------------------------------------------------------
# context_nodes RLS
# ---------------------------------------------------------------------------


class TestContextNodesRLS:
    """RLS tests for the context_nodes table."""

    def test_cross_tenant_select_returns_zero(
        self, sql_dsn, tenant_a, tenant_b, cleanup_context_nodes
    ):
        """Even with raw SQL WHERE account_id = tenant-A, a connection
        bound to tenant-B must see 0 rows."""
        # 1. Insert a row for tenant-A (bind to tenant-A so WITH CHECK passes)
        conn_a = psycopg2.connect(sql_dsn)
        try:
            with conn_a.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "INSERT INTO context_nodes "
                    "(uri, account_id, owner_space, category, context_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        f"ctx://{tenant_a}/users/u1/memories/profile",
                        tenant_a,
                        "user:u1",
                        "profile",
                        "MEMORY",
                    ),
                )
            conn_a.commit()
        finally:
            conn_a.close()

        # 2. Connect as tenant-B and try to read tenant-A's rows
        conn_b = psycopg2.connect(sql_dsn)
        try:
            with conn_b.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_b,))
                cur.execute(
                    "SELECT count(*) FROM context_nodes "
                    "WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
        finally:
            conn_b.rollback()
            conn_b.close()

        assert count == 0, (
            "RLS FAILED: tenant-B can see tenant-A rows via raw SQL"
        )

    def test_same_tenant_select_returns_row(
        self, sql_dsn, tenant_a, cleanup_context_nodes
    ):
        """A connection bound to tenant-A can see its own rows."""
        conn = psycopg2.connect(sql_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "INSERT INTO context_nodes "
                    "(uri, account_id, owner_space, category, context_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        f"ctx://{tenant_a}/users/u1/memories/profile",
                        tenant_a,
                        "user:u1",
                        "profile",
                        "MEMORY",
                    ),
                )
            conn.commit()

            # Re-bind and query
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "SELECT count(*) FROM context_nodes "
                    "WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
            conn.rollback()
        finally:
            conn.close()

        assert count == 1

    def test_cross_tenant_insert_blocked(
        self, sql_dsn, tenant_a, tenant_b, cleanup_context_nodes
    ):
        """A connection bound to tenant-B cannot INSERT rows with
        account_id = tenant-A (WITH CHECK clause)."""
        conn = psycopg2.connect(sql_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_b,))
                with pytest.raises(Exception):
                    cur.execute(
                        "INSERT INTO context_nodes "
                        "(uri, account_id, owner_space, category, context_type) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            f"ctx://{tenant_a}/users/u1/memories/profile",
                            tenant_a,  # Wrong account_id
                            "user:u1",
                            "profile",
                            "MEMORY",
                        ),
                    )
        finally:
            conn.rollback()
            conn.close()

    def test_no_binding_sees_nothing(
        self, sql_dsn, tenant_a, cleanup_context_nodes
    ):
        """A connection that never sets app.account_id sees 0 rows
        (current_setting returns NULL → safe default)."""
        # Insert with proper binding
        conn_a = psycopg2.connect(sql_dsn)
        try:
            with conn_a.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "INSERT INTO context_nodes "
                    "(uri, account_id, owner_space, category, context_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        f"ctx://{tenant_a}/users/u1/memories/profile",
                        tenant_a,
                        "user:u1",
                        "profile",
                        "MEMORY",
                    ),
                )
            conn_a.commit()
        finally:
            conn_a.close()

        # Query WITHOUT binding
        conn_no_bind = psycopg2.connect(sql_dsn)
        try:
            with conn_no_bind.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM context_nodes "
                    "WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
        finally:
            conn_no_bind.rollback()
            conn_no_bind.close()

        assert count == 0, (
            "RLS FAILED: unbound connection sees rows"
        )


# ---------------------------------------------------------------------------
# relation_edges RLS
# ---------------------------------------------------------------------------


class TestRelationEdgesRLS:
    """RLS tests for the relation_edges table."""

    def test_cross_tenant_select_returns_zero(
        self, sql_dsn, tenant_a, tenant_b, cleanup_relation_edges
    ):
        conn_a = psycopg2.connect(sql_dsn)
        try:
            with conn_a.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "INSERT INTO relation_edges "
                    "(from_uri, to_uri, relation_type, account_id) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        f"ctx://{tenant_a}/users/u1/memories/profile",
                        f"ctx://{tenant_a}/users/u1/memories/entities/x",
                        "related_to",
                        tenant_a,
                    ),
                )
            conn_a.commit()
        finally:
            conn_a.close()

        conn_b = psycopg2.connect(sql_dsn)
        try:
            with conn_b.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_b,))
                cur.execute(
                    "SELECT count(*) FROM relation_edges "
                    "WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
        finally:
            conn_b.rollback()
            conn_b.close()

        assert count == 0


# ---------------------------------------------------------------------------
# session_archives RLS
# ---------------------------------------------------------------------------


class TestSessionArchivesRLS:
    """RLS tests for the session_archives table."""

    def test_cross_tenant_select_returns_zero(
        self, sql_dsn, tenant_a, tenant_b, cleanup_session_archives
    ):
        conn_a = psycopg2.connect(sql_dsn)
        try:
            with conn_a.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_a,))
                cur.execute(
                    "INSERT INTO session_archives "
                    "(archive_id, session_id, account_id) "
                    "VALUES (%s, %s, %s)",
                    ("arch-1", "sess-1", tenant_a),
                )
            conn_a.commit()
        finally:
            conn_a.close()

        conn_b = psycopg2.connect(sql_dsn)
        try:
            with conn_b.cursor() as cur:
                cur.execute("SET LOCAL app.account_id = %s", (tenant_b,))
                cur.execute(
                    "SELECT count(*) FROM session_archives "
                    "WHERE account_id = %s",
                    (tenant_a,),
                )
                count = cur.fetchone()[0]
        finally:
            conn_b.rollback()
            conn_b.close()

        assert count == 0
