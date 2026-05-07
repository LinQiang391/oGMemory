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
