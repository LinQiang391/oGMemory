"""Exercise the real vector MERGE's identity/ownership on PostgreSQL 16+.

A schema-local text domain stands in for the vector type: these tests check
the DML/tenant boundary, not ANN behavior or openGauss-specific execution.
"""

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from providers.vector_index import opengauss_index

DSN = os.environ.get("OGMEM_TEST_DSN") or os.environ.get("GRAPH_DSN")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OGMEM_SQL_INTEGRATION") != "1" or not DSN,
        reason="set OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN (PostgreSQL 16+)",
    ),
]


@pytest.fixture
def index(monkeypatch):
    import psycopg2
    from psycopg2.extras import Json

    monkeypatch.setattr(opengauss_index, "Json", Json)
    conn = psycopg2.connect(DSN)
    schema = f"vector_tenant_test_{uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        cur.execute("CREATE DOMAIN vector AS text")
        cur.execute("""
            CREATE TABLE test_vectors (
                id text PRIMARY KEY, uri text, level integer, text text,
                embedding vector, filters jsonb, metadata jsonb,
                account_id text, owner_space text, node_uri text, node_id text,
                record_kind text, node_status text, space_id text,
                updated_at timestamptz
            )
        """)
    conn.commit()
    index = opengauss_index.OpenGaussVectorIndex.__new__(opengauss_index.OpenGaussVectorIndex)
    index._table_name = "test_vectors"
    index._get_connection = lambda: conn
    index._return_connection = lambda c: None
    try:
        yield index, conn
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()


def test_global_id_collision_cannot_transfer_owner_or_content(index):
    from core.models import IndexRecord

    vi, conn = index
    record = IndexRecord(
        id="legacy-global-id", uri="ctx://a/users/u/memories/events/legacy", level=2, text="private a",
        filters={"account_id": "a"}, metadata={"_embedding": [1.0, 0.0]},
    )
    vi.upsert([record])
    vi.upsert([replace(
        record, filters={"account_id": "b"}, text="overwrite attempt",
        metadata={"_embedding": [0.0, 1.0]},
    )])
    with conn.cursor() as cur:
        cur.execute("SELECT account_id, text, embedding FROM test_vectors")
        assert cur.fetchall() == [("a", "private a", "[1.00000000,0.00000000]")]
    # Collision rollback must not break a subsequent legitimate update.
    vi.upsert([replace(record, text="updated a")])
    with conn.cursor() as cur:
        cur.execute("SELECT account_id, text FROM test_vectors")
        assert cur.fetchall() == [("a", "updated a")]


def test_explicit_migration_backfills_once_without_cross_account_node_links(index):
    vi, conn = index
    with conn.cursor() as cur:
        cur.execute('CREATE TABLE context_nodes (node_id text, uri text, account_id text)')
        cur.execute("""INSERT INTO context_nodes VALUES
            ('node-a', 'ctx://a', 'a'), ('node-b', 'ctx://b', 'b')""")
        cur.execute("""INSERT INTO test_vectors (id, uri, filters) VALUES
            ('legacy-a', 'ctx://a/content.md', '{"account_id":"a","owner_space":"u"}'),
            ('foreign-link', 'ctx://b', '{"account_id":"a"}')""")
    conn.commit()
    assert vi.migrate_legacy_rows() is True
    with conn.cursor() as cur:
        cur.execute('SELECT id, account_id, node_id, node_uri FROM test_vectors ORDER BY id')
        assert cur.fetchall() == [
            ('foreign-link', 'a', None, 'ctx://b'),
            ('legacy-a', 'a', 'node-a', 'ctx://a/content.md'),
        ]
        # Even a newly inserted legacy-shaped row must not trigger another scan.
        cur.execute("INSERT INTO test_vectors (id, uri, filters) VALUES ('late', 'ctx://a', '{\"account_id\":\"a\"}')")
    conn.commit()
    assert vi.migrate_legacy_rows() is False
    with conn.cursor() as cur:
        cur.execute("SELECT account_id FROM test_vectors WHERE id = 'late'")
        assert cur.fetchone() == (None,)
