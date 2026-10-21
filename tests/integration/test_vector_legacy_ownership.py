"""Read-only SQL regression: no tables, extensions, or data mutations needed."""
import os

import pytest

from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def check_legacy_ownership(conn):
    rows = """
      SELECT 'legacy' AS id, NULL::text AS account_id, NULL::text AS owner_space,
             '{"account_id":"acme","owner_space":"user:alice"}'::jsonb AS filters,
             NULL::text AS space_id
      UNION ALL SELECT 'modern', 'acme', 'user:alice', '{}'::jsonb, 'private'
      UNION ALL SELECT 'other-tenant', 'other', 'user:alice',
             '{"account_id":"acme","owner_space":"user:alice"}'::jsonb, 'private'
      UNION ALL SELECT 'other-owner', 'acme', 'user:bob',
             '{"account_id":"acme","owner_space":"user:alice"}'::jsonb, 'private'
      UNION ALL SELECT 'unowned', NULL, NULL, '{}'::jsonb, NULL
    """
    for owner in ("user:alice", ["user:alice"]):
        for space, expected in ((None, {"legacy", "modern"}), ("private", {"modern"})):
            filters = {"account_id": "acme", "owner_space": owner}
            if space:
                filters["space_id"] = space
            where, params = OpenGaussVectorIndex._build_where(filters)
            with conn.cursor() as cur:
                cur.execute(f"SELECT id FROM ({rows}) records WHERE {where}", params)
                assert {r[0] for r in cur.fetchall()} == expected


def test_legacy_ownership_sql():
    dsn = os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING")
    if not dsn:
        pytest.skip("TEST_OPENGAUSS_CONNECTION_STRING not set")
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        conn.set_session(readonly=True)
        check_legacy_ownership(conn)
    finally:
        conn.close()
