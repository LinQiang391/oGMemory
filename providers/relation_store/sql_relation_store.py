"""PostgreSQL-backed RelationStore implementation.

Stores relation edges in the ``relation_edges`` table instead of
per-node .relations.json files.
"""

from __future__ import annotations

import logging

from core.errors import AccessDeniedError
from core.models import RelationEdge, RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


class SQLRelationStore(PoolAdapterMixin):
    """PostgreSQL implementation of RelationStore.

    Edges are stored as rows in ``relation_edges`` with a composite
    primary key ``(from_uri, to_uri, relation_type)`` for idempotent
    upsert.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLRelationStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Connection pool (standalone fallback — used when pool=None)
    # ------------------------------------------------------------------

    def _get_connection(self):
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn
        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn):
        if len(self._pool) < self._pool_size and conn.closed == 0:
            self._pool.append(conn)
        else:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn)
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"Failed to ensure schema: {exc}"
            ) from exc
        finally:
            self._return_connection(conn)

    def _extract_account_id(self, uri: str) -> str:
        from fs.sql_adapter.sql_context_fs import parse_uri
        return parse_uri(uri)["account"]

    def _ensure_accessible(self, uri: str, ctx: RequestContext) -> None:
        from fs.sql_adapter.sql_context_fs import parse_uri

        try:
            account_id = parse_uri(uri)["account"]
        except ValueError as exc:
            raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format") from exc

        if account_id != ctx.account_id:
            raise AccessDeniedError(
                uri,
                ctx.account_id,
                f"URI belongs to account '{account_id}'",
            )

    # ------------------------------------------------------------------
    # RelationStore Protocol
    # ------------------------------------------------------------------

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_uri, to_uri, relation_type, weight, reason
                    FROM relation_edges
                    WHERE from_uri = %s AND account_id = %s
                    """,
                    (uri, ctx.account_id),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        return [
            RelationEdge(
                from_uri=r[0],
                to_uri=r[1],
                relation_type=r[2],
                weight=float(r[3]),
                reason=r[4],
            )
            for r in rows
        ]

    def upsert_edges(
        self, edges: list[RelationEdge], ctx: RequestContext
    ) -> None:
        if not edges:
            return

        for edge in edges:
            self._ensure_accessible(edge.from_uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                for edge in edges:
                    cur.execute(
                        """
                        INSERT INTO relation_edges
                            (from_uri, to_uri, relation_type, weight, reason, account_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (from_uri, to_uri, relation_type) DO UPDATE SET
                            weight = EXCLUDED.weight,
                            reason = EXCLUDED.reason,
                            updated_at = NOW()
                        """,
                        (
                            edge.from_uri,
                            edge.to_uri,
                            edge.relation_type,
                            edge.weight,
                            edge.reason,
                            ctx.account_id,
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_one_hop(
        self, uri: str, ctx: RequestContext, limit: int = 3
    ) -> list[RelationEdge]:
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_uri, to_uri, relation_type, weight, reason
                    FROM relation_edges
                    WHERE from_uri = %s AND account_id = %s
                    ORDER BY weight DESC
                    LIMIT %s
                    """,
                    (uri, ctx.account_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        return [
            RelationEdge(
                from_uri=r[0],
                to_uri=r[1],
                relation_type=r[2],
                weight=float(r[3]),
                reason=r[4],
            )
            for r in rows
        ]
