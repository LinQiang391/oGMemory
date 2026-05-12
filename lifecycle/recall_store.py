"""PostgreSQL-backed RecallStore implementation.

Stores recall events in a ``dream_recalls`` table for tracking
which memories are being retrieved during searches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.models import RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


@dataclass
class RecallEvent:
    """A single recall event."""

    uri: str
    account_id: str
    owner_space: str
    query_text: str | None = None
    score: float | None = None
    category: str | None = None
    recalled_at: str | None = None


class SQLRecallStore(PoolAdapterMixin):
    """PostgreSQL-backed recall event storage.

    Provides batch insertion, statistics querying, and pruning
    of recall events for memory consolidation dreaming.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLRecallStore. "
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_insert(
        self,
        events: list[RecallEvent],
        ctx: RequestContext,
    ) -> bool:
        """Batch insert recall events.

        Returns True if successful, False otherwise.
        """
        if not events:
            return True

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                now = datetime.now(timezone.utc)
                cur.executemany(
                    """
                    INSERT INTO dream_recalls
                        (uri, account_id, owner_space, query_text, score, category, recalled_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            event.uri,
                            event.account_id,
                            event.owner_space,
                            event.query_text,
                            event.score,
                            event.category,
                            event.recalled_at or now.isoformat(),
                        )
                        for event in events
                    ],
                )
            conn.commit()
            logger.debug(f"Inserted {len(events)} recall events")
            return True
        except Exception as exc:
            conn.rollback()
            logger.error(f"Failed to insert recall events: {exc}")
            return False
        finally:
            self._return_connection(conn)

    def get_recall_stats(
        self,
        account_id: str,
        window_days: int = 30,
        ctx: RequestContext | None = None,
    ) -> dict:
        """Get recall statistics for an account within a time window.

        Returns a dict with:
        - total_recalls: total number of recall events
        - unique_uris: number of unique memories recalled
        - top_categories: dict of category -> count
        - avg_score: average recall score (for scored recalls)
        """
        conn = self._get_connection()
        try:
            if ctx:
                self.bind_tenant(conn, ctx.account_id)

            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            with conn.cursor() as cur:
                # Get total recalls and unique URIs
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_recalls,
                        COUNT(DISTINCT uri) as unique_uris,
                        AVG(score) as avg_score
                    FROM dream_recalls
                    WHERE account_id = %s AND recalled_at >= %s
                    """,
                    (account_id, cutoff),
                )
                row = cur.fetchone()
                total_recalls, unique_uris, avg_score = row or (0, 0, None)

                # Get category breakdown
                cur.execute(
                    """
                    SELECT category, COUNT(*) as cnt
                    FROM dream_recalls
                    WHERE account_id = %s AND recalled_at >= %s
                        AND category IS NOT NULL
                    GROUP BY category
                    ORDER BY cnt DESC
                    """,
                    (account_id, cutoff),
                )
                top_categories = {cat: count for cat, count in cur.fetchall()}

                return {
                    "total_recalls": total_recalls,
                    "unique_uris": unique_uris,
                    "top_categories": top_categories,
                    "avg_score": float(avg_score) if avg_score else None,
                }
        except Exception as exc:
            logger.error(f"Failed to get recall stats: {exc}")
            return {
                "total_recalls": 0,
                "unique_uris": 0,
                "top_categories": {},
                "avg_score": None,
            }
        finally:
            conn.rollback()
            self._return_connection(conn)

    def prune(
        self,
        before_days: int = 90,
        ctx: RequestContext | None = None,
    ) -> int:
        """Prune recall events older than the specified days.

        Returns the number of rows deleted.
        """
        conn = self._get_connection()
        try:
            if ctx:
                self.bind_tenant(conn, ctx.account_id)

            cutoff = datetime.now(timezone.utc) - timedelta(days=before_days)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM dream_recalls
                    WHERE recalled_at < %s
                    """,
                    (cutoff,),
                )
                deleted = cur.rowcount
            conn.commit()
            logger.info(f"Pruned {deleted} recall events older than {before_days} days")
            return deleted
        except Exception as exc:
            conn.rollback()
            logger.error(f"Failed to prune recall events: {exc}")
            return 0
        finally:
            self._return_connection(conn)
