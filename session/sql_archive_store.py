"""PostgreSQL-backed SessionArchiveStore implementation.

Stores session archives in a ``session_archives`` table instead of
per-archive AGFS directories.  Same public interface as
``session.archive_store.SessionArchiveStore``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from core.models import RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin
from session.models import ArchiveEntry, ArchiveWriteResult

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)


class SQLSessionArchiveStore(PoolAdapterMixin):
    """PostgreSQL-backed session archive storage.

    Provides the same ``write_archive``, ``list_archives``,
    ``read_archive``, and ``read_archive_abstract`` methods as the
    AGFS-based ``SessionArchiveStore``, but backed by a single
    ``session_archives`` table.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLSessionArchiveStore. "
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
    # Public API (same signatures as SessionArchiveStore)
    # ------------------------------------------------------------------

    def write_archive(
        self,
        session_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
        archive_id: str | None = None,
    ) -> ArchiveWriteResult:
        """Write a session archive to PostgreSQL.

        Uses INSERT ON CONFLICT for idempotent upsert.
        """
        if archive_id is None:
            archive_id = (
                f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
                f"_{uuid.uuid4().hex[:8]}"
            )

        uri = (
            f"ctx://{ctx.account_id}/sessions/{session_id}"
            f"/history/{archive_id}"
        )
        now = datetime.now(UTC).isoformat()

        metadata = {
            "archive_id": archive_id,
            "session_id": session_id,
            "created_at": now,
            "message_count": len(messages),
        }

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_archives
                        (archive_id, session_id, account_id,
                         abstract, overview, messages, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (account_id, session_id, archive_id) DO UPDATE SET
                        abstract  = EXCLUDED.abstract,
                        overview  = EXCLUDED.overview,
                        messages  = EXCLUDED.messages,
                        metadata  = EXCLUDED.metadata,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        archive_id,
                        session_id,
                        ctx.account_id,
                        abstract or "",
                        overview or "",
                        Json(messages),
                        Json(metadata),
                        now,
                    ),
                )
            conn.commit()
            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=True,
                created_at=now,
            )
        except Exception as exc:
            conn.rollback()
            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=False,
                error=str(exc),
            )
        finally:
            self._return_connection(conn)

    def list_archives(
        self, session_id: str, ctx: RequestContext
    ) -> list[ArchiveEntry]:
        """List all archives for a session, newest first."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE account_id = %s AND session_id = %s
                    ORDER BY created_at DESC
                    """,
                    (ctx.account_id, session_id),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            conn.rollback()
            self._return_connection(conn)

        entries: list[ArchiveEntry] = []
        for row in rows:
            (
                archive_id,
                sess_id,
                abstract,
                overview,
                messages_raw,
                metadata_raw,
                created_at,
            ) = row

            meta = (
                json.loads(metadata_raw)
                if isinstance(metadata_raw, str)
                else (metadata_raw or {})
            )

            entries.append(
                ArchiveEntry(
                    archive_id=archive_id,
                    session_id=sess_id,
                    overview=overview or "",
                    abstract=abstract or "",
                    messages=[],  # Don't include full messages in list
                    created_at=str(created_at) if created_at else "",
                    metadata=meta,
                )
            )
        return entries

    def list_archives_since(
        self,
        since: datetime,
        ctx: RequestContext,
        limit: int = 50,
    ) -> list[ArchiveEntry]:
        """List archives created since a given timestamp, across all sessions.

        Args:
            since: Only return archives created after this timestamp
            ctx: RequestContext for account_id
            limit: Maximum number of archives to return (default 50)

        Returns:
            List of ArchiveEntry with messages included, ordered by created_at DESC
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE account_id = %s AND created_at > %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (ctx.account_id, since, limit),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            conn.rollback()
            self._return_connection(conn)

        entries: list[ArchiveEntry] = []
        for row in rows:
            (
                archive_id,
                sess_id,
                abstract,
                overview,
                messages_raw,
                metadata_raw,
                created_at,
            ) = row

            # Parse messages (include full messages for dreaming)
            if isinstance(messages_raw, str):
                messages = json.loads(messages_raw)
            elif messages_raw is not None:
                messages = list(messages_raw)
            else:
                messages = []

            meta = (
                json.loads(metadata_raw)
                if isinstance(metadata_raw, str)
                else (metadata_raw or {})
            )

            entries.append(
                ArchiveEntry(
                    archive_id=archive_id,
                    session_id=sess_id,
                    overview=overview or "",
                    abstract=abstract or "",
                    messages=messages,  # Include full messages for dreaming
                    created_at=str(created_at) if created_at else "",
                    metadata=meta,
                )
            )
        return entries

    def read_archive(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> ArchiveEntry | None:
        """Read a full session archive including messages."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT archive_id, session_id, abstract, overview,
                           messages, metadata, created_at
                    FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                row = cur.fetchone()
        except Exception:
            return None
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        (
            aid,
            sess_id,
            abstract,
            overview,
            messages_raw,
            metadata_raw,
            created_at,
        ) = row

        # Parse messages
        if isinstance(messages_raw, str):
            messages = json.loads(messages_raw)
        elif messages_raw is not None:
            messages = list(messages_raw)
        else:
            messages = []

        meta = (
            json.loads(metadata_raw)
            if isinstance(metadata_raw, str)
            else (metadata_raw or {})
        )

        return ArchiveEntry(
            archive_id=aid,
            session_id=sess_id,
            overview=overview or "",
            abstract=abstract or "",
            messages=messages,
            created_at=str(created_at) if created_at else "",
            metadata=meta,
        )

    def read_archive_abstract(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> str | None:
        """Read only the abstract from an archive."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT abstract FROM session_archives
                    WHERE archive_id = %s AND session_id = %s AND account_id = %s
                    """,
                    (archive_id, session_id, ctx.account_id),
                )
                row = cur.fetchone()
        except Exception:
            return None
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None
        return row[0] or None
