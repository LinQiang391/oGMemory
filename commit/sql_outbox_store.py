"""PostgreSQL-backed OutboxStore implementation.

Replaces file-based .outbox/*.json events with rows in outbox_events table.
Uses ``LISTEN/NOTIFY`` for low-latency wakeups and ``UPDATE … RETURNING`` for
atomic batch claims instead of filesystem lock files.
"""

import json
import uuid
from datetime import UTC, datetime

from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger
from core.models import ContextNode, IndexRecord, OutboxEvent, RequestContext

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

logger = get_logger(__name__)


class SQLOutboxStore:
    """PostgreSQL-backed outbox event storage.

    Stores OutboxEvents as rows in the ``outbox_events`` table.
    Locking uses ``UPDATE … RETURNING`` for atomic batch claims.
    """

    supports_batch_claim = True
    notify_channel = "ogmem_outbox"

    def __init__(
        self,
        connection_string: str | None = None,
        fs: ContextFS | None = None,
        pool_size: int = 5,
        pool=None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLOutboxStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._fs = fs
        from fs.sql_adapter.pool import PoolAdapterMixin
        PoolAdapterMixin._init_pool(self, pool, connection_string, pool_size)
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
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_account_id(self, uri: str) -> str:
        from fs.sql_adapter.sql_context_fs import parse_uri
        return parse_uri(uri)["account"]

    def _node_uri_to_directory_uri(self, node_uri: str) -> str:
        parts = node_uri.rstrip("/").rsplit("/", 1)
        return parts[0] + "/"

    def _collect_sibling_abstracts(
        self, node_uri: str, ctx: RequestContext
    ) -> list[str]:
        dir_uri = self._node_uri_to_directory_uri(node_uri)
        try:
            siblings = self._fs.list_children(dir_uri, ctx)
        except Exception:
            return []
        abstracts = []
        for sibling_uri in siblings[:20]:
            try:
                node = self._fs.read_node(sibling_uri, ctx)
                if node and node.abstract:
                    abstracts.append(node.abstract[:200])
            except Exception:
                continue
        return abstracts

    def _insert_event(self, event: OutboxEvent, account_id: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO outbox_events
                        (event_id, event_type, uri, account_id, payload,
                         status, retry_count, created_at, next_retry_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        event.event_type,
                        event.uri,
                        account_id,
                        Json(event.payload),
                        event.status,
                        event.retry_count,
                        event.created_at or datetime.now(UTC).isoformat(),
                        event.next_retry_at or None,
                    ),
                )
                cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    (
                        self.notify_channel,
                        json.dumps(
                            {
                                "event_id": event.event_id,
                                "account_id": account_id,
                                "event_type": event.event_type,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _row_to_event(self, row) -> tuple[str, OutboxEvent]:
        # Rows from claim_batch include a trailing seq column;
        # rows from list_pending do not.  Slice uniformly.
        (
            event_id,
            event_type,
            uri,
            payload_raw,
            status,
            retry_count,
            created_at,
            next_retry_at,
        ) = row[:8]
        payload = (
            json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
        ) or {}
        event = OutboxEvent(
            event_id=event_id,
            event_type=event_type,
            uri=uri,
            payload=payload,
            status=status,
            retry_count=retry_count,
            created_at=str(created_at) if created_at else "",
            next_retry_at=str(next_retry_at) if next_retry_at else "",
        )
        return uri, event

    # ------------------------------------------------------------------
    # OutboxStore Protocol
    # ------------------------------------------------------------------

    def build_write_event(self, node: ContextNode) -> OutboxEvent:
        """Build an OutboxEvent for a node write without persisting it.

        Used by SQLContextFS.write_node_with_outbox to insert the event
        in the same transaction as the node write.
        """
        from index.index_record_builder import build_index_records

        index_records = build_index_records(node)
        records_data = [
            {
                "id": r.id,
                "uri": r.uri,
                "level": r.level,
                "text": r.text,
                "filters": r.filters,
                "metadata": r.metadata,
            }
            for r in index_records
        ]

        return OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.UPSERT_CONTEXT.value,
            uri=node.uri,
            payload={"records": records_data},
            status="PENDING",
        )

    def register_write(
        self, node: ContextNode, ctx: RequestContext
    ) -> OutboxEvent:
        event = self.build_write_event(node)
        account_id = self._extract_account_id(node.uri)
        self._insert_event(event, account_id)
        return event

    def register_delete(
        self, uri: str, ctx: RequestContext
    ) -> OutboxEvent:
        # Pre-compute ids_to_delete so the worker always has correct L2
        # ids (L2 uses uri/content.md suffix, not bare uri).
        l2_uri = uri.rstrip("/") + "/content.md"
        ids_to_delete = [
            IndexRecord.generate_id(uri, 0),
            IndexRecord.generate_id(uri, 1),
            IndexRecord.generate_id(l2_uri, 2),
        ]
        event = OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DELETE_CONTEXT.value,
            uri=uri,
            payload={"ids_to_delete": ids_to_delete},
            status="PENDING",
        )
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    def register_archive(
        self, uri: str, ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Pre-computes ids_to_delete so the worker can remove vector records
        for the archived node.

        Args:
            uri: URI of the archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        from core.models import IndexRecord

        l2_uri = uri.rstrip("/") + "/content.md"
        ids_to_delete = [
            IndexRecord.generate_id(uri, 0),
            IndexRecord.generate_id(uri, 1),
            IndexRecord.generate_id(l2_uri, 2),
        ]
        event = OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ARCHIVE_CONTEXT.value,
            uri=uri,
            payload={"ids_to_delete": ids_to_delete},
            status="PENDING",
        )
        account_id = self._extract_account_id(uri)
        self._insert_event(event, account_id)
        return event

    def register_directory(
        self, node: ContextNode, ctx: RequestContext
    ) -> OutboxEvent:
        dir_uri = self._node_uri_to_directory_uri(node.uri)
        abstracts = self._collect_sibling_abstracts(node.uri, ctx)

        payload = {
            "directory_uri": dir_uri,
            "child_abstracts": abstracts,
            "filters": {
                "account_id": ctx.account_id,
                "owner_space": node.owner_space,
            },
        }

        event = OutboxEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.UPSERT_DIRECTORY.value,
            uri=dir_uri,
            payload=payload,
            status="PENDING",
        )
        account_id = self._extract_account_id(node.uri)
        self._insert_event(event, account_id)
        return event

    def list_pending(self, account_id: str) -> list[tuple[str, OutboxEvent]]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, uri, payload, status,
                           retry_count, created_at, next_retry_at
                    FROM outbox_events
                    WHERE account_id = %s
                      AND status = 'PENDING'
                      AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                    ORDER BY created_at ASC
                    """,
                    (account_id,),
                )
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def claim_batch(
        self,
        worker_id: str,
        limit: int = 100,
        timeout_seconds: int = 300,
    ) -> list[tuple[str, OutboxEvent]]:
        """Atomically claim a batch of ready events across all accounts."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH ready AS (
                        SELECT event_id
                        FROM outbox_events
                        WHERE (
                                status = 'PENDING'
                                AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                              )
                           OR (
                                status = 'PROCESSING'
                                AND locked_at < NOW() - %s * interval '1 second'
                              )
                        ORDER BY seq ASC NULLS LAST, created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT %s
                    ),
                    claimed AS (
                        UPDATE outbox_events oe
                        SET status = 'PROCESSING',
                            worker_id = %s,
                            locked_at = NOW()
                        FROM ready
                        WHERE oe.event_id = ready.event_id
                        RETURNING oe.event_id, oe.event_type, oe.uri, oe.payload,
                                  oe.status, oe.retry_count, oe.created_at, oe.next_retry_at,
                                  oe.seq
                    )
                    SELECT * FROM claimed ORDER BY seq ASC NULLS LAST
                    """,
                    (timeout_seconds, limit, worker_id),
                )
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return [self._row_to_event(row) for row in rows]

    def open_listener_connection(self):
        """Create a dedicated autocommit connection for LISTEN/NOTIFY."""
        conn = psycopg2.connect(self._connection_string)
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute(f"LISTEN {self.notify_channel}")
        return conn

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET status = 'PROCESSING' "
                    "WHERE event_id = %s",
                    (event.event_id,),
                )
            conn.commit()
            event.status = "PROCESSING"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def is_event_current(self, event_id: str) -> bool:
        """Check whether an event still exists in the outbox table.

        Returns True if the event row exists, False if it was deleted
        (e.g. by move_node).  Raises on database errors so the caller
        can distinguish "tombstoned" from "DB unavailable".
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM outbox_events WHERE event_id = %s",
                    (event_id,),
                )
                return cur.fetchone() is not None
        finally:
            self._return_connection(conn)

    def mark_done(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM outbox_events WHERE event_id = %s",
                    (event.event_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def move_to_dlq(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET status = 'FAILED' "
                    "WHERE event_id = %s",
                    (event.event_id,),
                )
            conn.commit()
            event.status = "FAILED"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def increment_retry(
        self,
        event: OutboxEvent,
        node_uri: str,
        next_retry_at: datetime | None = None,
    ) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET retry_count = retry_count + 1,
                        status = 'PENDING',
                        next_retry_at = %s,
                        worker_id = NULL,
                        locked_at = NULL
                    WHERE event_id = %s
                    """,
                    (next_retry_at, event.event_id),
                )
            conn.commit()
            event.retry_count += 1
            event.status = "PENDING"
            if next_retry_at:
                event.next_retry_at = next_retry_at.isoformat()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def try_acquire(
        self,
        event: OutboxEvent,
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Atomically claim an event using UPDATE … RETURNING."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PROCESSING',
                        worker_id = %s,
                        locked_at = NOW()
                    WHERE event_id = %s
                      AND (
                          status = 'PENDING'
                          OR (status = 'PROCESSING' AND locked_at < NOW() - %s * interval '1 second')
                      )
                    RETURNING event_id
                    """,
                    (worker_id, event.event_id, timeout_seconds),
                )
                acquired = cur.fetchone() is not None
            conn.commit()
            if acquired:
                event.status = "PROCESSING"
            return acquired
        except Exception:
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def release(self, event: OutboxEvent, node_uri: str) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbox_events
                    SET status = 'PENDING', worker_id = NULL, locked_at = NULL
                    WHERE event_id = %s AND status = 'PROCESSING'
                    """,
                    (event.event_id,),
                )
            conn.commit()
            event.status = "PENDING"
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
