"""PostgreSQL implementation of ContextFS protocol.

Stores all node data (content, abstract, overview, relations, metadata)
in a single `context_nodes` table. A single INSERT ON CONFLICT replaces
the AGFS atomic 4-step write order — PostgreSQL transactions guarantee
atomicity.

Follows the same psycopg2 connection pool pattern as OpenGaussVectorIndex.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from urllib.parse import unquote

from core.enums import NodeStatus
from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    NodeBrokenError,
    NodeNotFoundError,
)
from core.models import ContextNode, OutboxEvent, RelationEdge, RequestContext
from fs.access_control import check_uri_access
from fs.sql_adapter.pool import PoolAdapterMixin, SharedConnectionPool

# ---------------------------------------------------------------------------
# URI parsing (independent copy — no dependency on agfs_adapter)
# ---------------------------------------------------------------------------

_SKILL_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/skills/(?P<skill_name>[^/]+)$'
)
_PROFILE_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/users/(?P<owner_id>[^/]+)/memories/profile$'
)
_MEMORY_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/'
    r'memories/(?P<category>[^/]+)/(?P<slug>[^/]+)$'
)
_CATEGORY_DIR_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/'
    r'memories/(?P<category>[^/]+)/?$'
)
_MEMORIES_ROOT_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/memories/?$'
)
_SESSION_ARCHIVE_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/(?P<archive_id>[^/]+)$'
)
_SESSION_HISTORY_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/?$'
)
_SESSION_STATE_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/state(?:\.json)?$'
)


def parse_uri(uri: str) -> dict:
    """Parse a ContextEngine URI into its components."""
    match = _SKILL_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': 'agents',
                'owner_id': g['owner_id'], 'category': 'skills',
                'slug': unquote(g['skill_name'])}

    match = _PROFILE_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': 'users',
                'owner_id': g['owner_id'], 'category': 'profile',
                'slug': 'profile'}

    match = _MEMORY_URI_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': g['owner_type'],
                'owner_id': g['owner_id'], 'category': g['category'],
                'slug': unquote(g['slug'])}

    stripped = uri.rstrip('/')
    match = _CATEGORY_DIR_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': g['owner_type'],
                'owner_id': g['owner_id'], 'category': g['category'],
                'slug': ''}

    match = _MEMORIES_ROOT_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': g['owner_type'],
                'owner_id': g['owner_id'], 'category': '', 'slug': ''}

    match = _SESSION_ARCHIVE_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': 'sessions',
                'owner_id': g['session_id'], 'category': 'history',
                'slug': g['archive_id']}

    match = _SESSION_HISTORY_PATTERN.match(stripped)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': 'sessions',
                'owner_id': g['session_id'], 'category': 'history',
                'slug': ''}

    match = _SESSION_STATE_PATTERN.match(uri)
    if match:
        g = match.groupdict()
        return {'account': g['account'], 'owner_type': 'sessions',
                'owner_id': g['session_id'], 'category': 'state',
                'slug': 'state'}

    raise ValueError(f"Invalid URI format: {uri}")

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None
    Json = None

logger = logging.getLogger(__name__)


class SQLContextFS(PoolAdapterMixin):
    """PostgreSQL-backed implementation of ContextFS.

    Each ContextNode maps to a single row in ``context_nodes``.
    The atomic 4-step write from AGFS is replaced by a single SQL
    transaction (INSERT … ON CONFLICT), which is inherently atomic.

    Usage::

        fs = SQLContextFS("host=localhost dbname=ogmemory")
        fs.write_node(node, ctx)
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool: SharedConnectionPool | None = None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLContextFS. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Connection pool (same pattern as OpenGaussVectorIndex)
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
    # Schema management
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
    # Access control (reuses parse_uri from agfs_adapter)
    # ------------------------------------------------------------------

    def _is_accessible(self, uri: str, ctx: RequestContext) -> bool:
        """Check URI accessibility with visible_owner_spaces support (SQL mode)."""
        try:
            components = parse_uri(uri)
            return check_uri_access(ctx, components, strict_mode=False)
        except ValueError:
            return False

    def _ensure_accessible(self, uri: str, ctx: RequestContext) -> None:
        if not self._is_accessible(uri, ctx):
            try:
                components = parse_uri(uri)
                owner_id = components.get("owner_id", "")
                if owner_id:
                    raise AccessDeniedError(
                        uri,
                        ctx.account_id,
                        f"owner mismatch: URI owner={owner_id}, request user={ctx.user_id}",
                    )
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    f"URI belongs to account '{components['account']}'",
                )
            except ValueError:
                raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format")

    def _extract_account_id(self, uri: str) -> str:
        """Extract account_id from URI for row-level storage."""
        components = parse_uri(uri)
        return components["account"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_row_data(self, node: ContextNode) -> tuple[list, dict, int, str]:
        """Prepare relations JSON, metadata dict, new version, and timestamp."""
        now = datetime.now(UTC).isoformat()
        relations = node.metadata.get("_relations", [])
        relations_json = [
            {
                "from_uri": e.from_uri,
                "to_uri": e.to_uri,
                "relation_type": e.relation_type,
                "weight": e.weight,
                "reason": e.reason,
            }
            for e in relations
        ]

        extra_meta = {
            k: v for k, v in node.metadata.items() if not k.startswith("_")
        }
        extra_meta.setdefault("created_at", now)
        extra_meta["updated_at"] = now

        expected_version = node.metadata.get("expected_version")
        if expected_version is not None:
            new_version = expected_version + 1
        else:
            new_version = node.metadata.get("version", 1)
        extra_meta["version"] = new_version

        return relations_json, extra_meta, new_version, now

    # ------------------------------------------------------------------
    # ContextFS Protocol implementation
    # ------------------------------------------------------------------

    def _execute_write(self, cur, node, account_id, expected_version,
                       relations_json, extra_meta, new_version, now):
        """Execute the INSERT/UPDATE SQL for a node write on the given cursor.

        Shared by write_node and write_node_with_outbox to avoid duplication.
        """
        if expected_version is not None:
            cur.execute(
                """
                UPDATE context_nodes SET
                    status       = 'ACTIVE',
                    content      = %s,
                    abstract     = %s,
                    overview     = %s,
                    relations    = %s,
                    metadata     = %s,
                    version      = version + 1,
                    updated_at   = NOW()
                WHERE uri = %s AND version = %s
                """,
                (
                    node.content, node.abstract, node.overview,
                    Json(relations_json), Json(extra_meta),
                    node.uri, expected_version,
                ),
            )
            if cur.rowcount == 0:
                cur.execute(
                    "SELECT version FROM context_nodes WHERE uri = %s",
                    (node.uri,),
                )
                row = cur.fetchone()
                actual = row[0] if row else 0
                raise ConcurrentModificationError(
                    node.uri, expected_version, actual
                )
        else:
            cur.execute(
                """
                INSERT INTO context_nodes
                    (uri, account_id, owner_space, category, context_type,
                     level, status, content, abstract, overview,
                     relations, metadata, version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s,
                        %s, %s, %s, %s, NOW())
                ON CONFLICT (uri) DO UPDATE SET
                    status       = 'ACTIVE',
                    content      = EXCLUDED.content,
                    abstract     = EXCLUDED.abstract,
                    overview     = EXCLUDED.overview,
                    relations    = EXCLUDED.relations,
                    metadata     = EXCLUDED.metadata,
                    version      = EXCLUDED.version,
                    updated_at   = NOW()
                """,
                (
                    node.uri, account_id, node.owner_space,
                    node.category, node.context_type, node.level,
                    node.content, node.abstract, node.overview,
                    Json(relations_json), Json(extra_meta),
                    new_version, extra_meta.get("created_at", now),
                ),
            )

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node — atomic upsert with optimistic locking.

        For merge operations (expected_version present), uses a single
        UPDATE … WHERE version = expected_version to avoid TOCTOU races.

        For initial creates, uses INSERT … ON CONFLICT.

        Args:
            node: ContextNode to write.
            ctx:  RequestContext for access control.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            ConcurrentModificationError: On optimistic lock failure.
        """
        self._ensure_accessible(node.uri, ctx)
        account_id = self._extract_account_id(node.uri)
        expected_version = node.metadata.get("expected_version")
        relations_json, extra_meta, new_version, now = self._build_row_data(node)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._execute_write(
                    cur, node, account_id, expected_version,
                    relations_json, extra_meta, new_version, now,
                )
            conn.commit()
        except ConcurrentModificationError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def write_node_with_outbox(self, node: ContextNode, ctx: RequestContext,
                               outbox_event: OutboxEvent | None = None) -> None:
        """Write a node and register an outbox event in a single transaction.

        Both the context_nodes INSERT/UPDATE and the outbox_events INSERT
        execute on the same connection within one commit, guaranteeing
        atomicity: either both succeed or both roll back.

        Args:
            node: ContextNode to write.
            ctx:  RequestContext for access control.
            outbox_event: Pre-built OutboxEvent (from SQLOutboxStore.build_write_event).
                         If None, only the node is written (delegates to write_node).

        Raises:
            AccessDeniedError: If account_id doesn't match.
            ConcurrentModificationError: On optimistic lock failure.
        """
        if outbox_event is None:
            return self.write_node(node, ctx)

        self._ensure_accessible(node.uri, ctx)
        account_id = self._extract_account_id(node.uri)
        expected_version = node.metadata.get("expected_version")
        relations_json, extra_meta, new_version, now = self._build_row_data(node)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Step 1: Write the node (shared logic)
                self._execute_write(
                    cur, node, account_id, expected_version,
                    relations_json, extra_meta, new_version, now,
                )

                # Step 2: Register outbox event in same transaction.
                # pg_notify wakes the outbox worker so it processes the
                # event promptly instead of waiting for the next poll.
                cur.execute(
                    """
                    INSERT INTO outbox_events
                        (event_id, event_type, uri, account_id, payload,
                         status, retry_count, created_at, next_retry_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        outbox_event.event_id,
                        outbox_event.event_type,
                        outbox_event.uri,
                        account_id,
                        Json(outbox_event.payload),
                        outbox_event.status,
                        outbox_event.retry_count,
                        outbox_event.created_at or datetime.now(UTC).isoformat(),
                        None,
                    ),
                )
                cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    (
                        "ogmem_outbox",
                        json.dumps(
                            {
                                "event_id": outbox_event.event_id,
                                "account_id": account_id,
                                "event_type": outbox_event.event_type,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            conn.commit()
        except ConcurrentModificationError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from PostgreSQL.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist or status != ACTIVE.
            NodeBrokenError: If node is in BROKEN state.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT uri, context_type, category, level, owner_space,
                           abstract, overview, content, relations, metadata,
                           status, created_at, updated_at, version
                    FROM context_nodes
                    WHERE uri = %s
                    """,
                    (uri,),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        if row is None:
            raise NodeNotFoundError(uri)

        (
            uri_val,
            context_type,
            category,
            level,
            owner_space,
            abstract,
            overview,
            content,
            relations_raw,
            metadata_raw,
            status,
            created_at,
            updated_at,
            version,
        ) = row

        if status == NodeStatus.BROKEN.value:
            raise NodeBrokenError(uri)
        if status != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        # Parse relations
        relations = []
        if relations_raw:
            rel_list = (
                json.loads(relations_raw)
                if isinstance(relations_raw, str)
                else relations_raw
            )
            relations = [
                RelationEdge(
                    from_uri=r["from_uri"],
                    to_uri=r["to_uri"],
                    relation_type=r["relation_type"],
                    weight=r["weight"],
                    reason=r["reason"],
                )
                for r in rel_list
            ]

        # Parse metadata
        meta = (
            json.loads(metadata_raw)
            if isinstance(metadata_raw, str)
            else (metadata_raw or {})
        )
        meta.setdefault("created_at", str(created_at) if created_at else "")
        meta.setdefault("updated_at", str(updated_at) if updated_at else "")
        meta.setdefault("version", version)
        meta["_relations"] = relations

        return ContextNode(
            uri=uri_val,
            context_type=context_type,
            category=category,
            level=level,
            owner_space=owner_space,
            abstract=abstract or "",
            overview=overview or "",
            content=content or "",
            metadata=meta,
        )

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if node exists and is ACTIVE."""
        if not self._is_accessible(uri, ctx):
            return False

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM context_nodes "
                    "WHERE uri = %s AND status = 'ACTIVE' LIMIT 1",
                    (uri,),
                )
                return cur.fetchone() is not None
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node.

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM context_nodes WHERE uri = %s",
                    (uri,),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(uri)
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def archive_node(self, uri: str, ctx: RequestContext) -> None:
        """Archive a node (soft delete). Marks status as ARCHIVED.

        Args:
            uri: URI of the node to archive
            ctx: RequestContext for this operation

        Raises:
            AccessDeniedError: If account_id doesn't match.
            NodeNotFoundError: If node doesn't exist.
        """
        self._ensure_accessible(uri, ctx)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE context_nodes SET status = 'ARCHIVED', "
                    "metadata = metadata || %s, "
                    "updated_at = NOW() "
                    "WHERE uri = %s AND status = 'ACTIVE'",
                    (
                        Json({"archived_at": datetime.now(UTC).isoformat()}),
                        uri,
                    ),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(uri)
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List immediate child URIs under a given parent URI.

        Finds all rows whose URI starts with ``parent_uri/`` and has
        exactly one additional path segment.
        """
        self._ensure_accessible(uri, ctx)
        account_id = self._extract_account_id(uri)

        # Normalise parent: ensure no trailing slash for prefix matching
        parent = uri.rstrip("/")
        prefix = parent + "/"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT uri FROM context_nodes
                    WHERE uri LIKE %s
                      AND account_id = %s
                      AND status = 'ACTIVE'
                    """,
                    (prefix + "%", account_id),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()  # Reset SET LOCAL for RLS
            self._return_connection(conn)

        children = []
        for (child_uri,) in rows:
            # Extract the segment after the parent prefix
            remainder = child_uri[len(prefix) :]
            # Only include immediate children (no further slashes)
            if "/" not in remainder:
                children.append(child_uri)

        return children

    def move_node(
        self, from_uri: str, to_uri: str, ctx: RequestContext
    ) -> None:
        """Move/rename a node (and cascade to children + relation/outbox refs).

        Raises:
            AccessDeniedError: If either URI account doesn't match.
            NodeNotFoundError: If source node doesn't exist.
        """
        self._ensure_accessible(from_uri, ctx)
        self._ensure_accessible(to_uri, ctx)
        account_id = self._extract_account_id(from_uri)

        from_prefix = from_uri.rstrip("/") + "/"
        to_prefix = to_uri.rstrip("/") + "/"

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # 1) Move the node itself
                cur.execute(
                    "UPDATE context_nodes SET uri = %s, updated_at = NOW() "
                    "WHERE uri = %s",
                    (to_uri, from_uri),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(from_uri)

                # 2) Cascade: update children with the old prefix
                cur.execute(
                    "UPDATE context_nodes "
                    "SET uri = concat(%s, substring(uri from %s)), "
                    "    updated_at = NOW() "
                    "WHERE uri LIKE %s",
                    (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                )

                # 3) Cascade: update relation_edges (only if table exists)
                cur.execute(
                    "SELECT to_regclass('relation_edges')"
                )
                if cur.fetchone()[0] is not None:
                    cur.execute(
                        "UPDATE relation_edges SET from_uri = %s "
                        "WHERE from_uri = %s",
                        (to_uri, from_uri),
                    )
                    cur.execute(
                        "UPDATE relation_edges SET to_uri = %s "
                        "WHERE to_uri = %s",
                        (to_uri, from_uri),
                    )
                    cur.execute(
                        "UPDATE relation_edges "
                        "SET from_uri = concat(%s, substring(from_uri from %s)) "
                        "WHERE from_uri LIKE %s",
                        (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                    )
                    cur.execute(
                        "UPDATE relation_edges "
                        "SET to_uri = concat(%s, substring(to_uri from %s)) "
                        "WHERE to_uri LIKE %s",
                        (to_prefix, len(from_prefix) + 1, from_prefix + "%"),
                    )

                # 4) Cascade: update outbox_events (only if table exists)
                cur.execute(
                    "SELECT to_regclass('outbox_events')"
                )
                if cur.fetchone()[0] is not None:
                    import uuid as _uuid

                    # --- 4a) Delete PENDING and PROCESSING events for old URIs.
                    #
                    # We delete both PENDING and PROCESSING events.  The
                    # PROCESSING events may have been claimed by a worker that
                    # has read the payload into memory.  However, if we leave
                    # them, the worker will upsert stale records at the old URI
                    # *after* our DELETE_CONTEXT runs, undoing the move.
                    #
                    # By deleting the rows, the worker's final mark_done DELETE
                    # matches 0 rows (harmless), and we rely on the tombstone
                    # check in _process_upsert_context to catch the rare case
                    # where the worker has already started processing.
                    cur.execute(
                        "DELETE FROM outbox_events "
                        "WHERE uri = %s AND status IN ('PENDING', 'PROCESSING')",
                        (from_uri,),
                    )
                    cur.execute(
                        "DELETE FROM outbox_events "
                        "WHERE uri LIKE %s AND status IN ('PENDING', 'PROCESSING')",
                        (from_prefix + "%",),
                    )

                    # --- 4b) Emit DELETE_CONTEXT for old URIs.
                    #
                    # Build ids using the canonical IndexRecord.generate_id().
                    from core.models import IndexRecord as _IndexRecord

                    def _record_id(uri: str, level: int) -> str:
                        return _IndexRecord.generate_id(uri, level)

                    def _parent_uri(uri: str) -> str:
                        """Derive parent URI (same logic as ContextNode.parent_uri)."""
                        s = uri.rstrip("/")
                        if "/" not in s:
                            return ""
                        return s.rsplit("/", 1)[0] + "/"

                    # Collect ALL descendants (not just immediate children).
                    # The context_nodes UPDATE already moved every row whose
                    # URI started with from_prefix, so we reconstruct old
                    # URIs by swapping the prefix back.
                    cur.execute(
                        "SELECT uri FROM context_nodes "
                        "WHERE uri = %s OR uri LIKE %s",
                        (to_uri, to_prefix + "%"),
                    )
                    all_new_uris = [row[0] for row in cur.fetchall()]

                    for new_uri in all_new_uris:
                        old_uri = from_uri if new_uri == to_uri \
                            else from_prefix + new_uri[len(to_prefix):]
                        # Build ids for L0, L1, L2 (with /content.md suffix)
                        ids_to_delete = [
                            _record_id(old_uri, 0),
                            _record_id(old_uri, 1),
                            _record_id(old_uri.rstrip("/") + "/content.md", 2),
                        ]
                        cur.execute(
                            """
                            INSERT INTO outbox_events
                                (event_id, event_type, uri, account_id,
                                 payload, status, retry_count, created_at)
                            VALUES (%s, 'DELETE_CONTEXT', %s, %s,
                                    %s, 'PENDING', 0, NOW())
                            """,
                            (
                                str(_uuid.uuid4()),
                                old_uri,
                                account_id,
                                Json({"ids_to_delete": ids_to_delete}),
                            ),
                        )

                    # --- 4c) Emit UPSERT_CONTEXT for new URIs.
                    #
                    # Build IndexRecords directly from context_nodes data
                    # using hashlib (no index-package dependency).  The
                    # outbox worker embeds the text and upserts to the
                    # vector index, making the node immediately searchable
                    # at its new URI without waiting for a next write.
                    cur.execute(
                        "SELECT uri, abstract, overview, content, "
                        "owner_space, category, context_type "
                        "FROM context_nodes "
                        "WHERE uri = %s OR uri LIKE %s",
                        (to_uri, to_prefix + "%"),
                    )
                    moved_rows = cur.fetchall()

                    for row in moved_rows:
                        r_uri, r_abstract, r_overview, r_content, \
                            r_owner, r_cat, r_ct = row

                        records = []
                        base_filters = {
                            "account_id": account_id,
                            "owner_space": r_owner,
                            "category": r_cat,
                            "context_type": r_ct,
                        }
                        # Shared metadata — must match build_index_records()
                        base_meta = {
                            "category": r_cat,
                            "context_type": r_ct,
                            "parent_uri": _parent_uri(r_uri),
                            "has_overview": bool(r_overview),
                            "has_content": bool(r_content),
                        }
                        # L0: abstract (always present)
                        records.append({
                            "id": _record_id(r_uri, 0),
                            "uri": r_uri, "level": 0,
                            "text": r_abstract or "",
                            "filters": dict(base_filters),
                            "metadata": {"level": "abstract",
                                         **base_meta},
                        })
                        # L1: overview
                        if r_overview:
                            records.append({
                                "id": _record_id(r_uri, 1),
                                "uri": r_uri, "level": 1,
                                "text": r_overview,
                                "filters": dict(base_filters),
                                "metadata": {"level": "overview",
                                             **base_meta},
                            })
                        # L2: content (id uses /content.md suffix)
                        l2_uri = r_uri.rstrip("/") + "/content.md"
                        if r_content:
                            records.append({
                                "id": _record_id(l2_uri, 2),
                                "uri": l2_uri, "level": 2,
                                "text": r_content,
                                "filters": dict(base_filters),
                                "metadata": {"level": "content",
                                             **base_meta,
                                             "parent_uri": r_uri},
                            })

                        if records:
                            cur.execute(
                                """
                                INSERT INTO outbox_events
                                    (event_id, event_type, uri, account_id,
                                     payload, status, retry_count, created_at)
                                VALUES (%s, 'UPSERT_CONTEXT', %s, %s,
                                        %s, 'PENDING', 0, NOW())
                                """,
                                (
                                    str(_uuid.uuid4()),
                                    r_uri,
                                    account_id,
                                    Json({"records": records}),
                                ),
                            )
            conn.commit()
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
