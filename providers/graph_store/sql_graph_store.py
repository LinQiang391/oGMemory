"""PostgreSQL implementation of GraphStore protocol.

Stores heterogeneous graph nodes and edges in PostgreSQL with
incremental PageRank computation.

Usage::

    from providers.graph_store import SQLGraphStore
    from core.models import RequestContext

    store = SQLGraphStore("host=localhost dbname=ogmemory")
    ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                         session_id="s1", trace_id="t1")

    # Create nodes
    raw_node = create_raw_node(RawDataType.PROMPT, "Write a story", "acc1")
    store.create_node(raw_node.serialize(), ctx)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Any

from core.errors import AccessDeniedError, NodeNotFoundError
from core.graph_enums import (
    ExtractionStatus,
    GraphEdgeType,
    GraphNodeType,
    NodeStatus,
    RawDataType,
)
from core.graph_models import (
    ExtractedNode,
    GraphEdge,
    GraphNode,
    GraphStats,
    MemoryNode,
    node_from_payload,
    RawNode,
    SubgraphResult,
    SupportingChain,
    create_graph_edge,
    create_raw_node,
)
from core.interfaces import GraphStore
from core.models import RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin, SharedConnectionPool

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None
    Json = None

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length non-zero vectors.

    Returns 0.0 when either vector has zero norm or the lengths differ
    rather than raising — search stays robust against dirty data.
    Mirrors :meth:`MemoryGraphStore._cosine`; kept as a free function so
    the SQL store can rank without instantiating a graph store.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SQLGraphStore(PoolAdapterMixin, GraphStore):
    """PostgreSQL-backed implementation of GraphStore.

    Stores nodes in `graph_nodes` table and edges in `graph_edges` table.
    Implements incremental PageRank updates on edge creation.

    Usage::

        store = SQLGraphStore("host=localhost dbname=ogmemory")
        store.create_node(raw_node.serialize(), ctx)
    """

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool: SharedConnectionPool | None = None,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLGraphStore. "
                "Install with: pip install psycopg2-binary"
            )
        self._init_pool(pool, connection_string, pool_size)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection pool
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

    def _ensure_schema(self) -> None:
        """Ensure graph tables exist."""
        from pathlib import Path

        conn = self._get_connection()
        try:
            schema_path = Path(__file__).parent.parent.parent / "fs" / "sql_adapter" / "graph_schema.sql"
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    sql = f.read()
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning(f"Schema initialization warning: {exc}")
        finally:
            self._return_connection(conn)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_access(self, node_account_id: str, ctx: RequestContext) -> None:
        """Check if the request context has access to the node."""
        if node_account_id != ctx.account_id:
            raise AccessDeniedError(
                f"graph_node:{node_account_id}",
                ctx.account_id,
                f"Node belongs to account '{node_account_id}'",
            )

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Create a new graph node.

        Args:
            payload: JSON-ready node payload
            ctx: RequestContext for access control

        Returns:
            The created node with generated ID
        """
        node = node_from_payload(payload)
        payload = node.serialize()
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                self._insert_node(cur, payload)
            conn.commit()
            return node
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _insert_node(self, cur, payload: dict[str, Any]) -> None:
        """Insert a node into the database."""

        cur.execute(
            """
            INSERT INTO graph_nodes
                (node_id, node_type, status, raw_type, content, summary,
                 properties, metadata, extracted_content,
                 source_raw_id, extractor_type, extraction_schema,
                 extraction_status, confidence,
                 supporting_node_ids, topic, tags, parent_memory_id,
                 pagerank, account_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s::text[], %s, %s, %s, %s, %s)
            """,
            (
                payload["node_id"],
                payload["node_type"],
                payload["status"],
                payload.get("raw_type"),
                payload.get("content"),
                payload.get("summary"),
                Json(payload.get("properties", {})),
                Json(payload.get("metadata", {})),
                Json(payload.get("extracted_content", {})),
                payload.get("source_raw_id"),
                payload.get("extractor_type"),
                payload.get("extraction_schema"),
                payload.get("extraction_status"),
                payload.get("confidence") if payload.get("node_type") == GraphNodeType.EXTRACTED.value else None,
                payload.get("supporting_node_ids", []),
                payload.get("topic"),
                payload.get("tags", []),
                payload.get("parent_memory_id"),
                payload["pagerank"],
                payload["account_id"],
                payload["created_at"],
                payload["updated_at"],
            ),
        )

    def get_node(self, node_id: str, ctx: RequestContext) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            GraphNode if found and accessible, None otherwise
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE node_id = %s
                    """,
                    (node_id,),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        return self._row_to_node(row)

    def update_node(self, payload: dict[str, Any], ctx: RequestContext) -> GraphNode:
        """Update an existing node.

        Args:
            payload: JSON-ready node payload
            ctx: RequestContext for access control

        Returns:
            The updated node
        """
        node = node_from_payload(payload)
        payload = node.serialize()
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE graph_nodes SET
                        status = %s,
                        content = %s,
                        summary = %s,
                        properties = %s,
                        metadata = %s,
                        extracted_content = %s,
                        extraction_status = %s,
                        confidence = %s,
                        supporting_node_ids = %s::uuid[],
                        topic = %s,
                        tags = %s::text[],
                        pagerank = %s,
                        updated_at = NOW()
                    WHERE node_id = %s
                    """,
                    (
                        payload["status"],
                        payload.get("content"),
                        payload.get("summary"),
                        Json(payload.get("properties", {})),
                        Json(payload.get("metadata", {})),
                        Json(payload.get("extracted_content", {})),
                        payload.get("extraction_status"),
                        payload.get("confidence") if payload.get("node_type") == GraphNodeType.EXTRACTED.value else None,
                        payload.get("supporting_node_ids", []),
                        payload.get("topic"),
                        payload.get("tags", []),
                        payload["pagerank"],
                        payload["node_id"],
                    ),
                )
                if cur.rowcount == 0:
                    raise NodeNotFoundError(node.node_id)
            conn.commit()
            node.updated_at = self._get_current_timestamp()
            return node
        except NodeNotFoundError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all its edges.

        Args:
            node_id: UUID of the node to delete
            ctx: RequestContext for access control
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_nodes WHERE node_id = %s",
                    (node_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def list_nodes(
        self,
        node_type: GraphNodeType | None,
        ctx: RequestContext,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GraphNode]:
        """List nodes of a specific type.

        Args:
            node_type: Filter by node type (None for all types)
            ctx: RequestContext for access control
            limit: Maximum number of nodes to return
            offset: Number of nodes to skip

        Returns:
            List of matching nodes
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                if node_type:
                    cur.execute(
                        """
                        SELECT node_id, node_type, status, raw_type, content, summary,
                               properties, metadata, extracted_content,
                               source_raw_id, extractor_type, extraction_schema,
                               extraction_status, confidence,
                               supporting_node_ids, topic, tags, parent_memory_id,
                               pagerank, account_id, created_at, updated_at
                        FROM graph_nodes
                        WHERE node_type = %s
                        ORDER BY updated_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (node_type.value, limit, offset),
                    )
                else:
                    cur.execute(
                        """
                        SELECT node_id, node_type, status, raw_type, content, summary,
                               properties, metadata, extracted_content,
                               source_raw_id, extractor_type, extraction_schema,
                               extraction_status, confidence,
                               supporting_node_ids, topic, tags, parent_memory_id,
                               pagerank, account_id, created_at, updated_at
                        FROM graph_nodes
                        ORDER BY updated_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (limit, offset),
                    )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def create_edge(self, edge: GraphEdge, ctx: RequestContext) -> GraphEdge:
        """Create a new edge between nodes.

        Automatically triggers incremental PageRank update.

        Args:
            edge: GraphEdge to create
            ctx: RequestContext for access control

        Returns:
            The created edge
        """
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Insert edge
                cur.execute(
                    """
                    INSERT INTO graph_edges
                        (edge_id, source_id, target_id, edge_type, weight, properties, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        properties = EXCLUDED.properties
                    """,
                    (
                        edge.edge_id,
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type.value,
                        edge.weight,
                        Json(edge.properties),
                        edge.account_id,
                    ),
                )

                # Incremental PageRank update
                self._update_pagerank_incremental(cur, edge.target_id, edge.weight)

            conn.commit()
            return edge
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def _update_pagerank_incremental(
        self, cur, target_node_id: str, incoming_weight: float
    ) -> None:
        """Incrementally update PageRank for a node based on incoming edge."""
        # Get all incoming edges for this node
        cur.execute(
            """
            SELECT source_id, weight
            FROM graph_edges
            WHERE target_id = %s
            """,
            (target_node_id,),
        )
        incoming_edges = cur.fetchall()

        if not incoming_edges:
            return

        # Calculate outgoing count for each source
        total_incoming_weight = sum(weight for _, weight in incoming_edges)

        # Update each source node's PageRank based on its outgoing edges
        for source_id, weight in incoming_edges:
            cur.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE source_id = %s",
                (source_id,),
            )
            outgoing_count = cur.fetchone()[0] or 1

            # PR = sum(incoming_weight / outgoing_count)
            contribution = weight / outgoing_count

            cur.execute(
                """
                UPDATE graph_nodes
                SET pagerank = pagerank + %s
                WHERE node_id = %s
                """,
                (contribution, source_id),
            )

        # Update target node's PageRank
        cur.execute(
            """
            UPDATE graph_nodes
            SET pagerank = %s
            WHERE node_id = %s
            """,
            (total_incoming_weight, target_node_id),
        )

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge by ID."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_edges WHERE edge_id = %s",
                    (edge_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_edge(self, edge_id: str, ctx: RequestContext) -> GraphEdge | None:
        """Get an edge by ID."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, source_id, target_id, edge_type, weight, properties, account_id, created_at
                    FROM graph_edges
                    WHERE edge_id = %s
                    """,
                    (edge_id,),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        if row is None:
            return None

        return GraphEdge(
            edge_id=row[0],
            source_id=row[1],
            target_id=row[2],
            edge_type=GraphEdgeType(row[3]),
            weight=row[4],
            properties=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
            account_id=row[6],
            created_at=str(row[7]) if row[7] else "",
        )

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                results = []

                if direction in ("outgoing", "both"):
                    cur.execute(
                        """
                        SELECT n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at,
                               e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.target_id
                        WHERE e.source_id = %s
                        AND (%s IS NULL OR e.edge_type = ANY(%s))
                        """,
                        (node_id, edge_types, [t.value for t in edge_types] if edge_types else None),
                    )
                    for row in cur.fetchall():
                        neighbor = self._row_to_node(row[:22])
                        edge = self._row_to_edge(row[22:])
                        results.append((neighbor, edge, "outgoing"))

                if direction in ("incoming", "both"):
                    cur.execute(
                        """
                        SELECT n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at,
                               e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.source_id
                        WHERE e.target_id = %s
                        AND (%s IS NULL OR e.edge_type = ANY(%s))
                        """,
                        (node_id, edge_types, [t.value for t in edge_types] if edge_types else None),
                    )
                    for row in cur.fetchall():
                        neighbor = self._row_to_node(row[:22])
                        edge = self._row_to_edge(row[22:])
                        results.append((neighbor, edge, "incoming"))

        finally:
            conn.rollback()
            self._return_connection(conn)

        return results

    def traverse(
        self,
        start_node_id: str,
        depth: int,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        """BFS traversal from a starting node."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # BFS traversal
                visited = {start_node_id}
                queue = [(start_node_id, 0)]
                nodes_map = {}
                edges_list = []

                while queue:
                    current_id, current_depth = queue.pop(0)

                    if current_depth >= depth:
                        continue

                    # Get outgoing edges
                    cur.execute(
                        """
                        SELECT e.edge_id, e.source_id, e.target_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at,
                               n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.target_id
                        WHERE e.source_id = %s
                        AND (%s IS NULL OR e.edge_type = ANY(%s))
                        """,
                        (current_id, edge_types, [t.value for t in edge_types] if edge_types else None),
                    )

                    for row in cur.fetchall():
                        edge = self._row_to_edge(row[:8])
                        neighbor_id = row[9]

                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, current_depth + 1))
                            nodes_map[neighbor_id] = self._row_to_node(row[8:])

                        edges_list.append(edge)

                # Get the start node
                cur.execute(
                    """
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes WHERE node_id = %s
                    """,
                    (start_node_id,),
                )
                row = cur.fetchone()
                if row:
                    nodes_map[start_node_id] = self._row_to_node(row)

        finally:
            conn.rollback()
            self._return_connection(conn)

        return SubgraphResult(
            nodes=list(nodes_map.values()),
            edges=edges_list,
        )

    def get_supporting_chain(
        self, memory_id: str, ctx: RequestContext
    ) -> SupportingChain:
        """Get the evidence chain for a memory node."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Get memory node
                cur.execute(
                    """
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE node_id = %s AND node_type = 'memory'
                    """,
                    (memory_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise NodeNotFoundError(memory_id)
                memory = self._row_to_node(row)

                # Get supporting nodes via edges
                raw_nodes = []
                extracted_nodes = []
                edges_list = []

                # BFS to find all supporting nodes
                visited = {memory_id}
                queue = [(memory_id, 0)]

                while queue:
                    current_id, current_depth = queue.pop(0)
                    if current_depth > 2:  # Limit depth
                        continue

                    cur.execute(
                        """
                        SELECT n.node_id, n.node_type, n.status, n.raw_type, n.content, n.summary,
                               n.properties, n.metadata, n.extracted_content,
                               n.source_raw_id, n.extractor_type, n.extraction_schema,
                               n.extraction_status, n.confidence,
                               n.supporting_node_ids, n.topic, n.tags, n.parent_memory_id,
                               n.pagerank, n.account_id, n.created_at, n.updated_at,
                               e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        JOIN graph_nodes n ON n.node_id = e.target_id
                        WHERE e.source_id = %s AND e.edge_type = 'supports'
                        """,
                        (current_id,),
                    )

                    for row in cur.fetchall():
                        neighbor = self._row_to_node(row[:22])
                        edge = self._row_to_edge(row[22:])

                        if neighbor.node_id not in visited:
                            visited.add(neighbor.node_id)
                            queue.append((neighbor.node_id, current_depth + 1))

                            if isinstance(neighbor, RawNode):
                                raw_nodes.append(neighbor)
                            elif isinstance(neighbor, ExtractedNode):
                                extracted_nodes.append(neighbor)

                        edges_list.append(edge)

                # Also get edges from extracted to raw
                for extracted in extracted_nodes:
                    cur.execute(
                        """
                        SELECT e.edge_id, e.edge_type, e.weight, e.properties, e.account_id, e.created_at
                        FROM graph_edges e
                        WHERE e.source_id = %s AND e.edge_type = 'extracted_from'
                        """,
                        (extracted.node_id,),
                    )
                    for row in cur.fetchall():
                        edges_list.append(GraphEdge(
                            edge_id=row[0],
                            source_id=extracted.node_id,
                            target_id=extracted.source_raw_id,
                            edge_type=GraphEdgeType(row[1]),
                            weight=row[2],
                            properties=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                            account_id=row[4],
                            created_at=str(row[5]) if row[5] else "",
                        ))

        finally:
            conn.rollback()
            self._return_connection(conn)

        return SupportingChain(
            memory=memory,
            raw_nodes=raw_nodes,
            extracted_nodes=extracted_nodes,
            edges=edges_list,
        )

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores for nodes."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                if node_ids:
                    # Compute for specific nodes
                    cur.execute(
                        """
                        WITH node_scores AS (
                            SELECT
                                n.node_id,
                                COALESCE(SUM(e.weight / NULLIF(outgoing.cnt, 0)), 0) as pr_score
                            FROM graph_nodes n
                            LEFT JOIN graph_edges e ON e.target_id = n.node_id
                            LEFT JOIN (
                                SELECT source_id, COUNT(*) as cnt
                                FROM graph_edges
                                GROUP BY source_id
                            ) outgoing ON outgoing.source_id = e.source_id
                            WHERE n.node_id = ANY(%s)
                            GROUP BY n.node_id
                        )
                        UPDATE graph_nodes
                        SET pagerank = (
                            SELECT (1 - %s) + %s * COALESCE(pr_score, 0)
                            FROM node_scores ns WHERE ns.node_id = graph_nodes.node_id
                        )
                        WHERE node_id = ANY(%s)
                        RETURNING node_id, pagerank
                        """,
                        (node_ids, damping, damping, node_ids),
                    )
                else:
                    # Compute for all nodes
                    cur.execute(
                        """
                        WITH node_scores AS (
                            SELECT
                                n.node_id,
                                COALESCE(SUM(e.weight / NULLIF(outgoing.cnt, 0)), 0) as pr_score
                            FROM graph_nodes n
                            LEFT JOIN graph_edges e ON e.target_id = n.node_id
                            LEFT JOIN (
                                SELECT source_id, COUNT(*) as cnt
                                FROM graph_edges
                                GROUP BY source_id
                            ) outgoing ON outgoing.source_id = e.source_id
                            GROUP BY n.node_id
                        )
                        UPDATE graph_nodes
                        SET pagerank = (
                            SELECT (1 - %s) + %s * COALESCE(pr_score, 0)
                            FROM node_scores ns WHERE ns.node_id = graph_nodes.node_id
                        )
                        RETURNING node_id, pagerank
                        """,
                        (damping, damping),
                    )

                rows = cur.fetchall()
                conn.commit()
                return {row[0]: row[1] for row in rows}
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a specific node."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pagerank FROM graph_nodes WHERE node_id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return row[0] if row else 0.0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memory nodes by text and filters."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Build query dynamically
                conditions = ["node_type = 'memory'"]
                params = []

                if query:
                    conditions.append("(summary ILIKE %s OR content ILIKE %s)")
                    params.extend([f"%{query}%", f"%{query}%"])

                if min_confidence > 0:
                    conditions.append("pagerank >= %s")
                    params.append(min_confidence)

                if topic:
                    conditions.append("topic = %s")
                    params.append(topic)

                where_clause = " AND ".join(conditions)

                cur.execute(
                    f"""
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE {where_clause}
                    ORDER BY pagerank DESC
                    LIMIT %s
                    """,
                    params + [top_k],
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by type and optional text query."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                conditions = ["node_type = 'raw'"]
                params = []

                if raw_type:
                    conditions.append("raw_type = %s")
                    params.append(raw_type.value)

                if query:
                    conditions.append("content ILIKE %s")
                    params.append(f"%{query}%")

                where_clause = " AND ".join(conditions)

                cur.execute(
                    f"""
                    SELECT node_id, node_type, status, raw_type, content, summary,
                           properties, metadata, extracted_content,
                           source_raw_id, extractor_type, extraction_schema,
                           extraction_status, confidence,
                           supporting_node_ids, topic, tags, parent_memory_id,
                           pagerank, account_id, created_at, updated_at
                    FROM graph_nodes
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    params + [top_k],
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        return [self._row_to_node(row) for row in rows]

    def search_by_embedding(
        self,
        query_vector: list[float],
        ctx: RequestContext,
        *,
        node_types: tuple[str, ...] = ("extracted",),
        extractor_types: tuple[str, ...] | None = None,
        video_id: str | None = None,
        before_clip: int | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Cosine-similarity search over node embeddings.

        Implementation: filter candidates with JSONB predicates against
        the ``extracted_content`` column, then rank in Python by cosine
        similarity. For M3-Bench scale (~3K episodic nodes) this fits
        comfortably in memory; a pgvector upgrade is v2.

        Behaviour matches :meth:`MemoryGraphStore.search_by_embedding`:
        dimension-mismatched embeddings, missing keys, and malformed
        clip metadata are silently skipped so a single bad row cannot
        poison the result set.

        See :meth:`GraphStore.search_by_embedding` (Protocol) for parameter
        semantics.
        """
        if top_k <= 0 or not query_vector:
            return []

        type_filters: list[str] = []
        for nt in node_types:
            try:
                type_filters.append(GraphNodeType(nt).value)
            except ValueError:
                continue
        if not type_filters:
            return []

        conditions: list[str] = [
            "account_id = %s",
            "node_type = ANY(%s)",
            "(extracted_content ? 'embedding')",
        ]
        params: list[Any] = [ctx.account_id, type_filters]

        if extractor_types is not None:
            conditions.append("extractor_type = ANY(%s)")
            params.append(list(extractor_types))
        if video_id is not None:
            conditions.append("extracted_content->>'_video_id' = %s")
            params.append(video_id)
        if before_clip is not None:
            conditions.append(
                "((extracted_content->>'_clip_id')::int <= %s "
                "AND (extracted_content->>'_clip_id') ~ '^-?\\d+$')"
            )
            params.append(before_clip)

        where_clause = " AND ".join(conditions)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT node_id, extracted_content
                    FROM graph_nodes
                    WHERE {where_clause}
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        finally:
            conn.rollback()
            self._return_connection(conn)

        dim = len(query_vector)
        ranked: list[tuple[str, float]] = []
        for row in rows:
            node_id, ext_content = row
            if isinstance(ext_content, str):
                # Driver compatibility: some psycopg2 builds return the JSONB
                # column as a raw string when there are no JSON codecs wired.
                try:
                    ext_content = json.loads(ext_content)
                except (TypeError, ValueError):
                    continue
            if not isinstance(ext_content, dict):
                continue
            embedding = ext_content.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != dim:
                continue
            score = _cosine(query_vector, embedding)
            ranked.append((str(node_id), score))

        # Stable, deterministic tie-break: on identical scores the
        # control loop must see the same node_id order from both the
        # SQL and Memory backends. Sort by (-score, node_id) so the
        # node_id is a strict ascending tiebreaker (matches memory).
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                # Count nodes by type
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE node_type = 'raw') as raw_count,
                        COUNT(*) FILTER (WHERE node_type = 'extracted') as extracted_count,
                        COUNT(*) FILTER (WHERE node_type = 'memory') as memory_count,
                        COALESCE(AVG(pagerank), 0) as avg_pagerank
                    FROM graph_nodes
                    """
                )
                total, raw_count, extracted_count, memory_count, avg_pr = cur.fetchone()

                # Count edges
                cur.execute("SELECT COUNT(*) FROM graph_edges")
                total_edges = cur.fetchone()[0]

                # Top memories by pagerank
                cur.execute(
                    """
                    SELECT node_id, summary, pagerank
                    FROM graph_nodes
                    WHERE node_type = 'memory'
                    ORDER BY pagerank DESC
                    LIMIT 5
                    """
                )
                top_memories = [
                    {"node_id": row[0], "summary": row[1], "pagerank": row[2]}
                    for row in cur.fetchall()
                ]

        finally:
            conn.rollback()
            self._return_connection(conn)

        return GraphStats(
            total_nodes=total,
            raw_nodes=raw_count,
            extracted_nodes=extracted_count,
            memory_nodes=memory_count,
            total_edges=total_edges,
            avg_pagerank=avg_pr,
            top_memories=top_memories,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode."""
        (
            node_id, node_type, status, raw_type, content, summary,
            properties, metadata, extracted_content,
            source_raw_id, extractor_type, extraction_schema,
            extraction_status, confidence,
            supporting_node_ids, topic, tags, parent_memory_id,
            pagerank, account_id, created_at, updated_at,
        ) = row

        properties = properties if isinstance(properties, dict) else json.loads(properties)
        metadata = metadata if isinstance(metadata, dict) else json.loads(metadata)

        node_type_enum = GraphNodeType(node_type)
        status_enum = NodeStatus(status) if status else NodeStatus.ACTIVE

        if node_type_enum == GraphNodeType.RAW:
            return RawNode(
                node_id=node_id,
                node_type=GraphNodeType.RAW,
                status=status_enum,
                raw_type=RawDataType(raw_type) if raw_type else RawDataType.PROMPT,
                content=content or "",
                metadata=metadata,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )
        elif node_type_enum == GraphNodeType.EXTRACTED:
            return ExtractedNode(
                node_id=node_id,
                node_type=GraphNodeType.EXTRACTED,
                status=status_enum,
                source_raw_id=source_raw_id or "",
                extractor_type=extractor_type or "",
                extraction_schema=extraction_schema or "",
                extracted_content=extracted_content if isinstance(extracted_content, dict) else json.loads(extracted_content),
                extraction_status=ExtractionStatus(extraction_status) if extraction_status else ExtractionStatus.PENDING,
                confidence=confidence or 0.0,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )
        else:  # MEMORY
            return MemoryNode(
                node_id=node_id,
                node_type=GraphNodeType.MEMORY,
                status=status_enum,
                summary=summary or "",
                content=content or "",
                supporting_node_ids=supporting_node_ids or [],
                topic=topic or "",
                tags=tags or [],
                parent_memory_id=parent_memory_id,
                pagerank=pagerank or 0.0,
                created_at=str(created_at) if created_at else "",
                updated_at=str(updated_at) if updated_at else "",
                account_id=account_id,
            )

    def _row_to_edge(self, row: tuple) -> GraphEdge:
        """Convert a database row to a GraphEdge."""
        edge_id, edge_type, weight, properties, account_id, created_at = row
        return GraphEdge(
            edge_id=edge_id,
            source_id="",  # Will be set by caller if needed
            target_id="",  # Will be set by caller if needed
            edge_type=GraphEdgeType(edge_type),
            weight=weight,
            properties=properties if isinstance(properties, dict) else json.loads(properties),
            account_id=account_id,
            created_at=str(created_at) if created_at else "",
        )

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO8601 string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()