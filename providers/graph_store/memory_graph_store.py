"""In-memory implementation of GraphStore for testing.

This implementation stores all nodes and edges in memory dictionaries.
Useful for unit testing without a PostgreSQL dependency.

Usage::

    from providers.graph_store import MemoryGraphStore
    from core.graph_models import create_raw_node

    store = MemoryGraphStore()
    node = create_raw_node(RawDataType.PROMPT, "Hello world", "acc1")
    store.create_node(node, ctx)
"""

from __future__ import annotations

import json
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
    RawNode,
    SubgraphResult,
    SupportingChain,
)
from core.models import RequestContext


class MemoryGraphStore:
    """In-memory implementation of GraphStore.

    Thread-unsafe, for testing only.
    """

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        # Index for faster lookups
        self._nodes_by_type: dict[GraphNodeType, list[str]] = {
            GraphNodeType.RAW: [],
            GraphNodeType.EXTRACTED: [],
            GraphNodeType.MEMORY: [],
        }
        self._edges_by_source: dict[str, list[str]] = {}
        self._edges_by_target: dict[str, list[str]] = {}

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

    def create_node(self, node: GraphNode, ctx: RequestContext) -> GraphNode:
        """Create a new graph node."""
        self._nodes[node.node_id] = node
        self._nodes_by_type[node.node_type].append(node.node_id)
        return node

    def get_node(self, node_id: str, ctx: RequestContext) -> GraphNode | None:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node and node.account_id != ctx.account_id:
            return None
        return node

    def update_node(self, node: GraphNode, ctx: RequestContext) -> GraphNode:
        """Update an existing node."""
        if node.node_id not in self._nodes:
            raise NodeNotFoundError(node.node_id)
        self._check_access(self._nodes[node.node_id].account_id, ctx)
        self._nodes[node.node_id] = node
        return node

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all its edges."""
        if node_id not in self._nodes:
            return

        node = self._nodes[node_id]
        self._check_access(node.account_id, ctx)

        # Remove from type index
        self._nodes_by_type[node.node_type].remove(node_id)

        # Remove connected edges
        for edge_id in list(self._edges.keys()):
            edge = self._edges[edge_id]
            if edge.source_id == node_id or edge.target_id == node_id:
                self._remove_edge(edge_id)

        del self._nodes[node_id]

    def list_nodes(
        self,
        node_type: GraphNodeType | None,
        ctx: RequestContext,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GraphNode]:
        """List nodes of a specific type."""
        if node_type:
            node_ids = self._nodes_by_type.get(node_type, [])
        else:
            node_ids = list(self._nodes.keys())

        result = []
        for node_id in node_ids:
            node = self._nodes.get(node_id)
            if node and node.account_id == ctx.account_id:
                result.append(node)

        result.sort(key=lambda n: n.updated_at, reverse=True)
        return result[offset:offset + limit]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def create_edge(self, edge: GraphEdge, ctx: RequestContext) -> GraphEdge:
        """Create a new edge between nodes."""
        self._edges[edge.edge_id] = edge

        # Update indexes
        if edge.source_id not in self._edges_by_source:
            self._edges_by_source[edge.source_id] = []
        self._edges_by_source[edge.source_id].append(edge.edge_id)

        if edge.target_id not in self._edges_by_target:
            self._edges_by_target[edge.target_id] = []
        self._edges_by_target[edge.target_id].append(edge.edge_id)

        # Incremental PageRank update
        self._update_pagerank_incremental(edge.target_id, edge.weight)

        return edge

    def _update_pagerank_incremental(
        self, target_node_id: str, incoming_weight: float
    ) -> None:
        """Incrementally update PageRank for a node based on incoming edge."""
        target = self._nodes.get(target_node_id)
        if not target:
            return

        # Calculate total incoming weight
        total_incoming = 0.0
        for edge_id in self._edges_by_target.get(target_node_id, []):
            edge = self._edges.get(edge_id)
            if edge:
                total_incoming += edge.weight

        target.pagerank = total_incoming

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge by ID."""
        self._remove_edge(edge_id)

    def _remove_edge(self, edge_id: str) -> None:
        """Remove an edge from storage."""
        edge = self._edges.pop(edge_id, None)
        if edge:
            if edge.edge_id in self._edges_by_source.get(edge.source_id, []):
                self._edges_by_source[edge.source_id].remove(edge.edge_id)
            if edge.edge_id in self._edges_by_target.get(edge.target_id, []):
                self._edges_by_target[edge.target_id].remove(edge.edge_id)

    def get_edge(self, edge_id: str, ctx: RequestContext) -> GraphEdge | None:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

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
        results = []

        if direction in ("outgoing", "both"):
            for edge_id in self._edges_by_source.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (edge_types is None or edge.edge_type in edge_types):
                    neighbor = self._nodes.get(edge.target_id)
                    if neighbor:
                        results.append((neighbor, edge, "outgoing"))

        if direction in ("incoming", "both"):
            for edge_id in self._edges_by_target.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (edge_types is None or edge.edge_type in edge_types):
                    neighbor = self._nodes.get(edge.source_id)
                    if neighbor:
                        results.append((neighbor, edge, "incoming"))

        return results

    def traverse(
        self,
        start_node_id: str,
        depth: int,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        """BFS traversal from a starting node."""
        visited = {start_node_id}
        queue = [(start_node_id, 0)]
        nodes_map = {start_node_id: self._nodes[start_node_id]}
        edges_list = []

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            for edge_id in self._edges_by_source.get(current_id, []):
                edge = self._edges.get(edge_id)
                if edge and (edge_types is None or edge.edge_type in edge_types):
                    neighbor = self._nodes.get(edge.target_id)
                    if neighbor and neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        queue.append((neighbor.node_id, current_depth + 1))
                        nodes_map[neighbor.node_id] = neighbor
                    edges_list.append(edge)

        return SubgraphResult(
            nodes=list(nodes_map.values()),
            edges=edges_list,
        )

    def get_supporting_chain(
        self, memory_id: str, ctx: RequestContext
    ) -> SupportingChain:
        """Get the evidence chain for a memory node."""
        memory = self._nodes.get(memory_id)
        if not memory or not isinstance(memory, MemoryNode):
            raise NodeNotFoundError(memory_id)

        raw_nodes = []
        extracted_nodes = []
        edges_list = []

        visited = {memory_id}
        queue = [(memory_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth > 2:
                continue

            # Step 1: Find supports edges (incoming to current node)
            # supports edge: source (supporting) -> target (memory)
            for edge_id in self._edges_by_target.get(current_id, []):
                edge = self._edges.get(edge_id)
                if edge and edge.edge_type == GraphEdgeType.SUPPORTS:
                    neighbor = self._nodes.get(edge.source_id)
                    if neighbor and neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        queue.append((neighbor.node_id, current_depth + 1))

                        if isinstance(neighbor, RawNode):
                            raw_nodes.append(neighbor)
                        elif isinstance(neighbor, ExtractedNode):
                            extracted_nodes.append(neighbor)

                    edges_list.append(edge)

            # Step 2: For extracted nodes, also find their source raw nodes
            # extracted_from edge: source (extracted) -> target (raw)
            # So we look at outgoing edges from extracted nodes
            current_node = self._nodes.get(current_id)
            if isinstance(current_node, ExtractedNode):
                for edge_id in self._edges_by_source.get(current_id, []):
                    edge = self._edges.get(edge_id)
                    if edge and edge.edge_type == GraphEdgeType.EXTRACTED_FROM:
                        neighbor = self._nodes.get(edge.target_id)
                        if neighbor and neighbor.node_id not in visited:
                            visited.add(neighbor.node_id)
                            queue.append((neighbor.node_id, current_depth + 1))

                            if isinstance(neighbor, RawNode):
                                raw_nodes.append(neighbor)

                        edges_list.append(edge)

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
        nodes_to_compute = node_ids or list(self._nodes.keys())

        # Build adjacency
        incoming: dict[str, list] = {nid: [] for nid in nodes_to_compute}
        outgoing_count: dict[str, int] = {nid: 0 for nid in nodes_to_compute}

        for edge in self._edges.values():
            if edge.source_id in nodes_to_compute and edge.target_id in nodes_to_compute:
                incoming[edge.target_id].append((edge.source_id, edge.weight))
                outgoing_count[edge.source_id] += 1

        # Iterative PageRank
        N = len(nodes_to_compute)
        scores = {nid: 1.0 / N for nid in nodes_to_compute}

        for _ in range(10):  # Max iterations
            new_scores = {}
            for nid in nodes_to_compute:
                sum_contrib = sum(
                    scores[src] * w / max(outgoing_count[src], 1)
                    for src, w in incoming[nid]
                )
                new_scores[nid] = (1 - damping) / N + damping * sum_contrib

            scores = new_scores

        # Update nodes
        for nid, score in scores.items():
            if nid in self._nodes:
                self._nodes[nid].pagerank = score

        return scores

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a specific node."""
        node = self._nodes.get(node_id)
        return node.pagerank if node else 0.0

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
        results = []

        for node in self._nodes.values():
            if not isinstance(node, MemoryNode):
                continue
            if node.account_id != ctx.account_id:
                continue
            if node.pagerank < min_confidence:
                continue
            if topic and node.topic != topic:
                continue
            if query:
                if query.lower() not in node.summary.lower() and query.lower() not in node.content.lower():
                    continue

            results.append(node)

        results.sort(key=lambda n: n.pagerank, reverse=True)
        return results[:top_k]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by type and optional text query."""
        results = []

        for node in self._nodes.values():
            if not isinstance(node, RawNode):
                continue
            if node.account_id != ctx.account_id:
                continue
            if raw_type and node.raw_type != raw_type:
                continue
            if query and query.lower() not in node.content.lower():
                continue

            results.append(node)

        results.sort(key=lambda n: n.updated_at, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph."""
        total = raw_count = extracted_count = memory_count = 0
        total_edges = len(self._edges)
        pageranks = []

        for node in self._nodes.values():
            if node.account_id != ctx.account_id:
                continue
            total += 1
            pageranks.append(node.pagerank)

            if isinstance(node, RawNode):
                raw_count += 1
            elif isinstance(node, ExtractedNode):
                extracted_count += 1
            elif isinstance(node, MemoryNode):
                memory_count += 1

        avg_pr = sum(pageranks) / len(pageranks) if pageranks else 0.0

        # Top memories
        memories = [
            (n.node_id, n.summary, n.pagerank)
            for n in self._nodes.values()
            if isinstance(n, MemoryNode) and n.account_id == ctx.account_id
        ]
        memories.sort(key=lambda x: x[2], reverse=True)
        top_memories = [
            {"node_id": nid, "summary": summary, "pagerank": pr}
            for nid, summary, pr in memories[:5]
        ]

        return GraphStats(
            total_nodes=total,
            raw_nodes=raw_count,
            extracted_nodes=extracted_count,
            memory_nodes=memory_count,
            total_edges=total_edges,
            avg_pagerank=avg_pr,
            top_memories=top_memories,
        )