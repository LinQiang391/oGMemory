"""In-memory graph implementation of RelationStore protocol.

Provides efficient multi-hop relation traversal for hierarchical retrieval.
Phase 2 implementation: In-memory storage using Python built-in types.
Phase 3+: Can migrate to external graph database without changing interface.

See docs/graph_schema.md for detailed design.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from core.models import RequestContext, RelationEdge
from core.interfaces import RelationStore


@dataclass
class NodeMetadata:
    """Metadata for a node in the graph.

    Nodes are tracked even if they have no edges.
    Deleted nodes are marked exists=False but edges remain.
    """
    uri: str
    exists: bool = True
    account_id: str = ""


@dataclass
class RelatedNode:
    """A node reachable via graph traversal.

    Returned by get_related_nodes() with path information.
    """
    uri: str
    score: float  # Path score (min weight along path)
    distance: int  # 1 for 1-hop, 2 for 2-hop
    path: list[RelationEdge] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)


class GraphRelationStore(RelationStore):
    """In-memory graph relation store.

    Stores edges in dual-index structure for efficient bidirectional traversal:
    - edges_by_source: from_uri -> to_uri -> relation_type -> RelationEdge
    - edges_by_target: to_uri -> from_uri -> relation_type -> RelationEdge

    Thread-safe: uses locks for write operations, reads are lock-free.
    """

    def __init__(self):
        """Initialize empty graph store."""
        self._nodes: dict[str, NodeMetadata] = {}
        self._edges_by_source: dict[str, dict[str, dict[str, RelationEdge]]] = {}
        self._edges_by_target: dict[str, dict[str, dict[str, RelationEdge]]] = {}
        self._lock = threading.Lock()

    # ========================================================================
    # RelationStore Protocol Implementation
    # ========================================================================

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        """Get all outgoing edges from a node.

        Args:
            uri: ContextEngine URI of the source node
            ctx: RequestContext for access control

        Returns:
            List of RelationEdge objects. Empty list if node has no relations
            or doesn't exist.
        """
        if not self._is_accessible(uri, ctx):
            return []

        return self._get_outbound_edges(uri)

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        """Add or update relation edges.

        Idempotent: same edge (from_uri, to_uri, relation_type) updates
        the existing edge.

        Args:
            edges: List of RelationEdge objects to upsert
            ctx: RequestContext for access control
        """
        if not edges:
            return

        with self._lock:
            for edge in edges:
                # Validate access: must own source node
                if edge.from_uri and edge.from_uri not in self._nodes:
                    # Auto-register node with account from context
                    self._nodes[edge.from_uri] = NodeMetadata(
                        uri=edge.from_uri,
                        exists=True,
                        account_id=ctx.account_id,
                    )
                elif edge.from_uri and self._nodes[edge.from_uri].account_id != ctx.account_id:
                    # Skip edges for nodes we don't own
                    continue

                # Register target node if unknown
                if edge.to_uri and edge.to_uri not in self._nodes:
                    self._nodes[edge.to_uri] = NodeMetadata(
                        uri=edge.to_uri,
                        exists=True,
                        account_id=ctx.account_id,  # Assume same account
                    )

                self._upsert_single_edge(edge)

    # ========================================================================
    # Extended Graph API
    # ========================================================================

    def get_related_nodes(
        self,
        uri: str,
        ctx: RequestContext,
        direction: str = "OUTBOUND",
        depth: int = 1,
        relation_types: Optional[list[str]] = None,
        min_weight: float = 0.0,
    ) -> list[RelatedNode]:
        """Get related nodes via graph traversal.

        Args:
            uri: Source URI to start traversal
            ctx: RequestContext for access control
            direction: OUTBOUND | INBOUND | BOTH
            depth: 1 for 1-hop, 2 for 2-hop traversal
            relation_types: Optional filter by relation type
            min_weight: Minimum edge weight to include

        Returns:
            List of RelatedNode with uri, score, path, and metadata.
            Sorted by score descending.
        """
        if not self._is_accessible(uri, ctx):
            return []

        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")

        if direction not in ("OUTBOUND", "INBOUND", "BOTH"):
            raise ValueError("direction must be OUTBOUND, INBOUND, or BOTH")

        related: list[RelatedNode] = []

        if depth == 1:
            related = self._traverse_1hop(uri, direction, relation_types, min_weight)
        else:  # depth == 2
            related = self._traverse_2hop(uri, direction, relation_types, min_weight)

        # Sort by score descending
        return sorted(related, key=lambda n: n.score, reverse=True)

    def node_exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if a node exists in the graph.

        Args:
            uri: URI to check
            ctx: RequestContext for access control

        Returns:
            True if node exists and is accessible
        """
        return self._is_accessible(uri, ctx)

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Mark a node as deleted (edges remain for history).

        Args:
            uri: URI to delete
            ctx: RequestContext for access control
        """
        if not self._is_accessible(uri, ctx):
            return

        with self._lock:
            if uri in self._nodes:
                self._nodes[uri] = NodeMetadata(
                    uri=uri,
                    exists=False,
                    account_id=self._nodes[uri].account_id,
                )

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _is_accessible(self, uri: str, ctx: RequestContext) -> bool:
        """Check if URI is accessible in the current context.

        Args:
            uri: URI to check
            ctx: RequestContext with account_id

        Returns:
            True if node exists and belongs to the account
        """
        if uri not in self._nodes:
            return False
        node = self._nodes[uri]
        return node.exists and node.account_id == ctx.account_id

    def _get_outbound_edges(self, uri: str) -> list[RelationEdge]:
        """Get all outbound edges from a URI."""
        if uri not in self._edges_by_source:
            return []

        all_edges: list[RelationEdge] = []
        for target_dict in self._edges_by_source[uri].values():
            all_edges.extend(target_dict.values())
        return all_edges

    def _get_inbound_edges(self, uri: str) -> list[RelationEdge]:
        """Get all inbound edges to a URI."""
        if uri not in self._edges_by_target:
            return []

        all_edges: list[RelationEdge] = []
        for source_dict in self._edges_by_target[uri].values():
            all_edges.extend(source_dict.values())
        return all_edges

    def _upsert_single_edge(self, edge: RelationEdge) -> None:
        """Insert or update a single edge in both indexes.

        Thread-safe: caller must hold _lock.
        """
        # Update source index
        if edge.from_uri not in self._edges_by_source:
            self._edges_by_source[edge.from_uri] = {}
        if edge.to_uri not in self._edges_by_source[edge.from_uri]:
            self._edges_by_source[edge.from_uri][edge.to_uri] = {}
        self._edges_by_source[edge.from_uri][edge.to_uri][edge.relation_type] = edge

        # Update target index (reverse)
        if edge.to_uri not in self._edges_by_target:
            self._edges_by_target[edge.to_uri] = {}
        if edge.from_uri not in self._edges_by_target[edge.to_uri]:
            self._edges_by_target[edge.to_uri][edge.from_uri] = {}
        self._edges_by_target[edge.to_uri][edge.from_uri][edge.relation_type] = edge

    def _traverse_1hop(
        self,
        uri: str,
        direction: str,
        relation_types: Optional[list[str]],
        min_weight: float,
    ) -> list[RelatedNode]:
        """Perform 1-hop graph traversal."""
        edges: list[RelationEdge] = []

        if direction in ("OUTBOUND", "BOTH"):
            edges.extend(self._get_outbound_edges(uri))

        if direction in ("INBOUND", "BOTH"):
            inbound = self._get_inbound_edges(uri)
            # For inbound, we need to reverse from/to in the path
            for edge in inbound:
                # Create reversed edge for path consistency
                edges.append(RelationEdge(
                    from_uri=edge.to_uri,
                    to_uri=edge.from_uri,
                    relation_type=edge.relation_type,
                    weight=edge.weight,
                    reason=f"(inbound) {edge.reason}",
                ))

        # Filter by relation type and weight
        if relation_types:
            edges = [e for e in edges if e.relation_type in relation_types]
        if min_weight > 0:
            edges = [e for e in edges if e.weight >= min_weight]

        # Convert to RelatedNode
        return [
            RelatedNode(
                uri=edge.to_uri,
                score=edge.weight,
                distance=1,
                path=[edge],
                relation_types=[edge.relation_type],
            )
            for edge in edges
        ]

    def _traverse_2hop(
        self,
        uri: str,
        direction: str,
        relation_types: Optional[list[str]],
        min_weight: float,
    ) -> list[RelatedNode]:
        """Perform 2-hop graph traversal."""
        # Get 1-hop neighbors
        hop1 = self._traverse_1hop(uri, direction, relation_types, min_weight)

        # Collect 2-hop results with deduplication
        results: dict[str, RelatedNode] = {}

        for node1 in hop1:
            # Traverse from each 1-hop neighbor
            hop2 = self._traverse_1hop(node1.uri, direction, relation_types, min_weight)

            for node2 in hop2:
                # Skip the source node (cycles)
                if node2.uri == uri:
                    continue

                # Calculate path score (min weight along path)
                path_score = min(node1.score, node2.score)

                # Combine paths
                combined_path = node1.path + node2.path
                combined_types = list(set(node1.relation_types + node2.relation_types))

                # Keep best score for each target URI
                if node2.uri not in results or results[node2.uri].score < path_score:
                    results[node2.uri] = RelatedNode(
                        uri=node2.uri,
                        score=path_score,
                        distance=2,
                        path=combined_path,
                        relation_types=combined_types,
                    )

        return list(results.values())
