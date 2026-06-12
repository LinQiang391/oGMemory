"""Graph service for heterogeneous multimodal memory system.

Provides high-level operations for creating and managing graph nodes,
edges, and traversing the evidence chain for memories.

Usage::

    from service.graph_service import GraphService
    from core.models import RequestContext
    from core.graph_enums import RawDataType, GraphEdgeType
    from core.graph_models import create_raw_node, create_graph_edge

    service = GraphService(graph_store)
    ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                         session_id="s1", trace_id="t1")

    # Create a raw prompt node
    raw_node = create_raw_node(
        raw_type=RawDataType.PROMPT,
        content="Write a story about a dragon",
        account_id="acc1"
    )
    service.create_raw_node(raw_node, ctx)

    # Create extracted node from raw
    extracted = service.create_extracted_node(
        source_raw_id=raw_node.node_id,
        extractor_type="prompt_analyzer",
        extraction_schema="prompt_structure_v1",
        extracted_content={"intent": "creative_writing", "subject": "dragon"},
        ctx=ctx
    )

    # Create memory from extracted
    memory = service.create_memory_node(
        summary="User wants to write a dragon story",
        supporting_node_ids=[extracted.node_id],
        ctx=ctx
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.graph_enums import (
    ExtractionStatus,
    GraphEdgeType,
    GraphNodeType,
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
    create_extracted_node,
    create_graph_edge,
    create_memory_node,
    create_raw_node,
)
from core.interfaces import GraphStore
from core.models import RequestContext

if TYPE_CHECKING:
    from core.errors import ContextEngineError

logger = logging.getLogger(__name__)


class GraphService:
    """High-level service for heterogeneous graph memory operations.

    Provides convenience methods that combine multiple store operations
    and handle edge cases like creating supporting edges automatically.
    """

    def __init__(self, graph_store: GraphStore):
        """Initialize the graph service.

        Args:
            graph_store: Underlying graph storage implementation
        """
        self._store = graph_store

    # ====================================================================
    # Raw Node Operations
    # ====================================================================

    def create_raw_node(
        self,
        raw_type: RawDataType,
        content: str,
        ctx: RequestContext,
        metadata: dict | None = None,
        node_id: str | None = None,
    ) -> RawNode:
        """Create a raw multimodal data node.

        Args:
            raw_type: Type of raw data (prompt, image, video, etc.)
            content: The raw content
            ctx: RequestContext for access control
            metadata: Optional metadata dict
            node_id: Optional pre-generated node ID

        Returns:
            The created RawNode
        """
        node = create_raw_node(
            raw_type=raw_type,
            content=content,
            account_id=ctx.account_id,
            metadata=metadata,
            node_id=node_id,
        )
        return self._store.create_node(node, ctx)

    def get_raw_node(
        self, node_id: str, ctx: RequestContext
    ) -> RawNode | None:
        """Get a raw node by ID.

        Args:
            node_id: UUID of the raw node
            ctx: RequestContext for access control

        Returns:
            RawNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, RawNode):
            return node
        return None

    def list_raw_nodes(
        self,
        raw_type: RawDataType | None,
        ctx: RequestContext,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RawNode]:
        """List raw nodes optionally filtered by type.

        Args:
            raw_type: Filter by raw data type (None for all)
            ctx: RequestContext for access control
            limit: Max results
            offset: Skip count

        Returns:
            List of RawNodes
        """
        nodes = self._store.list_nodes(GraphNodeType.RAW, ctx, limit, offset)
        if raw_type:
            return [n for n in nodes if isinstance(n, RawNode) and n.raw_type == raw_type]
        return [n for n in nodes if isinstance(n, RawNode)]

    def search_raw_nodes(
        self,
        raw_type: RawDataType | None,
        query: str,
        ctx: RequestContext,
        top_k: int = 10,
    ) -> list[RawNode]:
        """Search raw nodes by content.

        Args:
            raw_type: Filter by type
            query: Search query
            ctx: RequestContext for access control
            top_k: Max results

        Returns:
            List of matching RawNodes
        """
        return self._store.search_raw_nodes(raw_type, ctx, query, top_k)

    # ====================================================================
    # Extracted Node Operations
    # ====================================================================

    def create_extracted_node(
        self,
        source_raw_id: str,
        extractor_type: str,
        extraction_schema: str,
        ctx: RequestContext,
        extracted_content: dict | None = None,
        node_id: str | None = None,
    ) -> ExtractedNode:
        """Create an extracted node from a raw node.

        Automatically creates the extracted_from edge.

        Args:
            source_raw_id: ID of the source raw node
            extractor_type: Type/name of the extractor
            extraction_schema: Schema used for extraction
            ctx: RequestContext for access control
            extracted_content: The extraction results
            node_id: Optional pre-generated node ID

        Returns:
            The created ExtractedNode
        """
        node = create_extracted_node(
            source_raw_id=source_raw_id,
            extractor_type=extractor_type,
            extraction_schema=extraction_schema,
            account_id=ctx.account_id,
            extracted_content=extracted_content,
            node_id=node_id,
        )
        self._store.create_node(node, ctx)

        # Create extracted_from edge
        edge = create_graph_edge(
            source_id=node.node_id,
            target_id=source_raw_id,
            edge_type=GraphEdgeType.EXTRACTED_FROM,
            account_id=ctx.account_id,
            weight=1.0,
        )
        self._store.create_edge(edge, ctx)

        return node

    def get_extracted_node(
        self, node_id: str, ctx: RequestContext
    ) -> ExtractedNode | None:
        """Get an extracted node by ID.

        Args:
            node_id: UUID of the extracted node
            ctx: RequestContext for access control

        Returns:
            ExtractedNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, ExtractedNode):
            return node
        return None

    def list_extracted_nodes(
        self,
        extractor_type: str | None,
        ctx: RequestContext,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ExtractedNode]:
        """List extracted nodes optionally filtered by extractor type.

        Args:
            extractor_type: Filter by extractor type
            ctx: RequestContext for access control
            limit: Max results
            offset: Skip count

        Returns:
            List of ExtractedNodes
        """
        nodes = self._store.list_nodes(GraphNodeType.EXTRACTED, ctx, limit, offset)
        result = [n for n in nodes if isinstance(n, ExtractedNode)]
        if extractor_type:
            return [n for n in result if n.extractor_type == extractor_type]
        return result

    def mark_extraction_completed(
        self,
        node_id: str,
        extracted_content: dict,
        confidence: float,
        ctx: RequestContext,
    ) -> ExtractedNode:
        """Mark an extraction as completed with results.

        Args:
            node_id: ID of the extracted node
            extracted_content: The extraction results
            confidence: Confidence score (0-1)
            ctx: RequestContext for access control

        Returns:
            The updated ExtractedNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, ExtractedNode):
            raise ValueError(f"Extracted node not found: {node_id}")

        node.mark_completed(extracted_content, confidence)
        return self._store.update_node(node, ctx)

    def mark_extraction_failed(
        self, node_id: str, error: str, ctx: RequestContext
    ) -> ExtractedNode:
        """Mark an extraction as failed.

        Args:
            node_id: ID of the extracted node
            error: Error message
            ctx: RequestContext for access control

        Returns:
            The updated ExtractedNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, ExtractedNode):
            raise ValueError(f"Extracted node not found: {node_id}")

        node.mark_failed(error)
        return self._store.update_node(node, ctx)

    # ====================================================================
    # Memory Node Operations
    # ====================================================================

    def create_memory_node(
        self,
        summary: str,
        ctx: RequestContext,
        content: str = "",
        supporting_node_ids: list[str] | None = None,
        topic: str = "",
        tags: list[str] | None = None,
        parent_memory_id: str | None = None,
        node_id: str | None = None,
    ) -> MemoryNode:
        """Create a memory node with supporting nodes.

        Automatically creates supports edges for each supporting node.

        Args:
            summary: Memory summary
            ctx: RequestContext for access control
            content: Optional full memory content
            supporting_node_ids: IDs of supporting nodes
            topic: Optional topic
            tags: Optional tags
            parent_memory_id: ID of parent memory (for evolution)
            node_id: Optional pre-generated node ID

        Returns:
            The created MemoryNode
        """
        supporting_node_ids = supporting_node_ids or []
        node = create_memory_node(
            summary=summary,
            account_id=ctx.account_id,
            content=content,
            supporting_node_ids=supporting_node_ids,
            topic=topic,
            tags=tags,
            node_id=node_id,
            parent_memory_id=parent_memory_id,
        )
        self._store.create_node(node, ctx)

        # Create supports edges
        for supporting_id in supporting_node_ids:
            edge = create_graph_edge(
                source_id=supporting_id,
                target_id=node.node_id,
                edge_type=GraphEdgeType.SUPPORTS,
                account_id=ctx.account_id,
                weight=1.0,
            )
            self._store.create_edge(edge, ctx)

        # Create evolves edge if parent exists
        if parent_memory_id:
            edge = create_graph_edge(
                source_id=parent_memory_id,
                target_id=node.node_id,
                edge_type=GraphEdgeType.EVOLVES,
                account_id=ctx.account_id,
                weight=1.0,
            )
            self._store.create_edge(edge, ctx)

        return node

    def get_memory_node(
        self, node_id: str, ctx: RequestContext
    ) -> MemoryNode | None:
        """Get a memory node by ID.

        Args:
            node_id: UUID of the memory node
            ctx: RequestContext for access control

        Returns:
            MemoryNode if found, None otherwise
        """
        node = self._store.get_node(node_id, ctx)
        if node and isinstance(node, MemoryNode):
            return node
        return None

    def search_memories(
        self,
        query: str,
        ctx: RequestContext,
        min_confidence: float = 0.0,
        topic: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memory nodes.

        Args:
            query: Search query
            ctx: RequestContext for access control
            min_confidence: Minimum confidence threshold
            topic: Filter by topic
            tags: Filter by tags (OR logic)
            top_k: Max results

        Returns:
            List of matching MemoryNodes sorted by confidence
        """
        return self._store.search_memories(
            query, ctx, min_confidence, topic, tags, top_k
        )

    def update_memory(
        self,
        node_id: str,
        ctx: RequestContext,
        summary: str | None = None,
        content: str | None = None,
        topic: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryNode:
        """Update a memory node.

        Args:
            node_id: ID of the memory node
            ctx: RequestContext for access control
            summary: New summary
            content: New content
            topic: New topic
            tags: New tags

        Returns:
            The updated MemoryNode
        """
        node = self._store.get_node(node_id, ctx)
        if not node or not isinstance(node, MemoryNode):
            raise ValueError(f"Memory node not found: {node_id}")

        if summary is not None:
            node.summary = summary
        if content is not None:
            node.content = content
        if topic is not None:
            node.topic = topic
        if tags is not None:
            node.tags = tags

        node.touch()
        return self._store.update_node(node, ctx)

    def add_supporting_node(
        self, memory_id: str, supporting_id: str, ctx: RequestContext
    ) -> MemoryNode:
        """Add a supporting node to a memory.

        Args:
            memory_id: ID of the memory node
            supporting_id: ID of the supporting node
            ctx: RequestContext for access control

        Returns:
            The updated MemoryNode
        """
        memory = self._store.get_node(memory_id, ctx)
        if not memory or not isinstance(memory, MemoryNode):
            raise ValueError(f"Memory node not found: {memory_id}")

        memory.add_support(supporting_id)

        # Create the supports edge
        edge = create_graph_edge(
            source_id=supporting_id,
            target_id=memory_id,
            edge_type=GraphEdgeType.SUPPORTS,
            account_id=ctx.account_id,
            weight=1.0,
        )
        self._store.create_edge(edge, ctx)

        return self._store.update_node(memory, ctx)

    # ====================================================================
    # Graph Traversal
    # ====================================================================

    def get_neighbors(
        self,
        node_id: str,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[GraphNode, GraphEdge, str]]:
        """Get neighboring nodes and edges.

        Args:
            node_id: UUID of the center node
            ctx: RequestContext for access control
            edge_types: Filter by edge types
            direction: "incoming", "outgoing", or "both"

        Returns:
            List of (neighbor_node, edge, direction) tuples
        """
        return self._store.get_neighbors(node_id, ctx, edge_types, direction)

    def traverse(
        self,
        start_node_id: str,
        depth: int,
        ctx: RequestContext,
        edge_types: list[GraphEdgeType] | None = None,
    ) -> SubgraphResult:
        """BFS traversal from a starting node.

        Args:
            start_node_id: UUID of the starting node
            depth: Maximum traversal depth
            ctx: RequestContext for access control
            edge_types: Filter by edge types

        Returns:
            SubgraphResult with nodes and edges within depth limit
        """
        return self._store.traverse(start_node_id, depth, ctx, edge_types)

    def get_supporting_chain(
        self, memory_id: str, ctx: RequestContext
    ) -> SupportingChain:
        """Get the evidence chain for a memory node.

        Returns the memory and all supporting raw/extracted nodes
        with their relationships.

        Args:
            memory_id: UUID of the memory node
            ctx: RequestContext for access control

        Returns:
            SupportingChain with memory, supporting nodes, and edges
        """
        return self._store.get_supporting_chain(memory_id, ctx)

    # ====================================================================
    # Edge Operations
    # ====================================================================

    def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: GraphEdgeType,
        ctx: RequestContext,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> GraphEdge:
        """Create an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            ctx: RequestContext for access control
            weight: Edge weight
            properties: Additional edge properties

        Returns:
            The created GraphEdge
        """
        edge = create_graph_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            account_id=ctx.account_id,
            weight=weight,
            properties=properties,
        )
        return self._store.create_edge(edge, ctx)

    def delete_edge(self, edge_id: str, ctx: RequestContext) -> None:
        """Delete an edge.

        Args:
            edge_id: UUID of the edge to delete
            ctx: RequestContext for access control
        """
        self._store.delete_edge(edge_id, ctx)

    # ====================================================================
    # PageRank
    # ====================================================================

    def compute_pagerank(
        self,
        ctx: RequestContext,
        node_ids: list[str] | None = None,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank scores.

        Args:
            ctx: RequestContext for access control
            node_ids: Specific nodes to compute for (None = all)
            damping: Damping factor

        Returns:
            Dict mapping node_id to PageRank score
        """
        return self._store.compute_pagerank(ctx, node_ids, damping)

    def get_node_pagerank(self, node_id: str, ctx: RequestContext) -> float:
        """Get the PageRank score for a node.

        Args:
            node_id: UUID of the node
            ctx: RequestContext for access control

        Returns:
            PageRank score
        """
        return self._store.get_node_pagerank(node_id, ctx)

    # ====================================================================
    # Statistics
    # ====================================================================

    def get_graph_stats(self, ctx: RequestContext) -> GraphStats:
        """Get statistics about the graph.

        Args:
            ctx: RequestContext for access control

        Returns:
            GraphStats with node and edge counts, top memories
        """
        return self._store.get_graph_stats(ctx)

    # ====================================================================
    # Node Deletion
    # ====================================================================

    def delete_node(self, node_id: str, ctx: RequestContext) -> None:
        """Delete a node and all connected edges.

        Args:
            node_id: UUID of the node to delete
            ctx: RequestContext for access control
        """
        self._store.delete_node(node_id, ctx)