"""Graph models for heterogeneous multimodal memory system.

Phase 0 frozen — these dataclasses define the shape of all graph domain objects.

Architecture:
- RawNode: Raw multimodal data (prompt, image, video, document, webpage)
- ExtractedNode: Structured extraction results from raw data
- MemoryNode: Assembled memories supported by multiple extractions
- GraphEdge: Relationships between nodes
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

from core.graph_enums import (
    GraphNodeType,
    GraphEdgeType,
    RawDataType,
    NodeStatus,
    ExtractionStatus,
)


def now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def generate_node_id() -> str:
    """Generate a new unique node ID."""
    return uuid.uuid4().hex


def generate_edge_id() -> str:
    """Generate a new unique edge ID."""
    return uuid.uuid4().hex


def _json_value(value: Any) -> Any:
    """Convert graph model values to JSON-serializable primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_json_value(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Base Node
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Base class for all graph nodes.

    All nodes have a unique ID, type, properties, pagerank score,
    timestamps, and account_id for multi-tenant isolation.
    """
    node_id: str
    node_type: GraphNodeType
    status: NodeStatus = NodeStatus.ACTIVE
    pagerank: float = 0.0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    account_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            self.node_id = generate_node_id()

    def touch(self):
        """Update the updated_at timestamp."""
        self.updated_at = now_iso()

    def serialize(self) -> dict[str, Any]:
        """Return JSON-ready base node fields for graph store backends."""
        return {
            "node_id": self.node_id,
            "node_type": _json_value(self.node_type),
            "status": _json_value(self.status),
            "pagerank": self.pagerank,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "account_id": self.account_id,
        }


# ---------------------------------------------------------------------------
# Raw Node
# ---------------------------------------------------------------------------

@dataclass
class RawNode(GraphNode):
    """Raw multimodal data node.

    Stores the original content of multimodal data sources.
    The raw_type determines the content format and available metadata fields.
    """
    raw_type: RawDataType = RawDataType.PROMPT
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = generate_node_id()
        # Enforce node_type
        object.__setattr__(self, 'node_type', GraphNodeType.RAW)

    @property
    def uri(self) -> str:
        """Generate URI for this raw node."""
        return f"raw:{self.raw_type.value}:{self.node_id}"

    def get_embedding(self) -> list[float] | None:
        """Get embedding from metadata if available."""
        return self.metadata.get("embedding")

    def set_embedding(self, embedding: list[float]):
        """Set embedding in metadata."""
        self.metadata["embedding"] = embedding

    def serialize(self) -> dict[str, Any]:
        """Return JSON-ready raw node fields for graph store backends."""
        payload = super().serialize()
        payload.update({
            "raw_type": _json_value(self.raw_type),
            "content": self.content,
            "metadata": _json_value(self.metadata),
        })
        return payload


# ---------------------------------------------------------------------------
# Extracted Node
# ---------------------------------------------------------------------------

@dataclass
class ExtractedNode(GraphNode):
    """Extracted structured data node.

    Stores the result of applying an extractor to a raw node.
    Links back to source raw node via metadata.
    """
    source_raw_id: str = ""
    extractor_type: str = ""
    extraction_schema: str = ""
    extracted_content: dict[str, Any] = field(default_factory=dict)
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    confidence: float = 0.0

    def __post_init__(self):
        if not self.node_id:
            self.node_id = generate_node_id()
        object.__setattr__(self, 'node_type', GraphNodeType.EXTRACTED)

    @property
    def uri(self) -> str:
        """Generate URI for this extracted node."""
        return f"extracted:{self.extractor_type}:{self.node_id}"

    def mark_completed(self, extracted_content: dict[str, Any], confidence: float):
        """Mark extraction as completed with results."""
        self.extraction_status = ExtractionStatus.COMPLETED
        self.extracted_content = extracted_content
        self.confidence = confidence
        self.touch()

    def mark_failed(self, error: str):
        """Mark extraction as failed with error message."""
        self.extraction_status = ExtractionStatus.FAILED
        self.extracted_content["error"] = error
        self.touch()

    def serialize(self) -> dict[str, Any]:
        """Return JSON-ready extracted node fields for graph store backends."""
        payload = super().serialize()
        payload.update({
            "source_raw_id": self.source_raw_id,
            "extractor_type": self.extractor_type,
            "extraction_schema": self.extraction_schema,
            "extracted_content": _json_value(self.extracted_content),
            "extraction_status": _json_value(self.extraction_status),
            "confidence": self.confidence,
        })
        return payload


# ---------------------------------------------------------------------------
# Memory Node
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode(GraphNode):
    """Assembled memory node.

    A memory is a coherent piece of knowledge supported by multiple
    extracted nodes. The supporting_nodes list tracks the evidence chain.
    """
    summary: str = ""
    content: str = ""
    supporting_node_ids: list[str] = field(default_factory=list)
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    parent_memory_id: str | None = None  # For memory evolution

    def __post_init__(self):
        if not self.node_id:
            self.node_id = generate_node_id()
        object.__setattr__(self, 'node_type', GraphNodeType.MEMORY)

    @property
    def uri(self) -> str:
        """Generate URI for this memory node."""
        return f"memory:{self.node_id}"

    @property
    def confidence(self) -> float:
        """Calculate confidence based on supporting nodes.

        Confidence increases with:
        - Number of supporting nodes
        - Average pagerank of supporting nodes
        - Diversity of extractor types
        """
        if not self.supporting_node_ids:
            return 0.0
        return min(1.0, len(self.supporting_node_ids) * 0.1 + self.pagerank * 0.5)

    def add_support(self, node_id: str):
        """Add a supporting node ID."""
        if node_id not in self.supporting_node_ids:
            self.supporting_node_ids.append(node_id)
            self.touch()

    def remove_support(self, node_id: str):
        """Remove a supporting node ID."""
        if node_id in self.supporting_node_ids:
            self.supporting_node_ids.remove(node_id)
            self.touch()

    def serialize(self) -> dict[str, Any]:
        """Return JSON-ready memory node fields for graph store backends."""
        payload = super().serialize()
        payload.update({
            "summary": self.summary,
            "content": self.content,
            "supporting_node_ids": _json_value(self.supporting_node_ids),
            "topic": self.topic,
            "tags": _json_value(self.tags),
            "parent_memory_id": self.parent_memory_id,
            "confidence": self.confidence,
        })
        return payload


# ---------------------------------------------------------------------------
# Graph Edge
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    """A directed edge between two graph nodes.

    Represents relationships like:
    - extracted_from: extracted -> raw
    - supports: extracted -> memory
    - references: any -> any
    - evolves: memory -> memory (older -> newer)
    """
    edge_id: str
    source_id: str
    target_id: str
    edge_type: GraphEdgeType
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    account_id: str = ""
    created_at: str = field(default_factory=now_iso)

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = generate_edge_id()

    def serialize(self) -> dict[str, Any]:
        """Return JSON-ready edge fields for graph store backends."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": _json_value(self.edge_type),
            "weight": self.weight,
            "properties": _json_value(self.properties),
            "account_id": self.account_id,
            "created_at": self.created_at,
        }


@dataclass
class GraphStats:
    """Statistics about a graph."""
    total_nodes: int = 0
    raw_nodes: int = 0
    extracted_nodes: int = 0
    memory_nodes: int = 0
    total_edges: int = 0
    avg_pagerank: float = 0.0
    top_memories: list[dict] = field(default_factory=list)


@dataclass
class SupportingChain:
    """The evidence chain for a memory node.

    Used for visualization and debugging memory provenance.
    """
    memory: MemoryNode
    raw_nodes: list[RawNode] = field(default_factory=list)
    extracted_nodes: list[ExtractedNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class SubgraphResult:
    """Result of a subgraph query."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_raw_node(
    raw_type: RawDataType,
    content: str,
    account_id: str,
    metadata: dict[str, Any] | None = None,
    node_id: str | None = None,
) -> RawNode:
    """Create a new raw node."""
    return RawNode(
        node_id=node_id or generate_node_id(),
        node_type=GraphNodeType.RAW,
        raw_type=raw_type,
        content=content,
        metadata=metadata or {},
        account_id=account_id,
    )


def create_extracted_node(
    source_raw_id: str,
    extractor_type: str,
    extraction_schema: str,
    account_id: str,
    extracted_content: dict[str, Any] | None = None,
    node_id: str | None = None,
) -> ExtractedNode:
    """Create a new extracted node."""
    return ExtractedNode(
        node_id=node_id or generate_node_id(),
        node_type=GraphNodeType.EXTRACTED,
        source_raw_id=source_raw_id,
        extractor_type=extractor_type,
        extraction_schema=extraction_schema,
        extracted_content=extracted_content or {},
        account_id=account_id,
    )


def create_memory_node(
    summary: str,
    account_id: str,
    content: str = "",
    supporting_node_ids: list[str] | None = None,
    topic: str = "",
    tags: list[str] | None = None,
    node_id: str | None = None,
    parent_memory_id: str | None = None,
) -> MemoryNode:
    """Create a new memory node."""
    return MemoryNode(
        node_id=node_id or generate_node_id(),
        node_type=GraphNodeType.MEMORY,
        summary=summary,
        content=content,
        supporting_node_ids=supporting_node_ids or [],
        topic=topic,
        tags=tags or [],
        account_id=account_id,
        parent_memory_id=parent_memory_id,
    )


def create_graph_edge(
    source_id: str,
    target_id: str,
    edge_type: GraphEdgeType,
    account_id: str,
    weight: float = 1.0,
    properties: dict[str, Any] | None = None,
    edge_id: str | None = None,
) -> GraphEdge:
    """Create a new graph edge."""
    return GraphEdge(
        edge_id=edge_id or generate_edge_id(),
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        weight=weight,
        properties=properties or {},
        account_id=account_id,
    )