"""Graph enumerations for heterogeneous memory system.

Defines node types, edge types, and related enums for the graph-based
multimodal memory architecture.
"""

from enum import StrEnum


class GraphNodeType(StrEnum):
    """Types of nodes in the heterogeneous graph.

    Three layers:
    - raw: Raw multimodal data (prompt, image, video, document, webpage)
    - extracted: Structured extraction results from raw data
    - memory: Assembled memories supported by multiple extractions
    """
    RAW = "raw"
    EXTRACTED = "extracted"
    MEMORY = "memory"


class RawDataType(StrEnum):
    """Types of raw multimodal data that can be stored as raw nodes.

    These describe the *input* shape of data flowing into the graph —
    bytes on disk (or in-memory) that have not yet been processed by
    an operator. Operator outputs (face detections, object detections,
    episodic memories, etc.) live on :class:`ExtractedNode` /
    :class:`MemoryNode` instead, never here, so the boundary between
    raw data and derived structure stays clean.
    """
    PROMPT = "prompt"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    AUDIO = "audio"


class GraphEdgeType(StrEnum):
    """Types of edges in the heterogeneous graph.

    - extracted_from: Links extracted node to its source raw node
    - supports: Links supporting extracted nodes to a memory node
    - references: General reference relationship
    - evolves: Memory evolution (older -> newer)
    - same_identity: Two ExtractedNodes depict the same real-world
      identity (e.g. the same face across clips, the same voice
      across audio segments). Used by face_clustering and
      voice_clustering to collapse cross-clip detections. Edges
      are undirected in meaning; the source/target order is the
      newer detection pointing at the cluster anchor.
    - co_occurs_in: Two ExtractedNodes appear together in the
      same source (e.g. two faces in the same frame, two voices
      in the same clip). Carries a weight property for the joint
      occurrence count. Used as a feature for clustering and as
      evidence in cross-modal retrieval.
    """
    EXTRACTED_FROM = "extracted_from"
    SUPPORTS = "supports"
    REFERENCES = "references"
    EVOLVES = "evolves"
    SAME_IDENTITY = "same_identity"
    CO_OCCURS_IN = "co_occurs_in"


class NodeStatus(StrEnum):
    """Status of a graph node."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING = "pending"


class ExtractionStatus(StrEnum):
    """Status of an extraction operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
