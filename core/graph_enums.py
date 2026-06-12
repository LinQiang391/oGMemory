"""Graph enumerations for heterogeneous memory system.

Defines node types, edge types, and related enums for the graph-based
multimodal memory architecture.
"""

from enum import Enum


class GraphNodeType(str, Enum):
    """Types of nodes in the heterogeneous graph.

    Three layers:
    - raw: Raw multimodal data (prompt, image, video, document, webpage)
    - extracted: Structured extraction results from raw data
    - memory: Assembled memories supported by multiple extractions
    """
    RAW = "raw"
    EXTRACTED = "extracted"
    MEMORY = "memory"


class RawDataType(str, Enum):
    """Types of raw multimodal data that can be stored as raw nodes."""
    PROMPT = "prompt"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    AUDIO = "audio"


class GraphEdgeType(str, Enum):
    """Types of edges in the heterogeneous graph.

    - extracted_from: Links extracted node to its source raw node
    - supports: Links supporting extracted nodes to a memory node
    - references: General reference relationship
    - evolves: Memory evolution (older -> newer)
    """
    EXTRACTED_FROM = "extracted_from"
    SUPPORTS = "supports"
    REFERENCES = "references"
    EVOLVES = "evolves"


class NodeStatus(str, Enum):
    """Status of a graph node."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING = "pending"


class ExtractionStatus(str, Enum):
    """Status of an extraction operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"