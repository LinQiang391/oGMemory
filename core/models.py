"""Core domain models for ContextEngine.

Phase 0 frozen — these dataclasses define the shape of all domain objects.
All models use stdlib dataclass for minimal dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib


@dataclass(frozen=True)
class RequestContext:
    """Execution context for all operations.

    Enforces multi-tenant isolation by carrying account_id and owner_space.
    Passed to all ContextFS and RelationStore methods for access control.
    """
    account_id: str
    user_id: str
    agent_id: str
    session_id: str
    trace_id: str


@dataclass
class ContextNode:
    """A node in the context knowledge graph.

    Persisted to AGFS as a directory with content.md, .relations.json,
    .abstract.md, .overview.md, .meta.json files.

    See CLAUDE.md §2 AGFS Directory Spec for file layout.
    See CLAUDE.md §1 URI Spec for URI patterns by category.
    """
    uri: str
    context_type: str  # MEMORY | SKILL | RESOURCE
    category: str  # profile|preference|entity|event|case|pattern|skill
    level: int  # Depth in directory tree (0 for root-level nodes)
    owner_space: str  # user_space_name() or agent_space_name()
    abstract: str  # ≤100 chars summary (→ .abstract.md)
    overview: str  # Structured overview (→ .overview.md)
    content: str  # Full original content (→ content.md)
    metadata: dict[str, Any] = field(default_factory=dict)  # tags / timestamps / status


@dataclass
class RelationEdge:
    """A directed relationship between two ContextNodes.

    Persisted in .relations.json at the source node.
    Weight indicates confidence/strength (0.0-1.0).
    """
    from_uri: str
    to_uri: str
    relation_type: str  # e.g. "related_to" | "derived_from" | "contradicts" | "SEQUENCE"
    weight: float  # 0.0-1.0
    reason: str


@dataclass
class CandidateMemory:
    """Extracted candidate before merge policy decisions.

    Produced by CandidateExtractor from message history.
    Consumed by MergePolicy to produce WritePlan.
    """
    category: str  # Same as ContextNode.category
    owner_scope: str  # "user" | "agent"
    routing_key: str  # slug or event_id candidate value
    abstract: str
    overview: str
    content: str
    confidence: float  # 0.0-1.0, below threshold triggers skip


@dataclass
class WritePlan:
    """Output of MergePolicy — determines write action.

    Action values:
    - create: New node, does not exist
    - merge: Existing node, merge content
    - append: New version to existing node (events/cases)
    - skip: Duplicate or below confidence threshold
    """
    action: str  # create | merge | append | skip
    target_uri: str
    merged_fields: dict[str, Any] = field(default_factory=dict)  # For merge actions
    relation_edges: list[RelationEdge] = field(default_factory=list)


@dataclass
class IndexRecord:
    """A record in the vector index.

    Each ContextNode expands to 3 IndexRecords (L0/L1/L2).
    The id is stable: sha256(uri + ":" + str(level))[:16].

    CRITICAL: filters MUST contain account_id + owner_space.
    Missing filters cause silent cross-tenant leakage (no runtime error).
    Contract test enforces this invariant.
    """
    id: str
    uri: str
    level: int  # 0=abstract / 1=overview / 2=content
    text: str  # Text content for embedding
    filters: dict[str, Any]  # MUST contain account_id + owner_space
    metadata: dict[str, Any] = field(default_factory=dict)  # category / context_type / has_overview / has_content

    @staticmethod
    def generate_id(uri: str, level: int) -> str:
        """Generate stable IndexRecord id.

        Stable id enables idempotent upsert to vector index.
        """
        return hashlib.sha256(f"{uri}:{level}".encode()).hexdigest()[:16]


@dataclass
class OutboxEvent:
    """Event for async index synchronization.

    Written to .outbox/{event_id}.json, processed by OutboxWorker.
    Events are durable and survive process restarts (stored in AGFS).
    """
    event_id: str  # uuid4
    event_type: str  # UPSERT_CONTEXT|DELETE_CONTEXT|MOVE_CONTEXT|UPSERT_RELATION
    uri: str
    payload: dict[str, Any]
    status: str  # PENDING | PROCESSING | DONE | FAILED
    retry_count: int = 0
    created_at: str = ""  # ISO8601, set to datetime.utcnow().isoformat() if empty

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TypedQuery:
    """A query with type annotations for filtered retrieval.

    account_id is injected by service layer, cannot be overridden by caller.
    categories is inferred by Planner from query text.
    target_uri allows direct lookup skipping vector recall.
    """
    text: str  # Original query text, used for vectorization
    context_type: str  # MEMORY | SKILL | RESOURCE
    categories: list[str]  # Planner-inferred categories for pre-filtering
    target_uri: str | None = None  # If known, direct read, skip vector recall
    top_k: int = 10  # L0 recall count limit
    account_id: str = ""  # Injected by service layer
    owner_space: str | None = None  # Further limit to user or agent scope


@dataclass
class SeedHit:
    """Result from L0 vector recall.

    Returned by VectorIndex.search(), consumed by AssemblyService.
    """
    uri: str
    score: float  # Similarity score
    abstract: str  # L0 abstract text
    has_overview: bool
    has_content: bool
