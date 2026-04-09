"""Core domain models for ContextEngine.

Phase 0 frozen — these dataclasses define the shape of all domain objects.
All models use stdlib dataclass for minimal dependencies.

Retrieval-chain models (SeedHit → SearchMemoryResult) added for the
search_memory pipeline; they follow the same conventions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import hashlib
import uuid


def user_space_name(user_id: str) -> str:
    """Generate owner_space identifier for a user.

    Uses colon format for consistency across the codebase.
    Format: "user:{user_id}"

    Args:
        user_id: User identifier

    Returns:
        Owner space string in format "user:{user_id}"
    """
    return f"user:{user_id}"


def agent_space_name(agent_id: str) -> str:
    """Generate owner_space identifier for an agent.

    Uses colon format for consistency across the codebase.
    Format: "agent:{agent_id}"

    Args:
        agent_id: Agent identifier

    Returns:
        Owner space string in format "agent:{agent_id}"
    """
    return f"agent:{agent_id}"


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

    def user_space_name(self) -> str:
        """Generate owner_space identifier for user scope.

        Returns owner_space in format 'user:{user_id}'.
        """
        return f"user:{self.user_id}"

    def agent_space_name(self) -> str:
        """Generate owner_space identifier for agent scope.

        Returns owner_space in format 'agent:{agent_id}'.
        """
        return f"agent:{self.agent_id}"


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

    @property
    def parent_uri(self) -> str:
        """Get parent URI by removing the last path segment.

        For example:
        - "ctx://acme/users/alice/memories/preferences/coding_style"
          → "ctx://acme/users/alice/memories/preferences/"
        - "ctx://acme/users/alice/memories/profile"
          → "ctx://acme/users/alice/memories/"
        - "ctx://acme/users/alice/memories/preferences/"
          → "ctx://acme/users/alice/memories/"
        """
        if "/" not in self.uri:
            return ""

        uri = self.uri
        if uri.endswith("/"):
            uri = uri.rstrip("/")

        if "/" not in uri:
            return ""

        return uri.rsplit("/", 1)[0] + "/"


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
    event_type: str  # UPSERT_CONTEXT|UPSERT_DIRECTORY|DELETE_CONTEXT|MOVE_CONTEXT|UPSERT_RELATION
    uri: str
    payload: dict[str, Any]
    status: str  # PENDING | PROCESSING | DONE | FAILED
    retry_count: int = 0
    created_at: str = ""  # ISO8601, set to datetime.utcnow().isoformat() if empty
    next_retry_at: str = ""  # ISO8601, earliest time for next retry (exponential backoff)

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


# ---------------------------------------------------------------------------
# Retrieval pipeline models
# ---------------------------------------------------------------------------

class RetrieverMode(str, Enum):
    """Controls depth of scoring: QUICK skips reranker, THINKING uses it."""
    THINKING = "thinking"
    QUICK = "quick"


@dataclass
class SeedHit:
    """Raw result from VectorIndex.

    Carries level and parent_uri needed for hierarchical expansion.
    """
    uri: str
    score: float
    level: int = 2
    parent_uri: str | None = None
    context_type: str = ""
    category: str = ""
    owner_space: str = ""
    abstract: str = ""
    has_overview: bool = False
    has_content: bool = False
    active_count: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedResult:
    """Output of SeedRetriever — two streams after level-based split."""
    starting_points: list[SeedHit] = field(default_factory=list)
    initial_candidates: list[SeedHit] = field(default_factory=list)
    query_vector: list[float] = field(default_factory=list)
    root_uris: list[str] = field(default_factory=list)


@dataclass
class LeafHit:
    """L2 node produced by HierarchicalSearcher after recursive expansion."""
    uri: str
    score: float
    level: int = 2
    depth: int = 0
    parent_uri: str | None = None
    category: str = ""
    owner_space: str = ""
    abstract: str = ""
    has_overview: bool = False
    has_content: bool = False
    hierarchy_path: list[str] = field(default_factory=list)
    active_count: int = 0
    updated_at: str | None = None


@dataclass
class RetrievedBlock:
    """Final output block returned by search_memory or read_memory."""
    uri: str
    level_hit: str = "L2"
    score: float = 0.0
    category: str = ""
    owner_space: str = ""
    abstract: str | None = None
    overview: str | None = None
    content_excerpt: str | None = None
    relations: list[RelationEdge] = field(default_factory=list)
    has_overview: bool = False
    has_content: bool = False
    match_reason: str = ""


@dataclass
class SearchMemoryResult:
    """Top-level result of search_memory — pure structured data."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    typed_queries: list[TypedQuery] = field(default_factory=list)
    hits: list[RetrievedBlock] = field(default_factory=list)
    trace: Any = None


@dataclass
class RetrievalConfig:
    """Tunables shared across all retrieval pipeline stages."""
    max_top_k: int = 50
    default_top_k: int = 10
    global_search_topk: int = 5
    max_convergence_rounds: int = 3
    score_propagation_alpha: float = 0.5
    hotness_alpha: float = 0.2
    hotness_half_life_days: float = 7.0
    default_score_threshold: float = 0.0
