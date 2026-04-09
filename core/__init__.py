"""core 包 — Phase 0 由 core-architect 实现
包含：models.py / interfaces.py / errors.py / enums.py / logging_config.py / validation.py
"""

# Export core models
from core.models import (
    RequestContext,
    ContextNode,
    RelationEdge,
    CandidateMemory,
    WritePlan,
    IndexRecord,
    OutboxEvent,
    TypedQuery,
    SeedHit,
    SeedResult,
    LeafHit,
    RetrievedBlock,
    SearchMemoryResult,
    RetrievalConfig,
    RetrieverMode,
    user_space_name,
    agent_space_name,
)

# Export core interfaces
from core.interfaces import (
    ContextFS,
    RelationStore,
    CandidateExtractor,
    MergePolicy,
    VectorIndex,
    Embedder,
    LLM,
    OutboxStore,
)

# Export core errors
from core.errors import (
    ContextEngineError,
    AccessDeniedError,
    NodeNotFoundError,
    NodeBrokenError,
    OutboxError,
    MergePolicyError,
    EmbeddingError,
    VectorIndexError,
    ValidationError,
    ConcurrentModificationError,
    RetrievalError,
    RetrievalPlannerError,
    RetrievalSearchError,
)

# Export core enums
from core.enums import (
    NodeStatus,
    ContextType,
    EventType,
    OwnerScope,
)

# Export logging configuration
from core.logging_config import setup_logging, get_logger, with_context

# Export validation functions
from core.validation import (
    validate_candidate,
    validate_text_length,
    validate_utf8,
    sanitize_html,
    validate_slug,
    normalize_slug,
)

__all__ = [
    # Models
    "RequestContext",
    "ContextNode",
    "RelationEdge",
    "CandidateMemory",
    "WritePlan",
    "IndexRecord",
    "OutboxEvent",
    "TypedQuery",
    "SeedHit",
    "SeedResult",
    "LeafHit",
    "RetrievedBlock",
    "SearchMemoryResult",
    "RetrievalConfig",
    "RetrieverMode",
    "user_space_name",
    "agent_space_name",
    # Interfaces
    "ContextFS",
    "RelationStore",
    "CandidateExtractor",
    "MergePolicy",
    "VectorIndex",
    "Embedder",
    "LLM",
    "OutboxStore",
    # Errors
    "ContextEngineError",
    "AccessDeniedError",
    "NodeNotFoundError",
    "NodeBrokenError",
    "OutboxError",
    "MergePolicyError",
    "EmbeddingError",
    "VectorIndexError",
    "ValidationError",
    "ConcurrentModificationError",
    "RetrievalError",
    "RetrievalPlannerError",
    "RetrievalSearchError",
    # Enums
    "NodeStatus",
    "ContextType",
    "EventType",
    "OwnerScope",
    # Logging
    "setup_logging",
    "get_logger",
    "with_context",
    # Validation
    "validate_candidate",
    "validate_text_length",
    "validate_utf8",
    "sanitize_html",
    "validate_slug",
    "normalize_slug",
]
