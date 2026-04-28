"""core 包 — Phase 0 由 core-architect 实现
包含：models.py / interfaces.py / errors.py / enums.py / logging_config.py / validation.py
"""

# Export core models
# Export core enums
from core.enums import (
    ContextType,
    EventType,
    NodeStatus,
)

# Export core errors
from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    ContextEngineError,
    EmbeddingError,
    MergePolicyError,
    NodeBrokenError,
    NodeNotFoundError,
    OutboxError,
    RetrievalError,
    RetrievalPlannerError,
    RetrievalSearchError,
    ValidationError,
    VectorIndexError,
)

# Export core interfaces
from core.interfaces import (
    LLM,
    CandidateExtractor,
    ContextFS,
    Embedder,
    MergePolicy,
    OutboxStore,
    RelationStore,
    VectorIndex,
)

# Export logging configuration
from core.logging_config import get_logger, setup_logging, with_context
from core.models import (
    CandidateMemory,
    ContextNode,
    IndexRecord,
    LeafHit,
    OutboxEvent,
    RelationEdge,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
    SeedHit,
    SeedResult,
    TypedQuery,
    WritePlan,
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
    # Logging
    "setup_logging",
    "get_logger",
    "with_context",
]
