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
from core.react_loop import (
    ReactIterationTrace,
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
    ReactTrace,
)
from core.resource_identity import (
    CODE_INDEX_METADATA_FIELDS,
    AmbiguousRootBindingError,
    ResourceCollisionError,
    ResourceIdentityError,
    RootBindingError,
    canonical_node_uri,
    code_index_metadata,
    compute_chunk_key,
    compute_file_key,
    compute_root_fingerprint,
    digest_slug,
    guard_full_key_collision,
    metadata_matches_exact,
    normalize_posix_relative_path,
    normalize_resource_relative_path,
    redact_path,
    resolve_root_slug,
    resource_owner_space,
    select_nodes_by_metadata,
    validate_root_binding,
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
    # Resource identity
    "CODE_INDEX_METADATA_FIELDS",
    "ResourceIdentityError",
    "ResourceCollisionError",
    "RootBindingError",
    "AmbiguousRootBindingError",
    "resource_owner_space",
    "compute_root_fingerprint",
    "normalize_resource_relative_path",
    "normalize_posix_relative_path",
    "compute_file_key",
    "compute_chunk_key",
    "digest_slug",
    "guard_full_key_collision",
    "validate_root_binding",
    "resolve_root_slug",
    "metadata_matches_exact",
    "select_nodes_by_metadata",
    "canonical_node_uri",
    "redact_path",
    "code_index_metadata",
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
    # React loop
    "ReactLoop",
    "ReactLoopResult",
    "ReactIterationTrace",
    "ReactTaskContext",
    "ReactTrace",
]
