"""Domain error types for ContextEngine.

Phase 0 frozen — specific exceptions enable precise error handling.
All errors inherit from standard Exception for compatibility.
"""


class ContextEngineError(Exception):
    """Base exception for all ContextEngine errors."""
    pass


class AccessDeniedError(ContextEngineError):
    """Access control violation.

    Raised when RequestContext.account_id does not match URI account prefix.
    Also raised for unauthorized cross-agent or user-space access.
    """
    def __init__(self, uri: str, account_id: str, reason: str = ""):
        self.uri = uri
        self.account_id = account_id
        self.reason = reason
        message = f"Access denied to {uri} for account {account_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class NodeNotFoundError(ContextEngineError):
    """Requested node does not exist or is not ACTIVE.

    Distinguished from HTTP 404 — node may exist but be PENDING/BROKEN.
    """
    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Node not found or not ACTIVE: {uri}")


class NodeBrokenError(ContextEngineError):
    """Node exists but is in BROKEN state.

    Indicates Repair Job detected corruption requiring manual intervention.
    """
    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Node is BROKEN: {uri}")


class OutboxError(ContextEngineError):
    """Outbox processing failure.

    Raised by OutboxWorker when event processing exceeds retry limit.
    Event is moved to DLQ (dead letter queue).
    """
    def __init__(self, event_id: str, reason: str):
        self.event_id = event_id
        self.reason = reason
        super().__init__(f"Outbox event {event_id} failed: {reason}")


class MergePolicyError(ContextEngineError):
    """Merge policy evaluation failure.

    Raised when MergePolicy cannot determine a valid WritePlan.
    """
    def __init__(self, candidate_uri: str, reason: str):
        self.candidate_uri = candidate_uri
        self.reason = reason
        super().__init__(f"Merge policy failed for {candidate_uri}: {reason}")


class EmbeddingError(ContextEngineError):
    """Embedding generation failure.

    Raised when Embedder fails to produce vectors.
    """
    def __init__(self, text_preview: str, reason: str):
        self.text_preview = text_preview[:100]
        self.reason = reason
        super().__init__(f"Embedding failed for '{text_preview[:50]}...': {reason}")


class VectorIndexError(ContextEngineError):
    """Vector index operation failure.

    Raised when VectorIndex upsert/delete/search fails.
    """
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector index {operation} failed: {reason}")


class ValidationError(ContextEngineError):
    """Schema or validation failure.

    Raised when input does not match expected schema.
    """
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation failed for {field}: {reason}")


class ConcurrentModificationError(ContextEngineError):
    """Optimistic lock failure during concurrent merge.

    Raised when a merge write detects that the node was modified
    between read and write (version mismatch).

    Indicates the operation should be retried with fresh data.
    """
    def __init__(self, uri: str, expected_version: int, actual_version: int):
        self.uri = uri
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrent modification detected for {uri}: "
            f"expected version {expected_version}, found {actual_version}"
        )


# ---------------------------------------------------------------------------
# Retrieval pipeline errors
# ---------------------------------------------------------------------------

class RetrievalError(ContextEngineError):
    """Base exception for retrieval pipeline failures."""
    def __init__(self, stage: str, reason: str, *, retryable: bool = False):
        self.stage = stage
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Retrieval [{stage}] failed: {reason}")


class RetrievalPlannerError(RetrievalError):
    """Query planner could not classify the query."""
    def __init__(self, reason: str):
        super().__init__("planner", reason)


class RetrievalSearchError(RetrievalError):
    """Vector search or hierarchical expansion failed."""
    def __init__(self, reason: str, *, retryable: bool = True):
        super().__init__("search", reason, retryable=retryable)
