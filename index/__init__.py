"""Index package: Async index synchronization and repair.

This package handles:
- ContextNode → IndexRecord expansion (3-level)
- Outbox event processing with retry/DLQ
- Repair job for corrupted nodes

Key invariant: All IndexRecords MUST contain account_id + owner_space in filters.
"""

from index.index_record_builder import (
    build_index_records,
    build_single_record,
    _extract_account_id,
    _validate_filters,
    FilterValidationError,
)

from index.outbox_worker import (
    OutboxWorker,
    WorkerResult,
    WorkerConfig,
    create_upsert_event,
)

__all__ = [
    "build_index_records",
    "build_single_record",
    "_extract_account_id",
    "_validate_filters",
    "FilterValidationError",
    "OutboxWorker",
    "WorkerResult",
    "WorkerConfig",
    "create_upsert_event",
]
