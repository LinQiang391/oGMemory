"""OutboxWorker: Async index synchronization from OutboxEvents.

Processes OutboxEvents to update the vector index:
1. Read OutboxEvent from store
2. Extract IndexRecords from event payload
3. Embed text using Embedder
4. Upsert to VectorIndex
5. Delete event on success

Happy path: event → embed → upsert → delete
Failure path: retry with exponential backoff → DLQ after MAX_RETRY
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import uuid

from core.interfaces import VectorIndex, Embedder
from core.models import OutboxEvent, IndexRecord


logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_RETRY: int = 5
DEFAULT_BASE_DELAY_MS: int = 1000  # 1 second
DEFAULT_MAX_DELAY_MS: int = 60000  # 1 minute


@dataclass
class WorkerResult:
    """Result of processing a single OutboxEvent."""

    event_id: str
    success: bool
    records_upserted: int = 0
    error_message: str = ""
    moved_to_dlq: bool = False


@dataclass
class WorkerConfig:
    """Configuration for OutboxWorker."""

    max_retry: int = DEFAULT_MAX_RETRY
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS


class OutboxWorker:
    """Processes OutboxEvents to synchronize the vector index.

    The worker reads events, generates embeddings, upserts to the vector index,
    and deletes successfully processed events.

    Events that fail are retried with exponential backoff.
    Events exceeding MAX_RETRY are moved to the dead letter queue (DLQ).
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: WorkerConfig | None = None,
    ) -> None:
        """Initialize the OutboxWorker.

        Args:
            vector_index: Vector index for upsert
            embedder: Embedder for generating embeddings
            config: Worker configuration (optional)
        """
        self._vector_index = vector_index
        self._embedder = embedder
        self._config = config or WorkerConfig()

    def process_event(self, event: OutboxEvent) -> WorkerResult:
        """Process a single OutboxEvent.

        Happy path flow:
        1. Validate event type and payload
        2. Extract IndexRecords from payload
        3. Embed record texts
        4. Upsert to vector index
        5. Return success (caller deletes event)

        On failure:
        - Increment retry_count
        - Return error for retry logic

        Args:
            event: OutboxEvent to process

        Returns:
            WorkerResult with success status
        """
        if event.retry_count >= self._config.max_retry:
            return self._handle_max_retry_exceeded(event)

        try:
            # Extract IndexRecords from event payload
            records = self._extract_records(event)
            if not records:
                logger.warning(f"Event {event.event_id} has no records to index")
                return WorkerResult(
                    event_id=event.event_id,
                    success=True,
                    records_upserted=0,
                )

            # Embed all record texts
            texts = [record.text for record in records]
            embeddings = self._embedder.embed_texts(texts)

            # Augment records with embeddings (for in-memory index)
            # Note: Some VectorIndex implementations embed internally
            augmented_records = self._augment_with_embeddings(records, embeddings)

            # Upsert to vector index
            self._vector_index.upsert(augmented_records)

            logger.info(
                f"Successfully processed event {event.event_id}, "
                f"upserted {len(records)} records for URI {event.uri}"
            )

            return WorkerResult(
                event_id=event.event_id,
                success=True,
                records_upserted=len(records),
            )

        except Exception as e:
            logger.error(
                f"Failed to process event {event.event_id}: {e}",
                exc_info=True,
            )
            return WorkerResult(
                event_id=event.event_id,
                success=False,
                records_upserted=0,
                error_message=str(e),
            )

    def calculate_backoff(self, retry_count: int) -> timedelta:
        """Calculate exponential backoff delay.

        Args:
            retry_count: Current retry count

        Returns:
            Timedelta for delay before next retry
        """
        delay_ms = self._config.base_delay_ms * (2 ** retry_count)
        delay_ms = min(delay_ms, self._config.max_delay_ms)
        return timedelta(milliseconds=delay_ms)

    def should_retry(self, event: OutboxEvent, result: WorkerResult) -> bool:
        """Check if event should be retried.

        Args:
            event: The OutboxEvent that was processed
            result: Result from process_event

        Returns:
            True if event should be retried
        """
        if result.success:
            return False
        if event.retry_count >= self._config.max_retry:
            return False
        return True

    def _extract_records(self, event: OutboxEvent) -> list[IndexRecord]:
        """Extract IndexRecords from event payload.

        Args:
            event: OutboxEvent with payload

        Returns:
            List of IndexRecords to upsert

        Raises:
            ValueError: If payload format is invalid
        """
        payload = event.payload

        # Payload should contain "records" array
        if "records" not in payload:
            raise ValueError(f"Event payload missing 'records': {event.event_id}")

        records_data = payload["records"]
        if not isinstance(records_data, list):
            raise ValueError(f"Event 'records' is not a list: {event.event_id}")

        # Reconstruct IndexRecord objects
        records: list[IndexRecord] = []
        for record_dict in records_data:
            # Validate required fields
            required_fields = {"id", "uri", "level", "text", "filters"}
            missing = required_fields - record_dict.keys()
            if missing:
                raise ValueError(
                    f"Record missing fields {missing}: {event.event_id}"
                )

            # Validate mandatory filters (cross-tenant leakage prevention)
            if "account_id" not in record_dict["filters"]:
                raise ValueError(
                    f"Record missing account_id in filters: {event.event_id}. "
                    "This causes SILENT cross-tenant leakage!"
                )
            if "owner_space" not in record_dict["filters"]:
                raise ValueError(
                    f"Record missing owner_space in filters: {event.event_id}. "
                    "This causes SILENT cross-tenant leakage!"
                )

            record = IndexRecord(
                id=record_dict["id"],
                uri=record_dict["uri"],
                level=record_dict["level"],
                text=record_dict["text"],
                filters=record_dict["filters"],
                metadata=record_dict.get("metadata", {}),
            )
            records.append(record)

        # SECURITY: Validate all record account_ids match event URI
        # This prevents cross-tenant injection via malicious payloads
        try:
            expected_account = event.uri.split("://")[1].split("/")[0]
        except (IndexError, AttributeError):
            raise ValueError(
                f"Cannot extract account from event URI: {event.uri}"
            )

        for record in records:
            record_account = record.filters.get("account_id", "")
            if record_account != expected_account:
                raise ValueError(
                    f"Record account_id '{record_account}' does not match "
                    f"event URI account '{expected_account}': {event.event_id}. "
                    "Possible cross-tenant injection attempt!"
                )

        return records

    def _augment_with_embeddings(
        self,
        records: list[IndexRecord],
        embeddings: list[list[float]],
    ) -> list[IndexRecord]:
        """Augment IndexRecords with embedding vectors.

        Some VectorIndex implementations need embeddings in the record.
        This adds embeddings to metadata for those implementations.

        Args:
            records: IndexRecords without embeddings
            embeddings: Embedding vectors

        Returns:
            Augmented records with embeddings in metadata
        """
        augmented: list[IndexRecord] = []

        for record, embedding in zip(records, embeddings):
            # Create a copy with embedding in metadata
            new_metadata = dict(record.metadata)
            new_metadata["_embedding"] = embedding

            # Create new IndexRecord with augmented metadata
            # Note: IndexRecord is a dataclass, we can create a new instance
            augmented_record = IndexRecord(
                id=record.id,
                uri=record.uri,
                level=record.level,
                text=record.text,
                filters=record.filters,
                metadata=new_metadata,
            )
            augmented.append(augmented_record)

        return augmented

    def _handle_max_retry_exceeded(self, event: OutboxEvent) -> WorkerResult:
        """Handle event that has exceeded max retry attempts.

        Args:
            event: OutboxEvent with retry_count >= MAX_RETRY

        Returns:
            WorkerResult indicating DLQ transfer
        """
        logger.error(
            f"Event {event.event_id} exceeded max retry "
            f"({self._config.max_retry}), moving to DLQ"
        )
        return WorkerResult(
            event_id=event.event_id,
            success=False,
            records_upserted=0,
            error_message="Max retry exceeded",
            moved_to_dlq=True,
        )

    def run_once(
        self,
        outbox_store,  # OutboxStore type, avoid import to prevent circular dependency
        account_ids: list[str],
        worker_id: str = "default",
    ) -> dict:
        """Single polling cycle: scan → process → ack/retry/dlq.

        Call this from a scheduler (e.g., APScheduler, cron, or asyncio loop).

        Args:
            outbox_store: OutboxStore instance with list_pending/mark_done/etc.
            account_ids: List of account IDs to scan
            worker_id: Unique identifier for this worker (for distributed locking)

        Returns:
            Dict with counts: processed / succeeded / failed / moved_to_dlq / skipped
        """
        stats = {"processed": 0, "succeeded": 0, "failed": 0, "moved_to_dlq": 0, "skipped": 0}

        for account_id in account_ids:
            try:
                pending = outbox_store.list_pending(account_id)
            except Exception as e:
                logger.error("Failed to list pending events for %s: %s", account_id, e)
                continue

            for node_uri, event in pending:
                # Try to acquire lock for this event
                if not outbox_store.try_acquire(event, node_uri, worker_id):
                    stats["skipped"] += 1
                    continue

                stats["processed"] += 1

                try:
                    result = self.process_event(event)

                    if result.success:
                        outbox_store.mark_done(event, node_uri)
                        stats["succeeded"] += 1
                    elif result.moved_to_dlq:
                        outbox_store.move_to_dlq(event, node_uri)
                        stats["moved_to_dlq"] += 1
                    else:
                        outbox_store.increment_retry(event, node_uri)
                        outbox_store.release(event, node_uri)
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(
                        "Exception processing event %s: %s",
                        event.event_id, e, exc_info=True
                    )
                    outbox_store.release(event, node_uri)
                    stats["failed"] += 1

        return stats


def create_upsert_event(
    uri: str,
    records: list[IndexRecord],
) -> OutboxEvent:
    """Create an UPSERT_CONTEXT OutboxEvent.

    Utility function for write-path to create events.

    Args:
        uri: ContextNode URI
        records: IndexRecords to upsert

    Returns:
        OutboxEvent ready for serialization
    """
    # Serialize IndexRecords to dict for JSON storage
    records_data = [
        {
            "id": r.id,
            "uri": r.uri,
            "level": r.level,
            "text": r.text,
            "filters": r.filters,
            "metadata": r.metadata,
        }
        for r in records
    ]

    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type="UPSERT_CONTEXT",
        uri=uri,
        payload={
            "uri": uri,
            "records": records_data,
        },
        status="PENDING",
        retry_count=0,
    )
