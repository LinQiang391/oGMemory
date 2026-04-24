"""OutboxWorker: Async index synchronization from OutboxEvents.

Processes OutboxEvents to update the vector index:
1. Read OutboxEvent from store
2. Handle event by type:
   - UPSERT_CONTEXT: Embed and upsert leaf node records
   - UPSERT_DIRECTORY: Generate directory summary and upsert
   - DELETE_CONTEXT: Remove from vector index
3. Delete event on success

Happy path: event → process → delete
Failure path: retry with exponential backoff → DLQ after MAX_RETRY
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import uuid
import re

from core.interfaces import VectorIndex, Embedder, ContextFS, LLM
from core.models import OutboxEvent, IndexRecord, RequestContext


logger = logging.getLogger(__name__)

_MEMORY_OWNER_PATTERN = re.compile(
    r"^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/"
)

DEFAULT_MAX_RETRY: int = 5
DEFAULT_BASE_DELAY_MS: int = 1000
DEFAULT_MAX_DELAY_MS: int = 60000


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

    Event types:
    - UPSERT_CONTEXT: Embed and upsert leaf node records
    - UPSERT_DIRECTORY: Generate directory summary and upsert
    - DELETE_CONTEXT: Remove from vector index

    Events that fail are retried with exponential backoff.
    Events exceeding MAX_RETRY are moved to the dead letter queue (DLQ).
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: WorkerConfig | None = None,
        fs: ContextFS | None = None,
        llm: LLM | None = None,
        directory_summary_enabled: bool = True,
    ) -> None:
        """Initialize the OutboxWorker.

        Args:
            vector_index: Vector index for upsert
            embedder: Embedder for generating embeddings
            config: Worker configuration (optional)
            fs: ContextFS for reading directory children (required for UPSERT_DIRECTORY)
            llm: LLM for generating directory summaries (required for UPSERT_DIRECTORY)
            directory_summary_enabled: If False, skip UPSERT_DIRECTORY events
        """
        self._vector_index = vector_index
        self._embedder = embedder
        self._config = config or WorkerConfig()
        self._fs = fs
        self._llm = llm
        self._directory_summary_enabled = directory_summary_enabled

    def process_event(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None = None,
    ) -> WorkerResult:
        """Process a single OutboxEvent.

        Routes to appropriate handler based on event_type.

        Args:
            event: OutboxEvent to process
            ctx: RequestContext (required for UPSERT_DIRECTORY)

        Returns:
            WorkerResult with success status
        """
        if event.retry_count >= self._config.max_retry:
            return self._handle_max_retry_exceeded(event)

        try:
            event_type = event.event_type

            if event_type == "UPSERT_CONTEXT":
                return self._process_upsert_context(event)
            elif event_type == "UPSERT_DIRECTORY":
                return self._process_upsert_directory(event, ctx)
            elif event_type == "DELETE_CONTEXT":
                return self._process_delete_context(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return WorkerResult(
                    event_id=event.event_id,
                    success=True,
                    records_upserted=0,
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

    def _process_upsert_context(self, event: OutboxEvent) -> WorkerResult:
        """Process UPSERT_CONTEXT event for leaf node.

        Flow:
        1. Extract IndexRecords from payload
        2. Embed record texts
        3. Upsert to vector index

        Args:
            event: OutboxEvent with leaf node records

        Returns:
            WorkerResult with success status
        """
        records = self._extract_records(event)
        if not records:
            logger.warning(f"Event {event.event_id} has no records to index")
            return WorkerResult(
                event_id=event.event_id,
                success=True,
                records_upserted=0,
            )

        texts = [record.text for record in records]
        embeddings = self._embedder.embed_texts(texts)
        augmented_records = self._augment_with_embeddings(records, embeddings)
        self._vector_index.upsert(augmented_records)

        logger.info(
            f"Successfully processed UPSERT_CONTEXT {event.event_id}, "
            f"upserted {len(records)} records for URI {event.uri}"
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=len(records),
        )

    def _process_upsert_directory(
        self,
        event: OutboxEvent,
        ctx: RequestContext | None,
    ) -> WorkerResult:
        """Process UPSERT_DIRECTORY event for parent directory.

        Flow:
        1. Check fs and llm are available
        2. Use DirectoryEventHandler for DAG-style bottom-up processing
        3. Handler waits for subdirectory summaries before generating parent
        4. Handler uses fallback when LLM fails
        5. Handler vectorizes and stores to database

        Args:
            event: OutboxEvent with directory URI
            ctx: RequestContext (required for FS access)

        Returns:
            WorkerResult with success status
        """
        if not self._directory_summary_enabled:
            logger.info("UPSERT_DIRECTORY skipped (directory summary disabled): %s", event.uri)
            return WorkerResult(
                event_id=event.event_id,
                success=True,
                records_upserted=0,
            )

        if not self._fs or not self._llm:
            raise ValueError(
                "UPSERT_DIRECTORY requires fs and llm. "
                "Initialize OutboxWorker with fs and llm parameters."
            )

        if not ctx:
            raise ValueError(
                "UPSERT_DIRECTORY requires RequestContext. "
                "Pass ctx parameter to process_event()."
            )

        from index.directory_event_handler import DirectoryEventHandler

        handler = DirectoryEventHandler(
            fs=self._fs,
            llm=self._llm,
            embedder=self._embedder,
            vector_index=self._vector_index,
        )

        result = handler.process_directory_event(event, ctx)

        if not result.success:
            logger.error(
                "Failed to process directory event %s: %s",
                event.uri, result.error_message
            )
            return WorkerResult(
                event_id=event.event_id,
                success=False,
                records_upserted=0,
                error_message=result.error_message,
            )

        logger.info(
            "Successfully processed UPSERT_DIRECTORY %s, "
            "completed %d directories, failed %d directories",
            event.event_id,
            result.stats.completed_dirs,
            result.stats.failed_dirs,
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=result.stats.completed_dirs,
        )

    def _process_delete_context(self, event: OutboxEvent) -> WorkerResult:
        """Process DELETE_CONTEXT event.

        Args:
            event: OutboxEvent with deleted URI

        Returns:
            WorkerResult with success status
        """
        from index.index_record_builder import build_record_id

        ids_to_delete = [
            build_record_id(event.uri, level)
            for level in [0, 1, 2]
        ]

        self._vector_index.delete(ids_to_delete)

        logger.info(
            f"Successfully processed DELETE_CONTEXT {event.event_id}, "
            f"deleted {len(ids_to_delete)} records for URI {event.uri}"
        )

        return WorkerResult(
            event_id=event.event_id,
            success=True,
            records_upserted=0,
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

    def _handle_processing_result(
        self,
        event: OutboxEvent,
        node_uri: str,
        outbox_store,
        result: WorkerResult,
        stats: dict,
    ) -> None:
        """Handle the result of processing an event.
        
        Args:
            event: The OutboxEvent that was processed
            node_uri: URI of the associated node
            outbox_store: OutboxStore instance
            result: Processing result
            stats: Statistics dict to update
        """
        if result.success:
            outbox_store.mark_done(event, node_uri)
            stats["succeeded"] += 1
        elif result.moved_to_dlq or not self.should_retry(event, result):
            # Move to DLQ if explicitly marked or retries exhausted
            outbox_store.move_to_dlq(event, node_uri)
            stats["moved_to_dlq"] += 1
            if not result.moved_to_dlq:
                logger.error(
                    "Event %s exceeded max retry, moved to DLQ",
                    event.event_id
                )
        else:
            # Schedule for retry
            backoff = self.calculate_backoff(event.retry_count)
            next_retry_at = datetime.now(timezone.utc) + backoff
            outbox_store.increment_retry(event, node_uri, next_retry_at)
            outbox_store.release(event, node_uri)
            stats["failed"] += 1
            logger.info(
                "Event %s scheduled for retry #%d in %s",
                event.event_id, event.retry_count + 1, backoff
            )

    def _build_request_context(self, event: OutboxEvent) -> RequestContext | None:
        """Build a minimal RequestContext for event processing when possible."""
        match = _MEMORY_OWNER_PATTERN.match(event.uri)
        if not match:
            return None

        account_id = match.group("account")
        owner_type = match.group("owner_type")
        owner_id = match.group("owner_id")

        return RequestContext(
            account_id=account_id,
            user_id=owner_id if owner_type == "users" else "outbox-worker",
            agent_id=owner_id if owner_type == "agents" else "outbox-worker",
            session_id=f"outbox:{event.event_id}",
            trace_id=f"outbox:{event.event_id}",
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

        # Collect all pending events across accounts, grouped by type.
        # UPSERT_CONTEXT events are batch-embedded for efficiency.
        all_pending: list[tuple[str, object]] = []  # (node_uri, event)
        for account_id in account_ids:
            try:
                pending = outbox_store.list_pending(account_id)
            except Exception as e:
                logger.error("Failed to list pending events for %s: %s", account_id, e)
                continue
            all_pending.extend(pending)

        # Phase 1: Batch-embed all UPSERT_CONTEXT events
        upsert_batch: list[tuple[object, str, list[IndexRecord]]] = []  # (event, node_uri, records)
        other_events: list[tuple[object, str]] = []  # (event, node_uri)

        for node_uri, event in all_pending:
            if not outbox_store.try_acquire(event, node_uri, worker_id):
                stats["skipped"] += 1
                continue

            if event.event_type == "UPSERT_CONTEXT":
                try:
                    records = self._extract_records(event)
                    if records:
                        upsert_batch.append((event, node_uri, records))
                    else:
                        # Empty records — mark as succeeded immediately
                        outbox_store.mark_done(event, node_uri)
                        stats["processed"] += 1
                        stats["succeeded"] += 1
                except Exception as e:
                    logger.error("Failed to extract records for %s: %s", event.event_id, e)
                    failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                    self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)
            else:
                other_events.append((event, node_uri))

        # Batch embed all UPSERT_CONTEXT records in one call
        if upsert_batch:
            try:
                all_records = []
                boundaries = []
                for _, _, records in upsert_batch:
                    boundaries.append(len(all_records))
                    all_records.extend(records)

                all_texts = [r.text for r in all_records]
                all_embeddings = self._embedder.embed_texts(all_texts)

                # Distribute embeddings back and upsert per-event
                for idx, (event, node_uri, records) in enumerate(upsert_batch):
                    stats["processed"] += 1
                    start = boundaries[idx]
                    end = start + len(records)
                    embeddings = all_embeddings[start:end]
                    augmented = self._augment_with_embeddings(records, embeddings)
                    try:
                        self._vector_index.upsert(augmented)
                        result = WorkerResult(event_id=event.event_id, success=True, records_upserted=len(records))
                        self._handle_processing_result(event, node_uri, outbox_store, result, stats)
                        logger.info(
                            "Batch-embedded UPSERT_CONTEXT %s: %d records for %s",
                            event.event_id, len(records), event.uri,
                        )
                    except Exception as e:
                        logger.error("Upsert failed for %s: %s", event.event_id, e)
                        failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                        self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)
            except Exception as e:
                # Embedding failed for the whole batch — mark all as failed
                logger.error("Batch embedding failed: %s", e, exc_info=True)
                for event, node_uri, records in upsert_batch:
                    stats["processed"] += 1
                    failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                    self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)

        # Phase 2: Process non-UPSERT_CONTEXT events individually
        for event, node_uri in other_events:
            stats["processed"] += 1
            try:
                ctx = self._build_request_context(event)
                result = self.process_event(event, ctx)
                self._handle_processing_result(event, node_uri, outbox_store, result, stats)
            except Exception as e:
                logger.error("Exception processing event %s: %s", event.event_id, e, exc_info=True)
                failure_result = WorkerResult(event_id=event.event_id, success=False, error_message=str(e))
                self._handle_processing_result(event, node_uri, outbox_store, failure_result, stats)

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
