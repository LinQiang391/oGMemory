"""Unit tests for OutboxWorker.

Tests verify the happy path and retry/DLQ behavior.
Uses mock VectorIndex and mock Embedder for isolated testing.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import pytest

from core.models import OutboxEvent, IndexRecord
from index.outbox_worker import (
    OutboxWorker,
    WorkerResult,
    WorkerConfig,
    create_upsert_event,
)


# Mock VectorIndex for testing
class MockVectorIndex:
    """In-memory vector index for testing."""

    def __init__(self) -> None:
        self.records: dict[str, IndexRecord] = {}
        self.deleted_ids: list[str] = []

    def upsert(self, records: list[IndexRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            if id_ in self.records:
                del self.records[id_]
            self.deleted_ids.append(id_)

    def search(self, query):
        return []  # Not used in these tests


# Mock Embedder for testing
class MockEmbedder:
    """Mock embedder that returns fixed vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        # Return unique vectors per text
        return [[float(i) for i in range(self._dimension)] for _ in texts]


class FailingMockEmbedder(MockEmbedder):
    """Mock embedder that always fails."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding failed!")


class FailingMockVectorIndex(MockVectorIndex):
    """Mock vector index that always fails."""

    def upsert(self, records: list[IndexRecord]) -> None:
        raise RuntimeError("Vector index down!")


class TestCreateUpsertEvent:
    """Tests for create_upsert_event utility."""

    def test_creates_valid_event(self):
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        assert event.event_type == "UPSERT_CONTEXT"
        assert event.uri == "ctx://test/users/u1/memories/profile"
        assert event.status == "PENDING"
        assert event.retry_count == 0
        assert "records" in event.payload

    def test_serializes_records_correctly(self):
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
                metadata={"level": "abstract"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        payload_records = event.payload["records"]
        assert len(payload_records) == 1
        assert payload_records[0]["id"] == "rec1"
        assert payload_records[0]["level"] == 0
        assert payload_records[0]["filters"]["account_id"] == "test"


class TestOutboxWorkerProcessEvent:
    """Tests for OutboxWorker.process_event happy path."""

    def test_successful_processing(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)

        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 1
        assert result.event_id == event.event_id

    def test_multiple_records_upserted(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id=f"rec{i}",
                uri="ctx://test/users/u1/memories/profile",
                level=i,
                text=f"text{i}",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
            for i in range(3)
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 3
        assert len(vector_index.records) == 3

    def test_embedder_called_once_per_batch(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        worker.process_event(event)

        assert embedder.call_count == 1

    def test_event_with_no_records_succeeds(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Create event with empty records list
        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is True
        assert result.records_upserted == 0

    def test_records_stored_in_vector_index(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        worker.process_event(event)

        assert "rec1" in vector_index.records
        stored_record = vector_index.records["rec1"]
        assert stored_record.uri == "ctx://test/users/u1/memories/profile"
        # Embedding should be added to metadata
        assert "_embedding" in stored_record.metadata


class TestOutboxWorkerFailureCases:
    """Tests for OutboxWorker failure handling."""

    def test_embedding_failure_returns_error(self):
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is False
        assert "Embedding failed" in result.error_message
        assert result.records_upserted == 0

    def test_vector_index_failure_returns_error(self):
        vector_index = FailingMockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]

        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        assert result.success is False
        assert "Vector index down" in result.error_message


class TestOutboxWorkerValidation:
    """Tests for OutboxWorker input validation."""

    def test_missing_records_field_raises_error(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={},  # Missing "records"
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing 'records'" in result.error_message

    def test_records_not_list_raises_error(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": "not-a-list"},
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "not a list" in result.error_message

    def test_record_missing_account_id_raises_error(self):
        """CRITICAL: Missing account_id causes silent leakage - must reject."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={
                "records": [
                    {
                        "id": "rec1",
                        "uri": "ctx://test/users/u1/memories/profile",
                        "level": 0,
                        "text": "abstract",
                        "filters": {"owner_space": "user:u1"},  # Missing account_id
                    }
                ]
            },
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing account_id" in result.error_message

    def test_record_missing_owner_space_raises_error(self):
        """CRITICAL: Missing owner_space causes silent leakage - must reject."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={
                "records": [
                    {
                        "id": "rec1",
                        "uri": "ctx://test/users/u1/memories/profile",
                        "level": 0,
                        "text": "abstract",
                        "filters": {"account_id": "test"},  # Missing owner_space
                    }
                ]
            },
            status="PENDING",
        )

        result = worker.process_event(event)

        assert result.success is False
        assert "missing owner_space" in result.error_message


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_backoff_increases_exponentially(self):
        config = WorkerConfig(base_delay_ms=1000, max_delay_ms=60000)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        delay_0 = worker.calculate_backoff(0)
        delay_1 = worker.calculate_backoff(1)
        delay_2 = worker.calculate_backoff(2)

        assert delay_0.total_seconds() == 1.0
        assert delay_1.total_seconds() == 2.0
        assert delay_2.total_seconds() == 4.0

    def test_backoff_capped_at_max_delay(self):
        config = WorkerConfig(base_delay_ms=1000, max_delay_ms=3000)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        # With base=1000, max=3000, retry 3 would give 8000ms but should cap at 3000
        delay_3 = worker.calculate_backoff(3)

        assert delay_3.total_seconds() == 3.0


class TestShouldRetry:
    """Tests for retry logic."""

    def test_success_should_not_retry(self):
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
        )

        result = WorkerResult(
            event_id="test-event",
            success=True,
            records_upserted=0,
        )

        assert worker.should_retry(event, result) is False

    def test_failure_within_retry_limit_should_retry(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=2,
        )

        result = WorkerResult(
            event_id="test-event",
            success=False,
            records_upserted=0,
            error_message="Temporary error",
        )

        assert worker.should_retry(event, result) is True

    def test_failure_exceeding_retry_limit_should_not_retry(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=5,  # At max retry
        )

        result = WorkerResult(
            event_id="test-event",
            success=False,
            records_upserted=0,
            error_message="Permanent error",
        )

        assert worker.should_retry(event, result) is False

    def test_max_retry_event_returns_dlq_result(self):
        config = WorkerConfig(max_retry=5)
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config)

        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={"records": []},
            status="PENDING",
            retry_count=5,
        )

        result = worker.process_event(event)

        assert result.success is False
        assert result.moved_to_dlq is True
        assert "Max retry exceeded" in result.error_message

    def test_cross_tenant_account_id_mismatch_rejected(self):
        """Test that records with mismatched account_id are rejected (security fix C-3)."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Event URI has account "test" but record has account "other-account"
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://other-account/users/u1/memories/profile",  # Different account
                level=0,
                text="abstract",
                filters={"account_id": "other-account", "owner_space": "user:u1"},
            )
        ]

        # Event URI is for "test" account
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        result = worker.process_event(event)

        # Should fail due to account mismatch
        assert result.success is False
        assert "does not match" in result.error_message
        assert "cross-tenant injection" in result.error_message


class TestOutboxWorkerRunOnce:
    """Tests for OutboxWorker.run_once() polling entry point."""

    def test_run_once_happy_path(self):
        """Test run_once processes pending events successfully."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()

        # Create pending event
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"])

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 1
        assert stats["failed"] == 0
        assert stats["moved_to_dlq"] == 0

        # Verify event was marked done
        outbox_store.mark_done.assert_called_once_with(event, event.uri)

    def test_run_once_with_failures(self):
        """Test run_once handles failures correctly."""
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()  # Will fail
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()

        # Create pending event
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        event.retry_count = 0
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"])

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 1  # Failed but under max retry
        assert stats["moved_to_dlq"] == 0

        # Verify event retry was incremented
        outbox_store.increment_retry.assert_called_once()

    def test_run_once_moves_to_dlq_after_max_retry(self):
        """Test run_once moves events to DLQ after max retry."""
        vector_index = MockVectorIndex()
        embedder = FailingMockEmbedder()
        worker = OutboxWorker(vector_index, embedder, config=WorkerConfig(max_retry=3))

        # Mock outbox store
        outbox_store = Mock()

        # Create pending event at max retry
        records = [
            IndexRecord(
                id="rec1",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            )
        ]
        event = create_upsert_event("ctx://test/users/u1/memories/profile", records)
        event.retry_count = 3  # At max retry
        outbox_store.list_pending.return_value = [(event.uri, event)]

        # Run polling cycle
        stats = worker.run_once(outbox_store, ["test"])

        # Verify stats
        assert stats["processed"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 0
        assert stats["moved_to_dlq"] == 1

        # Verify event was moved to DLQ
        outbox_store.move_to_dlq.assert_called_once()

    def test_run_once_handles_multiple_accounts(self):
        """Test run_once processes events from multiple accounts."""
        vector_index = MockVectorIndex()
        embedder = MockEmbedder()
        worker = OutboxWorker(vector_index, embedder)

        # Mock outbox store
        outbox_store = Mock()

        # Create events for two accounts
        def make_side_effect(account_id):
            records = [
                IndexRecord(
                    id=f"rec_{account_id}",
                    uri=f"ctx://{account_id}/users/u1/memories/profile",
                    level=0,
                    text="abstract",
                    filters={"account_id": account_id, "owner_space": "user:u1"},
                )
            ]
            event = create_upsert_event(f"ctx://{account_id}/users/u1/memories/profile", records)
            return [(event.uri, event)]

        outbox_store.list_pending.side_effect = make_side_effect

        # Run polling cycle for multiple accounts
        stats = worker.run_once(outbox_store, ["account1", "account2"])

        # Verify both accounts were processed
        assert stats["processed"] == 2
        assert stats["succeeded"] == 2
        assert outbox_store.list_pending.call_count == 2
