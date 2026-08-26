"""Unit regressions for AGFS-specific Outbox worker behavior."""

from __future__ import annotations

from unittest.mock import Mock

from core.models import OutboxEvent, RequestContext
from index.outbox_worker import OutboxWorker, WorkerConfig
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="worker",
        agent_id="worker",
        session_id="session",
        trace_id="trace",
    )


def _delete_event() -> OutboxEvent:
    return OutboxEvent(
        event_id="delete-code",
        event_type="DELETE_CONTEXT",
        uri="ctx://acct/resources/chunk",
        payload={
            "ids_to_delete": ["code-l0", "code-l2"],
            "durable_delete": True,
        },
        status="PENDING",
    )


def test_delete_does_not_treat_bare_mock_fs_as_agfs_cleanup_capability():
    vector_index = InMemoryVectorIndex(dimension=8)
    fs = Mock()
    worker = OutboxWorker(vector_index, MockEmbedder(dimension=8), fs=fs)

    result = worker.process_event(_delete_event(), _ctx())

    assert result.success is True
    assert "is_tombstoned" not in fs.__dict__
    assert "delete_node_payload" not in fs.__dict__


def test_tombstoned_delete_retries_payload_cleanup_before_ack():
    event = _delete_event()

    class TombstonedFS:
        def __init__(self):
            self.tombstones = {event.uri}
            self.payload_present = True
            self.cleanup_attempts = 0

        def is_tombstoned(self, uri, ctx):
            return uri in self.tombstones

        def delete_node_payload(self, uri, ctx):
            self.cleanup_attempts += 1
            if self.cleanup_attempts == 1:
                raise RuntimeError("AGFS payload delete temporarily unavailable")
            self.payload_present = False

    fs = TombstonedFS()

    class RetryStore:
        supports_batch_claim = False

        def __init__(self):
            self.done = False
            self.retries = 0

        def list_pending(self, account_id):
            return [] if self.done else [(event.uri, event)]

        def try_acquire(self, pending_event, node_uri, worker_id):
            return True

        def increment_retry(self, failed_event, node_uri, next_retry_at):
            failed_event.retry_count += 1
            self.retries += 1

        def release(self, failed_event, node_uri):
            return None

        def mark_done(self, completed_event, node_uri):
            assert fs.payload_present is False
            fs.tombstones.discard(node_uri)
            self.done = True

        def move_to_dlq(self, failed_event, node_uri):
            raise AssertionError("transient cleanup must remain retryable")

    store = RetryStore()
    worker = OutboxWorker(
        InMemoryVectorIndex(dimension=8),
        MockEmbedder(dimension=8),
        fs=fs,
    )

    first = worker.run_once(store, ["acct"], worker_id="worker")
    second = worker.run_once(store, ["acct"], worker_id="worker")

    assert first["failed"] == 1
    assert second["succeeded"] == 1
    assert fs.cleanup_attempts == 2
    assert fs.tombstones == set()
    assert store.retries == 1


def test_durable_delete_never_moves_to_dlq_after_retry_limit():
    event = _delete_event()
    event.retry_count = 1

    class FailingCleanupFS:
        def is_tombstoned(self, uri, ctx):
            return True

        def delete_node_payload(self, uri, ctx):
            raise RuntimeError("AGFS payload delete remains unavailable")

    class RetryStore:
        supports_batch_claim = False

        def __init__(self):
            self.retries = 0

        def list_pending(self, account_id):
            return [(event.uri, event)]

        def try_acquire(self, pending_event, node_uri, worker_id):
            return True

        def increment_retry(self, failed_event, node_uri, next_retry_at):
            failed_event.retry_count += 1
            self.retries += 1

        def release(self, failed_event, node_uri):
            return None

        def move_to_dlq(self, failed_event, node_uri):
            raise AssertionError("durable DELETE must remain live")

    store = RetryStore()
    worker = OutboxWorker(
        InMemoryVectorIndex(dimension=8),
        MockEmbedder(dimension=8),
        fs=FailingCleanupFS(),
        config=WorkerConfig(max_retry=1, base_delay_ms=1, max_delay_ms=1),
    )

    stats = worker.run_once(store, ["acct"], worker_id="worker")

    assert stats["failed"] == 1
    assert stats["moved_to_dlq"] == 0
    assert store.retries == 1
    assert event.retry_count == 2


def test_non_batch_agfs_liveness_uses_node_uri_and_rolls_back_stale_upsert():
    event = OutboxEvent(
        event_id="upsert-code",
        event_type="UPSERT_CONTEXT",
        uri="ctx://acct/resources/chunk",
        payload={
            "records": [
                {
                    "id": "code-l0",
                    "uri": "ctx://acct/resources/chunk",
                    "level": 0,
                    "text": "parse config",
                    "filters": {
                        "account_id": "acct",
                        "owner_space": "resource:acct",
                    },
                    "metadata": {},
                }
            ]
        },
        status="PENDING",
    )
    liveness_calls: list[tuple[str, str]] = []

    class AGFSLikeStore:
        supports_batch_claim = False

        def is_event_current(self, event_id, node_uri):
            liveness_calls.append((event_id, node_uri))
            return len(liveness_calls) == 1

        def list_pending(self, account_id):
            return [(event.uri, event)]

        def try_acquire(self, pending_event, node_uri, worker_id):
            return True

        def mark_done(self, completed_event, node_uri):
            return None

    index = InMemoryVectorIndex(dimension=8)
    worker = OutboxWorker(index, MockEmbedder(dimension=8))

    stats = worker.run_once(AGFSLikeStore(), ["acct"])

    assert stats["succeeded"] == 1
    assert liveness_calls == [(event.event_id, event.uri)] * 2
    assert index.count() == 0


def test_resource_event_builds_account_scoped_worker_context():
    worker = OutboxWorker(
        InMemoryVectorIndex(dimension=8),
        MockEmbedder(dimension=8),
    )

    ctx = worker._build_request_context(_delete_event())

    assert ctx is not None
    assert ctx.account_id == "acct"
    assert ctx.user_id == ctx.agent_id == "outbox-worker"
