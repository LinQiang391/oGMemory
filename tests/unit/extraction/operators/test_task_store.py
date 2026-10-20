"""Unit tests for InMemoryTaskStore."""

from __future__ import annotations

import threading
import time

from extraction.operators.task_store import InMemoryTaskStore


class TestInMemoryTaskStore:
    def test_submit_creates_processing_record(self):
        store = InMemoryTaskStore()
        rec = store.submit(raw_node_id="r1", operator_service="ops")
        assert rec.task_id
        assert rec.status == "processing"
        assert rec.raw_node_id == "r1"
        assert store.get(rec.task_id) is rec

    def test_mark_running(self):
        store = InMemoryTaskStore()
        rec = store.submit(raw_node_id="r1", operator_service="ops")
        assert rec.started_at is None
        updated = store.mark_running(rec.task_id)
        assert updated is not None
        assert updated.started_at is not None

    def test_mark_completed(self):
        store = InMemoryTaskStore()
        rec = store.submit(raw_node_id="r1", operator_service="ops")
        result = {"steps": [{"operator": "cap", "ok": True}]}
        updated = store.mark_completed(rec.task_id, result)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.result == result
        assert updated.finished_at is not None

    def test_mark_failed(self):
        store = InMemoryTaskStore()
        rec = store.submit(raw_node_id="r1", operator_service="ops")
        updated = store.mark_failed(rec.task_id, "boom")
        assert updated is not None
        assert updated.status == "failed"
        assert updated.error == "boom"

    def test_get_unknown_returns_none(self):
        store = InMemoryTaskStore()
        assert store.get("nope") is None

    def test_updates_on_unknown_return_none(self):
        store = InMemoryTaskStore()
        assert store.mark_running("nope") is None
        assert store.mark_completed("nope", {}) is None
        assert store.mark_failed("nope", "x") is None

    def test_prune_removes_old_terminal_tasks(self):
        store = InMemoryTaskStore()
        rec = store.submit(raw_node_id="r1", operator_service="ops")
        # Backdate finished_at to ancient
        rec.finished_at = "2000-01-01T00:00:00+00:00"
        rec.status = "completed"

        rec2 = store.submit(raw_node_id="r2", operator_service="ops")
        store.mark_completed(rec2.task_id, {})

        removed = store.prune(older_than_seconds=60)
        assert removed == 1
        assert store.get(rec.task_id) is None
        assert store.get(rec2.task_id) is rec2

    def test_concurrent_access(self):
        store = InMemoryTaskStore()

        def submit_one(i):
            store.submit(raw_node_id=f"r{i}", operator_service="ops")

        threads = [threading.Thread(target=submit_one, args=(i,)) for i in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        # All should be reachable
        for rec in store._tasks.values():
            assert store.get(rec.task_id) is rec
