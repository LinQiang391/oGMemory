from __future__ import annotations

import threading
from unittest.mock import Mock

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def test_session_manager_lazy_initialization_is_thread_safe():
    service = MemoryService(config=OgMemConfig())
    barrier = threading.Barrier(16)
    managers = []

    def get_manager() -> None:
        barrier.wait()
        managers.append(service.get_session_manager())

    threads = [threading.Thread(target=get_manager) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(managers) == 16
    assert len({id(manager) for manager in managers}) == 1


def test_concurrent_after_turn_claims_each_message_once():
    service = MemoryService(
        config=OgMemConfig(
            after_turn_threshold=1,
            session_state_bridge_enabled=False,
        )
    )
    ctx = RequestContext(
        account_id="acct-test",
        user_id="user-test",
        agent_id="agent-test",
        session_id="session-threaded",
        trace_id="trace-threaded",
    )
    service._write_api = Mock()
    service._async_drain = Mock()
    mgr = service.get_session_manager()
    mgr.save_session_state = Mock(return_value=True)

    claim_barrier = threading.Barrier(2)
    buf = mgr.get_or_create("session-threaded", ctx=ctx)
    real_claim = buf.claim_incremental_extraction

    def synchronized_claim():
        claim_barrier.wait(timeout=3)
        return real_claim()

    buf.claim_incremental_extraction = synchronized_claim

    extract_started = threading.Event()
    release_extract = threading.Event()
    extraction_message_ids = []

    def fake_extract(**kwargs):
        extraction_message_ids.append(tuple(message["id"] for message in kwargs["messages"]))
        extract_started.set()
        release_extract.wait(timeout=3)
        return {
            "ok": True,
            "candidates_extracted": 1,
            "candidates_filtered": 0,
            "writes_completed": 1,
            "writes_skipped": 0,
            "writes_failed": 0,
            "plans": [],
        }

    service._chunk_and_extract = fake_extract
    mgr.commit_snapshot = Mock(
        return_value={"ok": True, "archived": True, "status": "completed"}
    )

    results = []

    def invoke_after_turn(index: int):
        results.append(service.after_turn({
            "sessionId": "session-threaded",
            "messages": [{
                "role": "user",
                "content": f"message-{index}-" + ("x" * 80),
            }],
            "_ctx": ctx,
        }))

    request_threads = [
        threading.Thread(target=invoke_after_turn, args=(index,))
        for index in range(2)
    ]
    for thread in request_threads:
        thread.start()
    for thread in request_threads:
        thread.join(timeout=3)

    assert extract_started.wait(timeout=3)
    assert len(extraction_message_ids) == 1
    assert len(extraction_message_ids[0]) == 2
    assert sorted(result["status"] for result in results) == ["processing", "processing"]

    release_extract.set()
    assert buf.extraction_done_event.wait(timeout=3)
    mgr.commit_snapshot.assert_called_once()
