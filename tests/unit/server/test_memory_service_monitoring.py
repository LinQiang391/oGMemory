import threading
from types import SimpleNamespace

from core.models import RequestContext
from server.memory_service import MemoryService


def test_monitoring_overview_maps_runtime_health_and_account_queue() -> None:
    service = MemoryService.__new__(MemoryService)
    service._cfg = SimpleNamespace(
        storage_backend="sql",
        retrieval_hybrid_enabled=True,
        vector_db_type="opengauss",
    )
    service._provider_cfg = SimpleNamespace(
        embedding_provider="openai",
        provider="openai",
        st_model="",
        openai_embedding_model="text-embedding-3-small",
    )
    service._outbox_thread = SimpleNamespace(is_alive=lambda: True)
    service.health = lambda: {
        "backend": "og-memory",
        "storage_backend": "sql",
        "sql": "connected",
        "status": "ok",
        "llm": "gpt-5.6-luna",
        "lexical": {"available": True},
        "lexical_backfill": {"queued": 0, "running": 0},
    }
    service._count_outbox_unfinished = lambda ctx: {
        "pending": 2,
        "processing": 1,
        "total": 3,
    }
    service._get_shared_vector_index = lambda: SimpleNamespace(
        count=lambda filters: 7
    )
    context = RequestContext("acct-a", "alice", "", "http", "trace")

    overview = service.get_monitoring_overview(context)

    assert overview["account_id"] == "acct-a"
    assert overview["status"] == "ok"
    assert overview["healthy_count"] == overview["total_count"] == 6
    queue = next(item for item in overview["components"] if item["id"] == "queue")
    assert queue["details"] == {
        "账户": "acct-a",
        "待处理": 2,
        "处理中": 1,
        "总计": 3,
        "后台 Worker": "运行中",
        "错误": "",
    }
    index = next(item for item in overview["components"] if item["id"] == "index")
    assert index["details"]["向量数量"] == 7
    assert "get_vector_index" not in str(overview)


def test_monitoring_overview_marks_unreachable_agfs_storage_as_error() -> None:
    service = MemoryService.__new__(MemoryService)
    service._cfg = SimpleNamespace(
        storage_backend="agfs",
        retrieval_hybrid_enabled=False,
        vector_db_type="chroma",
    )
    service._provider_cfg = SimpleNamespace(
        embedding_provider="mock",
        provider="mock",
        st_model="",
        openai_embedding_model="",
    )
    service._outbox_thread = None
    service.health = lambda: {
        "backend": "og-memory",
        "storage_backend": "agfs",
        "agfs": True,
        "status": "error",
        "error": "connection refused",
    }
    service._count_outbox_unfinished = lambda ctx: {
        "pending": 0,
        "processing": 0,
        "total": 0,
    }

    overview = service.get_monitoring_overview(
        RequestContext("acct-a", "alice", "", "http", "trace")
    )

    storage = next(item for item in overview["components"] if item["id"] == "storage")
    assert storage["status"] == "error"


def test_retrieval_monitoring_respects_visible_owner_spaces() -> None:
    service = MemoryService.__new__(MemoryService)
    service._retrieval_monitoring = {}
    service._retrieval_monitoring_lock = threading.Lock()
    alice = RequestContext("acct-a", "alice", "", "http", "trace")
    bob = RequestContext("acct-a", "bob", "", "http", "trace")

    service.record_retrieval_metric(
        alice, request_id="alice-request", duration_ms=20, scores=[0.8, 0.4],
        categories=["event", "event"], rerank_used=True,
    )
    service.record_retrieval_metric(
        bob, request_id="bob-request", duration_ms=40, scores=[], categories=[], rerank_used=False,
    )

    metrics = service.get_retrieval_monitoring_metrics("acct-a", ("user:alice",))

    assert metrics["queries"] == 1
    assert metrics["results"] == 2
    assert metrics["request_id"] == "alice-request"
    assert metrics["by_type"] == {"event": 2}
