from __future__ import annotations

import json
import os
import queue
import stat
import threading
import time

import pytest

from core.models import RequestContext, RetrievalConfig, SearchMemoryResult
from retrieval.pipeline import RetrievalPipeline
from retrieval.trace import (
    RetrievalTrace,
    StageTrace,
    TraceTimer,
    _NoTraceRedirect,
    _write_retrieval_trace_http,
    emit_retrieval_trace,
    trace_capable,
)
from service.api import ReadAPI


def test_stage_trace_serializes_metrics_and_children():
    child = StageTrace(stage="embed_query", elapsed_ms=1.234, input_count=1, output_count=1)
    child.metrics["embedding_dim"] = 8
    parent = StageTrace(stage="seed_retrieval", elapsed_ms=3.456, children=[child])
    parent.metrics["mode"] = "QUICK"

    assert parent.to_dict() == {
        "stage": "seed_retrieval",
        "elapsed_ms": 3.46,
        "input_count": 0,
        "output_count": 0,
        "metrics": {"mode": "QUICK"},
        "children": [
            {
                "stage": "embed_query",
                "elapsed_ms": 1.23,
                "input_count": 1,
                "output_count": 1,
                "metrics": {"embedding_dim": 8},
                "children": [],
                "warnings": [],
                "error": None,
            }
        ],
        "warnings": [],
        "error": None,
    }


def test_failed_nested_timer_is_attached_to_parent():
    with pytest.raises(RuntimeError):
        with TraceTimer("parent") as parent:
            with TraceTimer("child"):
                raise RuntimeError("boom")

    assert [child.stage for child in parent.children] == ["child"]
    assert parent.children[0].error == "RuntimeError"


def test_real_seed_retriever_records_production_substages(ctx):
    from core.models import TypedQuery
    from retrieval.seed_retriever import SeedRetriever

    class Embedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2]]

    class VectorIndex:
        def search_by_vector(self, *, query_vector, filters, top_k):
            return []

    parent = StageTrace(stage="seed_retrieval")
    retriever = SeedRetriever(VectorIndex(), Embedder(), RetrievalConfig())
    query = TypedQuery(text="question", context_type="MEMORY", categories=[], top_k=3)

    retriever.search(query, ctx, trace_stage=parent)

    names = [child.stage for child in parent.children]
    assert names == [
        "embed_query",
        "build_root_uris",
        "build_vector_filters",
        "vector_search",
        "split_starting_points",
    ]
    assert parent.children[0].metrics["embedding_dim"] == 2
    assert parent.children[3].metrics["candidate_top_k"] > 0


def test_http_trace_adds_auth_and_tenant_headers(monkeypatch):
    captured = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("OG_AUTH_API_KEY", "must-not-leak")
    monkeypatch.setenv("OGMEM_TRACE_HTTP_BEARER_TOKEN", "trace-secret")
    monkeypatch.setattr("retrieval.trace._trace_http_opener.open", fake_urlopen)

    assert _write_retrieval_trace_http(
        {"account_id": "acct", "user_id": "user"}, "https://trace.test/events"
    )
    assert "X-api-key" not in captured["headers"]
    assert captured["headers"]["Authorization"] == "Bearer trace-secret"
    assert "X-account-id" not in captured["headers"]
    assert "X-user-id" not in captured["headers"]
    assert captured["body"]["account_id"].startswith("sha256:")
    assert captured["body"]["user_id"].startswith("sha256:")
    assert captured["body"]["account_id"] != "acct"

    monkeypatch.setenv("OGMEM_TRACE_INCLUDE_IDENTIFIERS", "true")
    assert _write_retrieval_trace_http(
        {"account_id": "acct", "user_id": "user"}, "https://trace.test/events"
    )
    assert captured["headers"]["X-account-id"] == "acct"
    assert captured["headers"]["X-user-id"] == "user"
    assert captured["body"]["account_id"] == "acct"


def test_http_trace_redirects_are_disabled():
    handler = _NoTraceRedirect()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://other.test") is None


def test_remote_plaintext_http_sink_is_rejected(monkeypatch):
    monkeypatch.setenv("OGMEM_TRACE_SINK", "http://trace.example/events")

    assert not emit_retrieval_trace({"request_id": "insecure"})


def test_http_emit_runs_through_background_worker(monkeypatch):
    import retrieval.trace as trace_module

    delivered = threading.Event()
    monkeypatch.setenv("OGMEM_TRACE_SINK", "https://trace.test/events")
    monkeypatch.setattr(
        trace_module,
        "_write_retrieval_trace_http",
        lambda payload, url: delivered.set() or True,
    )
    monkeypatch.setattr(trace_module, "_http_trace_queue", None)
    monkeypatch.setattr(trace_module, "_http_trace_worker_started", False)
    monkeypatch.setattr(trace_module, "_http_sink_states", {})

    assert emit_retrieval_trace({"request_id": "async-request"})
    assert delivered.wait(timeout=1)

    worker_queue = trace_module._http_trace_queue
    assert worker_queue is not None
    worker_queue.put(None)
    worker_queue.join()


def test_http_queue_full_drops_new_event(monkeypatch):
    import retrieval.trace as trace_module

    full_queue = queue.Queue(maxsize=1)
    full_queue.put(({"request_id": "existing"}, "https://trace.test", time.perf_counter()))
    monkeypatch.setattr(trace_module, "_http_trace_queue", full_queue)
    monkeypatch.setattr(trace_module, "_http_trace_worker_started", True)

    assert not trace_module._enqueue_retrieval_trace_http(
        {"request_id": "dropped"}, "https://trace.test"
    )


def test_http_worker_retries_and_opens_per_url_circuit(monkeypatch):
    import retrieval.trace as trace_module

    work_queue = queue.Queue()
    attempts = []
    monkeypatch.setattr(trace_module, "_http_trace_queue", work_queue)
    monkeypatch.setattr(trace_module, "_http_sink_states", {})
    monkeypatch.setattr(trace_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        trace_module,
        "_write_retrieval_trace_http",
        lambda payload, url: attempts.append(url) or False,
    )
    url = "https://failing.test/events"
    now = time.perf_counter()
    for index in range(3):
        work_queue.put(({"request_id": str(index)}, url, now))
    work_queue.put(None)

    worker = threading.Thread(target=trace_module._http_trace_worker)
    worker.start()
    work_queue.join()
    worker.join(timeout=1)

    assert len(attempts) == 9
    assert trace_module._http_sink_states[url][1] > 0


def test_pipeline_trace_contains_seed_substages(ctx):
    class Planner:
        def plan(self, query, ctx, **kwargs):
            from core.models import TypedQuery

            return [TypedQuery(text=query, context_type="MEMORY", categories=[], top_k=3)]

    class Seed:
        @trace_capable
        def search(self, typed_query, ctx, *, mode, trace_stage=None):
            from core.models import SeedResult

            if trace_stage is not None:
                trace_stage.add_child(
                    StageTrace(stage="embed_query", elapsed_ms=1.0, input_count=1, output_count=1)
                )
                trace_stage.add_child(
                    StageTrace(stage="vector_search", elapsed_ms=2.0, input_count=1, output_count=0)
                )
            return SeedResult(query_vector=[0.1])

    class Assembly:
        def assemble(self, typed_query, leaf_hits, seed_result, **kwargs):
            return []

    pipe = RetrievalPipeline(
        Planner(),
        Seed(),
        None,
        Assembly(),
        RetrievalConfig(),
    )

    result = pipe.run("where did they meet?", ctx, top_k=3)
    seed_stage = next(stage for stage in result.trace.stages if stage.stage == "seed_retrieval")

    assert [child.stage for child in seed_stage.children] == ["embed_query", "vector_search"]


def test_pipeline_trace_keeps_failed_stage(ctx):
    class Planner:
        def plan(self, query, ctx, **kwargs):
            from core.models import TypedQuery

            return [TypedQuery(text=query, context_type="MEMORY", categories=[], top_k=3)]

    class Seed:
        def search(self, typed_query, ctx, *, mode, trace_stage=None):
            time.sleep(0.001)
            raise RuntimeError("provider unavailable")

    trace = RetrievalTrace()
    pipeline = RetrievalPipeline(Planner(), Seed(), None, object(), RetrievalConfig())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        pipeline.run("question", ctx, top_k=3, trace=trace)

    assert [stage.stage for stage in trace.stages] == ["planner", "seed_retrieval"]
    assert trace.stages[-1].error == "RuntimeError"
    assert trace.total_ms > 0


def test_read_api_writes_trace_jsonl_when_enabled(tmp_path, monkeypatch):
    trace_path = tmp_path / "retrieval_trace.jsonl"
    monkeypatch.setenv("OGMEM_RETRIEVAL_TRACE", "1")
    monkeypatch.setenv("OGMEM_RETRIEVAL_TRACE_JSONL", str(trace_path))

    class Pipeline:
        def run(self, *args, **kwargs):
            trace = StageTrace(stage="pipeline_run", elapsed_ms=1.0)
            from retrieval.trace import RetrievalTrace

            rt = RetrievalTrace()
            rt.add_stage(trace)
            rt.total_ms = 1.0
            return SearchMemoryResult(request_id=rt.request_id, query=args[0], hits=[], trace=rt)

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="t")

    result = api.search_memory("question", ctx, top_k=1, include_debug=True)

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["request_id"] == result.request_id
    assert rows[0]["query_length"] == len("question")
    assert rows[0]["hit_count"] == 0
    assert rows[0]["trace"]["stages"][0]["stage"] == "read_api"


def test_configured_jsonl_sink_enables_and_writes_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "configured-trace.jsonl"
    monkeypatch.delenv("OGMEM_RETRIEVAL_TRACE", raising=False)
    monkeypatch.delenv("OGMEM_RETRIEVAL_TRACE_JSONL", raising=False)
    monkeypatch.setenv("OGMEM_TRACE_SINK", f"jsonl:{trace_path}")

    assert emit_retrieval_trace({"request_id": "req-config"}) is True

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"request_id": "req-config"}]


def test_jsonl_sink_hashes_identifiers_and_uses_private_file(tmp_path, monkeypatch):
    trace_path = tmp_path / "private" / "trace.jsonl"
    monkeypatch.setenv("OGMEM_TRACE_SINK", f"jsonl:{trace_path}")

    assert emit_retrieval_trace(
        {"request_id": "req", "account_id": "acct", "user_id": "user"}
    )

    row = json.loads(trace_path.read_text(encoding="utf-8"))
    assert row["account_id"].startswith("sha256:")
    assert row["user_id"].startswith("sha256:")
    if os.name != "nt":
        assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(trace_path.parent.stat().st_mode) == 0o700


def test_trace_sink_failure_does_not_fail_successful_search(tmp_path, monkeypatch):
    monkeypatch.setenv("OGMEM_TRACE_SINK", f"jsonl:{tmp_path}")

    class Pipeline:
        @trace_capable
        def run(self, *args, trace=None, **kwargs):
            return SearchMemoryResult(
                request_id=trace.request_id,
                query=args[0],
                hits=[],
                trace=trace,
            )

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="t")

    result = api.search_memory("question", ctx, top_k=1)

    assert result.hits == []


def test_malformed_http_sink_does_not_fail_successful_search(monkeypatch):
    monkeypatch.setenv("OGMEM_TRACE_SINK", "https://[::1")

    class Pipeline:
        @trace_capable
        def run(self, *args, trace=None, **kwargs):
            return SearchMemoryResult(
                request_id=trace.request_id,
                query=args[0],
                hits=[],
                trace=trace,
            )

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="t")

    result = api.search_memory("question", ctx, top_k=1)

    assert result.hits == []


def test_malformed_http_sink_preserves_pipeline_error(monkeypatch):
    monkeypatch.setenv("OGMEM_TRACE_SINK", "https://[::1")

    class Pipeline:
        @trace_capable
        def run(self, *args, trace=None, **kwargs):
            raise RuntimeError("provider unavailable")

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="t")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        api.search_memory("question", ctx, top_k=1)


def test_read_api_persists_failed_request_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "failed-trace.jsonl"
    monkeypatch.setenv("OGMEM_TRACE_SINK", f"jsonl:{trace_path}")

    class Pipeline:
        @trace_capable
        def run(self, *args, trace=None, **kwargs):
            raise RuntimeError("provider unavailable")

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(
        account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="trace-1"
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        api.search_memory("question", ctx, top_k=1)

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["trace_id"].startswith("sha256:")
    read_stage = rows[0]["trace"]["stages"][0]
    pipeline_stage = next(
        child for child in read_stage["children"] if child["stage"] == "pipeline_run"
    )
    assert pipeline_stage["error"] == "RuntimeError"


def test_read_api_nests_pipeline_stages_without_double_counting(tmp_path, monkeypatch):
    trace_path = tmp_path / "nested-trace.jsonl"
    monkeypatch.setenv("OGMEM_TRACE_SINK", f"jsonl:{trace_path}")

    class Pipeline:
        @trace_capable
        def run(self, *args, trace=None, **kwargs):
            trace.add_stage(StageTrace(stage="planner", elapsed_ms=1.0))
            trace.add_stage(StageTrace(stage="seed_retrieval", elapsed_ms=2.0))
            trace.total_ms = 3.0
            return SearchMemoryResult(
                request_id=trace.request_id,
                query=args[0],
                hits=[],
                trace=trace,
            )

    api = ReadAPI(Pipeline(), config=RetrievalConfig())
    ctx = RequestContext(
        account_id="acct", user_id="u", agent_id="a", session_id="s", trace_id="trace-2"
    )

    result = api.search_memory("question", ctx, top_k=1, include_debug=True)

    assert [stage.stage for stage in result.trace.stages] == ["read_api"]
    pipeline_stage = next(
        child for child in result.trace.stages[0].children if child.stage == "pipeline_run"
    )
    assert [child.stage for child in pipeline_stage.children] == ["planner", "seed_retrieval"]
    assert result.request_id == result.trace.request_id


def test_read_api_preserves_legacy_adapter_debug_trace(ctx):
    class LegacyPipeline:
        def run(self, query, ctx, **kwargs):
            return SearchMemoryResult(
                request_id="legacy-request",
                query=query,
                hits=[],
                trace={"legacy": "debug"},
            )

    api = ReadAPI(LegacyPipeline(), config=RetrievalConfig())

    result = api.search_memory("question", ctx, top_k=1, include_debug=True)

    assert result.trace == {"legacy": "debug"}
    assert result.request_id == "legacy-request"


def test_pipeline_trace_records_skipped_expansion(ctx):
    class Planner:
        def plan(self, query, ctx, **kwargs):
            from core.models import TypedQuery

            return [TypedQuery(text=query, context_type="MEMORY", categories=[], top_k=3)]

    class Seed:
        def search(self, typed_query, ctx, *, mode, trace_stage=None):
            from core.models import SeedResult

            return SeedResult(query_vector=[0.1])

    class Assembly:
        def assemble(self, typed_query, leaf_hits, seed_result, **kwargs):
            return []

    result = RetrievalPipeline(Planner(), Seed(), None, Assembly(), RetrievalConfig()).run(
        "question", ctx, top_k=3
    )

    expansion = next(stage for stage in result.trace.stages if stage.stage == "expansion")
    assert expansion.metrics["skipped_reason"] == "no_candidates"
    assert expansion.input_count == 0
    assert expansion.output_count == 0
    assert expansion.children == []


def test_pipeline_trace_records_l2_fallback_stage(ctx):
    from core.models import SeedHit

    class Planner:
        def plan(self, query, ctx, **kwargs):
            from core.models import TypedQuery

            return [TypedQuery(text=query, context_type="MEMORY", categories=[], top_k=3)]

    class Seed:
        def search(self, typed_query, ctx, *, mode, trace_stage=None):
            from core.models import SeedResult

            return SeedResult(
                initial_candidates=[SeedHit(uri="ctx://acct/l2", score=0.9, level=2)],
                query_vector=[0.1],
            )

    class Assembly:
        def assemble(self, typed_query, leaf_hits, seed_result, **kwargs):
            return []

    result = RetrievalPipeline(Planner(), Seed(), None, Assembly(), RetrievalConfig()).run(
        "question", ctx, top_k=3
    )

    expansion = next(stage for stage in result.trace.stages if stage.stage == "expansion")
    assert expansion.metrics["path"] == "l2_fallback"
    assert [child.stage for child in expansion.children] == ["fallback_l2_from_seed"]
    assert expansion.output_count == 1
