"""Exact neighboring evidence, independently budgeted and scope-safe."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.clip_lookup import clip_ordinal, validate_clip_lookup
from core.graph_models import create_extracted_node
from core.models import IndexRecord
from m3.context_retrieval import neighbor_plan, validate_context_options
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, SearchConfig
from m3.search import EpisodicSearcher, format_evidence_block
from providers.llm import MockLLM
from service.graph_service import ClipScope, GraphService
from tests.unit.m3.test_retrieval_core import Embedder, ctx, graph_search, hit, record
from tests.unit.m3.test_transcript_recall import sql_index


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "bad", float("nan")])
def test_invalid_ordinals_are_not_guessed(value):
    assert clip_ordinal(value) is None
    assert neighbor_plan([("v", value, "n")], radius=2) == {}


def test_neighborhood_plan_deduplicates_prioritizes_distance_and_excludes_main():
    plan = neighbor_plan([("v", 2, "a"), ("v", 4, "b")], radius=2, before_clip=4)
    assert list(plan) == [("v", 1), ("v", 3), ("v", 0)]
    assert plan[("v", 3)] == (1, 0, "a")


@pytest.mark.parametrize("radius,budget", [(-1, 0), (True, 1), (1, -1), (1, True), (0, 1)])
def test_invalid_context_configuration(radius, budget):
    with pytest.raises(ValueError):
        validate_context_options(radius, budget)


def make_searcher():
    # Many high-similarity distractors previously hid the true neighbors from
    # the video-wide vector top-8 lookup used by _expand_neighbors.
    records = [record("main", 10, "She points to the door.", [1., 0.], start=100)]
    records += [record(f"d{i}", i, "irrelevant", [.99, .01], start=i * 10) for i in range(8)]
    records += [record("before", 9, "She says: let's go to the library.", [0., 1.], start=90),
                record("after", 11, "She picks up her bag.", [-1., 0.], start=110),
                record("foreign", 9, "secret", [1., 0.], video="foreign", start=90)]
    embedder = Mock(wraps=Embedder())
    return EpisodicSearcher.from_records(records, embedder=embedder, hybrid_weight=0), embedder


def test_low_similarity_neighbors_reach_prompt_without_extra_embeddings():
    s, embedder = make_searcher()
    vector = s._index.search_by_vector = Mock(wraps=s._index.search_by_vector)
    lookup = s._index.get_records_by_clips = Mock(wraps=s._index.get_records_by_clips)
    result = s.search("where are they going", video_id="v", top_k=1, candidate_k=1,
                      include_neighbors=1, context_k=2, before_sec=120)
    assert [ev.source_extracted_node_id for ev in result] == ["main", "before", "after"]
    assert result[1].metadata["context_for"] == "main"
    assert result[1].metadata["evidence_role"] == "context"
    assert result[1].metadata["retrieval_sources"] == ["exact_neighbor"]
    assert "evidence_id=main" in format_evidence_block(result)
    assert "library" in format_evidence_block(result)
    assert "role=context context_for=main" in format_evidence_block(result)
    assert vector.call_count == embedder.embed_texts.call_count == lookup.call_count == 1
    assert set(lookup.call_args.kwargs["clip_ids"]) == {9, 11}


@pytest.mark.parametrize("bounds", [{"before_sec": 110}, {"before_clip": 10}])
def test_neighbor_future_is_excluded_without_losing_past(bounds):
    s, _ = make_searcher()
    result = s.search("where", video_id="v", top_k=1, candidate_k=1,
                      include_neighbors=1, context_k=2, **bounds)
    assert [ev.source_extracted_node_id for ev in result] == ["main", "before"]


def test_separate_budget_can_be_disabled_and_missing_neighbors_not_fabricated():
    s, _ = make_searcher()
    assert len(s.search("where", video_id="v", top_k=1, include_neighbors=1)) == 1
    result = s.search("where", video_id="v", top_k=1, candidate_k=1, include_neighbors=1, context_k=1)
    assert len(result) == 2
    assert len(s.search("where", video_id="v", top_k=1, include_neighbors=0)) == 1
    s._index.get_records_by_clips = Mock(return_value=[])
    assert len(s.search("where", video_id="v", top_k=1, include_neighbors=1, context_k=4)) == 1


def test_neighbor_request_batches_and_deduplicates_overlap():
    s = EpisodicSearcher.from_records([
        record("a", 1, "main", [1., 0.]), record("b", 3, "main", [1., 0.]),
        record("c", 2, "context", [0., 1.]),
    ], embedder=Embedder(), hybrid_weight=0)
    lookup = s._index.get_records_by_clips = Mock(wraps=s._index.get_records_by_clips)
    result = s.search("main", video_id="v", top_k=2, include_neighbors=1, context_k=5)
    assert [ev.source_extracted_node_id for ev in result] == ["a", "b", "c"]
    assert lookup.call_count == 1
    assert result[-1].metadata["context_for"] == "a"


def test_memory_exact_lookup_account_video_type_and_time():
    s, _ = make_searcher()
    filters = {"account_id": "m3bench", "video_id": "v", "before_sec": 110,
               "extractor_type": "episodic_memory_generator"}
    result = s._index.get_records_by_clips(filters, [9, 11, 9])
    assert [r.metadata["extracted_node_id"] for r in result] == ["before"]
    assert s._index.get_records_by_clips({**filters, "account_id": "other"}, [9]) == []
    assert s._index.get_records_by_clips({**filters, "extractor_type": "other"}, [9]) == []
    assert s._index.get_records_by_clips(filters, []) == []
    with pytest.raises(ValueError):
        validate_clip_lookup({}, [1])


def test_sql_exact_batch_query_does_not_use_vector_and_returns_full_records():
    index, conn, cursor = sql_index()
    cursor.fetchall.return_value = [{"id": "r", "uri": "u", "level": 2, "text": "complete dialogue",
                                     "filters": {"account_id": "a"}, "metadata": {"clip_id": 2}}]
    result = index.get_records_by_clips({"account_id": "a", "video_id": "v", "before_sec": 25,
                                        "before_clip": 2, "extractor_type": "episodic_memory_generator"}, [2, 2])
    sql, params = cursor.execute.call_args.args
    assert "before_clip = ANY(%(clip_ids)s)" in sql
    assert "<=>" not in sql and "embedding" not in sql
    assert params["clip_ids"] == [2]
    assert params["f_account_id"] == "a" and params["f_video_id"] == "v"
    assert params["f_before_sec"] == 25 and params["f_before_clip"] == 2
    assert result[0].text == "complete dialogue"
    conn.rollback.assert_called_once()
    index._return_connection.assert_called_once_with(conn)


def test_sql_empty_and_invalid_requests_do_not_connect_and_errors_cleanup():
    index, conn, cursor = sql_index()
    filters = {"account_id": "a", "video_id": "v"}
    assert index.get_records_by_clips(filters, []) == []
    with pytest.raises(ValueError):
        index.get_records_by_clips(filters, [True])
    with pytest.raises(ValueError):
        index.get_records_by_clips({"account_id": "a"}, [1])
    index._get_connection.assert_not_called()
    cursor.execute.side_effect = RuntimeError("db failed")
    with pytest.raises(RuntimeError):
        index.get_records_by_clips(filters, [1])
    conn.rollback.assert_called_once()
    index._return_connection.assert_called_once_with(conn)


@pytest.mark.parametrize("change", [{}, {"source_video_id": "other"}, {"source_clip_id": 2},
                                     {"source_start_sec": 30}])
def test_graph_exact_lookup_rechecks_current_node(change):
    node = create_extracted_node(source_raw_id="raw", account_id="m3bench",
                                 extractor_type="episodic_memory_generator", extraction_schema="m3",
                                 extracted_content={"source_video_id": "v", "source_clip_id": 1,
                                                    "source_start_sec": 10, "source_duration_sec": 10, **change})
    store = Mock()
    store.get_node.return_value = node
    service = GraphService(store, Mock())
    service._vector_index.get_records_by_clips.return_value = [IndexRecord(
        id="r", uri="u", level=2, text="", filters={}, metadata={"extracted_node_id": node.node_id})]
    result = service.get_evidence_by_clips([1], ctx(), scope=ClipScope("v", before_sec=20))
    assert len(result) == (0 if change else 1)


def test_react_context_budget_cache_and_seen_evidence(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    graph = Mock()
    graph.search_by_embedding.return_value = [replace(hit("main", 1.), clip_id=2)]
    graph.get_evidence_by_clips.return_value = [replace(hit("before", 0.), clip_id=1),
                                              replace(hit("after", 0.), clip_id=3)]
    config = SearchConfig(video_id="v", context_k=1, include_neighbors=1)
    result = graph_search(graph, config, top_k=1)
    assert [h["node_id"] for h in result["hits"]] == ["main"]
    assert [h["node_id"] for h in result["context_hits"]] == ["before"]
    assert result["context_hits"][0]["context_for"] == "main"
    assert result["context_hits"][0]["source_scores"] == {}
    adapter = OGMemoryControlAdapter(MockLLM(), graph, Embedder(), context_k=1)
    task = M3TaskContext(question="room", video_id="v")
    adapter._build_messages(ctx(), task)
    adapter._execute_tool("Search", {"query": "room", "top_k": 1}, ctx())
    calls = graph.get_evidence_by_clips.call_count
    adapter._execute_tool("Search", {"query": "room", "top_k": 1}, ctx())
    assert graph.get_evidence_by_clips.call_count == calls
    assert task.search_history[0]["node_ids"] == ["before", "main"]
    assert task.search_history[0]["context_node_ids"] == ["before"]
    assert task.search_history[0]["new_evidence_count"] == 2
    assert task.search_history[1]["new_evidence_count"] == 0
    assert task.search_history[1]["retrieval_branches"] == []
    assert set(next(t for t in adapter._get_tools() if t["name"] == "Search")["input_schema"]["properties"]) == {"query", "top_k"}


def test_react_context_off_never_reads_neighbors(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    graph = Mock()
    graph.search_by_embedding.return_value = [hit("main", 1.)]
    assert graph_search(graph, SearchConfig(video_id="v"))["context_hits"] == []
    graph.get_evidence_by_clips.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["qa", "react", "eval", "search"])
def test_context_cli(entrypoint):
    from m3 import eval as evaluation
    from m3 import qa, search
    parser, argv = {"qa": (qa.parse_args, ["--annotations", "x.json"]),
                    "react": (qa.parse_react_qa_args, ["--annotations", "x.json"]),
                    "eval": (evaluation.parse_args, ["--annotations", "x.json"]),
                    "search": (search.parse_args, ["--query", "room"])}[entrypoint]
    assert parser(argv).context_k == 0
    args = parser([*argv, "--context-k", "3", "--include-neighbors", "2"])
    assert (args.context_k, args.include_neighbors) == (3, 2)
    for extra in (["--context-k", "-1"], ["--include-neighbors", "-1"],
                  ["--context-k", "1", "--include-neighbors", "0"]):
        with pytest.raises(SystemExit):
            parser([*argv, *extra])


def test_benchmark_pipeline_forwards_context_and_retrieval_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from m3.pipeline.spec import StageSpec
    from m3.pipeline.stages import qa_stage

    runner = Mock(return_value=0)
    monkeypatch.setattr("m3.qa.run_react_qa", runner)
    context = SimpleNamespace(work_dir=tmp_path, graph_store="memory", config_path=tmp_path / "config.yaml",
                              connection_string=None, graph=Mock(), artifact=lambda name: tmp_path / name)
    stage = StageSpec(id="qa", type="qa", params={"context_k": 3, "include_neighbors": 2,
                                                "candidate_k": 20, "transcript_top_k": 10,
                                                "transcript_fusion": "weighted"})
    qa_stage(context, stage)
    args = runner.call_args.args[0]
    assert (args.context_k, args.include_neighbors, args.candidate_k) == (3, 2, 20)
    assert (args.transcript_top_k, args.transcript_fusion) == (10, "weighted")
    assert runner.call_args.kwargs["graph"] is context.graph


def test_unsupported_exact_backend_fails_explicitly():
    service = GraphService(Mock())
    with pytest.raises(NotImplementedError, match="clip-capable"):
        service.get_evidence_by_clips([1], ctx(), scope=ClipScope("v"))
    s, _ = make_searcher()
    s._index.get_records_by_clips = None
    with pytest.raises(NotImplementedError, match="clip-capable"):
        s.search("where", video_id="v", top_k=1, include_neighbors=1, context_k=2)


@pytest.mark.parametrize("fault", ["account", "extractor", "deleted"])
def test_graph_context_rejects_foreign_wrong_type_and_deleted(fault):
    node = create_extracted_node(source_raw_id="raw", account_id="other" if fault == "account" else "m3bench",
                                 extractor_type="wrong" if fault == "extractor" else "episodic_memory_generator",
                                 extraction_schema="m3", extracted_content={"source_video_id": "v", "source_clip_id": 1})
    service = GraphService(Mock(), Mock())
    service._store.get_node.return_value = None if fault == "deleted" else node
    service._vector_index.get_records_by_clips.return_value = [IndexRecord(
        id="r", uri="u", level=2, text="", filters={}, metadata={"extracted_node_id": node.node_id})]
    assert service.get_evidence_by_clips([1], ctx(), scope=ClipScope("v"),
                                         extractor_types=("episodic_memory_generator",)) == []


def test_direct_rechecks_backend_returning_wrong_clip_or_foreign_account():
    s, _ = make_searcher()
    all_records = list(s._index._records.values())
    # A faulty backend ignores exact clip filters, and returns a foreign record
    # whose video/clip metadata looks valid. Neither may enter the prompt.
    foreign = replace(next(r for r in all_records if r.metadata["extracted_node_id"] == "before"),
                      filters={"account_id": "other", "video_id": "v", "extractor_type": "episodic_memory_generator"})
    wrong = next(r for r in all_records if r.metadata["extracted_node_id"] == "d1")
    s._index.get_records_by_clips = Mock(return_value=[wrong, foreign])
    result = s.search("where", video_id="v", top_k=1, candidate_k=1, include_neighbors=1, context_k=2)
    assert [h.source_extracted_node_id for h in result] == ["main"]
