"""Offline contracts shared by graph-backed and vector-index-backed M3 QA."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from core.tools.operational_tool_spec import call_with_injection
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, Search, SearchConfig
from m3.retrieval_core import RetrievalBudget, select_evidence
from m3.search import EpisodicSearcher
from providers.llm import MockLLM
from service.graph_service import EvidenceHit


@pytest.mark.parametrize("final,candidate,legacy,expected", [
    (3, None, None, 3), (3, None, 6, 6), (3, 8, None, 8),
    (8, 3, None, 8), (3, 4, 6, 4),
])
def test_candidate_and_final_budgets_are_independent(final, candidate, legacy, expected):
    budget = RetrievalBudget(final, candidate)
    assert budget.final_top_k == final
    assert budget.fetch_k(legacy_default=legacy) == expected


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
@pytest.mark.parametrize("field", ["final_top_k", "candidate_k"])
def test_invalid_budget_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        RetrievalBudget(**{"final_top_k": 3, field: value})


def test_union_keeps_best_occurrence_sorts_then_limits_without_mutation():
    items = [("a", .1), ("b", .8), ("a", .9), ("c", .8)]
    original = list(items)
    selected = select_evidence(items, identity=lambda x: x[0], score=lambda x: x[1], top_k=2)
    assert selected == [("a", .9), ("b", .8)]
    assert items == original
    assert select_evidence([], identity=lambda x: x, score=lambda x: 0) == []


def test_unknown_identities_remain_distinct_and_nonfinite_scores_excluded():
    items = [(None, .2), (None, .3), ("a", float("nan")), ("b", float("inf"))]
    assert select_evidence(items, identity=lambda x: x[0], score=lambda x: x[1]) == [
        (None, .3), (None, .2),
    ]


class Embedder:
    def embed_texts(self, texts):
        return [[1., 0.] for _ in texts]


def ctx():
    return RequestContext(account_id="m3bench", user_id="u", agent_id="a", session_id="s", trace_id="t")


def hit(node_id, score, *, transcript=None):
    return EvidenceHit(
        node=SimpleNamespace(node_id=node_id, extracted_content={
            "description": "room", "speech_transcript": transcript or [],
        }),
        score=score, video_id="v", clip_id=0,
    )


def graph_search(graph, config, top_k=2):
    return call_with_injection(
        Search._operational_tool_spec, {"query": "Lewis album", "top_k": top_k},
        ctx=ctx(), graph=graph, embedder=Embedder(), search_config=config,
    )


@pytest.fixture
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())


def test_dual_query_union_without_transcripts_is_sorted_and_capped(offline_tokenizer):
    graph = Mock()
    graph.search_by_embedding.side_effect = [
        [hit("a", .6), hit("b", .2)],
        [hit("c", .9), hit("a", .8)],
    ]
    result = graph_search(graph, SearchConfig(video_id="v", before_clip=2, before_sec=25, candidate_k=7))
    assert [h["node_id"] for h in result["hits"]] == ["c", "a"]
    assert [h["score"] for h in result["hits"]] == [.9, .8]
    assert result["candidate_k"] == 7
    assert result["final_top_k"] == 2
    assert result["retrieval_queries"] == ["Lewis album", "album"]
    for call in graph.search_by_embedding.call_args_list:
        assert call.kwargs["top_k"] == 7
        scope = call.kwargs["scope"]
        assert (scope.video_id, scope.before_clip, scope.before_sec) == ("v", 2, 25)
        assert call.kwargs["ctx"].account_id == "m3bench"


def test_final_cap_is_applied_after_reranking_all_candidates(monkeypatch):
    graph = Mock()
    graph.search_by_embedding.side_effect = [[hit("a", .9), hit("b", .8)], [hit("c", .6)]]

    def rerank(query, hits, **kwargs):
        assert {h.node.node_id for h in hits} == {"a", "b", "c"}
        return [replace(h, score=1 if h.node.node_id == "c" else h.score) for h in hits]

    monkeypatch.setattr("m3.transcript_rerank.rerank_with_transcript_bm25", rerank)
    result = graph_search(graph, SearchConfig(candidate_k=3), top_k=1)
    assert [h["node_id"] for h in result["hits"]] == ["c"]


def record(node_id, clip, text, vector, *, video="v", start=0, duration=10):
    return {
        "source_extracted_node_id": node_id, "video_id": video, "clip_id": clip,
        "text": text, "embedding": vector,
        "metadata": {"source_start_sec": start, "source_duration_sec": duration},
    }


def test_direct_wider_candidates_can_rescue_keyword_hit_without_neighbors():
    searcher = EpisodicSearcher.from_records([
        record("a", 0, "room", [1., 0.]),
        record("b", 1, "album", [.8, .6]),
        record("future", 2, "album", [1., 0.], start=20),
        record("other", 0, "album", [1., 0.], video="other"),
    ], embedder=Embedder(), hybrid_weight=1)
    scope = dict(video_id="v", before_clip=2, before_sec=15, top_k=1)
    assert [h.source_extracted_node_id for h in searcher.search("album", **scope)] == ["a"]
    spy = Mock(wraps=searcher._index.search_by_vector)
    searcher._index.search_by_vector = spy
    evidence = searcher.search("album", candidate_k=4, **scope)
    assert [h.source_extracted_node_id for h in evidence] == ["b"]
    assert spy.call_args.kwargs["top_k"] == 4
    assert spy.call_args.kwargs["filters"] == {
        "account_id": "m3bench", "video_id": "v", "before_clip": 2, "before_sec": 15,
        "extractor_type": "episodic_memory_generator",
    }


def test_both_entrypoints_deduplicate_by_node_identity(offline_tokenizer):
    graph = Mock()
    graph.search_by_embedding.side_effect = [[hit("a", .5), hit("a", 1.)], [hit("b", .8)]]
    graph_result = graph_search(graph, SearchConfig(candidate_k=4))
    searcher = EpisodicSearcher.from_records([
        record("a", 0, "room", [.5, .8660254]),
        record("a", 1, "room", [1., 0.]),
        record("b", 2, "room", [.8, .6]),
    ], embedder=Embedder(), hybrid_weight=0)
    direct = searcher.search("room", top_k=2, candidate_k=4)
    assert [h["node_id"] for h in graph_result["hits"]] == [h.source_extracted_node_id for h in direct] == ["a", "b"]
    assert [h.score for h in direct] == pytest.approx([1., .8])


def test_neighbor_expansion_cannot_exceed_final_cap_or_repeat_identity(monkeypatch):
    searcher = EpisodicSearcher.from_records([
        record("a", 0, "room", [1., 0.]),
    ], embedder=Embedder())
    b = record("b", 1, "room", [1., 0.])
    monkeypatch.setattr(searcher, "_expand_neighbors", lambda selected, **kw: [
        *selected, (.8, b), (.9, b), (.7, record("c", 2, "room", [1., 0.])),
    ])
    result = searcher.search("room", top_k=2, include_neighbors=1)
    assert [h.source_extracted_node_id for h in result] == ["a", "b"]


def test_runtime_candidate_config_is_hidden_isolated_and_audited(offline_tokenizer):
    graph = Mock()
    graph.search_by_embedding.return_value = [hit("a", .9), hit("b", .8)]
    a = OGMemoryControlAdapter(MockLLM(), graph, Embedder(), candidate_k=6)
    b = OGMemoryControlAdapter(MockLLM(), graph, Embedder(), candidate_k=9)
    task = M3TaskContext(question="room")
    a._build_messages(ctx(), task)
    b._build_messages(ctx(), M3TaskContext(question="room"))
    a._execute_tool("Search", {"query": "room", "top_k": 1}, ctx())
    a._execute_tool("Search", {"query": "room", "top_k": 1}, ctx())
    assert graph.search_by_embedding.call_count == 1
    assert graph.search_by_embedding.call_args.kwargs["top_k"] == 6
    assert task.search_history[0]["node_ids"] == ["a"]  # unseen candidates aren't evidence
    assert task.search_history[1]["candidate_k"] == 6
    assert task.search_history[1]["cache_hit"] is True
    assert a._search_config.candidate_k == 6
    assert b._search_config.candidate_k == 9
    schema = next(t for t in a._get_tools() if t["name"] == "Search")["input_schema"]
    assert set(schema["properties"]) == {"query", "top_k"}


@pytest.mark.parametrize("entrypoint", ["qa", "react", "eval", "search"])
def test_candidate_cli_configuration_and_validation(entrypoint):
    from m3 import eval as eval_module
    from m3 import qa, search
    parser, argv = {
        "qa": (qa.parse_args, ["--annotations", "x.json"]),
        "react": (qa.parse_react_qa_args, ["--annotations", "x.json"]),
        "eval": (eval_module.parse_args, ["--annotations", "x.json"]),
        "search": (search.parse_args, ["--query", "room"]),
    }[entrypoint]
    assert parser(argv).candidate_k is None
    assert parser([*argv, "--candidate-k", "16"]).candidate_k == 16
    for invalid in ["0", "-1"]:
        with pytest.raises(SystemExit):
            parser([*argv, "--candidate-k", invalid])
