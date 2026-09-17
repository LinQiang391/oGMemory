"""Offline regressions for approximate question times and local segmentation."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.graph_models import create_extracted_node
from core.temporal import contains_time, within_time
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, SearchConfig
from m3.search import EpisodicSearcher
from m3.temporal import resolve_question_scope
from providers.graph_store.memory_graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import ClipScope, GraphService
from tests.unit.m3.test_retrieval_core import Embedder, ctx, graph_search, record
from tests.unit.m3.test_transcript_recall import sql_index


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    durations = [30.066667, 30.16599, 30.1, 30.16599, .16599]
    transcripts = [
        [(0.85, 2.13, "So happy today"), (23.34, 24.62, "I'm exhausted"),
         (25.61, 27.09, "But I'm still very happy")],
        [],
        [(.69, 10.4, "put our photos in the album"),
         (15.41, 20.15, "fetch the pictures in my bag")],
        [(14.51, 16.69, "I need the white one"), (25.71, 27.89, "the white one")],
        [],
    ]
    records = []
    store = MemoryGraphStore()
    for clip, duration in enumerate(durations):
        data = {
            "source_video_id": "v", "source_clip_id": clip,
            "source_start_sec": clip * 30, "source_duration_sec": duration,
            "description": "In the bedroom" if clip == 0 else f"clip {clip}",
            "speech_transcript": [
                {"start_sec": a, "end_sec": b, "text": text} for a, b, text in transcripts[clip]
            ],
            "embedding": [1., 0.],
        }
        node = create_extracted_node(
            source_raw_id=f"raw{clip}", extractor_type="episodic_memory_generator",
            extraction_schema="m3", account_id="m3bench", extracted_content=data,
        )
        store.create_node(node.serialize(), ctx())
        row = record(node.node_id, clip, data["description"], [1., 0.],
                     start=clip * 30, duration=duration)
        row["metadata"].update(data)
        records.append(row)
    searcher = EpisodicSearcher.from_records(records, embedder=Embedder(), hybrid_weight=0)
    graph = GraphService(store, vector_index=searcher._index)
    return searcher, graph


@pytest.mark.parametrize("timestamp,annotation,current,end,answer", [
    (5, 2, 0, 30.066667, "bedroom"),
    (27, 2, 0, 30.066667, "exhausted"),
    (68, 4, 2, 90.1, "album"),
    (78, 4, 2, 90.1, "pictures in my bag"),
    (118, 5, 3, 120.16599, "white"),
])
def test_smoke_answers_eligible_in_both_paths(evidence, timestamp, annotation, current, end, answer):
    searcher, graph = evidence
    def lookup(t):
        return searcher.get_clip_at_time(video_id="v", timestamp=t)
    clip, cutoff, audit = resolve_question_scope(annotation, timestamp, clip_lookup=lookup)
    assert (clip, cutoff) == (current, end)
    graph_bounds = resolve_question_scope(
        annotation, timestamp,
        clip_lookup=lambda t: graph.get_clip_at_time(ctx(), video_id="v", timestamp=t),
    )
    assert graph_bounds == (clip, cutoff, audit)
    assert audit["future_allowance_sec"] == pytest.approx(end - timestamp)
    direct = searcher.search("photos", video_id="v", before_clip=clip, before_sec=cutoff,
                             top_k=8, transcript_top_k=8, include_neighbors=2, context_k=4)
    result = graph_search(
        graph, SearchConfig(video_id="v", before_clip=clip, before_sec=cutoff,
                            transcript_top_k=8, include_neighbors=2, context_k=4), top_k=8,
    )
    assert {ev.clip_id for ev in direct} == set(range(current + 1))
    assert {hit["clip_id"] for hit in result["hits"]} == set(range(current + 1))
    assert not result["boundary_hits"]
    assert answer in json.dumps([ev.to_dict() for ev in direct])
    assert answer in json.dumps(result)


def test_strict_boundary_uses_timestamp_not_annotation_ordinal(evidence):
    searcher, graph = evidence
    # Even a low-level inclusive upper bound need not equal the current clip.
    for bound in (6, None):
        hits = graph.get_boundary_transcript(ctx(), scope=ClipScope("v", bound, 118))
        assert [hit.clip_id for hit in hits] == [3]
        assert "white" in json.dumps(hits[0].node.extracted_content)
        assert "description" not in hits[0].node.extracted_content
        direct = searcher.search("white", video_id="v", before_clip=bound, before_sec=118, top_k=8)
        assert direct[-1].metadata["evidence_role"] == "boundary_transcript"
        assert direct[-1].clip_id == 3
    # Explicit low-level clip restrictions are not discarded.
    assert graph.get_boundary_transcript(ctx(), scope=ClipScope("v", 2, 118)) == []


@pytest.mark.parametrize("timestamp,phrase", [(68, "album"), (78, "pictures")])
def test_lookahead_finishes_speech_without_admitting_full_future_summary(evidence, timestamp, phrase):
    searcher, graph = evidence
    for policy, grace, present in [("strict", 0, False), ("lookahead", 3, True)]:
        clip, cutoff, audit = resolve_question_scope(
            -1, timestamp, policy=policy, lookahead_seconds=grace,
            clip_lookup=lambda t: graph.get_clip_at_time(ctx(), video_id="v", timestamp=t),
        )
        assert clip == 2  # Timestamp wins over incompatible annotation.
        assert cutoff == timestamp + grace
        result = graph_search(graph, SearchConfig(video_id="v", before_clip=clip, before_sec=cutoff), top_k=8)
        assert (phrase in json.dumps([*result["hits"], *result["boundary_hits"]])) is present
        assert all(h["clip_id"] < 2 for h in result["hits"])
        assert all("description" not in h or not h["description"] for h in result["boundary_hits"])
        direct = searcher.search("photos", video_id="v", before_clip=clip, before_sec=cutoff, top_k=8)
        assert (phrase in json.dumps([ev.to_dict() for ev in direct])) is present
        assert audit["future_allowance_sec"] == grace


@pytest.mark.parametrize("timestamp,expected", [(0, 0), (30, 1), (60, 2), (90, 3), (120, 4)])
def test_overlaps_and_exact_starts_choose_latest_interval(evidence, timestamp, expected):
    searcher, graph = evidence
    assert searcher.get_clip_at_time(video_id="v", timestamp=timestamp)["clip_id"] == expected
    assert graph.get_clip_at_time(ctx(), video_id="v", timestamp=timestamp)["clip_id"] == expected


@pytest.mark.parametrize("policy", ["strict", "current_clip", "lookahead"])
def test_missing_provenance_keeps_seconds_bound(policy):
    clip, cutoff, audit = resolve_question_scope(99, 25, policy=policy, clip_lookup=lambda t: None)
    assert (clip, cutoff) == (None, 25)
    assert audit["resolution"] == "no_containing_clip_strict_fallback"
    assert not within_time({}, cutoff)


def test_missing_timestamp_preserves_clip_bound_and_unrestricted_skips_lookup():
    lookup = Mock(side_effect=AssertionError("unexpected lookup"))
    assert resolve_question_scope(2, None, clip_lookup=lookup)[:2] == (2, None)
    assert resolve_question_scope("bad", "bad", policy="unrestricted", clip_lookup=lookup)[:2] == (None, None)
    assert resolve_question_scope("bad", "bad", enabled=False, lookahead_seconds=5)[:2] == (None, None)


@pytest.mark.parametrize("kwargs", [
    {"policy": "typo"}, {"timestamp": "nan"}, {"lookahead_seconds": -1},
    {"lookahead_seconds": 5}, {"lookahead_seconds": True}, {"policy": "lookahead", "lookahead_seconds": "inf"},
    {"clip_lookup": lambda t: {"clip_id": 1, "source_start_sec": 30, "source_duration_sec": 30}},
    {"clip_lookup": lambda t: {"clip_id": True, "source_start_sec": 0, "source_duration_sec": 30}},
])
def test_bad_policy_or_provenance_is_not_unrestricted(kwargs):
    args = {"timestamp": 5, "clip_lookup": lambda t: None, **kwargs}
    with pytest.raises(ValueError):
        resolve_question_scope(**args)


def test_scope_is_resolved_once_and_disclosed_to_model_and_audit(evidence):
    _, graph = evidence
    graph.get_clip_at_time = Mock(wraps=graph.get_clip_at_time)
    adapter = OGMemoryControlAdapter(graph=graph, embedder=Embedder(), llm=MockLLM())
    task = M3TaskContext(
        question="Where at the start?", video_id="v", temporal_policy="current_clip",
        annotation_before_clip=2, annotation_timestamp="00:05",
    )
    messages = adapter._build_messages(ctx(), task)
    assert task.before_clip == 0 and task.before_sec == 30.066667
    assert "retrospective context" in messages[0]["content"]
    assert task.search_stats["temporal_scope"]["question_sec"] == 5
    for query in ("bedroom", "room"):
        result = adapter._execute_tool("Search", {"query": query}, ctx())
        assert [h["clip_id"] for h in result["hits"]] == [0]
        assert result["hits"][0]["source_start_sec"] == 0
    graph.get_clip_at_time.assert_called_once()
    assert task.search_history[-1]["temporal_scope"] == task.temporal_scope


def test_interval_lookup_is_scoped_and_live_provenance_is_rechecked(evidence):
    searcher, graph = evidence
    index = searcher._index
    filters = {"account_id": "m3bench", "video_id": "v", "extractor_type": "episodic_memory_generator"}
    for field, value in [("account_id", "other"), ("video_id", "other"), ("extractor_type", "other")]:
        assert index.get_records_at_time({**filters, field: value}, 5) == []
    with pytest.raises(ValueError):
        index.get_records_at_time({"video_id": "v"}, 5)
    real = graph._store.get_node
    graph._store.get_node = Mock(side_effect=lambda node_id, context: replace(
        real(node_id, context),
        extracted_content={**real(node_id, context).extracted_content, "source_start_sec": 999},
    ))
    assert graph.get_clip_at_time(ctx(), video_id="v", timestamp=5) is None


def test_sql_interval_query_is_parameterized_and_filters_before_returning():
    index, conn, cursor = sql_index()
    good = {"source_start_sec": 60, "source_duration_sec": 30.1, "source_end_sec": 90.1, "clip_id": 2}
    cursor.fetchall.return_value = [
        {"id": "r", "uri": "u", "level": 2, "text": "complete", "filters": {}, "metadata": good},
        {"id": "bad", "uri": "u2", "level": 2, "text": "", "filters": {},
         "metadata": {**good, "source_end_sec": 80}},
    ]
    records = index.get_records_at_time(
        {"account_id": "a", "video_id": "v", "extractor_type": "episodic_memory_generator"}, 68,
    )
    assert [r.id for r in records] == ["r"]
    sql, params = cursor.execute.call_args.args
    assert params["at_sec"] == 68
    assert params["f_account_id"] == "a" and params["f_video_id"] == "v"
    assert params["f_extractor_type"] == "episodic_memory_generator"
    assert "CASE WHEN" in sql and " <= %(at_sec)s" in sql and " > %(at_sec)s" in sql
    assert "LIMIT" not in sql and "<=>" not in sql
    conn.rollback.assert_called_once()
    index._return_connection.assert_called_once_with(conn)


def test_sql_bad_time_or_missing_scope_does_not_connect():
    index, _, _ = sql_index()
    for filters, time in [({}, 5), ({"account_id": "a", "video_id": "v"}, float("nan"))]:
        with pytest.raises(ValueError):
            index.get_records_at_time(filters, time)
    index._get_connection.assert_not_called()


def test_contains_time_rejects_unknown_and_inconsistent_provenance():
    assert not contains_time({}, 5)
    assert not contains_time({"source_start_sec": 0, "source_duration_sec": 10, "source_end_sec": 9}, 5)
    assert not contains_time({"source_start_sec": 0, "source_duration_sec": 10}, 10)
