"""Offline temporal contract tests: no future or unknown complete evidence."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.graph_models import create_extracted_node
from core.models import RequestContext, SeedHit
from core.temporal import temporal_metadata, within_time
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter
from m3.index_episodic import _upsert_to_vector_index, index_record_from_episodic
from m3.search import EpisodicSearcher
from m3.temporal import annotation_scope
from providers.graph_store.memory_graph_store import MemoryGraphStore
from providers.graph_store.sql_graph_store import SQLGraphStore
from providers.llm import MockLLM
from providers.vector_index.opengauss_index import OpenGaussVectorIndex
from service.graph_service import ClipScope, GraphService


class Embedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize(
    "raw,expected", [(0, 0), ("00:00", 0), ("01:02.5", 62.5), ("01:02:03", 3723), (62.5, 62.5)]
)
def test_annotation_absolute_seconds(raw, expected):
    assert annotation_scope(1, raw) == (2, expected)


def test_annotation_without_timestamp_never_includes_current_clip():
    assert annotation_scope(1, None) == (1, None)
    assert annotation_scope(-1, None) == (-1, None)
    assert annotation_scope(None, 0) == (None, 0)
    assert annotation_scope("bad", "bad", enabled=False) == (None, None)


@pytest.mark.parametrize("bad", [True, -1, "nan", "inf", "bad", "1:60", "1:2:3:4", "1.2:02"])
def test_invalid_cutoff_is_not_unrestricted(bad):
    with pytest.raises(ValueError):
        annotation_scope(0, bad)


@pytest.mark.parametrize("bad", [True, 1.5, "nan", "c0"])
def test_invalid_clip_is_not_unrestricted(bad):
    with pytest.raises(ValueError):
        annotation_scope(bad, None)


@pytest.mark.parametrize(
    "content",
    [
        {},
        {"source_start_sec": 0},
        {"source_start_sec": 0, "source_duration_sec": float("nan")},
        {"source_start_sec": 20, "source_duration_sec": 20},
        {"source_start_sec": 0, "source_duration_sec": 10, "source_end_sec": 2},
        {
            "source_start_sec": 0,
            "source_duration_sec": 10,
            "speech_transcript": [{"start_sec": 9, "end_sec": 40}],
        },
    ],
)
def test_unknown_crossing_and_contaminated_summaries_are_excluded(content):
    assert not within_time(content, 30)
    assert within_time(content, None)


def test_clip_relative_asr_and_inclusive_video_cutoff():
    content = {
        "source_start_sec": 20,
        "source_duration_sec": 10,
        "speech_transcript": [{"start_sec": 1, "end_sec": 10}],
    }
    assert within_time(content, 30)
    assert not within_time(content, 29.999)


def record(clip, start, duration, *, vector=None, video="v"):
    return {
        "video_id": video,
        "clip_id": clip,
        "source_extracted_node_id": f"ext-{clip}",
        "text": f"past and future facts in clip {clip}",
        "embedding": vector or [1.0, 0.0],
        "metadata": {"source_start_sec": start, "source_duration_sec": duration},
    }


def test_direct_search_filters_before_topk_and_neighbors():
    searcher = EpisodicSearcher.from_records(
        [
            record(0, 0, 10, vector=[0.5, 0.5]),  # safe, lower similarity
            record(1, 10, 20),  # both before/after facts: exclude entire summary
            record(2, None, None),
            record(3, 0, 10, video="other"),
        ],
        embedder=Embedder(),
    )
    hits = searcher.search("facts", video_id="v", before_clip=3, before_sec=15, top_k=1)
    assert [h.clip_id for h in hits] == [0]
    hits = searcher.search("facts", video_id="v", before_sec=15, top_k=5, include_neighbors=3)
    assert [h.clip_id for h in hits] == [0]
    assert len(searcher.search("facts", video_id="v", top_k=5)) == 3


def test_direct_final_check_rejects_backend_ignoring_filters():
    index = SimpleNamespace(
        search_by_vector=lambda **kw: [
            SeedHit(
                uri="m3://v/clips/1/episodic",
                score=1,
                metadata={
                    "video_id": "v",
                    "clip_id": 1,
                    "source_start_sec": 10,
                    "source_duration_sec": 20,
                },
            )
        ]
    )
    searcher = EpisodicSearcher(index, embedder=Embedder())
    index.get_records_at_time = MagicMock(return_value=[])
    assert searcher.search("future", video_id="v", before_sec=15) == []


def ctx():
    return RequestContext(account_id="a", user_id="u", agent_id="a", session_id="s", trace_id="t")


def test_adapter_passes_resolved_scope_without_plus_one():
    graph = MagicMock()
    graph.search_by_embedding.return_value = []
    adapter = OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=Embedder())
    clip, cutoff = annotation_scope(1, "00:25")
    adapter._build_messages(
        ctx(), M3TaskContext(question="facts", video_id="v", before_clip=clip, before_sec=cutoff)
    )
    adapter._execute_tool("Search", {"query": "facts"}, ctx())
    assert graph.search_by_embedding.call_args.kwargs["scope"] == ClipScope(
        video_id="v", before_clip=2, before_sec=25
    )


def test_graph_store_and_service_filter_before_topk_and_recheck_stale_index():
    store = MemoryGraphStore()
    nodes = []
    for start, duration, emb in [(0, 10, [0.5, 0.5]), (10, 20, [1, 0])]:
        node = create_extracted_node(
            source_raw_id="raw",
            extractor_type="episodic_memory_generator",
            extraction_schema="m3",
            account_id="a",
            extracted_content={
                "source_video_id": "v",
                "source_clip_id": len(nodes),
                "source_start_sec": start,
                "source_duration_sec": duration,
                "embedding": emb,
            },
        )
        store.create_node(node.serialize(), ctx())
        nodes.append(node)
    service = GraphService(store)
    scope = ClipScope(video_id="v", before_sec=15)
    assert [
        h.node.node_id for h in service.search_by_embedding([1, 0], ctx(), scope=scope, top_k=1)
    ] == [nodes[0].node_id]
    index = MagicMock()
    index.search_by_vector.return_value = [
        SeedHit(
            uri="stale",
            score=1,
            metadata={"extracted_node_id": nodes[1].node_id, "source_end_sec": 5},
        )
    ]
    service._vector_index = index
    assert service.search_by_embedding([1, 0], ctx(), scope=scope) == []
    assert index.search_by_vector.call_args.kwargs["filters"]["before_sec"] == 15


def test_sql_graph_filters_intervals_before_ranking_without_database():
    store = SQLGraphStore.__new__(SQLGraphStore)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("future", {"embedding": [1, 0], "source_start_sec": 10, "source_duration_sec": 20}),
        ("past", {"embedding": [0.5, 0.5], "source_start_sec": 0, "source_duration_sec": 10}),
        ("unknown", {"embedding": [1, 0]}),
    ]
    store._get_connection = lambda: conn
    store._return_connection = lambda c: None
    store.bind_tenant = lambda *a: None
    assert store.search_by_embedding([1, 0], ctx(), before_sec=15, top_k=1)[0][0] == "past"


def test_index_preserves_temporal_provenance():
    rec = {
        "video_id": "v",
        "clip_id": 0,
        "extracted_node_id": "ext",
        "description": "past",
        "source_start_sec": 20,
        "source_duration_sec": 10,
    }
    ir = index_record_from_episodic(rec, [1, 0], include_embedding=True)
    index = MagicMock()
    _upsert_to_vector_index([ir], index, SimpleNamespace(account_id="a"))
    meta = index.upsert.call_args.args[0][0].metadata
    assert temporal_metadata(meta) == {
        "source_start_sec": 20,
        "source_duration_sec": 10,
        "source_end_sec": 30,
    }


def test_sql_cutoff_zero_is_parameterized_and_casts_guarded():
    where, params = OpenGaussVectorIndex._build_where({"before_sec": 0})
    assert params["f_before_sec"] == 0
    assert "CASE WHEN" in where and "source_end_sec" in where
    assert "length(" in where and "{1,3}" in where
    with pytest.raises(ValueError):
        OpenGaussVectorIndex._build_where({"before_sec": float("nan")})


@pytest.mark.parametrize("timestamp", [None, "00:25", 0])
@pytest.mark.parametrize("enabled", [True, False])
def test_both_qa_entrypoints_use_same_annotation_scope(tmp_path, monkeypatch, timestamp, enabled):
    import m3.qa as qa

    annotation = tmp_path / "annotations.json"
    annotation.write_text("{}")
    question = {
        "video_id": "v",
        "question_id": "q",
        "question_index": 0,
        "question": "facts",
        "answer": "past",
        "type": "test",
        "before_clip": 1,
        "timestamp": timestamp,
    }
    monkeypatch.setattr(qa, "load_questions", lambda path: [question])
    monkeypatch.setattr(qa, "build_embedder", lambda *a, **kw: Embedder())
    monkeypatch.setattr(qa, "build_llm", lambda *a, **kw: MockLLM())
    monkeypatch.setattr(
        qa, "answer_question", lambda *a, **kw: {"answer": "past", "reasoning": "", "confidence": 1}
    )
    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.get_clip_at_time.return_value = {
        "clip_id": 0, "source_start_sec": 0, "source_duration_sec": 30.1,
    }
    monkeypatch.setattr(qa.EpisodicSearcher, "from_records", lambda **kw: searcher)
    argv = ["--annotations", str(annotation), "--output-jsonl", str(tmp_path / "direct.jsonl"),
            "--candidate-k", "11", "--transcript-top-k", "7", "--transcript-fusion", "weighted", "--context-k", "3", "--include-neighbors", "2"]
    if not enabled:
        argv.append("--ignore-before-clip")
    args = qa.parse_args(argv)
    args.skip_vector_index = True
    assert qa.run_qa(args) == 0
    direct = searcher.search.call_args.kwargs

    adapter = MagicMock()
    adapter.run.return_value = SimpleNamespace(
        output=["past"], iterations=1, finish_reason="completed", completed=True
    )
    factory = MagicMock(return_value=adapter)
    monkeypatch.setattr("m3.control_adapter.OGMemoryControlAdapter", factory)
    args = qa.parse_react_qa_args(argv)
    assert qa.run_react_qa(args, graph=MagicMock(), embedder=Embedder(), llm=MockLLM()) == 0
    task = adapter.run.call_args.args[1]
    assert (direct["transcript_top_k"], direct["transcript_fusion"]) == (7, "weighted")
    assert factory.call_args.kwargs["transcript_top_k"] == 7
    assert factory.call_args.kwargs["transcript_fusion"] == "weighted"
    assert (direct["context_k"], direct["include_neighbors"]) == (3, 2)
    assert factory.call_args.kwargs["context_k"] == 3
    assert factory.call_args.kwargs["include_neighbors"] == 2
    assert direct["candidate_k"] == 11
    assert factory.call_args.kwargs["candidate_k"] == 11
    from m3.temporal import resolve_question_scope
    *expected, _ = resolve_question_scope(
        1, timestamp, enabled=enabled, clip_lookup=lambda t: searcher.get_clip_at_time.return_value,
    )
    assert (direct["before_clip"], direct["before_sec"]) == tuple(expected)
    assert (task.annotation_before_clip, task.annotation_timestamp) == (1, timestamp)
    assert task.temporal_policy == ("current_clip" if enabled else "unrestricted")


def test_eval_entrypoint_carries_timestamp_and_rejects_invalid():
    from m3.eval import answer_with_adapter

    adapter = MagicMock()
    adapter.run.return_value = SimpleNamespace(
        output=["past"], iterations=1, finish_reason="completed", completed=True
    )
    result = answer_with_adapter(
        adapter=adapter,
        question="facts",
        video_id="v",
        question_id="q",
        before_clip=1,
        timestamp=25,
    )
    assert not result.error
    task = adapter.run.call_args.args[1]
    assert (task.annotation_before_clip, task.annotation_timestamp) == (1, 25)
    assert task.temporal_policy == "current_clip"
    adapter.reset_mock()
    result = answer_with_adapter(
        adapter=adapter, question="facts", video_id="v", question_id="q", timestamp="bad"
    )
    assert result.error
    adapter.run.assert_not_called()
