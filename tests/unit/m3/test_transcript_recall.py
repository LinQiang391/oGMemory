"""Offline independent ASR recall and experimental fusion contracts."""
from dataclasses import replace
from unittest.mock import MagicMock, Mock

import pytest

from core.graph_models import create_extracted_node
from core.models import SeedHit
from core.transcript_lexical import (
    indexed_metadata,
    query_terms,
    transcript_score,
    validate_transcript_request,
)
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter, SearchConfig
from m3.retrieval_core import fuse_evidence
from m3.search import EpisodicSearcher
from providers.llm import MockLLM
from providers.vector_index.opengauss_index import OpenGaussVectorIndex
from service.graph_service import ClipScope, GraphService
from tests.unit.m3.test_retrieval_core import Embedder, ctx, graph_search, hit, record


def asr(text):
    return {"speech_transcript": [{"text": text, "start_sec": 0, "end_sec": 1}]}


def test_normalization_and_only_asr_text_is_searchable():
    metadata = {**asr("Ａlbum, ALBUM café!"), "description": "secret", "transcript_search_text": "forged"}
    assert indexed_metadata(metadata)["transcript_search_text"] == " album café "
    assert transcript_score("album CAFÉ secret", metadata) == pytest.approx(2 / 3)
    assert metadata["transcript_search_text"] == "forged"
    assert indexed_metadata({"transcript_search_text": "stale"})["transcript_search_text"] == ""
    assert transcript_score("text start_sec", {"speech_transcript": [{"start_sec": 1}]}) == 0
    assert transcript_score("!!!", metadata) == 0
    assert len(query_terms(" ".join(map(str, range(50))))) == 32


@pytest.mark.parametrize("filters", [{}, {"account_id": "a"}, {"account_id": " ", "video_id": "v"}])
def test_scope_required(filters):
    with pytest.raises(ValueError):
        validate_transcript_request("album", filters, 1)


@pytest.mark.parametrize("method", ["rrf", "weighted"])
def test_fusion_deduplicates_before_ranks_and_preserves_sources(method):
    vector = [("a", .8), ("a", .9), ("b", .7)]
    fused = fuse_evidence(vector, [("a", 1.), ("c", .5)], identity=lambda x: x[0], score=lambda x: x[1], method=method)
    assert fused[0].evidence == ("a", .9)
    assert fused[0].score == pytest.approx(2 / 61 if method == "rrf" else .95)
    assert fused[0].source_scores == {"vector": .9, "transcript": 1.}
    assert len(fused) == 3
    assert len(vector) == 3


def test_fusion_tie_prefers_transcript_and_missing_ids_remain_distinct():
    args = dict(identity=lambda x: x[0], score=lambda x: x[1])
    result = fuse_evidence([("v", 1)], [("t", 1)], **args)
    assert result[0].evidence[0] == "t"
    assert len(fuse_evidence([(None, 1), (None, .8)], [(None, 1)], **args)) == 3


def searcher():
    records = [record("vector", 0, "room", [1., 0.]), record("lexical", 1, "room", [0., 1.])]
    records[1]["metadata"].update(asr("album"))
    return EpisodicSearcher.from_records(records, embedder=Embedder(), hybrid_weight=0)


@pytest.mark.parametrize("method", ["rrf", "weighted"])
def test_independent_lexical_rescues_outside_vector_pool(method):
    s = searcher()
    assert s.search("album", video_id="v", top_k=1, candidate_k=1)[0].source_extracted_node_id == "vector"
    results = s.search("album", video_id="v", top_k=1, candidate_k=1, transcript_top_k=2, transcript_fusion=method)
    assert len(results) == 1
    assert results[0].source_extracted_node_id == "lexical"


def test_memory_lexical_has_no_vector_dependency_and_tracks_updates():
    s = searcher()
    index = s._index
    index.search_by_vector = Mock(side_effect=AssertionError("vector call"))
    filters = {"account_id": "m3bench", "video_id": "v"}
    hits = index.search_by_transcript("album", filters, 1)
    assert len(hits) == 1
    from core.models import IndexRecord
    rec = index.get_record(IndexRecord.generate_id(hits[0].uri, hits[0].level))
    index.upsert([replace(rec, metadata={**rec.metadata, **asr("changed")})])
    assert index.search_by_transcript("album", filters, 1) == []
    assert len(index.search_by_transcript("changed", filters, 1)) == 1
    index.delete([rec.id])
    assert index.search_by_transcript("changed", filters, 1) == []


@pytest.mark.parametrize("overrides", [{"video_id": "other"}, {"account_id": "other"}, {"before_clip": 0}, {"before_sec": 5}, {"extractor_type": "other"}])
def test_memory_lexical_scope(overrides):
    filters = {"account_id": "m3bench", "video_id": "v", **overrides}
    assert searcher()._index.search_by_transcript("album", filters, 1) == []


def test_direct_preflight_precedes_embedding():
    s = searcher()
    s._embedder = Mock()
    with pytest.raises(ValueError, match="video_id"):
        s.search("album", transcript_top_k=1)
    s._index = object()
    with pytest.raises(NotImplementedError):
        s.search("album", video_id="v", transcript_top_k=1)
    s._embedder.embed_texts.assert_not_called()


def test_graph_search_fusion_and_cache_audit(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    graph = Mock()
    graph.search_by_embedding.return_value = [hit("vector", 1.)]
    graph.search_by_transcript.return_value = [hit("lexical", 1., transcript=[{"text": "album"}])]
    result = graph_search(graph, SearchConfig(video_id="v", transcript_top_k=3), top_k=1)
    assert [h["node_id"] for h in result["hits"]] == ["lexical"]
    assert result["ranking_policy"] == "transcript_rrf"
    adapter = OGMemoryControlAdapter(MockLLM(), graph, Embedder(), transcript_top_k=3)
    task = M3TaskContext(question="album", video_id="v")
    adapter._build_messages(ctx(), task)
    adapter._execute_tool("Search", {"query": "album", "top_k": 1}, ctx())
    adapter._execute_tool("Search", {"query": "album", "top_k": 1}, ctx())
    assert task.search_history[0]["retrieval_branches"][-1]["kind"] == "transcript"
    assert task.search_history[1]["retrieval_branches"] == []
    assert task.search_history[1]["cache_hit"]
    schema = next(t for t in adapter._get_tools() if t["name"] == "Search")["input_schema"]
    assert set(schema["properties"]) == {"query", "top_k"}


@pytest.mark.parametrize("change", [{}, {"source_video_id": "other"}, {"source_start_sec": 20}, {"speech_transcript": [{"text": "changed"}]}])
def test_graph_rechecks_live_scope_and_asr(change):
    content = {"source_video_id": "v", "source_clip_id": None, "_clip_id": 0,
               "source_start_sec": 0, "source_duration_sec": 10, **asr("album"), **change}
    node = create_extracted_node(source_raw_id="raw", extractor_type="episodic_memory_generator",
                                 extraction_schema="m3", account_id="m3bench", extracted_content=content)
    store = Mock()
    store.get_node.return_value = node
    service = GraphService(store)
    service._vector_index = Mock()
    service._vector_index.search_by_transcript.return_value = [SeedHit(uri="x", score=1, metadata={"extracted_node_id": node.node_id})]
    result = service.search_by_transcript("album", ctx(), scope=ClipScope(video_id="v", before_sec=15))
    assert len(result) == (0 if change else 1)
    if result:
        assert result[0].clip_id == 0


def sql_index():
    index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
    index._table_name = "test_index"
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    index._get_connection = Mock(return_value=conn)
    index._return_connection = Mock()
    return index, conn, cursor


def test_sql_query_is_bound_scoped_and_not_vector_retrieval():
    index, conn, cursor = sql_index()
    index.search_by_transcript("ALBUM ' OR 1=1", {"account_id": "a", "video_id": "v", "before_clip": 2, "before_sec": 15, "extractor_type": ["episodic_memory_generator"]}, 3)
    sql, params = cursor.execute.call_args.args
    assert "POSITION" in sql and "transcript_search_text" in sql
    assert "ORDER BY score DESC, uri ASC" in sql and "LIMIT %(topk)s" in sql
    assert "qvec" not in params and "<=>" not in sql
    assert params["lex_0"] == " album " and params["topk"] == 3
    assert "album" not in sql.lower()
    assert "a" in params.values() and "v" in params.values()
    conn.rollback.assert_called_once()
    index._return_connection.assert_called_once_with(conn)


def test_sql_no_connection_on_empty_query_or_missing_scope_and_cleanup_on_error():
    index, conn, cursor = sql_index()
    assert index.search_by_transcript("!!!", {"account_id": "a", "video_id": "v"}, 1) == []
    with pytest.raises(ValueError):
        index.search_by_transcript("album", {}, 1)
    index._get_connection.assert_not_called()
    cursor.execute.side_effect = RuntimeError("database failure")
    with pytest.raises(RuntimeError):
        index.search_by_transcript("album", {"account_id": "a", "video_id": "v"}, 1)
    conn.rollback.assert_called_once()
    index._return_connection.assert_called_once_with(conn)


@pytest.mark.parametrize("entrypoint", ["qa", "react", "eval", "search"])
def test_transcript_cli_options(entrypoint):
    from m3 import eval as evaluation
    from m3 import qa, search
    parser, argv = {"qa": (qa.parse_args, ["--annotations", "x.json"]),
                    "react": (qa.parse_react_qa_args, ["--annotations", "x.json"]),
                    "eval": (evaluation.parse_args, ["--annotations", "x.json"]),
                    "search": (search.parse_args, ["--query", "album"])}[entrypoint]
    assert parser(argv).transcript_top_k == 0
    args = parser([*argv, "--transcript-top-k", "12", "--transcript-fusion", "weighted"])
    assert (args.transcript_top_k, args.transcript_fusion) == (12, "weighted")
    with pytest.raises(SystemExit):
        parser([*argv, "--transcript-top-k", "-1"])


def test_sql_upsert_overwrites_derived_field_without_mutating_record(monkeypatch):
    from core.models import IndexRecord
    # Avoid psycopg2 adaptation and other modules' MockJson patches.
    monkeypatch.setattr("providers.vector_index.opengauss_index.Json", lambda obj: obj)
    index, conn, cursor = sql_index()
    rec = IndexRecord(id="r", uri="m3://v/r", level=2, text="description secret",
                      filters={"account_id": "a", "video_id": "v"},
                      metadata={"_embedding": [1., 0.], **asr("ALBUM"), "transcript_search_text": "stale"})
    index.upsert([rec])
    params = cursor.execute.call_args.args[1]
    derived = [p for p in params if isinstance(p, dict) and "transcript_search_text" in p]
    assert derived and all(p["transcript_search_text"] == " album " for p in derived)
    assert rec.metadata["transcript_search_text"] == "stale"
    conn.commit.assert_called()


@pytest.mark.parametrize("account,extractor,deleted", [("other", "episodic_memory_generator", False), ("m3bench", "other", False), ("m3bench", "episodic_memory_generator", True)])
def test_graph_drops_foreign_wrong_type_and_deleted_nodes(account, extractor, deleted):
    node = create_extracted_node(source_raw_id="raw", extractor_type=extractor,
                                 extraction_schema="m3", account_id=account,
                                 extracted_content={"source_video_id": "v", **asr("album")})
    store = Mock()
    store.get_node.return_value = None if deleted else node
    service = GraphService(store)
    service._vector_index = Mock()
    service._vector_index.search_by_transcript.return_value = [SeedHit(uri="x", score=1, metadata={"extracted_node_id": node.node_id})]
    assert service.search_by_transcript("album", ctx(), scope=ClipScope(video_id="v"), extractor_types=("episodic_memory_generator",)) == []


def test_graph_only_backend_explicitly_unsupported():
    service = GraphService(Mock())
    with pytest.raises(NotImplementedError):
        service.search_by_transcript("album", ctx(), scope=ClipScope(video_id="v"))


def test_default_off_never_calls_lexical_and_run_settings_are_isolated(monkeypatch):
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    graph = Mock()
    graph.search_by_embedding.return_value = []
    graph_search(graph, SearchConfig(video_id="v"))
    graph.search_by_transcript.assert_not_called()
    off = OGMemoryControlAdapter(MockLLM(), graph, Embedder())
    on = OGMemoryControlAdapter(MockLLM(), graph, Embedder(), transcript_top_k=8, transcript_fusion="weighted")
    off._build_messages(ctx(), M3TaskContext(question="album", video_id="v"))
    on._build_messages(ctx(), M3TaskContext(question="album", video_id="v"))
    assert off._search_config.transcript_top_k == 0
    assert (on._search_config.transcript_top_k, on._search_config.transcript_fusion) == (8, "weighted")


def test_memory_scope_applied_before_lexical_topk():
    valid = record("valid", 0, "room", [0., 1.])
    valid["metadata"].update(asr("album"))
    future = record("future", 2, "room", [1., 0.], start=20)
    future["metadata"].update(asr("album secret"))
    other = record("other", 0, "room", [1., 0.], video="other")
    other["metadata"].update(asr("album secret"))
    s = EpisodicSearcher.from_records([future, other, valid], embedder=Embedder())
    hits = s._index.search_by_transcript("album secret", {"account_id": "m3bench", "video_id": "v", "before_sec": 15}, 1)
    assert [h.metadata["extracted_node_id"] for h in hits] == ["valid"]
