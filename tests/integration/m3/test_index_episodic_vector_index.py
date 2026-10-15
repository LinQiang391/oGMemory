"""Tests that m3.index_episodic upserts embeddings to a VectorIndex.

After the migration off ``ExtractedNode.extracted_content["embedding"]``,
the canonical retrieval source is a :class:`VectorIndex`
(``OpenGaussVectorIndex`` in production, ``InMemoryVectorIndex`` for
these tests). The driver builds one :class:`IndexRecord` per episodic
clip and upserts it; ``GraphService.search_by_embedding`` reads from
the same instance via its constructor-injected ``vector_index``.

What we assert:

- The driver builds the right :class:`IndexRecord` shape (id from
  ``IndexRecord.generate_id``, ``filters`` carrying account_id +
  video_id + extractor_type, ``metadata`` carrying the
  extracted_node_id forward-channel + the M3 scope columns).
- The audit JSONL no longer carries the per-record embedding (the
  manifest is text+metadata only now).
- ``GraphService.search_by_embedding`` returns hits by routing
  through the injected VectorIndex — proving the round-trip works
  end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.models import IndexRecord, RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from providers.graph_store import MemoryGraphStore
from providers.vector_index import InMemoryVectorIndex
from service.graph_service import GraphService


def _make_ctx(account_id: str = M3BENCH_ACCOUNT_ID) -> RequestContext:
    return RequestContext(
        account_id=account_id, user_id="u", agent_id="a",
        session_id="m3bench-index-test", trace_id="t-test",
    )


def _ext_id(i: int) -> str:
    return f"ext_{i:06d}".ljust(32, "x")


def _vid_id(i: int) -> str:
    return f"vid_{i:06d}".ljust(32, "x")


def _write_episodic(path: Path, n: int = 3) -> None:
    """Synthesise an episodic_memory.jsonl with `n` records, each with
    a stable extracted_node_id so the test can look it back up."""
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "video_id": f"vid_{i}",
            "clip_id": i,
            "video_raw_node_id": _vid_id(i),
            "extracted_node_id": _ext_id(i),
            "description": f"Episode {i}: the woman walks into the room.",
            "characters": [],
            "events": [],
            "entities": [],
            "frame_count": 4,
            "has_audio": False,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_graph(graph: GraphService, ctx: RequestContext, n: int) -> list[str]:
    """Pre-populate the graph with `n` ExtractedNodes so the
    VectorIndex → GraphService.search_by_embedding round-trip has
    live nodes to forward-resolve to."""
    ids = []
    for i in range(n):
        ext_id = _ext_id(i)
        from core.graph_models import create_extracted_node
        node = create_extracted_node(
            source_raw_id=_vid_id(i),
            extractor_type="episodic_memory_generator",
            extraction_schema="m3_bench_v1",
            account_id=ctx.account_id,
            extracted_content={
                "description": f"Episode {i}",
                "source_video_id": f"vid_{i}",
                "source_clip_id": i,
            },
            node_id=ext_id,
        )
        graph._store.create_node(node.serialize(), ctx)
        ids.append(ext_id)
    return ids


def _make_args(episodic_path: Path, index_path: Path) -> list[str]:
    return [
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]


def test_run_upserts_embeddings_to_vector_index(tmp_path: Path):
    """The driver builds one IndexRecord per episodic clip and upserts
    it to the injected VectorIndex. We assert on the IndexRecord
    shape (id, filters, metadata) so a future refactor of the writer
    that preserves these fields keeps the test green."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    vector_index = InMemoryVectorIndex(dimension=16)
    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _make_ctx()
    _seed_graph(graph, ctx, n=3)

    from m3.index_episodic import parse_args, run
    rc = run(parse_args(_make_args(episodic_path, index_path)),
             graph=graph, vector_index=vector_index)
    assert rc == 0

    # Every clip should now be in the VectorIndex.
    assert vector_index.count() == 3

    # Spot-check one record's shape. ``IndexRecord.generate_id`` is
    # sha256(uri + ":2")[:16]; the URI follows the m3://<video>/
    # clips/<clip>/episodic convention so re-indexing the same
    # clip produces the same id (idempotent upsert).
    target_uri = f"m3://accounts/{M3BENCH_ACCOUNT_ID}/videos/vid_0/clips/0/episodic"
    target_id = IndexRecord.generate_id(target_uri, level=2)
    record = vector_index.get_record(target_id)
    assert record is not None, f"missing IndexRecord for {target_uri}"
    assert record.uri == target_uri
    assert record.level == 2
    assert record.text.startswith("Description: Episode 0")
    # Filters carry the tenant + M3 scope columns used by
    # GraphService.search_by_embedding to push the QA scope down to
    # the index.
    assert record.filters["account_id"] == M3BENCH_ACCOUNT_ID
    assert record.filters["video_id"] == "vid_0"
    assert record.filters["extractor_type"] == "episodic_memory_generator"
    # Metadata carries the forward channel back to the graph node
    # plus the M3 scope columns the index's WHERE clause filters on.
    assert record.metadata["extracted_node_id"] == _ext_id(0)
    assert record.metadata["video_id"] == "vid_0"
    assert record.metadata["clip_id"] == 0
    assert record.metadata["before_clip"] == 0
    assert "_embedding" in record.metadata
    assert len(record.metadata["_embedding"]) == 16


def test_run_skips_records_missing_extracted_node_id(tmp_path: Path):
    """Records without an ``extracted_node_id`` are silently dropped —
    they can't reverse-resolve to a graph node, so an embedding-less
    index entry would be worse than no entry."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    # Manually write one valid + one invalid (no extracted_node_id).
    valid = json.dumps({
        "video_id": "vid_0",
        "clip_id": 0,
        "video_raw_node_id": _vid_id(0),
        "extracted_node_id": _ext_id(0),
        "description": "ok",
    })
    invalid = json.dumps({
        "video_id": "vid_1",
        "clip_id": 1,
        "video_raw_node_id": _vid_id(1),
        # no extracted_node_id
        "description": "orphan",
    })
    episodic_path.write_text(valid + "\n" + invalid + "\n", encoding="utf-8")

    vector_index = InMemoryVectorIndex(dimension=16)
    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    _seed_graph(graph, _make_ctx(), n=2)

    from m3.index_episodic import parse_args, run
    rc = run(parse_args(_make_args(episodic_path, index_path)),
             graph=graph, vector_index=vector_index)
    assert rc == 0
    # Only the valid record landed in the index.
    assert vector_index.count() == 1
    # The audit JSONL still lists both records (it's a debug artefact,
    # not a retrieval source). The invalid one survives because we
    # surface all text we processed.
    records = [json.loads(l) for l in index_path.read_text().splitlines() if l]
    assert len(records) == 2


def test_run_without_injected_vector_index_returns_zero(tmp_path: Path):
    """Backwards compat: ``run`` with neither ``graph`` nor
    ``vector_index`` injected must construct a VectorIndex from the
    CLI args / YAML config. We can't stand up an OpenGauss here, so
    we override ``--vector-index-backend=memory`` to force an
    in-memory instance."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=2)

    from m3.index_episodic import parse_args, run
    args_list = _make_args(episodic_path, index_path) + [
        "--vector-index-backend", "memory",
    ]
    rc = run(parse_args(args_list))
    assert rc == 0
    assert index_path.exists()
    records = [json.loads(l) for l in index_path.read_text().splitlines() if l]
    assert len(records) == 2


def test_audit_jsonl_omits_embedding_field(tmp_path: Path):
    """The audit JSONL is text+metadata only — embeddings live in the
    VectorIndex. A future writer that accidentally re-adds the field
    trips this test so we don't silently regress to the doubled-disk
    layout the migration set out to eliminate."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=2)

    vector_index = InMemoryVectorIndex(dimension=16)
    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)

    from m3.index_episodic import parse_args, run
    run(parse_args(_make_args(episodic_path, index_path)),
        graph=graph, vector_index=vector_index)

    records = [json.loads(l) for l in index_path.read_text().splitlines() if l]
    assert len(records) == 2
    for rec in records:
        assert "embedding" not in rec, (
            "episodic_index.jsonl is a debug artefact; embeddings live "
            "in the VectorIndex. Re-adding the field wastes disk and "
            "drifts from the canonical store."
        )


def test_indexed_embeddings_are_searchable_via_graph_service(tmp_path: Path):
    """End-to-end round-trip: ``run`` writes to the VectorIndex, and
    ``graph.search_by_embedding`` returns hits by routing through it.

    This is the M3-Bench hot-path assertion. A query embedding equal
    to one of the indexed vectors must surface that clip as the top
    hit with score ~1.0.
    """
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    vector_index = InMemoryVectorIndex(dimension=16)
    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _make_ctx()
    _seed_graph(graph, ctx, n=3)

    from m3.index_episodic import parse_args, run
    run(parse_args(_make_args(episodic_path, index_path)),
        graph=graph, vector_index=vector_index)

    # Query with the same vector that was indexed for vid_0. The
    # VectorIndex path computes cosine similarity (not 1-cosine
    # distance) so score should be ~1.0 for identical vectors.
    target_uri = f"m3://accounts/{M3BENCH_ACCOUNT_ID}/videos/vid_0/clips/0/episodic"
    target_record = vector_index.get_record(IndexRecord.generate_id(target_uri, level=2))
    query_emb = list(target_record.metadata["_embedding"])

    hits = graph.search_by_embedding(
        query_emb, ctx, top_k=3, extractor_types=("episodic_memory_generator",),
    )
    assert len(hits) >= 1
    # The VectorIndex path uses the SeedHit's score directly; for an
    # InMemoryVectorIndex on identical vectors this is ~1.0.
    assert hits[0].node.node_id == _ext_id(0)
    assert hits[0].score > 0.99
    # Provenance lifted from ExtractedNode.extracted_content (the
    # same keys the legacy store path used).
    assert hits[0].video_id == "vid_0"
    assert hits[0].clip_id == 0


def test_search_routes_through_vector_index_filter(tmp_path: Path):
    """The ``ClipScope.video_id`` filter is pushed down to the
    VectorIndex's ``_build_where`` so cross-video bleed is impossible
    at the index level. Three clips across two videos, query for
    video B must return only video B hits."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    lines = []
    for vid, cid in (("vid_a", 0), ("vid_a", 1), ("vid_b", 0)):
        lines.append(json.dumps({
            "video_id": vid,
            "clip_id": cid,
            "video_raw_node_id": f"vid_raw_{vid}_{cid}".ljust(32, "x"),
            "extracted_node_id": f"ext_{vid}_{cid}".ljust(32, "x"),
            "description": f"{vid} clip {cid}",
        }))
    episodic_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vector_index = InMemoryVectorIndex(dimension=8)
    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _make_ctx()
    from core.graph_models import create_extracted_node
    for line in lines:
        item = json.loads(line)
        node = create_extracted_node(
            source_raw_id=item["video_raw_node_id"],
            extractor_type="episodic_memory_generator", extraction_schema="m3_bench_v1",
            account_id=ctx.account_id, node_id=item["extracted_node_id"],
            extracted_content={"source_video_id": item["video_id"], "source_clip_id": item["clip_id"]},
        )
        graph._store.create_node(node.serialize(), ctx)

    from m3.index_episodic import parse_args, run
    run(parse_args(_make_args(episodic_path, index_path)),
        graph=graph, vector_index=vector_index)

    # Issue a search scoped to vid_b — must NOT return vid_a hits
    # even though the underlying index contains both. The
    # VectorIndex ``_build_where`` clause is the safety net here:
    # a Python bug in the searcher's scope handling would silently
    # leak cross-video clips, and the M3-Bench scoring harness
    # would credit wrong-clip answers as correct.
    from service.graph_service import ClipScope
    record = vector_index.get_record(IndexRecord.generate_id(
        f"m3://accounts/{M3BENCH_ACCOUNT_ID}/videos/vid_b/clips/0/episodic", 2,
    ))
    query_emb = record.metadata["_embedding"]
    hits = graph.search_by_embedding(
        query_emb, ctx,
        top_k=10,
        scope=ClipScope(video_id="vid_b"),
    )
    assert hits, "no hits returned for vid_b query"
    for hit in hits:
        assert hit.video_id == "vid_b", (
            f"ClipScope(video_id='vid_b') leaked a {hit.video_id} hit"
        )
