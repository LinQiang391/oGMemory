"""Tests that m3.index_episodic mirrors embeddings into the graph.

Before this fix, the index_episodic stage wrote embeddings only to
episodic_index.jsonl — the JSONL was the only place vectors lived, so
GraphService.search_by_embedding returned nothing for real pipelines.
The driver now writes each record's embedding back to its
corresponding ExtractedNode via set_embedding + update_extracted_node.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path

import pytest

from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


def _make_ctx(account_id: str = M3BENCH_ACCOUNT_ID) -> RequestContext:
    return RequestContext(
        account_id=account_id, user_id="u", agent_id="a",
        session_id="m3bench-index-test", trace_id="t-test",
    )


def _ext_id(i: int) -> str:
    """Stable 32-char id for the i-th test record.

    ``f"ext-{i:032d}"[:32]`` collides for small i because the prefix
    is shared and the int format pads to 32 chars (truncated). Use a
    fixed-width suffix instead so each record gets a distinct id.
    """
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
    """Pre-populate the graph with `n` ExtractedNodes so index_episodic
    has live nodes to write embeddings into. Returns the inserted ids."""
    ids = []
    for i in range(n):
        ext_id = _ext_id(i)
        from core.graph_models import create_extracted_node
        node = create_extracted_node(
            source_raw_id=_vid_id(i),
            extractor_type="episodic_memory_generator",
            extraction_schema="m3_bench_v1",
            account_id=ctx.account_id,
            extracted_content={"description": f"Episode {i}"},
            node_id=ext_id,
        )
        graph._store.create_node(node.serialize(), ctx)
        ids.append(ext_id)
    return ids


def test_run_persists_embeddings_to_graph(tmp_path: Path):
    """End-to-end: write episodic_memory.jsonl + seed graph → run the
    CLI driver → assert each ExtractedNode now has its embedding."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    graph = GraphService(MemoryGraphStore())
    ctx = _make_ctx()
    seeded_ids = _seed_graph(graph, ctx, n=3)

    from m3.index_episodic import parse_args, run
    rc = run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--graph-store", "memory",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph)
    assert rc == 0, f"run() returned {rc}"

    # Each seeded ExtractedNode must now have a non-empty embedding.
    for ext_id in seeded_ids:
        node = graph.get_extracted_node(ext_id, ctx)
        assert node is not None, f"ExtractedNode {ext_id} missing"
        emb = node.get_embedding()
        assert emb is not None, f"ExtractedNode {ext_id} has no embedding"
        assert len(emb) == 16, f"unexpected embedding dim {len(emb)} for {ext_id}"

    # And the JSONL index must still be written (the change is additive,
    # not a replacement of the file).
    assert index_path.exists()
    records = [json.loads(l) for l in index_path.read_text().splitlines() if l]
    assert len(records) == 3
    for rec in records:
        assert len(rec["embedding"]) == 16


def test_run_skips_extracted_nodes_missing_from_graph(tmp_path: Path):
    """If the JSONL mentions an extracted_node_id that isn't in the
    graph, the driver must skip it (log + continue) rather than crash
    on the first missing row."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    graph = GraphService(MemoryGraphStore())
    ctx = _make_ctx()
    # Seed only 1 of the 3 expected nodes.
    seeded_ids = _seed_graph(graph, ctx, n=1)

    from m3.index_episodic import parse_args, run
    rc = run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--graph-store", "memory",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph)
    assert rc == 0

    # Only the seeded one carries an embedding; the others remain None.
    seeded = graph.get_extracted_node(seeded_ids[0], ctx)
    assert seeded.get_embedding() is not None
    missing = graph.get_extracted_node(_ext_id(1), ctx)
    # We seeded only the first node, so the others are not in the graph
    # at all — ``get_extracted_node`` returns None for missing nodes.
    assert missing is None
    missing2 = graph.get_extracted_node(_ext_id(2), ctx)
    assert missing2 is None


def test_run_without_injected_graph_still_writes_jsonl(tmp_path: Path):
    """Backwards compat: passing graph=None (legacy CLI behaviour)
    must still produce a valid JSONL index. The embeddings just won't
    be mirrored anywhere."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=2)

    from m3.index_episodic import parse_args, run
    rc = run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--graph-store", "memory",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]))  # graph=None intentionally
    assert rc == 0
    assert index_path.exists()
    records = [json.loads(l) for l in index_path.read_text().splitlines() if l]
    assert len(records) == 2


def test_indexed_embeddings_are_searchable_via_graph_service(tmp_path: Path):
    """The whole point of the change: after index_episodic runs,
    GraphService.search_by_embedding must find the indexed
    ExtractedNodes by vector similarity."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    graph = GraphService(MemoryGraphStore())
    ctx = _make_ctx()
    seeded_ids = _seed_graph(graph, ctx, n=3)

    from m3.index_episodic import parse_args, run
    run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--graph-store", "memory",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph)

    # Search by one of the indexed embeddings — it should come back
    # as a hit with score ~1.0 (cosine of identical vectors).
    target = graph.get_extracted_node(seeded_ids[0], ctx)
    query_emb = list(target.get_embedding())
    hits = graph.search_by_embedding(
        query_emb, ctx, top_k=3, extractor_types=("episodic_memory_generator",),
    )
    assert len(hits) >= 1
    assert hits[0].node.node_id == seeded_ids[0]
    assert hits[0].score > 0.99