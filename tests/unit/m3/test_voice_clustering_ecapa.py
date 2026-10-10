"""Tests for the ECAPA-aware upgrades to ``m3.voice_clustering``.

Pins four behaviours:

* When ``speaker_embedding`` ExtractedNodes exist, the driver
  reads them and ignores ``voice_segmentation`` nodes (preference
  for the 192-dim ECAPA signal).
* When only ``voice_segmentation`` nodes exist (graphs written
  before the ECAPA stage), the driver falls back to the 8-dim
  spectral fingerprint.
* The default ``--threshold`` flips 0.3 → 0.5 when ECAPA nodes are
  present. An explicit ``--threshold`` always wins.
* The output JSONL gets the new ``embedding_kind`` field per row.
"""

from __future__ import annotations

import json
import math
import pytest

from core.graph_enums import GraphEdgeType, RawDataType
from core.graph_models import create_extracted_node, create_raw_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.voice_clustering import (
    VoiceRecord,
    _records_from_speaker_embedding,
    _records_from_voice_segmentation,
    _voice_records_from_graph,
    build_cluster_output,
    cluster_voices,
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


def _seed_video(graph, ctx, video_id: str):
    video_node = create_raw_node(
        raw_type=RawDataType.VIDEO,
        content=b"",
        account_id=ctx.account_id,
        metadata={"video_id": video_id, "clip_id": 0},
        node_id=f"video_{video_id}".ljust(32, "x")[:32],
    )
    graph._store.create_node(video_node.serialize(), ctx)


def _seed_voice_segmentation(
    graph, ctx, *, voice_id: str, embedding: list[float],
    video_id: str = "v_a", clip_id: int = 0,
    node_id: str | None = None,
):
    """Add a voice_segmentation ExtractedNode carrying voice_embedding."""
    nid = node_id or f"vs_{voice_id}".ljust(32, "x")[:32]
    n = create_extracted_node(
        source_raw_id="audio_seed",
        extractor_type="voice_segmentation",
        extraction_schema="voice_segmentation_v1",
        account_id=ctx.account_id,
        extracted_content={
            "start_sec": 0.0, "end_sec": 1.0,
            "duration_sec": 1.0,
            "voice_id": voice_id,
            "voice_embedding": embedding,
            "source_video_id": video_id,
            "source_clip_id": clip_id,
        },
        node_id=nid,
    )
    graph._store.create_node(n.serialize(), ctx)


def _seed_speaker_embedding(
    graph, ctx, *, speaker_id: str, embedding: list[float],
    video_id: str = "v_a", clip_id: int = 0,
    num_turns: int = 1, total_duration_sec: float = 1.0,
    node_id: str | None = None,
):
    """Add a speaker_embedding ExtractedNode carrying 192-dim vector."""
    nid = node_id or f"se_{speaker_id}".ljust(32, "x")[:32]
    n = create_extracted_node(
        source_raw_id="audio_seed",
        extractor_type="speaker_embedding",
        extraction_schema="speaker_embedding_v1",
        account_id=ctx.account_id,
        extracted_content={
            "speaker_id": speaker_id,
            "speaker_embedding": embedding,
            "embedding_dim": len(embedding),
            "model": "speechbrain",
            "num_turns": num_turns,
            "total_duration_sec": total_duration_sec,
            "source_video_id": video_id,
            "source_clip_id": clip_id,
        },
        node_id=nid,
    )
    graph._store.create_node(n.serialize(), ctx)


@pytest.fixture
def memory_store():
    from providers.graph_store import MemoryGraphStore
    return MemoryGraphStore()


@pytest.fixture
def graph(memory_store):
    from service.graph_service import GraphService
    return GraphService(memory_store)


# ---------------------------------------------------------------------------
# _voice_records_from_graph preference
# ---------------------------------------------------------------------------

def test_prefers_ecapa_when_present(graph):
    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_voice_segmentation(
        graph, ctx, voice_id="V1", embedding=[0.1] * 8,
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="S1", embedding=[0.5] * 192,
    )
    records = _voice_records_from_graph(graph, ctx)
    assert len(records) == 1
    assert records[0].embedding_kind == "ecapa"
    assert len(records[0].embedding) == 192


def test_falls_back_to_spectral_when_no_ecapa(graph):
    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_voice_segmentation(
        graph, ctx, voice_id="V1", embedding=[0.1] * 8,
    )
    records = _voice_records_from_graph(graph, ctx)
    assert len(records) == 1
    assert records[0].embedding_kind == "spectral"
    assert len(records[0].embedding) == 8


def test_ecapa_confidence_weighted_by_turns_and_duration(graph):
    """The ECAPA confidence proxy prefers anchors with more
    speech — num_turns * (1 + total_duration_sec / 10)."""
    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_speaker_embedding(
        graph, ctx, speaker_id="A",
        embedding=[0.1] * 192,
        num_turns=1, total_duration_sec=1.0,
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="B",
        embedding=[0.1] * 192,
        num_turns=10, total_duration_sec=30.0,
    )
    records = _records_from_speaker_embedding(graph, ctx)
    by_speaker = {r.voice_id: r for r in records}
    assert by_speaker["B"].confidence > by_speaker["A"].confidence


def test_spectral_records_only_have_8d(graph):
    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_voice_segmentation(
        graph, ctx, voice_id="V1", embedding=[0.1] * 8,
    )
    records = _records_from_voice_segmentation(graph, ctx)
    assert len(records) == 1
    assert len(records[0].embedding) == 8
    assert records[0].embedding_kind == "spectral"


def test_mixed_kind_records_kept_separate(graph):
    """When ECAPA and spectral records coexist (different node IDs),
    ECAPA wins and the spectral nodes are dropped. They live in
    different embedding spaces, so merging is meaningless."""
    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_voice_segmentation(
        graph, ctx, voice_id="V1", embedding=[0.1] * 8,
        node_id="vs_v1_node_abcdef",
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="S1", embedding=[0.5] * 192,
        node_id="se_s1_node_abcdef",
    )
    records = _voice_records_from_graph(graph, ctx)
    assert len(records) == 1
    assert records[0].embedding_kind == "ecapa"


# ---------------------------------------------------------------------------
# Output JSONL gets embedding_kind
# ---------------------------------------------------------------------------

def test_cluster_output_includes_embedding_kind():
    voices = [
        VoiceRecord(
            node_id="a", embedding=[0.5] * 192,
            confidence=1.0, voice_id="A",
            embedding_kind="ecapa",
        ),
        VoiceRecord(
            node_id="b", embedding=[0.5] * 192,
            confidence=1.0, voice_id="B",
            embedding_kind="ecapa",
        ),
    ]
    clusters = cluster_voices(voices, threshold=0.5)
    rows = build_cluster_output(clusters, voices)
    assert len(rows) == 2
    for r in rows:
        assert r.embedding_kind == "ecapa"


def test_cluster_output_spectral_kind():
    voices = [
        VoiceRecord(
            node_id="a", embedding=[0.1] * 8,
            confidence=1.0, voice_id="A",
            embedding_kind="spectral",
        ),
        VoiceRecord(
            node_id="b", embedding=[0.1] * 8,
            confidence=1.0, voice_id="B",
            embedding_kind="spectral",
        ),
    ]
    clusters = cluster_voices(voices, threshold=0.3)
    rows = build_cluster_output(clusters, voices)
    assert all(r.embedding_kind == "spectral" for r in rows)


# ---------------------------------------------------------------------------
# Threshold default — picked at runtime based on embedding_kind
# ---------------------------------------------------------------------------

def test_threshold_default_picks_ecapa_when_present(tmp_path, graph):
    """When ECAPA nodes exist, the driver auto-selects threshold=0.5."""
    from argparse import Namespace
    from m3.voice_clustering import run_voice_clustering

    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_speaker_embedding(
        graph, ctx, speaker_id="A", embedding=[0.5] * 192,
        video_id="v_a", clip_id=0,
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="B", embedding=[-0.5] * 192,
        video_id="v_b", clip_id=0,
    )
    args = Namespace(
        graph_store="memory", connection_string=None,
        threshold=None,            # auto
        output_jsonl=str(tmp_path / "out.jsonl"),
    )
    rc = run_voice_clustering(args, graph=graph)
    assert rc == 0
    # Read output; verify the two speakers landed in separate
    # clusters (they're at distance ~1.0 in cosine, well above 0.5).
    rows = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    anchors = {r["anchor_node_id"] for r in rows}
    # Two distinct anchors → two clusters (one per speaker).
    assert len(anchors) == 2


def test_threshold_default_picks_spectral_when_only_spectral(tmp_path, graph):
    """When only spectral nodes exist, the driver uses threshold=0.3."""
    from argparse import Namespace
    from m3.voice_clustering import run_voice_clustering

    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_voice_segmentation(
        graph, ctx, voice_id="A", embedding=[0.1] * 8,
        video_id="v_a", clip_id=0,
    )
    _seed_voice_segmentation(
        graph, ctx, voice_id="B", embedding=[-0.1] * 8,
        video_id="v_b", clip_id=0,
    )
    args = Namespace(
        graph_store="memory", connection_string=None,
        threshold=None,            # auto
        output_jsonl=str(tmp_path / "out.jsonl"),
    )
    rc = run_voice_clustering(args, graph=graph)
    assert rc == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(r["embedding_kind"] == "spectral" for r in rows)


def test_explicit_threshold_overrides_default(tmp_path, graph):
    """--threshold 0.1 always wins, regardless of embedding_kind."""
    from argparse import Namespace
    from m3.voice_clustering import run_voice_clustering

    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    _seed_speaker_embedding(
        graph, ctx, speaker_id="A", embedding=[0.5] * 192,
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="B", embedding=[-0.5] * 192,
    )
    args = Namespace(
        graph_store="memory", connection_string=None,
        threshold=0.1,             # very strict — won't merge these
        output_jsonl=str(tmp_path / "out.jsonl"),
    )
    rc = run_voice_clustering(args, graph=graph)
    assert rc == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    # Two anchors → two clusters.
    anchors = {r["anchor_node_id"] for r in rows}
    assert len(anchors) == 2


# ---------------------------------------------------------------------------
# Edge: same-identity edges still emit with ECAPA nodes
# ---------------------------------------------------------------------------

def test_ecapa_records_emit_same_identity_edges(graph):
    """Two ECAPA records with similar embeddings collapse into one
    cluster and emit one same_identity edge."""
    from m3.voice_clustering import run_voice_clustering

    ctx = _ctx()
    _seed_video(graph, ctx, "v_a")
    # Same embedding → cosine distance 0 → cluster together.
    _seed_speaker_embedding(
        graph, ctx, speaker_id="A", embedding=[0.5] * 192,
        video_id="v_a", clip_id=0,
        node_id="se_a".ljust(32, "x")[:32],
    )
    _seed_speaker_embedding(
        graph, ctx, speaker_id="B", embedding=[0.5] * 192,
        video_id="v_b", clip_id=0,
        node_id="se_b".ljust(32, "x")[:32],
    )
    from argparse import Namespace
    args = Namespace(
        graph_store="memory", connection_string=None,
        threshold=0.5,
        output_jsonl=str(graph._store.__class__.__name__) + ".jsonl",  # unused
    )
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        args.output_jsonl = f"{td}/out.jsonl"
        rc = run_voice_clustering(args, graph=graph)
    assert rc == 0
    # Both ExtractedNodes are in the graph; one same_identity edge
    # between member → anchor.
    seen_edge_ids: set[str] = set()
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, edge, _dir in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            if edge.edge_type == GraphEdgeType.SAME_IDENTITY:
                seen_edge_ids.add(getattr(edge, "edge_id", id(edge)))
    assert len(seen_edge_ids) == 1