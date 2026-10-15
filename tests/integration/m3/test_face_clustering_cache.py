"""Identity clustering uses real graph persistence; no model services."""

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_extracted_node
from m3 import face_clustering as module
from tests.integration.m3.test_voice_asr_resume import cache_graph  # noqa: F401


def seed_face(graph, ctx, clip, vector, confidence=0.9):
    raw = graph.create_raw_node(
        raw_type=RawDataType.VIDEO, content="", ctx=ctx,
        metadata={"video_id": "v", "clip_id": clip},
    )
    node = create_extracted_node(
        source_raw_id=raw.node_id, extractor_type="face_detection",
        extraction_schema="face_v1", account_id=ctx.account_id,
        extracted_content={"embedding": vector, "confidence": confidence,
                           "source_video_id": "v", "source_clip_id": clip},
    )
    graph._store.create_node(node.serialize(), ctx)
    return graph._store.get_node(node.node_id, ctx)


def test_shared_cache_and_edge_repair(cache_graph, tmp_path, monkeypatch):  # noqa: F811
    graph, ctx = cache_graph
    a = seed_face(graph, ctx, 0, [1., 0.], 1.)
    b = seed_face(graph, ctx, 1, [0.99, 0.01])
    argv = ["--account-id", ctx.account_id, "--cache-dir", str(tmp_path / "cache")]
    args = module.parse_args(argv + ["--output-jsonl", str(tmp_path / "v1" / "faces.jsonl")])
    def no_neighbors(*args, **kwargs):
        pytest.fail("dedup must not issue per-node neighbor queries")
    monkeypatch.setattr(graph, "get_neighbors", no_neighbors)
    assert module.run_face_clustering(args, graph=graph) == 0
    assert graph.list_identity_pairs([b.node_id], ctx) == [(b.node_id, a.node_id)]
    ranks = [graph._store.get_node(n.node_id, ctx).pagerank for n in (a, b)]
    def no_compute(*args, **kwargs):
        pytest.fail("same account/input should reuse clustering across video directories")
    monkeypatch.setattr(module, "cluster_faces", no_compute)
    args.output_jsonl = str(tmp_path / "v2" / "faces.jsonl")
    assert module.run_face_clustering(args, graph=graph) == 0
    summary = json.loads((tmp_path / "v2" / "faces.summary.json").read_text())
    assert summary["cache_hit"] and summary["emitted_edges"] == 0
    assert ranks == [graph._store.get_node(n.node_id, ctx).pagerank for n in (a, b)]
    # A valid computation cache never masks missing DB relations.
    edges = graph._store.get_neighbors(b.node_id, ctx, direction="outgoing")
    for _, edge, _ in edges:
        graph.delete_edge(edge.edge_id, ctx)
    assert module.run_face_clustering(args, graph=graph) == 0
    assert graph.list_identity_pairs([b.node_id], ctx) == [(b.node_id, a.node_id)]


def test_batch_noop_and_account_isolation(cache_graph):  # noqa: F811
    graph, ctx = cache_graph
    a = seed_face(graph, ctx, 0, [1., 0.])
    b = seed_face(graph, ctx, 1, [1., 0.])
    other = replace(ctx, account_id="other-" + uuid4().hex)
    assert graph.ensure_identity_edges([(b.node_id, a.node_id, 0.9)] * 2, ctx) == 1
    ranks = [graph._store.get_node(n.node_id, ctx).pagerank for n in (a, b)]
    assert graph.ensure_identity_edges([(b.node_id, a.node_id, 0.1)], ctx) == 0
    assert graph.list_identity_pairs([b.node_id], other) == []
    with pytest.raises(ValueError, match="endpoint"):
        graph.ensure_identity_edges([(b.node_id, a.node_id, 0.9)], other)
    with pytest.raises(ValueError, match="endpoint"):
        graph.ensure_identity_edges([(b.node_id, str(uuid4()), 0.9)], ctx)
    assert ranks == [graph._store.get_node(n.node_id, ctx).pagerank for n in (a, b)]


def test_batch_multiple_pages_and_atomic_validation(cache_graph):  # noqa: F811
    graph, ctx = cache_graph
    anchor = seed_face(graph, ctx, 0, [1., 0.])
    members = [
        graph.create_raw_node(raw_type=RawDataType.VIDEO, content="", ctx=ctx)
        for _ in range(501)
    ]
    pairs = [(n.node_id, anchor.node_id, 0.9) for n in members]
    with pytest.raises(ValueError, match="endpoint"):
        graph.ensure_identity_edges(pairs + [(str(uuid4()), anchor.node_id, 0.9)], ctx)
    assert graph.list_identity_pairs([n.node_id for n in members], ctx) == []
    assert graph.ensure_identity_edges(pairs, ctx) == 501
    rank = graph._store.get_node(anchor.node_id, ctx).pagerank
    assert len(graph.list_identity_pairs([n.node_id for n in members], ctx)) == 501
    assert graph.ensure_identity_edges(pairs, ctx) == 0
    assert graph._store.get_node(anchor.node_id, ctx).pagerank == rank


def test_empty_faces_replace_old_summary(cache_graph, tmp_path):  # noqa: F811
    graph, ctx = cache_graph
    output = tmp_path / "faces.jsonl"
    summary = output.with_suffix(".summary.json")
    summary.write_text('{"faces": 100, "cache_hit": true}')
    args = module.parse_args(["--account-id", ctx.account_id, "--output-jsonl", str(output)])
    assert module.run_face_clustering(args, graph=graph) == 0
    receipt = json.loads(summary.read_text())
    assert receipt["faces"] == 0 and not receipt["cache_hit"]
    assert output.read_text() == ""


@pytest.mark.parametrize("change", ["threshold", "embedding", "account", "corrupt"])
def test_computation_cache_invalidation(tmp_path, monkeypatch, change):
    faces = [
        module.FaceRecord("a", [1., 0.], 0.9, [], "v", 0),
        module.FaceRecord("b", [0.9, 0.1], 0.8, [], "v", 1),
    ]
    _, hit = module._cached_clusters(faces, 0.4, "a", tmp_path)
    assert not hit
    threshold, account = 0.4, "a"
    if change == "threshold":
        threshold = 0.2
    elif change == "account":
        account = "b"
    elif change == "embedding":
        faces[1].embedding = [0., 1.]
    else:
        next(tmp_path.glob("*.json")).write_text("{broken")
    calls = []
    original = module.cluster_faces
    def compute(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "cluster_faces", compute)
    _, hit = module._cached_clusters(faces, threshold, account, tmp_path)
    assert not hit and len(calls) == 1


def test_face_lookup_is_uncapped(monkeypatch):
    from core.models import RequestContext
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService

    graph = GraphService(MemoryGraphStore())
    ctx = RequestContext(account_id="a", user_id="u", agent_id="a", session_id="s", trace_id="t")
    # Pin both the type filter and removal of the old 4096-node cap.
    def list_faces(kind, context, limit):
        assert kind == "face_detection" and context is ctx and limit is None
        return []
    monkeypatch.setattr(graph, "list_extracted_nodes", list_faces)
    assert module._face_records_from_graph(graph, ctx) == []
