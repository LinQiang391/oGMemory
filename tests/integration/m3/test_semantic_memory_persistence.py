"""Persistence contract: strict offline store plus opt-in real PostgreSQL.

SQL cases require OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN pointing to a
dedicated disposable test database. No fallback to the experiment GRAPH_DSN.
Each test uses fresh account IDs and removes only its own nodes.
"""

import json
import os
from uuid import uuid4
from unittest.mock import Mock

import pytest

from core.graph_enums import GraphEdgeType, GraphNodeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.pipeline.context import PipelineContext
from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import PipelineSpec, StageSpec
from m3.pipeline.stages import semantic_memory_consolidator_stage
from m3.semantic_memory_consolidator import (
    consolidate, parse_args, run_semantic_memory_consolidator,
)
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


class StrictMemoryStore(MemoryGraphStore):
    """Reject duplicate IDs/edges instead of hiding accidental INSERTs."""

    def create_node(self, payload, ctx):
        if payload["node_id"] in self._nodes:
            raise ValueError("duplicate node primary key")
        return super().create_node(payload, ctx)

    def create_edge(self, edge, ctx):
        if any(
            (e.source_id, e.target_id, e.edge_type)
            == (edge.source_id, edge.target_id, edge.edge_type)
            for e in self._edges.values()
        ):
            raise ValueError("duplicate edge")
        return super().create_edge(edge, ctx)


@pytest.fixture(params=["strict-memory", pytest.param("sql", marks=pytest.mark.integration)])
def graph(request):
    if request.param == "sql":
        dsn = os.environ.get("OGMEM_TEST_DSN")
        if os.environ.get("OGMEM_SQL_INTEGRATION") != "1" or not dsn:
            pytest.skip("Requires opt-in OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN")
        from providers.graph_store import SQLGraphStore
        store = SQLGraphStore(connection_string=dsn)
    else:
        store = StrictMemoryStore()
    return GraphService(store)


@pytest.fixture
def contexts(graph):
    owned = []

    def make():
        ctx = RequestContext(
            account_id=f"semantic_test_{uuid4().hex}", user_id="test",
            agent_id="test", session_id="test", trace_id="test",
        )
        owned.append(ctx)
        return ctx

    yield make
    for ctx in owned:
        for node in graph._store.list_nodes(None, ctx):
            graph.delete_node(node.node_id, ctx)


def seed(graph, ctx, video="target", clip=0, description="original"):
    node = create_extracted_node(
        source_raw_id=str(uuid4()), extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1", account_id=ctx.account_id,
        extracted_content={
            "source_video_id": video, "source_clip_id": clip,
            "description": description,
        },
        node_id=str(uuid4()),
    )
    graph._store.create_node(node.serialize(), ctx)
    return node


def args_for(tmp_path, ctx, video="target"):
    argv = ["--account-id", ctx.account_id,
            "--output-jsonl", str(tmp_path / "semantic.jsonl")]
    if video:
        argv += ["--video", video]
    return parse_args(argv)


def read_rows(args):
    from pathlib import Path
    return [json.loads(line) for line in Path(args.output_jsonl).read_text().splitlines()]


def read_summary(args):
    from pathlib import Path
    return json.loads(Path(args.output_jsonl).with_suffix(".summary.json").read_text())


def edges(graph, ctx, memory_id):
    return [
        edge for _, edge, _ in graph.get_neighbors(
            memory_id, ctx, direction="incoming",
            edge_types=[GraphEdgeType.SUPPORTS, GraphEdgeType.EVOLVES],
        )
    ]


def test_rerun_updates_content_and_reconciles_changed_sources(graph, contexts, tmp_path):
    ctx = contexts()
    first = seed(graph, ctx)
    retired = seed(graph, ctx, clip=1)
    args = args_for(tmp_path, ctx)
    assert run_semantic_memory_consolidator(args, graph) == 0
    memory_id = read_rows(args)[0]["memory_node_id"]
    created_at = graph.get_memory_node(memory_id, ctx).created_at
    original_edges = {e.edge_id for e in edges(graph, ctx, memory_id)}
    assert len(original_edges) == 4

    assert run_semantic_memory_consolidator(args, graph) == 0
    assert read_rows(args)[0]["evolves_edge_count"] == 0
    assert {e.edge_id for e in edges(graph, ctx, memory_id)} == original_edges
    assert len(graph._store.list_nodes(GraphNodeType.MEMORY, ctx)) == 1

    first.extracted_content["description"] = "corrected evidence"
    graph._store.update_node(first.serialize(), ctx)
    # Keep this node alive so stale edges cannot disappear by SQL cascade.
    retired.extracted_content["source_video_id"] = "retired_video"
    graph._store.update_node(retired.serialize(), ctx)
    added = seed(graph, ctx, clip=2, description="new evidence")
    assert run_semantic_memory_consolidator(args, graph) == 0
    row = read_rows(args)[0]
    assert row["memory_node_id"] == memory_id
    assert row["evolves_edge_count"] == 1
    memory = graph.get_memory_node(memory_id, ctx)
    assert memory.created_at == created_at
    assert memory.supporting_node_ids == [first.node_id, added.node_id]
    assert json.loads(memory.content)["description"] == "corrected evidence\n\nnew evidence"
    provenance = edges(graph, ctx, memory_id)
    assert len(provenance) == 4
    assert {(e.source_id, e.edge_type) for e in provenance} == {
        (n.node_id, kind) for n in (first, added)
        for kind in (GraphEdgeType.SUPPORTS, GraphEdgeType.EVOLVES)
    }


def test_update_failure_keeps_old_memory_and_fails_stage(
    graph, contexts, tmp_path, monkeypatch,
):
    ctx = contexts()
    node = seed(graph, ctx)
    args = args_for(tmp_path, ctx)
    assert run_semantic_memory_consolidator(args, graph) == 0
    memory_id = read_rows(args)[0]["memory_node_id"]
    before = graph.get_memory_node(memory_id, ctx).serialize()
    edge_ids = {e.edge_id for e in edges(graph, ctx, memory_id)}
    node.extracted_content["description"] = "changed"
    graph._store.update_node(node.serialize(), ctx)

    def fail(*a, **kw):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(graph._store, "update_node", fail)
    pipe_ctx = PipelineContext(
        video=tmp_path / "video.mp4", video_id="target", seconds=0,
        work_dir=tmp_path, config_path=tmp_path / "unused.yaml",
        graph_store="memory", connection_string=None, account_id=ctx.account_id,
        artifacts={}, variables={}, graph=graph,
    )
    stage = StageSpec(
        id="semantic", type="m3.semantic_memory_consolidator",
        params={"output_jsonl": args.output_jsonl},
    )
    downstream = Mock(side_effect=AssertionError("must not run after failure"))
    runner = PipelineRunner(registry={
        stage.type: semantic_memory_consolidator_stage, "test.downstream": downstream,
    }, probes={})
    summary = runner.run(PipelineSpec(name="failure-test", version=1, stages=[
        stage, StageSpec(id="next", type="test.downstream", needs=["semantic"]),
    ]), pipe_ctx)
    result = pipe_ctx.stage_results["semantic"]
    downstream.assert_not_called()
    assert summary["stages"]["semantic"]["ok"] is False
    saved_summary = json.loads((tmp_path / "pipeline_summary.json").read_text())
    assert saved_summary["stages"]["semantic"]["errors"] == result.errors
    assert result.ok is False
    assert result.to_dict()["status"] == "failed"
    assert result.counts["failed_videos"] == 1
    assert result.errors[0]["message"] == "injected update failure"
    assert read_rows(args) == []
    assert read_summary(args)["status"] == "failed"
    assert graph.get_memory_node(memory_id, ctx).serialize() == before
    assert {e.edge_id for e in edges(graph, ctx, memory_id)} == edge_ids


@pytest.mark.parametrize("failed_kind", [GraphEdgeType.SUPPORTS, GraphEdgeType.EVOLVES])
def test_partial_edge_failure_is_retryable(
    graph, contexts, tmp_path, monkeypatch, failed_kind,
):
    ctx = contexts()
    seed(graph, ctx)
    second = seed(graph, ctx, clip=1)
    args = args_for(tmp_path, ctx)
    real_create = graph._store.create_edge

    def fail_second(edge, ctx):
        if edge.source_id == second.node_id and edge.edge_type == failed_kind:
            raise RuntimeError("injected edge failure")
        return real_create(edge, ctx)

    monkeypatch.setattr(graph._store, "create_edge", fail_second)
    assert run_semantic_memory_consolidator(args, graph) == 2
    assert read_rows(args) == []
    assert read_summary(args)["errors"][0]["message"] == "injected edge failure"
    monkeypatch.setattr(graph._store, "create_edge", real_create)
    assert run_semantic_memory_consolidator(args, graph) == 0
    row = read_rows(args)[0]
    assert row["evolves_edge_count"] == (2 if failed_kind == GraphEdgeType.SUPPORTS else 1)
    assert len(edges(graph, ctx, row["memory_node_id"])) == 4
    assert len(graph._store.list_nodes(GraphNodeType.MEMORY, ctx)) == 1
    assert read_summary(args)["failed_videos"] == 0


def test_partial_create_failure_reports_failed_video(graph, contexts, tmp_path, monkeypatch):
    ctx = contexts()
    seed(graph, ctx, video="good")
    seed(graph, ctx, video="bad")
    args = args_for(tmp_path, ctx, video=None)
    create = graph.create_memory_node

    def fail_bad(**kw):
        if kw["topic"] == "semantic:bad":
            raise RuntimeError("injected insert failure")
        return create(**kw)

    monkeypatch.setattr(graph, "create_memory_node", fail_bad)
    assert run_semantic_memory_consolidator(args, graph) == 2
    assert [r["video_id"] for r in read_rows(args)] == ["good"]
    summary = read_summary(args)
    assert summary["completed_videos"] == summary["failed_videos"] == 1
    assert summary["errors"][0]["video_id"] == "bad"
    monkeypatch.setattr(graph, "create_memory_node", create)
    assert run_semantic_memory_consolidator(args, graph) == 0
    counts = {r["video_id"]: r["evolves_edge_count"] for r in read_rows(args)}
    assert counts == {"good": 0, "bad": 1}


def test_edge_counts_are_per_video_not_cumulative(graph, contexts, tmp_path):
    ctx = contexts()
    seed(graph, ctx, video="one")
    seed(graph, ctx, video="one", clip=1)
    seed(graph, ctx, video="two")
    args = args_for(tmp_path, ctx, video=None)
    assert run_semantic_memory_consolidator(args, graph) == 0
    counts = {r["video_id"]: r["evolves_edge_count"] for r in read_rows(args)}
    assert counts == {"one": 2, "two": 1}
    assert read_summary(args)["evolves_edges_emitted"] == 3


def test_same_video_in_two_accounts_has_distinct_memories(graph, contexts, tmp_path):
    left, right = contexts(), contexts()
    left_source = seed(graph, left, description="left only")
    right_source = seed(graph, right, description="right only")
    memory_ids = []
    for ctx, source in ((left, left_source), (right, right_source)):
        args = args_for(tmp_path, ctx)
        assert run_semantic_memory_consolidator(args, graph) == 0
        memory_id = read_rows(args)[0]["memory_node_id"]
        memory_ids.append(memory_id)
        memory = graph.get_memory_node(memory_id, ctx)
        assert memory.account_id == ctx.account_id
        assert memory.supporting_node_ids == [source.node_id]
    assert memory_ids[0] != memory_ids[1]
    assert graph.get_memory_node(memory_ids[0], left).supporting_node_ids == [left_source.node_id]


def test_legacy_custom_account_memory_is_updated_in_place(graph, contexts, tmp_path):
    ctx = contexts()
    source = seed(graph, ctx)
    video = f"legacy_{uuid4().hex}"
    source.extracted_content["source_video_id"] = video
    graph._store.update_node(source.serialize(), ctx)
    legacy_id = consolidate([], video).memory_node_id
    graph.create_memory_node(
        summary="legacy", ctx=ctx, topic=f"semantic:{video}", node_id=legacy_id,
        supporting_node_ids=[source.node_id],
    )
    args = args_for(tmp_path, ctx, video)
    assert run_semantic_memory_consolidator(args, graph) == 0
    assert graph.get_memory_node(legacy_id, ctx).summary == "original"
    assert len(graph._store.list_nodes(GraphNodeType.MEMORY, ctx)) == 1
    assert len(edges(graph, ctx, legacy_id)) == 2


def test_legacy_memory_in_another_account_is_not_adopted(graph, contexts, tmp_path):
    owner, other = contexts(), contexts()
    video = f"foreign_legacy_{uuid4().hex}"
    legacy_id = consolidate([], video).memory_node_id
    graph.create_memory_node(
        summary="keep original", ctx=owner, topic=f"semantic:{video}", node_id=legacy_id,
    )
    source = seed(graph, other, video=video)
    args = args_for(tmp_path, other, video)
    assert run_semantic_memory_consolidator(args, graph) == 0
    assert read_rows(args)[0]["memory_node_id"] != legacy_id
    assert graph.get_memory_node(legacy_id, owner).summary == "keep original"
    own_id = consolidate([], video, other.account_id).memory_node_id
    assert graph.get_memory_node(own_id, other).supporting_node_ids == [source.node_id]


def test_default_account_keeps_legacy_id():
    import hashlib
    assert consolidate([], "target", M3BENCH_ACCOUNT_ID).memory_node_id == (
        hashlib.sha1(b"semantic/target").hexdigest()[:32]
    )
