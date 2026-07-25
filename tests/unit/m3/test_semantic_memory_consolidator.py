"""Tests for ``m3.semantic_memory_consolidator`` — per-video
memory consolidation.

Three layers:

* Pure consolidation — consolidate() merges descriptions,
  deduplicates characters / events / entities across clips, and
  produces a stable MemoryNode id. No I/O.

* Driver — run_semantic_memory_consolidator with a seeded graph.
  Asserts a MemoryNode lands in the graph per video with the
  expected content + topic + tags + supporting_node_ids, and that
  each source EpisrictedNode has an EVOLVES edge to the consolidated
  MemoryNode. Re-runs are idempotent.

* Pipeline stage — semantic_memory_consolidator_stage through
  PipelineRunner, asserting the stage counts match the driver
  output and edges land in the graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.semantic_memory_consolidator import (
    EpisodicItem,
    _normalize_character,
    _normalize_event,
    _normalize_entity,
    consolidate,
    run_semantic_memory_consolidator,
)
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-semantic-test", trace_id="t-test",
    )


def _item(node_id: str, video_id: str | None, clip_id: int | None,
          description: str,
          characters: list[dict] | None = None,
          events: list[dict] | None = None,
          entities: list[dict] | None = None) -> EpisodicItem:
    return EpisodicItem(
        node_id=node_id,
        video_id=video_id,
        clip_id=clip_id,
        description=description,
        characters=characters or [],
        events=events or [],
        entities=entities or [],
    )


# ---------------------------------------------------------------------------
# Pure helpers — _normalize_*
# ---------------------------------------------------------------------------

def test_normalize_character_lowercases_and_strips():
    assert _normalize_character({"name": "Alice", "role": "Owner"}) == \
        ("alice", "owner")
    assert _normalize_character({"name": "  ALICE ", "role": "OWNER"}) == \
        ("alice", "owner")
    assert _normalize_character({}) == ("", "")


def test_normalize_event_lowercases_type_and_summary():
    assert _normalize_event({"type": "Observation", "summary": "Walked in"}) == \
        ("observation", "walked in")


def test_normalize_entity_lowercases_name_and_type():
    assert _normalize_entity({"name": "Mug", "type": "Object"}) == \
        ("mug", "object")


# ---------------------------------------------------------------------------
# Pure consolidation
# ---------------------------------------------------------------------------

def test_consolidate_empty_input_returns_empty_memory():
    mem = consolidate([], "vid_a")
    assert mem.video_id == "vid_a"
    assert mem.memory_node_id  # stable hash
    assert mem.description == ""
    assert mem.characters == []
    assert mem.events == []
    assert mem.entities == []
    assert mem.source_node_ids == []


def test_consolidate_sorts_by_clip_id_for_stable_description_order():
    """Two items for the same video with different clip_ids must be
    concatenated in clip order, not in the order they were passed."""
    items = [
        _item("n_clip5", "v", 5, "Clip five description."),
        _item("n_clip0", "v", 0, "Clip zero description."),
        _item("n_clip3", "v", 3, "Clip three description."),
    ]
    mem = consolidate(items, "v")
    assert mem.description == (
        "Clip zero description.\n\n"
        "Clip three description.\n\n"
        "Clip five description."
    )


def test_consolidate_stable_memory_node_id_per_video():
    items = [_item("n1", "v_a", 0, "desc")]
    a = consolidate(items, "v_a")
    b = consolidate(items, "v_a")
    assert a.memory_node_id == b.memory_node_id
    # Different video → different id.
    c = consolidate(items, "v_b")
    assert a.memory_node_id != c.memory_node_id


def test_consolidate_dedups_characters_across_clips():
    items = [
        _item("n1", "v", 0, "clip 0", characters=[
            {"name": "Alice", "role": "Owner"},
            {"name": "Bob", "role": "Visitor"},
        ]),
        _item("n2", "v", 1, "clip 1", characters=[
            {"name": "Alice", "role": "owner"},  # same as clip 0 (lowercased)
            {"name": "Carol", "role": "Visitor"},
        ]),
    ]
    mem = consolidate(items, "v")
    names = sorted(c["name"] for c in mem.characters)
    assert names == ["Alice", "Bob", "Carol"]
    # Dedup picks the FIRST occurrence (clip 0's Alice).
    alice = [c for c in mem.characters if c["name"] == "Alice"][0]
    assert alice["role"] == "Owner"  # capitalized from clip 0


def test_consolidate_dedups_events_across_clips():
    items = [
        _item("n1", "v", 0, "c0", events=[
            {"type": "entry", "summary": "Woman walked in"},
        ]),
        _item("n2", "v", 1, "c1", events=[
            {"type": "Entry", "summary": "WOMAN WALKED IN"},  # same, different case
            {"type": "exit", "summary": "Woman left"},
        ]),
    ]
    mem = consolidate(items, "v")
    assert len(mem.events) == 2
    summaries = sorted(e["summary"].lower() for e in mem.events)
    assert summaries == ["woman left", "woman walked in"]


def test_consolidate_dedups_entities_across_clips():
    items = [
        _item("n1", "v", 0, "c0", entities=[
            {"name": "Mug", "type": "Object"},
        ]),
        _item("n2", "v", 1, "c1", entities=[
            {"name": "MUG", "type": "OBJECT"},  # same as n1
            {"name": "Table", "type": "Furniture"},
        ]),
    ]
    mem = consolidate(items, "v")
    names = sorted(e["name"] for e in mem.entities)
    assert names == ["Mug", "Table"]


def test_consolidate_carries_source_node_ids():
    items = [
        _item("n1", "v", 0, "c0"),
        _item("n2", "v", 1, "c1"),
        _item("n3", "v", 2, "c2"),
    ]
    mem = consolidate(items, "v")
    assert mem.source_node_ids == ["n1", "n2", "n3"]


def test_consolidate_handles_non_dict_characters_events_entities():
    """Some upstream operators might emit non-dict entries (e.g.
    strings instead of {name, role} objects); the consolidator
    must skip them rather than crash."""
    items = [
        _item("n1", "v", 0, "c0",
              characters=["bad-string", {"name": "Alice", "role": "Owner"}],
              events=[None, {"type": "x", "summary": "y"}],
              entities=[42, {"name": "Mug", "type": "Object"}]),
    ]
    mem = consolidate(items, "v")
    assert len(mem.characters) == 1
    assert mem.characters[0]["name"] == "Alice"
    assert len(mem.events) == 1
    assert mem.events[0]["summary"] == "y"
    assert len(mem.entities) == 1
    assert mem.entities[0]["name"] == "Mug"


# ---------------------------------------------------------------------------
# Driver — graph I/O
# ---------------------------------------------------------------------------

def _seed_episode(
    graph, ctx, *, node_id, video_id, clip_id,
    description, characters=None, events=None, entities=None,
):
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "characters": characters or [],
            "events": events or [],
            "entities": entities or [],
            # Match the operator's modern provenance keys so the
            # driver can read them.
            "source_video_id": video_id,
            "source_clip_id": clip_id,
            "_video_id": video_id,
            "_clip_id": clip_id,
        },
        node_id=node_id,
    )
    graph._store.create_node(node.serialize(), ctx)


def test_driver_creates_one_memory_per_video(tmp_path: Path):
    """Seed 2 EpisrictedNodes for vid_a and 1 for vid_b. Expect 2
    MemoryNodes, each with the right topic + supporting_node_ids."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, node_id="ep_a1", video_id="vid_a",
                  clip_id=0, description="A: clip 0.")
    _seed_episode(graph, ctx, node_id="ep_a2", video_id="vid_a",
                  clip_id=1, description="A: clip 1.")
    _seed_episode(graph, ctx, node_id="ep_b1", video_id="vid_b",
                  clip_id=0, description="B: clip 0.")

    output_jsonl = tmp_path / "smem.jsonl"
    from m3.semantic_memory_consolidator import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(output_jsonl),
    ])
    rc = run_semantic_memory_consolidator(args, graph=graph)
    assert rc == 0

    # 2 MemoryNodes in the graph.
    from core.graph_models import MemoryNode
    memory_nodes = [
        n for n in graph._store.list_nodes(None, ctx)
        if isinstance(n, MemoryNode)
    ]
    assert len(memory_nodes) == 2
    by_topic = {n.topic: n for n in memory_nodes}
    assert "semantic:vid_a" in by_topic
    assert "semantic:vid_b" in by_topic
    # Each consolidates 1-2 EpisrictedNodes.
    a_id = by_topic["semantic:vid_a"].node_id
    assert sorted(by_topic["semantic:vid_a"].supporting_node_ids) == ["ep_a1", "ep_a2"]
    assert sorted(by_topic["semantic:vid_b"].supporting_node_ids) == ["ep_b1"]

    # JSONL has 2 rows.
    rows = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(rows) == 2


def test_driver_emits_evolves_edges_to_memory(tmp_path: Path):
    """Every source EpisrictedNode gets an EVOLVES edge to the
    consolidated MemoryNode."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, node_id="ep_a1", video_id="vid_a",
                  clip_id=0, description="clip 0")
    _seed_episode(graph, ctx, node_id="ep_a2", video_id="vid_a",
                  clip_id=1, description="clip 1")

    output_jsonl = tmp_path / "smem.jsonl"
    from m3.semantic_memory_consolidator import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(output_jsonl),
    ])
    run_semantic_memory_consolidator(args, graph=graph)

    # Find the consolidated MemoryNode.
    from core.graph_models import MemoryNode
    mem_nodes = [
        n for n in graph._store.list_nodes(None, ctx)
        if isinstance(n, MemoryNode) and n.topic == "semantic:vid_a"
    ]
    assert len(mem_nodes) == 1
    mem_id = mem_nodes[0].node_id

    # 2 EVOLVES edges: ep_a1 → mem, ep_a2 → mem.
    edges: set = set()
    for src in ("ep_a1", "ep_a2"):
        for _nb, e, _d in graph.get_neighbors(
            src, ctx,
            edge_types=[GraphEdgeType.EVOLVES],
        ):
            if e.edge_type == GraphEdgeType.EVOLVES and e.target_id == mem_id:
                edges.add(src)
    assert edges == {"ep_a1", "ep_a2"}


def test_driver_is_idempotent_on_rerun(tmp_path: Path):
    """Re-running must NOT create a duplicate MemoryNode or duplicate
    EVOLVES edges."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, node_id="ep_a1", video_id="vid_a",
                  clip_id=0, description="clip 0")

    from m3.semantic_memory_consolidator import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "smem.jsonl"),
    ])
    run_semantic_memory_consolidator(args, graph=graph)
    from core.graph_models import MemoryNode
    mem_count_first = sum(
        1 for n in graph._store.list_nodes(None, ctx)
        if isinstance(n, MemoryNode)
    )
    # Count EVOLVES edges uniquely (each edge is seen from both
    # endpoints when iterating all nodes, so dedupe via frozenset
    # of (source, target)).
    edges_first = {
        frozenset([e.source_id, e.target_id])
        for n in graph._store.list_nodes(None, ctx)
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.EVOLVES],
        )
    }
    run_semantic_memory_consolidator(args, graph=graph)
    mem_count_second = sum(
        1 for n in graph._store.list_nodes(None, ctx)
        if isinstance(n, MemoryNode)
    )
    edges_second = {
        frozenset([e.source_id, e.target_id])
        for n in graph._store.list_nodes(None, ctx)
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.EVOLVES],
        )
    }
    assert mem_count_first == mem_count_second == 1
    assert edges_first == edges_second
    assert len(edges_first) == 1


def test_driver_video_filter_narrows_input(tmp_path: Path):
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, node_id="ep_a", video_id="vid_a",
                  clip_id=0, description="a")
    _seed_episode(graph, ctx, node_id="ep_b", video_id="vid_b",
                  clip_id=0, description="b")

    from m3.semantic_memory_consolidator import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "smem.jsonl"),
        "--video", "vid_a",
    ])
    rc = run_semantic_memory_consolidator(args, graph=graph)
    assert rc == 0
    rows = [json.loads(l) for l in (tmp_path / "smem.jsonl").read_text().splitlines() if l]
    assert len(rows) == 1
    assert rows[0]["video_id"] == "vid_a"


def test_driver_no_episodes_exits_cleanly(tmp_path: Path):
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    # No episodes seeded — driver should still write an empty
    # JSONL (so downstream consumers can rely on the file).
    output_jsonl = tmp_path / "smem.jsonl"
    from m3.semantic_memory_consolidator import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(output_jsonl),
    ])
    rc = run_semantic_memory_consolidator(args, graph=graph)
    assert rc == 0
    assert output_jsonl.exists()
    assert output_jsonl.read_text() == ""


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def test_semantic_memory_consolidator_stage_is_registered():
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.semantic_memory_consolidator" in DEFAULT_STAGE_REGISTRY


def test_semantic_memory_consolidator_stage_runs_end_to_end(tmp_path: Path):
    """Seed 2 EpisrictedNodes → run the stage → assert one
    MemoryNode landed + 2 EVOLVES edges."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, node_id="ep_a1", video_id="vid_a",
                  clip_id=0, description="clip 0",
                  characters=[{"name": "Alice", "role": "Owner"}])
    _seed_episode(graph, ctx, node_id="ep_a2", video_id="vid_a",
                  clip_id=1, description="clip 1",
                  characters=[{"name": "Alice", "role": "owner"}],
                  events=[{"type": "entry", "summary": "in"}])

    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    pipe_ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="vid_a", seconds=10, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
        artifacts={"semantic_memories": tmp_path / "run" / "semantic_memories.jsonl"},
        variables={}, graph=graph,
    )
    stage = StageSpec(
        id="semantic_memory_consolidator", type="m3.semantic_memory_consolidator",
        needs=["episodic_memory"], enabled=True,
        params={},
    )
    result = stages_mod.semantic_memory_consolidator_stage(pipe_ctx, stage)
    assert result.ok is True
    assert result.counts["memories"] == 1

    # MemoryNode + edges landed.
    from core.graph_models import MemoryNode
    mem_nodes = [
        n for n in graph._store.list_nodes(None, ctx)
        if isinstance(n, MemoryNode) and n.topic == "semantic:vid_a"
    ]
    assert len(mem_nodes) == 1
    mem = mem_nodes[0]
    assert sorted(mem.supporting_node_ids) == ["ep_a1", "ep_a2"]
    # Description merged (in clip order).
    content = json.loads(mem.content)
    assert "clip 0" in content["description"]
    assert "clip 1" in content["description"]
    # Characters deduped.
    assert len(content["characters"]) == 1
    assert content["characters"][0]["name"] == "Alice"
    # Events carried through.
    assert len(content["events"]) == 1

    # 2 EVOLVES edges from the EpisrictedNodes to the MemoryNode.
    evolves_targets = set()
    for src in ("ep_a1", "ep_a2"):
        for _nb, e, _d in graph.get_neighbors(
            src, ctx,
            edge_types=[GraphEdgeType.EVOLVES],
        ):
            if e.target_id == mem.node_id:
                evolves_targets.add(src)
    assert evolves_targets == {"ep_a1", "ep_a2"}


def test_semantic_memory_consolidator_yaml_stage_wired_after_episodic_memory():
    """semantic_memory_consolidator must depend on episodic_memory so
    the EpisrictedNodes exist before consolidation runs."""
    import yaml
    spec = yaml.safe_load(
        Path("config/m3/pipelines/video_episodic_faces.yaml").read_text()
    )
    stage_ids = [s["id"] for s in spec["stages"]]
    assert "semantic_memory_consolidator" in stage_ids
    smc = next(s for s in spec["stages"]
               if s["id"] == "semantic_memory_consolidator")
    assert smc["type"] == "m3.semantic_memory_consolidator"
    assert "episodic_memory" in smc["needs"]
    assert smc["enabled"] is False
    assert "semantic_memories" in spec["artifacts"]