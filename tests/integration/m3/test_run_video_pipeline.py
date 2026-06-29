"""Tests for :mod:`m3.run_video_pipeline`."""

from __future__ import annotations

import json
from pathlib import Path

from m3.run_video_pipeline import create_memory_nodes


def test_create_memory_nodes_from_episodic_jsonl(tmp_path: Path):
    from core.graph_enums import RawDataType
    from core.models import RequestContext
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    from service.operator_service import OperatorService

    graph = GraphService(MemoryGraphStore())
    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)
    ctx = RequestContext(
        account_id="m3bench",
        user_id="m3bench-user",
        agent_id="m3bench-agent",
        session_id="test",
        trace_id="test",
    )

    video = graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        ctx=ctx,
        metadata={"video_id": "bedroom_03", "clip_id": 0},
        node_id="video-0",
    )
    from extraction.operators.builtin import EpisodicMemoryGenerator
    from providers.llm import MockLLM
    op = EpisodicMemoryGenerator(llm=MockLLM())
    extraction = op.extract(
        video,
        ctx,
        frame_nodes=[graph.create_raw_node(
            raw_type=RawDataType.IMAGE,
            content=b"\xff\xd8\xff\xe0jpeg",
            ctx=ctx,
            node_id="frame-0",
        )],
    )
    persisted = op_service.persist_extraction(
        raw_node=video,
        operator=op,
        extraction=extraction,
        ctx=ctx,
        desired_node_id="episodic-0",
    )
    episodic = tmp_path / "episodic_memory.jsonl"
    episodic.write_text(json.dumps({
        "video_id": "bedroom_03",
        "clip_id": 0,
        "video_raw_node_id": video.node_id,
        "extracted_node_id": persisted["extracted_node_id"],
        "description": "The bedroom clip shows a bed and a nightstand.",
        "frame_count": 1,
        "has_audio": False,
    }) + "\n", encoding="utf-8")

    summary = create_memory_nodes(
        episodic_jsonl=episodic,
        graph_store="memory",
        connection_string=None,
        graph=graph,
        ctx=ctx,
    )
    assert summary["created_memory_nodes"] == 1
    assert summary["missing_support"] == 0
    memory_id = "337f223400b2043f0275cd0388aa30f8"
    node = graph.get_memory_node(memory_id, ctx)
    assert node is not None
    assert node.topic == "m3:bedroom_03"
    assert "episodic-0" in node.supporting_node_ids
