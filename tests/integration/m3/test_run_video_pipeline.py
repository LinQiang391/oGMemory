"""Tests for :mod:`m3.run_video_pipeline`."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import load_pipeline_spec
from m3.pipeline.stages import create_memory_nodes


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


_FFMPEG = shutil.which("ffmpeg")
_requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None, reason="ffmpeg not available on PATH",
)


@_requires_ffmpeg
def test_runner_shares_graph_across_stages(tmp_path: Path):
    """Regression: with --graph-store memory, every stage must see the
    same graph service so VIDEO RawNodes from load_clips are visible
    to extract_audio / extract_frames / episodic_memory.

    Before this fix each driver's parse_args/run call built its own
    MemoryGraphStore() and the downstream stages silently produced
    zero records. This test runs the default pipeline end-to-end on
    a 30-second synthetic clip and asserts that extract_frames,
    episodic_memory, and create_memory_nodes all produced non-empty
    results — which is only possible if the graph survives the stage
    boundary.
    """
    video = tmp_path / "src.mp4"
    # 30-second synthetic clip with both video and audio tracks so
    # load_clips, extract_audio, extract_frames, and episodic_memory
    # all have inputs to work with.
    subprocess.run(
        [
            _FFMPEG, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=30:r=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(video),
        ],
        check=True,
    )

    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    # Disable detect_faces (no CV service in CI) and build_index (needs
    # an embedder); keep load_clips + extract_audio + extract_frames +
    # episodic_memory + create_memory_nodes to exercise the graph flow.
    for stage in spec.stages:
        if stage.id == "episodic_memory":
            stage.params["llm"] = "mock"
        if stage.id in {"detect_faces", "build_index"}:
            stage.enabled = False

    runner = PipelineRunner()
    ctx = runner.build_context(
        spec,
        video=video,
        video_id="bedroom_01",
        seconds=30,
        work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory",
        connection_string=None,
        account_id="m3bench",
    )

    summary = runner.run(spec, ctx)

    stages = summary["stages"]
    # load_clips persists VIDEO RawNodes into the shared graph.
    assert stages["load_clips"]["counts"]["clips"] == 1
    # extract_audio finds the VIDEO node and writes a WAV + AUDIO node.
    assert stages["extract_audio"]["counts"]["records"] == 1
    assert stages["extract_audio"]["counts"]["new_records"] == 1
    assert (tmp_path / "run" / "audio" / "bedroom_01" / "0.wav").exists()
    # extract_frames finds the VIDEO node and materializes FRAME nodes.
    # The non-zero count is the key assertion: pre-fix this was 0.
    assert stages["extract_frames"]["counts"]["records"] == 1
    assert stages["extract_frames"]["counts"]["frames"] >= 1
    # episodic_memory walks the graph for VIDEO+FRAME+AUDIO children
    # and produces a description. Pre-fix this was 0.
    assert stages["episodic_memory"]["counts"]["records"] == 1
    assert stages["episodic_memory"]["counts"]["description_chars"] > 0
    # create_memory_nodes reads the episodic jsonl and writes a
    # MemoryNode. Pre-fix this was 0.
    assert stages["create_memory_nodes"]["counts"]["created_memory_nodes"] == 1
    assert stages["create_memory_nodes"]["counts"]["missing_support"] == 0

    # And the PipelineContext itself should now expose the shared
    # graph service — stages that build their own OperatorService
    # via this handle will write into the same store.
    assert ctx.graph is not None
    assert ctx.op_service is not None
