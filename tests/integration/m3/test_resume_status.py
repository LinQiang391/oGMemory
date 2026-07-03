"""Tests that m3 drivers respect extraction status when short-circuiting on rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3.detect_faces import detect_faces_for_frame, extracted_node_id
from m3.extract_frames import extract_frames_for_clip, extracted_node_id as extract_extracted_node_id


def _build_services():
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
    return graph, op_service, ctx


def test_extract_frames_does_not_resume_pending_node(tmp_path: Path):
    """A leftover PENDING ExtractedNode from a previous failed run must be re-processed."""
    from core.graph_enums import RawDataType
    from core.graph_enums import ExtractionStatus

    graph, op_service, ctx = _build_services()
    import subprocess
    src_mp4 = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=red:size=64x64:rate=25:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src_mp4),
        ],
        check=True,
        capture_output=True,
    )
    video = graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        ctx=ctx,
        metadata={
            "video_id": "v_pending",
            "clip_id": 0,
            "source_path": str(src_mp4),
        },
        node_id="video-pending",
    )
    ext_id = extract_extracted_node_id("video_frame_extractor", video.node_id)

    # Simulate a leftover PENDING row from a previous failed run.
    graph.create_extracted_node(
        source_raw_id=video.node_id,
        extractor_type="video_frame_extractor",
        extraction_schema="video_frame_extractor_v1",
        ctx=ctx,
        extracted_content={
            "error": "previous run failed",
            "child_count": 0,
            "child_node_ids": [],
        },
        node_id=ext_id,
    )
    pending_node = graph.get_extracted_node(ext_id, ctx)
    assert pending_node.extraction_status == ExtractionStatus.PENDING.value

    result = extract_frames_for_clip(
        graph=graph, op_service=op_service, ctx=ctx,
        video_raw_node=video,
        video_id="v_pending",
        clip_id=0,
        frames_root=tmp_path / "frames",
        fps=5,
    )
    assert result["resumed"] is False
    assert result["child_count"] >= 0


def test_detect_faces_does_not_resume_pending_node(tmp_path: Path):
    """A leftover PENDING ExtractedNode from a previous failed run must be re-processed."""
    from core.graph_enums import ExtractionStatus, RawDataType

    graph, op_service, ctx = _build_services()
    frame = graph.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"\xff\xd8\xff\xe0jpeg",
        ctx=ctx,
        metadata={"video_id": "v", "clip_id": 0, "frame_idx": 0},
        node_id="frame-pending",
    )
    ext_id = extracted_node_id("face_detection", frame.node_id)

    graph.create_extracted_node(
        source_raw_id=frame.node_id,
        extractor_type="face_detection",
        extraction_schema="face_detection_v1",
        ctx=ctx,
        extracted_content={"error": "previous run failed", "face_count": 0},
        node_id=ext_id,
    )
    pending_node = graph.get_extracted_node(ext_id, ctx)
    assert pending_node.extraction_status == ExtractionStatus.PENDING.value

    from extraction.operators.backends import MockBackend

    result = detect_faces_for_frame(
        graph=graph, op_service=op_service, ctx=ctx,
        frame_raw_node=frame,
        backend=MockBackend(face_count=1),
    )
    assert result["resumed"] is False
    assert result["face_count"] == 1
