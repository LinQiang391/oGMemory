"""Tests that m3 drivers respect extraction status when short-circuiting on rerun."""

from __future__ import annotations

from pathlib import Path

from m3.detect_faces import detect_faces_for_frame, extracted_node_id
from m3.extract_frames import extract_frames_for_clip
from m3.extract_frames import extracted_node_id as extract_extracted_node_id
from m3.resume_status import is_completed_extraction


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
    from core.graph_enums import ExtractionStatus, RawDataType

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


def test_extract_frames_marks_extracted_node_completed(tmp_path: Path):
    """Regression: ``OperatorService.persist_extraction`` must flip the
    parent ``ExtractedNode`` to ``COMPLETED`` so the next run can
    short-circuit via :func:`m3.resume_status.is_completed_extraction`.

    Without the explicit ``mark_extraction_completed`` call the row
    would stay ``PENDING`` and the idempotency pre-check would never
    fire, causing every rerun to redo all the work.
    """
    from core.graph_enums import ExtractionStatus, RawDataType

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
            "video_id": "v_complete",
            "clip_id": 0,
            "source_path": str(src_mp4),
        },
        node_id="video-complete",
    )

    # First run: completes the extraction and writes the parent node.
    result1 = extract_frames_for_clip(
        graph=graph, op_service=op_service, ctx=ctx,
        video_raw_node=video,
        video_id="v_complete",
        clip_id=0,
        frames_root=tmp_path / "frames",
        fps=5,
    )
    assert result1["resumed"] is False
    ext_id = extract_extracted_node_id("video_frame_extractor", video.node_id)
    stored = graph.get_extracted_node(ext_id, ctx)
    assert stored is not None
    assert is_completed_extraction(stored.extraction_status)
    assert stored.extraction_status == ExtractionStatus.COMPLETED

    # Second run: the idempotency pre-check must short-circuit and
    # return the cached child_node_ids rather than re-extracting.
    result2 = extract_frames_for_clip(
        graph=graph, op_service=op_service, ctx=ctx,
        video_raw_node=video,
        video_id="v_complete",
        clip_id=0,
        frames_root=tmp_path / "frames",
        fps=5,
    )
    assert result2["resumed"] is True
    assert result2["extracted_node_id"] == ext_id
    assert result2["child_node_ids"] == result1["child_node_ids"]


def test_detect_faces_marks_extracted_node_completed(tmp_path: Path):
    """Same regression for the face detection driver."""
    from core.graph_enums import ExtractionStatus, RawDataType

    graph, op_service, ctx = _build_services()
    frame = graph.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"\xff\xd8\xff\xe0jpeg",
        ctx=ctx,
        metadata={"video_id": "v", "clip_id": 0, "frame_idx": 0},
        node_id="frame-complete",
    )

    from extraction.operators.backends import MockBackend

    result1 = detect_faces_for_frame(
        graph=graph, op_service=op_service, ctx=ctx,
        frame_raw_node=frame,
        backend=MockBackend(face_count=2),
    )
    assert result1["resumed"] is False
    assert result1["face_count"] == 2
    ext_id = extracted_node_id("face_detection", frame.node_id)
    stored = graph.get_extracted_node(ext_id, ctx)
    assert stored is not None
    assert is_completed_extraction(stored.extraction_status)
    assert stored.extraction_status == ExtractionStatus.COMPLETED

    result2 = detect_faces_for_frame(
        graph=graph, op_service=op_service, ctx=ctx,
        frame_raw_node=frame,
        backend=MockBackend(face_count=2),
    )
    assert result2["resumed"] is True
    assert result2["child_node_ids"] == result1["child_node_ids"]
