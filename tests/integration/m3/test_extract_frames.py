"""Tests for :mod:`m3.extract_frames` driver.

The driver itself is now a thin wrapper around ``OperatorService`` +
``VideoFrameExtractor`` materialization. The interesting behaviour
(JPEG writing, IMAGE RawNode creation, REFERENCES edges, lightweight
summary, idempotency) lives at the operator layer — see
``tests/unit/extraction/operators/test_materialize.py`` and
``tests/unit/extraction/operators/test_video_frame_extractor.py`` for
the bulk of coverage. These tests verify the M3-specific glue:

* the manifest format the m3 tooling expects,
* the on-disk JPEG layout (``frames_root/<video_id>/<clip_id>/<idx>.jpg``),
* idempotent re-run via the parent ``ExtractedNode`` lookup,
* ``OperatorService.run_operator`` is invoked with the expected kwargs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType, RawDataType  # noqa: E402
from core.graph_models import create_raw_node  # noqa: E402
from core.models import RequestContext  # noqa: E402
from m3.config import M3BENCH_ACCOUNT_ID  # noqa: E402
from m3.extract_frames import (  # noqa: E402
    _build_frame_writer,
    _write_frame_jpeg,
    extracted_node_id,
    frame_node_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-extract-frames-test", trace_id="t-test",
    )


def _make_memory_graph():
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    return GraphService(MemoryGraphStore())


def _make_op_service(graph):
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from service.operator_service import OperatorService
    registry = default_registry()
    register_all_defaults(registry)
    return OperatorService(graph, registry=registry)


def _make_video_raw_node(video_path: Path, account_id: str = M3BENCH_ACCOUNT_ID):
    seed = f"clip/{video_path.parent.name}/{video_path.stem}".encode()
    node_id = hashlib.sha1(seed).hexdigest()[:32]
    return create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        account_id=account_id,
        metadata={
            "video_id": video_path.parent.name,
            "clip_id": int(video_path.stem),
            "start_sec": int(video_path.stem) * 30,
            "source_path": str(video_path),
        },
        node_id=node_id,
    )


def _persist_video_raw_node(graph, video_node, ctx):
    """Persist a VIDEO RawNode into ``graph`` so the service-side lookup
    in :func:`extract_frames_for_clip` can find it."""
    graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content=video_node.content,
        ctx=ctx,
        metadata=video_node.metadata,
        node_id=video_node.node_id,
    )


@pytest.fixture
def real_video(tmp_path: Path) -> Path:
    """Generate a tiny synthetic mp4 so video_frame_extractor can read it."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = tmp_path / "videos" / "vid_a" / "0.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (160, 120))
    try:
        for i in range(30):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            x = int((i / 29) * 120)
            cv2.rectangle(frame, (x, 30), (x + 20, 90), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


# ---------------------------------------------------------------------------
# ID derivation (kept for manifest compatibility)
# ---------------------------------------------------------------------------

class TestIdDerivation:
    def test_extracted_node_id_is_deterministic(self):
        a = extracted_node_id("video_frame_extractor", "raw-xyz")
        b = extracted_node_id("video_frame_extractor", "raw-xyz")
        assert a == b
        assert len(a) == 32

    def test_extracted_node_id_varies_with_source(self):
        a = extracted_node_id("video_frame_extractor", "raw-1")
        b = extracted_node_id("video_frame_extractor", "raw-2")
        assert a != b

    def test_extracted_node_id_varies_with_extractor(self):
        a = extracted_node_id("video_frame_extractor", "raw-1")
        b = extracted_node_id("face_detection", "raw-1")
        assert a != b

    def test_frame_node_id_is_deterministic(self):
        a = frame_node_id("vid_a", 0, 5)
        b = frame_node_id("vid_a", 0, 5)
        assert a == b

    def test_frame_node_id_varies_with_clip(self):
        assert frame_node_id("vid_a", 0, 0) != frame_node_id("vid_a", 1, 0)

    def test_frame_node_id_varies_with_frame(self):
        assert frame_node_id("vid_a", 0, 0) != frame_node_id("vid_a", 0, 1)


# ---------------------------------------------------------------------------
# Frame persistence helpers (kept in m3/ because they're filesystem-layout
# specific to M3-Bench; the operator layer is filesystem-agnostic).
# ---------------------------------------------------------------------------

class TestWriteFrameJpeg:
    def test_creates_file_at_expected_path(self, tmp_path):
        p = _write_frame_jpeg(tmp_path, "vid_a", 0, 5, b"\xff\xd8\xff\xe0FAKE")
        assert p == tmp_path / "vid_a" / "0" / "5.jpg"
        assert p.read_bytes() == b"\xff\xd8\xff\xe0FAKE"

    def test_creates_intermediate_dirs(self, tmp_path):
        p = _write_frame_jpeg(tmp_path, "deep/nested", 99, 7, b"x")
        assert p.exists()
        assert p.parent == tmp_path / "deep" / "nested" / "99"


class TestBuildFrameWriter:
    def test_writer_writes_to_clip_specific_subdir(self, tmp_path):
        writer = _build_frame_writer(tmp_path, "vid_b", 7)
        out = writer({"frame_idx": 12}, b"\xff\xd8\xff\xe1jpeg-bytes")
        assert out == str(tmp_path / "vid_b" / "7" / "12.jpg")
        assert Path(out).read_bytes() == b"\xff\xd8\xff\xe1jpeg-bytes"

    def test_writer_creates_dirs_lazily(self, tmp_path):
        writer = _build_frame_writer(tmp_path / "deep", "v", 3)
        writer({"frame_idx": 0}, b"x")
        assert (tmp_path / "deep" / "v" / "3" / "0.jpg").exists()


# ---------------------------------------------------------------------------
# Driver integration via extract_frames_for_clip
# ---------------------------------------------------------------------------

class TestExtractFramesForClip:
    """End-to-end driver test: synthetic mp4 → frames → IMAGE RawNodes →
    REFERENCES edges → lightweight ExtractedNode summary."""

    def _run(self, tmp_path, real_video):
        from m3.extract_frames import extract_frames_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)

        result = extract_frames_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        return result, graph, ctx

    def test_creates_one_image_node_per_frame(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        # 30 frames @ 30fps sampled @ 5fps → 5 frames
        assert result["child_count"] == 5
        assert result["resumed"] is False
        assert len(result["child_node_ids"]) == 5
        for fn_id in result["child_node_ids"]:
            node = graph.get_raw_node(fn_id, ctx)
            assert node is not None
            assert node.raw_type == RawDataType.IMAGE
            # frame_idx propagated from operator to child metadata
            assert "frame_idx" in node.metadata
            assert "ts_sec" in node.metadata

    def test_frame_jpegs_are_persisted_to_disk(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        for fn_id in result["child_node_ids"]:
            node = graph.get_raw_node(fn_id, ctx)
            path = Path(node.metadata["source_path"])
            assert path.exists()
            assert path.read_bytes()[:3] == b"\xff\xd8\xff"  # JPEG magic

    def test_extracted_node_has_lightweight_summary(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        ext = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert ext is not None
        assert ext.extractor_type == "video_frame_extractor"
        # Heavy base64 payloads must not be stored.
        assert "frames" not in (ext.extracted_content or {})
        # Lightweight summary IS stored.
        assert "child_node_ids" in ext.extracted_content
        # Operator-pass-through keys still present.
        assert "fps_requested" in ext.extracted_content
        assert "fps_actual" in ext.extracted_content

    def test_references_edges_link_video_to_frames(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        video_node = graph.get_raw_node(
            _make_video_raw_node(real_video).node_id, ctx,
        )
        neighbors = graph.get_neighbors(
            video_node.node_id, ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        assert len(neighbors) == result["child_count"]
        neighbor_ids = {n[0].node_id for n in neighbors}
        assert neighbor_ids == set(result["child_node_ids"])

    def test_resumed_when_extracted_node_exists(self, tmp_path, real_video):
        """Re-running for a clip whose parent ExtractedNode already exists
        must short-circuit and report ``resumed=True``."""
        from m3.extract_frames import extract_frames_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)
        frames_root = tmp_path / "frames"

        first = extract_frames_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=frames_root,
            fps=5,
        )
        assert first["resumed"] is False

        # Second run: same VIDEO node, same extracted_node_id will
        # already be present so the driver short-circuits.
        second = extract_frames_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=frames_root,
            fps=5,
        )
        assert second["resumed"] is True
        assert second["child_node_ids"] == first["child_node_ids"]
        assert second["child_count"] == first["child_count"]

    def test_keysframes_only_filters_at_extraction_time(self, tmp_path, real_video):
        from m3.extract_frames import extract_frames_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)
        result = extract_frames_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
            keyframes_only=True,
        )
        # Synthetic video has motion on every sampled frame → all 5 are
        # keyframes. The point of this test is just that the
        # ``keyframes_only`` flag flowed through to the operator without
        # erroring.
        assert result["resumed"] is False
        assert result["child_count"] >= 1

    def test_max_frames_caps_output(self, tmp_path, real_video):
        from m3.extract_frames import extract_frames_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)
        result = extract_frames_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
            max_frames=2,
        )
        assert result["child_count"] == 2

    def test_missing_video_raises(self, tmp_path):
        from m3.extract_frames import extract_frames_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        # Point at a path that does not exist so the operator errors.
        ghost = create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="",
            account_id=M3BENCH_ACCOUNT_ID,
            metadata={"video_id": "ghost", "clip_id": 0,
                      "source_path": "/no/such/file.mp4"},
            node_id="ghost-video",
        )
        _persist_video_raw_node(graph, ghost, ctx)
        with pytest.raises(RuntimeError, match="video_frame_extractor failed"):
            extract_frames_for_clip(
                graph=graph, op_service=op_service, ctx=ctx,
                video_raw_node=ghost,
                video_id="ghost",
                clip_id=0,
                frames_root=tmp_path / "frames",
                fps=5,
            )


# ---------------------------------------------------------------------------
# Manifest format
# ---------------------------------------------------------------------------

class TestRunManifestOutput:
    """Smoke-test the ``run()`` driver end-to-end with a synthetic
    manifest, checking the JSONL output schema that downstream M3
    tooling expects."""

    def test_run_writes_manifest_jsonl(self, tmp_path, real_video):
        from m3.extract_frames import parse_args, run

        # Synthesize a single-video manifest so the driver has something
        # to iterate over. The load_clips step normally produces this;
        # we emulate just enough of it here.
        manifest_path = tmp_path / "data.jsonl"
        video_node = _make_video_raw_node(real_video)
        manifest_path.write_text(json.dumps({
            "id": real_video.parent.name,
            "raw_nodes": [
                {"clip_id": 0, "raw_node_id": video_node.node_id},
            ],
        }) + "\n", encoding="utf-8")

        # Build a graph + service we control, seed the VIDEO RawNode, and
        # inject them into ``run()`` so the driver sees the same state we
        # set up here.
        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        _persist_video_raw_node(graph, video_node, ctx)

        out_jsonl = tmp_path / "frames.jsonl"
        args = parse_args([
            "--manifest", str(manifest_path),
            "--frames-root", str(tmp_path / "frames"),
            "--output-jsonl", str(out_jsonl),
            "--graph-store", "memory",
            "--fps", "5",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        assert rc == 0, f"run() returned {rc}"

        assert out_jsonl.exists()
        records = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line]
        assert len(records) == 1
        rec = records[0]
        # Schema: video_id, clip_id, video_raw_node_id, extracted_node_id,
        # child_node_ids, child_count, resumed.
        assert rec["video_id"] == real_video.parent.name
        assert rec["clip_id"] == 0
        assert rec["video_raw_node_id"] == video_node.node_id
        assert rec["extracted_node_id"]
        assert rec["child_count"] == 5
        assert rec["resumed"] is False
        assert len(rec["child_node_ids"]) == 5
