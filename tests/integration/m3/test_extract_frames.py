"""Tests for :mod:`m3.extract_frames`."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType, RawDataType  # noqa: E402
from core.graph_models import create_raw_node  # noqa: E402
from core.models import RequestContext  # noqa: E402
from m3.config import M3BENCH_ACCOUNT_ID  # noqa: E402
from m3.extract_frames import (  # noqa: E402
    _decode_frame,
    _write_frame_jpeg,
    extract_frames_for_clip,
    extracted_node_id,
    frame_node_id,
    parse_args,
    run,
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
# ID derivation
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
# _write_frame_jpeg
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


# ---------------------------------------------------------------------------
# _decode_frame
# ---------------------------------------------------------------------------

class TestDecodeFrame:
    def test_round_trip(self):
        raw = b"\xff\xd8\xff\xe0payload"
        encoded = {"image_bytes_b64": base64.b64encode(raw).decode()}
        assert _decode_frame(encoded) == raw

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="image_bytes_b64"):
            _decode_frame({"frame_idx": 0})


# ---------------------------------------------------------------------------
# extract_frames_for_clip
# ---------------------------------------------------------------------------

class TestExtractFramesForClip:
    def test_creates_one_image_node_per_frame(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)

        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        # 30 frames at 30 fps sampled at 5 fps -> 5 frames
        assert result["frame_count"] == 5
        assert result["resumed"] is False
        assert len(result["frame_node_ids"]) == 5
        for fn_id in result["frame_node_ids"]:
            node = graph.get_raw_node(fn_id, ctx)
            assert node is not None
            assert node.raw_type == RawDataType.IMAGE

    def test_frame_jpegs_are_persisted_to_disk(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        for fn_id in result["frame_node_ids"]:
            node = graph.get_raw_node(fn_id, ctx)
            path = Path(node.metadata["source_path"])
            assert path.exists()
            assert path.read_bytes()[:3] == b"\xff\xd8\xff"  # JPEG magic

    def test_extracted_node_has_frame_node_ids_no_heavy_payload(
        self, tmp_path, real_video,
    ):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        ext = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert ext is not None
        assert ext.extractor_type == "video_frame_extractor"
        # Heavy base64 payloads must not be stored.
        assert "frames" not in (ext.extracted_content or {})
        # Lightweight summary IS stored.
        assert "frame_node_ids" in ext.extracted_content
        assert ext.extracted_content["frame_count"] == 5
        # Each frame_node_id corresponds to a real RawNode.
        for fn_id in ext.extracted_content["frame_node_ids"]:
            assert graph.get_raw_node(fn_id, ctx) is not None

    def test_references_edges_link_video_to_frames(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        neighbors = graph.get_neighbors(
            video_node.node_id, ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        frame_ids = set(result["frame_node_ids"])
        # The video node should be source of one edge per frame.
        edges_to_frames = [
            (n, e) for n, e, d in neighbors
            if n.node_id in frame_ids
        ]
        assert len(edges_to_frames) == result["frame_count"]

    def test_extracted_from_edge_auto_created(self, tmp_path, real_video):
        # The graph service creates extracted_from automatically; the
        # references edges we add go in addition to that.
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        ext = graph.get_extracted_node(result["extracted_node_id"], ctx)
        # Walking from the source VIDEO node should reach the ExtractedNode.
        neighbors = graph.get_neighbors(
            video_node.node_id, ctx,
            edge_types=[GraphEdgeType.EXTRACTED_FROM],
        )
        ext_ids = {n.node_id for n, _, _ in neighbors}
        assert ext.node_id in ext_ids

    def test_resume_is_noop(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        kwargs = dict(
            graph=graph, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        r1 = extract_frames_for_clip(**kwargs)
        r2 = extract_frames_for_clip(**kwargs)
        assert r1["extracted_node_id"] == r2["extracted_node_id"]
        assert r1["frame_node_ids"] == r2["frame_node_ids"]
        assert r1["frame_count"] == r2["frame_count"]
        # Second run is a true no-op: no additional nodes, no extra edges.
        # The video node was never written by the script (it was passed
        # in pre-populated by the test), so the graph only contains the
        # 5 IMAGE RawNodes + 1 EXTRACTED node the script produced.
        stats = graph.get_graph_stats(ctx)
        assert stats.raw_nodes == r1["frame_count"]
        assert stats.extracted_nodes == 1
        # Edges: 5 has_frame (video -> frames) + 1 extracted_from
        assert stats.total_edges == r1["frame_count"] + 1
        # And the second result is marked as a resume hit.
        assert r2["resumed"] is True

    def test_frame_node_id_is_deterministic(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        # The operator samples every 6th frame for a 30fps video at
        # 5fps, so frame_idx is one of {0, 6, 12, 18, 24}. Read each
        # frame's actual frame_idx back from the graph and verify the
        # deterministic id formula.
        sampled = [
            graph.get_raw_node(fn_id, ctx).metadata["frame_idx"]
            for fn_id in result["frame_node_ids"]
        ]
        assert sorted(sampled) == [0, 6, 12, 18, 24]
        for fn_id in result["frame_node_ids"]:
            node = graph.get_raw_node(fn_id, ctx)
            expected = frame_node_id(
                real_video.parent.name, 0, node.metadata["frame_idx"],
            )
            assert fn_id == expected

    def test_max_frames_caps_extraction(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=30,
            max_frames=2,
        )
        assert result["frame_count"] == 2
        assert len(result["frame_node_ids"]) == 2

    def test_keyframes_only_drops_non_keyframes(self, tmp_path, real_video):
        # The synthetic video is all black except a moving white square.
        # Many frames will be keyframes; with keyframes_only=True the
        # result is at most the unfiltered count.
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=30,
            keyframe_threshold=0.05,
            keyframes_only=True,
        )
        # 30 frames at 30 fps sampled at 30 fps = 30 frames; moving
        # square means most are keyframes.
        assert 0 < result["frame_count"] <= 30

    def test_metadata_round_trips(self, tmp_path, real_video):
        graph = _make_memory_graph()
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        result = extract_frames_for_clip(
            graph, ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            frames_root=tmp_path / "frames",
            fps=5,
        )
        first_frame_id = result["frame_node_ids"][0]
        node = graph.get_raw_node(first_frame_id, ctx)
        assert node.metadata["video_id"] == real_video.parent.name
        assert node.metadata["clip_id"] == 0
        assert node.metadata["frame_idx"] in range(result["frame_count"])
        assert node.metadata["source_video_node_id"] == video_node.node_id
        assert "ts_sec" in node.metadata
        assert "keyframe" in node.metadata


# ---------------------------------------------------------------------------
# run() end-to-end
# ---------------------------------------------------------------------------

class TestRun:
    def test_end_to_end_writes_manifest(
        self, tmp_path, sample_annotation, sample_video_layout, real_video,
    ):
        # Both load_clips and extract_frames build their own
        # GraphService internally, so we patch build_graph_service to
        # share a single MemoryGraphStore-backed service across the two
        # drivers.
        from m3 import load_clips as lc_mod
        from providers.graph_store import MemoryGraphStore
        from service.graph_service import GraphService

        shared_graph = GraphService(MemoryGraphStore())

        def _fake_build(*a, **kw):
            return shared_graph, "memory"

        videos_dir = sample_video_layout["data_root"] / "videos" / "robot"
        for vid in ("living_room_06", "kitchen_22"):
            shutil.copy(real_video, videos_dir / f"{vid}.mp4")
            clip_dir = tmp_path / "clips" / vid
            clip_dir.mkdir(parents=True, exist_ok=True)
            # The clip file itself must be a real video for
            # video_frame_extractor to open it. Copy the synthetic
            # video into each clip slot.
            for i in range(2):
                shutil.copy(real_video, clip_dir / f"{i}.mp4")

        original = lc_mod.build_graph_service
        lc_mod.build_graph_service = _fake_build
        try:
            rc = lc_mod.run(lc_mod.parse_args([
                "--annotations", str(sample_annotation),
                "--clips-root", str(tmp_path / "clips"),
                "--videos-root", str(videos_dir),
                "--output-jsonl", str(tmp_path / "data.jsonl"),
            ]))
            assert rc == 0

            out = tmp_path / "frames.jsonl"
            args = parse_args([
                "--manifest", str(tmp_path / "data.jsonl"),
                "--frames-root", str(tmp_path / "frames"),
                "--output-jsonl", str(out),
            ])
            rc = run(args)
        finally:
            lc_mod.build_graph_service = original
        assert rc == 0
        lines = out.read_text().strip().split("\n")
        # 2 videos x 2 clips = 4 extraction records
        assert len(lines) == 4
        records = [json.loads(line) for line in lines]
        for r in records:
            assert r["frame_count"] == 5  # 5 fps sampling, 30-frame video

    def test_video_filter_picks_subset(
        self, tmp_path, sample_annotation, sample_video_layout, real_video,
    ):
        from m3 import load_clips as lc_mod
        from providers.graph_store import MemoryGraphStore
        from service.graph_service import GraphService

        shared_graph = GraphService(MemoryGraphStore())

        def _fake_build(*a, **kw):
            return shared_graph, "memory"

        videos_dir = sample_video_layout["data_root"] / "videos" / "robot"
        for vid in ("living_room_06", "kitchen_22"):
            shutil.copy(real_video, videos_dir / f"{vid}.mp4")
            clip_dir = tmp_path / "clips" / vid
            clip_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(real_video, clip_dir / "0.mp4")

        original = lc_mod.build_graph_service
        lc_mod.build_graph_service = _fake_build
        try:
            lc_mod.run(lc_mod.parse_args([
                "--annotations", str(sample_annotation),
                "--clips-root", str(tmp_path / "clips"),
                "--videos-root", str(videos_dir),
                "--output-jsonl", str(tmp_path / "data.jsonl"),
            ]))

            out = tmp_path / "frames.jsonl"
            args = parse_args([
                "--manifest", str(tmp_path / "data.jsonl"),
                "--frames-root", str(tmp_path / "frames"),
                "--output-jsonl", str(out),
                "--video", "kitchen_22",
            ])
            rc = run(args)
        finally:
            lc_mod.build_graph_service = original
        assert rc == 0
        records = [
            json.loads(line) for line in out.read_text().strip().split("\n")
        ]
        assert {r["video_id"] for r in records} == {"kitchen_22"}


class TestParseArgs:
    def test_required_manifest(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self):
        args = parse_args(["--manifest", "x.jsonl"])
        assert args.frames_root == "data/frames"
        assert args.output_jsonl == "data/frames.jsonl"
        assert args.graph_store == "memory"
        assert args.fps == 5
        assert args.keyframe_threshold == 0.1
        assert args.keyframes_only is False
