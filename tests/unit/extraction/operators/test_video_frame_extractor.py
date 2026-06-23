"""Unit tests for ``extraction.operators.builtin.video_frame_extractor``."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")  # tests need OpenCV at runtime
np = pytest.importorskip("numpy")


from core.graph_enums import RawDataType  # noqa: E402
from core.graph_models import create_raw_node  # noqa: E402
from core.models import RequestContext  # noqa: E402
from extraction.operators.builtin.video_frame_extractor import (  # noqa: E402
    VideoFrameExtractor,
    _try_import_cv2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_synthetic_video(
    path: Path,
    *,
    n_frames: int = 30,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    pattern: str = "static",
) -> None:
    """Write a tiny mp4 at ``path``.

    ``pattern`` controls the visual content:
        - "static"     : all-black frames (no motion; only first is a keyframe)
        - "moving"     : a moving bright square (every frame is a keyframe)
        - "alternating": every other frame is black (one keyframe per 2)
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open for {path}"
    try:
        for i in range(n_frames):
            if pattern == "static":
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif pattern == "moving":
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # bright square moving across the frame
                x = int((i / max(1, n_frames - 1)) * (width - 40))
                cv2.rectangle(frame, (x, 80), (x + 40, 160), (255, 255, 255), -1)
            elif pattern == "alternating":
                frame = np.zeros((height, width, 3), dtype=np.uint8) if i % 2 == 0 \
                    else np.full((height, width, 3), 255, dtype=np.uint8)
            else:
                raise ValueError(f"Unknown pattern: {pattern}")
            writer.write(frame)
    finally:
        writer.release()
    assert path.exists() and path.stat().st_size > 0, f"failed to write {path}"


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    p = tmp_path / "static.mp4"
    _write_synthetic_video(p, n_frames=30, fps=30, pattern="static")
    return p


@pytest.fixture
def moving_video(tmp_path: Path) -> Path:
    p = tmp_path / "moving.mp4"
    _write_synthetic_video(p, n_frames=30, fps=30, pattern="moving")
    return p


@pytest.fixture
def alternating_video(tmp_path: Path) -> Path:
    p = tmp_path / "alt.mp4"
    _write_synthetic_video(p, n_frames=30, fps=30, pattern="alternating")
    return p


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="test-account",
        user_id="u1", agent_id="a1",
        session_id="s1", trace_id="t1",
    )


def _make_raw_node(video_path: Path, **metadata_extras) -> "object":
    node = create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",  # bytes stay on disk; loader convention
        account_id="test-account",
        metadata={"source_path": str(video_path), **metadata_extras},
    )
    return node


# ---------------------------------------------------------------------------
# Constructor / config validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        op = VideoFrameExtractor()
        assert op._fps == 5
        assert op._max_frames is None
        assert op._resize == 720
        assert op._keyframes_only is False

    def test_zero_fps_rejected(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            VideoFrameExtractor(fps=0)

    def test_negative_fps_rejected(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            VideoFrameExtractor(fps=-1)

    def test_keyframe_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="keyframe_threshold"):
            VideoFrameExtractor(keyframe_threshold=1.5)
        with pytest.raises(ValueError, match="keyframe_threshold"):
            VideoFrameExtractor(keyframe_threshold=-0.1)

    def test_jpeg_quality_out_of_range(self):
        with pytest.raises(ValueError, match="jpeg_quality"):
            VideoFrameExtractor(jpeg_quality=0)
        with pytest.raises(ValueError, match="jpeg_quality"):
            VideoFrameExtractor(jpeg_quality=101)


class TestProtocolConformance:
    def test_supported_types(self):
        op = VideoFrameExtractor()
        assert op.supported_types == [RawDataType.VIDEO]

    def test_extractor_type(self):
        assert VideoFrameExtractor().extractor_type == "video_frame_extractor"

    def test_schema_name(self):
        assert VideoFrameExtractor().schema_name == "video_frame_extractor_v1"

    def test_info_has_config_schema(self):
        info = VideoFrameExtractor().info
        assert info.operator_type == "video_frame_extractor"
        assert "fps" in info.config_schema["properties"]
        assert "keyframe_threshold" in info.config_schema["properties"]


# ---------------------------------------------------------------------------
# Source path resolution
# ---------------------------------------------------------------------------

class TestSourcePathResolution:
    def test_source_path_takes_priority(self, tmp_path, static_video):
        op = VideoFrameExtractor(fps=5)
        # Wrong path in content; correct in source_path metadata.
        node = create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="/this/does/not/exist.mp4",
            account_id="acc",
            metadata={"source_path": str(static_video)},
        )
        result = op.extract(node, _ctx())
        assert result.error is None, result.error
        assert result.extracted_content["video_path"] == str(static_video)

    def test_legacy_video_path_metadata(self, tmp_path, static_video):
        op = VideoFrameExtractor(fps=5)
        node = create_raw_node(
            raw_type=RawDataType.VIDEO, content="",
            account_id="acc",
            metadata={"video_path": str(static_video)},
        )
        result = op.extract(node, _ctx())
        assert result.error is None
        assert result.extracted_content["video_path"] == str(static_video)

    def test_no_source_returns_error(self, tmp_path):
        op = VideoFrameExtractor(fps=5)
        node = create_raw_node(
            raw_type=RawDataType.VIDEO, content="",
            account_id="acc",
            metadata={},
        )
        result = op.extract(node, _ctx())
        assert result.error is not None
        assert "not found" in result.error

    def test_missing_file_returns_error(self, tmp_path):
        op = VideoFrameExtractor(fps=5)
        node = _make_raw_node(tmp_path / "no_such.mp4")
        result = op.extract(node, _ctx())
        assert result.error is not None
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Sampling behaviour
# ---------------------------------------------------------------------------

class TestSampling:
    def test_static_video_yields_only_first_keyframe(self, static_video):
        # 30 frames at 30 fps -> 1 second. At 5 fps sampling -> 5 frames
        # but only the first differs from "nothing" -> 1 keyframe.
        op = VideoFrameExtractor(fps=5, keyframe_threshold=0.05)
        node = _make_raw_node(static_video)
        result = op.extract(node, _ctx())
        assert result.error is None
        content = result.extracted_content
        assert content["frame_count"] == 5
        assert content["keyframe_count"] == 1
        # Only frame_idx 0 should be a keyframe
        keyframe_indices = [f["frame_idx"] for f in content["frames"] if f["keyframe"]]
        assert keyframe_indices == [0]

    def test_moving_video_marks_many_keyframes(self, moving_video):
        op = VideoFrameExtractor(fps=5, keyframe_threshold=0.05)
        node = _make_raw_node(moving_video)
        result = op.extract(node, _ctx())
        content = result.extracted_content
        assert content["frame_count"] == 5
        # Moving square -> significant pixel change between samples -> many keyframes
        assert content["keyframe_count"] >= 3

    def test_max_frames_caps_output(self, moving_video):
        op = VideoFrameExtractor(fps=30, max_frames=3)
        node = _make_raw_node(moving_video)
        result = op.extract(node, _ctx())
        assert result.extracted_content["frame_count"] == 3

    def test_keyframes_only_filters(self, alternating_video):
        # Sanity check: every sampled frame is a keyframe (huge pixel diff
        # between black/white). When we turn the filter on, the result
        # is unchanged but every returned frame has keyframe=True.
        op_all = VideoFrameExtractor(fps=30, keyframe_threshold=0.05)
        result_all = op_all.extract(_make_raw_node(alternating_video), _ctx())
        total = result_all.extracted_content["frame_count"]
        keyframes = result_all.extracted_content["keyframe_count"]
        assert total > 0
        assert keyframes == total  # alternating black/white: all are keyframes

        op_only = VideoFrameExtractor(
            fps=30, keyframe_threshold=0.05, require_keyframes_only=True,
        )
        result_only = op_only.extract(_make_raw_node(alternating_video), _ctx())
        # Filter keeps all (since all are keyframes)
        assert result_only.extracted_content["frame_count"] == keyframes
        assert all(f["keyframe"] for f in result_only.extracted_content["frames"])

    def test_keyframes_only_drops_non_keyframes(self, moving_video):
        # moving_video at 30fps with sampling at 5fps and a high
        # keyframe_threshold should give us a few keyframes interspersed
        # with non-keyframes. With require_keyframes_only, only the
        # keyframes survive.
        op_all = VideoFrameExtractor(fps=30, keyframe_threshold=0.5)
        result_all = op_all.extract(_make_raw_node(moving_video), _ctx())
        total = result_all.extracted_content["frame_count"]
        keyframes = result_all.extracted_content["keyframe_count"]
        assert total > keyframes > 0  # some but not all are keyframes

        op_only = VideoFrameExtractor(
            fps=30, keyframe_threshold=0.5, require_keyframes_only=True,
        )
        result_only = op_only.extract(_make_raw_node(moving_video), _ctx())
        assert result_only.extracted_content["frame_count"] == keyframes
        assert result_only.extracted_content["frame_count"] < total

    def test_resize_changes_dimensions(self, moving_video):
        # Source is 320x240; resize to 64 -> frames must report smaller dims
        op = VideoFrameExtractor(fps=5, resize=64)
        node = _make_raw_node(moving_video)
        result = op.extract(node, _ctx())
        frames = result.extracted_content["frames"]
        assert frames
        for f in frames:
            assert max(f["width"], f["height"]) <= 64

    def test_resize_none_keeps_native(self, moving_video):
        op = VideoFrameExtractor(fps=5, resize=None)
        result = op.extract(_make_raw_node(moving_video), _ctx())
        # Native is 320x240
        assert result.extracted_content["frames"][0]["width"] == 320
        assert result.extracted_content["frames"][0]["height"] == 240

    def test_jpeg_bytes_are_valid(self, moving_video):
        op = VideoFrameExtractor(fps=5, jpeg_quality=70)
        result = op.extract(_make_raw_node(moving_video), _ctx())
        for f in result.extracted_content["frames"]:
            raw = base64.b64decode(f["image_bytes_b64"])
            assert raw[:3] == b"\xff\xd8\xff", f"not a JPEG (frame {f['frame_idx']})"
            # Round-trip through cv2 to confirm it can be decoded
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            assert img is not None
            assert img.shape[2] == 3

    def test_zero_fps_video_returns_error(self, tmp_path):
        # A 1-frame mp4 has effective fps issues; OpenCV may report 0.
        # We write a deliberately-broken video by truncating the header.
        path = tmp_path / "broken.mp4"
        path.write_bytes(b"not a real video")
        op = VideoFrameExtractor(fps=5)
        result = op.extract(_make_raw_node(path), _ctx())
        assert result.error is not None


# ---------------------------------------------------------------------------
# Metadata + confidence
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_confidence_is_keyframe_fraction(self, moving_video):
        op = VideoFrameExtractor(fps=5, keyframe_threshold=0.05)
        result = op.extract(_make_raw_node(moving_video), _ctx())
        content = result.extracted_content
        expected = content["keyframe_count"] / content["frame_count"]
        assert result.confidence == pytest.approx(expected)

    def test_confidence_zero_when_no_frames(self, tmp_path):
        # Empty / zero-frame video -> no sampled frames -> confidence 0
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        op = VideoFrameExtractor(fps=5)
        result = op.extract(_make_raw_node(path), _ctx())
        # Either error or empty frames; both should yield 0 confidence
        assert result.confidence == 0.0

    def test_ts_sec_increases_monotonically(self, moving_video):
        op = VideoFrameExtractor(fps=5)
        result = op.extract(_make_raw_node(moving_video), _ctx())
        ts = [f["ts_sec"] for f in result.extracted_content["frames"]]
        assert ts == sorted(ts)
        assert len(set(ts)) == len(ts)  # no duplicates

    def test_frame_idx_records_source_index(self, moving_video):
        op = VideoFrameExtractor(fps=5)
        result = op.extract(_make_raw_node(moving_video), _ctx())
        # 30fps video, sampled at 5fps -> indices are 0, 6, 12, 18, 24
        assert [f["frame_idx"] for f in result.extracted_content["frames"]] == [0, 6, 12, 18, 24]


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_try_import_cv2_succeeds_when_present(self):
        cv2_mod, np_mod = _try_import_cv2()
        assert cv2_mod is not None
        assert np_mod is not None
