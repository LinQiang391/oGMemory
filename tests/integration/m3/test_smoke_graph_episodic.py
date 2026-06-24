"""Tests for :mod:`m3.smoke_graph_episodic`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3.smoke_graph_episodic import (
    DEFAULT_DATA_JSONL,
    DEFAULT_FRAMES_JSONL,
    DEFAULT_OUTPUT_JSONL,
    load_env_file,
    parse_args,
    run_graph_smoke,
    write_frames_manifest,
    write_video_manifest,
)

# ---------------------------------------------------------------------------
# Env + args
# ---------------------------------------------------------------------------

class TestEnvLoading:
    def test_load_env_file_sets_missing_keys(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("OGMEM_MINIMAX_API_KEY=abc\nX='quoted'\n")
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        loaded = load_env_file(env)
        assert loaded["OGMEM_MINIMAX_API_KEY"] == "abc"
        assert loaded["X"] == "quoted"

    def test_load_env_file_noop_when_missing(self, tmp_path):
        assert load_env_file(tmp_path / "missing.env") == {}


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.video.endswith("data/m3/videos/robot/kitchen_15.mp4")
        assert args.video_records_manifest == DEFAULT_DATA_JSONL
        assert args.frames_manifest == DEFAULT_FRAMES_JSONL
        assert args.output_jsonl == DEFAULT_OUTPUT_JSONL
        assert args.llm == "minimax"
        assert args.max_frames == 2
        assert args.fps == 1
        assert args.resize == 512
        assert args.no_audio is True
        assert args.load_env is True

    def test_overrides(self, tmp_path):
        args = parse_args([
            "--video", "v.mp4",
            "--clip-id", "7",
            "--video-records-manifest", str(tmp_path / "data.jsonl"),
            "--frames-manifest", str(tmp_path / "frames.jsonl"),
            "--frames-root", str(tmp_path / "frames"),
            "--output-jsonl", str(tmp_path / "out.jsonl"),
            "--llm", "mock",
            "--max-frames", "3",
            "--fps", "2",
            "--resize", "0",
            "--jpeg-quality", "70",
            "--with-audio",
            "--no-load-env",
            "-v",
        ])
        assert args.video == "v.mp4"
        assert args.clip_id == 7
        assert args.llm == "mock"
        assert args.max_frames == 3
        assert args.fps == 2
        assert args.resize is None
        assert args.jpeg_quality == 70
        assert args.no_audio is False
        assert args.load_env is False
        assert args.verbose is True

    def test_unknown_llm_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--llm", "magic"])


# ---------------------------------------------------------------------------
# Manifest writers
# ---------------------------------------------------------------------------

class TestManifestWriters:
    def test_write_video_manifest(self, tmp_path):
        class Node:
            node_id = "v-node"
            metadata = {"video_id": "vid", "clip_id": 3}
        path = tmp_path / "data.jsonl"
        rec = write_video_manifest(path, video_node=Node())
        assert rec == {"id": "vid", "raw_nodes": [{"clip_id": 3, "raw_node_id": "v-node"}]}
        assert json.loads(path.read_text()) == rec

    def test_write_frames_manifest(self, tmp_path):
        path = tmp_path / "frames.jsonl"
        result = {
            "video_id": "vid",
            "clip_id": 0,
            "video_raw_node_id": "v-node",
            "extracted_node_id": "ext",
            "child_node_ids": ["f1", "f2"],
            "child_count": 2,
            "resumed": False,
        }
        rec = write_frames_manifest(path, frame_result=result)
        assert rec == result
        assert json.loads(path.read_text()) == result


# ---------------------------------------------------------------------------
# Full in-process smoke (mock LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 64),
    )
    try:
        for i in range(30):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :, 2] = min(255, i * 8)
            cv2.circle(frame, (20 + i % 30, 32), 10, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


class TestRunGraphSmoke:
    def test_mock_graph_smoke_outputs_manifests_and_json(self, tmp_path, tiny_video):
        args = parse_args([
            "--video", str(tiny_video),
            "--llm", "mock",
            "--max-frames", "2",
            "--fps", "2",
            "--video-records-manifest", str(tmp_path / "data.jsonl"),
            "--frames-manifest", str(tmp_path / "frames.jsonl"),
            "--frames-root", str(tmp_path / "frames"),
            "--output-jsonl", str(tmp_path / "episodic.jsonl"),
            "--no-load-env",
        ])
        summary = run_graph_smoke(args)
        assert Path(args.video_records_manifest).exists()
        assert Path(args.frames_manifest).exists()
        assert Path(args.output_jsonl).exists()
        assert summary["frame_record"]["child_count"] == 2
        ep = summary["episodic_record"]
        assert ep["description"]
        assert ep["frame_count"] == 2
        assert ep["has_audio"] is False
        assert ep["resumed"] is False

    def test_main_writes_output(self, tmp_path, tiny_video, capsys):
        from m3.smoke_graph_episodic import main
        out = tmp_path / "episodic.jsonl"
        rc = main([
            "--video", str(tiny_video),
            "--llm", "mock",
            "--max-frames", "2",
            "--fps", "2",
            "--video-records-manifest", str(tmp_path / "data.jsonl"),
            "--frames-manifest", str(tmp_path / "frames.jsonl"),
            "--frames-root", str(tmp_path / "frames"),
            "--output-jsonl", str(out),
            "--no-load-env",
        ])
        assert rc == 0
        assert out.exists()
        captured = capsys.readouterr()
        assert "wrote" in captured.out

    def test_missing_video_raises(self, tmp_path):
        args = parse_args([
            "--video", str(tmp_path / "missing.mp4"),
            "--llm", "mock",
            "--no-load-env",
        ])
        with pytest.raises(FileNotFoundError):
            run_graph_smoke(args)
