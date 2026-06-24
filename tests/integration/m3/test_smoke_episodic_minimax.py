"""Tests for :mod:`m3.smoke_episodic_minimax`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3.smoke_episodic_minimax import (
    load_env_file,
    parse_args,
    run_smoke,
    summarise_minimax_env,
)

# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

class TestEnvLoading:
    def test_load_env_file_sets_missing_keys(self, tmp_path, monkeypatch):
        env = tmp_path / ".env.local"
        env.write_text("OGMEM_MINIMAX_API_KEY=abc123\n# comment\nBADLINE\nX='quoted'\n")
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        loaded = load_env_file(env)
        assert loaded["OGMEM_MINIMAX_API_KEY"] == "abc123"
        assert loaded["X"] == "quoted"

    def test_load_env_file_does_not_override_by_default(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("OGMEM_MINIMAX_API_KEY=file-key\n")
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "existing")
        loaded = load_env_file(env)
        assert "OGMEM_MINIMAX_API_KEY" not in loaded
        assert summarise_minimax_env()["api_key"] == "<set,len=8>"

    def test_load_env_file_override_true(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("OGMEM_MINIMAX_API_KEY=file-key\n")
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "existing")
        loaded = load_env_file(env, override=True)
        assert loaded["OGMEM_MINIMAX_API_KEY"] == "file-key"
        assert summarise_minimax_env()["api_key"] == "<set,len=8>"

    def test_missing_file_noop(self, tmp_path):
        assert load_env_file(tmp_path / "missing.env") == {}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.video.endswith("data/m3/videos/robot/kitchen_15.mp4")
        assert args.output_json == "/tmp/episodic_smoke.json"
        assert args.llm == "minimax"
        assert args.model == "MiniMax-M3"
        assert args.max_frames == 2
        assert args.resize == 512
        assert args.image_detail == "low"
        assert args.load_env is True

    def test_resize_zero_disables_resize(self):
        args = parse_args(["--resize", "0"])
        assert args.resize is None

    def test_overrides(self, tmp_path):
        out = tmp_path / "out.json"
        args = parse_args([
            "--video", "v.mp4",
            "--output-json", str(out),
            "--llm", "mock",
            "--model", "custom",
            "--base-url", "https://api.minimaxi.com/v1/chat/completions",
            "--temperature", "0.1",
            "--max-tokens", "512",
            "--max-frames", "3",
            "--fps", "2",
            "--resize", "384",
            "--jpeg-quality", "70",
            "--image-detail", "auto",
            "--no-load-env",
            "-v",
        ])
        assert args.video == "v.mp4"
        assert args.output_json == str(out)
        assert args.llm == "mock"
        assert args.model == "custom"
        assert args.base_url == "https://api.minimaxi.com/v1/chat/completions"
        assert args.temperature == 0.1
        assert args.max_tokens == 512
        assert args.max_frames == 3
        assert args.fps == 2
        assert args.resize == 384
        assert args.jpeg_quality == 70
        assert args.image_detail == "auto"
        assert args.load_env is False
        assert args.verbose is True

    def test_invalid_llm_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--llm", "magic"])


# ---------------------------------------------------------------------------
# run_smoke
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
            frame[:, :, 1] = min(255, i * 8)
            cv2.rectangle(frame, (10 + i % 20, 15), (40 + i % 20, 45), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


class TestRunSmoke:
    def test_mock_smoke_outputs_expected_json(self, tiny_video):
        args = parse_args([
            "--video", str(tiny_video),
            "--llm", "mock",
            "--max-frames", "2",
            "--fps", "2",
            "--no-load-env",
        ])
        out = run_smoke(args)
        assert out["description"] == "Mock episodic memory for smoke testing."
        assert out["frame_count"] == 2
        assert out["has_audio"] is False

    def test_missing_video_raises(self, tmp_path):
        args = parse_args([
            "--video", str(tmp_path / "missing.mp4"),
            "--llm", "mock",
            "--no-load-env",
        ])
        with pytest.raises(FileNotFoundError):
            run_smoke(args)

    def test_main_writes_output_file(self, tiny_video, tmp_path, capsys):
        from m3.smoke_episodic_minimax import main
        out = tmp_path / "smoke.json"
        rc = main([
            "--video", str(tiny_video),
            "--llm", "mock",
            "--max-frames", "2",
            "--output-json", str(out),
            "--no-load-env",
        ])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["description"] == "Mock episodic memory for smoke testing."
        captured = capsys.readouterr()
        assert "wrote" in captured.out
