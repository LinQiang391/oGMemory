"""Tests for :mod:`m3.preprocess` (ffmpeg slicing driver)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3.config import PreprocessConfig, video_id_from_path
from m3.preprocess import (
    cut_clip,
    expected_clip_count,
    ffprobe_duration,
    list_video_paths,
    main,
    parse_args,
    run,
    slice_video,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestExpectedClipCount:
    """``expected_clip_count`` must match M3-Agent's reference bash script
    which uses ``duration/30 + 1`` rounded up via integer division."""

    def test_exact_multiple(self):
        assert expected_clip_count(60.0, 30) == 2

    def test_partial_last_clip(self):
        # 65s -> 3 clips of [0-30, 30-60, 60-65]
        assert expected_clip_count(65.0, 30) == 3

    def test_subsecond_remainder(self):
        assert expected_clip_count(30.4, 30) == 2

    def test_very_short_video_still_one_clip(self):
        # Edge: 5s video should still produce 1 clip, not 0.
        assert expected_clip_count(5.0, 30) == 1


class TestVideoIdFromPath:
    """``video_id_from_path`` extracts the filename stem."""

    def test_simple(self):
        assert video_id_from_path("data/videos/robot/living_room_06.mp4") == "living_room_06"

    def test_absolute(self):
        assert video_id_from_path("/abs/data/videos/robot/kitchen_22.mp4") == "kitchen_22"

    def test_no_extension(self):
        assert video_id_from_path("living_room_06") == "living_room_06"

    def test_other_extension(self):
        assert video_id_from_path("/x/y/foo.webm") == "foo"


# ---------------------------------------------------------------------------
# Driver behaviour
# ---------------------------------------------------------------------------

class TestSliceVideo:
    def test_writes_expected_clip_files(self, tmp_path, fake_ffmpeg):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"\x00\x00\x00\x00ftypmp42")
        clips_root = tmp_path / "clips"

        # ffprobe returns 60s -> 2 clips of 30s
        written = slice_video(
            src, clips_root,
            clip_seconds=30,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            log_level="error",
        )
        assert written == 2
        assert (clips_root / "src" / "0.mp4").exists()
        assert (clips_root / "src" / "1.mp4").exists()
        # ffprobe + 2 ffmpeg calls
        assert len(fake_ffmpeg) == 3
        # Last ffmpeg call writes clip 1 starting at t=30
        last_cmd = fake_ffmpeg[-1]
        assert "-ss" in last_cmd and "30" in last_cmd
        assert "-t" in last_cmd and "30" in last_cmd

    def test_resume_is_noop_when_clips_exist(self, tmp_path, fake_ffmpeg):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"\x00\x00\x00\x00ftypmp42")
        clips_root = tmp_path / "clips"
        # Pre-create clip files
        out_dir = clips_root / "src"
        out_dir.mkdir(parents=True)
        (out_dir / "0.mp4").write_bytes(b"existing")
        (out_dir / "1.mp4").write_bytes(b"existing")

        written = slice_video(
            src, clips_root,
            clip_seconds=30,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            log_level="error",
        )
        assert written == 0
        # Only ffprobe called, no ffmpeg
        assert len(fake_ffmpeg) == 1

    def test_force_re_slices_existing(self, tmp_path, fake_ffmpeg):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"\x00\x00\x00\x00ftypmp42")
        clips_root = tmp_path / "clips"
        out_dir = clips_root / "src"
        out_dir.mkdir(parents=True)
        (out_dir / "0.mp4").write_bytes(b"existing")
        (out_dir / "1.mp4").write_bytes(b"existing")

        written = slice_video(
            src, clips_root,
            clip_seconds=30,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            log_level="error",
            force=True,
        )
        assert written == 2
        assert len(fake_ffmpeg) == 3


class TestListVideoPaths:
    def test_resolves_from_project_root(self, sample_annotation, sample_video_layout):
        # sample_annotation is at tmp_path/data/annotations/annotations.json
        # sample_video_layout["data_root"] is tmp_path/data
        paths = list_video_paths(
            sample_annotation,
            sample_video_layout["data_root"] / "videos",
            project_root=sample_video_layout["data_root"].parent,
        )
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert {p.stem for p in paths} == {"living_room_06", "kitchen_22"}

    def test_falls_back_to_videos_root(self, tmp_path):
        # Move data to a location where the project_root path doesn't exist,
        # so the fallback (videos_root / subset / stem) is exercised.
        annotations = {
            "vid_a": {"video_path": "data/videos/robot/vid_a.mp4", "mem_path": "x.pkl", "qa_list": []},
        }
        ann_dir = tmp_path / "ann"
        ann_dir.mkdir()
        ann_file = ann_dir / "ann.json"
        ann_file.write_text(json.dumps(annotations))
        # Place video at videos_root/robot/vid_a.mp4 (no project_root equivalent)
        vr = tmp_path / "videos_only"
        (vr / "robot").mkdir(parents=True)
        (vr / "robot" / "vid_a.mp4").write_bytes(b"x")
        paths = list_video_paths(
            ann_file, vr,
            project_root=tmp_path / "no_such_project_root",
        )
        assert len(paths) == 1
        assert paths[0].name == "vid_a.mp4"

    def test_warns_on_missing(self, sample_annotation, tmp_path, caplog):
        # Empty videos root and unreachable project root -> warnings logged
        with caplog.at_level("WARNING"):
            paths = list_video_paths(
                sample_annotation,
                tmp_path / "no_videos",
                project_root=tmp_path / "no_such_project_root",
            )
        assert paths == []
        assert any("not found" in r.message for r in caplog.records)


class TestRun:
    def test_end_to_end_writes_clips(
        self, tmp_path, sample_annotation, sample_video_layout, fake_ffmpeg
    ):
        args = parse_args([
            "--annotations", str(sample_annotation),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
            "--clips-root", str(tmp_path / "clips"),
            "--project-root", str(sample_video_layout["data_root"].parent),
        ])
        rc = run(args)
        assert rc == 0

        clips_root = tmp_path / "clips"
        assert (clips_root / "living_room_06" / "0.mp4").exists()
        assert (clips_root / "living_room_06" / "1.mp4").exists()
        assert (clips_root / "kitchen_22" / "0.mp4").exists()
        assert (clips_root / "kitchen_22" / "1.mp4").exists()

    def test_reports_failures_when_ffprobe_blows_up(
        self, tmp_path, sample_annotation, sample_video_layout, monkeypatch
    ):
        import m3.preprocess as pp
        def bad_probe(*args, **kwargs):
            raise RuntimeError("ffprobe exploded")
        monkeypatch.setattr(pp, "ffprobe_duration", bad_probe)

        args = parse_args([
            "--annotations", str(sample_annotation),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
            "--clips-root", str(tmp_path / "clips"),
            "--project-root", str(sample_video_layout["data_root"].parent),
        ])
        rc = run(args)
        assert rc == 2  # all videos failed
        failures_path = sample_video_layout["data_root"].parent / "preprocess_failures.json"
        assert failures_path.exists()
        failures = json.loads(failures_path.read_text())
        assert len(failures) == 2
        assert {f["video_id"] for f in failures} == {"living_room_06", "kitchen_22"}


class TestParseArgs:
    def test_required_annotation(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self):
        args = parse_args(["--annotations", "x.json"])
        assert args.clip_seconds == 30
        assert args.fps == 5
        assert args.videos_root == "data/videos"
        assert args.clips_root == "data/clips"
        assert args.force is False
