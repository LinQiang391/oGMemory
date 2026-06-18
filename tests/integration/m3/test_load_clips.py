"""Tests for :mod:`m3.load_clips` (clip → RawNode batch loader)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from m3.config import M3BENCH_ACCOUNT_ID
from m3.load_clips import (
    build_graph_service,
    clip_duration_seconds,
    clip_id_from_path,
    discover_clip_files,
    load_clips_for_video,
    main,
    parse_args,
    run,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestClipIdFromPath:
    def test_extracts_integer(self, tmp_path):
        p = tmp_path / "living_room_06" / "12.mp4"
        p.parent.mkdir()
        p.write_bytes(b"x")
        assert clip_id_from_path(p) == 12

    def test_zero(self, tmp_path):
        p = tmp_path / "v" / "0.mp4"
        p.parent.mkdir()
        p.write_bytes(b"x")
        assert clip_id_from_path(p) == 0


class TestDiscoverClipFiles:
    def test_returns_sorted_mp4s(self, tmp_path):
        d = tmp_path / "v"
        d.mkdir()
        (d / "0.mp4").write_bytes(b"a")
        (d / "1.mp4").write_bytes(b"b")
        (d / "2.mp4").write_bytes(b"c")
        (d / "ignore.txt").write_bytes(b"x")
        out = discover_clip_files(tmp_path, "v")
        assert [p.name for p in out] == ["0.mp4", "1.mp4", "2.mp4"]

    def test_skips_empty_files(self, tmp_path):
        d = tmp_path / "v"
        d.mkdir()
        (d / "0.mp4").write_bytes(b"")
        (d / "1.mp4").write_bytes(b"a")
        out = discover_clip_files(tmp_path, "v")
        assert [p.name for p in out] == ["1.mp4"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert discover_clip_files(tmp_path, "missing") == []


class TestClipDurationFallback:
    def test_falls_back_when_ffprobe_missing(self, tmp_path, monkeypatch):
        # Simulate ffprobe not on PATH
        import m3.preprocess as pp
        monkeypatch.setattr(pp.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"x")
        assert clip_duration_seconds(clip) == 30.0


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------

def _make_ctx():
    from core.models import RequestContext
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id="m3bench-user",
        agent_id="m3bench-agent",
        session_id="test",
        trace_id="trace-load-test",
    )


def _make_memory_graph():
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    return GraphService(MemoryGraphStore())


class TestBuildGraphService:
    def test_memory_store(self):
        graph, kind = build_graph_service("memory", None)
        assert kind == "memory"
        assert graph is not None

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown graph store"):
            build_graph_service("redis", None)


class TestLoadClipsForVideo:
    def test_inserts_one_raw_node_per_clip(self, tmp_path):
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        for i in range(3):
            (out_dir / f"{i}.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        records = load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=discover_clip_files(clips, "vid"),
            clips_root=clips,
            ffprobe_bin="ffprobe",
        )
        assert len(records) == 3
        for r in records:
            assert r["video_id"] == "vid"
            assert r["clip_id"] in {0, 1, 2}
            assert r["raw_node_id"]

        # Graph should hold three nodes
        stats = graph.get_graph_stats(ctx)
        assert stats.raw_nodes == 3

    def test_metadata_round_trips(self, tmp_path):
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        (out_dir / "0.mp4").write_bytes(b"x")
        (out_dir / "1.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        records = load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=discover_clip_files(clips, "vid"),
            clips_root=clips,
            ffprobe_bin="ffprobe",
        )
        node = graph.get_raw_node(records[1]["raw_node_id"], ctx)
        assert node is not None
        assert node.metadata["clip_id"] == 1
        assert node.metadata["start_sec"] == 30
        assert node.metadata["video_id"] == "vid"
        # Source path is absolute and points inside the clips dir
        assert Path(node.metadata["source_path"]).exists()

    def test_deterministic_node_id(self, tmp_path):
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        (out_dir / "5.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        records = load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=[out_dir / "5.mp4"],
            clips_root=clips,
            ffprobe_bin="ffprobe",
        )
        expected = hashlib.sha1(b"vid/5").hexdigest()[:32]
        assert records[0]["raw_node_id"] == expected


class TestRun:
    def test_end_to_end_writes_manifest(
        self, tmp_path, sample_annotation, sample_video_layout, monkeypatch
    ):
        # Pre-create clips on disk (the preprocess step is tested separately)
        for vid in ["living_room_06", "kitchen_22"]:
            cd = tmp_path / "clips" / vid
            cd.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (cd / f"{i}.mp4").write_bytes(b"x")

        out_jsonl = tmp_path / "data.jsonl"
        args = parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--output-jsonl", str(out_jsonl),
        ])
        rc = run(args)
        assert rc == 0

        lines = out_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert {r["id"] for r in records} == {"living_room_06", "kitchen_22"}
        for r in records:
            assert len(r["raw_nodes"]) == 2
            assert r["account_id"] == M3BENCH_ACCOUNT_ID

    def test_missing_clips_recorded_as_failure(
        self, tmp_path, sample_annotation, sample_video_layout
    ):
        # Clips root exists but is empty -> no clips to load -> rc=2.
        (tmp_path / "clips").mkdir()
        args = parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--output-jsonl", str(tmp_path / "data.jsonl"),
        ])
        rc = run(args)
        # All videos missing -> rc 2
        assert rc == 2
        missing = json.loads((tmp_path / "load_missing_videos.json").read_text())
        assert set(missing) == {"living_room_06", "kitchen_22"}


class TestParseArgs:
    def test_defaults_to_memory_store(self):
        args = parse_args(["--annotations", "x.json"])
        assert args.graph_store == "memory"
        assert args.account_id == M3BENCH_ACCOUNT_ID
        assert args.only_annotated is True
