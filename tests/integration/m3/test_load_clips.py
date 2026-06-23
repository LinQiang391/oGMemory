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
            clip_seconds=30,
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
            clip_seconds=30,
            ffprobe_bin="ffprobe",
        )
        node = graph.get_raw_node(records[1]["raw_node_id"], ctx)
        assert node is not None
        assert node.metadata["clip_id"] == 1
        assert node.metadata["start_sec"] == 30
        assert node.metadata["video_id"] == "vid"
        assert Path(node.metadata["source_path"]).exists()

    def test_start_sec_honours_clip_seconds(self, tmp_path):
        # Regression for the hardcoded `clip_id * 30` bug: when the
        # slicer used a non-default clip length, the loader's start_sec
        # must follow that value, not 30.
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        for i in range(4):
            (out_dir / f"{i}.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        records = load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=discover_clip_files(clips, "vid"),
            clip_seconds=60,
            ffprobe_bin="ffprobe",
        )
        # clip_id 0 -> 0s, clip_id 1 -> 60s, clip_id 2 -> 120s, ...
        for r in records:
            node = graph.get_raw_node(r["raw_node_id"], ctx)
            assert node.metadata["start_sec"] == r["clip_id"] * 60

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
            clip_seconds=30,
            ffprobe_bin="ffprobe",
        )
        expected = hashlib.sha1(b"vid/5").hexdigest()[:32]
        assert records[0]["raw_node_id"] == expected

    def test_references_edges_link_clips_in_time_order(self, tmp_path):
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        for i in range(4):
            (out_dir / f"{i}.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        records = load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=discover_clip_files(clips, "vid"),
            clip_seconds=30,
            ffprobe_bin="ffprobe",
        )
        # 4 clips -> 3 sequential references edges (0->1, 1->2, 2->3)
        from core.graph_enums import GraphEdgeType
        for prev, curr in zip(records, records[1:]):
            neighbors = graph.get_neighbors(
                prev["raw_node_id"], ctx, edge_types=[GraphEdgeType.REFERENCES],
            )
            # (neighbor_node, edge, direction)
            assert any(n.node_id == curr["raw_node_id"] for n, _, _ in neighbors)

    def test_single_clip_creates_no_edges(self, tmp_path):
        clips = tmp_path / "clips"
        out_dir = clips / "vid"
        out_dir.mkdir(parents=True)
        (out_dir / "0.mp4").write_bytes(b"x")

        graph = _make_memory_graph()
        ctx = _make_ctx()
        load_clips_for_video(
            graph, ctx,
            video_id="vid",
            source_video_path=tmp_path / "src.mp4",
            clip_files=discover_clip_files(clips, "vid"),
            clip_seconds=30,
            ffprobe_bin="ffprobe",
        )
        from core.graph_enums import GraphEdgeType
        neighbors = graph.get_neighbors(
            (out_dir / "0.mp4").name and "", ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        assert neighbors == []  # not testable as-is; rely on stats instead

        # Better assertion: graph has zero edges.
        node_id = hashlib.sha1(b"vid/0").hexdigest()[:32]
        edges = graph.get_neighbors(node_id, ctx, edge_types=[GraphEdgeType.REFERENCES])
        assert edges == []


class TestSQLDuplicateHandling:
    """The deterministic node_id + get_node-first pattern makes the
    loader idempotent for both MemoryGraphStore and SQLGraphStore
    without relying on exception-driven control flow."""

    def test_sql_unique_violation_is_not_swallowed(self, tmp_path):
        # Regression for the previous `except Exception` branch that
        # silently re-classified any error matching "duplicate" /
        # "unique" / "already exists" — including FK violations, ACL
        # errors, and connection drops. The fix is to do a get_node
        # probe up front and let real exceptions propagate.
        from m3 import load_clips

        class _FakeUniqueViolationError(Exception):
            pass

        class _RaisingStore:
            def __init__(self):
                self.created = 0
                self.edges = 0

            def get_raw_node(self, node_id, ctx):
                return None  # never short-circuits

            def create_raw_node(self, *a, **kw):
                self.created += 1
                raise _FakeUniqueViolationError("duplicate key value violates unique constraint")

            def create_edge(self, *a, **kw):
                self.edges += 1

        # Pre-create a real clip file so load_clips_for_video's loop
        # has something to iterate over.
        clip_path = tmp_path / "0.mp4"
        clip_path.write_bytes(b"x")
        graph = _RaisingStore()
        from core.models import RequestContext
        ctx = RequestContext(account_id="acc", user_id="u", agent_id="a",
                             session_id="s", trace_id="t")
        with pytest.raises(_FakeUniqueViolationError):
            load_clips.load_clips_for_video(
                graph, ctx,
                video_id="vid",
                source_video_path=tmp_path / "src.mp4",
                clip_files=[clip_path],
                clip_seconds=30,
                ffprobe_bin="ffprobe",
            )
        # The first create_raw_node must raise; the loader must NOT
        # swallow it under a broad except (which is the regression).
        assert graph.created == 1
        # Edge creation must not be reached on a per-clip failure.
        assert graph.edges == 0


class TestRun:
    def test_end_to_end_writes_manifest(
        self, tmp_path, sample_annotation, sample_video_layout
    ):
        for vid in ["living_room_06", "kitchen_22"]:
            cd = tmp_path / "clips" / vid
            cd.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (cd / f"{i}.mp4").write_bytes(b"x")

        out_jsonl = tmp_path / "data.jsonl"
        args = parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
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
            # source videos are resolved to real paths now
            assert r["video_path"] != ""
            assert Path(r["video_path"]).exists()

    def test_idempotent_repeated_run(self, tmp_path, sample_annotation, sample_video_layout):
        """Running the loader twice must not duplicate nodes or edges.

        The run() function builds its own GraphService internally, so
        we monkey-patch build_graph_service to inject a single
        MemoryGraphStore-backed service that we can assert on.
        """
        for vid in ["living_room_06", "kitchen_22"]:
            cd = tmp_path / "clips" / vid
            cd.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (cd / f"{i}.mp4").write_bytes(b"x")

        from core.models import RequestContext
        from providers.graph_store import MemoryGraphStore
        from service.graph_service import GraphService

        shared_graph = GraphService(MemoryGraphStore())
        ctx = RequestContext(
            account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
            session_id="s", trace_id="t",
        )

        def _fake_build(*a, **kw):
            return shared_graph, "memory"

        import m3.load_clips as lc
        original = lc.build_graph_service
        lc.build_graph_service = _fake_build
        try:
            for _ in range(2):
                run(parse_args([
                    "--annotations", str(sample_annotation),
                    "--clips-root", str(tmp_path / "clips"),
                    "--videos-root", str(sample_video_layout["data_root"] / "videos"),
                    "--output-jsonl", str(tmp_path / "data.jsonl"),
                ]))
        finally:
            lc.build_graph_service = original

        stats = shared_graph.get_graph_stats(ctx)
        # 2 videos x 2 clips = 4 raw nodes, no duplicates
        assert stats.raw_nodes == 4
        # Sequential references edges: 2 videos x 1 edge each = 2
        assert stats.total_edges == 2

    def test_account_id_override_propagates(self, tmp_path, sample_annotation, sample_video_layout):
        for vid in ["living_room_06", "kitchen_22"]:
            cd = tmp_path / "clips" / vid
            cd.mkdir(parents=True, exist_ok=True)
            (cd / "0.mp4").write_bytes(b"x")

        run(parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
            "--output-jsonl", str(tmp_path / "data.jsonl"),
            "--account-id", "tenant-x",
        ]))
        lines = (tmp_path / "data.jsonl").read_text().strip().split("\n")
        for line in lines:
            assert json.loads(line)["account_id"] == "tenant-x"

    def test_missing_clips_recorded_as_failure(
        self, tmp_path, sample_annotation, sample_video_layout
    ):
        (tmp_path / "clips").mkdir()
        rc = run(parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
            "--output-jsonl", str(tmp_path / "data.jsonl"),
        ]))
        assert rc == 2
        missing = json.loads((tmp_path / "load_missing_videos.json").read_text())
        assert set(missing) == {"living_room_06", "kitchen_22"}

    def test_all_clips_mode_processes_unannotated_videos(
        self, tmp_path, sample_annotation, sample_video_layout
    ):
        # --all-clips should pick up video dirs even if missing from annotations.
        extra_vid = "extra_vid"
        cd = tmp_path / "clips" / extra_vid
        cd.mkdir(parents=True)
        (cd / "0.mp4").write_bytes(b"x")

        # Make sure the annotated videos don't have clips, so the only
        # processed one is the extra.
        rc = run(parse_args([
            "--annotations", str(sample_annotation),
            "--clips-root", str(tmp_path / "clips"),
            "--videos-root", str(sample_video_layout["data_root"] / "videos"),
            "--output-jsonl", str(tmp_path / "data.jsonl"),
            "--all-clips",
        ]))
        assert rc == 0
        records = [json.loads(line) for line in
                   (tmp_path / "data.jsonl").read_text().strip().split("\n")]
        assert {r["id"] for r in records} == {extra_vid}


class TestParseArgs:
    def test_defaults_to_memory_store(self):
        args = parse_args(["--annotations", "x.json"])
        assert args.graph_store == "memory"
        assert args.account_id == M3BENCH_ACCOUNT_ID
        assert args.only_annotated is True
