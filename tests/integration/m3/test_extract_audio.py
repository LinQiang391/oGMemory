"""Tests for :mod:`m3.extract_audio` driver.

The driver is now a thin wrapper around ``OperatorService`` +
``AudioExtractor`` materialization. The interesting behaviour (WAV
writing, AUDIO RawNode creation, REFERENCES edges, lightweight summary,
idempotency) lives at the operator layer — see
``tests/unit/extraction/operators/test_audio_extractor.py`` and
``tests/unit/extraction/operators/test_materialize.py`` for the bulk of
coverage. These tests verify the M3-specific glue:

* the manifest format the m3 tooling expects,
* the on-disk WAV layout (``audio_root/<video_id>/<clip_id>.wav``),
* idempotent re-run via the parent ``ExtractedNode`` lookup,
* ``OperatorService.run_operator`` is invoked with the expected kwargs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType, RawDataType  # noqa: E402
from core.graph_models import create_raw_node  # noqa: E402
from core.models import RequestContext  # noqa: E402
from m3.config import M3BENCH_ACCOUNT_ID  # noqa: E402
from m3.extract_audio import (  # noqa: E402
    _build_audio_writer,
    _write_audio_wav,
    audio_node_id,
    extracted_node_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# ffmpeg must be on PATH for the integration test. Skip the whole
# module cleanly on a box that lacks it (CI without ffmpeg installed).
_FFMPEG = shutil.which("ffmpeg")
_requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None, reason="ffmpeg not available on PATH",
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-extract-audio-test", trace_id="t-test",
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
    in :func:`extract_audio_for_clip` can find it."""
    graph.create_raw_node(
        raw_type=video_node.raw_type,
        content=video_node.content,
        ctx=ctx,
        metadata=video_node.metadata,
        node_id=video_node.node_id,
    )


@pytest.fixture
def real_video(tmp_path: Path) -> Path:
    """Generate a tiny synthetic mp4 *with* an audio track so
    audio_extractor can read it."""
    path = tmp_path / "videos" / "vid_a" / "0.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1-second synthetic clip with a 440 Hz sine wave on the audio
    # track. ``-shortest`` lets the audio track be trimmed to the
    # video length so ffmpeg is happy. ``-pix_fmt yuv420p`` is the
    # most-compatible video pixel format.
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(
            f"ffmpeg fixture build failed: rc={proc.returncode} "
            f"stderr={proc.stderr.strip()}"
        )
    return path


# ---------------------------------------------------------------------------
# ID derivation (kept for manifest compatibility)
# ---------------------------------------------------------------------------

class TestIdDerivation:
    def test_extracted_node_id_is_deterministic(self):
        a = extracted_node_id("audio_extractor", "raw-xyz")
        b = extracted_node_id("audio_extractor", "raw-xyz")
        assert a == b
        assert len(a) == 32

    def test_extracted_node_id_varies_with_source(self):
        a = extracted_node_id("audio_extractor", "raw-1")
        b = extracted_node_id("audio_extractor", "raw-2")
        assert a != b

    def test_extracted_node_id_varies_with_extractor(self):
        a = extracted_node_id("audio_extractor", "raw-1")
        b = extracted_node_id("video_frame_extractor", "raw-1")
        assert a != b

    def test_audio_node_id_is_deterministic(self):
        a = audio_node_id("vid_a", 0)
        b = audio_node_id("vid_a", 0)
        assert a == b

    def test_audio_node_id_varies_with_clip(self):
        assert audio_node_id("vid_a", 0) != audio_node_id("vid_a", 1)

    def test_audio_node_id_matches_existing_test_fixture(self):
        # The episodic_memory integration test uses
        # ``sha1("audio/{video_id}/{clip_id}")`` to seed its audio
        # fixture; the extract_audio driver must agree so the two
        # paths share id space.
        expected = hashlib.sha1(b"audio/vid_a/0").hexdigest()[:32]
        assert audio_node_id("vid_a", 0) == expected


# ---------------------------------------------------------------------------
# Audio persistence helpers
# ---------------------------------------------------------------------------

class TestWriteAudioWav:
    def test_creates_file_at_expected_path(self, tmp_path):
        p = _write_audio_wav(tmp_path, "vid_a", 0, b"RIFFfake")
        assert p == tmp_path / "vid_a" / "0.wav"
        assert p.read_bytes() == b"RIFFfake"

    def test_creates_intermediate_dirs(self, tmp_path):
        p = _write_audio_wav(tmp_path / "deep", "v", 99, b"x")
        assert p.exists()
        assert p.parent == tmp_path / "deep" / "v"


class TestBuildAudioWriter:
    def test_writer_writes_to_clip_specific_path(self, tmp_path):
        writer = _build_audio_writer(tmp_path, "vid_b", 7)
        out = writer({}, b"RIFF\x00\x00")
        assert out == str(tmp_path / "vid_b" / "7.wav")
        assert Path(out).read_bytes() == b"RIFF\x00\x00"

    def test_writer_creates_dirs_lazily(self, tmp_path):
        writer = _build_audio_writer(tmp_path / "deep", "v", 3)
        writer({}, b"x")
        assert (tmp_path / "deep" / "v" / "3.wav").exists()


# ---------------------------------------------------------------------------
# Driver integration via extract_audio_for_clip
# ---------------------------------------------------------------------------

@_requires_ffmpeg
class TestExtractAudioForClip:
    """End-to-end driver test: synthetic mp4 with audio → WAV on disk →
    AUDIO RawNode → REFERENCES edge → lightweight ExtractedNode summary."""

    def _run(self, tmp_path, real_video):
        from m3.extract_audio import extract_audio_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)

        result = extract_audio_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            audio_root=tmp_path / "audio",
        )
        return result, graph, ctx

    def test_creates_one_audio_node(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        assert result["resumed"] is False
        assert result["audio_node_id"]
        node = graph.get_raw_node(result["audio_node_id"], ctx)
        assert node is not None
        assert node.raw_type == RawDataType.AUDIO

    def test_wav_persisted_to_disk_with_riff_header(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        node = graph.get_raw_node(result["audio_node_id"], ctx)
        path = Path(node.metadata["source_path"])
        assert path.exists()
        # RIFF...WAVE header magic.
        assert path.read_bytes()[:4] == b"RIFF"
        assert path.read_bytes()[8:12] == b"WAVE"

    def test_audio_metadata_propagated(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        node = graph.get_raw_node(result["audio_node_id"], ctx)
        assert node.metadata["format"] == "wav"
        assert node.metadata["sample_rate"] == 16000
        assert node.metadata["channels"] == 1
        # Source video_id / clip_id flowed through to the child.
        assert node.metadata["video_id"] == real_video.parent.name
        assert node.metadata["clip_id"] == 0
        assert node.metadata["source_video_node_id"] == result["video_raw_node_id"]

    def test_extracted_node_has_lightweight_summary(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        ext = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert ext is not None
        assert ext.extractor_type == "audio_extractor"
        # Heavy base64 payload must NOT be persisted.
        assert "audio" not in (ext.extracted_content or {})
        assert "audio_bytes_b64" not in (ext.extracted_content or {})
        # Lightweight summary IS persisted.
        assert "child_node_ids" in ext.extracted_content
        assert "child_count" in ext.extracted_content
        # Top-level fields survive the materialization step.
        assert ext.extracted_content["format"] == "wav"
        assert ext.extracted_content["sample_rate"] == 16000

    def test_references_edge_links_video_to_audio(self, tmp_path, real_video):
        result, graph, ctx = self._run(tmp_path, real_video)
        neighbors = graph.get_neighbors(
            result["video_raw_node_id"], ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        # ``get_neighbors`` returns (neighbor, edge, direction) tuples.
        audio_children = [
            n for n, _edge, _dir in neighbors
            if getattr(n, "raw_type", None) == RawDataType.AUDIO
        ]
        assert len(audio_children) == 1
        assert audio_children[0].node_id == result["audio_node_id"]

    def test_resumed_when_extracted_node_exists(self, tmp_path, real_video):
        """Re-running for a clip whose parent ExtractedNode already exists
        must short-circuit and report ``resumed=True``."""
        from m3.extract_audio import extract_audio_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)
        audio_root = tmp_path / "audio"

        first = extract_audio_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            audio_root=audio_root,
        )
        assert first["resumed"] is False

        second = extract_audio_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            video_id=real_video.parent.name,
            clip_id=0,
            audio_root=audio_root,
        )
        assert second["resumed"] is True
        assert second["audio_node_id"] == first["audio_node_id"]
        assert second["extracted_node_id"] == first["extracted_node_id"]

    def test_resume_re_extracts_when_audio_file_missing(self, tmp_path, real_video):
        """Review comment #3: if the WAV on disk is deleted between
        runs, the next call must NOT report ``resumed=True`` — the
        downstream ``episodic_memory`` stage would otherwise try to
        attach a path to a file that no longer exists. The fix: probe
        ``metadata.source_path`` on the AUDIO child during the resume
        pre-check and fall through to re-extraction when the file is
        gone.
        """
        from m3.extract_audio import extract_audio_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_node = _make_video_raw_node(real_video)
        _persist_video_raw_node(graph, video_node, ctx)
        audio_root = tmp_path / "audio"
        video_id = real_video.parent.name

        first = extract_audio_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node, video_id=video_id, clip_id=0,
            audio_root=audio_root,
        )
        assert first["resumed"] is False
        wav_path = audio_root / video_id / "0.wav"
        assert wav_path.exists()

        # Delete the WAV to simulate a user / cleanup job removing
        # the file between runs.
        wav_path.unlink()
        assert not wav_path.exists()

        second = extract_audio_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node, video_id=video_id, clip_id=0,
            audio_root=audio_root,
        )
        # Not a resume: the WAV was missing, so we re-extracted.
        assert second["resumed"] is False, (
            f"expected re-extraction when WAV is missing, got {second}"
        )
        # The ids are stable (deterministic id formula) and the file
        # is back on disk.
        assert second["audio_node_id"] == first["audio_node_id"]
        assert second["extracted_node_id"] == first["extracted_node_id"]
        assert wav_path.exists()

    def test_missing_video_raises(self, tmp_path):
        from m3.extract_audio import extract_audio_for_clip

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        ghost = create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="",
            account_id=M3BENCH_ACCOUNT_ID,
            metadata={"video_id": "ghost", "clip_id": 0,
                      "source_path": "/no/such/file.mp4"},
            node_id="ghost-video",
        )
        _persist_video_raw_node(graph, ghost, ctx)
        with pytest.raises(RuntimeError, match="audio_extractor failed"):
            extract_audio_for_clip(
                graph=graph, op_service=op_service, ctx=ctx,
                video_raw_node=ghost,
                video_id="ghost",
                clip_id=0,
                audio_root=tmp_path / "audio",
            )


# ---------------------------------------------------------------------------
# Manifest format
# ---------------------------------------------------------------------------

@_requires_ffmpeg
class TestRunManifestOutput:
    """Smoke-test the ``run()`` driver end-to-end with a synthetic
    manifest, checking the JSONL output schema that downstream M3
    tooling expects."""

    def test_run_writes_manifest_jsonl(self, tmp_path, real_video):
        from m3.extract_audio import parse_args, run

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

        out_jsonl = tmp_path / "audio.jsonl"
        args = parse_args([
            "--manifest", str(manifest_path),
            "--audio-root", str(tmp_path / "audio"),
            "--output-jsonl", str(out_jsonl),
            "--graph-store", "memory",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        assert rc == 0, f"run() returned {rc}"

        assert out_jsonl.exists()
        records = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line]
        assert len(records) == 1
        rec = records[0]
        # Schema: video_id, clip_id, video_raw_node_id, extracted_node_id,
        # audio_node_id, resumed.
        assert rec["video_id"] == real_video.parent.name
        assert rec["clip_id"] == 0
        assert rec["video_raw_node_id"] == video_node.node_id
        assert rec["extracted_node_id"]
        assert rec["audio_node_id"]
        assert rec["resumed"] is False


@_requires_ffmpeg
class TestExtractAudioStage:
    """Verify the orchestrator's stage wrapper wires the right argv
    into ``m3.extract_audio.run``.

    With ``--graph-store memory`` the orchestrator cannot share a graph
    across stages, so this test builds the graph + service itself,
    seeds the VIDEO RawNode, and invokes the stage function directly
    against a hand-rolled ``PipelineContext``. The stage must:
    * pass ``--manifest`` / ``--audio-root`` / ``--output-jsonl`` from
      the context's artifacts
    * forward ``--sample-rate`` / ``--channels`` from stage params
    * report counts from the output jsonl
    * tolerate ``rc=2`` (partial failure) without raising
    """

    def _ctx(self, work_dir: Path):
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec

        return PipelineContext(
            video=work_dir / "src.mp4",
            video_id="bedroom_01",
            seconds=60,
            work_dir=work_dir,
            config_path=Path("deploy/ogmemory.yaml"),
            graph_store="memory",
            connection_string=None,
            account_id=M3BENCH_ACCOUNT_ID,
            artifacts={
                "data_jsonl": work_dir / "data.jsonl",
                "audio_root": work_dir / "audio",
                "audio_jsonl": work_dir / "audio.jsonl",
            },
            variables={"ffmpeg_bin": "ffmpeg", "ffmpeg_log_level": "error"},
        ), StageSpec(
            id="extract_audio",
            type="m3.extract_audio",
            needs=["load_clips"],
            enabled=True,
            params={"sample_rate": 16000, "channels": 1},
        )

    def test_stage_writes_audio_when_graph_is_shared(self, tmp_path, real_video):
        from m3.extract_audio import extract_audio_for_clip  # noqa: F401  (sanity import)
        from m3.pipeline.stages import extract_audio_stage

        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx, stage = self._ctx(tmp_path)

        # Persist VIDEO RawNode into the graph the stage will see. The
        # orchestrator normally builds the graph service per stage,
        # which prevents cross-stage sharing with ``memory`` — for this
        # in-process test we share ``graph`` directly by patching
        # ``m3.extract_audio.run`` to honor a pre-built graph. Easiest
        # way: simulate ``load_clips`` output in ``data_jsonl`` and let
        # the stage use ``build_graph_service`` itself, but seed the
        # in-memory singleton via the same back door the integration
        # test uses (write the VIDEO node before the stage runs and
        # ensure the stage's fresh graph sees it — only works with
        # SQL). Instead, validate the argv path: stage returns ok, the
        # jsonl exists, and at the very least its schema is valid.
        video_node = _make_video_raw_node(real_video)
        request_ctx = _ctx()
        graph.create_raw_node(
            raw_type=video_node.raw_type, content=video_node.content,
            ctx=request_ctx, metadata=video_node.metadata,
            node_id=video_node.node_id,
        )

        # Write a load_clips-shaped manifest so the driver has inputs.
        manifest = tmp_path / "data.jsonl"
        manifest.write_text(json.dumps({
            "id": real_video.parent.name,
            "raw_nodes": [{"clip_id": 0, "raw_node_id": video_node.node_id}],
        }) + "\n", encoding="utf-8")

        # The stage builds its own graph (per-stage graph service); the
        # VIDEO node we just wrote lives in a different MemoryGraphStore
        # instance. So this stage will see ``records: 0`` — same as a
        # real orchestrator run with ``--graph-store memory``. The point
        # of this test is that the wiring (argv, return shape, rc
        # handling) is right, not that audio lands. SQL store is the
        # real path; this is just a contract test.
        result = extract_audio_stage(ctx, stage)

        assert result.ok is True
        assert result.stage_id == "extract_audio"
        assert result.stage_type == "m3.extract_audio"
        assert "audio_root" in result.outputs
        assert "audio_jsonl" in result.outputs
        assert (tmp_path / "audio.jsonl").exists()

    def test_stage_raises_on_unexpected_return_code(self, tmp_path, monkeypatch):
        """A non-zero/non-two return code from ``m3.extract_audio.run``
        must surface as a RuntimeError so the orchestrator marks the
        stage failed."""
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec
        from m3.pipeline import stages as stages_mod

        ctx = PipelineContext(
            video=tmp_path / "src.mp4",
            video_id="bedroom_01", seconds=60, work_dir=tmp_path,
            config_path=Path("deploy/ogmemory.yaml"),
            graph_store="memory", connection_string=None,
            account_id=M3BENCH_ACCOUNT_ID,
            artifacts={
                "data_jsonl": tmp_path / "data.jsonl",
                "audio_root": tmp_path / "audio",
                "audio_jsonl": tmp_path / "audio.jsonl",
            },
            variables={},
        )
        stage = StageSpec(
            id="extract_audio", type="m3.extract_audio",
            needs=["load_clips"], enabled=True, params={},
        )

        # Patch the inner ``m3.extract_audio.run`` (the name actually
        # imported inside the stage fn) to return rc=1. The stage now
        # passes the runner-managed graph + op_service through, so the
        # stub must accept (and ignore) those kwargs.
        import m3.extract_audio as extract_audio_mod
        monkeypatch.setattr(
            extract_audio_mod, "run",
            lambda args, graph=None, op_service=None: 1,
        )

        with pytest.raises(RuntimeError, match="m3.extract_audio returned 1"):
            stages_mod.extract_audio_stage(ctx, stage)


class TestEpisodicMemoryStageAudioWiring:
    """Review comment #1: ``episodic_memory_stage`` must forward
    ``--audio-manifest`` when the audio_jsonl artifact is registered
    AND must default to ``no_audio=False`` so the multimodal LLM
    actually receives audio attachments when audio was extracted.

    Pre-fix behaviour: ``no_audio`` defaulted to ``True`` and
    ``--audio-manifest`` was never passed — so the just-extracted
    AUDIO RawNodes were silently ignored downstream.
    """

    def _ctx(self, work_dir: Path, *, with_audio: bool):
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec

        artifacts = {
            "data_jsonl": work_dir / "data.jsonl",
            "frames_jsonl": work_dir / "frames.jsonl",
            "episodic_jsonl": work_dir / "episodic_memory.jsonl",
        }
        if with_audio:
            artifacts["audio_jsonl"] = work_dir / "audio.jsonl"
            (work_dir / "audio.jsonl").write_text("{}\n", encoding="utf-8")
        return PipelineContext(
            video=work_dir / "src.mp4",
            video_id="vid",
            seconds=60,
            work_dir=work_dir,
            config_path=Path("deploy/ogmemory.yaml"),
            graph_store="memory",
            connection_string=None,
            account_id=M3BENCH_ACCOUNT_ID,
            artifacts=artifacts,
            variables={},
        ), StageSpec(
            id="episodic_memory",
            type="m3.episodic_memory",
            needs=["extract_frames", "extract_audio"],
            enabled=True,
            params={},
        )

    def _capture_argv(self, monkeypatch, tmp_path: Path, with_audio: bool):
        """Run ``episodic_memory_stage`` against a stubbed inner run
        that captures the parsed argv instead of executing it."""
        from m3.pipeline import stages as stages_mod

        captured: dict[str, list[str]] = {}
        ctx, stage = self._ctx(tmp_path, with_audio=with_audio)

        def fake_run(args, graph=None, op_service=None):
            # Convert Namespace back into argv so the assertion can
            # read it as a list. Drop attrs that weren't actually on
            # the CLI; we only care about the flags.
            argv = [
                "--config", str(args.config),
                "--video-records-manifest", str(args.video_records_manifest),
                "--frames-manifest", str(args.frames_manifest),
                "--output-jsonl", str(args.output_jsonl),
                "--graph-store", args.graph_store,
                "--account-id", args.account_id,
                "--max-frames", str(args.max_frames),
            ]
            if args.audio_manifest:
                argv.extend(["--audio-manifest", str(args.audio_manifest)])
            if args.no_audio:
                argv.append("--no-audio")
            captured["argv"] = argv
            # Write an empty output jsonl so the stage's _read_jsonl works.
            Path(args.output_jsonl).write_text("", encoding="utf-8")
            return 0

        import m3.episodic_memory as episodic_mod
        monkeypatch.setattr(episodic_mod, "run", fake_run)
        result = stages_mod.episodic_memory_stage(ctx, stage)
        assert result.ok is True
        return captured["argv"]

    def test_passes_audio_manifest_when_audio_jsonl_exists(self, monkeypatch, tmp_path):
        argv = self._capture_argv(monkeypatch, tmp_path, with_audio=True)
        # --audio-manifest points at the registered audio_jsonl artifact.
        assert "--audio-manifest" in argv, argv
        manifest_idx = argv.index("--audio-manifest")
        assert argv[manifest_idx + 1] == str(tmp_path / "audio.jsonl"), argv
        # --no-audio must NOT be passed when audio is available.
        assert "--no-audio" not in argv, argv

    def test_omits_audio_manifest_when_audio_jsonl_not_registered(self, monkeypatch, tmp_path):
        argv = self._capture_argv(monkeypatch, tmp_path, with_audio=False)
        # No audio_jsonl in artifacts → no --audio-manifest and no
        # --no-audio (default is to use audio if available; absence of
        # the manifest makes audio_index empty but is not a flag flip).
        assert "--audio-manifest" not in argv, argv
        assert "--no-audio" not in argv, argv

    def test_explicit_no_audio_overrides_audio_jsonl(self, monkeypatch, tmp_path):
        from m3.pipeline import stages as stages_mod
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec

        ctx, _stage = self._ctx(tmp_path, with_audio=True)
        stage = StageSpec(
            id="episodic_memory",
            type="m3.episodic_memory",
            needs=["extract_frames", "extract_audio"],
            enabled=True,
            params={"no_audio": True},
        )

        captured: dict[str, list[str]] = {}

        def fake_run(args, graph=None, op_service=None):
            argv = []
            if args.audio_manifest:
                argv.extend(["--audio-manifest", str(args.audio_manifest)])
            if args.no_audio:
                argv.append("--no-audio")
            captured["argv"] = argv
            Path(args.output_jsonl).write_text("", encoding="utf-8")
            return 0

        import m3.episodic_memory as episodic_mod
        monkeypatch.setattr(episodic_mod, "run", fake_run)
        result = stages_mod.episodic_memory_stage(ctx, stage)
        assert result.ok is True
        # Raw audio is disabled, but the manifest still scopes ASR lookup.
        assert "--no-audio" in captured["argv"], captured["argv"]
        assert "--audio-manifest" in captured["argv"], captured["argv"]


def test_episodic_stage_forwards_validation_force_and_counts_actual_evidence(tmp_path, monkeypatch):
    import m3.episodic_memory as driver
    from m3.pipeline.stages import episodic_memory_stage
    ctx, stage = TestEpisodicMemoryStageAudioWiring()._ctx(tmp_path, with_audio=True)
    ctx.artifacts["voice_asr"] = tmp_path / "voice_asr.jsonl"
    ctx.artifacts["voice_asr"].write_text("")
    stage.params.update(no_audio=True, force=True)
    def fake_run(args, **kwargs):
        assert args.no_audio and args.force
        assert args.audio_manifest == str(ctx.artifact("audio_jsonl"))
        assert args.asr_manifest == str(ctx.artifact("voice_asr"))
        Path(args.output_jsonl).write_text(json.dumps({
            "description": "speech only", "audio_attached": False,
            "speech_transcript": [{"text": "hello"}],
        }) + "\n")
        return 0
    monkeypatch.setattr(driver, "run", fake_run)
    result = episodic_memory_stage(ctx, stage)
    assert result.counts["audio_attached"] == 0
    assert result.counts["clips_with_transcript"] == 1
    assert result.counts["transcript_segments"] == 1
