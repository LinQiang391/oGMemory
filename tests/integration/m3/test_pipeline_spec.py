"""Tests for declarative M3 pipeline specs."""

from __future__ import annotations

from pathlib import Path

from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import load_pipeline_spec


def test_load_default_video_episodic_spec():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    assert spec.name == "video_episodic"
    assert "selected_video" in spec.artifacts
    assert [s.id for s in spec.stages][:3] == ["trim", "annotate", "preprocess"]
    detect = next(s for s in spec.stages if s.id == "detect_faces")
    assert detect.enabled is False


def test_default_video_episodic_declares_audio_stage():
    """The audio stage must be declared after load_clips (VIDEO nodes
    exist) and before episodic_memory (walks the graph for audio)."""
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    stage_ids = [s.id for s in spec.stages]
    assert "extract_audio" in stage_ids
    audio = next(s for s in spec.stages if s.id == "extract_audio")
    assert audio.type == "m3.extract_audio"
    assert "load_clips" in audio.needs
    # Enabled by default — users run --disable-stage extract_audio to opt
    # out (saves the ffmpeg cost for vision-only smoke tests).
    assert audio.enabled is True
    # Position: extract_audio must precede episodic_memory so AUDIO
    # RawNodes land in the graph before episodic_memory looks for them.
    assert stage_ids.index("extract_audio") < stage_ids.index("episodic_memory")
    # Artifacts wired up so the stage can find its inputs/outputs.
    assert "audio_root" in spec.artifacts
    assert "audio_jsonl" in spec.artifacts


def test_faces_video_episodic_declares_audio_stage():
    """Same contract as the default pipeline — the faces variant must
    declare both the audio stage AND the audio artifacts it consumes.
    A previous revision shipped the stage entry without the artifact
    declarations, which made the stage crash with ``KeyError: unknown
    pipeline artifact: audio_root`` the moment it was enabled."""
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    stage_ids = [s.id for s in spec.stages]
    assert "extract_audio" in stage_ids
    audio = next(s for s in spec.stages if s.id == "extract_audio")
    assert audio.type == "m3.extract_audio"
    assert "load_clips" in audio.needs
    assert stage_ids.index("extract_audio") < stage_ids.index("episodic_memory")
    # Both artifact entries must be present; without these the stage
    # raises KeyError the first time it asks ctx.artifact("audio_root").
    assert "audio_root" in spec.artifacts
    assert "audio_jsonl" in spec.artifacts


def test_extract_audio_stage_is_registered():
    """Orchestrator rejects unknown stage types at validation; without
    this registration the YAML entry would either error or be skipped."""
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.extract_audio" in DEFAULT_STAGE_REGISTRY


def test_load_faces_spec_enables_detect_faces():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    detect = next(s for s in spec.stages if s.id == "detect_faces")
    assert detect.enabled is True


def test_runner_artifact_interpolation(tmp_path: Path):
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    runner = PipelineRunner()
    video = tmp_path / "bedroom_03.mp4"
    video.write_bytes(b"not real video")
    ctx = runner.build_context(
        spec,
        video=video,
        video_id="bedroom_03",
        seconds=120,
        work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory",
        connection_string=None,
        account_id="m3bench",
    )
    assert ctx.artifact("selected_video") == (tmp_path / "run" / "videos" / "robot" / "bedroom_03.mp4").resolve()
    assert ctx.artifact("summary_json") == (tmp_path / "run" / "pipeline_summary.json").resolve()


def test_stage_overrides_enable_disable_and_set():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    runner = PipelineRunner()
    runner.apply_overrides(
        spec,
        enable_stages=["detect_faces", "build_index"],
        disable_stages=["create_memory_nodes"],
        set_values=[
            "extract_frames.params.max_frames=16",
            "detect_faces.params.backend=mock",
        ],
    )
    stages = {s.id: s for s in spec.stages}
    assert stages["detect_faces"].enabled is True
    assert stages["build_index"].enabled is True
    assert stages["create_memory_nodes"].enabled is False
    assert stages["extract_frames"].params["max_frames"] == 16
    assert stages["detect_faces"].params["backend"] == "mock"
