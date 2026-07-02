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
