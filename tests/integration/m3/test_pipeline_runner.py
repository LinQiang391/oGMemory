"""Tests for spec-driven m3.run_video_pipeline CLI behavior."""

from __future__ import annotations

from m3.run_video_pipeline import parse_args
from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import load_pipeline_spec


def test_legacy_detect_faces_and_build_index_flags_map_to_stage_overrides():
    args = parse_args([
        "--video", "v.mp4",
        "--detect-faces",
        "--build-index",
        "--face-backend", "mock",
        "--embedder", "mock",
        "--embedding-dimension", "384",
    ])
    spec = load_pipeline_spec(args.pipeline)
    runner = PipelineRunner()
    enable = []
    if args.detect_faces:
        enable.append("detect_faces")
    if args.build_index:
        enable.append("build_index")
    from m3.run_video_pipeline import _legacy_set_overrides
    runner.apply_overrides(spec, enable_stages=enable, set_values=_legacy_set_overrides(args))
    stages = {s.id: s for s in spec.stages}
    assert stages["detect_faces"].enabled is True
    assert stages["detect_faces"].params["backend"] == "mock"
    assert stages["build_index"].enabled is True
    assert stages["build_index"].params["embedder"] == "mock"
    assert stages["build_index"].params["embedding_dimension"] == 384


def test_parse_args_accepts_pipeline_and_set_overrides():
    args = parse_args([
        "--pipeline", "config/m3/pipelines/video_episodic_faces.yaml",
        "--video", "v.mp4",
        "--enable-stage", "build_index",
        "--disable-stage", "detect_faces",
        "--set", "extract_frames.params.max_frames=2",
    ])
    assert args.pipeline.endswith("video_episodic_faces.yaml")
    assert args.enable_stage == ["build_index"]
    assert args.disable_stage == ["detect_faces"]
    assert args.set == ["extract_frames.params.max_frames=2"]
