"""Tests for spec-driven m3.run_video_pipeline CLI behavior."""

from __future__ import annotations

import pytest

from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import load_pipeline_spec
from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
from m3.run_video_pipeline import parse_args


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
    runner.apply_overrides(spec, enable_stages=enable, set_values=[v for v in _legacy_set_overrides(args)
        if v.split(".", 1)[0] in {stage.id for stage in spec.stages}])
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


def test_audio_default_preserves_yaml_and_explicit_flags_override():
    from m3.run_video_pipeline import _legacy_set_overrides
    for flags, expected in [([], None), (["--with-audio"], False), (["--no-audio"], True)]:
        args = parse_args(["--video", "v.mp4"] + flags)
        values = [v for v in _legacy_set_overrides(args) if ".no_audio=" in v]
        assert values == ([] if expected is None else
                          [f"episodic_memory.params.no_audio={str(expected).lower()}"])
        spec = load_pipeline_spec("config/m3/pipelines/video_episodic_full.yaml")
        PipelineRunner().apply_overrides(spec, set_values=values)
        memory = next(s for s in spec.stages if s.id == "episodic_memory")
        assert memory.params.get("no_audio", False) is (False if expected is None else expected)


def test_apply_overrides_coerces_bool_and_int_strings():
    """``--set`` values must be coerced so ``enabled=false`` lands as
    a real bool, not the truthy string ``"false"``.
    """
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    runner = PipelineRunner()
    runner.apply_overrides(
        spec,
        set_values=[
            "detect_faces.enabled=false",
            "extract_frames.params.max_frames=16",
            "episodic_memory.params.no_audio=no",
        ],
    )
    stages = {s.id: s for s in spec.stages}
    assert stages["detect_faces"].enabled is False
    assert stages["extract_frames"].params["max_frames"] == 16
    assert stages["episodic_memory"].params["no_audio"] is False


def test_apply_overrides_rejects_malformed_set():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic.yaml")
    runner = PipelineRunner()
    with pytest.raises(ValueError, match="stage.path=value"):
        runner.apply_overrides(spec, set_values=["no_equals_sign"])
    with pytest.raises(ValueError, match="stray '='"):
        runner.apply_overrides(
            spec, set_values=["extract_frames.params.max_frames=8=extra"],
        )
    with pytest.raises(ValueError, match="unknown stage"):
        runner.apply_overrides(
            spec, set_values=["nope.params.max_frames=1"],
        )


def test_load_pipeline_spec_validates_against_registry():
    """Surface typos at load time when a registry is supplied."""
    spec = load_pipeline_spec(
        "config/m3/pipelines/video_episodic.yaml",
        registry=DEFAULT_STAGE_REGISTRY,
    )
    assert spec.stages, "default spec should load"

    bad_yaml = tmp_yaml_factory(
        """
name: bad
version: 1
stages:
  - id: unknown_stage
    type: not.in.registry
""".strip(),
    )
    with pytest.raises(ValueError, match="unknown stage type"):
        load_pipeline_spec(bad_yaml, registry=DEFAULT_STAGE_REGISTRY)


def tmp_yaml_factory(text: str):
    """Inline helper to write a temp YAML and return its path."""
    import tempfile
    from pathlib import Path

    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    )
    fd.write(text)
    fd.close()
    return Path(fd.name)
