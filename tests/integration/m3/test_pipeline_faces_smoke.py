"""Tests that the faces preset actually enables the detect_faces stage."""

from __future__ import annotations

from pathlib import Path

from m3.pipeline import PipelineRunner, load_pipeline_spec
from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY


def test_faces_preset_enables_detect_faces():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    detect = next(s for s in spec.stages if s.id == "detect_faces")
    assert detect.enabled is True
    assert detect.type == "m3.detect_faces"
    assert "extract_frames" in detect.needs


def test_apply_overrides_preserves_yaml_enabled_state():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    runner = PipelineRunner()
    runner.apply_overrides(spec)
    detect = next(s for s in spec.stages if s.id == "detect_faces")
    assert detect.enabled is True


def test_faces_stage_dispatched_in_runner(tmp_path: Path):
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_faces.yaml")
    runner = PipelineRunner()
    runner.apply_overrides(spec)
    runner.validate(spec)

    executed: list[str] = []
    ctx_args: list[tuple] = []

    def fake_stage(ctx, stage):
        executed.append(stage.id)
        ctx_args.append((ctx.video, stage.id, stage.enabled))
        from m3.pipeline.context import StageResult
        return StageResult(stage_id=stage.id, stage_type=stage.type, ok=True)

    custom_registry = dict(DEFAULT_STAGE_REGISTRY)
    for stage in spec.stages:
        custom_registry[stage.type] = fake_stage
    runner._registry = custom_registry

    from m3.pipeline.context import PipelineContext
    ctx = PipelineContext(
        video=Path("/tmp/fake.mp4"),
        video_id="v",
        seconds=10,
        work_dir=tmp_path,
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory",
        connection_string=None,
        account_id="m3bench",
        artifacts={"data_jsonl": tmp_path / "data.jsonl"},
        variables={},
    )
    runner.run(spec, ctx)
    assert "detect_faces" in executed, f"detect_faces not executed; ran: {executed}"
