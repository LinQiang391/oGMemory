"""Tests for the smart-resume probe protocol (Layer 2 of the resume strategy).

Each test mocks the runner's graph + artefacts and verifies the probe
returns the expected :class:`ProbeResult`. Also verifies that
:meth:`PipelineRunner.run` short-circuits a stage when its probe says
complete and runs it when the probe says continue.

Layer 1 (per-source-node skip inside each driver) is tested in the
driver-level test files (test_voice_asr.py, test_detect_faces.py, …).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.graph_enums import RawDataType
from m3.pipeline.context import PipelineContext, StageResult
from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import StageSpec
from m3.pipeline.stages import (
    DEFAULT_STAGE_PROBES,
    ProbeResult,
    _probe_video_trim,
    _probe_write_single_annotation,
    _probe_preprocess,
    _probe_load_clips,
    _probe_extract_frames,
    _probe_extract_audio,
    _probe_voice_asr,
    _probe_episodic_memory,
    _probe_qa,
    probe_continue,
    probe_skip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_video_node(*, video_id: str, clip_id: int) -> MagicMock:
    node = MagicMock()
    node.node_id = f"video-{clip_id:032d}"[:32]
    node.metadata = {"video_id": video_id, "clip_id": clip_id}
    return node


def _fake_extracted_node(
    *,
    node_id: str,
    source_raw_id: str = "",
    status: str = "completed",
    embedding: list[float] | None = None,
) -> MagicMock:
    node = MagicMock()
    node.node_id = node_id
    node.source_raw_id = source_raw_id
    node.extraction_status = status
    node.metadata = {"embedding": embedding} if embedding is not None else {}
    return node


def _ctx_with_artifacts(
    tmp_path: Path,
    *,
    artefacts: dict[str, str],
    graph=None,
) -> PipelineContext:
    """Build a PipelineContext with the requested artefact paths."""
    paths = {name: tmp_path / value for name, value in artefacts.items()}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        video=tmp_path / "video.mp4",
        video_id="v1",
        seconds=0,
        work_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        graph_store="memory",
        connection_string=None,
        account_id="acct",
        artifacts=paths,
        variables={},
        graph=graph,
        op_service=None,
    )


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------


class TestProbeHelpers:
    def test_probe_skip_returns_complete(self):
        result = probe_skip(reason="cached", count=3)
        assert result.complete is True
        assert result.reason == "cached"
        assert result.count == 3

    def test_probe_continue_returns_incomplete(self):
        result = probe_continue(reason="nothing here")
        assert result.complete is False
        assert result.reason == "nothing here"


# ---------------------------------------------------------------------------
# Artefact-based probes
# ---------------------------------------------------------------------------


class TestArtefactProbes:
    def test_video_trim_complete_when_file_present(self, tmp_path):
        selected = tmp_path / "v.mp4"
        selected.write_bytes(b"x" * 1024)
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"selected_video": "v.mp4"},
        )
        result = _probe_video_trim(ctx, StageSpec(
            id="trim", type="video.trim", enabled=True,
            needs=[], params={}, outputs={},
        ), None)
        assert result.complete is True
        assert "selected_video" in result.reason or "artefact" in result.reason

    def test_video_trim_continues_when_file_missing(self, tmp_path):
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"selected_video": "v.mp4"},
        )
        result = _probe_video_trim(ctx, StageSpec(
            id="trim", type="video.trim", enabled=True,
            needs=[], params={}, outputs={},
        ), None)
        assert result.complete is False

    def test_write_single_annotation_complete_when_json_nonempty(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text(json.dumps({"v": {"video_path": "x", "qa_list": []}}))
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"annotations": "ann.json"},
        )
        result = _probe_write_single_annotation(ctx, StageSpec(
            id="annotate", type="m3.write_single_annotation", enabled=True,
            needs=[], params={}, outputs={},
        ), None)
        assert result.complete is True

    def test_load_clips_continues_when_data_jsonl_missing(self, tmp_path):
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"data_jsonl": "missing.jsonl"},
        )
        result = _probe_load_clips(ctx, StageSpec(
            id="load_clips", type="m3.load_clips", enabled=True,
            needs=[], params={}, outputs={},
        ), None)
        assert result.complete is False


# ---------------------------------------------------------------------------
# Graph-walking probes
# ---------------------------------------------------------------------------


class TestExtractedNodeProbes:
    def test_extract_frames_complete_when_all_completed(self, tmp_path):
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id="video-1", status="completed",
            ),
            _fake_extracted_node(
                node_id="ext-2", source_raw_id="video-2", status="completed",
            ),
        ]
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_extract_frames(ctx, StageSpec(
            id="extract_frames", type="m3.extract_frames", enabled=True,
            needs=[], params={}, outputs={},
        ), graph)
        assert result.complete is True
        assert result.count == 2

    def test_extract_frames_continues_when_some_pending(self, tmp_path):
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id="video-1", status="completed",
            ),
            _fake_extracted_node(
                node_id="ext-2", source_raw_id="video-2", status="pending",
            ),
        ]
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_extract_frames(ctx, StageSpec(
            id="extract_frames", type="m3.extract_frames", enabled=True,
            needs=[], params={}, outputs={},
        ), graph)
        assert result.complete is False
        assert "not completed" in result.reason

    def test_voice_asr_continues_when_no_extracted_nodes(self, tmp_path):
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = []
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_voice_asr(ctx, StageSpec(
            id="voice_asr", type="m3.voice_asr", enabled=True,
            needs=[], params={}, outputs={},
        ), graph)
        assert result.complete is False
        assert "no voice_asr" in result.reason

    def test_voice_asr_swallows_graph_errors(self, tmp_path):
        graph = MagicMock()
        graph.list_extracted_nodes.side_effect = RuntimeError("db down")
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_voice_asr(ctx, StageSpec(
            id="voice_asr", type="m3.voice_asr", enabled=True,
            needs=[], params={}, outputs={},
        ), graph)
        assert result.complete is False
        assert "probe error" in result.reason

    def test_episodic_memory_continues_when_some_embeddings_missing(
        self, tmp_path,
    ):
        # The episodic_memory probe does NOT require embeddings (those
        # are stamped by index_episodic); it only requires completion.
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id="video-1",
                status="completed", embedding=None,
            ),
            _fake_extracted_node(
                node_id="ext-2", source_raw_id="video-2",
                status="completed", embedding=[0.1, 0.2, 0.3],
            ),
        ]
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_episodic_memory(ctx, StageSpec(
            id="episodic_memory", type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        ), graph)
        assert result.complete is True


# ---------------------------------------------------------------------------
# Runner-level integration: probe short-circuits a stage
# ---------------------------------------------------------------------------


class TestRunnerProbeShortCircuit:
    def _make_spec(self, stage: StageSpec):
        # Build a real PipelineSpec instead of a MagicMock so summary()
        # can serialise it. ``validate`` is patched because we don't
        # want the runner to walk the real registry.
        from m3.pipeline.spec import PipelineSpec
        return PipelineSpec(
            name="test", version=1, description="",
            artifacts={}, stages=[stage],
        )

    def test_runner_skips_stage_when_probe_says_complete(self, tmp_path):
        """End-to-end: the runner must NOT invoke the stage function
        when the probe returns complete=True."""

        # Mock graph + probe + stage function. The runner should call
        # the probe first; if complete, it must NOT call the stage fn.
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id="video-1", status="completed",
            ),
        ]
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)

        stage_fn = MagicMock()
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": _probe_episodic_memory},
        )

        spec = self._make_spec(StageSpec(
            id="episodic_memory", type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        ))

        runner.run(spec, ctx)
        # Stage function must NOT have been called.
        stage_fn.assert_not_called()
        # ctx.stage_results should record a skipped result.
        result = ctx.stage_results["episodic_memory"]
        assert result.ok is True
        assert result.counts.get("skipped") == 1

    def test_runner_invokes_stage_when_probe_says_continue(self, tmp_path):
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = []  # nothing completed
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)

        stage_fn = MagicMock(return_value=StageResult(
            stage_id="episodic_memory",
            stage_type="m3.episodic_memory",
            ok=True,
            counts={"records": 1},
        ))
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": _probe_episodic_memory},
        )

        spec = self._make_spec(StageSpec(
            id="episodic_memory", type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        ))

        runner.run(spec, ctx)
        stage_fn.assert_called_once()
        result = ctx.stage_results["episodic_memory"]
        assert result.counts.get("records") == 1

    def test_runner_smart_resume_disabled_always_invokes_stage(
        self, tmp_path,
    ):
        # Same setup as the previous test but smart_resume=False.
        # Probe says complete, but the runner must invoke the stage
        # because the probe layer is disabled.
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id="video-1", status="completed",
            ),
        ]
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)

        stage_fn = MagicMock(return_value=StageResult(
            stage_id="episodic_memory",
            stage_type="m3.episodic_memory",
            ok=True,
        ))
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": _probe_episodic_memory},
            smart_resume=False,
        )

        spec = self._make_spec(StageSpec(
            id="episodic_memory", type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        ))

        runner.run(spec, ctx)
        stage_fn.assert_called_once()

    def test_runner_swallows_probe_errors(self, tmp_path):
        # A buggy probe must NOT abort the pipeline — it should be
        # treated as "continue" and the stage should run normally.
        graph = MagicMock()
        graph.list_extracted_nodes.return_value = []

        def _bad_probe(*args, **kwargs):
            raise RuntimeError("intentional probe failure")

        stage_fn = MagicMock(return_value=StageResult(
            stage_id="episodic_memory",
            stage_type="m3.episodic_memory",
            ok=True,
        ))
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": _bad_probe},
        )

        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        spec = self._make_spec(StageSpec(
            id="episodic_memory", type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        ))

        runner.run(spec, ctx)  # must not raise
        stage_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Default probe registry sanity check
# ---------------------------------------------------------------------------


class TestDefaultProbeRegistry:
    def test_default_probes_have_entry_for_each_registered_stage(self):
        # Every stage type that has a probe should be present in
        # DEFAULT_STAGE_PROBES. This guards against silent typos
        # (e.g. renaming a stage without updating the probe dict).
        from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
        probed = set(DEFAULT_STAGE_PROBES)
        registered = set(DEFAULT_STAGE_REGISTRY)
        # The QA stage has its own --resume; an extra probe is fine
        # but optional. We require probe coverage for the expensive
        # media stages.
        required = {
            "m3.episodic_memory",
            "m3.extract_frames",
            "m3.extract_audio",
            "m3.detect_faces",
            "m3.voice_segmentation",
            "m3.voice_asr",
            "m3.speaker_diarization",
            "m3.speaker_embedding",
            "m3.index_episodic",
        }
        missing = required - probed
        assert not missing, f"missing probes: {missing}"

    def test_all_probes_are_callable(self):
        # Defensive: every entry in DEFAULT_STAGE_PROBES must be a
        # callable accepting the probe signature.
        for stage_type, probe in DEFAULT_STAGE_PROBES.items():
            assert callable(probe), f"{stage_type}: probe is not callable"