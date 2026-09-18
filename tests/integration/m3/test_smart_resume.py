"""Tests for the smart-resume probe protocol (Layer 2 of the resume strategy).

Each test mocks the runner's graph + artefacts and verifies the probe
returns the expected :class:`ProbeResult`. Also verifies that
:meth:`PipelineRunner.run` short-circuits a stage when its probe says
complete and runs it when the probe says continue.

Layer 1 (per-source-node skip inside each driver) is tested in the
driver-level test files (test_voice_asr.py, test_detect_faces.py, …).

The probes enumerate the *current video's* parent raw nodes and check
that each has a completed ExtractedNode (by ``source_raw_id``). This
makes them safe in multi-video shared-account runs — the pre-fix
behaviour counted nodes account-wide, so the second video's probes
saw the first video's completed nodes and skipped extraction.
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
    node.node_id = f"video-{video_id}-clip-{clip_id}"
    node.metadata = {"video_id": video_id, "clip_id": clip_id}
    return node


def _fake_image_node(
    *, node_id: str, source_video_node_id: str,
) -> MagicMock:
    node = MagicMock()
    node.node_id = node_id
    node.metadata = {"source_video_node_id": source_video_node_id}
    return node


def _fake_audio_node(*, video_id: str, clip_id: int) -> MagicMock:
    node = MagicMock()
    node.node_id = f"audio-{video_id}-clip-{clip_id}"
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


def _stage_spec(stage_id: str, type_: str) -> StageSpec:
    return StageSpec(
        id=stage_id, type=type_, enabled=True,
        needs=[], params={}, outputs={},
    )


def _ctx_with_artifacts(
    tmp_path: Path,
    *,
    artefacts: dict[str, str],
    graph=None,
    video_id: str = "v1",
) -> PipelineContext:
    """Build a PipelineContext with the requested artefact paths."""
    paths = {name: tmp_path / value for name, value in artefacts.items()}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        video=tmp_path / "video.mp4",
        video_id=video_id,
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


def _wire_graph_raw_nodes(
    graph: MagicMock,
    *,
    video_nodes: list = None,
    image_nodes: list = None,
    audio_nodes: list = None,
) -> MagicMock:
    """Configure ``graph.list_raw_nodes`` with one page per raw type.

    The probe walks VIDEO first, then IMAGE, then AUDIO; each call
    returns its respective page and the pagination loop terminates
    after a single page (``len(page) < 1000``).
    """
    graph.list_raw_nodes.side_effect = [
        video_nodes or [],
        image_nodes or [],
        audio_nodes or [],
    ]
    return graph


def _set_extracted_pages(
    graph: MagicMock, *pages: list,
) -> MagicMock:
    """Configure ``graph.list_extracted_nodes`` page-by-page."""
    graph.list_extracted_nodes.side_effect = list(pages) + [[]]
    return graph


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
        result = _probe_video_trim(ctx, _stage_spec("trim", "video.trim"), None)
        assert result.complete is True
        assert "selected_video" in result.reason or "artefact" in result.reason

    def test_video_trim_continues_when_file_missing(self, tmp_path):
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"selected_video": "v.mp4"},
        )
        result = _probe_video_trim(ctx, _stage_spec("trim", "video.trim"), None)
        assert result.complete is False

    def test_write_single_annotation_complete_when_json_nonempty(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text(json.dumps({"v": {"video_path": "x", "qa_list": []}}))
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"annotations": "ann.json"},
        )
        result = _probe_write_single_annotation(
            ctx, _stage_spec("annotate", "m3.write_single_annotation"), None,
        )
        assert result.complete is True

    def test_load_clips_continues_when_data_jsonl_missing(self, tmp_path):
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={"data_jsonl": "missing.jsonl"},
        )
        result = _probe_load_clips(
            ctx, _stage_spec("load_clips", "m3.load_clips"), None,
        )
        assert result.complete is False


# ---------------------------------------------------------------------------
# Graph-walking probes (video-scoped)
# ---------------------------------------------------------------------------


class TestExtractedNodeProbes:
    def test_extract_frames_complete_when_all_completed(self, tmp_path):
        video_node = _fake_video_node(video_id="v1", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[video_node])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id=video_node.node_id,
                status="completed",
            ),
        ])
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is True
        assert result.count == 1
        assert "for video 'v1'" in result.reason

    def test_extract_frames_continues_when_some_pending(self, tmp_path):
        # Two parent VIDEO raw nodes for v1; one has a completed
        # extracted node, the other is unfinished.
        v1 = _fake_video_node(video_id="v1", clip_id=1)
        v2 = _fake_video_node(video_id="v1", clip_id=2)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[v1, v2])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id=v1.node_id,
                status="completed",
            ),
            _fake_extracted_node(
                node_id="ext-2", source_raw_id=v2.node_id,
                status="pending",
            ),
        ])
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is False
        assert "incomplete" in result.reason
        assert "for video 'v1'" in result.reason

    def test_voice_asr_continues_when_no_parents(self, tmp_path):
        # No AUDIO raw nodes for v1 — pre-extraction state.
        graph = MagicMock()
        _wire_graph_raw_nodes(graph)  # all raw pages empty
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_voice_asr(
            ctx, _stage_spec("voice_asr", "m3.voice_asr"), graph,
        )
        assert result.complete is False
        assert "no audio" in result.reason
        assert "video 'v1'" in result.reason

    def test_voice_asr_continues_when_no_extracted_nodes(self, tmp_path):
        # Has AUDIO parents but no voice_asr ExtractedNodes yet.
        audio = _fake_audio_node(video_id="v1", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, audio_nodes=[audio])
        _set_extracted_pages(graph, [])
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_voice_asr(
            ctx, _stage_spec("voice_asr", "m3.voice_asr"), graph,
        )
        assert result.complete is False
        assert "missing" in result.reason

    def test_voice_asr_swallows_graph_errors(self, tmp_path):
        graph = MagicMock()
        graph.list_raw_nodes.side_effect = RuntimeError("db down")
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_voice_asr(
            ctx, _stage_spec("voice_asr", "m3.voice_asr"), graph,
        )
        assert result.complete is False
        assert "probe error" in result.reason

    def test_episodic_memory_requires_evidence_validation(self, tmp_path):
        # Completed nodes may still have missing or outdated speech.
        v1 = _fake_video_node(video_id="v1", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[v1])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id=v1.node_id,
                status="completed", embedding=None,
            ),
        ])
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_episodic_memory(
            ctx, _stage_spec("episodic_memory", "m3.episodic_memory"), graph,
        )
        assert result.complete is False

    # ----- Multi-video scenarios (FIX-2 regression coverage) -----

    def test_extract_frames_continues_when_other_video_complete(self, tmp_path):
        """Video A's extracted nodes exist; B has only raw nodes —
        probe for B must NOT skip extraction (pre-fix bug returned skip
        counting A's nodes account-wide)."""
        raw_a = _fake_video_node(video_id="A", clip_id=1)
        raw_b = _fake_video_node(video_id="B", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[raw_a, raw_b])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-a-1", source_raw_id=raw_a.node_id,
                status="completed",
            ),
        ])
        # ctx for video B; raw_b has NO extracted node.
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={}, graph=graph, video_id="B",
        )
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is False, (
            "video B should not be marked complete just because video A's "
            "extraction finished (pre-fix bug returned skip here)"
        )
        assert "missing" in result.reason
        assert "for video 'B'" in result.reason

    def test_extract_frames_skips_when_current_video_all_completed(
        self, tmp_path,
    ):
        """B's VIDEO raw nodes each have a completed
        ``video_frame_extractor`` extracted node → skip."""
        b1 = _fake_video_node(video_id="B", clip_id=1)
        b2 = _fake_video_node(video_id="B", clip_id=2)
        a1 = _fake_video_node(video_id="A", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[a1, b1, b2])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-a-1", source_raw_id=a1.node_id,
                status="completed",
            ),
            _fake_extracted_node(
                node_id="ext-b-1", source_raw_id=b1.node_id,
                status="completed",
            ),
            _fake_extracted_node(
                node_id="ext-b-2", source_raw_id=b2.node_id,
                status="completed",
            ),
        ])
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={}, graph=graph, video_id="B",
        )
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is True
        assert result.count == 2
        assert "for video 'B'" in result.reason

    def test_probe_reports_missing_and_incomplete_split(self, tmp_path):
        """B has one parent with an unfinished child, one parent with
        no child at all — probe continues and reports both counts in
        the reason."""
        b1 = _fake_video_node(video_id="B", clip_id=1)
        b2 = _fake_video_node(video_id="B", clip_id=2)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[b1, b2])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-b-1", source_raw_id=b1.node_id,
                status="in_progress",  # unfinished → counts as incomplete
            ),
        ])
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={}, graph=graph, video_id="B",
        )
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is False
        assert "1 missing" in result.reason
        assert "1 incomplete" in result.reason
        assert "of 2 video_frame_extractor nodes for video 'B'" in result.reason

    def test_probe_continues_when_no_raw_nodes_for_video(self, tmp_path):
        """B's account has VIDEO raw nodes for A only — probe for B
        must fail open with a 'no raw nodes yet' message."""
        a1 = _fake_video_node(video_id="A", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[a1])
        _set_extracted_pages(graph, [])
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={}, graph=graph, video_id="B",
        )
        result = _probe_extract_frames(
            ctx, _stage_spec("extract_frames", "m3.extract_frames"), graph,
        )
        assert result.complete is False
        assert "no video" in result.reason.lower()
        assert "video 'B'" in result.reason

    def test_probe_continues_when_embedding_missing(self, tmp_path):
        """``require_embeddings=True`` flag must catch a completed
        extracted node whose ``metadata.embedding`` is unset."""
        from m3.pipeline.stages import _probe_from_extracted_nodes

        v1 = _fake_video_node(video_id="v1", clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[v1])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id=v1.node_id,
                status="completed", embedding=None,
            ),
        ])
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_from_extracted_nodes(
            ctx, "episodic_memory_generator", require_embeddings=True,
        )
        assert result.complete is False
        assert "1 incomplete" in result.reason
        assert "video 'v1'" in result.reason

    def test_probe_continues_when_extractor_unknown(self, tmp_path):
        """An extractor type not in the parent-raw-type map must fail
        open (probe_continue), never wrongly skip a stage."""
        from m3.pipeline.stages import _probe_from_extracted_nodes

        graph = MagicMock()
        # list_raw_nodes / list_extracted_nodes shouldn't be called at
        # all when the extractor is unknown; make sure that's true.
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        result = _probe_from_extracted_nodes(ctx, "totally_new_extractor")
        assert result.complete is False
        assert "no parent-raw-type mapping" in result.reason
        graph.list_raw_nodes.assert_not_called()
        graph.list_extracted_nodes.assert_not_called()

    def test_face_detection_keys_off_image_not_video(self, tmp_path):
        """``face_detection`` ExtractedNodes' ``source_raw_id`` is an
        IMAGE raw node id (frame child), NOT a VIDEO raw node id. A
        completed face node whose ``source_raw_id`` is an A-video
        frame id must NOT count toward B's completeness."""
        from m3.pipeline.stages import _probe_detect_faces

        # A's VIDEO raw node.
        raw_a_video = _fake_video_node(video_id="A", clip_id=1)
        # A's IMAGE frame child (referencing A's VIDEO raw node).
        raw_a_frame = _fake_image_node(
            node_id="frame-A-1", source_video_node_id=raw_a_video.node_id,
        )
        # B's VIDEO raw node.
        raw_b_video = _fake_video_node(video_id="B", clip_id=1)
        # B's IMAGE frame child.
        raw_b_frame = _fake_image_node(
            node_id="frame-B-1", source_video_node_id=raw_b_video.node_id,
        )
        graph = MagicMock()
        _wire_graph_raw_nodes(
            graph,
            video_nodes=[raw_a_video, raw_b_video],
            image_nodes=[raw_a_frame, raw_b_frame],
        )
        # A frame has a face_detection extracted node (source_raw_id =
        # A frame id); B's frame doesn't.
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="face-A-1", source_raw_id=raw_a_frame.node_id,
                status="completed",
            ),
        ])
        ctx = _ctx_with_artifacts(
            tmp_path, artefacts={}, graph=graph, video_id="B",
        )
        result = _probe_detect_faces(
            ctx, _stage_spec("detect_faces", "m3.detect_faces"), graph,
        )
        assert result.complete is False, (
            "A's face_detection node (keyed on A's IMAGE raw node id) "
            "must not count toward B's completeness"
        )


# ---------------------------------------------------------------------------
# Runner-level integration: probe short-circuits a stage
# ---------------------------------------------------------------------------


class TestRunnerProbeShortCircuit:
    def _make_spec(self, stage: StageSpec):
        from m3.pipeline.spec import PipelineSpec
        return PipelineSpec(
            name="test", version=1, description="",
            artifacts={}, stages=[stage],
        )

    def _completed_video_graph(self, video_id: str = "v1") -> MagicMock:
        """Return a graph mock where the named video's episodic
        extraction is fully complete."""
        raw = _fake_video_node(video_id=video_id, clip_id=1)
        graph = MagicMock()
        _wire_graph_raw_nodes(graph, video_nodes=[raw])
        _set_extracted_pages(graph, [
            _fake_extracted_node(
                node_id="ext-1", source_raw_id=raw.node_id,
                status="completed",
            ),
        ])
        return graph

    def test_runner_skips_stage_when_probe_says_complete(self, tmp_path):
        from m3.pipeline.stages import probe_skip
        graph = self._completed_video_graph()
        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)

        stage_fn = MagicMock()
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": lambda *args: probe_skip(reason="test")},
        )

        spec = self._make_spec(_stage_spec("episodic_memory", "m3.episodic_memory"))

        runner.run(spec, ctx)
        stage_fn.assert_not_called()
        result = ctx.stage_results["episodic_memory"]
        assert result.ok is True
        assert result.counts.get("skipped") == 1

    def test_runner_invokes_stage_when_probe_says_continue(self, tmp_path):
        graph = MagicMock()
        _wire_graph_raw_nodes(graph)  # no raw nodes → continue
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

        spec = self._make_spec(_stage_spec("episodic_memory", "m3.episodic_memory"))

        runner.run(spec, ctx)
        stage_fn.assert_called_once()
        result = ctx.stage_results["episodic_memory"]
        assert result.counts.get("records") == 1

    def test_runner_smart_resume_disabled_always_invokes_stage(self, tmp_path):
        graph = self._completed_video_graph()
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

        spec = self._make_spec(_stage_spec("episodic_memory", "m3.episodic_memory"))

        runner.run(spec, ctx)
        stage_fn.assert_called_once()

    def test_runner_swallows_probe_errors(self, tmp_path):
        graph = MagicMock()
        graph.list_raw_nodes.side_effect = RuntimeError("intentional probe failure")

        stage_fn = MagicMock(return_value=StageResult(
            stage_id="episodic_memory",
            stage_type="m3.episodic_memory",
            ok=True,
        ))
        runner = PipelineRunner(
            registry={"m3.episodic_memory": stage_fn},
            probes={"m3.episodic_memory": _probe_episodic_memory},
        )

        ctx = _ctx_with_artifacts(tmp_path, artefacts={}, graph=graph)
        spec = self._make_spec(_stage_spec("episodic_memory", "m3.episodic_memory"))

        runner.run(spec, ctx)
        stage_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Default probe registry sanity check
# ---------------------------------------------------------------------------


class TestDefaultProbeRegistry:
    def test_default_probes_have_entry_for_each_registered_stage(self):
        from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
        probed = set(DEFAULT_STAGE_PROBES)
        registered = set(DEFAULT_STAGE_REGISTRY)
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
        for stage_type, probe in DEFAULT_STAGE_PROBES.items():
            assert callable(probe), f"{stage_type}: probe is not callable"


@pytest.mark.parametrize('present', [0, 1, 2])
def test_load_clips_requires_all_manifest_nodes_in_current_graph(tmp_path, present):
    ctx = _ctx_with_artifacts(tmp_path, artefacts={'data_jsonl': 'data.jsonl'})
    ctx.artifact('data_jsonl').write_text(json.dumps({
        'raw_nodes': [{'raw_node_id': 'clip-0'}, {'raw_node_id': 'clip-1'}],
    }) + '\n')
    graph = MagicMock()
    graph.get_raw_node.side_effect = lambda node_id, request: (
        object() if node_id in ['clip-0', 'clip-1'][:present] else None
    )
    result = _probe_load_clips(ctx, _stage_spec('load_clips', 'm3.load_clips'), graph)
    assert result.complete is (present == 2)
