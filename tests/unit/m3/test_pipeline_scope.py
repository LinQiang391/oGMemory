"""Offline regressions for single-video pipeline scope and legacy driver CLIs."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from m3.config import M3BENCH_ACCOUNT_ID
from m3.pipeline.context import PipelineContext
from m3.pipeline.spec import StageSpec
from m3.pipeline import stages


DRIVERS = [
    ("voice_segmentation", "VoiceSegmentationExtractor"),
    ("voice_asr", "VoiceASRExtractor"),
    ("speaker_diarization", "SpeakerDiarizationExtractor"),
    ("speaker_embedding", "SpeakerEmbeddingExtractor"),
]
NAMES = [name for name, _ in DRIVERS] + ["semantic_memory_consolidator"]


def _context(tmp_path):
    return PipelineContext(
        video=tmp_path / "target.mp4", video_id="target", seconds=0,
        work_dir=tmp_path, config_path=tmp_path / "config.yaml",
        graph_store="memory", connection_string=None, account_id="custom-account",
        artifacts={}, variables={}, graph=MagicMock(),
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("explicit_video", [None, "target"])
def test_stage_forwards_scope(tmp_path, monkeypatch, name, explicit_video):
    module = importlib.import_module(f"m3.{name}")
    run = MagicMock(return_value=0)
    monkeypatch.setattr(module, f"run_{name}", run)
    params = {"force": True}
    if name != "voice_segmentation":
        params["backend"] = "mock"
    if explicit_video is not None:
        params["video"] = explicit_video
    ctx = _context(tmp_path)
    getattr(stages, f"{name}_stage")(ctx, StageSpec(
        id=name, type=f"m3.{name}", params=params,
    ))
    args = run.call_args.args[0]
    assert args.video == "target"
    assert args.account_id == "custom-account"
    assert run.call_args.kwargs["graph"] is ctx.graph
    if name != "semantic_memory_consolidator":
        assert args.force is True


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("video", ["other", "target,other", ""])
def test_stage_rejects_conflicting_scope(tmp_path, monkeypatch, name, video):
    module = importlib.import_module(f"m3.{name}")
    run = MagicMock()
    monkeypatch.setattr(module, f"run_{name}", run)
    with pytest.raises(ValueError, match="conflicts"):
        getattr(stages, f"{name}_stage")(_context(tmp_path), StageSpec(
            id=name, type=f"m3.{name}", params={"video": video},
        ))
    run.assert_not_called()


@pytest.mark.parametrize("name", NAMES)
def test_cli_scope_is_backwards_compatible(name):
    module = importlib.import_module(f"m3.{name}")
    args = module.parse_args([])
    assert args.account_id == M3BENCH_ACCOUNT_ID
    assert args.video is None
    args = module.parse_args(["--account-id", "custom", "--video", "one,two"])
    assert args.account_id == "custom"
    assert args.video == "one,two"


@pytest.mark.parametrize("name,extractor_name", DRIVERS)
@pytest.mark.parametrize("video,expected", [
    ("target", ["target"]), ("target,other", ["target", "other"]),
    (None, ["target", "other"]),
])
def test_force_only_processes_selected_scope(
    tmp_path, monkeypatch, name, extractor_name, video, expected,
):
    """Exercise the real drivers, stubbing only storage and inference."""
    module = importlib.import_module(f"m3.{name}")
    nodes = [
        SimpleNamespace(node_id=v, metadata={"video_id": v, "clip_id": 0})
        for v in ("target", "other")
    ]
    graph = MagicMock()

    def list_raw(raw_type, ctx):
        assert ctx.account_id == "custom-account"
        return nodes

    graph.list_raw_nodes.side_effect = list_raw
    graph.list_extracted_nodes.return_value = []
    if name == "speaker_embedding":
        # This scope test uses certified silence; nonempty inference scope is
        # covered with real persistence in test_speaker_embedding_cache.py.
        path = tmp_path / "audio.wav"
        path.write_bytes(b"audio")
        for node in nodes:
            node.metadata["source_path"] = str(path)
        monkeypatch.setattr(module, "_diarization_turns", lambda *a: ([], {}))
    extractor = MagicMock()
    extractor.extract.return_value = SimpleNamespace(extracted_content={}, error=None)
    monkeypatch.setattr(
        f"extraction.operators.builtin.{name}.{extractor_name}",
        MagicMock(return_value=extractor),
    )
    service = MagicMock()
    service.persist_extraction.return_value = {
        "materialized_children": [], "extracted_node_id": "parent",
    }
    graph.get_extracted_node.return_value = SimpleNamespace(
        extracted_content={"child_node_ids": [], "child_count": 0},
        confidence=1.0,
    )
    monkeypatch.setattr(
        "service.operator_service.OperatorService", MagicMock(return_value=service),
    )
    argv = ["--account-id", "custom-account", "--force",
            "--output-jsonl", str(tmp_path / "out.jsonl")]
    if name != "voice_segmentation":
        argv += ["--backend", "mock"]
    if video:
        argv += ["--video", video]
    assert getattr(module, f"run_{name}")(module.parse_args(argv), graph=graph) == 0
    assert [c.args[0].node_id for c in extractor.extract.call_args_list] == (
        [] if name == "speaker_embedding" else expected
    )
    assert [c.kwargs["raw_node"].node_id for c in service.persist_extraction.call_args_list] == expected
    for call in extractor.extract.call_args_list:
        assert call.args[1].account_id == "custom-account"
    assert service.persist_extraction.call_count == len(expected)
    for call in service.persist_extraction.call_args_list:
        assert call.kwargs["ctx"].account_id == "custom-account"
