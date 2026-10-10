"""Diarization cache and speaker writeback with real persistence, no inference service."""

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from core.graph_enums import RawDataType
from extraction.operators.base import ExtractionResult
from extraction.operators.builtin.speaker_diarization import SpeakerDiarizationExtractor
from m3.audio_cache import RECEIPT, _id
from m3.speaker_diarization import parse_args, run_speaker_diarization
from tests.integration.m3.test_voice_asr_resume import cache_graph  # noqa: F401


@pytest.fixture
def case(cache_graph, tmp_path, monkeypatch):  # noqa: F811
    graph, ctx = cache_graph
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    audio = graph.create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=ctx,
        metadata={"source_path": str(path), "video_id": "v", "clip_id": 1},
    )
    asr = graph.create_extracted_node(
        source_raw_id=audio.node_id,
        extractor_type="voice_asr",
        extraction_schema="voice_asr_v1",
        ctx=ctx,
        extracted_content={},
    )
    graph.mark_extraction_completed(
        asr.node_id,
        {
            "asr_id": uuid4().hex,
            "text": "hello",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "words": [],
            "source_audio_node_id": audio.node_id,
        },
        1.0,
        ctx,
    )
    args = parse_args(
        [
            "--backend",
            "mock",
            "--account-id",
            ctx.account_id,
            "--video",
            "v",
            "--output-jsonl",
            str(tmp_path / "turns.jsonl"),
        ]
    )
    calls = []
    turns = [{"speaker_id": "S0", "start_sec": 0.0, "end_sec": 1.0}]
    state = {"speaker": "S0", "fail": False, "omit": False}

    def extract(self, raw, context, *, asr_extracted_node):
        calls.append(raw.node_id)
        if state["fail"]:
            raise RuntimeError("test inference failure")
        segments = asr_extracted_node.extracted_content["segments"]
        current = deepcopy(turns)
        for turn in current:
            turn["asr_segment_ids"] = [s["asr_id"] for s in segments]
        return ExtractionResult(
            node_id=raw.node_id,
            schema_used="speaker_diarization_v1",
            confidence=1.0,
            extracted_content={
                "turns": current,
                "num_speakers": len({t["speaker_id"] for t in current}),
                "per_segment_speaker": {}
                if state["omit"]
                else {s["asr_id"]: state["speaker"] for s in segments},
            },
        )

    monkeypatch.setattr(SpeakerDiarizationExtractor, "extract", extract)

    def run():
        return run_speaker_diarization(args, graph=graph)

    def summary():
        return json.loads(Path(args.output_jsonl).with_suffix(".summary.json").read_text())

    def child():
        return graph.get_extracted_node(asr.node_id, ctx)

    def change_asr(**values):
        node = child()
        graph.mark_extraction_completed(
            node.node_id, {**node.extracted_content, **values}, 1.0, ctx
        )

    return locals()


def test_cache_manifest_and_annotations(case):
    c = case
    assert c["run"]() == 0
    before = Path(c["args"].output_jsonl).read_text()
    assert c["child"]().extracted_content["speaker_id"] == "S0"
    assert c["run"]() == 0
    assert Path(c["args"].output_jsonl).read_text() == before
    assert json.loads(before)["num_speakers"] == 1
    assert len(c["calls"]) == 1
    assert c["summary"]()["cached_audio"] == 1
    assert c["summary"]()["executed_audio"] == 0


@pytest.mark.parametrize(
    "change",
    ["text", "words", "time", "speaker", "force", "config", "bytes", "missing-turn", "legacy"],
)
def test_dependency_or_output_change_invalidates(case, change):
    c = case
    assert c["run"]() == 0
    if change == "text":
        c["change_asr"](text="updated")
    elif change == "words":
        c["change_asr"](words=[{"word": "updated", "start": 0, "end": 1}])
    elif change == "time":
        c["change_asr"](end_sec=2.0)
    elif change == "speaker":
        c["change_asr"](speaker_id="wrong")
    elif change == "force":
        c["args"].force = True
    elif change == "config":
        c["args"].max_speakers = 3
    elif change == "bytes":
        c["path"].write_bytes(b"changed")
    else:
        nodes = c["graph"].list_extracted_nodes("speaker_diarization", c["ctx"])
        if change == "missing-turn":
            n = next(n for n in nodes if "start_sec" in n.extracted_content)
            c["graph"].delete_node(n.node_id, c["ctx"])
        else:
            n = next(n for n in nodes if RECEIPT in n.extracted_content)
            ec = dict(n.extracted_content)
            ec.pop(RECEIPT)
            c["graph"].mark_extraction_completed(n.node_id, ec, 1.0, c["ctx"])
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
    assert c["child"]().extracted_content["speaker_id"] == "S0"


def test_failed_writeback_never_certified_and_retry_repairs(case, monkeypatch):
    c = case
    assert c["run"]() == 0
    c["args"].force = True
    c["state"]["speaker"] = "S1"
    original = c["graph"].mark_extraction_completed

    def fail(node_id, *args, **kwargs):
        if _id(node_id) == _id(c["asr"].node_id):
            raise RuntimeError("writeback failed")
        return original(node_id, *args, **kwargs)

    monkeypatch.setattr(c["graph"], "mark_extraction_completed", fail)
    assert c["run"]() == 2
    assert c["child"]().extracted_content["speaker_id"] == "S0"
    assert c["summary"]()["failed_audio"] == 1
    assert not any(
        RECEIPT in n.extracted_content
        for n in c["graph"].list_extracted_nodes("speaker_diarization", c["ctx"])
    )
    monkeypatch.setattr(c["graph"], "mark_extraction_completed", original)
    c["args"].force = False
    assert c["run"]() == 0
    assert c["child"]().extracted_content["speaker_id"] == "S1"
    assert len(c["calls"]) == 3
    assert c["run"]() == 0
    assert len(c["calls"]) == 3


def test_silence_clears_stale_speaker_and_is_cached(case):
    c = case
    assert c["run"]() == 0
    c["args"].force = True
    c["turns"].clear()
    c["state"]["speaker"] = None
    assert c["run"]() == 0
    assert c["child"]().extracted_content["speaker_id"] is None
    assert len(c["graph"].list_extracted_nodes("speaker_diarization", c["ctx"])) == 1
    c["args"].force = False
    assert c["run"]() == 0
    assert c["summary"]()["cached_audio"] == 1
    assert len(c["calls"]) == 2


def test_missing_alignment_fails(case):
    case["state"]["omit"] = True
    assert case["run"]() == 2
    assert case["summary"]()["completed_audio"] == 0


@pytest.mark.parametrize("incomplete", ["failed-child", "missing-child"])
def test_incomplete_upstream_rejected_before_inference(case, incomplete):
    c = case
    if incomplete == "failed-child":
        c["graph"].mark_extraction_failed(c["asr"].node_id, "incomplete", c["ctx"])
    else:
        parent = c["graph"].create_extracted_node(
            source_raw_id=c["audio"].node_id,
            extractor_type="voice_asr",
            extraction_schema="voice_asr_v1",
            ctx=c["ctx"],
            extracted_content={},
        )
        c["graph"].mark_extraction_completed(
            parent.node_id,
            {
                "child_count": 2,
                "child_node_ids": [c["asr"].node_id, uuid4().hex],
            },
            1.0,
            c["ctx"],
        )
    assert c["run"]() == 2
    assert not c["calls"]


def test_partial_failure_and_other_video_untouched(case):
    c = case
    other = c["graph"].create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=c["ctx"],
        metadata={"source_path": str(c["path"]), "video_id": "other"},
    )
    missing = c["graph"].create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=c["ctx"],
        metadata={"source_path": str(c["path"]), "video_id": "v"},
    )
    original = SpeakerDiarizationExtractor.extract
    from unittest.mock import patch

    def extract(self, raw, *args, **kwargs):
        if raw.node_id == missing.node_id:
            raise RuntimeError("one audio failed")
        return original(self, raw, *args, **kwargs)

    with patch.object(SpeakerDiarizationExtractor, "extract", extract):
        assert c["run"]() == 2
    assert c["summary"]()["completed_audio"] == 1
    assert c["summary"]()["failed_audio"] == 1
    assert c["summary"]()["executed_audio"] == 2
    assert other.node_id not in c["calls"]


def test_same_turn_id_refresh_updates_links_and_prunes_old_turns(case):
    c = case
    c["turns"].append({"speaker_id": "S1", "start_sec": 2.0, "end_sec": 3.0})
    assert c["run"]() == 0
    old = [json.loads(line) for line in Path(c["args"].output_jsonl).read_text().splitlines()]
    new_id = uuid4().hex
    c["change_asr"](asr_id=new_id)
    c["turns"].pop()
    assert c["run"]() == 0
    row = json.loads(Path(c["args"].output_jsonl).read_text())
    assert row["turn_node_id"] == old[0]["turn_node_id"]
    node = c["graph"].get_extracted_node(row["turn_node_id"], c["ctx"])
    assert node.extracted_content["asr_segment_ids"] == [new_id]
    assert node.extracted_content["num_speakers"] == 1
    assert c["graph"].get_extracted_node(old[1]["turn_node_id"], c["ctx"]) is None


def test_failed_forced_inference_cannot_reuse_old_receipt(case):
    c = case
    assert c["run"]() == 0
    c["args"].force = True
    c["state"]["fail"] = True
    assert c["run"]() == 2
    c["args"].force = False
    c["state"]["fail"] = False
    assert c["run"]() == 0
    assert len(c["calls"]) == 3


def test_valid_cache_usable_when_backend_preflight_unavailable(case, monkeypatch):
    c = case
    c["args"].backend = "pyannote"
    c["args"].hf_token = "test-token"
    monkeypatch.setattr("m3.speaker_diarization.importlib.util.find_spec", lambda name: object())
    assert c["run"]() == 0
    c["args"].hf_token = None
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert c["run"]() == 0
    assert c["summary"]()["cached_audio"] == 1
    c["change_asr"](text="changed dependency")
    assert c["run"]() == 2
    assert len(c["calls"]) == 1
    assert c["summary"]()["executed_audio"] == 0


def test_asr_added_after_unaligned_diarization_invalidates_cache(case):
    c = case
    content = dict(c["child"]().extracted_content)
    c["graph"].delete_node(c["asr"].node_id, c["ctx"])
    assert c["run"]() == 0
    assert c["run"]() == 0
    assert len(c["calls"]) == 1
    node = c["graph"].create_extracted_node(
        source_raw_id=c["audio"].node_id,
        extractor_type="voice_asr",
        extraction_schema="voice_asr_v1",
        ctx=c["ctx"],
        extracted_content={},
    )
    c["graph"].mark_extraction_completed(node.node_id, content, 1.0, c["ctx"])
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
    assert (
        c["graph"].get_extracted_node(node.node_id, c["ctx"]).extracted_content["speaker_id"]
        == "S0"
    )
