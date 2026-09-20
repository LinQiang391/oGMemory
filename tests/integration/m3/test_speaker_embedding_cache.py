"""Dependency-aware speaker embedding persistence; stub only inference."""

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from core.graph_enums import RawDataType
from extraction.operators.base import ExtractionResult
from extraction.operators.builtin.speaker_diarization import SpeakerDiarizationExtractor
from extraction.operators.builtin.speaker_embedding import SpeakerEmbeddingExtractor
from m3.audio_cache import RECEIPT, _id
from m3.speaker_diarization import parse_args as diar_args
from m3.speaker_diarization import run_speaker_diarization
from m3.speaker_embedding import parse_args, run_speaker_embedding
from tests.integration.m3.test_voice_asr_resume import cache_graph  # noqa: F401


@pytest.fixture
def case(cache_graph, tmp_path, monkeypatch):  # noqa: F811
    graph, ctx = cache_graph
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake-audio")
    audio = graph.create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=ctx,
        metadata={"source_path": str(path), "video_id": "v", "clip_id": 0},
    )
    turns = [
        {"speaker_id": "S0", "start_sec": 0.0, "end_sec": 1.0},
        {"speaker_id": "S1", "start_sec": 1.0, "end_sec": 2.0},
    ]
    state = {
        "error": None,
        "partial": False,
        "vector": None,
        "omit": False,
        "duplicate": False,
        "raise": False,
        "value": 1.0,
    }
    calls = []

    def diar_extract(self, raw, context, **kwargs):
        return ExtractionResult(
            node_id=raw.node_id,
            schema_used=self.schema_name,
            confidence=1.0,
            extracted_content={
                "turns": deepcopy(turns),
                "num_speakers": len({t["speaker_id"] for t in turns}),
            },
        )

    monkeypatch.setattr(SpeakerDiarizationExtractor, "extract", diar_extract)
    dargs = diar_args(
        [
            "--account-id",
            ctx.account_id,
            "--backend",
            "mock",
            "--force",
            "--video",
            "v",
            "--output-jsonl",
            str(tmp_path / "diar.jsonl"),
        ]
    )

    def refresh_diar():
        assert run_speaker_diarization(dargs, graph=graph) == 0

    refresh_diar()
    args = parse_args(
        [
            "--account-id",
            ctx.account_id,
            "--backend",
            "mock",
            "--embedding-dim",
            "3",
            "--video",
            "v",
            "--output-jsonl",
            str(tmp_path / "emb.jsonl"),
        ]
    )

    def extract(self, raw, context, *, diarization_extracted_node):
        calls.append(raw.node_id)
        if state["raise"]:
            raise RuntimeError("inference failure")
        grouped = {}
        for turn in diarization_extracted_node.extracted_content["turns"]:
            grouped.setdefault(turn["speaker_id"], []).append(turn)
        speakers = [
            {
                "speaker_id": sid,
                "speaker_embedding": (
                    [state["value"]] * args.embedding_dim
                    if state["vector"] is None
                    else state["vector"]
                ),
                "embedding_dim": args.embedding_dim,
                "model": "mock",
                "num_turns": len(ts),
                "total_duration_sec": sum(t["end_sec"] - t["start_sec"] for t in ts),
            }
            for sid, ts in grouped.items()
        ]
        if state["omit"]:
            speakers.pop()
        if state["duplicate"]:
            speakers.append(deepcopy(speakers[0]))
        content = {"speakers": speakers, "model": "mock", "embedding_dim": args.embedding_dim}
        if state["partial"]:
            content["per_speaker_error"] = {"S1": "backend error"}
        return ExtractionResult(
            node_id=raw.node_id,
            extracted_content=content,
            confidence=1.0,
            schema_used=self.schema_name,
            error=state["error"],
        )

    original_extract = SpeakerEmbeddingExtractor.extract
    monkeypatch.setattr(SpeakerEmbeddingExtractor, "extract", extract)

    def run():
        return run_speaker_embedding(args, graph=graph)

    def rows():
        return [json.loads(line) for line in Path(args.output_jsonl).read_text().splitlines()]

    def summary():
        return json.loads(Path(args.output_jsonl).with_suffix(".summary.json").read_text())

    def nodes(kind="speaker_embedding"):
        return graph.list_extracted_nodes(kind, ctx, limit=None)

    return locals()


def test_valid_cache_regenerates_exact_manifest(case):
    c = case
    assert c["run"]() == 0
    before = c["rows"]()
    assert len(before) == 2
    assert c["run"]() == 0
    assert c["rows"]() == before
    assert len(c["calls"]) == 1
    assert c["summary"]()["cached_audio"] == 1
    assert c["summary"]()["executed_audio"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "turn-time",
        "speaker",
        "config",
        "model",
        "device",
        "bytes",
        "force",
        "payload",
        "missing-child",
        "legacy",
    ],
)
def test_changes_invalidate_cache(case, change):
    c = case
    assert c["run"]() == 0
    if change in ("turn-time", "speaker"):
        c["turns"][0]["end_sec" if change == "turn-time" else "speaker_id"] = (
            0.5 if change == "turn-time" else "S2"
        )
        c["refresh_diar"]()
    elif change == "config":
        c["args"].embedding_dim = 4
    elif change in ("model", "device"):
        setattr(c["args"], change, "changed")
    elif change == "bytes":
        c["path"].write_bytes(b"changed audio")
    elif change == "force":
        c["args"].force = True
    else:
        node = next(
            n for n in c["nodes"]() if (RECEIPT in n.extracted_content) == (change == "legacy")
        )
        if change == "missing-child":
            c["graph"].delete_node(node.node_id, c["ctx"])
        else:
            content = dict(node.extracted_content)
            if change == "legacy":
                content.pop(RECEIPT)
            else:
                content["speaker_embedding"] = [9.0] * 3
            c["graph"].mark_extraction_completed(node.node_id, content, 1.0, c["ctx"])
    assert c["run"]() == 0
    assert len(c["calls"]) == 2


def test_refresh_same_id_vectors_and_remove_obsolete_speakers(case):
    c = case
    assert c["run"]() == 0
    old = {r["speaker_id"]: r["speaker_node_id"] for r in c["rows"]()}
    c["turns"].pop()
    c["refresh_diar"]()
    c["state"]["value"] = 0.5
    assert c["run"]() == 0
    row = c["rows"]()[0]
    assert row["speaker_node_id"] == old["S0"]
    assert (
        c["graph"].get_extracted_node(old["S0"], c["ctx"]).extracted_content["speaker_embedding"]
        == [0.5] * 3
    )
    assert c["graph"].get_extracted_node(old["S1"], c["ctx"]) is None
    assert len(c["nodes"]()) == 2


@pytest.mark.parametrize(
    "failure", ["error", "partial", "omit", "duplicate", "raise", "nan", "zero", "dimension"]
)
def test_failed_or_incomplete_embedding_never_certified(case, failure):
    c = case
    assert c["run"]() == 0
    c["args"].force = True
    if failure == "error":
        c["state"]["error"] = "backend failed"
    elif failure in ("nan", "zero", "dimension"):
        c["state"]["vector"] = {"nan": [float("nan")] * 3, "zero": [0.0] * 3, "dimension": [1.0]}[
            failure
        ]
    else:
        c["state"][failure] = True
    assert c["run"]() == 2
    assert c["summary"]()["failed_audio"] == 1
    assert c["summary"]()["executed_audio"] == 1
    assert not any(RECEIPT in n.extracted_content for n in c["nodes"]())
    assert c["rows"]() == []


@pytest.mark.parametrize(
    "broken", ["missing", "legacy", "missing-turn", "failed-parent", "tampered-turn"]
)
def test_upstream_not_certified_is_failure_even_with_embedding_cache(case, broken):
    c = case
    assert c["run"]() == 0
    parent = next(n for n in c["nodes"]("speaker_diarization") if RECEIPT in n.extracted_content)
    child = next(n for n in c["nodes"]("speaker_diarization") if "start_sec" in n.extracted_content)
    if broken == "missing":
        for node in c["nodes"]("speaker_diarization"):
            c["graph"].delete_node(node.node_id, c["ctx"])
    elif broken == "missing-turn":
        c["graph"].delete_node(child.node_id, c["ctx"])
    elif broken == "failed-parent":
        c["graph"].mark_extraction_failed(parent.node_id, "failed", c["ctx"])
    else:
        node = parent if broken == "legacy" else child
        content = dict(node.extracted_content)
        if broken == "legacy":
            content.pop(RECEIPT)
        else:
            content["end_sec"] = 20
        c["graph"].mark_extraction_completed(node.node_id, content, 1.0, c["ctx"])
    assert c["run"]() == 2
    assert len(c["calls"]) == 1
    assert c["summary"]()["executed_audio"] == 0


def test_certified_silence_cleans_old_speakers_without_inference(case):
    c = case
    assert c["run"]() == 0
    c["turns"].clear()
    c["refresh_diar"]()
    assert c["run"]() == 0
    assert c["rows"]() == []
    assert c["summary"]()["completed_audio"] == 1
    assert c["summary"]()["executed_audio"] == 0
    assert len(c["nodes"]()) == 1
    assert c["run"]() == 0
    assert c["summary"]()["cached_audio"] == 1
    assert len(c["calls"]) == 1


def test_failed_child_write_retries_without_old_receipt(case, monkeypatch):
    c = case
    assert c["run"]() == 0
    c["args"].force = True
    c["state"]["value"] = 0.25
    child_id = c["rows"]()[0]["speaker_node_id"]
    original = c["graph"].mark_extraction_completed

    def fail(node_id, *args, **kwargs):
        if _id(node_id) == _id(child_id):
            raise RuntimeError("child write failed")
        return original(node_id, *args, **kwargs)

    monkeypatch.setattr(c["graph"], "mark_extraction_completed", fail)
    assert c["run"]() == 2
    assert not any(RECEIPT in n.extracted_content for n in c["nodes"]())
    monkeypatch.setattr(c["graph"], "mark_extraction_completed", original)
    c["args"].force = False
    assert c["run"]() == 0
    assert len(c["calls"]) == 3
    assert (
        c["graph"].get_extracted_node(child_id, c["ctx"]).extracted_content["speaker_embedding"]
        == [0.25] * 3
    )
    assert c["run"]() == 0
    assert len(c["calls"]) == 3


def test_partial_audio_failure_and_other_video_not_touched(case):
    c = case
    for video in ("v", "other"):
        c["graph"].create_raw_node(
            raw_type=RawDataType.AUDIO,
            content="",
            ctx=c["ctx"],
            metadata={"source_path": str(c["path"]), "video_id": video},
            node_id=uuid4().hex,
        )
    assert c["run"]() == 2  # selected second audio has no diarization
    assert len(c["calls"]) == 1
    assert c["summary"]()["attempted_audio"] == 2
    assert c["summary"]()["completed_audio"] == 1
    assert c["summary"]()["failed_audio"] == 1
    assert len(c["rows"]()) == 2


@pytest.mark.parametrize("failure", ["cleanup", "receipt"])
def test_cleanup_or_receipt_failure_is_retryable(case, monkeypatch, failure):
    c = case
    assert c["run"]() == 0
    old_ids = {r["speaker_id"]: r["speaker_node_id"] for r in c["rows"]()}
    c["turns"].pop()
    c["refresh_diar"]()
    original_delete = c["graph"].delete_node
    original_mark = c["graph"].mark_extraction_completed

    def delete(node_id, ctx):
        if _id(node_id) == _id(old_ids["S1"]):
            raise RuntimeError("cleanup failed")
        return original_delete(node_id, ctx)

    def mark(node_id, *args, **kwargs):
        if RECEIPT in (kwargs.get("extracted_content") or {}):
            raise RuntimeError("receipt write failed")
        return original_mark(node_id, *args, **kwargs)

    if failure == "cleanup":
        monkeypatch.setattr(c["graph"], "delete_node", delete)
    else:
        monkeypatch.setattr(c["graph"], "mark_extraction_completed", mark)
    assert c["run"]() == 2
    assert not any(RECEIPT in n.extracted_content for n in c["nodes"]())
    monkeypatch.setattr(c["graph"], "delete_node", original_delete)
    monkeypatch.setattr(c["graph"], "mark_extraction_completed", original_mark)
    assert c["run"]() == 0
    assert len(c["rows"]()) == 1
    assert len(c["calls"]) == 3
    assert c["run"]() == 0
    assert len(c["calls"]) == 3


def test_real_operator_with_mock_backend(case, monkeypatch):
    c = case
    # Restore the pure operator; only its explicitly selected mock backend runs.
    monkeypatch.setattr(SpeakerEmbeddingExtractor, "extract", c["original_extract"])
    assert c["run"]() == 0
    before = c["rows"]()
    assert len(before) == 2
    assert c["run"]() == 0
    assert c["rows"]() == before
    assert c["summary"]()["cached_audio"] == 1
