"""ASR/VAD cache contract using real graph persistence and stubbed inference."""

import importlib
import json
import os
from copy import deepcopy
from uuid import uuid4

import pytest

from core.graph_enums import RawDataType
from core.models import RequestContext
from extraction.operators.base import ExtractionResult
from m3.audio_cache import AudioCache, signature
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


@pytest.fixture(params=["memory", "sql"])
def cache_graph(request):
    # Reject duplicate INSERTs: ordinary MemoryGraphStore silently overwrites.
    class StrictStore(MemoryGraphStore):
        def create_node(self, payload, ctx):
            assert payload["node_id"] not in self._nodes
            return super().create_node(payload, ctx)

    if request.param == "sql":
        if os.environ.get("OGMEM_SQL_INTEGRATION") != "1" or not os.environ.get("OGMEM_TEST_DSN"):
            pytest.skip("Requires explicit dedicated SQL test database")
        from providers.graph_store import SQLGraphStore

        store = SQLGraphStore(connection_string=os.environ["OGMEM_TEST_DSN"])
    else:
        store = StrictStore()
    graph = GraphService(store)
    ctx = RequestContext(
        account_id="cache-test-" + uuid4().hex,
        user_id="test",
        agent_id="test",
        session_id="test",
        trace_id="test",
    )
    yield graph, ctx
    for node in store.list_nodes(None, ctx):
        graph.delete_node(node.node_id, ctx)


@pytest.fixture(params=["voice_asr", "voice_segmentation"])
def run_case(request, tmp_path, monkeypatch, cache_graph):
    kind = request.param
    module = importlib.import_module("m3." + kind)
    cls = getattr(
        importlib.import_module("extraction.operators.builtin." + kind),
        "VoiceASRExtractor" if kind == "voice_asr" else "VoiceSegmentationExtractor",
    )
    graph, ctx = cache_graph
    path = tmp_path / "audio.wav"
    path.write_bytes(b"fake-audio-input")
    audio = graph.create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=ctx,
        metadata={"source_path": str(path), "video_id": "v", "clip_id": 0},
    )
    argv = [
        "--account-id",
        ctx.account_id,
        "--video",
        "v",
        "--output-jsonl",
        str(tmp_path / "out.jsonl"),
    ]
    if kind == "voice_asr":
        argv += ["--backend", "mock"]
    args = module.parse_args(argv)
    calls = []
    segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "old text",
            "confidence": 0.8,
            "asr_id": uuid4().hex,
            "voice_id": uuid4().hex,
        }
    ]

    def extract(self, raw, context):
        calls.append(raw.node_id)
        return ExtractionResult(
            node_id=raw.node_id,
            extracted_content={"segments": deepcopy(segments), "language": "en"},
            confidence=1.0,
            schema_used="test",
        )

    monkeypatch.setattr(cls, "extract", extract)

    def run():
        return getattr(module, "run_" + kind)(args, graph=graph)

    return locals()


def rows(case):
    from pathlib import Path

    return [json.loads(line) for line in Path(case["args"].output_jsonl).read_text().splitlines()]


def test_valid_cache_regenerates_manifest_without_inference(run_case):
    c = run_case
    assert c["run"]() == 0
    before = rows(c)
    assert c["run"]() == 0
    assert len(c["calls"]) == 1
    assert rows(c) == before


@pytest.mark.parametrize(
    "change", ["config", "bytes", "force", "missing-child", "legacy", "payload", "pending"]
)
def test_invalid_cache_recomputes(run_case, change):
    c = run_case
    assert c["run"]() == 0
    children = AudioCache(c["graph"], c["ctx"], c["kind"]).lookup(
        c["audio"], signature(c["audio"], c["args"], c["kind"], c["ctx"].account_id)
    )
    child = children[0]
    if change == "config":
        if c["kind"] == "voice_asr":
            c["args"].beam_size += 1
        else:
            c["args"].pad_sec += 0.1
    elif change == "bytes":
        c["path"].write_bytes(b"changed audio")
    elif change == "force":
        c["args"].force = True
    elif change == "missing-child":
        c["graph"].delete_node(child.node_id, c["ctx"])
    elif change == "legacy":
        AudioCache(c["graph"], c["ctx"], c["kind"]).invalidate(c["audio"])
    elif change == "pending":
        c["graph"].mark_extraction_failed(child.node_id, "failed", c["ctx"])
    else:
        c["graph"].mark_extraction_completed(child.node_id, {"changed": True}, 1.0, c["ctx"])
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
    assert len(c["graph"].list_extracted_nodes(c["kind"], c["ctx"])) == 2


def test_refresh_updates_same_id_payload_and_removes_obsolete_segments(run_case):
    c = run_case
    c["segments"].append(
        dict(c["segments"][0], asr_id=uuid4().hex, voice_id=uuid4().hex, start_sec=2.0, end_sec=3.0)
    )
    assert c["run"]() == 0
    old = rows(c)
    c["segments"].pop()
    c["segments"][0].update(text="new text", confidence=0.2)
    c["args"].force = True
    assert c["run"]() == 0
    current = rows(c)
    assert len(current) == 1
    assert current[0]["segment_node_id"] == old[0]["segment_node_id"]
    child = c["graph"].get_extracted_node(current[0]["segment_node_id"], c["ctx"])
    field = "text" if c["kind"] == "voice_asr" else "confidence"
    assert child.extracted_content[field] == c["segments"][0][field]
    assert c["graph"].get_extracted_node(old[1]["segment_node_id"], c["ctx"]) is None
    c["args"].force = False
    assert c["run"]() == 0
    assert len(c["calls"]) == 2


def test_silence_is_valid_cached_result(run_case):
    c = run_case
    c["segments"].clear()
    assert c["run"]() == c["run"]() == 0
    assert len(c["calls"]) == 1
    assert rows(c) == []


def test_missing_file_does_not_reuse_receipt(run_case):
    c = run_case
    assert c["run"]() == 0
    c["path"].unlink()
    # Stub inference still succeeds; the production extractor reports missing file.
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
    assert signature(c["audio"], c["args"], c["kind"], c["ctx"].account_id) is None


def test_failed_forced_refresh_invalidates_old_receipt(run_case, monkeypatch):
    c = run_case
    assert c["run"]() == 0
    original = c["cls"].extract

    def fail(*args):
        raise RuntimeError("backend down")

    monkeypatch.setattr(c["cls"], "extract", fail)
    c["args"].force = True
    assert c["run"]() == 2
    assert rows(c) == []
    monkeypatch.setattr(c["cls"], "extract", original)
    c["args"].force = False
    assert c["run"]() == 0
    assert len(c["calls"]) == 2


def test_partial_materialization_cannot_be_cached(run_case, monkeypatch):
    from service.operator_service import OperatorService

    c = run_case
    original = OperatorService.persist_extraction

    def partial(self, **kwargs):
        result = original(self, **kwargs)
        result["materialization"]["failed"] = 1
        return result

    monkeypatch.setattr(OperatorService, "persist_extraction", partial)
    assert c["run"]() == 2
    assert rows(c) == []
    monkeypatch.setattr(OperatorService, "persist_extraction", original)
    assert c["run"]() == 0
    assert len(c["calls"]) == 2


def test_downstream_speaker_annotation_preserves_asr_cache(run_case):
    c = run_case
    if c["kind"] != "voice_asr":
        return
    assert c["run"]() == 0
    child = c["graph"].get_extracted_node(rows(c)[0]["segment_node_id"], c["ctx"])
    c["graph"].mark_extraction_completed(
        child.node_id, {**child.extracted_content, "speaker_id": "speaker-1"}, 1.0, c["ctx"]
    )
    assert c["run"]() == 0
    assert len(c["calls"]) == 1


def test_refresh_does_not_touch_other_video(run_case):
    c = run_case
    other = c["graph"].create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=c["ctx"],
        metadata={"source_path": str(c["path"]), "video_id": "other", "clip_id": 0},
    )
    foreign_child = c["graph"].create_extracted_node(
        source_raw_id=other.node_id,
        extractor_type=c["kind"],
        extraction_schema="test",
        extracted_content={"sentinel": "keep"},
        ctx=c["ctx"],
    )
    c["args"].force = True
    assert c["run"]() == 0
    assert len(c["calls"]) == 1
    assert c["graph"].get_extracted_node(foreign_child.node_id, c["ctx"]).extracted_content == {
        "sentinel": "keep"
    }


def test_partial_audio_failure_returns_nonzero_and_reports_counts(run_case, monkeypatch):
    c = run_case
    c["graph"].create_raw_node(
        raw_type=RawDataType.AUDIO,
        content="",
        ctx=c["ctx"],
        metadata={"source_path": str(c["path"]) + ".other", "video_id": "v", "clip_id": 1},
    )
    original = c["cls"].extract

    def extract(self, raw, context):
        if raw.metadata["clip_id"] == 1:
            raise RuntimeError("one bad clip")
        return original(self, raw, context)

    monkeypatch.setattr(c["cls"], "extract", extract)
    assert c["run"]() == 2
    assert len(rows(c)) == 1
    from pathlib import Path

    summary = json.loads(Path(c["args"].output_jsonl).with_suffix(".summary.json").read_text())
    assert summary["attempted_audio"] == 2
    assert summary["failed_audio"] == 1
    assert summary["completed_audio"] == 1


def test_old_receipt_not_reused_after_cleanup_failure(run_case, monkeypatch):
    c = run_case
    assert c["run"]() == 0
    original = c["graph"].delete_node

    def fail(*args):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(c["graph"], "delete_node", fail)
    c["args"].force = True
    assert c["run"]() == 2
    monkeypatch.setattr(c["graph"], "delete_node", original)
    c["args"].force = False
    assert c["run"]() == 0
    assert len(c["calls"]) == 3
    assert len(c["graph"].list_extracted_nodes(c["kind"], c["ctx"])) == 2


def test_retry_repairs_child_created_before_provenance_edge_failure(run_case, monkeypatch):
    from core.graph_enums import GraphEdgeType

    c = run_case
    original = c["graph"]._store.create_edge
    attempts = []

    def create_edge(edge, context):
        attempts.append(edge)
        if len(attempts) == 1:
            raise RuntimeError("edge write failed after child INSERT")
        return original(edge, context)

    monkeypatch.setattr(c["graph"]._store, "create_edge", create_edge)
    assert c["run"]() == 2
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
    child_id = rows(c)[0]["segment_node_id"]
    neighbors = c["graph"].get_neighbors(
        child_id, c["ctx"], edge_types=[GraphEdgeType.EXTRACTED_FROM], direction="outgoing"
    )
    assert len(neighbors) == 1


def test_failed_receipt_write_does_not_publish_cache(run_case, monkeypatch):
    from m3.audio_cache import RECEIPT

    c = run_case
    original = c["graph"]._store.update_node

    def update(payload, ctx):
        if RECEIPT in (payload.get("extracted_content") or {}):
            raise RuntimeError("receipt write failed")
        return original(payload, ctx)

    monkeypatch.setattr(c["graph"]._store, "update_node", update)
    assert c["run"]() == 2
    assert all(
        RECEIPT not in n.extracted_content
        for n in c["graph"].list_extracted_nodes(c["kind"], c["ctx"])
    )
    monkeypatch.setattr(c["graph"]._store, "update_node", original)
    assert c["run"]() == 0
    assert len(c["calls"]) == 2
