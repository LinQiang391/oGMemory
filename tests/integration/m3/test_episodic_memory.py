"""Tests for :mod:`m3.episodic_memory` driver.

The driver is now a thin wrapper around :class:`EpisodicMemoryGenerator`
+ :class:`OperatorService.persist_extraction`. The interesting
behaviour (LLM multimodal call construction, frame subsampling,
operator materialization, idempotency) lives at the operator layer —
see ``tests/unit/extraction/operators/test_episodic_memory_generator.py``
for coverage. These tests verify the M3-specific glue:

* the ``episodic_memory.jsonl`` manifest format,
* manifest-driven iteration + frame / audio child lookup,
* idempotent re-run via the parent ``ExtractedNode`` id,
* graceful handling of missing frames / audio,
* backend selection (``--llm minimax|mock|openai``).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from core.graph_enums import RawDataType  # noqa: E402
from core.models import RequestContext  # noqa: E402
from m3.config import M3BENCH_ACCOUNT_ID  # noqa: E402
from m3.episodic_memory import (  # noqa: E402
    _build_llm,
    _index_audio_records,
    _index_frame_records,
    extracted_node_id,
    generate_episodic_for_clip,
    parse_args,
    run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-episodic-test", trace_id="t-test",
    )


def _make_memory_graph():
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    return GraphService(MemoryGraphStore())


def _make_op_service(graph):
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from service.operator_service import OperatorService
    registry = default_registry()
    register_all_defaults(registry)
    return OperatorService(graph, registry=registry)


def _persist_video(graph, video_id, clip_id, ctx):
    seed = f"clip/{video_id}/{clip_id}".encode()
    node_id = hashlib.sha1(seed).hexdigest()[:32]
    graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content=b"",
        ctx=ctx,
        metadata={
            "video_id": video_id, "clip_id": clip_id,
            "start_sec": clip_id * 30,
            "duration_sec": 30.0,
            "source_path": f"/tmp/{video_id}/{clip_id}.mp4",
        },
        node_id=node_id,
    )
    return node_id


def _persist_frame(graph, video_id, clip_id, frame_idx, ctx):
    """Create + persist a frame IMAGE RawNode and wire it via REFERENCES
    to the given video node id (so the driver can find it via
    ``get_neighbors``)."""
    seed = f"frame/{video_id}/{clip_id}/{frame_idx}".encode()
    node_id = hashlib.sha1(seed).hexdigest()[:32]
    graph.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"\xff\xd8\xff\xe0jpeg" + bytes([frame_idx]),
        ctx=ctx,
        metadata={"video_id": video_id, "clip_id": clip_id,
                  "frame_idx": frame_idx, "ts_sec": float(frame_idx)},
        node_id=node_id,
    )
    # The driver itself doesn't traverse edges — it looks up frames by
    # id from the frames manifest. So edges are not required here.
    return node_id


def _persist_audio(graph, video_id, clip_id, ctx):
    seed = f"audio/{video_id}/{clip_id}".encode()
    node_id = hashlib.sha1(seed).hexdigest()[:32]
    graph.create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"RIFF" + b"\x00" * 36,  # WAV magic + filler
        ctx=ctx,
        metadata={"video_id": video_id, "clip_id": clip_id,
                  "duration_sec": 30.0},
        node_id=node_id,
    )
    return node_id


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------

class TestExtractedNodeId:
    def test_deterministic(self):
        a = extracted_node_id("episodic_memory_generator", "vid-xyz")
        b = extracted_node_id("episodic_memory_generator", "vid-xyz")
        assert a == b
        assert len(a) == 32

    def test_varies_with_extractor(self):
        a = extracted_node_id("episodic_memory_generator", "v-1")
        b = extracted_node_id("face_detection", "v-1")
        assert a != b

    def test_varies_with_source(self):
        a = extracted_node_id("episodic_memory_generator", "v-1")
        b = extracted_node_id("episodic_memory_generator", "v-2")
        assert a != b


# ---------------------------------------------------------------------------
# Manifest indexes
# ---------------------------------------------------------------------------

class TestIndexFrameRecords:
    def test_empty_path_returns_empty_index(self, tmp_path):
        assert _index_frame_records(tmp_path / "nonexistent.jsonl") == {}

    def test_groups_by_video_clip_key(self, tmp_path):
        path = tmp_path / "frames.jsonl"
        path.write_text("\n".join([
            json.dumps({
                "video_id": "v1", "clip_id": 0,
                "child_node_ids": ["f1", "f2"],
            }),
            json.dumps({
                "video_id": "v1", "clip_id": 1,
                "child_node_ids": ["f3"],
            }),
            json.dumps({
                "video_id": "v2", "clip_id": 0,
                "child_node_ids": ["f4"],
            }),
        ]))
        idx = _index_frame_records(path)
        assert idx["v1|0"] == ["f1", "f2"]
        assert idx["v1|1"] == ["f3"]
        assert idx["v2|0"] == ["f4"]
        assert len(idx) == 3

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "frames.jsonl"
        path.write_text("\n" + json.dumps({"video_id": "v", "clip_id": 0, "child_node_ids": ["a"]}) + "\n\n")
        idx = _index_frame_records(path)
        assert idx == {"v|0": ["a"]}


class TestIndexAudioRecords:
    def test_none_path_returns_empty(self):
        assert _index_audio_records(None) == {}

    def test_missing_path_returns_empty(self, tmp_path):
        assert _index_audio_records(tmp_path / "nonexistent.jsonl") == {}

    def test_returns_audio_node_id_per_clip(self, tmp_path):
        path = tmp_path / "audio.jsonl"
        path.write_text("\n".join([
            json.dumps({"video_id": "v1", "clip_id": 0, "audio_node_id": "a1"}),
            json.dumps({"video_id": "v1", "clip_id": 1, "audio_node_id": "a2"}),
        ]))
        idx = _index_audio_records(path)
        assert idx["v1|0"] == "a1"
        assert idx["v1|1"] == "a2"

    def test_skips_records_without_audio_node_id(self, tmp_path):
        path = tmp_path / "audio.jsonl"
        path.write_text(json.dumps({
            "video_id": "v1", "clip_id": 0, "audio_node_id": None,
        }))
        assert _index_audio_records(path) == {}


# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------

class TestBuildLLM:
    def test_mock_backend_returns_mock_llm(self):
        llm = _build_llm(parse_args(["--video-records-manifest", "v.jsonl", "--frames-manifest", "f.jsonl", "--llm", "mock"]))
        from providers.llm import MockLLM
        assert isinstance(llm, MockLLM)

    def test_minimax_backend_picks_minimax_class(self):
        from providers.llm import get_minimax_llm
        args = parse_args([
            "--video-records-manifest", "v.jsonl",
            "--frames-manifest", "f.jsonl",
            "--llm", "minimax",
            "--minimax-model", "MiniMax-M3",
        ])
        # Skip API-key validation by setting it via env first.
        import os
        old = os.environ.get("OGMEM_MINIMAX_API_KEY")
        os.environ["OGMEM_MINIMAX_API_KEY"] = "test-key"
        try:
            llm = _build_llm(args)
            assert isinstance(llm, get_minimax_llm())
            assert llm._model == "MiniMax-M3"
        finally:
            if old is None:
                os.environ.pop("OGMEM_MINIMAX_API_KEY", None)
            else:
                os.environ["OGMEM_MINIMAX_API_KEY"] = old


# ---------------------------------------------------------------------------
# Driver integration via generate_episodic_for_clip
# ---------------------------------------------------------------------------

class TestGenerateEpisodicForClip:
    def _setup(self, tmp_path):
        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_id = "v1"
        clip_id = 0
        video_node_id = _persist_video(graph, video_id, clip_id, ctx)
        frame_ids = [
            _persist_frame(graph, video_id, clip_id, i, ctx)
            for i in range(3)
        ]
        audio_id = _persist_audio(graph, video_id, clip_id, ctx)
        video_node = graph.get_raw_node(video_node_id, ctx)
        frame_nodes = [graph.get_raw_node(fid, ctx) for fid in frame_ids]
        audio_node = graph.get_raw_node(audio_id, ctx)
        return graph, op_service, ctx, video_node, frame_nodes, audio_node

    def test_writes_extracted_node_with_multimodal_output(
        self, tmp_path,
    ):
        from providers.llm import MockLLM
        llm = MockLLM()
        llm._mock_multimodal_response = {
            "description": "A robot picks up a cup.",
            "characters": [{"name": "robot_arm"}],
            "events": [],
            "entities": [{"name": "cup", "type": "object"}],
        }
        graph, op_service, ctx, video_node, frame_nodes, audio_node = (
            self._setup(tmp_path)
        )

        result = generate_episodic_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node,
            frame_nodes=frame_nodes,
            audio_node=audio_node,
            llm=llm,
            max_frames=8,
            include_audio=True,
            temperature=0.2,
        )
        assert result["resumed"] is False
        assert result["description"] == "A robot picks up a cup."
        assert result["frame_count"] == 3
        assert result["has_audio"] is True

        parent = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert parent is not None
        assert parent.extractor_type == "episodic_memory_generator"
        assert parent.extracted_content["description"] == "A robot picks up a cup."

    def test_resumed_when_parent_extracted_node_exists(self, tmp_path):
        from providers.llm import MockLLM
        llm = MockLLM()
        llm._mock_multimodal_response = {
            "description": "memo",
            "characters": [], "events": [], "entities": [],
        }
        graph, op_service, ctx, video_node, frame_nodes, audio_node = (
            self._setup(tmp_path)
        )
        first = generate_episodic_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node, frame_nodes=frame_nodes,
            audio_node=audio_node, llm=llm,
            max_frames=8, include_audio=True, temperature=0.2,
        )
        assert first["resumed"] is False

        # Second run with a fresh LLM that returns different content;
        # the driver should short-circuit and return the cached value.
        llm2 = MockLLM()
        llm2._mock_multimodal_response = {"description": "different"}
        second = generate_episodic_for_clip(
            graph=graph, op_service=op_service, ctx=ctx,
            video_raw_node=video_node, frame_nodes=frame_nodes,
            audio_node=audio_node, llm=llm2,
            max_frames=8, include_audio=True, temperature=0.2,
        )
        assert second["resumed"] is True
        assert second["description"] == "memo"  # cached, not "different"


# ---------------------------------------------------------------------------
# Driver entry point: run() with manifest
# ---------------------------------------------------------------------------

class TestRunDriver:
    def _make_setup(self, tmp_path, *, frame_count=3, with_audio=True):
        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_id = "vid_a"
        clip_id = 0
        video_node_id = _persist_video(graph, video_id, clip_id, ctx)
        frame_ids = [
            _persist_frame(graph, video_id, clip_id, i, ctx)
            for i in range(frame_count)
        ]
        records = [{
            "id": video_id,
            "raw_nodes": [{
                "clip_id": clip_id,
                "raw_node_id": video_node_id,
            }],
        }]
        video_manifest = tmp_path / "data.jsonl"
        video_manifest.write_text(
            json.dumps(records[0], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        frames_manifest = tmp_path / "frames.jsonl"
        frames_manifest.write_text(
            json.dumps({
                "video_id": video_id, "clip_id": clip_id,
                "video_raw_node_id": video_node_id,
                "extracted_node_id": "fake",
                "child_node_ids": frame_ids,
                "child_count": frame_count,
                "resumed": False,
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        audio_manifest = None
        if with_audio:
            audio_id = _persist_audio(graph, video_id, clip_id, ctx)
            audio_manifest = tmp_path / "audio.jsonl"
            audio_manifest.write_text(
                json.dumps({
                    "video_id": video_id, "clip_id": clip_id,
                    "audio_node_id": audio_id,
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        return graph, op_service, video_manifest, frames_manifest, audio_manifest

    def test_run_writes_episodic_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")

        graph, op_service, video_manifest, frames_manifest, audio_manifest = (
            self._make_setup(tmp_path, frame_count=3, with_audio=True)
        )

        out_jsonl = tmp_path / "episodic_memory.jsonl"
        args = parse_args([
            "--video-records-manifest", str(video_manifest),
            "--frames-manifest", str(frames_manifest),
            "--audio-manifest", str(audio_manifest),
            "--output-jsonl", str(out_jsonl),
            "--llm", "mock",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        assert rc == 0

        records = [
            json.loads(line) for line in out_jsonl.read_text().splitlines() if line
        ]
        assert len(records) == 1
        rec = records[0]
        assert rec["video_id"] == "vid_a"
        assert rec["clip_id"] == 0
        assert rec["video_raw_node_id"]
        assert rec["extracted_node_id"]
        assert rec["description"]  # MockLLM's schema-derived text
        assert rec["frame_count"] == 3
        assert rec["has_audio"] is True
        assert rec["resumed"] is False

    def test_run_vision_only_when_no_audio(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        graph, op_service, video_manifest, frames_manifest, _ = (
            self._make_setup(tmp_path, frame_count=2, with_audio=False)
        )

        out_jsonl = tmp_path / "episodic_memory.jsonl"
        args = parse_args([
            "--video-records-manifest", str(video_manifest),
            "--frames-manifest", str(frames_manifest),
            "--output-jsonl", str(out_jsonl),
            "--llm", "mock",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        assert rc == 0
        records = [
            json.loads(line) for line in out_jsonl.read_text().splitlines() if line
        ]
        assert records[0]["has_audio"] is False

    def test_run_skips_clips_with_no_frames(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        video_id = "vid_a"
        video_node_id = _persist_video(graph, video_id, 0, ctx)
        # No frames persisted + frames manifest empty.
        video_manifest = tmp_path / "data.jsonl"
        video_manifest.write_text(json.dumps({
            "id": video_id,
            "raw_nodes": [{"clip_id": 0, "raw_node_id": video_node_id}],
        }) + "\n")
        frames_manifest = tmp_path / "frames.jsonl"
        frames_manifest.write_text("")  # empty

        out_jsonl = tmp_path / "episodic_memory.jsonl"
        args = parse_args([
            "--video-records-manifest", str(video_manifest),
            "--frames-manifest", str(frames_manifest),
            "--output-jsonl", str(out_jsonl),
            "--llm", "mock",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        # 0 records written but failure path ran — rc=2 per driver contract.
        assert rc == 2
        assert not out_jsonl.exists() or out_jsonl.read_text() == ""

    def test_missing_video_manifest_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        args = parse_args([
            "--video-records-manifest", str(tmp_path / "nope.jsonl"),
            "--frames-manifest", str(tmp_path / "f.jsonl"),
            "--output-jsonl", str(tmp_path / "o.jsonl"),
            "--llm", "mock",
        ])
        assert run(args) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = parse_args([
            "--video-records-manifest", "v.jsonl",
            "--frames-manifest", "f.jsonl",
        ])
        assert args.llm is None
        assert args.minimax_model is None
        assert args.config is None
        assert args.temperature == 0.2
        assert args.max_frames == 8
        assert args.no_audio is False
        assert args.graph_store == "memory"
        assert args.audio_manifest is None

    def test_no_audio_flag(self):
        args = parse_args([
            "--video-records-manifest", "v.jsonl",
            "--frames-manifest", "f.jsonl",
            "--no-audio",
        ])
        assert args.no_audio is True

    def test_unknown_llm_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([
                "--video-records-manifest", "v.jsonl",
                "--frames-manifest", "f.jsonl",
                "--llm", "magic",
            ])
