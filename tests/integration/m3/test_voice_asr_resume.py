"""Tests for voice_asr Layer-1 per-AUDIO-node resume.

When a previous run already produced voice_asr ExtractedNode children
for an AUDIO node, the driver must:
- skip the Whisper call (no ``extractor.extract`` invocation)
- re-emit the JSONL rows from the cached children
- record the per-AUDIO-node resume in the log line

The driver also respects ``--force`` to bypass the cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from m3.voice_asr import run_voice_asr


def _make_audio_node(node_id: str, *, video_id: str, clip_id: int) -> MagicMock:
    node = MagicMock()
    node.node_id = node_id
    node.metadata = {"video_id": video_id, "clip_id": clip_id}
    return node


def _make_asr_child(
    *,
    node_id: str,
    source_audio_node_id: str,
    start_sec: float,
    end_sec: float,
    text: str,
    status: str = "completed",
) -> MagicMock:
    child = MagicMock()
    child.node_id = node_id
    child.source_raw_id = source_audio_node_id
    child.extraction_status = status
    child.extracted_content = {
        "source_audio_node_id": source_audio_node_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": round(end_sec - start_sec, 3),
        "text": text,
        "asr_id": node_id,
        "language": "en",
        "words": [],
        "no_speech_prob": 0.0,
    }
    return child


def _build_graph(*, audio_nodes: list, voice_asr_children: list) -> MagicMock:
    graph = MagicMock()
    graph.list_raw_nodes.side_effect = (
        lambda raw_type, ctx: audio_nodes if raw_type.name == "AUDIO" else []
    )
    graph.list_extracted_nodes.side_effect = (
        lambda extractor_type, ctx: (
            voice_asr_children if extractor_type == "voice_asr" else []
        )
    )
    return graph


class TestVoiceAsrResume:
    def test_resume_skips_whisper_when_cached(
        self, tmp_path, monkeypatch,
    ):
        """Whisper should NOT be invoked when cached segments exist."""
        audio_node = _make_audio_node(
            "audio-1", video_id="v1", clip_id=0,
        )
        cached = _make_asr_child(
            node_id="asr-child-1",
            source_audio_node_id="audio-1",
            start_sec=0.85,
            end_sec=2.13,
            text="hello world",
        )
        graph = _build_graph(
            audio_nodes=[audio_node],
            voice_asr_children=[cached],
        )

        # Spy on Whisper to confirm it's never called.
        extractor = MagicMock()
        extractor.extract = MagicMock(side_effect=AssertionError(
            "Whisper should not be invoked when cached children exist"
        ))
        monkeypatch.setattr(
            "extraction.operators.builtin.voice_asr.VoiceASRExtractor",
            MagicMock(return_value=extractor),
        )

        output_jsonl = tmp_path / "voice_asr.jsonl"
        args = MagicMock()
        args.backend = "faster_whisper"
        args.model = None
        args.language = "en"
        args.beam_size = 5
        args.no_word_timestamps = False
        args.no_vad_filter = False
        args.base_url = None
        args.api_key = None
        args.device = "cpu"
        args.timeout_sec = 300.0
        args.video = None
        args.output_jsonl = str(output_jsonl)
        args.graph_store = "memory"
        args.connection_string = None
        args.account_id = "acct"
        args.force = False

        rc = run_voice_asr(args, graph=graph)
        assert rc == 0

        # The JSONL must have one row re-emitted from the cached child.
        lines = [
            json.loads(l)
            for l in output_jsonl.read_text().splitlines()
            if l.strip()
        ]
        assert len(lines) == 1
        assert lines[0]["text"] == "hello world"
        assert lines[0]["audio_node_id"] == "audio-1"
        assert lines[0]["video_id"] == "v1"
        assert lines[0]["clip_id"] == 0

    def test_resume_continues_whisper_when_no_cache(
        self, tmp_path, monkeypatch,
    ):
        """Without cached children, Whisper MUST run."""
        audio_node = _make_audio_node(
            "audio-2", video_id="v2", clip_id=0,
        )
        graph = _build_graph(
            audio_nodes=[audio_node],
            voice_asr_children=[],  # no cache
        )

        # Stub Whisper to return a single segment.
        extractor = MagicMock()
        extractor.extract = MagicMock(return_value=MagicMock(
            error=None,
            extracted_content={
                "segments": [{
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "duration_sec": 1.5,
                    "text": "fresh transcription",
                    "no_speech_prob": 0.0,
                    "asr_id": "asr-fresh-1",
                    "words": [],
                }],
                "language": "en",
                "duration_sec": 1.5,
                "model": "mock",
                "backend": "mock",
            },
            schema_used="voice_asr_v1",
            node_id="audio-2",
            confidence=0.9,
        ))
        monkeypatch.setattr(
            "extraction.operators.builtin.voice_asr.VoiceASRExtractor",
            MagicMock(return_value=extractor),
        )
        # Stub the op_service built inside run_voice_asr.
        op_service = MagicMock()
        op_service.persist_extraction = MagicMock(return_value={
            "error": None,
            "extracted_node_id": "ext-audio-2",
            "materialized_children": [
                {"child_node_id": "asr-fresh-1"},
            ],
        })
        monkeypatch.setattr(
            "service.operator_service.OperatorService",
            MagicMock(return_value=op_service),
        )

        output_jsonl = tmp_path / "voice_asr.jsonl"
        args = MagicMock()
        args.backend = "faster_whisper"
        args.model = None
        args.language = "en"
        args.beam_size = 5
        args.no_word_timestamps = False
        args.no_vad_filter = False
        args.base_url = None
        args.api_key = None
        args.device = "cpu"
        args.timeout_sec = 300.0
        args.video = None
        args.output_jsonl = str(output_jsonl)
        args.graph_store = "memory"
        args.connection_string = None
        args.account_id = "acct"
        args.force = False

        rc = run_voice_asr(args, graph=graph)
        assert rc == 0
        extractor.extract.assert_called_once()
        # JSONL has one row from the fresh Whisper call.
        lines = [
            json.loads(l)
            for l in output_jsonl.read_text().splitlines()
            if l.strip()
        ]
        assert len(lines) == 1
        assert lines[0]["text"] == "fresh transcription"

    def test_force_bypasses_cache(self, tmp_path, monkeypatch):
        audio_node = _make_audio_node(
            "audio-3", video_id="v3", clip_id=0,
        )
        cached = _make_asr_child(
            node_id="asr-cached-3",
            source_audio_node_id="audio-3",
            start_sec=0.0, end_sec=1.0, text="stale",
        )
        graph = _build_graph(
            audio_nodes=[audio_node],
            voice_asr_children=[cached],
        )

        # Even though the cache is hot, --force must run Whisper.
        extractor = MagicMock()
        extractor.extract = MagicMock(return_value=MagicMock(
            error=None,
            extracted_content={
                "segments": [{
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "duration_sec": 1.0,
                    "text": "fresh after force",
                    "no_speech_prob": 0.0,
                    "asr_id": "asr-fresh-3",
                    "words": [],
                }],
                "language": "en",
                "duration_sec": 1.0,
                "model": "mock",
                "backend": "mock",
            },
            schema_used="voice_asr_v1",
            node_id="audio-3",
            confidence=0.9,
        ))
        monkeypatch.setattr(
            "extraction.operators.builtin.voice_asr.VoiceASRExtractor",
            MagicMock(return_value=extractor),
        )
        op_service = MagicMock()
        op_service.persist_extraction = MagicMock(return_value={
            "error": None,
            "extracted_node_id": "ext-audio-3",
            "materialized_children": [{"child_node_id": "asr-fresh-3"}],
        })
        monkeypatch.setattr(
            "service.operator_service.OperatorService",
            MagicMock(return_value=op_service),
        )

        output_jsonl = tmp_path / "voice_asr.jsonl"
        args = MagicMock()
        args.backend = "faster_whisper"
        args.model = None
        args.language = "en"
        args.beam_size = 5
        args.no_word_timestamps = False
        args.no_vad_filter = False
        args.base_url = None
        args.api_key = None
        args.device = "cpu"
        args.timeout_sec = 300.0
        args.video = None
        args.output_jsonl = str(output_jsonl)
        args.graph_store = "memory"
        args.connection_string = None
        args.account_id = "acct"
        args.force = True  # the whole point

        rc = run_voice_asr(args, graph=graph)
        assert rc == 0
        extractor.extract.assert_called_once()