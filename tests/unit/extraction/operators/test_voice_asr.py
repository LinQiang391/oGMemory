"""Tests for the ``voice_asr`` operator and its backends.

Three layers:

* Pure dataclass / factory tests for
  :class:`MockASRBackend`, :class:`OpenAIHTTPBackend`, and
  :class:`FasterWhisperBackend`. No network, no model load.
* Backend factory tests — error handling for unknown names, lazy
  import of heavy deps (faster-whisper, requests).
* Operator test — ``VoiceASRExtractor.extract(audio_raw, ctx)``
  on a synthetic WAV via the Mock backend, asserting the
  parent ``ExtractedNode.extracted_content`` shape + the
  ``VOICE_ASR_MATERIALIZATION`` child payload shape.

All tests run in-process — no ffmpeg, no DB, no model downloads.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.asr_backends import (
    ASRBackendError,
    ASRResult,
    ASRSegment,
    ASRWord,
    FasterWhisperBackend,
    MockASRBackend,
    OpenAIHTTPBackend,
    build_asr_backend,
    make_asr_id,
)
from extraction.operators.builtin.voice_asr import VoiceASRExtractor
from extraction.operators.materialize import VOICE_ASR_MATERIALIZATION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_sine_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    duration_sec: float = 1.0,
    freq_hz: float = 440.0,
    amplitude: float = 0.5,
) -> None:
    """Write a 16-bit-PCM mono WAV containing a sine wave.

    Default amplitude is well above the operator's loudness floor
    so any backend would consider it speech. Tests that need silence
    pass ``amplitude=0.0``.
    """
    n = int(round(sample_rate * duration_sec))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(
                amplitude * 32767
                * math.sin(2 * math.pi * freq_hz * i / sample_rate)
            )
            frames.extend(struct.pack("<h", max(-32767, min(32767, sample))))
        w.writeframes(bytes(frames))


def _audio_node(path: Path, *, node_id: str = "audio_test_0") -> Any:
    return create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"",
        account_id="acct_test",
        metadata={
            "format": "wav", "sample_rate": 16000, "channels": 1,
            "source_path": str(path),
            "video_id": "vid_test", "clip_id": 0,
        },
        node_id=node_id.ljust(32, "x")[:32],
    )


@pytest.fixture
def req_ctx() -> RequestContext:
    return RequestContext(
        account_id="acct_test",
        user_id="u_test",
        agent_id="a_test",
        session_id="s_test",
        trace_id="t_test",
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_make_asr_id_is_deterministic():
    a = make_asr_id("/tmp/a.wav", 0.5, 1.5)
    b = make_asr_id("/tmp/a.wav", 0.5, 1.5)
    assert a == b
    # ID is a 32-char hex string (valid Postgres UUID, no dashes) -
    # the SQL graph store's node_id column is typed UUID, so a
    # prefixed id like "asr_..." would fail.
    assert len(a) == 32
    int(a, 16)  # raises ValueError if not hex


def test_make_asr_id_differs_on_start():
    assert make_asr_id("/tmp/a.wav", 0.0, 1.0) != make_asr_id("/tmp/a.wav", 0.5, 1.5)


def test_make_asr_id_differs_on_path():
    assert make_asr_id("/tmp/a.wav", 0.0, 1.0) != make_asr_id("/tmp/b.wav", 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

def test_mock_backend_default_one_segment():
    backend = MockASRBackend(default_text="hi")
    result = backend.transcribe(
        "/tmp/x.wav",
        language="en", beam_size=5,
        word_timestamps=True, vad_filter=True,
    )
    assert isinstance(result, ASRResult)
    assert result.language == "en"
    assert result.model == "mock"
    assert len(result.segments) == 1
    assert result.segments[0].text == "hi"
    assert result.segments[0].words == []


def test_mock_backend_fixture_pins_result():
    seg = ASRSegment(
        start_sec=0.5, end_sec=1.5, text="pinned",
        no_speech_prob=0.0,
        words=[ASRWord("pinned", 0.5, 1.5, 0.99)],
    )
    backend = MockASRBackend(
        fixture={"/tmp/x.wav": ASRResult(
            language="en", duration_sec=1.0,
            segments=[seg], model="mock",
        )},
    )
    result = backend.transcribe(
        "/tmp/x.wav",
        language=None, beam_size=1,
        word_timestamps=True, vad_filter=False,
    )
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.segments[0].text == "pinned"
    assert result.segments[0].words[0].word == "pinned"


def test_mock_backend_fail_on_path_substring():
    backend = MockASRBackend(
        default_text="x",
        fail_on_path_substring="bad",
    )
    with pytest.raises(ASRBackendError):
        backend.transcribe(
            "/tmp/bad_clip.wav",
            language="en", beam_size=1,
            word_timestamps=False, vad_filter=False,
        )


def test_mock_backend_uses_default_language_when_none():
    backend = MockASRBackend(default_text="x", default_language="zh")
    result = backend.transcribe(
        "/tmp/y.wav",
        language=None, beam_size=1,
        word_timestamps=False, vad_filter=False,
    )
    assert result.language == "zh"


# ---------------------------------------------------------------------------
# OpenAIHTTPBackend
# ---------------------------------------------------------------------------

def test_openai_http_backend_requires_base_url():
    with pytest.raises(ASRBackendError):
        OpenAIHTTPBackend(base_url="")


def test_openai_http_backend_parses_verbose_json(tmp_path: Path):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    backend = OpenAIHTTPBackend(
        base_url="http://example.test",
        model="whisper-1",
        api_key="k",
    )
    # Build the response payload the OpenAI /audio/transcriptions
    # endpoint returns when response_format=verbose_json +
    # timestamp_granularities=word.
    payload = {
        "task": "transcribe",
        "language": "en",
        "duration": 1.0,
        "text": "hello world",
        "segments": [
            {
                "id": 0,
                "start": 0.0, "end": 1.0,
                "text": "hello world",
                "no_speech_prob": 0.01,
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.98},
                ],
            },
        ],
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = payload
    fake_resp.text = json.dumps(payload)
    fake_requests = MagicMock()
    fake_requests.post.return_value = fake_resp
    with patch.object(backend, "_requests", return_value=fake_requests):
        result = backend.transcribe(
            str(wav),
            language="en", beam_size=1,
            word_timestamps=True, vad_filter=True,
        )
    assert result.language == "en"
    assert result.duration_sec == pytest.approx(1.0)
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.text == "hello world"
    assert len(seg.words) == 2
    assert seg.words[0].word == "hello"
    assert seg.words[0].probability == pytest.approx(0.99)


def test_openai_http_backend_http_error_returns_asr_error(tmp_path: Path):
    wav = tmp_path / "x.wav"
    _write_sine_wave = _write_sine_wav  # alias for readability
    _write_sine_wave(wav)
    backend = OpenAIHTTPBackend(base_url="http://example.test", model="whisper-1")
    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = "internal server error"
    fake_requests = MagicMock()
    fake_requests.post.return_value = fake_resp
    with patch.object(backend, "_requests", return_value=fake_requests):
        with pytest.raises(ASRBackendError) as excinfo:
            backend.transcribe(
                str(wav),
                language="en", beam_size=1,
                word_timestamps=False, vad_filter=False,
            )
    assert "500" in str(excinfo.value)


def test_openai_http_backend_degrades_when_words_missing(tmp_path: Path):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    backend = OpenAIHTTPBackend(base_url="http://example.test", model="whisper-1")
    payload = {
        "language": "en", "duration": 1.0, "text": "x",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "x", "no_speech_prob": 0.0},
        ],
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = payload
    fake_resp.text = json.dumps(payload)
    fake_requests = MagicMock()
    fake_requests.post.return_value = fake_resp
    with patch.object(backend, "_requests", return_value=fake_requests):
        result = backend.transcribe(
            str(wav),
            language="en", beam_size=1,
            word_timestamps=True, vad_filter=False,
        )
    assert result.segments[0].words == []  # graceful, no crash


# ---------------------------------------------------------------------------
# FasterWhisperBackend (mocked model)
# ---------------------------------------------------------------------------

def test_faster_whisper_backend_returns_asr_result(tmp_path: Path):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    backend = FasterWhisperBackend(model="tiny")
    # Build a fake model that mimics faster_whisper's API surface:
    # transcribe(...) returns (segments_iter, info).
    class FakeSeg:
        def __init__(self, start, end, text, words):
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    class FakeWord:
        def __init__(self, word, start, end, prob):
            self.word = word
            self.start = start
            self.end = end
            self.probability = prob

    segs = [
        FakeSeg(0.0, 0.5, "hi", [
            FakeWord("hi", 0.0, 0.5, 0.99),
        ]),
    ]
    info = MagicMock()
    info.language = "en"
    info.duration = 1.0
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(segs), info)
    with patch.object(backend, "_load", return_value=fake_model):
        result = backend.transcribe(
            str(wav),
            language="en", beam_size=5,
            word_timestamps=True, vad_filter=True,
        )
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.segments[0].text == "hi"
    assert result.segments[0].words[0].word == "hi"


def test_faster_whisper_backend_missing_dep_message():
    """When ``faster_whisper`` isn't installed, _load() raises
    ASRBackendError with an install hint. We simulate the missing
    import by patching the import to raise ImportError."""
    backend = FasterWhisperBackend()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        with pytest.raises(ASRBackendError) as excinfo:
            backend._load()
    assert "faster-whisper" in str(excinfo.value)
    assert "uv pip install" in str(excinfo.value)


# ---------------------------------------------------------------------------
# build_asr_backend factory
# ---------------------------------------------------------------------------

def test_build_asr_backend_default_is_faster_whisper():
    backend = build_asr_backend("faster_whisper")
    assert isinstance(backend, FasterWhisperBackend)


def test_build_asr_backend_aliases():
    assert isinstance(build_asr_backend("whisper"), FasterWhisperBackend)
    assert isinstance(build_asr_backend("faster-whisper"), FasterWhisperBackend)
    assert isinstance(
        build_asr_backend("openai", base_url="http://x.test"),
        OpenAIHTTPBackend,
    )
    assert isinstance(
        build_asr_backend("http", base_url="http://x.test"),
        OpenAIHTTPBackend,
    )
    assert isinstance(build_asr_backend("mock"), MockASRBackend)


def test_build_asr_backend_unknown_raises():
    with pytest.raises(ASRBackendError):
        build_asr_backend("nonexistent")


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def test_voice_asr_operator_happy_path(tmp_path: Path, req_ctx: RequestContext):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav, duration_sec=2.0)
    audio = _audio_node(wav)
    extractor = VoiceASRExtractor(backend="mock", language="en")
    result = extractor.extract(audio, req_ctx)
    assert result.error is None
    assert result.schema_used == "voice_asr_v1"
    # Language lives on the parent, not per-segment (the operator
    # runs Whisper once per audio, getting a single detected
    # language for the whole clip).
    assert result.extracted_content["language"] == "en"
    assert result.extracted_content["model"] == "mock"
    assert result.extracted_content["backend"] == "mock"
    assert len(result.extracted_content["segments"]) == 1
    seg = result.extracted_content["segments"][0]
    assert seg["text"] == "Hello world."
    assert seg["start_sec"] == 0.0
    # Mock backend spans [0, default_duration_sec=30.0] so downstream
    # diarization alignment has a real time range to overlap.
    assert seg["end_sec"] == pytest.approx(30.0)
    assert seg["asr_id"]
    # asr_id is a 32-char hex UUID (not prefixed) so the SQL graph
    # store's UUID-typed node_id column accepts it.
    assert len(seg["asr_id"]) == 32
    int(seg["asr_id"], 16)


def test_voice_asr_operator_missing_path(tmp_path: Path, req_ctx: RequestContext):
    audio = _audio_node(tmp_path / "does_not_exist.wav", node_id="missing")
    extractor = VoiceASRExtractor(backend="mock")
    result = extractor.extract(audio, req_ctx)
    assert result.error
    assert result.extracted_content["segments"] == []
    assert result.confidence == 0.0


def test_voice_asr_operator_backend_failure_is_caught(
    tmp_path: Path, req_ctx: RequestContext,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    extractor = VoiceASRExtractor(backend="mock")
    # Force a backend failure by patching the backend object.
    with patch.object(
        extractor._backend, "transcribe",
        side_effect=ASRBackendError("mock", "boom"),
    ):
        result = extractor.extract(audio, req_ctx)
    assert result.error
    assert "boom" in result.error
    assert result.extracted_content["backend"] == "mock"
    assert result.confidence == 0.0


def test_voice_asr_operator_stamps_asr_id(tmp_path: Path, req_ctx: RequestContext):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    extractor = VoiceASRExtractor(backend="mock")
    result = extractor.extract(audio, req_ctx)
    seg = result.extracted_content["segments"][0]
    # Mock backend spans [0, 30.0] by default.
    expected = make_asr_id(str(wav), 0.0, 30.0)
    assert seg["asr_id"] == expected


def test_voice_asr_operator_confidence_no_segments(
    tmp_path: Path, req_ctx: RequestContext,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    extractor = VoiceASRExtractor(backend="mock")

    # Force empty-segments result.
    empty_result = ASRResult(
        language="en", duration_sec=0.0, segments=[], model="mock",
    )
    with patch.object(extractor._backend, "transcribe", return_value=empty_result):
        result = extractor.extract(audio, req_ctx)
    assert result.confidence == 0.0


def test_voice_asr_extractor_metadata_fields():
    """info() advertises the right extractor_type / schema / supported_types."""
    extractor = VoiceASRExtractor(backend="mock")
    assert extractor.extractor_type == "voice_asr"
    assert extractor.schema_name == "voice_asr_v1"
    assert extractor.supported_types == [RawDataType.AUDIO]
    info = extractor.info
    assert info.materialization is VOICE_ASR_MATERIALIZATION


# ---------------------------------------------------------------------------
# Materialization policy
# ---------------------------------------------------------------------------

def test_voice_asr_materialization_child_id_from_valid_asr_id():
    """A valid 32-hex asr_id (UUID-shaped) passes through unchanged."""
    valid_uuid = "abcdef0123456789abcdef0123456789"  # 32 hex chars
    item = {"asr_id": valid_uuid, "start_sec": 0.0, "end_sec": 1.0}
    cid = VOICE_ASR_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    assert cid == valid_uuid
    int(cid, 16)  # still valid hex


def test_voice_asr_materialization_child_id_rejects_non_uuid_asr_id():
    """A non-UUID asr_id (e.g. legacy ``asr_abc...`` prefix) must NOT
    pass through - the SQL graph store's node_id column is typed UUID
    and would reject it. The policy hashes (source, start, end)
    instead and returns a valid 32-hex id."""
    item = {"asr_id": "asr_abcdef0123456789", "start_sec": 0.0, "end_sec": 1.0}
    cid = VOICE_ASR_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    assert len(cid) == 32
    int(cid, 16)  # valid hex UUID
    # Must NOT be the raw asr_id (which isn't a UUID).
    assert cid != "asr_abcdef0123456789"


def test_voice_asr_materialization_child_id_fallback():
    """When the operator didn't stamp an asr_id, the policy hashes
    (source, start, end) so re-runs are still deterministic."""
    item = {"start_sec": 1.5, "end_sec": 2.5}
    cid_a = VOICE_ASR_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    cid_b = VOICE_ASR_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    assert cid_a == cid_b
    assert len(cid_a) == 32
    assert cid_a != VOICE_ASR_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/y.wav"},
    )


def test_voice_asr_materialization_payload_shape():
    item = {
        "asr_id": "asr_aaa",
        "start_sec": 0.5, "end_sec": 1.5, "duration_sec": 1.0,
        "text": "hi", "no_speech_prob": 0.05, "language": "en",
        "words": [{"word": "hi", "start_sec": 0.5, "end_sec": 1.5, "probability": 0.9}],
    }
    payload = VOICE_ASR_MATERIALIZATION.child_payload_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav", "video_id": "v1", "clip_id": 7},
    )
    assert payload["text"] == "hi"
    assert payload["start_sec"] == 0.5
    assert payload["end_sec"] == 1.5
    assert payload["asr_id"] == "asr_aaa"
    assert payload["language"] == "en"
    assert payload["source_audio_node_id"] == "audio_xyz"
    assert payload["source_video_id"] == "v1"
    assert payload["source_clip_id"] == 7
    assert len(payload["words"]) == 1


def test_voice_asr_materialization_edge_props():
    item = {
        "asr_id": "asr_aaa", "start_sec": 0.5, "end_sec": 1.5,
    }
    props = VOICE_ASR_MATERIALIZATION.edge_properties_fn(
        item, "audio_xyz", None,
    )
    assert props["relation"] == "asr_segment"
    assert props["asr_id"] == "asr_aaa"
    assert props["start_sec"] == 0.5
    assert props["end_sec"] == 1.5


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------

def test_builtin_import_succeeds_without_faster_whisper():
    """``import extraction.operators.builtin`` must succeed even
    without faster-whisper / pyannote / speechbrain installed. The
    lazy import contract from the operators/__init__.py makes this
    possible."""
    from extraction.operators import builtin
    # VoiceASRExtractor should be importable (it's there as a class
    # stub) — construction may fail without the backend, but the
    # symbol exists.
    assert hasattr(builtin, "VoiceASRExtractor")