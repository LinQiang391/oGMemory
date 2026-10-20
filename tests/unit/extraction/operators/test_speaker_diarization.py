"""Tests for the ``speaker_diarization`` operator + backends.

Three layers:

* Pure alignment helpers — ``_stamp_speaker_on_asr_words``,
  ``_majority_speaker_id``, ``_turns_with_asr_segment_ids``.
* Backends — ``MockDiarizationBackend`` deterministic stub.
* Operator — ``SpeakerDiarizationExtractor.extract(audio, ctx,
  asr_extracted_node=...)`` with the mock backend; asserts the
  parent ``ExtractedNode.extracted_content`` shape + the
  ``SPEAKER_DIARIZATION_MATERIALIZATION`` child payload shape.

All tests run in-process — no pyannote / torch download, no DB.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.diarization_backends import (
    DiarizationBackendError,
    DiarizationResult,
    DiarizationTurn,
    MockDiarizationBackend,
    PyannoteDiarizationBackend,
    build_diarization_backend,
)
from extraction.operators.builtin.speaker_diarization import (
    SpeakerDiarizationExtractor,
    _majority_speaker_id,
    _stamp_speaker_on_asr_words,
    _turns_with_asr_segment_ids,
)
from extraction.operators.materialize import SPEAKER_DIARIZATION_MATERIALIZATION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_sine_wav(path: Path, *, duration_sec: float = 1.0) -> None:
    n = int(round(16000 * duration_sec))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(n):
            sample = int(0.5 * 32767 * math.sin(2 * math.pi * 440 * i / 16000))
            frames.extend(struct.pack("<h", max(-32767, min(32767, sample))))
        w.writeframes(bytes(frames))


def _audio_node(path: Path) -> Any:
    return create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"",
        account_id="acct_test",
        metadata={
            "format": "wav", "sample_rate": 16000, "channels": 1,
            "source_path": str(path),
            "video_id": "vid_test", "clip_id": 0,
        },
        node_id="audio_diar_test".ljust(32, "x")[:32],
    )


def _make_asr_node(
    audio_id: str, segments: list[dict],
) -> Any:
    """Build a fake voice_asr ExtractedNode for alignment tests."""
    return create_extracted_node(
        source_raw_id=audio_id,
        extractor_type="voice_asr",
        extraction_schema="voice_asr_v1",
        account_id="acct_test",
        extracted_content={
            "language": "en",
            "duration_sec": 30.0,
            "segments": segments,
            "source_audio_node_id": audio_id,
        },
        node_id="asr_test".ljust(32, "x")[:32],
    )


from core.graph_models import create_extracted_node  # noqa: E402  (intentional)


@pytest.fixture
def req_ctx() -> RequestContext:
    return RequestContext(
        account_id="acct_test",
        user_id="u_test", agent_id="a_test",
        session_id="s_test", trace_id="t_test",
    )


# ---------------------------------------------------------------------------
# Pure alignment helpers
# ---------------------------------------------------------------------------

def test_stamp_speaker_on_asr_words_basic():
    turns = [
        {"start_sec": 0.0, "end_sec": 2.0, "speaker_id": "SPEAKER_00"},
        {"start_sec": 2.0, "end_sec": 4.0, "speaker_id": "SPEAKER_01"},
    ]
    words = [
        {"word": "hello", "start_sec": 0.5, "end_sec": 1.0, "probability": 0.9},
        {"word": "hi", "start_sec": 2.5, "end_sec": 3.0, "probability": 0.95},
    ]
    out = _stamp_speaker_on_asr_words(words, turns)
    assert out[0]["speaker_id"] == "SPEAKER_00"
    assert out[1]["speaker_id"] == "SPEAKER_01"


def test_stamp_speaker_on_asr_words_outside_turns():
    turns = [
        {"start_sec": 1.0, "end_sec": 2.0, "speaker_id": "SPEAKER_00"},
    ]
    words = [
        # before any turn
        {"word": "early", "start_sec": 0.0, "end_sec": 0.5, "probability": 0.5},
        # inside the turn
        {"word": "mid", "start_sec": 1.2, "end_sec": 1.7, "probability": 0.5},
    ]
    out = _stamp_speaker_on_asr_words(words, turns)
    assert out[0]["speaker_id"] is None
    assert out[1]["speaker_id"] == "SPEAKER_00"


def test_stamp_speaker_on_asr_words_partial_overlap():
    """Half-open overlap: word overlaps turn iff ws < te AND we > ts."""
    turns = [
        {"start_sec": 1.0, "end_sec": 2.0, "speaker_id": "SPEAKER_00"},
    ]
    # Word ending exactly at 1.0 doesn't overlap (we == ts).
    words = [
        {"word": "a", "start_sec": 0.5, "end_sec": 1.0, "probability": 0.5},
    ]
    out = _stamp_speaker_on_asr_words(words, turns)
    assert out[0]["speaker_id"] is None


def test_majority_speaker_id_clear_winner():
    turns = [
        {"start_sec": 0.0, "end_sec": 10.0, "speaker_id": "SPEAKER_00"},
        {"start_sec": 5.0, "end_sec": 15.0, "speaker_id": "SPEAKER_01"},
    ]
    seg_words = [
        {"word": "a", "start_sec": 1.0, "end_sec": 1.2, "probability": 0.9},
        {"word": "b", "start_sec": 1.2, "end_sec": 1.5, "probability": 0.9},
        {"word": "c", "start_sec": 12.0, "end_sec": 12.5, "probability": 0.9},
    ]
    assert _majority_speaker_id(seg_words, turns) == "SPEAKER_00"


def test_majority_speaker_id_empty_words():
    turns = [{"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "X"}]
    assert _majority_speaker_id([], turns) is None


def test_majority_speaker_id_empty_turns():
    seg_words = [{"word": "a", "start_sec": 0.0, "end_sec": 0.5, "probability": 0.9}]
    assert _majority_speaker_id(seg_words, []) is None


def test_turns_with_asr_segment_ids():
    turns = [
        {"start_sec": 0.0, "end_sec": 5.0, "speaker_id": "SPEAKER_00"},
        {"start_sec": 5.0, "end_sec": 10.0, "speaker_id": "SPEAKER_01"},
    ]
    asr_segments = [
        {"asr_id": "asr_1", "start_sec": 1.0, "end_sec": 2.0},
        {"asr_id": "asr_2", "start_sec": 3.0, "end_sec": 4.0},
        {"asr_id": "asr_3", "start_sec": 6.0, "end_sec": 7.0},
        # outside any turn
        {"asr_id": "asr_4", "start_sec": 11.0, "end_sec": 12.0},
    ]
    out = _turns_with_asr_segment_ids(turns, asr_segments)
    assert out[0]["asr_segment_ids"] == ["asr_1", "asr_2"]
    assert out[1]["asr_segment_ids"] == ["asr_3"]


def test_turns_with_asr_segment_ids_skips_missing_asr_id():
    turns = [{"start_sec": 0.0, "end_sec": 5.0, "speaker_id": "X"}]
    asr_segments = [
        {"start_sec": 1.0, "end_sec": 2.0},  # no asr_id
        {"asr_id": "asr_1", "start_sec": 3.0, "end_sec": 4.0},
    ]
    out = _turns_with_asr_segment_ids(turns, asr_segments)
    assert out[0]["asr_segment_ids"] == ["asr_1"]


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

def test_mock_backend_default_two_speakers():
    backend = MockDiarizationBackend(
        default_speakers=2, default_duration_sec=20.0,
    )
    result = backend.diarize(
        "/tmp/x.wav",
        min_speakers=None, max_speakers=None,
    )
    assert isinstance(result, DiarizationResult)
    assert result.num_speakers == 2
    assert len(result.turns) == 2
    assert result.turns[0].speaker_id == "SPEAKER_00"
    assert result.turns[0].start_sec == 0.0
    assert result.turns[0].end_sec == pytest.approx(10.0)
    assert result.turns[1].speaker_id == "SPEAKER_01"


def test_mock_backend_synthetic_pattern_covers_full_duration():
    backend = MockDiarizationBackend(
        default_speakers=3, default_duration_sec=30.0,
    )
    result = backend.diarize(
        "/tmp/x.wav", min_speakers=None, max_speakers=None,
    )
    assert result.turns[0].start_sec == 0.0
    assert result.turns[-1].end_sec == pytest.approx(30.0)


def test_mock_backend_fixture_pins_result():
    fixture_turns = [
        DiarizationTurn(0.0, 1.5, "ALICE"),
        DiarizationTurn(1.5, 3.0, "BOB"),
    ]
    backend = MockDiarizationBackend(
        fixture={"/tmp/x.wav": DiarizationResult(
            turns=fixture_turns, num_speakers=2, model="mock",
        )},
    )
    result = backend.diarize(
        "/tmp/x.wav", min_speakers=None, max_speakers=None,
    )
    assert result.turns[0].speaker_id == "ALICE"
    assert result.turns[1].speaker_id == "BOB"


def test_mock_backend_fail_on_path_substring():
    backend = MockDiarizationBackend(
        default_speakers=2, fail_on_path_substring="bad",
    )
    with pytest.raises(DiarizationBackendError):
        backend.diarize("/tmp/bad_clip.wav", min_speakers=None, max_speakers=None)


# ---------------------------------------------------------------------------
# Pyannote backend (without actually loading)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wrapped", [False, True], ids=["3.x", "4.x"])
@pytest.mark.parametrize("empty", [False, True], ids=["speech", "empty"])
def test_pyannote_backend_output_compatibility(wrapped, empty):
    tracks = [] if empty else [
        (SimpleNamespace(start=2, end=4), "track_a", "SPEAKER_01"),
        (SimpleNamespace(start=0, end=3), "track_b", "SPEAKER_00"),
        (SimpleNamespace(start=4, end=5), "track_c", "SPEAKER_00"),
    ]
    annotation = SimpleNamespace(itertracks=Mock(return_value=iter(tracks)))
    output = SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=None,
    ) if wrapped else annotation
    backend = PyannoteDiarizationBackend()
    backend._pipeline = Mock(return_value=output)

    result = backend.diarize("clip.wav", min_speakers=1, max_speakers=3)

    backend._pipeline.assert_called_once_with(
        "clip.wav", min_speakers=1, max_speakers=3,
    )
    annotation.itertracks.assert_called_once_with(yield_label=True)
    assert result.model == "pyannote/speaker-diarization-3.1"
    assert result.num_speakers == (0 if empty else 2)
    assert result.turns == ([] if empty else [
        DiarizationTurn(0.0, 3.0, "SPEAKER_00"),
        DiarizationTurn(2.0, 4.0, "SPEAKER_01"),
        DiarizationTurn(4.0, 5.0, "SPEAKER_00"),
    ])


@pytest.mark.parametrize("output", [None, SimpleNamespace(speaker_diarization=None)])
def test_pyannote_backend_unsupported_output(output):
    backend = PyannoteDiarizationBackend()
    backend._pipeline = Mock(return_value=output)
    with pytest.raises(DiarizationBackendError, match="unsupported diarization output"):
        backend.diarize("clip.wav", min_speakers=None, max_speakers=None)
    backend._pipeline.assert_called_once_with("clip.wav")


def test_pyannote_backend_requires_token():
    backend = PyannoteDiarizationBackend()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(DiarizationBackendError) as excinfo:
            backend._resolve_token()
    assert "HUGGINGFACE_TOKEN" in str(excinfo.value)


def test_pyannote_backend_token_from_env():
    backend = PyannoteDiarizationBackend()
    with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "hf_x"}, clear=True):
        assert backend._resolve_token() == "hf_x"


def test_pyannote_backend_token_precedence_arg_over_env():
    backend = PyannoteDiarizationBackend(hf_token="hf_arg")
    with patch.dict(
        "os.environ",
        {"HUGGINGFACE_TOKEN": "hf_env", "HF_TOKEN": "hf_old"},
        clear=True,
    ):
        assert backend._resolve_token() == "hf_arg"


def test_pyannote_backend_token_falls_back_to_HF_TOKEN():
    backend = PyannoteDiarizationBackend()
    with patch.dict(
        "os.environ", {"HF_TOKEN": "hf_old"}, clear=True,
    ):
        assert backend._resolve_token() == "hf_old"


def test_pyannote_backend_name():
    backend = PyannoteDiarizationBackend()
    assert backend.name == "pyannote"


def test_pyannote_backend_load_error_when_pyannote_missing():
    backend = PyannoteDiarizationBackend(hf_token="hf_x")
    with patch.dict("sys.modules", {"pyannote": None, "pyannote.audio": None}):
        with pytest.raises(DiarizationBackendError) as excinfo:
            backend._load()
    assert "pyannote" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# build_diarization_backend factory
# ---------------------------------------------------------------------------

def test_build_diarization_backend_default_is_pyannote():
    backend = build_diarization_backend("pyannote", hf_token="hf_x")
    assert isinstance(backend, PyannoteDiarizationBackend)


def test_build_diarization_backend_aliases():
    assert isinstance(
        build_diarization_backend("pyannote.audio", hf_token="hf_x"),
        PyannoteDiarizationBackend,
    )
    assert isinstance(build_diarization_backend("mock"), MockDiarizationBackend)


def test_build_diarization_backend_unknown_raises():
    with pytest.raises(DiarizationBackendError):
        build_diarization_backend("nonexistent")


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def test_speaker_diarization_operator_no_asr(tmp_path: Path, req_ctx):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav, duration_sec=2.0)
    audio = _audio_node(wav)
    extractor = SpeakerDiarizationExtractor(backend="mock")
    extractor._backend = MockDiarizationBackend(
        default_speakers=2, default_duration_sec=2.0,
    )
    result = extractor.extract(audio, req_ctx)
    # 2 speakers, 2s default → 1s each.
    assert result.error is None
    assert result.extracted_content["num_speakers"] == 2
    assert len(result.extracted_content["turns"]) == 2


def test_speaker_diarization_operator_with_asr_alignment(
    tmp_path: Path, req_ctx,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav, duration_sec=20.0)
    audio = _audio_node(wav)
    asr_node = _make_asr_node(audio.node_id, [
        {
            "asr_id": "asr_1",
            "start_sec": 1.0, "end_sec": 3.0,
            "text": "hi",
            "no_speech_prob": 0.0,
            "words": [
                {"word": "hi", "start_sec": 1.0, "end_sec": 3.0,
                 "probability": 0.9},
            ],
        },
        {
            "asr_id": "asr_2",
            "start_sec": 11.0, "end_sec": 13.0,
            "text": "hello",
            "no_speech_prob": 0.0,
            "words": [
                {"word": "hello", "start_sec": 11.0, "end_sec": 13.0,
                 "probability": 0.9},
            ],
        },
    ])
    extractor = SpeakerDiarizationExtractor(backend="mock")
    extractor._backend = MockDiarizationBackend(
        default_speakers=2, default_duration_sec=20.0,
    )
    result = extractor.extract(audio, req_ctx, asr_extracted_node=asr_node)
    assert result.error is None
    # The per_segment_speaker map should map asr_1 → first speaker,
    # asr_2 → second speaker.
    per_seg = result.extracted_content["per_segment_speaker"]
    assert per_seg["asr_1"] == "SPEAKER_00"
    assert per_seg["asr_2"] == "SPEAKER_01"
    # Each ASR word got stamped with the right speaker_id.
    aligned = result.extracted_content["aligned_asr_extracted_content"]
    seg_words = aligned["segments"][0]["words"]
    assert seg_words[0]["speaker_id"] == "SPEAKER_00"


def test_speaker_diarization_operator_missing_path(
    tmp_path: Path, req_ctx,
):
    audio = _audio_node(tmp_path / "does_not_exist.wav")
    extractor = SpeakerDiarizationExtractor(backend="mock")
    result = extractor.extract(audio, req_ctx)
    assert result.error
    assert result.extracted_content["turns"] == []
    assert result.confidence == 0.0


def test_speaker_diarization_operator_backend_failure(
    tmp_path: Path, req_ctx,
):
    wav = tmp_path / "x.wav"
    _write_sine_wav(wav)
    audio = _audio_node(wav)
    extractor = SpeakerDiarizationExtractor(backend="mock")
    with patch.object(
        extractor._backend, "diarize",
        side_effect=DiarizationBackendError("mock", "boom"),
    ):
        result = extractor.extract(audio, req_ctx)
    assert result.error
    assert "boom" in result.error
    assert result.extracted_content["backend"] == "mock"


def test_speaker_diarization_operator_metadata_fields():
    extractor = SpeakerDiarizationExtractor(backend="mock")
    assert extractor.extractor_type == "speaker_diarization"
    assert extractor.schema_name == "speaker_diarization_v1"
    assert extractor.supported_types == [RawDataType.AUDIO]
    info = extractor.info
    assert info.materialization is SPEAKER_DIARIZATION_MATERIALIZATION


# ---------------------------------------------------------------------------
# Materialization policy
# ---------------------------------------------------------------------------

def test_speaker_diarization_materialization_child_id_deterministic():
    item = {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "SPEAKER_00"}
    cid_a = SPEAKER_DIARIZATION_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    cid_b = SPEAKER_DIARIZATION_MATERIALIZATION.child_id_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav"},
    )
    assert cid_a == cid_b
    assert len(cid_a) == 32


def test_speaker_diarization_materialization_payload_shape():
    item = {
        "start_sec": 0.5, "end_sec": 1.5,
        "speaker_id": "SPEAKER_00",
        "asr_segment_ids": ["asr_1", "asr_2"],
    }
    payload = SPEAKER_DIARIZATION_MATERIALIZATION.child_payload_fn(
        item, "audio_xyz", {"source_path": "/tmp/x.wav", "video_id": "v1", "clip_id": 7},
    )
    assert payload["start_sec"] == 0.5
    assert payload["end_sec"] == 1.5
    assert payload["speaker_id"] == "SPEAKER_00"
    assert payload["asr_segment_ids"] == ["asr_1", "asr_2"]
    assert payload["source_audio_node_id"] == "audio_xyz"
    assert payload["source_video_id"] == "v1"
    assert payload["source_clip_id"] == 7


def test_speaker_diarization_materialization_edge_props():
    item = {
        "start_sec": 0.5, "end_sec": 1.5,
        "speaker_id": "SPEAKER_00",
    }
    props = SPEAKER_DIARIZATION_MATERIALIZATION.edge_properties_fn(
        item, "audio_xyz", None,
    )
    assert props["relation"] == "speaker_turn"
    assert props["speaker_id"] == "SPEAKER_00"
    assert props["start_sec"] == 0.5
    assert props["end_sec"] == 1.5


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------

def test_builtin_import_succeeds_without_pyannote():
    from extraction.operators import builtin
    assert hasattr(builtin, "SpeakerDiarizationExtractor")
