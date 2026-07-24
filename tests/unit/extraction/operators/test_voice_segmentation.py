"""Tests for the voice_segmentation operator.

Three layers:

* Pure helpers — _pcm16le_to_float_mono, _frame_rms_dbfs,
  _segment_voice_id, detect_voice_segments. No I/O, no graph.
* Operator — VoiceSegmentationExtractor.extract(audio_raw, ctx) on
  a synthetic WAV file written to tmp_path.
* Materialization policy — VOICE_SEGMENTATION_MATERIALIZATION's
  child_id_fn / child_payload_fn / edge_properties_fn behave as
  documented.

All tests run in-process — no ffmpeg, no DB.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType, RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.voice_segmentation import (
    VoiceSegmentationExtractor,
    _frame_rms_dbfs,
    _pcm16le_to_float_mono,
    _segment_voice_id,
    detect_voice_segments,
)
from extraction.operators.materialize import VOICE_SEGMENTATION_MATERIALIZATION


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

    The default amplitude (0.5 of full scale, ~-6 dBFS) is well above
    the operator's default -40 dBFS threshold, so this clip produces
    a single contiguous voice segment. Tests that need silence use
    ``amplitude=0.0``.
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


def _write_pattern_wav(
    path: Path,
    *,
    pattern: list[tuple[float, float, float]],
    sample_rate: int = 16000,
) -> None:
    """Write a WAV from a pattern of (start_sec, end_sec, amplitude)
    intervals. Useful for tests that want speech + silence + speech.
    """
    duration = max(end for _, end, _ in pattern) if pattern else 1.0
    n = int(round(sample_rate * duration))
    samples = [0.0] * n
    for start, end, amp in pattern:
        s = int(round(start * sample_rate))
        e = int(round(end * sample_rate))
        for i in range(s, min(e, n)):
            # 1 kHz sine inside the segment so energy isn't pure silence.
            samples[i] = amp * math.sin(2 * math.pi * 1000 * i / sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for s in samples:
            frames.extend(struct.pack("<h", int(max(-1, min(1, s)) * 32767)))
        w.writeframes(bytes(frames))


def _ctx() -> RequestContext:
    # Use the M3BENCH_ACCOUNT_ID default — the driver builds its own
    # RequestContext with this id, so the test's seeded graph must
    # match or the access check filters the audio node out.
    from m3.config import M3BENCH_ACCOUNT_ID
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


def _audio_raw_node(path: Path, *, video_id: str = "v", clip_id: int = 0,
                    node_id: str | None = None) -> object:
    """Build a minimal AUDIO RawNode pointing at the given WAV path."""
    import uuid
    nid = node_id or uuid.uuid4().hex[:32]
    return create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"",
        account_id="acc_test",
        metadata={
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "source_path": str(path),
            "video_id": video_id,
            "clip_id": clip_id,
        },
        node_id=nid,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_pcm16le_to_float_mono_round_trip(tmp_path: Path):
    path = tmp_path / "x.wav"
    _write_sine_wav(path, duration_sec=0.1, freq_hz=440, amplitude=0.3)
    samples, sr = _pcm16le_to_float_mono(str(path))
    assert sr == 16000
    assert len(samples) == 1600  # 0.1s @ 16kHz
    # All samples in [-1, 1].
    assert all(-1.0 <= s <= 1.0 for s in samples)
    # Peak amplitude should be roughly the input amplitude.
    assert max(abs(s) for s in samples) == pytest.approx(0.3, abs=1e-2)


def test_pcm16le_to_float_mono_rejects_stereo(tmp_path: Path):
    path = tmp_path / "x.wav"
    # Write a stereo WAV manually.
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    with pytest.raises(ValueError, match="mono"):
        _pcm16le_to_float_mono(str(path))


def test_frame_rms_dbfs_zero_samples_is_neg100():
    assert _frame_rms_dbfs([]) == -100.0


def test_frame_rms_dbfs_full_scale_is_zero():
    # A 1.0-amplitude sine has RMS ~0.707, so dBFS = 20*log10(0.707) ≈ -3.
    samples = [math.sin(2 * math.pi * 1000 * i / 16000) for i in range(1600)]
    db = _frame_rms_dbfs(samples)
    assert db == pytest.approx(-3.01, abs=0.1)


def test_segment_voice_id_deterministic_and_unique_per_offset():
    a = _segment_voice_id("v1", "/path/audio.wav", 1.234)
    b = _segment_voice_id("v1", "/path/audio.wav", 1.234)
    c = _segment_voice_id("v1", "/path/audio.wav", 1.235)
    d = _segment_voice_id("v1", "/path/other.wav", 1.234)
    assert a == b
    assert a != c  # different offset → different id
    assert a != d  # different source → different id
    assert a.startswith("voice_")


def test_detect_voice_segments_continuous_speech_produces_one_segment():
    """1 second of constant 0.5-amplitude tone at 16 kHz → one segment
    spanning (approximately) the full duration."""
    samples = [
        0.5 * math.sin(2 * math.pi * 440 * i / 16000)
        for i in range(16000)
    ]
    segs = detect_voice_segments(samples, 16000)
    assert len(segs) == 1
    seg = segs[0]
    assert seg["start_sec"] == pytest.approx(0.0, abs=0.05)
    assert seg["end_sec"] == pytest.approx(1.0, abs=0.05)
    assert seg["duration_sec"] == pytest.approx(1.0, abs=0.1)
    assert seg["voice_id"].startswith("voice_")
    assert seg["rms_dbfs"] < -5  # -6 dBFS-ish for amp 0.5


def test_detect_voice_segments_silence_produces_no_segments():
    """All-zero samples → no segments at all."""
    segs = detect_voice_segments([0.0] * 16000, 16000)
    assert segs == []


def test_detect_voice_segments_short_burst_below_min_dur_filtered():
    """A 100 ms burst (below the 250 ms default min) → dropped."""
    samples = [0.0] * 15000
    # 100 ms of speech at 0.5 amplitude.
    for i in range(1500):
        samples[5000 + i] = 0.5 * math.sin(2 * math.pi * 440 * i / 16000)
    segs = detect_voice_segments(samples, 16000, min_segment_sec=0.25)
    assert segs == []


def test_detect_voice_segments_two_runs_with_silence_between():
    """Two 300 ms runs of speech separated by 200 ms silence → two
    segments."""
    samples = [0.0] * 16000  # 1 second
    for start in (0.0, 0.5):
        for i in range(int(0.3 * 16000)):
            samples[int(start * 16000) + i] = 0.5 * math.sin(
                2 * math.pi * 440 * i / 16000
            )
    segs = detect_voice_segments(samples, 16000)
    # 2 segments, sorted by start. The operator pads each segment by
    # 0.1s on each side, so the reported end of seg 0 may be ≥ the
    # reported start of seg 1 — that's expected (the padding is
    # there to preserve context). Verify via the raw segment body:
    # the true end (mid of silence) must be < the true start of seg 1.
    assert len(segs) == 2
    assert segs[0]["start_sec"] == pytest.approx(0.0, abs=0.15)
    assert segs[1]["start_sec"] == pytest.approx(0.5, abs=0.15)
    # The first burst ends at sample 12800 (0.8s), the second starts
    # at sample 8000 (0.5s). After the 0.1s pad they're 0.5 + 0.1 = 0.6
    # vs 0.8 - 0.1 = 0.7 — so a 100ms gap remains at the segment core.
    gap = segs[1]["start_sec"] - segs[0]["end_sec"]
    assert -0.05 <= gap <= 0.2  # pad causes overlap, but not deep


def test_detect_voice_segments_long_run_capped_by_max_segment_sec():
    """A 5-second continuous run, with max_segment_sec=2.0, splits
    into ~3 chunks."""
    samples = [
        0.5 * math.sin(2 * math.pi * 440 * i / 16000)
        for i in range(5 * 16000)
    ]
    segs = detect_voice_segments(
        samples, 16000, max_segment_sec=2.0, min_segment_sec=0.1,
    )
    assert len(segs) >= 2
    # Total covered time ≈ 5s, allow some pad overlap.
    total = sum(s["duration_sec"] for s in segs)
    assert total == pytest.approx(5.0, abs=0.5)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def test_operator_supported_types_and_metadata():
    op = VoiceSegmentationExtractor()
    assert op.supported_types == [RawDataType.AUDIO]
    assert op.extractor_type == "voice_segmentation"
    assert op.schema_name == "voice_segmentation_v1"
    info = op.info
    assert info.materialization is VOICE_SEGMENTATION_MATERIALIZATION


def test_operator_extracts_segments_from_synthetic_wav(tmp_path: Path):
    """End-to-end: write a 1s sine, run the operator, check the
    output shape (segments list, voice_id, duration)."""
    wav = tmp_path / "audio.wav"
    _write_sine_wav(wav, duration_sec=1.0, freq_hz=440, amplitude=0.5)
    raw = _audio_raw_node(wav)
    op = VoiceSegmentationExtractor()
    result = op.extract(raw, _ctx())
    assert result.error is None
    ec = result.extracted_content
    assert ec["format"] == "wav"
    assert ec["sample_rate"] == 16000
    assert ec["duration_sec"] == pytest.approx(1.0, abs=0.05)
    assert isinstance(ec["segments"], list)
    assert len(ec["segments"]) == 1
    seg = ec["segments"][0]
    assert seg["voice_id"].startswith("voice_")
    # voice_id is sha1-derived with source_path baked in.
    assert "audio.wav" not in seg["voice_id"]  # only the hash digest
    # Confidence is speech-coverage ratio — full clip is speech,
    # so it should be near 1.0.
    assert result.confidence == pytest.approx(1.0, abs=0.05)


def test_operator_extracts_stable_voice_id_across_runs(tmp_path: Path):
    """Same audio → same voice_id on a re-run."""
    wav = tmp_path / "audio.wav"
    _write_sine_wav(wav, duration_sec=0.6, freq_hz=440, amplitude=0.5)
    raw = _audio_raw_node(wav)
    op = VoiceSegmentationExtractor()

    result1 = op.extract(raw, _ctx())
    result2 = op.extract(raw, _ctx())
    assert result1.extracted_content["segments"][0]["voice_id"] == \
        result2.extracted_content["segments"][0]["voice_id"]


def test_operator_returns_error_when_audio_missing(tmp_path: Path):
    """Audio path doesn't exist → returns ExtractionResult with
    error set, segments empty, confidence 0."""
    raw = _audio_raw_node(tmp_path / "nope.wav")  # file doesn't exist
    op = VoiceSegmentationExtractor()
    result = op.extract(raw, _ctx())
    assert result.error is not None
    assert result.confidence == 0.0
    assert result.extracted_content["segments"] == []


def test_operator_handles_silence_with_zero_confidence(tmp_path: Path):
    """All-silence audio → 0 segments, confidence 0."""
    wav = tmp_path / "silent.wav"
    _write_sine_wav(wav, duration_sec=1.0, amplitude=0.0)
    raw = _audio_raw_node(wav)
    op = VoiceSegmentationExtractor()
    result = op.extract(raw, _ctx())
    assert result.extracted_content["segments"] == []
    assert result.confidence == 0.0


def test_operator_handles_two_runs_of_speech(tmp_path: Path):
    """Two speech runs separated by silence → 2 segments in the
    output."""
    wav = tmp_path / "audio.wav"
    _write_pattern_wav(wav, pattern=[
        (0.0, 0.4, 0.5),
        (0.7, 1.1, 0.5),
    ])
    raw = _audio_raw_node(wav)
    op = VoiceSegmentationExtractor()
    result = op.extract(raw, _ctx())
    segs = result.extracted_content["segments"]
    assert len(segs) == 2
    # The voice_ids are different (different start times → different
    # sha1 inputs) — that's the point: voice_clustering uses these
    # ids as cross-clip anchor candidates.
    assert segs[0]["voice_id"] != segs[1]["voice_id"]


# ---------------------------------------------------------------------------
# Materialization policy
# ---------------------------------------------------------------------------

def test_policy_child_id_uses_voice_id_when_present():
    item = {"voice_id": "voice_abc123def456abc1"}
    nid = VOICE_SEGMENTATION_MATERIALIZATION.child_id_fn(
        item, "raw_audio_id", {"video_id": "v", "clip_id": 0},
    )
    assert nid == "voice_abc123def456abc1"[:32]


def test_policy_child_id_falls_back_to_sha1_without_voice_id():
    item = {"start_sec": 1.5, "end_sec": 3.0}
    nid = VOICE_SEGMENTATION_MATERIALIZATION.child_id_fn(
        item, "raw_audio_id", {"source_path": "/path/audio.wav"},
    )
    assert len(nid) == 32  # sha1 hex


def test_policy_child_payload_carries_provenance():
    item = {
        "voice_id": "voice_xyz",
        "start_sec": 1.0, "end_sec": 2.5,
        "duration_sec": 1.5, "confidence": 0.9,
        "rms_dbfs": -22.5, "frame_count": 48,
    }
    source_metadata = {"video_id": "v", "clip_id": 3}
    payload = VOICE_SEGMENTATION_MATERIALIZATION.child_payload_fn(
        item, "raw_audio_id", source_metadata,
    )
    assert payload["voice_id"] == "voice_xyz"
    assert payload["start_sec"] == 1.0
    assert payload["end_sec"] == 2.5
    assert payload["duration_sec"] == 1.5
    assert payload["confidence"] == 0.9
    assert payload["rms_dbfs"] == -22.5
    assert payload["frame_count"] == 48
    # Provenance comes from the source metadata.
    assert payload["source_video_id"] == "v"
    assert payload["source_clip_id"] == 3
    assert payload["source_audio_node_id"] == "raw_audio_id"


def test_policy_edge_properties_carry_voice_id_and_times():
    item = {"voice_id": "voice_xyz", "start_sec": 1.0, "end_sec": 2.5}
    props = VOICE_SEGMENTATION_MATERIALIZATION.edge_properties_fn(
        item, "raw_audio_id", {"video_id": "v"},
    )
    assert props["relation"] == "voice_segment"
    assert props["voice_id"] == "voice_xyz"
    assert props["start_sec"] == 1.0
    assert props["end_sec"] == 2.5


def test_policy_is_extracted_kind_not_raw():
    """Voice segments are structured output (start/end/voice_id), not
    raw audio bytes — they live as ExtractedNodes, not RawNodes."""
    assert VOICE_SEGMENTATION_MATERIALIZATION.child_node_kind == "extracted"
    assert VOICE_SEGMENTATION_MATERIALIZATION.child_raw_type is None
    assert VOICE_SEGMENTATION_MATERIALIZATION.collection_field == "segments"


# ---------------------------------------------------------------------------
# Driver + stage wiring
# ---------------------------------------------------------------------------

def test_voice_segmentation_stage_is_registered():
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.voice_segmentation" in DEFAULT_STAGE_REGISTRY


def test_voice_segmentation_stage_emits_segments_through_pipeline(tmp_path: Path):
    """End-to-end: write a 1s sine WAV, create an AUDIO RawNode +
    VIDEO RawNode pointing at it, run the stage, assert one
    voice_segment ExtractedNode landed in the graph."""
    import struct as _st
    import wave as _w
    from core.graph_models import create_raw_node
    from core.graph_enums import RawDataType
    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService

    wav = tmp_path / "audio.wav"
    _write_sine_wav(wav, duration_sec=1.0, freq_hz=440, amplitude=0.5)

    graph = GraphService(MemoryGraphStore())
    req_ctx = _ctx()

    # VIDEO RawNode (parent).
    video_node = create_raw_node(
        raw_type=RawDataType.VIDEO,
        content=b"",
        account_id=req_ctx.account_id,
        metadata={
            "source_path": str(wav),  # not used by VAD but consistent shape
            "video_id": "vid_test", "clip_id": 0,
        },
        node_id="vid_test_0".ljust(32, "x"),
    )
    graph._store.create_node(video_node.serialize(), req_ctx)

    # AUDIO RawNode pointing at the WAV. The driver's own
    # RequestContext uses M3BENCH_ACCOUNT_ID, so the seed must too.
    audio_node = create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"",
        account_id=req_ctx.account_id,
        metadata={
            "format": "wav", "sample_rate": 16000, "channels": 1,
            "source_path": str(wav),
            "video_id": "vid_test", "clip_id": 0,
        },
        node_id="audio_vid_test_0".ljust(32, "x"),
    )
    graph._store.create_node(audio_node.serialize(), req_ctx)

    pipe_ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="vid_test", seconds=10, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=req_ctx.account_id,
        artifacts={"voice_segments": tmp_path / "run" / "voice_segments.jsonl"},
        variables={}, graph=graph,
    )
    stage = StageSpec(
        id="voice_segmentation", type="m3.voice_segmentation",
        needs=["extract_audio"], enabled=True,
        params={},
    )
    result = stages_mod.voice_segmentation_stage(pipe_ctx, stage)
    assert result.ok is True
    jsonl_path = tmp_path / "run" / "voice_segments.jsonl"
    assert jsonl_path.exists()
    assert result.counts["segments"] == 1

    # The voice_segment ExtractedNode landed in the graph with the
    # expected payload shape. Filter by voice_id presence to
    # distinguish the per-segment children from the parent
    # lightweight summary (both have extractor_type=voice_segmentation).
    voice_segments = [
        n for n in graph.list_extracted_nodes(None, req_ctx)
        if getattr(n, "extractor_type", "") == "voice_segmentation"
        and "voice_id" in (n.extracted_content or {})
    ]
    assert len(voice_segments) == 1
    vs = voice_segments[0]
    payload = vs.extracted_content
    assert payload["start_sec"] == pytest.approx(0.0, abs=0.15)
    assert payload["end_sec"] == pytest.approx(1.0, abs=0.15)
    assert payload["voice_id"].startswith("voice_")
    assert payload["source_video_id"] == "vid_test"
    assert payload["source_clip_id"] == 0


def test_voice_segmentation_yaml_stage_wired_after_extract_audio():
    """voice_segmentation must depend on extract_audio so AUDIO
    RawNodes with source_path exist before VAD runs."""
    import yaml
    spec = yaml.safe_load(
        Path("config/m3/pipelines/video_episodic_faces.yaml").read_text()
    )
    stage_ids = [s["id"] for s in spec["stages"]]
    assert "voice_segmentation" in stage_ids
    vs = next(s for s in spec["stages"] if s["id"] == "voice_segmentation")
    assert vs["type"] == "m3.voice_segmentation"
    assert "extract_audio" in vs["needs"]
    assert vs["enabled"] is False
    assert "voice_segments" in spec["artifacts"]