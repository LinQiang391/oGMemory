"""Voice activity detection (VAD) operator.

Splits a single audio track into voice segments (continuous speech
runs separated by silence). Pure RMS-energy VAD — no external VAD
model dependency, so the operator runs anywhere numpy runs. For the
M3-Bench MVP this is plenty: it finds the obvious speech runs, the
downstream ``voice_clustering`` operator is the one that cares about
per-voice identity.

Input contract
--------------
The operator consumes an ``AUDIO`` ``RawNode`` whose ``metadata``
contains ``source_path`` pointing at a 16 kHz mono 16-bit-PCM WAV
file (the format the ``audio_extractor`` operator emits). Reading
from ``source_path`` (not ``content``) matches the convention
established by ``audio_extractor``: audio bytes are large and live
on disk, the graph only carries the path.

Output contract
----------------
``extracted_content``::

    {
      "format": "wav",
      "sample_rate": 16000,
      "duration_sec": 30.0,
      "segments": [
        {
          "start_sec": 0.0,
          "end_sec": 1.5,
          "duration_sec": 1.5,
          "confidence": 0.95,
          "voice_id": "voice_abc123",
          "rms_dbfs": -18.4,
          "frame_count": 50,
        },
        ...
      ]
    }

Each segment becomes its own ``ExtractedNode`` via
``VOICE_SEGMENTATION_MATERIALIZATION``; the parent carries the
aggregate ``{"format", "sample_rate", "duration_sec", "segments_count"}``
plus a ``child_node_ids`` + ``child_count`` summary.

Voice ID derivation
-------------------
A deterministic per-segment id (``sha1(operator + audio_source_path +
start_sec)``) so re-runs are idempotent and downstream clustering can
key off it without recomputing.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)
from extraction.operators.materialize import VOICE_SEGMENTATION_MATERIALIZATION

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — also exercised by the unit tests directly.
# ---------------------------------------------------------------------------

def _pcm16le_to_float_mono(path: str) -> tuple[list[float], int]:
    """Decode a 16-bit-PCM mono WAV file to a Python float list.

    Returns ``(samples, sample_rate)``. Mono only — non-mono WAVs are
    rejected (the audio_extractor emits mono by default so this is
    the common case). Stereo inputs raise so callers can fall back
    to a downmix operator rather than silently producing wrong
    segments.
    """
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if n_channels != 1:
        raise ValueError(
            f"voice_segmentation expects mono audio, got {n_channels} channels"
        )
    if sampwidth != 2:
        raise ValueError(
            f"voice_segmentation expects 16-bit PCM, got sampwidth={sampwidth}"
        )
    count = len(raw) // 2
    samples = struct.unpack(f"<{count}h", raw)
    # int16 → float in [-1, 1]
    return [s / 32768.0 for s in samples], sr


def _frame_rms_dbfs(samples: list[float]) -> float:
    """Root-mean-square of a sample window, returned in dBFS.

    0 dBFS = full-scale sine, -inf dBFS = silence. Floors at
    -100 dBFS to avoid log(0).
    """
    if not samples:
        return -100.0
    sq = sum(s * s for s in samples)
    rms = math.sqrt(sq / len(samples))
    if rms <= 0.0:
        return -100.0
    return 20.0 * math.log10(rms)


def _segment_voice_id(
    operator_id: str, audio_source_path: str, start_sec: float,
) -> str:
    seed = f"{operator_id}|{audio_source_path}|{start_sec:.4f}".encode()
    return f"voice_{hashlib.sha1(seed).hexdigest()[:16]}"


def detect_voice_segments(
    samples: list[float],
    sample_rate: int,
    *,
    energy_threshold_dbfs: float = -40.0,
    frame_sec: float = 0.03,
    min_segment_sec: float = 0.25,
    max_segment_sec: float = 30.0,
    pad_sec: float = 0.1,
) -> list[dict]:
    """Pure VAD: split ``samples`` into segments via RMS-energy gating.

    Algorithm (pure, no I/O):

    1. Slice into ``frame_sec`` windows (default 30 ms).
    2. Compute RMS dBFS per frame; mark frame as speech if
       ``>= energy_threshold_dbfs`` (default -40 dBFS).
    3. Group consecutive speech frames into raw segments.
    4. Drop segments shorter than ``min_segment_sec`` (noise bursts).
    5. Cap segments at ``max_segment_sec`` (split overly long ones
       into fixed-length chunks).
    6. Pad each segment by ``pad_sec`` on both sides, clamped to
       the audio duration.
    7. Emit a ``voice_id`` per segment derived from operator id +
       source path + start time.

    Returns a list of segment dicts ready for the operator's
    ``extracted_content["segments"]`` slot. Stable order (ascending
    start_sec). Empty list if no speech detected.
    """
    if not samples or sample_rate <= 0:
        return []
    frame_n = max(1, int(round(frame_sec * sample_rate)))
    n_frames = len(samples) // frame_n
    if n_frames == 0:
        return []
    # Per-frame speech flags.
    is_speech = [
        _frame_rms_dbfs(samples[i * frame_n:(i + 1) * frame_n])
        >= energy_threshold_dbfs
        for i in range(n_frames)
    ]
    # Group consecutive speech frames.
    raw_segs: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if not is_speech[i]:
            i += 1
            continue
        j = i
        while j < n_frames and is_speech[j]:
            j += 1
        raw_segs.append((i, j))  # [start_frame, end_frame)
        i = j
    # Filter / cap / pad / number.
    out: list[dict] = []
    total_sec = len(samples) / float(sample_rate)
    for start_f, end_f in raw_segs:
        start_sec = start_f * frame_n / float(sample_rate)
        end_sec = end_f * frame_n / float(sample_rate)
        duration = end_sec - start_sec
        if duration < min_segment_sec:
            continue
        # Cap by splitting into chunks.
        chunks: list[tuple[float, float]] = []
        cur_start = start_sec
        while cur_start + max_segment_sec < end_sec - 1e-6:
            chunks.append((cur_start, cur_start + max_segment_sec))
            cur_start += max_segment_sec
        chunks.append((cur_start, end_sec))
        for cs, ce in chunks:
            pad_lo = max(0.0, cs - pad_sec)
            pad_hi = min(total_sec, ce + pad_sec)
            if pad_hi - pad_lo < min_segment_sec * 0.5:
                continue  # too tiny after padding, skip
            seg = {
                "start_sec": round(pad_lo, 3),
                "end_sec": round(pad_hi, 3),
                "duration_sec": round(pad_hi - pad_lo, 3),
                "voice_id": _segment_voice_id(
                    "voice_segmentation_v1", "",  # source_path filled
                    # in extract() below where we know it.
                    pad_lo,
                ),
                "rms_dbfs": round(
                    _frame_rms_dbfs(
                        samples[int(pad_lo * sample_rate):int(pad_hi * sample_rate)]
                    ), 1,
                ),
                "frame_count": int(round((pad_hi - pad_lo) / frame_sec)),
            }
            out.append(seg)
    return out


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class VoiceSegmentationExtractor:
    """Split an AUDIO track into voice segments via RMS-energy VAD."""

    def __init__(
        self,
        *,
        energy_threshold_dbfs: float = -40.0,
        frame_sec: float = 0.03,
        min_segment_sec: float = 0.25,
        max_segment_sec: float = 30.0,
        pad_sec: float = 0.1,
    ) -> None:
        self._energy_threshold_dbfs = float(energy_threshold_dbfs)
        self._frame_sec = float(frame_sec)
        self._min_segment_sec = float(min_segment_sec)
        self._max_segment_sec = float(max_segment_sec)
        self._pad_sec = float(pad_sec)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.AUDIO]

    @property
    def extractor_type(self) -> str:
        return "voice_segmentation"

    @property
    def schema_name(self) -> str:
        return "voice_segmentation_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Split an audio track into voice segments via RMS-energy "
            "voice activity detection. Each segment becomes an "
            "ExtractedNode child with start_sec / end_sec / voice_id; "
            "segments in the same clip share co_occurs_in edges."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "energy_threshold_dbfs": {
                    "type": "number",
                    "minimum": -80.0,
                    "maximum": 0.0,
                    "description": (
                        "Frames with RMS dBFS above this threshold "
                        "are marked as speech. Default -40 dBFS."
                    ),
                },
                "frame_sec": {
                    "type": "number", "minimum": 0.005, "maximum": 0.1,
                    "description": "Frame size for RMS computation.",
                },
                "min_segment_sec": {
                    "type": "number", "minimum": 0.05, "maximum": 5.0,
                    "description": "Drop segments shorter than this.",
                },
                "max_segment_sec": {
                    "type": "number", "minimum": 1.0, "maximum": 120.0,
                    "description": "Cap segment length; long runs split.",
                },
                "pad_sec": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                    "description": "Padding on each side of a segment.",
                },
            },
        }
        info.materialization = VOICE_SEGMENTATION_MATERIALIZATION
        return info

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        path = (raw_node.metadata or {}).get("source_path")
        if not path or not Path(path).exists():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": "audio source_path missing",
                                   "segments": []},
                confidence=0.0,
                schema_used=self.schema_name,
                error=f"audio source_path missing or unreadable: {path!r}",
            )
        try:
            samples, sample_rate = _pcm16le_to_float_mono(path)
        except Exception as exc:  # noqa: BLE001
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc), "segments": []},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )
        total_sec = len(samples) / float(sample_rate) if sample_rate else 0.0

        segments = detect_voice_segments(
            samples, sample_rate,
            energy_threshold_dbfs=self._energy_threshold_dbfs,
            frame_sec=self._frame_sec,
            min_segment_sec=self._min_segment_sec,
            max_segment_sec=self._max_segment_sec,
            pad_sec=self._pad_sec,
        )
        # Stamp the real source path into each voice_id so re-runs
        # on a different audio file can't collide. The pure helper
        # passes an empty path; this is the only place we know the
        # full path.
        for seg in segments:
            seg["voice_id"] = _segment_voice_id(
                self.schema_name, path, seg["start_sec"],
            )

        if not segments:
            confidence = 0.0
        else:
            # Average speech coverage (seconds of speech / total
            # audio seconds), capped at 1.0. Confident when speech
            # dominates the clip.
            speech_sec = sum(s["duration_sec"] for s in segments)
            confidence = min(1.0, speech_sec / max(1e-6, total_sec))

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "format": "wav",
                "sample_rate": sample_rate,
                "duration_sec": round(total_sec, 3),
                "segments": segments,
            },
            confidence=confidence,
            schema_used=self.schema_name,
        )