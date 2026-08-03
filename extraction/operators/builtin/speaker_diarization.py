"""Speaker diarization operator.

Labels each word in the ASR output with a ``speaker_id``
(``SPEAKER_00``, ``SPEAKER_01``, ...). Pyannote.audio 3.1 produces
contiguous, non-overlapping speaker turns; the operator aligns
those turns to the ASR words so downstream consumers can answer
"what did speaker_1 say in CLIP_3" instead of just "0:05-0:10 was
speaker_1".

The operator is pure — it does NOT fetch its own ASR children.
Drivers (``m3/speaker_diarization.py``) walk the graph, look up
the matching ``voice_asr`` ExtractedNode for each AUDIO RawNode,
and pass ``asr_segments`` explicitly. The operator then:

1. Runs diarization on the WAV file (time-based turns).
2. Stamps ``speaker_id`` onto each ASR word via timestamp overlap.
3. Picks a per-segment majority speaker and stamps it on each
   ``voice_asr`` child ExtractedNode (``speaker_id`` field added
   to ``extracted_content``).

Output contract
---------------
Parent ``ExtractedNode.extracted_content``::

    {
        "model": "pyannote/speaker-diarization-3.1",
        "num_speakers": 2,
        "turns": [
            {"start_sec": 0.32, "end_sec": 4.18, "speaker_id": "SPEAKER_00",
             "asr_segment_ids": ["asr_abc123"]},
            ...
        ],
    }

Each ``turns[i]`` becomes its own ``ExtractedNode`` child via
``SPEAKER_DIARIZATION_MATERIALIZATION``. Each ``voice_asr``
ExtractedNode child of the same audio also gets its
``speaker_id`` field set in-place (via a follow-up graph write
performed by the driver — see ``m3/speaker_diarization.py``).

Backends (selected by ``backend=``):

* ``pyannote`` — default; pyannote.audio 3.1. Requires HF token.
* ``mock`` — deterministic stub for tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)
from extraction.operators.builtin.diarization_backends import (
    DiarizationBackend,
    DiarizationBackendError,
    build_diarization_backend,
)
from extraction.operators.materialize import SPEAKER_DIARIZATION_MATERIALIZATION

if TYPE_CHECKING:
    from core.graph_models import ExtractedNode, RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (also exercised by unit tests)
# ---------------------------------------------------------------------------

def _stamp_speaker_on_asr_words(
    words: list[dict],
    turns: list[dict],
) -> list[dict]:
    """Mutate-copy each word dict in-place to add ``speaker_id``.

    Each word gets the speaker of the turn whose
    ``[start_sec, end_sec]`` overlaps the word's
    ``[start_sec, end_sec]``. Pyannote turns are contiguous and
    non-overlapping by construction, so an unambiguous lookup is
    always possible; if a word somehow falls outside every turn
    (shouldn't happen), it stays speaker-less.
    """
    out: list[dict] = []
    for w in words:
        ws = float(w.get("start_sec", 0.0))
        we = float(w.get("end_sec", 0.0))
        speaker_id: str | None = None
        for t in turns:
            ts = float(t.get("start_sec", 0.0))
            te = float(t.get("end_sec", 0.0))
            # Half-open overlap: word overlaps turn iff ws < te AND we > ts.
            if ws < te and we > ts:
                speaker_id = str(t.get("speaker_id", ""))
                break
        new_w = dict(w)
        new_w["speaker_id"] = speaker_id
        out.append(new_w)
    return out


def _majority_speaker_id(
    segment_words: list[dict],
    turns: list[dict],
    *,
    segment_start_sec: float | None = None,
    segment_end_sec: float | None = None,
) -> str | None:
    """Pick the majority speaker for a single ASR segment.

    For each word in the segment, find the turn whose
    ``[start_sec, end_sec]`` overlaps. The speaker with the most
    word-matches wins. When the segment has no words (some ASR
    backends don't emit per-word timestamps), fall back to
    segment-level overlap: the turn with the largest overlap with
    ``[segment_start_sec, segment_end_sec]`` wins. Returns None
    when no word/segment matched any turn.
    """
    counts: dict[str, int] = {}
    for w in segment_words:
        ws = float(w.get("start_sec", 0.0))
        we = float(w.get("end_sec", 0.0))
        for t in turns:
            ts = float(t.get("start_sec", 0.0))
            te = float(t.get("end_sec", 0.0))
            if ws < te and we > ts:
                sid = str(t.get("speaker_id", ""))
                counts[sid] = counts.get(sid, 0) + 1
                break
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    # Fallback: segment-level overlap when no words are available.
    if segment_start_sec is None or segment_end_sec is None:
        return None
    ss = float(segment_start_sec)
    se = float(segment_end_sec)
    best_sid: str | None = None
    best_overlap = 0.0
    for t in turns:
        ts = float(t.get("start_sec", 0.0))
        te = float(t.get("end_sec", 0.0))
        overlap = min(se, te) - max(ss, ts)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sid = str(t.get("speaker_id", ""))
    return best_sid


def _turns_with_asr_segment_ids(
    turns: list[dict],
    asr_segments: list[dict],
) -> list[dict]:
    """For each turn, list the ASR segment ids that overlap it."""
    out: list[dict] = []
    for t in turns:
        ts = float(t.get("start_sec", 0.0))
        te = float(t.get("end_sec", 0.0))
        seg_ids: list[str] = []
        for s in asr_segments:
            ss = float(s.get("start_sec", 0.0))
            se = float(s.get("end_sec", 0.0))
            if ss < te and se > ts:
                seg_id = s.get("asr_id", "")
                if seg_id:
                    seg_ids.append(str(seg_id))
        out.append({
            "start_sec": float(t.get("start_sec", 0.0)),
            "end_sec": float(t.get("end_sec", 0.0)),
            "speaker_id": str(t.get("speaker_id", "")),
            "asr_segment_ids": seg_ids,
        })
    return out


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class SpeakerDiarizationExtractor:
    """Diarize an AUDIO track and align speakers to ASR words."""

    def __init__(
        self,
        *,
        backend: str = "pyannote",
        model: str | None = None,
        hf_token: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: str = "cpu",
    ) -> None:
        self._backend_name = backend
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._backend: DiarizationBackend = build_diarization_backend(
            backend,
            model=model,
            hf_token=hf_token,
            device=device,
        )

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.AUDIO]

    @property
    def extractor_type(self) -> str:
        return "speaker_diarization"

    @property
    def schema_name(self) -> str:
        return "speaker_diarization_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Speaker diarization (who spoke when) aligned to ASR words. "
            "Each turn becomes an ExtractedNode child carrying speaker_id "
            "and the list of ASR segment ids that overlap it. Backed by "
            "pyannote.audio 3.1 with a Hugging Face token."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "backend": {
                    "enum": ["pyannote", "mock"],
                },
                "model": {"type": "string"},
                "min_speakers": {"type": ["integer", "null"], "minimum": 1},
                "max_speakers": {"type": ["integer", "null"], "minimum": 1},
            },
        }
        info.materialization = SPEAKER_DIARIZATION_MATERIALIZATION
        return info

    def extract(
        self,
        raw_node: RawNode,
        ctx: RequestContext,
        *,
        asr_extracted_node: ExtractedNode | None = None,
    ) -> ExtractionResult:
        """Run diarization + alignment.

        Args:
            raw_node: The AUDIO RawNode whose ``metadata.source_path``
                points at the WAV file.
            asr_extracted_node: Optional pre-fetched ``voice_asr``
                ExtractedNode for this audio. When present, the
                operator stamps ``speaker_id`` onto each ASR word
                (mutating its ``extracted_content`` in place) and
                picks a per-segment majority speaker. When absent,
                the operator emits time-based turns only.
        """
        path = (raw_node.metadata or {}).get("source_path")
        if not path or not Path(path).exists():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": "audio source_path missing",
                    "turns": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=f"audio source_path missing or unreadable: {path!r}",
            )

        try:
            diarization = self._backend.diarize(
                path,
                min_speakers=self._min_speakers,
                max_speakers=self._max_speakers,
            )
        except DiarizationBackendError as exc:
            logger.warning(
                "speaker_diarization backend %s failed for %s: %s",
                exc.backend, raw_node.node_id, exc,
            )
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": str(exc),
                    "backend": exc.backend,
                    "turns": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        turns_dicts: list[dict] = [
            {
                "start_sec": round(t.start_sec, 3),
                "end_sec": round(t.end_sec, 3),
                "speaker_id": t.speaker_id,
            }
            for t in diarization.turns
        ]

        # Align to ASR words + segments when the driver passed the
        # voice_asr ExtractedNode in. We mutate a *copy* of its
        # extracted_content so we don't surprise the caller (the
        # driver persists the copy via a separate graph write).
        aligned_asr_extracted_content: dict | None = None
        per_segment_speaker: dict[str, str | None] = {}
        if asr_extracted_node is not None:
            asr_ec = dict(asr_extracted_node.extracted_content or {})
            asr_segments = list(asr_ec.get("segments") or [])
            turns_with_segs = _turns_with_asr_segment_ids(
                turns_dicts, asr_segments,
            )
            new_segments: list[dict] = []
            for seg in asr_segments:
                seg_words = list(seg.get("words") or [])
                stamped_words = _stamp_speaker_on_asr_words(seg_words, turns_dicts)
                seg_speaker = _majority_speaker_id(
                    seg_words, turns_dicts,
                    segment_start_sec=seg.get("start_sec"),
                    segment_end_sec=seg.get("end_sec"),
                )
                new_seg = dict(seg)
                new_seg["words"] = stamped_words
                new_seg["speaker_id"] = seg_speaker
                new_segments.append(new_seg)
                seg_id = str(seg.get("asr_id", ""))
                if seg_id:
                    per_segment_speaker[seg_id] = seg_speaker
            aligned_asr_extracted_content = asr_ec
            aligned_asr_extracted_content["segments"] = new_segments
            # The per-turn asr_segment_ids were computed above; use
            # them in the parent output too.
            turns_dicts = turns_with_segs

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "model": diarization.model,
                "backend": self._backend.name,
                "num_speakers": diarization.num_speakers,
                "turns": turns_dicts,
                # Carried for the driver to persist back onto the
                # voice_asr ExtractedNode. Empty when no ASR was
                # passed in.
                "per_segment_speaker": per_segment_speaker,
                "aligned_asr_extracted_content": aligned_asr_extracted_content,
            },
            # Confidence = 1.0 - (silence_in_turns / total_dur). For
            # now just report 1.0 when any turns were detected,
            # 0.0 otherwise.
            confidence=1.0 if diarization.turns else 0.0,
            schema_used=self.schema_name,
        )