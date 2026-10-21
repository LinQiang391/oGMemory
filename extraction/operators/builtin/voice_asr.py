"""Voice ASR (automatic speech recognition) operator.

Transcribes a single ``AUDIO`` ``RawNode`` (16 kHz mono 16-bit-PCM
WAV — the format :class:`AudioExtractor` emits) into a structured
``ExtractedNode`` carrying one record per segment::

    {
        "language": "en",
        "duration_sec": 28.7,
        "model": "large-v3-turbo",
        "segments": [
            {
                "start_sec": 0.32, "end_sec": 4.18,
                "duration_sec": 3.86,
                "text": "Hello and welcome back to the show.",
                "no_speech_prob": 0.012,
                "asr_id": "asr_<sha1[:16]>",
                "words": [
                    {"word": "Hello", "start_sec": 0.32, "end_sec": 0.71, "probability": 0.98},
                    ...
                ],
            },
            ...
        ],
    }

Each ``segments[i]`` becomes its own ``ExtractedNode`` child via
``VOICE_ASR_MATERIALIZATION`` so downstream consumers (episodic
memory generator, semantic memory consolidator, future
speaker_diarization alignment) can read per-segment text + words
without re-reading the audio file.

Backends (selected by ``backend=``):

* ``faster_whisper`` — default; CTranslate2 Whisper. Lazy-imports
  ``faster_whisper`` so the package can be installed without the
  heavy ML deps for users who only run the HTTP or mock backend.
* ``openai_http`` — POSTs to an OpenAI-compatible
  ``/audio/transcriptions`` endpoint. Use when ASR runs on a
  separate NPU / GPU host.
* ``mock`` — deterministic stub for unit tests.

Whisper already does its own VAD (``vad_filter=True`` by default),
so this operator does NOT depend on the existing
``voice_segmentation`` RMS-VAD stage. Both can coexist — the RMS
VAD is cheap and useful as a "did this clip have speech?" filter,
the ASR result is what feeds the LLM.

Input contract
--------------
Same as :class:`VoiceSegmentationExtractor`: an ``AUDIO`` RawNode
whose ``metadata.source_path`` points at the WAV file. Bytes
travel on disk; the graph only carries the path.
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
from extraction.operators.builtin.asr_backends import (
    ASRBackend,
    ASRBackendError,
    build_asr_backend,
    make_asr_id,
)
from extraction.operators.materialize import VOICE_ASR_MATERIALIZATION

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class VoiceASRExtractor:
    """Transcribe an AUDIO track via Whisper (or compatible backend)."""

    def __init__(
        self,
        *,
        backend: str = "faster_whisper",
        model: str | None = None,
        language: str | None = "en",
        beam_size: int = 5,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        timeout_sec: float = 300.0,
    ) -> None:
        self._backend_name = backend
        self._language = language
        self._beam_size = int(beam_size)
        self._word_timestamps = bool(word_timestamps)
        self._vad_filter = bool(vad_filter)
        self._backend: ASRBackend = build_asr_backend(
            backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            device=device,
            compute_type=compute_type,
            timeout_sec=timeout_sec,
        )

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.AUDIO]

    @property
    def extractor_type(self) -> str:
        return "voice_asr"

    @property
    def schema_name(self) -> str:
        return "voice_asr_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Automatic speech recognition via Whisper. Each utterance "
            "becomes an ExtractedNode child carrying text + word-level "
            "timestamps; the parent carries the full transcript and "
            "language tag. Downstream consumers (episodic memory "
            "generator, semantic consolidator) read these children to "
            "ground cross-modal reasoning in actual speech."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "backend": {
                    "enum": ["faster_whisper", "openai_http", "mock"],
                    "description": "ASR backend implementation.",
                },
                "model": {
                    "type": "string",
                    "description": "Model name (faster-whisper) or "
                    "remote model id (openai_http).",
                },
                "language": {
                    "type": ["string", "null"],
                    "description": "BCP-47 language code or null for auto-detect.",
                },
                "beam_size": {"type": "integer", "minimum": 1, "maximum": 20},
                "word_timestamps": {"type": "boolean"},
                "vad_filter": {"type": "boolean"},
            },
        }
        info.materialization = VOICE_ASR_MATERIALIZATION
        return info

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        path = (raw_node.metadata or {}).get("source_path")
        if not path or not Path(path).exists():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": "audio source_path missing",
                    "segments": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=f"audio source_path missing or unreadable: {path!r}",
            )

        try:
            result = self._backend.transcribe(
                path,
                language=self._language,
                beam_size=self._beam_size,
                word_timestamps=self._word_timestamps,
                vad_filter=self._vad_filter,
            )
        except ASRBackendError as exc:
            logger.warning(
                "voice_asr backend %s failed for %s: %s",
                exc.backend, raw_node.node_id, exc,
            )
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": str(exc),
                    "backend": exc.backend,
                    "segments": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        # Stamp each segment with an asr_id so materialisation can
        # produce a deterministic child node id. The id is derived
        # from (audio source path, start, end) so re-runs on the same
        # clip are idempotent.
        segments_out = []
        for seg in result.segments:
            asr_id = make_asr_id(path, seg.start_sec, seg.end_sec)
            segments_out.append({
                "start_sec": round(seg.start_sec, 3),
                "end_sec": round(seg.end_sec, 3),
                "duration_sec": round(seg.end_sec - seg.start_sec, 3),
                "text": seg.text,
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "asr_id": asr_id,
                "words": [
                    {
                        "word": w.word,
                        "start_sec": round(w.start_sec, 3),
                        "end_sec": round(w.end_sec, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in seg.words
                ],
            })

        # Confidence heuristic: 1.0 - mean(no_speech_prob). Noisy
        # audio scores lower; silent clips get confidence=0 and an
        # empty segments list.
        if segments_out:
            mean_no_speech = sum(
                s["no_speech_prob"] for s in segments_out
            ) / len(segments_out)
            confidence = max(0.0, min(1.0, 1.0 - mean_no_speech))
        else:
            confidence = 0.0

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "language": result.language,
                "duration_sec": round(result.duration_sec, 3),
                "model": result.model,
                "backend": self._backend.name,
                "segments": segments_out,
            },
            confidence=confidence,
            schema_used=self.schema_name,
        )