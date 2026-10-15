"""Speaker embedding operator.

Produces a 192-dim speaker embedding per ``speaker_id`` in a clip,
backed by ECAPA-TDNN (``speechbrain/spkrec-ecapa-voxceleb``).
Replaces the 8-dim spectral fingerprint from
:class:`VoiceSegmentationExtractor` for cross-clip identity
clustering — the 192-dim vectors are actually discriminative
between distinct speakers, where the 8-dim fingerprint was just
"acoustic similarity".

The operator is pure — drivers (``m3/speaker_embedding.py``) walk
the graph, look up the matching ``speaker_diarization`` ExtractedNode
for each AUDIO RawNode, and pass ``turns`` explicitly. For each
unique ``speaker_id`` across the turns, the operator:

1. Concatenates the per-turn audio slices into one continuous
   signal.
2. Runs ECAPA-TDNN on the concatenated signal.
3. Stores the result as one child ``ExtractedNode`` per
   ``speaker_id`` with a 192-dim ``speaker_embedding``.

Output contract
---------------
Parent ``ExtractedNode.extracted_content``::

    {
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "speaker_embedding": [0.012, -0.034, ...],   # 192 floats
                "num_turns": 4,
                "total_duration_sec": 18.3,
                "model": "speechbrain/spkrec-ecapa-voxceleb",
            },
            ...
        ],
    }

Each ``speakers[i]`` becomes its own ``ExtractedNode`` child via
``SPEAKER_EMBEDDING_MATERIALIZATION``. The child node's
``extracted_content`` carries the 192-dim vector directly — no
separate embedding column.

Backends (selected by ``backend=``):

* ``speechbrain`` — default; ECAPA-TDNN via speechbrain.
* ``mock`` — deterministic stub for tests.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)
from extraction.operators.builtin.embedding_backends import (
    SpeakerEmbeddingBackend,
    SpeakerEmbeddingBackendError,
    SpeakerEmbeddingTooShortError,
    build_speaker_embedding_backend,
)
from extraction.operators.materialize import SPEAKER_EMBEDDING_MATERIALIZATION

if TYPE_CHECKING:
    from core.graph_models import ExtractedNode, RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _aggregate_turns_by_speaker(
    turns: list[dict],
) -> dict[str, list[dict]]:
    """Group turns by speaker_id, preserving order.

    Pyannote already numbers speakers consistently within a clip,
    so the per-clip grouping is straightforward. Cross-clip identity
    is the downstream ``voice_clustering`` driver's job.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        sid = str(t.get("speaker_id", ""))
        if not sid:
            continue
        out[sid].append({
            "start_sec": float(t.get("start_sec", 0.0)),
            "end_sec": float(t.get("end_sec", 0.0)),
        })
    return out


def _aggregate_total_duration(turns: list[dict]) -> float:
    return sum(
        max(0.0, float(t.get("end_sec", 0.0)) - float(t.get("start_sec", 0.0)))
        for t in turns
    )


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class SpeakerEmbeddingExtractor:
    """Compute a 192-dim embedding per speaker in an AUDIO clip."""

    def __init__(
        self,
        *,
        backend: str = "speechbrain",
        model: str | None = None,
        device: str = "cpu",
        embedding_dim: int = 192,
    ) -> None:
        self._backend_name = backend
        self._embedding_dim = int(embedding_dim)
        self._backend: SpeakerEmbeddingBackend = build_speaker_embedding_backend(
            backend,
            model=model,
            device=device,
            embedding_dim=embedding_dim,
        )

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.AUDIO]

    @property
    def extractor_type(self) -> str:
        return "speaker_embedding"

    @property
    def schema_name(self) -> str:
        return "speaker_embedding_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Compute a 192-dim ECAPA-TDNN embedding per unique speaker_id "
            "in an audio clip. Each speaker becomes one ExtractedNode child "
            "carrying speaker_embedding + provenance. The downstream "
            "voice_clustering driver uses these 192-dim vectors (in "
            "preference to the 8-dim spectral fingerprint from "
            "voice_segmentation) to compute cross-clip identity."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "backend": {"enum": ["speechbrain", "mock"]},
                "model": {"type": "string"},
                "embedding_dim": {"type": "integer", "minimum": 1, "maximum": 2048},
            },
        }
        info.materialization = SPEAKER_EMBEDDING_MATERIALIZATION
        return info

    def extract(
        self,
        raw_node: RawNode,
        ctx: RequestContext,
        *,
        diarization_extracted_node: ExtractedNode | None = None,
    ) -> ExtractionResult:
        """Compute one embedding per speaker_id.

        Args:
            raw_node: The AUDIO RawNode whose ``metadata.source_path``
                points at the WAV file.
            diarization_extracted_node: Optional pre-fetched
                ``speaker_diarization`` ExtractedNode for this audio.
                When present, its ``turns`` field drives which
                ``speaker_id``s get embedded. When absent, the
                operator reports missing turns (no speakers to process).
        """
        path = (raw_node.metadata or {}).get("source_path")
        if not path or not Path(path).exists():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": "audio source_path missing",
                    "speakers": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=f"audio source_path missing or unreadable: {path!r}",
            )

        turns: list[dict] = []
        if diarization_extracted_node is not None:
            dec = diarization_extracted_node.extracted_content or {}
            turns = list(dec.get("turns") or [])
        if not turns:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "model": "",
                    "embedding_dim": self._embedding_dim,
                    "speakers": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error="no diarization turns provided; nothing to embed",
            )

        grouped = _aggregate_turns_by_speaker(turns)
        if not grouped:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "model": "",
                    "embedding_dim": self._embedding_dim,
                    "speakers": [],
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error="turns had no speaker_id; nothing to embed",
            )

        # Compute each speaker's embedding. We feed the whole
        # concatenated range to the backend rather than per-turn
        # averages — ECAPA-TDNN is trained on longer segments and
        # single-turn utterances (often <1s) degrade embedding
        # quality.
        speakers_out: list[dict] = []
        skipped_speakers: list[dict] = []
        per_speaker_error: dict[str, str] = {}
        for sid, sid_turns in grouped.items():
            start_sec = min(t["start_sec"] for t in sid_turns)
            end_sec = max(t["end_sec"] for t in sid_turns)
            try:
                embedding = self._backend.embed(
                    path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    speaker_id=sid,
                )
            except SpeakerEmbeddingTooShortError as exc:
                skipped_speakers.append({
                    "speaker_id": sid,
                    "reason": "audio_too_short",
                    "duration_sec": exc.duration_sec,
                })
                logger.warning("Skipping speaker %s: %s", sid, exc)
                continue
            except SpeakerEmbeddingBackendError as exc:
                logger.warning(
                    "speaker_embedding backend %s failed for %s: %s",
                    exc.backend, sid, exc,
                )
                per_speaker_error[sid] = str(exc)
                continue
            speakers_out.append({
                "speaker_id": sid,
                "speaker_embedding": embedding,
                "embedding_dim": self._embedding_dim,
                "model": self._backend.name,
                "num_turns": len(sid_turns),
                "total_duration_sec": round(_aggregate_total_duration(sid_turns), 3),
            })

        confidence = (
            len(speakers_out) / len(grouped) if grouped else 0.0
        )
        out: dict = {
            "model": self._backend.name,
            "embedding_dim": self._embedding_dim,
            "speakers": speakers_out,
            "skipped_speakers": skipped_speakers,
        }
        if per_speaker_error:
            out["per_speaker_error"] = per_speaker_error
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=out,
            confidence=confidence,
            schema_used=self.schema_name,
        )
