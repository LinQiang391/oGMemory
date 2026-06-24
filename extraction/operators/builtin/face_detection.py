"""Face detection operator.

Uses a ``CVBackend`` to detect faces in an image, returning structured
``ExtractedNode`` content that records the number of faces and each
face's bounding box (normalised to image space).

Configuration (kwargs)::

    backend: CVBackend instance (default: MockBackend())
    min_confidence: float (default 0.5) — discard detections below this
    max_faces: int (default 20) — cap result list
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.backends import MockBackend
from extraction.operators.backends.base import CVBackend, CVBackendError
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)
from extraction.operators.materialize import FACE_DETECTION_MATERIALIZATION

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


class FaceDetectionExtractor:
    """Detect human faces in an image."""

    def __init__(
        self,
        backend: CVBackend | None = None,
        min_confidence: float = 0.5,
        max_faces: int = 20,
    ) -> None:
        self._backend = backend or MockBackend()
        self._min_confidence = float(min_confidence)
        self._max_faces = int(max_faces)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.IMAGE]

    @property
    def extractor_type(self) -> str:
        return "face_detection"

    @property
    def schema_name(self) -> str:
        return "face_detection_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = "Detect human faces in an image and return bounding boxes."
        info.config_schema = {
            "type": "object",
            "properties": {
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_faces": {"type": "integer", "minimum": 1, "maximum": 200},
            },
        }
        # Each detected face becomes its own ExtractedNode in the graph
        # (kind="extracted" — face metadata is structured output, not
        # raw pixels). OperatorService.persist_extraction handles the
        # node + auto-wired EXTRACTED_FROM edge; the driver just needs
        # to call it without bytes_writer since embeddings live directly
        # on the ExtractedNode.extracted_content.
        info.materialization = FACE_DETECTION_MATERIALIZATION
        return info

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        try:
            detections = self._backend.detect_faces(
                raw_node.content,
                threshold=self._min_confidence,
            )
        except CVBackendError as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc), "faces": []},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        kept = [d for d in detections if d.confidence >= self._min_confidence][:self._max_faces]
        if not kept:
            confidence = 0.0
        else:
            confidence = sum(d.confidence for d in kept) / len(kept)

        content = {
            "face_count": len(kept),
            "faces": [
                {
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    **({"metadata": d.metadata} if d.metadata else {}),
                }
                for d in kept
            ],
        }
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=content,
            confidence=confidence,
            schema_used=self.schema_name,
        )
