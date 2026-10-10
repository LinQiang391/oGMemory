"""Object detection operator.

Generic multi-class object detection (COCO-style labels) using a
``CVBackend``. Returns a list of detected objects with bounding boxes
and per-class confidences.

Configuration (kwargs)::

    backend: CVBackend instance (default: MockBackend())
    confidence_threshold: float (default 0.5)
    max_objects: int (default 50)
    allowed_labels: optional iterable — keep only these labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from core.graph_enums import RawDataType
from extraction.operators.backends import MockBackend
from extraction.operators.backends.base import CVBackend, CVBackendError
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


class ObjectDetectionExtractor:
    """Detect generic objects in an image (COCO-style 80 classes)."""

    def __init__(
        self,
        backend: CVBackend | None = None,
        confidence_threshold: float = 0.5,
        max_objects: int = 50,
        allowed_labels: Iterable[str] | None = None,
    ) -> None:
        self._backend = backend or MockBackend()
        self._threshold = float(confidence_threshold)
        self._max_objects = int(max_objects)
        self._allowed_labels = (
            frozenset(allowed_labels) if allowed_labels is not None else None
        )

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.IMAGE]

    @property
    def extractor_type(self) -> str:
        return "object_detection"

    @property
    def schema_name(self) -> str:
        return "object_detection_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Detect objects in an image and return bounding boxes with "
            "per-class confidence. Backend-dependent label set (COCO by default)."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_objects": {"type": "integer", "minimum": 1, "maximum": 500},
                "allowed_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        return info

    def extract(self, raw_node: "RawNode", ctx: "RequestContext") -> ExtractionResult:
        try:
            detections = self._backend.detect_objects(
                raw_node.content,
                threshold=self._threshold,
                max_items=self._max_objects,
            )
        except CVBackendError as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc), "objects": []},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        if self._allowed_labels is not None:
            detections = [d for d in detections if d.label in self._allowed_labels]
        detections = detections[: self._max_objects]

        if not detections:
            confidence = 0.0
            label_summary: dict[str, int] = {}
        else:
            confidence = sum(d.confidence for d in detections) / len(detections)
            label_summary = {}
            for d in detections:
                label_summary[d.label] = label_summary.get(d.label, 0) + 1

        content = {
            "object_count": len(detections),
            "labels": label_summary,
            "objects": [
                {
                    "label": d.label,
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    **({"metadata": d.metadata} if d.metadata else {}),
                }
                for d in detections
            ],
        }
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=content,
            confidence=confidence,
            schema_used=self.schema_name,
        )
