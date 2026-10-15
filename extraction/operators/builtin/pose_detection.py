"""Pose / keypoint detection operator.

Detects people in an image and (when the backend supports it) extracts
body keypoints. Currently uses the ``CVBackend``'s ``detect_pose``
method, which returns Detection objects with the person's bbox in
``bbox`` and a ``metadata["keypoints"]`` list.

The schema preserves the original keypoint structure so consumers can
do downstream analysis (action classification, body measurement, etc.).
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

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


class PoseDetectionExtractor:
    """Detect people + body keypoints in an image."""

    def __init__(
        self,
        backend: CVBackend | None = None,
        min_confidence: float = 0.5,
        max_persons: int = 10,
    ) -> None:
        self._backend = backend or MockBackend()
        self._min_confidence = float(min_confidence)
        self._max_persons = int(max_persons)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.IMAGE]

    @property
    def extractor_type(self) -> str:
        return "pose_detection"

    @property
    def schema_name(self) -> str:
        return "pose_detection_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Detect people in an image and extract body keypoints "
            "(nose, shoulders, elbows, wrists, hips, knees, ankles)."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_persons": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        }
        return info

    def extract(self, raw_node: "RawNode", ctx: "RequestContext") -> ExtractionResult:
        try:
            detections = self._backend.detect_pose(
                raw_node.content,
                threshold=self._min_confidence,
            )
        except CVBackendError as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc), "persons": []},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        kept = [d for d in detections if d.confidence >= self._min_confidence][:self._max_persons]

        if not kept:
            confidence = 0.0
        else:
            confidence = sum(d.confidence for d in kept) / len(kept)

        content = {
            "person_count": len(kept),
            "persons": [
                {
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    "keypoints": d.metadata.get("keypoints", []),
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
