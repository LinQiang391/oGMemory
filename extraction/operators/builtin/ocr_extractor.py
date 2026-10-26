"""OCR (text recognition) operator.

Extracts text from an image via a ``CVBackend``. The backend's
``detect_objects`` is repurposed to return word/line detections (each
detection's metadata contains the recognised text); the operator then
reconstructs the document text by sorting detections top-to-bottom,
left-to-right.

If a backend doesn't support text recognition, the operator returns
an empty text + an error message in ``extracted_content``.
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


class OCRExtractor:
    """Recognise text in an image."""

    def __init__(
        self,
        backend: CVBackend | None = None,
        min_confidence: float = 0.4,
        max_lines: int = 200,
    ) -> None:
        self._backend = backend or MockBackend()
        self._min_confidence = float(min_confidence)
        self._max_lines = int(max_lines)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.IMAGE]

    @property
    def extractor_type(self) -> str:
        return "ocr"

    @property
    def schema_name(self) -> str:
        return "ocr_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Recognise text in an image. Each detection's metadata['text'] "
            "is treated as a word/line; results are sorted top-to-bottom, "
            "left-to-right to reconstruct the document."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
        }
        return info

    def extract(self, raw_node: "RawNode", ctx: "RequestContext") -> ExtractionResult:
        try:
            detections = self._backend.detect_objects(
                raw_node.content,
                threshold=self._min_confidence,
                max_items=self._max_lines,
            )
        except CVBackendError as exc:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"error": str(exc), "text": ""},
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        # Filter to detections that have text in metadata
        text_items: list[tuple[float, float, str, float]] = []
        for d in detections:
            text = d.metadata.get("text")
            if not text or not isinstance(text, str):
                continue
            x1, y1, x2, y2 = d.bbox
            text_items.append((y1, x1, text, d.confidence))

        # Sort top-to-bottom, then left-to-right, with a small y-tolerance
        text_items.sort(key=lambda t: (round(t[0] / 0.02) * 0.02, t[1]))

        text = "\n".join(item[2] for item in text_items)
        avg_conf = (
            sum(item[3] for item in text_items) / len(text_items)
            if text_items else 0.0
        )

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "text": text,
                "line_count": len(text_items),
                "lines": [
                    {"text": item[2], "confidence": item[3], "bbox": None}
                    for item in text_items
                ],
            },
            confidence=avg_conf,
            schema_used=self.schema_name,
        )
