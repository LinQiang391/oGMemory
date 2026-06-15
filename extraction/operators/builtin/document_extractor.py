"""Document structure extraction operator.

Lightweight structure extraction from a DOCUMENT raw node's text
content. Splits the text into sections by detecting headings
(all-caps lines, lines ending with ':', or markdown '#' prefixes) and
records each section's title and body.

This operator is text-only — it does not parse PDFs / DOCX natively.
For those formats, pre-convert the document to text in the ingestion
pipeline, then store the text as a DOCUMENT raw node.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


_HEADING_PATTERNS = [
    re.compile(r"^#\s+(.+)$"),                    # Markdown ATX
    re.compile(r"^#{2,}\s+(.+)$"),                # Markdown ATX (deeper)
    re.compile(r"^(.+):\s*$"),                    # "Title: "
    re.compile(r"^([A-Z][A-Z0-9 _-]{2,})$"),      # ALL CAPS
]


class DocumentExtractor:
    """Extract section structure from a text document."""

    def __init__(
        self,
        min_section_length: int = 20,
        max_sections: int = 100,
    ) -> None:
        self._min_section_length = int(min_section_length)
        self._max_sections = int(max_sections)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.DOCUMENT]

    @property
    def extractor_type(self) -> str:
        return "document_structure"

    @property
    def schema_name(self) -> str:
        return "document_structure_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Split a text document into sections by detecting headings "
            "(markdown #, ALL-CAPS, 'Title: ' patterns). Each section's "
            "title and body are stored in extracted_content."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "min_section_length": {"type": "integer", "minimum": 1, "maximum": 1000},
                "max_sections": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
        }
        return info

    def extract(self, raw_node: "RawNode", ctx: "RequestContext") -> ExtractionResult:
        text = raw_node.content or ""
        if not text.strip():
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"sections": [], "section_count": 0, "char_count": 0},
                confidence=1.0,
                schema_used=self.schema_name,
            )

        sections: list[dict] = []
        current_title = "(preamble)"
        current_body: list[str] = []
        confidence_sum = 0.0
        confidence_count = 0

        for line in text.splitlines():
            heading = self._detect_heading(line)
            if heading:
                # Close current section (if non-empty)
                if current_body or current_title != "(preamble)":
                    body = "\n".join(current_body).strip()
                    if len(body) >= self._min_section_length or current_title != "(preamble)":
                        sections.append({"title": current_title, "body": body})
                        confidence_sum += 1.0
                        confidence_count += 1
                # Respect max_sections BEFORE starting a new one
                if len(sections) >= self._max_sections:
                    current_title = None  # signal: no more sections accepted
                    break
                current_title = heading
                current_body = []
            else:
                if current_title is not None:
                    current_body.append(line)

        # Tail section (only if we haven't been told to stop)
        if current_title is not None and (current_body or current_title != "(preamble)"):
            body = "\n".join(current_body).strip()
            sections.append({"title": current_title, "body": body})
            confidence_sum += 1.0
            confidence_count += 1

        avg_confidence = (confidence_sum / confidence_count) if confidence_count else 0.0
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "section_count": len(sections),
                "char_count": len(text),
                "sections": sections[: self._max_sections],
            },
            confidence=avg_confidence,
            schema_used=self.schema_name,
        )

    def _detect_heading(self, line: str) -> str | None:
        stripped = line.rstrip()
        if not stripped or len(stripped) > 200:
            return None
        for pat in _HEADING_PATTERNS:
            m = pat.match(stripped)
            if m:
                return m.group(1).strip()
        return None
