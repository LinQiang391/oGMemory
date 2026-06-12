"""Base class for multimodal extraction operators.

Each extractor handles a specific type of raw data and produces
structured extraction results that can be stored as ExtractedNodes.

Usage::

    class ImageExtractor(MultimodalExtractor):
        @property
        def supported_types(self) -> list[RawDataType]:
            return [RawDataType.IMAGE]

        def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
            # Call vision model to analyze image
            description = self._analyze_image(raw_node.content)
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"description": description},
                confidence=0.9,
                schema_used="image_description_v1"
            )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from core.graph_enums import RawDataType

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext


@dataclass
class ExtractionResult:
    """Result of an extraction operation.

    Contains the extracted structured content that will be stored
    as an ExtractedNode in the graph.
    """
    node_id: str  # ID of the source raw node
    extracted_content: dict  # Structured extraction results
    confidence: float  # Extraction confidence (0-1)
    schema_used: str  # Schema name used for extraction
    error: str | None = None  # Error message if extraction failed


class MultimodalExtractor(Protocol):
    """Protocol for multimodal extraction operators.

    Each extractor handles specific raw data types and produces
    structured extraction results.
    """

    @property
    def supported_types(self) -> list[RawDataType]:
        """Return the list of raw data types this extractor supports."""
        ...

    @property
    def extractor_type(self) -> str:
        """Return the type identifier for this extractor."""
        ...

    @property
    def schema_name(self) -> str:
        """Return the schema name used for extraction results."""
        ...

    def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
        """Extract structured content from raw data.

        Args:
            raw_node: The source raw node
            ctx: RequestContext for access control

        Returns:
            ExtractionResult with extracted content
        """
        ...

    def extract_batch(
        self, raw_nodes: list[RawNode], ctx: RequestContext
    ) -> list[ExtractionResult]:
        """Extract structured content from multiple raw nodes.

        Default implementation calls extract() for each node.
        Override for batch-optimized implementations.

        Args:
            raw_nodes: List of source raw nodes
            ctx: RequestContext for access control

        Returns:
            List of ExtractionResults
        """
        return [self.extract(node, ctx) for node in raw_nodes]