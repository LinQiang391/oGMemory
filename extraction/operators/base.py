"""Base class for multimodal extraction operators.

Each extractor handles a specific type of raw data and produces
structured extraction results that can be stored as ExtractedNodes.

The framework supports:
- Plug-and-play registration via OperatorRegistry (code or YAML)
- DAG-style orchestration via OperatorPipeline
- Async execution via TaskStore
- Pluggable CV backends via CVBackend Protocol

Usage::

    class ImageExtractor(MultimodalExtractor):
        @property
        def supported_types(self) -> list[RawDataType]:
            return [RawDataType.IMAGE]

        def extract(self, raw_node: RawNode, ctx: RequestContext) -> ExtractionResult:
            description = self._analyze_image(raw_node.content)
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={"description": description},
                confidence=0.9,
                schema_used="image_description_v1"
            )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from core.graph_enums import RawDataType

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext
    from extraction.operators.materialize import MaterializationPolicy


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


@dataclass
class OperatorInfo:
    """Public metadata for a registered operator.

    Used by GET /api/v1/operators to advertise capabilities and
    by OperatorRegistry.load_from_config to validate registrations.
    """
    operator_type: str                          # Unique id: "object_detection"
    display_name: str                           # Human-friendly: "YOLO Object Detection"
    description: str = ""
    supported_raw_types: list[RawDataType] = field(default_factory=list)
    schema_name: str = ""                       # Default schema for results
    version: str = "0.1.0"
    config_schema: dict[str, Any] | None = None  # JSON Schema for operator params
    # If set, ``OperatorService.run_operator`` will materialize the items
    # under ``extracted_content[materialization.collection_field]`` into
    # child ``RawNode``s of ``materialization.child_raw_type``, with edges
    # from the source node. Operators without a multi-output structure
    # (e.g. ``image_caption`` which returns a single description) leave
    # this ``None``.
    materialization: MaterializationPolicy | None = None


class MultimodalExtractor(Protocol):
    """Protocol for multimodal extraction operators.

    Each extractor handles specific raw data types and produces
    structured extraction results.

    Note: this Protocol is structural — use ``is_multimodal_extractor(obj)``
    to check at runtime. Python's ``@runtime_checkable`` decorator does
    not correctly detect Protocol properties (only methods).
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

    @property
    def info(self) -> OperatorInfo:
        """Return public metadata for this operator.

        Default implementation derives from the other properties,
        but operators are encouraged to override for richer descriptions
        and config_schema.
        """
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


def default_info(extractor: MultimodalExtractor) -> OperatorInfo:
    """Build a default OperatorInfo from an extractor's properties.

    Used when an extractor doesn't override .info; preserves backwards
    compatibility with operators that only define the original three
    properties.
    """
    return OperatorInfo(
        operator_type=extractor.extractor_type,
        display_name=extractor.extractor_type.replace("_", " ").title(),
        supported_raw_types=list(extractor.supported_types),
        schema_name=extractor.schema_name,
    )


# Members that an operator MUST expose to satisfy MultimodalExtractor.
# Used by is_multimodal_extractor() — see docstring on the Protocol.
_REQUIRED_OPERATOR_ATTRS = (
    "supported_types",
    "extractor_type",
    "schema_name",
    "extract",
)


def is_multimodal_extractor(obj: object) -> bool:
    """Duck-typed MultimodalExtractor conformance check.

    Use this in lieu of ``isinstance(obj, MultimodalExtractor)`` because
    Python's runtime Protocol check does not correctly detect Protocol
    properties at runtime (only methods). This helper explicitly
    verifies the presence of every required attribute.
    """
    if obj is None:
        return False
    for attr in _REQUIRED_OPERATOR_ATTRS:
        if not hasattr(obj, attr):
            return False
    return True
