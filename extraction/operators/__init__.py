"""Multimodal extraction operators.

Provides a pluggable framework for extracting structured content
from raw multimodal data:

- :class:`MultimodalExtractor` — the base Protocol every operator implements
- :class:`OperatorRegistry` — plug-and-play registration (code + YAML)
- :class:`OperatorPipeline` — DAG-style orchestration with conditional
  and parallel branches
- :class:`TaskStore` — async task tracking
- :class:`PromptExtractor` — built-in text-prompt analyzer

Add new operators via code::

    from extraction.operators import default_registry
    default_registry().register(MyOperator(...))

Or via YAML (``config/operators.yaml``) — see :meth:`OperatorRegistry.load_from_config`.
"""

from extraction.operators.base import (
    ExtractionResult,
    MultimodalExtractor,
    OperatorInfo,
    default_info,
)
from extraction.operators.pipeline import (
    OperatorPipeline,
    PipelineResult,
    PipelineSpec,
    PipelineSpecError,
    StepResult,
    StepSpec,
)
from extraction.operators.registry import (
    OperatorRegistry,
    OperatorRegistryError,
    default_registry,
    reset_default_registry,
)
from extraction.operators.task_store import (
    InMemoryTaskStore,
    TaskRecord,
    TaskStore,
)
from extraction.operators.prompt_extractor import PromptExtractor

__all__ = [
    # Base abstractions
    "ExtractionResult",
    "MultimodalExtractor",
    "OperatorInfo",
    "default_info",
    # Registry
    "OperatorRegistry",
    "OperatorRegistryError",
    "default_registry",
    "reset_default_registry",
    # Pipeline
    "OperatorPipeline",
    "PipelineSpec",
    "PipelineResult",
    "PipelineSpecError",
    "StepSpec",
    "StepResult",
    # Async tasks
    "TaskStore",
    "InMemoryTaskStore",
    "TaskRecord",
    # Built-in operators
    "PromptExtractor",
]
