"""core.tools — Tool framework building blocks for ReAct loops.

Three tool categories share this foundation:
- Operational tools: @operational_tool decorator + call_with_injection
- Extraction tools: YamlSpecRegistry → SchemaRegistry (in extraction/schemas/)
- Prompt-driven tools: YamlSpecRegistry → PromptDrivenSchemaRegistry (in core/tools/)
"""

from core.tools.operational_tool_spec import (
    OperationalToolSpec,
    operational_tool,
    call_with_injection,
)
from core.tools.yaml_tool_schema import (
    YamlSpecRegistry,
    build_input_schema_from_fields,
    validate_tool_input,
    FIELD_TYPE_JSON_SCHEMA,
)
from core.tools.prompt_driven_tool_spec import (
    PromptDrivenToolSpec,
    PromptDrivenSchemaRegistry,
    build_prompt_driven_tools,
)

__all__ = [
    "OperationalToolSpec",
    "operational_tool",
    "call_with_injection",
    "YamlSpecRegistry",
    "build_input_schema_from_fields",
    "validate_tool_input",
    "FIELD_TYPE_JSON_SCHEMA",
    "PromptDrivenToolSpec",
    "PromptDrivenSchemaRegistry",
    "build_prompt_driven_tools",
]
