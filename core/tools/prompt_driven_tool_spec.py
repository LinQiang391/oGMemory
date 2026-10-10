"""Prompt-driven tool spec framework.

Prompt-driven tools differ from operational tools and extraction intent tools:
- Operational tools (@operational_tool): deterministic function, LLM sees real results
- Extraction intent tools (YAML schema): LLM "fills form", returns {"status": "recorded"},
  parsed post-loop
- Prompt-driven tools: YAML schema + prompt_guidance + post_call_validator
  - LLM fills schema fields (structured content)
  - validator runs deterministic check (e.g., duplicate detection)
  - validator returns feedback LLM can see and react to within the loop
  - structured content collected as output, written via commit pipeline post-loop

This module provides:
- PromptDrivenToolSpec: metadata descriptor for a prompt-driven tool
- PromptDrivenSchemaRegistry: loads PromptDrivenToolSpec from YAML directory
- build_prompt_driven_tools: generates OpenAI tool definitions from registry

Internal fields excluded from the LLM-visible schema are listed in
_EXCLUDED_TOOL_FIELDS. Note: provenance_ids is NO longer excluded —
the LLM now explicitly provides source URIs via affected_uris /
source_uris, which _tool_input_to_dream_output() resolves into
provenance_ids via ProvenanceResolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from core.tools.yaml_tool_schema import (
    build_input_schema_from_fields,
    validate_tool_input,
    YamlSpecRegistry,
)

logger = getLogger(__name__)

# Fields kept in YAML definitions for internal metadata, but not exposed
# to the LLM as tool-call arguments. provenance_ids is NO longer excluded
# — the LLM now provides source URIs explicitly (affected_uris / source_uris),
# which are resolved to provenance_ids by ProvenanceResolver in
# _tool_input_to_dream_output().
_EXCLUDED_TOOL_FIELDS: set[str] = set()


@dataclass
class PromptDrivenToolSpec:
    """Metadata descriptor for a prompt-driven tool.

    Bound to tool definition via PromptDrivenSchemaRegistry lookup.
    """
    name: str                              # e.g. "process_correction"
    description: str                       # Tool description for LLM
    input_schema: dict                     # JSON Schema (LLM parameters only)
    prompt_guidance: str                   # YAML prompt_guidance → injected into system prompt
    post_call_validator: str               # Validator function name (dispatch key)
    category: str                          # e.g. "dream"
    dream_type: str | None = None          # e.g. "correction" / "consolidation"

    def tool_dict(self) -> dict:
        """Convert to OpenAI function-calling tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate(self, tool_input: dict) -> list[str]:
        """Validate LLM input against schema: required fields and basic type checks.

        Delegates to the shared validate_tool_input() function.
        Returns list of error strings (empty = valid).
        """
        return validate_tool_input(tool_input, self.input_schema)


class PromptDrivenSchemaRegistry(YamlSpecRegistry[PromptDrivenToolSpec]):
    """Registry for prompt-driven tool YAML definitions.

    Inherits common YAML loading, register/get/list_all mechanics
    from YamlSpecRegistry[PromptDrivenToolSpec].

    If schemas_dir is not provided, defaults to dream/tools/tool_specs/
    relative to this package (using dream.tools.__file__).
    This ensures correct path resolution regardless of cwd.
    """

    def __init__(self, schemas_dir: str | None = None):
        if schemas_dir is None:
            import dream.tools as tools_pkg
            schemas_dir = str(Path(tools_pkg.__file__).parent / "tool_specs")
        super().__init__(schemas_dir=schemas_dir)

    # ── YamlSpecRegistry hooks ──────────────────────────────────────────

    def _parse_yaml(self, data: dict) -> PromptDrivenToolSpec:
        """Parse a prompt-driven tool YAML dict into PromptDrivenToolSpec."""
        tool_name = data.get("tool_name", "")
        if not tool_name:
            raise ValueError("Prompt-driven YAML missing required field: tool_name")

        description = data.get("description", "")
        category = data.get("category", "")
        dream_type = data.get("dream_type")
        prompt_guidance = data.get("prompt_guidance", "")
        post_call_validator = data.get("post_call_validator", "")

        fields = data.get("fields", [])
        if not isinstance(fields, list):
            raise ValueError(f"Prompt-driven YAML {tool_name}: fields must be a list")

        input_schema = build_input_schema_from_fields(
            fields, excluded_fields=_EXCLUDED_TOOL_FIELDS
        )

        return PromptDrivenToolSpec(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            prompt_guidance=prompt_guidance,
            post_call_validator=post_call_validator,
            category=category,
            dream_type=dream_type,
        )

    def _should_load(self, data: dict) -> bool:
        """Only load YAML files that define a prompt-driven tool (have tool_name)."""
        return isinstance(data, dict) and data.get("tool_name") is not None

    def _key(self, spec: PromptDrivenToolSpec) -> str:
        """Use spec.name as the registry key."""
        return spec.name


def build_prompt_driven_tools(registry: PromptDrivenSchemaRegistry) -> list[dict]:
    """Build OpenAI tool definitions from all registered prompt-driven specs.

    Parallel to extraction/tool_builder.py build_extraction_tools().
    """
    return [spec.tool_dict() for spec in registry.list_all()]