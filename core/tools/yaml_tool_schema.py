"""Shared infrastructure for YAML-driven tool definitions.

Both extraction tools (SchemaRegistry + MemoryTypeSchema) and prompt-driven
tools (PromptDrivenSchemaRegistry + PromptDrivenToolSpec) share the same
mechanics for:

1. Building OpenAI-compatible input_schema from YAML field definitions
2. Validating LLM tool call input against that schema
3. Loading YAML files from a directory into a typed registry

This module provides the shared building blocks so that each tool category
only defines its own spec dataclass and YAML parsing logic — no duplication
of the schema-building, validation, or registry-loading boilerplate.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

logger = getLogger(__name__)

# ── Layer 1: fields → JSON Schema ──────────────────────────────────────────

# Field type mapping: YAML type strings → JSON Schema type strings.
# Used by both extraction/tool_builder.py and prompt_driven_tool_spec.py.
FIELD_TYPE_JSON_SCHEMA: dict[str, str] = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "list": "array",
}


def build_input_schema_from_fields(
    fields: list[dict],
    excluded_fields: set[str] | None = None,
) -> dict:
    """Build OpenAI-compatible input_schema from YAML field definitions.

    Each field dict has keys: name, type, required, description, enum.

    Args:
        fields: List of field definition dicts from YAML.
        excluded_fields: Field names to exclude from the LLM-visible schema
                        (e.g. internal metadata fields like provenance_ids).

    Returns:
        JSON Schema dict: {"type": "object", "properties": {...}, "required": [...]}
    """
    properties = {}
    required = []
    excluded = excluded_fields or set()

    for f in fields:
        name = f.get("name", "")
        if name in excluded:
            continue

        field_type = f.get("type", "string")
        json_type = FIELD_TYPE_JSON_SCHEMA.get(field_type, "string")

        prop_def: dict[str, Any] = {"type": json_type}
        if f.get("description"):
            prop_def["description"] = f["description"]
        if f.get("enum"):
            prop_def["enum"] = f["enum"]

        properties[name] = prop_def

        if f.get("required", False):
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ── Layer 2: LLM input validation ──────────────────────────────────────────

_TYPE_CHECKS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)),
    "integer": lambda v: isinstance(v, int),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
}


def validate_tool_input(
    tool_input: dict,
    input_schema: dict,
) -> list[str]:
    """Validate LLM tool call input against a JSON Schema.

    Checks:
    - Required fields are present and not None
    - Present fields match their declared type

    Args:
        tool_input: Dict of LLM-provided parameter values.
        input_schema: JSON Schema dict with "properties" and "required".

    Returns:
        List of error strings (empty = valid).
    """
    errors = []
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Check required fields
    for field_name in required:
        if field_name not in tool_input or tool_input[field_name] is None:
            errors.append(f"{field_name}: required field missing")

    # Check types for present fields
    for field_name, value in tool_input.items():
        if field_name in properties and value is not None:
            expected_type = properties[field_name].get("type")
            if expected_type and expected_type in _TYPE_CHECKS:
                if not _TYPE_CHECKS[expected_type](value):
                    errors.append(
                        f"{field_name}: expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )

    return errors


# ── Layer 3: Generic YAML registry ─────────────────────────────────────────

T = TypeVar("T")


class YamlSpecRegistry(Generic[T]):
    """Generic base class for YAML-driven spec registries.

    Provides the common loading, storage, and lookup mechanics.
    Subclasses define:
    - _parse_yaml(data) -> T: how to turn a YAML dict into a spec object
    - _should_load(data) -> bool: which YAML files belong to this registry
    - _key(spec) -> str: what field to use as the dict key (default: spec.name)

    Subclasses may also override _load_from_directory to customise the
    glob pattern, error handling, or filtering logic.
    """

    def __init__(self, schemas_dir: str | None = None):
        self._specs: dict[str, T] = {}

        if schemas_dir and Path(schemas_dir).exists():
            self._load_from_directory(schemas_dir)

    def _parse_yaml(self, data: dict) -> T:
        """Parse a YAML dict into a spec object. Subclass must implement."""
        raise NotImplementedError

    def _should_load(self, data: dict) -> bool:
        """Return True if this YAML dict belongs to this registry.

        Default: accept all dicts. Subclasses override to filter by
        discriminator fields (e.g. `memory_type` vs `tool_name`).
        """
        return isinstance(data, dict)

    def _key(self, spec: T) -> str:
        """Return the dict key for a spec. Default: spec.name if present."""
        return getattr(spec, "name", "")

    def _load_from_directory(self, dir_path: str) -> int:
        """Load all YAML files from a directory.

        For each file: yaml.safe_load → _should_load filter → _parse_yaml → register.

        Returns number of specs loaded.
        """
        count = 0
        dir_path_obj = Path(dir_path)

        for yaml_file in sorted(dir_path_obj.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not self._should_load(data):
                    continue
                spec = self._parse_yaml(data)
                self.register(spec)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        for yaml_file in sorted(dir_path_obj.glob("*.yml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not self._should_load(data):
                    continue
                spec = self._parse_yaml(data)
                self.register(spec)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {count} specs from {dir_path}")
        return count

    def register(self, spec: T) -> None:
        """Register a spec object."""
        key = self._key(spec)
        self._specs[key] = spec
        logger.debug(f"Registered spec: {key}")

    def get(self, name: str) -> T | None:
        """Get a spec by key."""
        return self._specs.get(name)

    def list_all(self) -> list[T]:
        """List all registered specs."""
        return list(self._specs.values())