# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Dynamic tool definition builder from SchemaRegistry.

Generates OpenAI function calling tool definitions from schema metadata,
replacing hardcoded tool_schemas.py with a schema-driven approach.
"""

from datetime import datetime, timezone
from logging import getLogger

from core.models import CandidateMemory
from extraction.schemas.models import FieldType, MemoryTypeSchema, SchemaField
from extraction.schemas.registry import SchemaRegistry

logger = getLogger(__name__)

# Map memory_type to its identifying field for routing_key derivation
_ROUTING_KEY_FIELDS = {
    "profile": "routing_key",
    "preference": "topic",
    "entity": "name",
    "event": "event_name",
    "case": "case_name",
    "pattern": "topic",
    "skill": "skill_name",
    "tool": "tool_identifier",
}

# Field type mapping to JSON Schema types
_FIELD_TYPE_JSON_SCHEMA = {
    FieldType.STRING: "string",
    FieldType.NUMBER: "number",
    FieldType.INTEGER: "integer",
    FieldType.BOOLEAN: "boolean",
}


def build_extraction_tools(registry: SchemaRegistry) -> list[dict]:
    """Generate OpenAI function calling tool definitions from enabled schemas.

    Each schema generates one tool definition with:
    - name: schema.tool_name (extract_{memory_type})
    - description: schema.description
    - input_schema: JSON Schema derived from schema.fields

    Args:
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        List of tool definition dicts compatible with OpenAI function calling
    """
    tools = []

    for schema in registry.list_enabled():
        # Build JSON Schema for input parameters
        properties = {}
        required = []

        for field in schema.fields:
            prop_def = {
                "type": _FIELD_TYPE_JSON_SCHEMA.get(field.field_type, "string"),
            }

            if field.description:
                prop_def["description"] = field.description

            if field.enum:
                prop_def["enum"] = field.enum

            properties[field.name] = prop_def

            if field.required:
                required.append(field.name)

        tools.append({
            "name": schema.tool_name,
            "description": schema.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })

        logger.debug(f"Built tool definition: {schema.tool_name}")

    return tools


def build_tool_to_category(registry: SchemaRegistry) -> dict[str, tuple[str, str]]:
    """Generate tool_name -> (memory_type, owner_scope) mapping.

    Args:
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        Dict mapping tool_name to (memory_type, owner_scope) tuple
    """
    return {
        schema.tool_name: (schema.memory_type, schema.owner_scope)
        for schema in registry.list_enabled()
    }


def parse_tool_call(
    tool_name: str,
    tool_input: dict,
    registry: SchemaRegistry,
) -> tuple[str, str, CandidateMemory] | None:
    """Validate and parse a tool call using schema definitions.

    Args:
        tool_name: Name of the tool being called (e.g., "extract_profile")
        tool_input: Raw input dict from the tool call
        registry: SchemaRegistry instance with loaded schemas

    Returns:
        Tuple of (memory_type, owner_scope, CandidateMemory) if valid,
        None if tool_name is unknown or validation fails
    """
    # Find schema by tool_name
    schema = None
    for s in registry.list_enabled():
        if s.tool_name == tool_name:
            schema = s
            break

    if schema is None:
        logger.warning(f"Unknown tool name: {tool_name}")
        return None

    # Pre-fill auto-generatable fields before validation so required checks pass
    tool_input = dict(tool_input)  # copy to avoid mutating caller's dict
    if schema.memory_type in ("event", "case") and "timestamp" not in tool_input:
        tool_input["timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if "confidence" not in tool_input or tool_input.get("confidence") is None:
        tool_input["confidence"] = 0.5
    if "content" not in tool_input or not tool_input.get("content"):
        tool_input["content"] = tool_input.get("abstract", "")
    if "overview" not in tool_input or not tool_input.get("overview"):
        tool_input["overview"] = tool_input.get("abstract", "")
    # Auto-generate key/name fields from abstract
    abstract = tool_input.get("abstract", "")
    if abstract:
        slug = "_".join(abstract.lower().split()[:3]).replace(",", "").replace(".", "")
    else:
        slug = "unnamed"
    key_field = _ROUTING_KEY_FIELDS.get(schema.memory_type, "routing_key")
    if key_field not in tool_input or not tool_input.get(key_field):
        tool_input[key_field] = slug
    if schema.memory_type == "entity" and "name" not in tool_input:
        tool_input["name"] = slug
    if schema.memory_type in ("preference", "pattern") and "topic" not in tool_input:
        tool_input["topic"] = slug
    # Event-specific: event_name fallback
    if schema.memory_type == "event" and "event_name" not in tool_input:
        tool_input["event_name"] = slug

    # Validate fields against schema
    validated_fields = {}
    errors = []

    for field in schema.fields:
        if field.name in tool_input:
            value = tool_input[field.name]

            # Type validation
            if field.field_type == FieldType.STRING:
                if value is not None and not isinstance(value, str):
                    try:
                        value = str(value)
                    except Exception:
                        errors.append(f"{field.name}: expected string, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.NUMBER:
                if value is not None and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{field.name}: expected number, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.INTEGER:
                if value is not None and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{field.name}: expected integer, got {type(value).__name__}")
                        continue

            elif field.field_type == FieldType.BOOLEAN:
                if value is not None and not isinstance(value, bool):
                    # Try to parse common string representations
                    if isinstance(value, str):
                        value_lower = value.lower()
                        if value_lower in ("true", "1", "yes"):
                            value = True
                        elif value_lower in ("false", "0", "no"):
                            value = False
                        else:
                            errors.append(f"{field.name}: expected boolean, got '{value}'")
                            continue
                    else:
                        errors.append(f"{field.name}: expected boolean, got {type(value).__name__}")
                        continue

            # Enum validation
            if field.enum and value not in field.enum:
                errors.append(f"{field.name}: value '{value}' not in enum {field.enum}")
                continue

            validated_fields[field.name] = value

        elif field.required:
            errors.append(f"{field.name}: required field missing")
            if field.default is not None:
                validated_fields[field.name] = field.default
        elif field.default is not None:
            validated_fields[field.name] = field.default

    if errors:
        logger.warning(f"Validation failed for tool {tool_name}: {errors}")
        return None

    # Extract common fields for CandidateMemory
    abstract = validated_fields.get("abstract", "")
    overview = validated_fields.get("overview", "")
    content = validated_fields.get("content", "")
    confidence = validated_fields.get("confidence", 0.5)

    # Auto-generate timestamp for event/case if not provided
    if schema.memory_type in ("event", "case"):
        if "timestamp" not in validated_fields or not validated_fields["timestamp"]:
            validated_fields["timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            logger.info(f"Auto-generated timestamp for {schema.memory_type}: {validated_fields['timestamp']}")

    # Derive routing_key from semantic field based on memory_type
    routing_key_field = _ROUTING_KEY_FIELDS.get(schema.memory_type, "routing_key")
    routing_key = validated_fields.get(routing_key_field, "")

    # For event and case, combine name with timestamp
    if schema.memory_type in ("event", "case"):
        ts = validated_fields.get("timestamp", "")
        routing_key = f"{routing_key}_{ts}" if routing_key and ts else routing_key or ts

    # Fallback: generate slug from abstract if no routing_key
    if not routing_key:
        slug = "_".join(abstract.lower().split()[:3]).replace(",", "").replace(".", "")
        routing_key = slug or "unnamed"
        logger.info(f"Auto-generated routing_key='{routing_key}' for {tool_name} from abstract")

    # Handle temporal/actor/location fields
    when = validated_fields.get("when")
    who = validated_fields.get("who")
    where = validated_fields.get("where")

    # Handle attribution fields (profile category)
    evidence_quote = validated_fields.get("evidence_quote") if schema.memory_type == "profile" else None
    attributed_speaker = validated_fields.get("attributed_speaker") if schema.memory_type == "profile" else None
    attribution_basis = validated_fields.get("attribution_basis") if schema.memory_type == "profile" else None

    # Handle tool-specific fields (for 'tool' category)
    tool_stats = None
    if schema.memory_type == "tool":
        tool_stats = {}
        tool_stat_fields = ["best_for", "optimal_params", "common_failures",
                           "recommendation", "tool_identifier"]
        for stat_field in tool_stat_fields:
            if stat_field in validated_fields and validated_fields[stat_field]:
                tool_stats[stat_field] = validated_fields[stat_field]
        if tool_stats:
            logger.debug(f"Included tool_stats for {tool_name}: {list(tool_stats.keys())}")

    return (
        schema.memory_type,
        schema.owner_scope,
        CandidateMemory(
            category=schema.memory_type,
            owner_scope=schema.owner_scope,
            routing_key=routing_key,
            abstract=abstract,
            overview=overview,
            content=content,
            confidence=confidence,
            when=when,
            who=who,
            where=where,
            tool_stats=tool_stats,
            evidence_quote=evidence_quote,
            attributed_speaker=attributed_speaker,
            attribution_basis=attribution_basis,
        ),
    )
