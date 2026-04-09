"""Pydantic models for ContextEngine extraction tool schemas.

Converts the 7 extraction tool schemas from plain dicts to typed Pydantic BaseModel definitions.
Provides utility functions to generate OpenAI function calling wire format and parse tool calls.
"""

from logging import getLogger
from typing import ClassVar

from pydantic import BaseModel, Field

from core.models import CandidateMemory


logger = getLogger(__name__)


class ToolInput(BaseModel):
    """Base class for all extraction tool inputs.

    Common fields across all extraction tools:
    - abstract: Brief summary (≤200 chars)
    - overview: Structured overview
    - content: Full detailed content
    - confidence: Confidence score 0.0-1.0
    """

    abstract: str = Field(description="Brief summary (≤200 chars)")
    overview: str = Field(description="Structured overview")
    content: str = Field(description="Full detailed content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")


class ExtractProfileInput(ToolInput):
    """Input model for extract_profile tool.

    Extract stable user profile information: name, background, role, long-term goals.
    Call at most once per conversation.

    No routing_key required - uses category as default.
    """

    tool_name: ClassVar[str] = "extract_profile"
    tool_description: ClassVar[str] = (
        "Extract stable user profile information: name, background, role, "
        "long-term goals. Call at most once per conversation. "
        "Do NOT extract: transient states, temporary preferences, "
        "specific events, or session-specific information."
    )
    category: ClassVar[str] = "profile"
    owner_scope: ClassVar[str] = "user"


class ExtractPreferenceInput(ToolInput):
    """Input model for extract_preference tool.

    Extract user preferences, opinions, and likes/dislikes.
    Call once per distinct topic (e.g., coffee, coding style, music).
    """

    routing_key: str = Field(
        description="Topic identifier (lowercase, underscored, e.g., 'coffee', 'coding_style')"
    )

    tool_name: ClassVar[str] = "extract_preference"
    tool_description: ClassVar[str] = (
        "Extract user preferences, opinions, and likes/dislikes. "
        "Call once per distinct topic (e.g., coffee, coding style, music). "
        "Each independent topic gets its own call."
    )
    category: ClassVar[str] = "preference"
    owner_scope: ClassVar[str] = "user"


class ExtractEntityInput(ToolInput):
    """Input model for extract_entity tool.

    Extract specific entities: people, places, organizations, objects.
    Call once per distinct entity mentioned.
    """

    routing_key: str = Field(
        description="Entity identifier (normalized name, e.g., 'alice', 'coffee_shop_downtown')"
    )

    tool_name: ClassVar[str] = "extract_entity"
    tool_description: ClassVar[str] = (
        "Extract specific entities: people, places, organizations, objects. "
        "Call once per distinct entity mentioned."
    )
    category: ClassVar[str] = "entity"
    owner_scope: ClassVar[str] = "user"


class ExtractEventInput(ToolInput):
    """Input model for extract_event tool.

    Extract time-bounded events: meetings, visits, actions, occurrences.
    routing_key should include timestamp (e.g., 'visit_20250315').
    """

    routing_key: str = Field(
        description="Event identifier with timestamp (e.g., 'meeting_20250315_1430')"
    )

    tool_name: ClassVar[str] = "extract_event"
    tool_description: ClassVar[str] = (
        "Extract time-bounded events: meetings, visits, actions, occurrences. "
        "routing_key should include timestamp (e.g., 'visit_20250315'). "
        "Call once per distinct event."
    )
    category: ClassVar[str] = "event"
    owner_scope: ClassVar[str] = "user"


class ExtractCaseInput(ToolInput):
    """Input model for extract_case tool.

    Extract problem-solving cases: a complete problem-resolution pair with outcome.
    Must include: 1) The problem/challenge, 2) Steps taken to resolve, 3) Final outcome.
    """

    routing_key: str = Field(
        description=(
            "Case identifier describing problem type (lowercase, underscored, "
            "e.g., 'debug_api_timeout', 'fix_memory_leak')"
        )
    )

    tool_name: ClassVar[str] = "extract_case"
    tool_description: ClassVar[str] = (
        "Extract problem-solving cases: a complete problem-resolution pair with outcome. "
        "Must include: 1) The problem/challenge, 2) Steps taken to resolve, 3) Final outcome. "
        "routing_key should describe the problem type (e.g., 'debug_api_timeout', "
        "'fix_memory_leak', 'resolve_merge_conflict'). Call once per resolved issue."
    )
    category: ClassVar[str] = "case"
    owner_scope: ClassVar[str] = "agent"


class ExtractPatternInput(ToolInput):
    """Input model for extract_pattern tool.

    Extract agent-learned patterns: 'I noticed that...', 'It seems that...'
    observations about user behavior or conversation patterns.
    """

    routing_key: str = Field(
        description="Pattern topic identifier (lowercase, underscored)"
    )

    tool_name: ClassVar[str] = "extract_pattern"
    tool_description: ClassVar[str] = (
        "Extract agent-learned patterns: 'I noticed that...', 'It seems that...' "
        "observations about user behavior or conversation patterns. "
        "Do NOT extract: complete reusable workflows with clear trigger conditions "
        "and success criteria — those belong to extract_skill."
    )
    category: ClassVar[str] = "pattern"
    owner_scope: ClassVar[str] = "agent"


class ExtractSkillInput(ToolInput):
    """Input model for extract_skill tool.

    Extract reusable, structured workflows.
    Must satisfy ALL three criteria:
    1) Clear trigger condition (when to use)
    2) Specific step sequence (what to do)
    3) Verifiable completion criteria (how to know it's done)
    """

    tool_name: ClassVar[str] = "extract_skill"
    tool_description: ClassVar[str] = (
        "Extract reusable, structured workflows. Must satisfy ALL three criteria: "
        "1) Clear trigger condition (when to use), "
        "2) Specific step sequence (what to do), "
        "3) Verifiable completion criteria (how to know it's done). "
        "Examples: debugging protocol, code review checklist, error handling workflow."
    )
    category: ClassVar[str] = "skill"
    owner_scope: ClassVar[str] = "agent"


# All tool input models in order
TOOL_MODELS = [
    ExtractProfileInput,
    ExtractPreferenceInput,
    ExtractEntityInput,
    ExtractEventInput,
    ExtractCaseInput,
    ExtractPatternInput,
    ExtractSkillInput,
]


def get_tool_definitions() -> list[dict]:
    """Generate OpenAI function calling wire format from Pydantic models.

    Returns a list of dicts with "name", "description", "input_schema" keys,
    matching the current EXTRACTION_TOOLS structure exactly.

    The generated schemas use model.model_json_schema() to produce the correct
    OpenAI function calling format.
    """
    tools = []
    for model in TOOL_MODELS:
        schema = model.model_json_schema()
        tools.append({
            "name": model.tool_name,
            "description": model.tool_description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        })

        # Build properties from schema
        properties = {}
        required = []

        for prop_name, prop_schema in schema.get("properties", {}).items():
            # Skip ClassVar metadata fields
            if prop_name in ("tool_name", "tool_description", "category", "owner_scope"):
                continue

            properties[prop_name] = {
                "type": prop_schema.get("type", "string"),
                "description": prop_schema.get("description", ""),
            }

            # Check if field is required
            if prop_name in schema.get("required", []):
                required.append(prop_name)

        tools[-1]["input_schema"]["properties"] = properties
        tools[-1]["input_schema"]["required"] = required

    return tools


def get_tool_to_category() -> dict[str, tuple[str, str]]:
    """Generate TOOL_TO_CATEGORY mapping from model metadata.

    Returns:
        Dict mapping tool_name -> (category, owner_scope)
    """
    return {
        model.tool_name: (model.category, model.owner_scope)
        for model in TOOL_MODELS
    }


def parse_tool_call(
    tool_name: str,
    tool_input: dict,
) -> tuple[str, str, CandidateMemory] | None:
    """Validate and parse a tool call using the appropriate Pydantic model.

    Args:
        tool_name: Name of the tool being called
        tool_input: Raw input dict from the tool call

    Returns:
        Tuple of (category, owner_scope, CandidateMemory) if valid,
        None if tool_name is unknown or validation fails
    """
    # Find the appropriate model
    model = None
    for m in TOOL_MODELS:
        if m.tool_name == tool_name:
            model = m
            break

    if model is None:
        logger.warning(f"Unknown tool name: {tool_name}")
        return None

    # Validate input using the model
    try:
        validated = model(**tool_input)
    except Exception as e:
        logger.warning(f"Validation failed for tool {tool_name}: {e}")
        return None

    # Determine routing_key
    routing_key = validated.routing_key if hasattr(validated, "routing_key") else ""
    if not routing_key:
        # Use category as default for profile and skill
        routing_key = model.category

    # Build CandidateMemory
    return (
        model.category,
        model.owner_scope,
        CandidateMemory(
            category=model.category,
            owner_scope=model.owner_scope,
            routing_key=routing_key,
            abstract=validated.abstract,
            overview=validated.overview,
            content=validated.content,
            confidence=validated.confidence,
        ),
    )
