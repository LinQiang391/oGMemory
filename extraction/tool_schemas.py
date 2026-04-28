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
    - when: Optional temporal information
    - who: Optional people involved
    - where: Optional location
    """

    abstract: str = Field(description="Brief summary (≤200 chars)")
    overview: str = Field(description="Structured overview")
    content: str = Field(description="Full detailed content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")

    when: str | None = Field(
        default=None,
        description=(
            "Temporal information: convert to absolute date when context allows. "
            "Examples: '2023-07-03', '7 May 2023 (yesterday)', 'summer 2023'. "
            "If no context to resolve, preserve original expression. "
            "Leave empty if no time reference."
        ),
    )
    who: str | None = Field(
        default=None,
        description="People involved: names, roles, relationships. Copy exact names.",
    )
    where: str | None = Field(
        default=None,
        description="Location or spatial context. Copy exact place names.",
    )


class ExtractProfileInput(ToolInput):
    """Input model for extract_profile tool.

    Extract individual profile attributes: name, location, occupation,
    background, etc. Call ONCE per attribute — each attribute becomes
    an independent node that can be updated without affecting others.

    routing_key identifies the attribute (e.g., 'name', 'location', 'occupation').
    """

    routing_key: str = Field(
        description="Attribute identifier (lowercase, underscored, e.g., 'name', 'location', 'occupation', 'background')"
    )

    evidence_quote: str = Field(
        description="Verbatim quote from the source text that supports this extraction. Copy the exact words."
    )
    attributed_speaker: str = Field(
        description="Who said or stated this fact. Use 'user' if the human operator stated it directly. "
        "Use the speaker's name if from a [Name]: prefix in group chat (e.g., 'Andrew', 'Audrey')."
    )
    attribution_basis: str = Field(
        description="How you determined the speaker: "
        "'self_first_person' (user says 'I X' in direct dialogue with no named speakers), "
        "'self_named' (user speaks via a [Name]: prefix that matches the user's identity), "
        "'other_named' (someone else speaks via [Name]: or is quoted — do NOT use extract_profile for this)"
    )

    tool_name: ClassVar[str] = "extract_profile"
    tool_description: ClassVar[str] = (
        "Extract individual user profile attributes: name, location, occupation, "
        "background, relationships. Call once per distinct attribute. "
        "Each attribute gets its own node (e.g., profile/name, profile/location). "
        "Only use for facts BY the user AND ABOUT the user. "
        "For facts about other people in group chats or forwarded content, use extract_entity instead. "
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

    routing_key: str = Field(
        description="Skill identifier (lowercase, underscored, e.g., 'debug_protocol', 'code_review_checklist')"
    )

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


class ExtractToolInput(ToolInput):
    """Input model for extract_tool tool.

    Extract tool usage insights: best scenarios, optimal parameters,
    common failures, and recommendations.
    Call once per distinct tool used in the conversation.
    """

    routing_key: str = Field(
        description="Tool name as identifier (e.g., 'bash', 'read', 'grep')"
    )

    tool_identifier: str = Field(
        description="Exact tool name from the conversation"
    )

    best_for: str = Field(
        default="",
        description="Best use scenarios for this tool",
    )
    optimal_params: str = Field(
        default="",
        description="Recommended parameters and their values",
    )
    common_failures: str = Field(
        default="",
        description="Common failure patterns and how to avoid them",
    )
    recommendation: str = Field(
        default="",
        description="General recommendations for using this tool effectively",
    )

    tool_name: ClassVar[str] = "extract_tool"
    tool_description: ClassVar[str] = (
        "Extract tool usage insights from the conversation. "
        "Reports best scenarios, optimal parameters, common failures, "
        "and recommendations for a specific tool. "
        "Call once per distinct tool used in the conversation."
    )
    category: ClassVar[str] = "tool"
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
    ExtractToolInput,
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
                "description": prop_schema.get("description", ""),
            }
            # Handle Optional[str] which produces {"anyOf": [{"type": "string"}, {"type": "null"}]}
            if "anyOf" in prop_schema:
                types = [s.get("type") for s in prop_schema["anyOf"] if s.get("type")]
                if "string" in types:
                    properties[prop_name]["type"] = "string"
                    if "null" in [s.get("type") for s in prop_schema["anyOf"]]:
                        properties[prop_name]["nullable"] = True
            else:
                properties[prop_name]["type"] = prop_schema.get("type", "string")

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

    # Auto-fill missing routing_key from abstract before validation
    if "routing_key" not in tool_input or not tool_input["routing_key"]:
        abstract = tool_input.get("abstract", "")
        # Generate slug from first 3 words of abstract
        slug = "_".join(abstract.lower().split()[:3]).replace(",", "").replace(".", "")
        tool_input["routing_key"] = slug or "unnamed"
        logger.info(f"Auto-generated routing_key='{tool_input['routing_key']}' for {tool_name}")

    # Validate input using the model
    try:
        validated = model(**tool_input)
    except Exception as e:
        logger.warning(f"Validation failed for tool {tool_name}: {e}")
        return None

    # Determine routing_key (all tools except profile already have it)
    routing_key = getattr(validated, "routing_key", "") or model.category

    # Build CandidateMemory (preserve temporal/actor/location fields)
    # For tool category, populate tool_stats with tool-specific fields
    tool_stats = None
    if isinstance(validated, ExtractToolInput):
        tool_stats = {}
        if validated.best_for:
            tool_stats["best_for"] = validated.best_for
        if validated.optimal_params:
            tool_stats["optimal_params"] = validated.optimal_params
        if validated.common_failures:
            tool_stats["common_failures"] = validated.common_failures
        if validated.recommendation:
            tool_stats["recommendation"] = validated.recommendation
        if validated.tool_identifier:
            tool_stats["tool_identifier"] = validated.tool_identifier

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
            when=validated.when if hasattr(validated, "when") else None,
            who=validated.who if hasattr(validated, "who") else None,
            where=validated.where if hasattr(validated, "where") else None,
            tool_stats=tool_stats,
            # Attribution fields (profile category only)
            evidence_quote=validated.evidence_quote if isinstance(validated, ExtractProfileInput) else None,
            attributed_speaker=validated.attributed_speaker if isinstance(validated, ExtractProfileInput) else None,
            attribution_basis=validated.attribution_basis if isinstance(validated, ExtractProfileInput) else None,
        ),
    )
