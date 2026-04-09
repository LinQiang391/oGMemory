"""Tool-use based candidate extraction.

Uses OpenAI function calling to extract candidate memories from conversations.
Single LLM call with 7 tools: profile/preference/entity/event/case/pattern/skill.
"""

from typing import Any

from core.interfaces import LLM
from core.models import RequestContext, CandidateMemory


# Tool definitions for LLM function calling
EXTRACTION_TOOLS = [
    {
        "name": "extract_profile",
        "description": (
            "Extract stable user profile information: name, background, role, "
            "long-term goals. Call at most once per conversation. "
            "Do NOT extract: transient states, temporary preferences, "
            "specific events, or session-specific information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of who the user is (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview of user profile",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed profile content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_preference",
        "description": (
            "Extract user preferences, opinions, and likes/dislikes. "
            "Call once per distinct topic (e.g., coffee, coding style, music). "
            "Each independent topic gets its own call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routing_key": {
                    "type": "string",
                    "description": "Topic identifier (lowercase, underscored, e.g., 'coffee', 'coding_style')",
                },
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of this preference (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview of the preference",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed preference content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["routing_key", "abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_entity",
        "description": (
            "Extract specific entities: people, places, organizations, objects. "
            "Call once per distinct entity mentioned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routing_key": {
                    "type": "string",
                    "description": "Entity identifier (normalized name, e.g., 'alice', 'coffee_shop_downtown')",
                },
                "abstract": {
                    "type": "string",
                    "description": "Brief description of the entity (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview of the entity",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed entity content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["routing_key", "abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_event",
        "description": (
            "Extract time-bounded events: meetings, visits, actions, occurrences. "
            "routing_key should include timestamp (e.g., 'visit_20250315'). "
            "Call once per distinct event."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routing_key": {
                    "type": "string",
                    "description": "Event identifier with timestamp (e.g., 'meeting_20250315_1430')",
                },
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of the event (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview of the event",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed event content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["routing_key", "abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_case",
        "description": (
            "Extract problem-solving cases: a complete problem-resolution pair with outcome. "
            "Must include: 1) The problem/challenge, 2) Steps taken to resolve, 3) Final outcome. "
            "routing_key should describe the problem type (e.g., 'debug_api_timeout', "
            "'fix_memory_leak', 'resolve_merge_conflict'). Call once per resolved issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routing_key": {
                    "type": "string",
                    "description": "Case identifier describing problem type (lowercase, underscored, e.g., 'debug_api_timeout', 'fix_memory_leak')",
                },
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of problem and solution (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview: Problem → Diagnosis → Solution → Outcome",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed case content with step-by-step resolution",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["routing_key", "abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_pattern",
        "description": (
            "Extract agent-learned patterns: 'I noticed that...', 'It seems that...' "
            "observations about user behavior or conversation patterns. "
            "Do NOT extract: complete reusable workflows with clear trigger conditions "
            "and success criteria — those belong to extract_skill."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "routing_key": {
                    "type": "string",
                    "description": "Pattern topic identifier (lowercase, underscored)",
                },
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of the pattern (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview of the pattern",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed pattern content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["routing_key", "abstract", "overview", "content", "confidence"],
        },
    },
    {
        "name": "extract_skill",
        "description": (
            "Extract reusable, structured workflows. Must satisfy ALL three criteria: "
            "1) Clear trigger condition (when to use), "
            "2) Specific step sequence (what to do), "
            "3) Verifiable completion criteria (how to know it's done). "
            "Examples: debugging protocol, code review checklist, error handling workflow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "Brief summary of the skill (≤200 chars)",
                },
                "overview": {
                    "type": "string",
                    "description": "Structured overview with trigger/steps/criteria",
                },
                "content": {
                    "type": "string",
                    "description": "Full detailed skill content",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                },
            },
            "required": ["abstract", "overview", "content", "confidence"],
        },
    },
]

# Map tool name to (category, owner_scope)
TOOL_TO_CATEGORY = {
    "extract_profile": ("profile", "user"),
    "extract_preference": ("preference", "user"),
    "extract_entity": ("entity", "user"),
    "extract_event": ("event", "user"),
    "extract_case": ("case", "agent"),
    "extract_pattern": ("pattern", "agent"),
    "extract_skill": ("skill", "agent"),
}


_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI agent.

Your task is to analyze conversation messages and extract candidate memories using the provided tools.

Guidelines:
- Be conservative: only extract when confident
- Use specific, descriptive routing keys
- Keep abstracts concise (≤200 characters)
- Set confidence scores appropriately:
  - 0.9+: Explicit, clear information
  - 0.7-0.9: Strongly implied but some ambiguity
  - 0.5-0.7: Reasonable inference
  - <0.5: Too uncertain, do not extract
- Profile: extract once per conversation at most
- Preferences: group by topic, one call per topic
- Entities: one call per distinct entity
- Events: time-bounded occurrences, include timestamps in routing_key
- Cases: problem-resolution pairs with clear outcome, must be RESOLVED issues
- Patterns vs Skills: Patterns are observations ("I noticed..."),
  Skills are actionable workflows with clear steps

Conversation history will be provided. Extract memories using the tools."""


class Extractor:
    """Tool-use based candidate memory extractor.

    Uses 7 extraction tools: profile/preference/entity/event/case/pattern/skill.
    Single LLM function-calling approach reduces token cost by ~75% compared to parallel calls.
    """

    def __init__(self, llm: LLM):
        """Initialize the Extractor.

        Args:
            llm: LLM instance with complete_with_tools() support
        """
        self._llm = llm

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Extract candidate memories from conversation messages.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # Step 1: Detect language
        language = self._detect_language(messages)

        # Step 2: Format conversation for LLM
        conversation = self._format_conversation(messages)

        # Step 3: Build prompt with language instruction
        prompt = (
            f"Please output all abstract/overview/content fields in {language}.\n\n"
            f"{_SYSTEM_PROMPT}\n\nConversation:\n{conversation}"
        )

        # Step 4: Call LLM with tools
        tool_calls = self._llm.complete_with_tools(
            prompt=prompt,
            tools=EXTRACTION_TOOLS,
            tool_choice="auto",
        )

        # Step 5: Convert tool calls to CandidateMemory
        candidates = []
        for call in tool_calls:
            candidate = self._to_candidate(call)
            if candidate and candidate.confidence >= 0.5:
                candidates.append(candidate)

        return candidates

    @staticmethod
    def _content_to_str(content) -> str:
        """Normalize message content to plain string.

        Handles both plain string and OpenClaw structured content
        (list of {"type": "text", "text": "..."} blocks).
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def _detect_language(self, messages: list[dict]) -> str:
        """Detect language from user messages.

        Returns:
            Language code: "zh-CN", "ja", "ko", "ru", or "en" (default)
        """
        import re

        # Extract user text for language detection
        user_text = " ".join(
            self._content_to_str(m.get("content", ""))
            for m in messages
            if m.get("role") == "user"
        )

        if not user_text.strip():
            return "en"

        total = len(re.findall(r"\S", user_text))
        if total == 0:
            return "en"

        # Japanese: check before Chinese (Hiragana/Katakana are unique to Japanese)
        if re.search(r'[\u3040-\u30ff]', user_text):
            return "ja"
        # Chinese: CJK ideographs
        if re.search(r'[\u4e00-\u9fff]', user_text):
            return "zh-CN"
        # Korean: need at least 2 characters and ≥10% of text
        ko = len(re.findall(r'[\uac00-\ud7af]', user_text))
        if ko >= 2 and ko / total >= 0.10:
            return "ko"
        # Russian: need at least 2 characters and ≥10% of text
        ru = len(re.findall(r'[\u0400-\u04ff]', user_text))
        if ru >= 2 and ru / total >= 0.10:
            return "ru"

        return "en"

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages into conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _to_candidate(self, tool_call: dict) -> CandidateMemory | None:
        """Convert tool call result to CandidateMemory.

        Args:
            tool_call: {"tool": str, "input": dict}

        Returns:
            CandidateMemory or None if conversion fails
        """
        tool_name = tool_call.get("tool", "")
        input_data = tool_call.get("input", {})

        # Look up category and owner_scope
        if tool_name not in TOOL_TO_CATEGORY:
            return None

        category, owner_scope = TOOL_TO_CATEGORY[tool_name]

        # Extract fields
        abstract = input_data.get("abstract", "")
        overview = input_data.get("overview", "")
        content = input_data.get("content", "")
        confidence = input_data.get("confidence", 0.5)

        # Determine routing_key
        routing_key = input_data.get("routing_key", "")
        if not routing_key:
            # Use category as default for profile and skill
            routing_key = category

        return CandidateMemory(
            category=category,
            owner_scope=owner_scope,
            routing_key=routing_key,
            abstract=abstract,
            overview=overview,
            content=content,
            confidence=confidence,
        )
