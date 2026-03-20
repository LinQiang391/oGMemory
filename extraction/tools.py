"""Tool-use based candidate extraction.

Uses OpenAI function calling to extract candidate memories from conversations.
Single LLM call with 6 tools instead of 4 parallel extractors.
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
                    "description": "Brief summary of who the user is (≤100 chars)",
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
                    "description": "Brief summary of this preference (≤100 chars)",
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
                    "description": "Brief description of the entity (≤100 chars)",
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
                    "description": "Brief summary of the event (≤100 chars)",
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
                    "description": "Brief summary of the pattern (≤100 chars)",
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
                    "description": "Brief summary of the skill (≤100 chars)",
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
    "extract_pattern": ("pattern", "agent"),
    "extract_skill": ("skill", "agent"),
}


_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI agent.

Your task is to analyze conversation messages and extract candidate memories using the provided tools.

Guidelines:
- Be conservative: only extract when confident
- Use specific, descriptive routing keys
- Keep abstracts concise (≤100 characters)
- Set confidence scores appropriately:
  - 0.9+: Explicit, clear information
  - 0.7-0.9: Strongly implied but some ambiguity
  - 0.5-0.7: Reasonable inference
  - <0.5: Too uncertain, do not extract
- Profile: extract once per conversation at most
- Preferences: group by topic, one call per topic
- Entities: one call per distinct entity
- Events: include timestamps in routing_key
- Patterns vs Skills: Patterns are observations ("I noticed..."),
  Skills are actionable workflows with clear steps

Conversation history will be provided. Extract memories using the tools."""


class Extractor:
    """Tool-use based candidate memory extractor.

    Replaces 4 parallel extractors with single LLM function-calling approach.
    Reduces token cost by ~75% compared to parallel calls.
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

        # Step 3: Build prompt with language hint
        prompt = f"""分析以下对话，提取所有值得长期记忆的信息。
请用 {language} 语言输出所有记忆内容。

{conversation}

对每类信息调用对应工具。同一类型多条时每条单独调用一次。
没有的类型不要调用工具。"""

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

    def _detect_language(self, messages: list[dict]) -> str:
        """Detect language from user messages.

        Returns:
            Language code: "zh-CN", "ja", "ko", "ru", or "en" (default)
        """
        import re

        # Extract user text for language detection
        user_text = " ".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
        )

        if not user_text:
            return "en"

        # Check for Japanese Hiragana/Katakana (unique to Japanese)
        # Must check before Chinese since Japanese also uses Kanji (CJK ideographs)
        if re.search(r'[\u3040-\u30ff]', user_text):
            return "ja"
        # Check for Chinese characters (CJK unified ideographs)
        if re.search(r'[\u4e00-\u9fff]', user_text):
            return "zh-CN"
        # Check for Korean Hangul
        if re.search(r'[\uac00-\ud7af]', user_text):
            return "ko"
        # Check for Cyrillic (Russian)
        if re.search(r'[\u0400-\u04ff]', user_text):
            return "ru"

        return "en"

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages into conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
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
