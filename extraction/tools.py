"""Tool-use based candidate extraction.

Uses OpenAI function calling to extract candidate memories from conversations.
Single LLM call with 7 tools: profile/preference/entity/event/case/pattern/skill.
"""

from typing import Any

from core.interfaces import LLM
from core.models import RequestContext, CandidateMemory

from extraction.tool_schemas import (
    get_tool_definitions,
    get_tool_to_category,
    parse_tool_call,
)

# Tool definitions for LLM function calling (generated from Pydantic models)
EXTRACTION_TOOLS = get_tool_definitions()

# Map tool name to (category, owner_scope) (generated from Pydantic models)
TOOL_TO_CATEGORY = get_tool_to_category()


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

        # Use parse_tool_call for validation and conversion
        result = parse_tool_call(tool_name, input_data)
        if result is None:
            return None

        # parse_tool_call returns (category, owner_scope, CandidateMemory)
        _, _, candidate = result
        return candidate
