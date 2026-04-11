"""Tool-use based candidate extraction.

Uses OpenAI function calling to extract candidate memories from conversations.
Single LLM call with 8 tools: profile/preference/entity/event/case/pattern/skill/tool.
Supports optional Jinja2 YAML template system for prompt management.
"""

from __future__ import annotations

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


class Extractor:
    """Tool-use based candidate memory extractor.

    Uses 8 extraction tools: profile/preference/entity/event/case/pattern/skill/tool.
    Single LLM function-calling approach reduces token cost by ~75% compared to parallel calls.
    Prompts are loaded from YAML templates via PromptManager.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_manager: PromptManager | None = None,
    ):
        """Initialize the Extractor.

        Args:
            llm: LLM instance with complete_with_tools() support
            prompt_manager: Optional PromptManager for template-based prompts.
                          If None, auto-creates one from default templates dir.
        """
        self._llm = llm
        if prompt_manager is None:
            from extraction.prompts import PromptManager as _PM
            prompt_manager = _PM()
        self._prompt_manager = prompt_manager

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> list[CandidateMemory]:
        """Extract candidate memories from conversation messages.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            session_time: Optional datetime for temporal resolution (defaults to now)
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text for tool category

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # Step 1: Detect language
        language = self._detect_language(messages)

        # Step 2: Format conversation
        # session_summary is rendered by PromptManager in system_prompt,
        # so skip it here to avoid double injection
        conversation = self._format_conversation(
            messages, session_time,
            session_summary="",
            tool_stats_text=tool_stats_text,
        )

        # Step 3: Build prompt from YAML template
        prompt = self._build_prompt(language, conversation, session_summary)

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

    def _build_prompt(
        self,
        language: str,
        conversation: str,
        session_summary: str = "",
    ) -> str:
        """Build extraction prompt from YAML template.

        Args:
            language: Output language code
            conversation: Formatted conversation text
            session_summary: Optional summary for incremental extraction

        Returns:
            Complete prompt string for LLM
        """
        system_prompt = self._prompt_manager.render(
            "extraction", "system_prompt",
            session_summary=session_summary,
        )
        examples = self._prompt_manager.render(
            "extraction", "examples",
        )
        output_instruction = self._prompt_manager.render(
            "extraction", "output_instruction",
            output_language=language,
        )
        return f"{system_prompt}\n\n{examples}\n\n{output_instruction}\n\nConversation:\n{conversation}"

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

    def _format_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> str:
        """Format messages into conversation text with session time header.

        Args:
            messages: List of message dicts
            session_time: Optional datetime for temporal resolution (defaults to now)
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text
        """
        from datetime import datetime as dt

        if session_time is None:
            session_time = dt.now()

        # Format session time header
        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"**Session Time:** {time_str} ({day_of_week})"]
        lines.append("Relative times (e.g., 'last week', 'next month') are based on Session Time, not today.")

        # Inject summary of previously extracted content (incremental extraction)
        if session_summary:
            lines.append("")
            lines.append("## Previously Extracted Context (DO NOT re-extract these — they are already saved)")
            lines.append(session_summary)

        # Inject tool usage statistics (tool category extraction)
        if tool_stats_text:
            lines.append("")
            lines.append(tool_stats_text)

        lines.append("")

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
