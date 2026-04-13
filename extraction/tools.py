"""Two-phase candidate extraction.

Phase 1 — Span Identification: Cheap JSON-mode LLM call identifies which
user message ranges contain extractable information.

Phase 2 — Span Structuring: For each span, focused tool-call extraction
with +/- 3 turns context.  Parallelized across spans (max_workers=4).

Falls back to single-span mode when Phase 1 fails or returns no spans,
preserving backward compatibility with mock / test scenarios.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.interfaces import LLM
from core.models import RequestContext, CandidateMemory

from extraction.tool_schemas import (
    get_tool_definitions,
    get_tool_to_category,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

# Tool definitions for LLM function calling (generated from Pydantic models)
EXTRACTION_TOOLS = get_tool_definitions()

# Map tool name to (category, owner_scope) (generated from Pydantic models)
TOOL_TO_CATEGORY = get_tool_to_category()


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """A contiguous message range identified as containing extractable info."""
    start: int
    end: int
    reason: str = ""
    categories: list[str] = field(default_factory=list)


# JSON schema for Phase 1 span identification
_SPAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "reason": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end"],
            },
        },
    },
    "required": ["spans"],
}

# Phase 1 prompt — lightweight, no tool definitions
_SPAN_PROMPT = """\
Analyze this conversation and identify contiguous message ranges (spans) \
that contain personal information worth remembering.

Rules:
- ONLY identify spans from non-assistant messages. Ignore lines starting \
with "[N] assistant:".
- Each span should contain a coherent unit of extractable information.
- Merge adjacent user messages about the same topic into one span.
- Skip pure greetings, acknowledgments, or general knowledge.
- If a "Previously Extracted Context" section is present, DO NOT re-identify \
content already listed there.

For each span, specify:
- start: message index (the [N] prefix number)
- end: message index (inclusive)
- reason: what extractable information this span contains
- categories: likely categories from: profile, preference, entity, event, \
case, pattern, skill, tool

{session_summary_block}\
Conversation:
{conversation}

Return JSON: {{"spans": [{{"start": int, "end": int, "reason": str, \
"categories": [str]}}]}}
If nothing is extractable: {{"spans": []}}"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor:
    """Two-phase candidate memory extractor.

    Phase 1 uses a cheap JSON-mode LLM call to identify which user message
    ranges contain extractable information.  Phase 2 runs focused tool-call
    extraction per span with surrounding context, parallelized across spans.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_manager: Any | None = None,
    ):
        """Initialize the Extractor.

        Args:
            llm: LLM instance with complete_with_tools() and complete_json().
            prompt_manager: Optional PromptManager for template-based prompts.
                          If None, auto-creates one from default templates dir.
        """
        self._llm = llm
        if prompt_manager is None:
            from extraction.prompts import PromptManager as _PM
            prompt_manager = _PM()
        self._prompt_manager = prompt_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # No user content to extract from
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") != "assistant"
        ]
        if not user_indices:
            return []

        # Detect language once
        language = self._detect_language(messages)

        # Phase 1: identify spans worth extracting
        spans = self._identify_spans(messages, session_time, session_summary)
        if not spans:
            return []

        # Phase 2: structure each span
        candidates = self._structure_spans(
            spans, messages, language, session_time, session_summary, tool_stats_text,
        )

        return [c for c in candidates if c.confidence >= 0.5]

    # ------------------------------------------------------------------
    # Phase 1 — Span Identification
    # ------------------------------------------------------------------

    def _identify_spans(
        self,
        messages: list[dict],
        session_time: Any | None,
        session_summary: str,
    ) -> list[_Span]:
        """Use a cheap JSON-mode call to find extractable message ranges."""
        conversation = self._format_indexed_conversation(messages, session_time)

        summary_block = ""
        if session_summary:
            summary_block = (
                "Previously Extracted Context "
                "(DO NOT re-identify these — they are already saved):\n"
                f"{session_summary}\n\n"
            )

        prompt = _SPAN_PROMPT.format(
            conversation=conversation,
            session_summary_block=summary_block,
        )

        try:
            result = self._llm.complete_json(prompt, schema=_SPAN_SCHEMA)
            logger.info(
                "Phase 1 returned %d spans (keys=%s)",
                len(result.get("spans", [])),
                list(result.keys()),
            )
            spans: list[_Span] = []
            for s in result.get("spans", []):
                span = _Span(
                    start=max(0, int(s.get("start", 0))),
                    end=min(len(messages) - 1, int(s.get("end", 0))),
                    reason=s.get("reason", ""),
                    categories=s.get("categories", []),
                )
                if span.start <= span.end:
                    spans.append(span)
            if not spans:
                logger.warning("Phase 1 returned 0 valid spans for %d messages", len(messages))
            return spans
        except Exception as exc:
            logger.error("Phase 1 span identification FAILED: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Phase 2 — Span Structuring
    # ------------------------------------------------------------------

    def _structure_spans(
        self,
        spans: list[_Span],
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
    ) -> list[CandidateMemory]:
        """Run focused tool-call extraction for each span."""
        # Single span → skip thread pool overhead
        if len(spans) == 1:
            return self._structure_span(
                spans[0], messages, language,
                session_time, session_summary, tool_stats_text,
            )

        all_candidates: list[CandidateMemory] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    self._structure_span, span, messages,
                    language, session_time, session_summary, tool_stats_text,
                ): span
                for span in spans
            }
            for future in as_completed(futures):
                span = futures[future]
                try:
                    all_candidates.extend(future.result())
                except Exception as exc:
                    logger.error(
                        "Span %d-%d structuring FAILED: %s",
                        span.start, span.end, exc, exc_info=True,
                    )
                    raise
        return all_candidates

    def _structure_span(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
    ) -> list[CandidateMemory]:
        """Extract structured candidates from a single span."""
        # Focused context: +/- 3 turns around the span
        ctx_start = max(0, span.start - 3)
        ctx_end = min(len(messages) - 1, span.end + 3)
        focused = messages[ctx_start: ctx_end + 1]

        conversation = self._format_conversation(
            focused, session_time,
            session_summary="",
            tool_stats_text=(
                tool_stats_text if span.end >= len(messages) - 3 else ""
            ),
        )
        prompt = self._build_prompt(language, conversation, session_summary)

        tool_calls = self._llm.complete_with_tools(
            prompt=prompt,
            tools=EXTRACTION_TOOLS,
            tool_choice="required",
        )

        candidates: list[CandidateMemory] = []
        for call in tool_calls:
            c = self._to_candidate(call)
            if c is not None:
                candidates.append(c)
        return candidates

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        language: str,
        conversation: str,
        session_summary: str = "",
    ) -> str:
        """Build extraction prompt from YAML template."""
        system_prompt = self._prompt_manager.render(
            "extraction", "system_prompt",
            session_summary=session_summary,
        )
        examples = self._prompt_manager.render("extraction", "examples")
        output_instruction = self._prompt_manager.render(
            "extraction", "output_instruction",
            output_language=language,
        )
        return (
            f"{system_prompt}\n\n{examples}\n\n"
            f"{output_instruction}\n\nConversation:\n{conversation}"
        )

    @staticmethod
    def _content_to_str(content) -> str:
        """Normalize message content to plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def _detect_language(self, messages: list[dict]) -> str:
        """Detect language from user messages."""
        from core.language import detect_language

        user_text = " ".join(
            self._content_to_str(m.get("content", ""))
            for m in messages
            if m.get("role") == "user"
        )
        return detect_language(user_text)

    def _format_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> str:
        """Format messages into conversation text with session time header."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"**Session Time:** {time_str} ({day_of_week})"]
        lines.append(
            "Relative times (e.g., 'last week', 'next month') are based "
            "on Session Time, not today."
        )

        if session_summary:
            lines.append("")
            lines.append(
                "## Previously Extracted Context "
                "(DO NOT re-extract these — they are already saved)"
            )
            lines.append(session_summary)

        if tool_stats_text:
            lines.append("")
            lines.append(tool_stats_text)

        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(
                    f"assistant [do not extract from this]: {content}"
                )
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_indexed_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
    ) -> str:
        """Format messages with [N] index prefixes for span identification."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"Session Time: {time_str} ({day_of_week})"]
        lines.append("")

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(f"[{i}] assistant: {content}")
            else:
                lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)

    def _to_candidate(self, tool_call: dict) -> CandidateMemory | None:
        """Convert tool call result to CandidateMemory."""
        tool_name = tool_call.get("tool", "")
        input_data = tool_call.get("input", {})
        result = parse_tool_call(tool_name, input_data)
        if result is None:
            return None
        _, _, candidate = result
        return candidate
