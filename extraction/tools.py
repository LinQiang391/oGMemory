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

Your task is to analyze the ENTIRE conversation and extract candidate memories using the provided tools.

**HIGH RECALL STRATEGY:**
- Extract liberally: missing information is worse than redundancy
- When uncertain whether something is worth extracting, extract it
- Ambiguous content → extract with lower confidence (0.5-0.7) rather than skipping
- One tool call per independent fact — do NOT merge unrelated facts
- Read the ENTIRE conversation before extracting — do not ignore later turns

**TEMPORAL PRECISION:**
- Convert relative time expressions to absolute dates when context allows
- Use the conversation timestamp to resolve: "yesterday" (spoken May 8) → "May 7, 2023"
- "the friday before 15 July" → compute the actual date if possible
- If no context to resolve: preserve the original expression verbatim
- ALWAYS include time context for events — even approximate ("summer 2023", "around 5pm on July 3")

**DETAIL PRESERVATION:**
- Always preserve specific proper nouns, numeric values, colors, quotes verbatim
- "went camping July 6-8 at Yellowstone" is useful; "went camping" is NOT
- "painting with pink sky" captures the feature; "painting" loses it
- Include distinguishing features, exact names, and specific descriptions

**CONFIDENCE SCORING:**
- 0.9+: Explicit, directly stated information
- 0.7-0.9: Strongly implied with clear evidence
- 0.5-0.7: Reasonable inference or partial information — STILL EXTRACT
- <0.5: Too uncertain, do not extract

**ROUTING KEYS:**
- Use specific, descriptive identifiers (lowercase, underscored)
- Events: include date stamp (e.g., 'meeting_20250315', 'road_trip_202310')
- Entities: use normalized name (e.g., 'alice', 'coffee_shop_downtown')
- Preferences: topic-based (e.g., 'coffee', 'coding_style')
- Cases: problem type (e.g., 'debug_api_timeout')

**INCLUSION CRITERIA — EXTRACT:**
- Personal info: name, age, location, occupation, relationships
- Preferences: likes, dislikes, habits, routines
- Events: specific occurrences with time/place/people
- Entities: people, organizations, objects with distinguishing features
- Cases: problem-resolution pairs with clear outcome
- Patterns: observations ("I noticed...", "It seems that...")
- Skills: actionable workflows with clear steps

**EXCLUSION CRITERIA — DO NOT EXTRACT:**
- General knowledge: "Water boils at 100°C"
- Common facts: "Paris is the capital of France"
- Transient chatter: "hello", "how are you", "okay"
- Technical docs: verbatim API reference unless personalized

**CATEGORY GUIDELINES:**
- Profile: extract once per conversation at most — stable self-description
- Preference: one call per distinct topic — group by theme
- Entity: one call per distinct entity — include distinguishing features!
- Event: time-bounded occurrences — ALWAYS include time context
- Case: problem + resolution with outcome — must be RESOLVED
- Pattern: observations and trends — NOT actionable workflows
- Skill: reusable workflows with trigger, steps, and completion criteria

**CATEGORY CONFUSION RULES:**
- Event vs Case: Event = occurrence; Case = problem + solution
- Preference vs Pattern: Preference = stated liking; Pattern = observed trend
- Entity vs Profile: Entity = external thing; Profile = self-description
- Multiple related facts → separate extractions, link via routing_key

**OVERVIEW FIELD STRUCTURE (Markdown templates):**
- Event: ## What happened / ## When / ## Who / ## Details
- Entity: ## Name / ## Key attributes / ## Related facts / ## Distinguishing features
- Preference: ## Topic / ## Details / ## Evidence
- Case: ## Problem / ## Solution / ## Outcome
- Pattern: ## Observation / ## Frequency / ## Context
- Profile: ## Name / ## Key attributes / ## Background / ## Current status
- Skill: ## Trigger / ## Steps / ## Completion criteria

Conversation history will be provided. Extract memories using the tools."""

_EXTRACTION_EXAMPLES = """
**EXAMPLES — Category-Specific Extraction:**

---
Example 1: Event with temporal expression
Input: "I went to a LGBTQ support group yesterday"
Context: Timestamp shows "1:56 pm on 8 May, 2023"
Extract: extract_event(
  routing_key="lgbtq_support_group_20230507",
  abstract="Attended LGBTQ support group meeting",
  overview="## What happened\\nCaroline attended an LGBTQ support group meeting.\\n## When\\n7 May 2023 (yesterday relative to May 8)\\n## Who\\nCaroline\\n## Details\\nDescribed it as powerful and meaningful.",
  content="Caroline: 'I went to a LGBTQ support group yesterday and it was so powerful.' (Spoken at 1:56 pm on 8 May, 2023, so event date is 7 May 2023)",
  when="7 May 2023",
  who="Caroline",
  confidence=0.9
)

---
Example 2: Entity with distinguishing details
Input: "I showed Caroline a painting inspired by sunsets with a pink sky"
Extract: extract_entity(
  routing_key="painting_sunset_pink_sky",
  abstract="Sunset painting with pink sky",
  overview="## Name\\nSunset painting\\n## Key attributes\\nInspired by sunsets, pink sky\\n## Distinguishing features\\nPink sky is the distinctive visual element",
  content="Melanie shared a painting she created, described as inspired by sunsets with a pink sky",
  who="Melanie (creator), Caroline (viewer)",
  confidence=0.95
)

---
Example 3: Multi-hop event chain
Input: "My son had a car accident during our road trip. He was scared but we reassured him."
Extract: extract_event(
  routing_key="car_accident_road_trip_202310",
  abstract="Son involved in car accident during family road trip",
  overview="## What happened\\nDuring a family road trip, the son was in a car accident.\\n## When\\nOctober 2023\\n## Who\\nSon, family\\n## Details\\nHe was scared but family reassured him. He was okay.",
  content="Melanie: 'My son had a car accident during our family road trip. It was scary but he was okay. He was scared but we reassured him.'",
  when="October 2023",
  who="Melanie's son, Melanie, family",
  confidence=0.9
)

---
Example 4: Preference with evidence
Input: "I really prefer quiet coffee shops, not those noisy ones"
Extract: extract_preference(
  routing_key="quiet_coffee_shops",
  abstract="Prefers quiet coffee shops over noisy ones",
  overview="## Topic\\nCoffee shop environment preference\\n## Details\\nPrefers quiet atmosphere, dislikes noise\\n## Evidence\\n'I really prefer quiet coffee shops, not those noisy ones'",
  content="Caroline expressed preference for quiet coffee shops, explicitly dislikes noisy ones",
  confidence=0.95
)

---
Example 5: Profile (self-description)
Input: "I'm Caroline, 28 years old, living in Portland, working as a software engineer"
Extract: extract_profile(
  abstract="28-year-old software engineer in Portland",
  overview="## Name\\nCaroline\\n## Age\\n28\\n## Location\\nPortland\\n## Occupation\\nSoftware engineer",
  content="Caroline is 28 years old, lives in Portland, and works as a software engineer.",
  confidence=0.95
)

---
Example 6: Pattern (observed trend)
Input: "I've noticed I tend to get more creative ideas in the morning"
Extract: extract_pattern(
  routing_key="morning_creativity_pattern",
  abstract="Observed pattern: more creative ideas in the morning",
  overview="## Observation\\nTends to get more creative ideas in the morning\\n## Frequency\\nOngoing pattern\\n## Context\\nPersonal productivity observation",
  content="Observed that she tends to have more creative ideas in the morning hours.",
  confidence=0.8
)

---
Example 7: Skill (actionable workflow)
Input: "To debug API timeouts, I first check the logs, then increase timeout, and finally add retry logic"
Extract: extract_skill(
  abstract="Debugging API timeouts: 3-step workflow",
  overview="## Trigger\\nAPI timeout errors occur\\n## Steps\\n1. Check logs for timeout details\\n2. Increase timeout duration\\n3. Add retry logic with exponential backoff\\n## Completion criteria\\nAPI calls succeed without timeout",
  content="Workflow for debugging API timeout issues: 1) Check logs to identify timeout patterns, 2) Increase the timeout duration if appropriate, 3) Implement retry logic with exponential backoff.",
  confidence=0.9
)

---
Example 8: Case (problem + solution)
Input: "The API kept timing out. I fixed it by increasing the timeout to 30 seconds."
Extract: extract_case(
  routing_key="api_timeout_fix",
  abstract="Fixed API timeout issue by increasing timeout duration",
  overview="## Problem\\nAPI calls were timing out\\n## Solution\\nIncreased timeout to 30 seconds\\n## Outcome\\nIssue resolved",
  content="User encountered API timeout errors. Fixed by increasing timeout parameter to 30 seconds.",
  confidence=0.9
)
"""


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
            f"{_SYSTEM_PROMPT}\n\n"
            f"{_EXTRACTION_EXAMPLES}\n\n"
            f"Conversation:\n{conversation}"
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
