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

**TEMPORAL PRECISION — CRITICAL:**
- ALWAYS convert relative times to ABSOLUTE dates using the Session Time
- "last week" (session: June 9, 2023) → "June 2-8, 2023" (NOT "last week (relative to June 9)")
- "yesterday" (session: May 8) → "May 7, 2023" (NOT "yesterday")
- "the friday before 15 July 2023" → "July 14, 2023"
- DO NOT store unresolved relative expressions like "last week (relative to X)" — COMPUTE the date
- If truly unresolvable (no session time, vague reference): preserve original "summer 2023", "around 5pm"
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
**Template** — Use function calls with specific fields:
- routing_key: lowercase_underscored (events: include date stamp)
- abstract: Brief summary (≤50 chars) — WHAT happened
- overview: Structured ## sections (see rules above)
- content: Verbatim quotes or evidence
- when/who: Temporal/actor context (when available)
- confidence: 0.5-1.0 (default 0.9)

**Positive Examples:**
✅ Event (temporal): "I went to a LGBTQ support group yesterday"
→ extract_event(routing_key="lgbtq_20230507", abstract="Attended LGBTQ support group meeting", when="7 May 2023", who="Caroline", confidence=0.9)

✅ Entity (distinguishing): "Melanie showed me a painting with pink sky inspired by sunsets"
→ extract_entity(routing_key="painting_sunset_pink_sky", abstract="Sunset painting with pink sky", who="Melanie", confidence=0.95)

✅ Preference (with evidence): "I prefer quiet coffee shops over noisy ones"
→ extract_preference(routing_key="quiet_coffee_shops", abstract="Prefers quiet coffee shops", overview="## Evidence\\n'I prefer quiet coffee shops, not those noisy ones'")

✅ Profile: "I'm Caroline, 28, software engineer in Portland"
→ extract_profile(abstract="Caroline: 28-year-old software engineer in Portland", overview="## Name\\nCaroline\\n## Age\\n28\\n## Occupation\\nSoftware engineer")

✅ Multi-hop event: "My son had a car accident during our road trip. He was scared but reassured."
→ extract_event(routing_key="car_accident_202310", abstract="Son in car accident during family road trip", when="October 2023", who="Son, family", confidence=0.9)

**Anti-Patterns:**
❌ Vague abstract: "User info" → ✅ "28-year-old software engineer in Portland"
❌ Mixed facets: abstract="User preferences: coffee, commute, tools" (unmergeable) → ✅ Split into 3 separate extract_preference calls
❌ Missing details: abstract="Sunset painting" → ✅ "Sunset painting with pink sky" (keep distinguishing features)
❌ Don't extract: General knowledge ("Water boils at 100°C") or transient chatter ("hello", "thanks")
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
        session_time: Any | None = None,
    ) -> list[CandidateMemory]:
        """Extract candidate memories from conversation messages.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            session_time: Optional datetime for temporal resolution (defaults to now)

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # Step 1: Detect language
        language = self._detect_language(messages)

        # Step 2: Format conversation for LLM
        conversation = self._format_conversation(messages, session_time)

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

    def _format_conversation(self, messages: list[dict], session_time: Any | None = None) -> str:
        """Format messages into conversation text with session time header.

        Args:
            messages: List of message dicts
            session_time: Optional datetime for temporal resolution (defaults to now)
        """
        from datetime import datetime as dt

        if session_time is None:
            session_time = dt.now()

        # Format session time header (OpenViking-style)
        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"**Session Time:** {time_str} ({day_of_week})"]
        lines.append("Relative times (e.g., 'last week', 'next month') are based on Session Time, not today.")
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
