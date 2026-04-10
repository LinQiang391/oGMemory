"""Tool-use based candidate extraction.

Uses OpenAI function calling to extract candidate memories from conversations.
Single LLM call with 8 tools: profile/preference/entity/event/case/pattern/skill/tool.
Supports optional Jinja2 YAML template system for prompt management.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from core.interfaces import LLM
from core.models import RequestContext, CandidateMemory

from extraction.tool_schemas import (
    get_tool_definitions,
    get_tool_to_category,
    parse_tool_call,
)

if TYPE_CHECKING:
    from extraction.prompts import PromptManager

# Tool definitions for LLM function calling (generated from Pydantic models)
EXTRACTION_TOOLS = get_tool_definitions()

# Map tool name to (category, owner_scope) (generated from Pydantic models)
TOOL_TO_CATEGORY = get_tool_to_category()


_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI agent.

Your task is to analyze the ENTIRE conversation and extract candidate memories using the provided tools.

**STRICT FACTUAL ACCURACY — MOST IMPORTANT:**
- NEVER infer, guess, or fabricate facts not explicitly stated in the conversation
- If someone says "I study {{FIELD_A}}", extract EXACTLY "{{FIELD_A}}" — do NOT change it to "{{FIELD_B}}" or any other field
- If someone says "I live in {{CITY_A}}", extract EXACTLY "{{CITY_A}}" — do NOT assume {{CITY_B}}
- Copy proper nouns, occupation names, field names, and specific details VERBATIM
- When in doubt, quote the exact words from the conversation rather than paraphrasing
- WRONG: conversation says one field → you extract a different field = HALLUCINATION
- RIGHT: conversation says X → you extract X exactly = CORRECT

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

**CATEGORY CLASSIFICATION — Core Decision Logic:**

| Question | Answer | Category |
|----------|--------|----------|
| Who is the user? | Identity, attributes | profile |
| What does the user prefer? | Preferences, habits | preference |
| What is this thing? | Person, project, organization | entity |
| What happened? | Decision, milestone, occurrence | event |
| How was it solved? | Problem + solution | case |
| What is the process? | Reusable steps | pattern |
| How to execute? | Workflow, strategy | skill |

**Precise Definition of Each Category:**
- **profile** — Describes "who the user is". Relatively stable personal attributes.
- **preference** — Describes "user tends to / habits". Changeable choices, styles.
  - Each memory = ONE independently updatable facet (single topic).
  - Do NOT mix unrelated facets in one call. Split food/commute/tools into separate calls.
  - If a new facet appears, create a new call instead of forcing into existing ones.
- **entity** — Describes "what is this named thing and its attributes". People, projects, organizations, objects.
- **event** — Describes "what happened". Has a time dimension. Includes past decisions, completed/ongoing/planned activities.
  - "Currently reading book X" → event (ongoing time-bound activity), NOT preference.
  - "Plan to do X tonight" → event (has time), NOT preference.
- **case** — Describes a specific problem AND its diagnosis/cause/solution/outcome. Both parts required.
- **pattern** — Describes "what process to follow in what situation". Reusable across scenarios.
- **skill** — Reusable workflows with trigger condition, step sequence, and completion criteria.

**Common Confusion Clarification:**
- "Plan to do X" → event (action, not entity)
- "User prefers X" → preference (not profile)
- "Encountered problem A, used solution B" → case (not event)
- "General process for handling X" → pattern (not case)
- "Currently reading/learning/studying X" → event (ongoing activity), NOT preference
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
# Few-shot Examples (using placeholders — substitute mentally for real values)

## profile Example
Conversation: "I'm {{NAME}}, {{AGE}}, {{OCCUPATION}} in {{LOCATION}}."

✅ Good — call extract_profile:
  abstract: "{{NAME}}: {{AGE}}-year-old {{OCCUPATION}} in {{LOCATION}}"
  overview: "## Background Info\\n- Name: {{NAME}}\\n- Age: {{AGE}}\\n- Occupation: {{OCCUPATION}}\\n- Location: {{LOCATION}}"
  content: "{{NAME}} is a {{AGE}}-year-old {{OCCUPATION}} based in {{LOCATION}}."
  confidence: 0.95

❌ Bad: abstract="User info" (too vague)
❌ Bad: abstract="{{OCCUPATION}}" (loses name, age, location)

## preferences Example — Granularity (CRITICAL)
Conversation: "I usually drink {{BEVERAGE}} in the morning, commute by {{TRANSPORT}}, and use {{TOOL}} for notes."

❌ Bad — ONE call mixing unrelated facets:
  abstract: "User preferences: {{BEVERAGE}}, {{TRANSPORT}} commute, {{TOOL}} notes"

✅ Good — THREE separate extract_preference calls:

Call 1 — extract_preference:
  routing_key: "beverage"
  abstract: "Beverage preference: Drinks {{BEVERAGE}} in the morning"
  overview: "## Preference Domain\\n- **Domain**: Beverage\\n\\n## Specific Preference\\n- Drinks {{BEVERAGE}} in the morning"
  content: "User habitually drinks {{BEVERAGE}} in the morning."
  confidence: 0.9

Call 2 — extract_preference:
  routing_key: "commute"
  abstract: "Commute preference: Uses {{TRANSPORT}}"
  overview: "## Preference Domain\\n- **Domain**: Commute\\n\\n## Specific Preference\\n- Commutes by {{TRANSPORT}}"
  content: "User commutes by {{TRANSPORT}}."
  confidence: 0.9

Call 3 — extract_preference:
  routing_key: "note_taking"
  abstract: "Note-taking preference: Uses {{TOOL}}"
  overview: "## Preference Domain\\n- **Domain**: Productivity tools\\n\\n## Specific Preference\\n- Prefers {{TOOL}} for note-taking"
  content: "User prefers {{TOOL}} for taking notes."
  confidence: 0.9

## entity Example
Conversation: "{{PERSON}} showed me a hand-thrown {{OBJECT}} with {{DISTINGUISHING_FEATURE}}, inspired by {{INSPIRATION}}. Sells at {{MARKET}}."

✅ Good — call extract_entity:
  routing_key: "{{OBJECT_TYPE}}"
  abstract: "{{PERSON}}'s {{OBJECT}}: {{DISTINGUISHING_FEATURE}}, inspired by {{INSPIRATION}}"
  overview: "## Basic Info\\n- **Creator**: {{PERSON}}\\n- **Object**: {{OBJECT}}\\n\\n## Key Attributes\\n- {{DISTINGUISHING_FEATURE}}\\n- Inspired by {{INSPIRATION}}\\n\\n## Related Facts\\n- Sells at {{MARKET}}"
  content: "{{PERSON}} makes {{OBJECT}} with {{DISTINGUISHING_FEATURE}} inspired by {{INSPIRATION}}. Sells at {{MARKET}}."
  who: "{{PERSON}}"
  where: "{{MARKET}}"
  confidence: 0.95

❌ Bad: abstract="{{OBJECT}}" (loses distinguishing features)
❌ Bad: abstract="{{PERSON}}'s art" (loses specific medium and detail)

## events Example
Conversation: "We decided to move the {{MEETING}} from {{OLD_TIME}} to {{NEW_TIME}} starting {{START_DATE}}, because {{REASON}}."
Session Time: {{SESSION_TIME}}

✅ Good — call extract_event:
  routing_key: "{{EVENT_TYPE}}_{{DATE_STAMP}}"
  abstract: "{{MEETING}} rescheduled from {{OLD_TIME}} to {{NEW_TIME}} starting {{START_DATE}}"
  overview: "## Decision Content\\nReschedule {{MEETING}} from {{OLD_TIME}} to {{NEW_TIME}}\\n\\n## Reason\\n{{REASON}}\\n\\n## Result\\nNew time takes effect {{START_DATE}}"
  content: "Team decided to move {{MEETING}} from {{OLD_TIME}} to {{NEW_TIME}} starting {{START_DATE}}. Reason: {{REASON}}."
  when: "{{START_DATE}}"
  who: "{{PARTICIPANTS}}"
  confidence: 0.95

❌ Bad: abstract="Meeting changed" (loses specific time, date, reason)
❌ Bad: when="next Monday" (MUST resolve to absolute date)

## cases Example
Conversation: "The {{SYSTEM}} kept returning {{ERROR}}. Turns out {{ROOT_CAUSE}}. After {{FIX}}, the errors stopped completely."

✅ Good — call extract_case:
  routing_key: "{{CASE_TYPE}}"
  abstract: "{{SYSTEM}} {{ERROR}} → {{ROOT_CAUSE}}, fixed by {{FIX}}"
  overview: "## Problem\\n{{SYSTEM}} returning {{ERROR}}\\n\\n## Root Cause\\n{{ROOT_CAUSE}}\\n\\n## Solution\\n{{FIX}}\\n\\n## Outcome\\nErrors stopped completely"
  content: "{{SYSTEM}} was returning {{ERROR}}. Root cause: {{ROOT_CAUSE}}. After {{FIX}}, all errors stopped."
  confidence: 0.95

❌ Bad: abstract="System error" (no cause or solution — useless as a case)

## patterns Example
Conversation: "I noticed that {{OBSERVATION}}. Been tracking this for {{DURATION}}."

✅ Good — call extract_pattern:
  routing_key: "{{PATTERN_TOPIC}}"
  abstract: "{{PATTERN_TOPIC}} pattern: {{OBSERVATION_SUMMARY}}"
  overview: "## Observation\\n{{OBSERVATION}}\\n\\n## Frequency\\nTracked over {{DURATION}}\\n\\n## Context\\n{{CONTEXT}}"
  content: "User observed over {{DURATION}}: {{OBSERVATION}}."
  confidence: 0.85

## Anti-Patterns Summary
❌ Vague abstract: "User info" → ✅ Specific: "{{AGE}}-year-old {{OCCUPATION}} in {{LOCATION}}"
❌ Mixed facets in one preference → ✅ Split into separate calls per facet
❌ Missing distinguishing details → ✅ Keep all specific features
❌ Relative time in events → ✅ Resolve to absolute date using Session Time
❌ Incomplete case → ✅ Must include problem + cause/solution
❌ General knowledge or transient chatter → ✅ Only extract personalized information
"""


class Extractor:
    """Tool-use based candidate memory extractor.

    Uses 8 extraction tools: profile/preference/entity/event/case/pattern/skill/tool.
    Single LLM function-calling approach reduces token cost by ~75% compared to parallel calls.
    Supports optional PromptManager for Jinja2 YAML template-based prompts.
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
                          If None, uses hardcoded Python string constants.
        """
        self._llm = llm
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

        # Step 2: Format conversation for LLM
        conversation = self._format_conversation(
            messages, session_time,
            session_summary=session_summary,
            tool_stats_text=tool_stats_text,
        )

        # Step 3: Build prompt (template-based or hardcoded fallback)
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
        """Build extraction prompt using template or hardcoded fallback.

        Args:
            language: Output language code
            conversation: Formatted conversation text
            session_summary: Optional summary for incremental extraction

        Returns:
            Complete prompt string for LLM
        """
        if self._prompt_manager and self._prompt_manager.has_template("extraction"):
            # Template-based mode
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
            return f"{output_instruction}\n\n{system_prompt}\n\n{examples}\n\nConversation:\n{conversation}"
        else:
            # Hardcoded fallback (original behavior)
            return (
                f"Please output all abstract/overview/content fields in {language}.\n\n"
                f"{_SYSTEM_PROMPT}\n\n"
                f"{_EXTRACTION_EXAMPLES}\n\n"
                f"Conversation:\n{conversation}"
            )

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
