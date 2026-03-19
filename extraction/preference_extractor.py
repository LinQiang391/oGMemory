"""Preference extraction from conversation history.

Extracts user preferences, opinions, and tastes.
These are more specific than profile and can change over time.
"""

from typing import Any

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor, LLM


class PreferenceExtractor(CandidateExtractor):
    """Extract user preferences from messages.

    Preference memories are user opinions and tastes:
    - Food and drink preferences
    - Entertainment preferences
    - Work style preferences
    - Communication preferences
    - Any "I like/dislike X" statements

    Returns CandidateMemory with category="preference".
    Routing key is the normalized preference topic (e.g., "coffee", "music").
    """

    def __init__(self, llm: LLM):
        """Initialize PreferenceExtractor.

        Args:
            llm: LLM instance for structured extraction
        """
        self._llm = llm

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract preferences from conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction

        Returns:
            List of CandidateMemory with preference candidates
        """
        conversation = self._format_conversation(messages)
        prompt = f"""Extract user preferences from this conversation:

{conversation}

Extract preferences such as:
- Food/drink preferences (e.g., coffee, dietary restrictions)
- Entertainment preferences (e.g., music genres, movies)
- Work/communication preferences
- Any explicit likes/dislikes

Return a JSON response with this schema:
{{
    "candidates": [
        {{
            "routing_key": "preference_topic_lowercase",
            "abstract": "Brief summary (≤100 chars)",
            "overview": "Detailed preference explanation",
            "content": "Full preference with context",
            "confidence": 0.0-1.0
        }}
    ]
}}

Only extract if there's actual preference information. Use lowercase underscore format for routing_key (e.g., "dark_coffee", "classical_music").
"""

        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "routing_key": {"type": "string"},
                            "abstract": {"type": "string"},
                            "overview": {"type": "string"},
                            "content": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                    },
                },
            },
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            candidates = []

            for c in result.get("candidates", []):
                if c.get("confidence", 0) < 0.5:
                    continue

                candidates.append(CandidateMemory(
                    category="preference",
                    owner_scope="user",
                    routing_key=c["routing_key"],
                    abstract=c["abstract"],
                    overview=c["overview"],
                    content=c["content"],
                    confidence=c["confidence"],
                ))

            return candidates

        except Exception:
            return []

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages for LLM prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
