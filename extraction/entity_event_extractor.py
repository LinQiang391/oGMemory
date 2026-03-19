"""Entity and Event extraction from conversation history.

Extracts entities (people, places, things) and events (time-bounded occurrences).
"""

from typing import Any

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor, LLM


class EntityEventExtractor(CandidateExtractor):
    """Extract entities and events from messages.

    Entity memories:
    - People (names, roles)
    - Places (locations, venues)
    - Organizations
    - Objects of importance

    Event memories:
    - Time-bounded occurrences
    - Meetings, visits, activities
    - Timestamped events

    Returns CandidateMemory with category="entity" or "event".
    """

    def __init__(self, llm: LLM):
        """Initialize EntityEventExtractor.

        Args:
            llm: LLM instance for structured extraction
        """
        self._llm = llm

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract entities and events from conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction

        Returns:
            List of CandidateMemory with entity and event candidates
        """
        conversation = self._format_conversation(messages)
        prompt = f"""Extract entities and events from this conversation:

{conversation}

Extract:
ENTITIES (people, places, organizations):
- routing_key: normalized name (lowercase with underscores, e.g., "john_smith", "downtown_coffee")
- abstract: "What/who is this?"
- overview: Key details and relationships

EVENTS (time-bounded occurrences):
- routing_key: timestamp-based ID (e.g., "meeting_20250315", "visit_20250315")
- abstract: "What happened?"
- overview: When, where, who was involved

Return JSON schema:
{{
    "candidates": [
        {{
            "category": "entity" | "event",
            "routing_key": "identifier",
            "abstract": "Brief summary (≤100 chars)",
            "overview": "Key details",
            "content": "Full information",
            "confidence": 0.0-1.0
        }}
    ]
}}

Only extract actual entities/events mentioned. For events, include date in routing_key if available.
"""

        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": ["entity", "event"]},
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

                category = c.get("category", "entity")
                if category not in ("entity", "event"):
                    continue

                candidates.append(CandidateMemory(
                    category=category,
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
