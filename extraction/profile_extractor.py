"""Profile extraction from conversation history.

Extracts user profile information including name, background, preferences,
and other stable user attributes.
"""

import json
from typing import Any

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor, LLM


class ProfileExtractor(CandidateExtractor):
    """Extract user profile information from messages.

    Profile memories are stable, long-term user attributes:
    - Name and demographic info
    - Background and context
    - Stable preferences (vs transient preferences)
    - Goals and interests

    Returns CandidateMemory with category="profile", routing_key="profile".
    """

    def __init__(self, llm: LLM):
        """Initialize ProfileExtractor.

        Args:
            llm: LLM instance for structured extraction
        """
        self._llm = llm

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract profile information from conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction

        Returns:
            List of CandidateMemory with profile candidates (usually 0-1)
        """
        # Build extraction prompt
        conversation = self._format_conversation(messages)
        prompt = f"""Extract user profile information from this conversation:

{conversation}

Extract the following if present:
- Name and basic info
- Background or context
- Long-term goals or interests
- Stable personality traits

Return a JSON response with this schema:
{{
    "candidates": [
        {{
            "routing_key": "profile",
            "abstract": "Brief summary of user profile (≤100 chars)",
            "overview": "Structured overview with key facts",
            "content": "Full profile content",
            "confidence": 0.0-1.0
        }}
    ]
}}

Only extract if there's actual profile information. Return empty candidates array if not found.
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
                # Filter by confidence threshold
                if c.get("confidence", 0) < 0.5:
                    continue

                candidates.append(CandidateMemory(
                    category="profile",
                    owner_scope="user",
                    routing_key=c["routing_key"],
                    abstract=c["abstract"],
                    overview=c["overview"],
                    content=c["content"],
                    confidence=c["confidence"],
                ))

            return candidates

        except Exception as e:
            # Log error and return empty list
            return []

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages for LLM prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
