"""Procedure/Pattern extraction from conversation history.

Extracts procedural knowledge, patterns, and skills learned by the agent.
These are agent-scoped memories (owner_scope="agent").
"""

from typing import Any

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor, LLM


class ProcedureExtractor(CandidateExtractor):
    """Extract procedures and patterns from messages.

    Procedural memories are agent-learned knowledge:
    - Problem-solving patterns
    - Error handling approaches
    - Best practices learned
    - Workflow improvements
    - Skill acquisition

    Returns CandidateMemory with category="pattern" or "skill".
    Owner scope is "agent" (these are the agent's learned patterns).
    """

    def __init__(self, llm: LLM):
        """Initialize ProcedureExtractor.

        Args:
            llm: LLM instance for structured extraction
        """
        self._llm = llm

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract procedures and patterns from conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction

        Returns:
            List of CandidateMemory with pattern candidates
        """
        conversation = self._format_conversation(messages)
        prompt = f"""Extract procedural patterns and skills learned from this conversation:

{conversation}

Extract patterns such as:
- Problem-solving approaches that worked
- Error handling patterns
- Best practices discovered
- Workflow improvements
- Techniques that should be remembered

Return JSON schema:
{{
    "candidates": [
        {{
            "routing_key": "pattern_name_lowercase",
            "abstract": "Brief pattern summary (≤100 chars)",
            "overview": "When to use this pattern",
            "content": "Full pattern with steps and rationale",
            "confidence": 0.0-1.0
        }}
    ]
}}

Only extract actual procedural learning, not facts about the user.
Use lowercase underscore format for routing_key (e.g., "error_retry_pattern", "user_delegation").
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
                    category="pattern",
                    owner_scope="agent",
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
