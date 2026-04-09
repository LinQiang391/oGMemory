"""Session compression using LLM.

Compresses conversation history into overview and abstract summaries.
"""

import json
from typing import Optional

from core.interfaces import LLM


class SessionCompressor:
    """Compresses session messages into condensed summaries.

    Uses LLM to generate:
    - overview: Structured summary with key topics, decisions, outcomes
    - abstract: Brief one-sentence summary (≤100 chars)

    Falls back to simple concatenation if LLM is not available.
    """

    def __init__(self, llm: Optional[LLM] = None):
        """Initialize the compressor.

        Args:
            llm: Optional LLM instance for smart compression
        """
        self.llm = llm

    def compress(self, messages: list[dict]) -> tuple[str, str]:
        """Compress a session message history.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Tuple of (overview, abstract)
            - overview: Structured summary of the session
            - abstract: Brief one-sentence summary (≤100 chars)
        """
        if not messages:
            return "", "Empty session"

        if self.llm is None:
            return self._fallback_compress(messages)

        return self._llm_compress(messages)

    def _llm_compress(self, messages: list[dict]) -> tuple[str, str]:
        """Use LLM to generate compressed summaries.

        Args:
            messages: List of message dicts

        Returns:
            Tuple of (overview, abstract)
        """
        # Build conversation text for LLM
        conversation = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )

        prompt = f"""Compress the following conversation into two summaries:

CONVERSATION:
{conversation}

Return JSON with this schema:
{{
    "overview": "Structured summary covering: main topics discussed, decisions made, outcomes achieved, and any action items",
    "abstract": "One-sentence summary (maximum 100 characters) capturing the essence of the conversation"
}}
"""

        schema = {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "abstract": {"type": "string", "maxLength": 100},
            },
            "required": ["overview", "abstract"],
        }

        try:
            result = self.llm.complete_json(prompt, schema)
            overview = result.get("overview", "")
            abstract = result.get("abstract", "")[:100]  # Ensure max length
            return overview, abstract

        except Exception:
            # Fall back to simple compression on error
            return self._fallback_compress(messages)

    def _fallback_compress(self, messages: list[dict]) -> tuple[str, str]:
        """Simple concatenation-based compression (no LLM).

        Args:
            messages: List of message dicts

        Returns:
            Tuple of (overview, abstract)
        """
        # Count messages by role
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

        # Build simple overview
        overview_parts = [
            f"Session with {len(messages)} messages ({user_count} from user, {assistant_count} from assistant).",
        ]

        # Extract first user message as topic hint
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    overview_parts.append(f"Started with: {preview}")
                    break

        overview = "\n\n".join(overview_parts)

        # Build abstract from first and last messages
        first_content = ""
        last_content = ""
        for msg in messages:
            if msg.get("role") == "user" and not first_content:
                first_content = msg.get("content", "")
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")

        if first_content and last_content:
            abstract = f"{first_content[:50]}... → {last_content[:30]}"
        elif first_content:
            abstract = first_content[:97] + "..."
        else:
            abstract = f"Session with {len(messages)} messages"

        # Ensure abstract is ≤100 chars
        abstract = abstract[:100]

        return overview, abstract
