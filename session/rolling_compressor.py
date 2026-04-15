"""Rolling compressor for session state layer (Layer 2).

Generates a full-rewrite compressed summary of recent conversation turns
every N turns or when token count exceeds threshold. The compressed text
replaces (not appends to) the previous window summary.

Requires LLM — no fallback. If LLM is unavailable, compression is skipped
and the window state remains unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.interfaces import LLM
from session.models import SessionMessage, SessionWindowState

logger = logging.getLogger("ogmem.session")


class RollingCompressor:
    """Compresses recent session messages into a rolling window summary.

    L0 (full rewrite) compression: regenerate the entire window summary
    from scratch on each compression cycle. Requires LLM — no fallback.
    """

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm

    def compress(
        self,
        messages: list[SessionMessage],
        window_state: SessionWindowState,
    ) -> SessionWindowState:
        """Compress messages into a new window state.

        Args:
            messages: All messages in the current session buffer.
            window_state: Current window state (updated in place).

        Returns:
            Updated SessionWindowState with new compressed text.
            If LLM is unavailable or returns empty, window_state is unchanged.
        """
        if not messages:
            return window_state

        if self._llm is None:
            logger.debug("RollingCompressor: no LLM, skipping compression")
            return window_state

        result = self._llm_compress(messages)
        if not result:
            logger.warning("RollingCompressor: LLM returned empty result, skipping")
            return window_state

        token_count = sum(m.estimated_tokens for m in messages)

        # Populate structured fields from LLM response
        window_state.active_task = result.get("active_task", "")
        window_state.confirmed_constraints = result.get("confirmed_constraints", [])
        window_state.recent_decisions = result.get("recent_decisions", [])
        window_state.open_loops = result.get("open_loops", [])
        window_state.uncertainties = result.get("uncertainties", [])
        window_state.compressed_text = result.get("summary", "")
        window_state.turn_count_at_last_compress = len(messages)
        window_state.token_count_at_last_compress = token_count
        return window_state

    def _llm_compress(self, messages: list[SessionMessage]) -> dict:
        """Use LLM to generate a full-rewrite window summary with structured fields."""
        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )

        prompt = (
            "Analyze this conversation and extract structured state.\n"
            "Output in the same language as the conversation.\n"
            "Preserve proper nouns, version numbers, file names, and error codes verbatim.\n"
            "Keep 'summary' field under 150 words.\n\n"
            "Fields to extract:\n"
            "- active_task: What is the user currently working on?\n"
            "- confirmed_constraints: What constraints have been established?\n"
            "- recent_decisions: What key decisions were made?\n"
            "- open_loops: What remains unresolved?\n"
            "- uncertainties: What is unclear or tentative?\n"
            "- summary: A concise narrative summary\n\n"
            f"CONVERSATION:\n{conversation}"
        )

        schema = {
            "type": "object",
            "properties": {
                "active_task": {
                    "type": "string",
                    "description": "Current task or sub-task being worked on",
                },
                "confirmed_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints that have been confirmed",
                },
                "recent_decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key decisions made in this segment",
                },
                "open_loops": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Unresolved items or questions",
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things we're not sure about",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise narrative summary of the conversation",
                },
            },
            "required": ["summary"],
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            return result
        except Exception as exc:
            logger.warning("RollingCompressor LLM failed: %s", exc)
            return {}
