"""Deterministic externalization selection policy."""

from __future__ import annotations

from dataclasses import dataclass

from .models import NormalizedMessage, ToolResultPart


@dataclass(frozen=True)
class SelectedPart:
    message_index: int
    part: ToolResultPart
    reason: str


class ToolResultPolicy:
    def __init__(self, single_result_chars: int = 20_000):
        if single_result_chars < 1:
            raise ValueError("tool-result threshold must be positive")
        self.single_result_chars = single_result_chars

    def select(self, messages: list[NormalizedMessage]) -> list[SelectedPart]:
        all_parts = [
            (message.message_index, part)
            for message in messages
            for part in message.tool_results
        ]
        selected: dict[tuple[int, str], SelectedPart] = {}
        for message_index, part in all_parts:
            if part.source_ref or part.source_collection:
                selected[(message_index, part.part_path)] = SelectedPart(
                    message_index, part, "source_aware_reuse",
                )
            elif part.original_chars >= self.single_result_chars:
                selected[(message_index, part.part_path)] = SelectedPart(
                    message_index, part, "single_result_threshold",
                )
        return sorted(selected.values(), key=lambda item: (item.message_index, item.part.part_path))
