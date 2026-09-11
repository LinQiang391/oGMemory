"""Shared candidate-budget and final-selection contracts for M3 retrieval.

Backends remain responsible for permission/scope filtering, hydration and
scoring. Scores from different ranking algorithms are not comparable; this
module only orders candidates scored by the same caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalBudget:
    """Per-branch candidate depth, separate from the final evidence limit.

    An explicit candidate depth is promoted to at least final_top_k. This lets
    a model request more final hits without invalidating runtime configuration.
    None preserves the caller's legacy candidate depth.
    """

    final_top_k: int
    candidate_k: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("final_top_k", self.final_top_k), ("candidate_k", self.candidate_k),
        ):
            if name == "candidate_k" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def fetch_k(self, *, legacy_default: int | None = None) -> int:
        depth = self.candidate_k if self.candidate_k is not None else legacy_default
        return max(self.final_top_k, depth or self.final_top_k)


def select_evidence(
    candidates: Iterable[T],
    *,
    identity: Callable[[T], Hashable | None],
    score: Callable[[T], float],
    top_k: int | None = None,
) -> list[T]:
    """Deduplicate by evidence identity, sort by score, then truncate.

    Keep the highest-scored occurrence, retaining first-seen order on ties.
    Missing identities are kept distinct rather than silently collapsed.
    Non-finite scores cannot be ranked and are excluded. Input is not mutated.
    Without top_k this also serves as a pre-rerank union; do not truncate a
    union until all scoring branches have had a chance to rank it.
    """
    if top_k is not None:
        RetrievalBudget(top_k)
    merged: dict[Hashable, tuple[float, T]] = {}
    for candidate in candidates:
        value = float(score(candidate))
        if not math.isfinite(value):
            continue
        key = identity(candidate)
        # object() cannot collide with a valid persisted evidence identity.
        if key is None:
            key = object()
        if key not in merged or value > merged[key][0]:
            merged[key] = (value, candidate)
    ranked = sorted(merged.values(), key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked[:top_k]]
