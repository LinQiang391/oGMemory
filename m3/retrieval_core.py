"""Shared candidate-budget and final-selection contracts for M3 retrieval.

Backends remain responsible for permission/scope filtering, hydration and
scoring. Scores from different ranking algorithms are not comparable; this
module only orders candidates scored by the same caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

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


def validate_transcript_options(top_k: int, fusion: str) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("transcript_top_k must be an integer >= 0 (0 disables)")
    if fusion not in ("rrf", "weighted"):
        raise ValueError("transcript_fusion must be rrf or weighted")


@dataclass(frozen=True)
class FusedEvidence(Generic[T]):
    evidence: T
    score: float
    source_scores: dict[str, float]


def fuse_evidence(
    vector: Iterable[T],
    transcript: Iterable[T],
    *,
    identity: Callable[[T], Hashable | None],
    score: Callable[[T], float],
    method: str = "rrf",
) -> list[FusedEvidence[T]]:
    """Experimental shared fusion, not a claim of calibrated confidence.

    RRF uses equal branch weights and k=60. The comparison baseline uses
    0.5 * clamp(vector, 0, 1) + 0.5 * transcript token coverage.
    Missing branch scores contribute zero. Deduplicate within each branch
    before assigning ranks; on exact fused ties prefer transcript matches.
    """
    validate_transcript_options(0, method)
    merged: dict[Hashable, FusedEvidence[T]] = {}
    totals: dict[Hashable, float] = {}
    for name, candidates in (("vector", vector), ("transcript", transcript)):
        ranked = select_evidence(candidates, identity=identity, score=score)
        for rank, candidate in enumerate(ranked, 1):
            key = identity(candidate)
            if key is None:
                key = object()
            raw = float(score(candidate))
            previous = merged.get(key)
            sources = dict(previous.source_scores) if previous else {}
            sources[name] = raw
            contribution = 1 / (60 + rank) if method == "rrf" else .5 * min(1., max(0., raw))
            totals[key] = totals.get(key, 0.) + contribution
            merged[key] = FusedEvidence(
                previous.evidence if previous else candidate, totals[key], sources,
            )
    return sorted(
        merged.values(),
        key=lambda hit: (-hit.score, -hit.source_scores.get("transcript", -1.)),
    )
