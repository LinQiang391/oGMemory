"""Candidate pool sizing helpers for retrieval stages."""

from __future__ import annotations

from core.models import RetrievalConfig


def candidate_top_k(
    config: RetrievalConfig,
    final_top_k: int | None,
    *,
    fallback_top_k: int | None = None,
) -> int:
    if not getattr(config, "candidate_pool_enabled", False):
        return max(1, int(fallback_top_k or final_top_k or config.default_top_k))

    requested = max(1, int(final_top_k or config.default_top_k))
    multiplier = max(1, int(getattr(config, "candidate_pool_multiplier", 1)))
    minimum = max(1, int(getattr(config, "min_candidate_pool_size", requested)))
    candidate_count = max(requested * multiplier, minimum)
    return min(config.global_search_topk, candidate_count)
