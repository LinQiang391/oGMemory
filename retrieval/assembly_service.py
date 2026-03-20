"""Stage 4 — Assembly service.

Filters to L2 leaf nodes only, sorts by score, truncates to top_k,
and converts into RetrievedBlock list.
"""

from __future__ import annotations

from core.models import (
    LeafHit,
    RetrievalConfig,
    RetrievedBlock,
    SeedResult,
    TypedQuery,
    SeedHit,
)


class AssemblyService:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.cfg = config or RetrievalConfig()

    def assemble(
        self,
        typed_query: TypedQuery,
        leaf_hits: list[LeafHit],
        seed_result: SeedResult | None,
    ) -> list[RetrievedBlock]:
        candidates = list(leaf_hits) if leaf_hits else []

        if not candidates and seed_result:
            candidates = self._l2_from_seed(seed_result)

        l2_only = [h for h in candidates if h.level == 2]
        l2_only.sort(key=lambda h: h.score, reverse=True)
        top = l2_only[: typed_query.top_k]

        blocks: list[RetrievedBlock] = []
        for hit in top:
            blocks.append(
                RetrievedBlock(
                    uri=hit.uri,
                    level_hit="L2",
                    score=hit.score,
                    category=hit.category,
                    owner_space=hit.owner_space,
                    abstract=hit.abstract or None,
                    match_reason=f"vector score {hit.score:.4f}",
                )
            )
        return blocks

    @staticmethod
    def _l2_from_seed(seed: SeedResult) -> list[LeafHit]:
        out: list[LeafHit] = []
        for v in seed.initial_candidates:
            if v.level == 2:
                out.append(_vec_to_leaf(v))
        return out


def _vec_to_leaf(v: SeedHit) -> LeafHit:
    return LeafHit(
        uri=v.uri,
        score=v.score,
        level=v.level,
        parent_uri=v.parent_uri,
        category=v.category,
        owner_space=v.owner_space,
        abstract=v.abstract,
        has_overview=v.has_overview,
        has_content=v.has_content,
        active_count=v.active_count,
        updated_at=v.updated_at,
    )
