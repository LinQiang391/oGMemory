"""Stage 4 — Result ranker.

Filters to L2 leaf nodes only, deduplicates by URI, sorts by score,
truncates to top_k, and converts into RetrievedBlock list.
"""

from __future__ import annotations

import logging
import re

from core.models import (
    LeafHit,
    RetrievalConfig,
    RetrievedBlock,
    RequestContext,
    SeedResult,
    TypedQuery,
    SeedHit,
)


logger = logging.getLogger(__name__)


_TEMPORAL_KEYWORDS = re.compile(
    r"\b("
    r"when|date|day|before|after|since|until|during|last|next|previous|"
    r"following|weekend|month|year|today|yesterday|tomorrow"
    r")\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def is_temporal_query(query: str | None) -> bool:
    text = query or ""
    return bool(_TEMPORAL_KEYWORDS.search(text) or _ISO_DATE.search(text))


def _date_tokens(text: str | None) -> set[str]:
    return set(_ISO_DATE.findall(text or ""))


def _temporal_bonus(hit: LeafHit, query_dates: set[str]) -> float:
    metadata = hit.metadata or {}
    when = str(metadata.get("when") or "")
    searchable = " ".join(
        str(value)
        for value in (
            when,
            hit.abstract,
        )
        if value
    )

    bonus = 0.0
    if when:
        bonus += 0.03
    if hit.category == "event":
        bonus += 0.02
    if query_dates:
        hit_dates = _date_tokens(searchable)
        if hit_dates & query_dates:
            bonus += 0.18
        elif hit_dates:
            bonus -= 0.04
    return bonus


def _vec_to_leaf(v: SeedHit) -> LeafHit:
    """Convert a SeedHit to a LeafHit.

    This is a shared utility used by ResultRanker and RetrievalPipeline.
    """
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
        metadata=dict(v.metadata or {}),
    )


class ResultRanker:
    def __init__(self, config: RetrievalConfig | None = None, relation_store=None) -> None:
        self.cfg = config or RetrievalConfig()
        self._relation_store = relation_store

    def assemble(
        self,
        typed_query: TypedQuery,
        leaf_hits: list[LeafHit],
        seed_result: SeedResult | None,
        ctx: RequestContext | None = None,
        *,
        fill_content_for_top_k: int = 0,
        context_reader=None,
    ) -> list[RetrievedBlock]:
        candidates = list(leaf_hits) if leaf_hits else []

        if not candidates and seed_result:
            candidates = self._l2_from_seed(seed_result)

        l2_only = [h for h in candidates if h.level == 2]

        # Deduplicate by URI, keeping the highest-scored hit per URI
        seen_uris: dict[str, LeafHit] = {}
        for hit in l2_only:
            if hit.uri not in seen_uris or hit.score > seen_uris[hit.uri].score:
                seen_uris[hit.uri] = hit
        l2_only = list(seen_uris.values())

        query_dates = _date_tokens(typed_query.text)
        if is_temporal_query(typed_query.text):
            l2_only.sort(
                key=lambda h: (h.score + _temporal_bonus(h, query_dates), h.score),
                reverse=True,
            )
        else:
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
                    has_overview=hit.has_overview,
                    has_content=hit.has_content,
                    match_reason=f"vector score {hit.score:.4f}",
                    metadata=dict(hit.metadata or {}),
                )
            )

        # Fill content_excerpt for top-K results (read full content from AGFS)
        if fill_content_for_top_k > 0 and context_reader and ctx:
            top_k_to_fill = min(fill_content_for_top_k, len(blocks))
            for i in range(top_k_to_fill):
                block = blocks[i]
                try:
                    content_block = context_reader.read(block.uri, ctx)
                    if content_block.overview:
                        block.overview = content_block.overview
                    if content_block.content_excerpt:
                        block.content_excerpt = content_block.content_excerpt
                except Exception as exc:
                    logger.debug("content fill failed for %s: %s", block.uri, exc)

        # Fill relations from relation store
        if self._relation_store and ctx:
            for block in blocks:
                try:
                    block.relations = self._relation_store.get_one_hop(block.uri, ctx, limit=3)
                except Exception as exc:
                    logger.debug("relation fetch failed for %s: %s", block.uri, exc)

        return blocks


    @staticmethod
    def _l2_from_seed(seed: SeedResult) -> list[LeafHit]:
        out: list[LeafHit] = []
        for v in seed.initial_candidates:
            if v.level == 2:
                out.append(_vec_to_leaf(v))
        return out
