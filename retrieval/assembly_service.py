"""Stage 4 — Assembly service.

Filters to L2 leaf nodes only, sorts by score, truncates to top_k,
and converts into RetrievedBlock list.

Token-aware assembly mode supports graceful degradation when budget
is constrained (full → abstract only → skip).
"""

from __future__ import annotations

from core.models import (
    AssembleResult,
    LeafHit,
    RetrievalConfig,
    RetrievedBlock,
    RequestContext,
    SeedResult,
    TokenBudget,
    TypedQuery,
    SeedHit,
)


class AssemblyService:
    def __init__(self, config: RetrievalConfig | None = None, relation_store=None) -> None:
        self.cfg = config or RetrievalConfig()
        self._relation_store = relation_store

    def assemble(
        self,
        typed_query: TypedQuery,
        leaf_hits: list[LeafHit],
        seed_result: SeedResult | None,
        ctx: RequestContext | None = None,
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

        # Fill relations from relation store
        if self._relation_store and ctx:
            for block in blocks:
                try:
                    block.relations = self._relation_store.get_one_hop(block.uri, ctx, limit=3)
                except Exception:
                    pass  # graceful degradation

        return blocks

    def assemble_with_budget(
        self,
        typed_query: TypedQuery,
        leaf_hits: list[LeafHit],
        seed_result: SeedResult | None,
        token_budget: TokenBudget,
        profile_text: str = "",
        ctx: RequestContext | None = None,
    ) -> AssembleResult:
        """Assemble memory hits into user messages with token budget awareness.

        Implements graceful degradation tiers:
        1. Full content (overview + content excerpt)
        2. Abstract only (when budget constrained)
        3. Skip low-scoring hits (when severely constrained)

        Args:
            typed_query: The typed query for filtering
            leaf_hits: L2 leaf hits from hierarchical search
            seed_result: Seed result for fallback
            token_budget: Token budget configuration
            profile_text: User profile text (counted against budget)
            ctx: Request context for relation loading

        Returns:
            AssembleResult with messages[], estimatedTokens, and metadata
        """
        candidates = list(leaf_hits) if leaf_hits else []

        if not candidates and seed_result:
            candidates = self._l2_from_seed(seed_result)

        # Filter to L2 only and sort by score
        l2_only = [h for h in candidates if h.level == 2]
        l2_only.sort(key=lambda h: h.score, reverse=True)

        # Reserve space for profile and system overhead
        profile_tokens = self._estimate_tokens(profile_text)
        overhead_tokens = 500  # System prompt, archive index, etc.
        available_for_memory = token_budget.total - profile_tokens - overhead_tokens

        if available_for_memory < token_budget.archive_limit:
            available_for_memory = token_budget.archive_limit

        messages: list[dict] = []
        total_tokens = profile_tokens + overhead_tokens
        stats = {
            "total_hits": len(l2_only),
            "included_hits": 0,
            "abstract_only": 0,
            "full_content": 0,
            "skipped": 0,
        }

        # Tier 1: Try full content for top hits
        for hit in l2_only:
            if total_tokens >= available_for_memory:
                break

            # Estimate tokens for this hit (abstract + content excerpt)
            hit_abstract_tokens = self._estimate_tokens(hit.abstract or "")
            # LeafHit has has_overview flag, not overview text content
            # Assume 500 tokens for content excerpt when has_content=True
            hit_content_tokens = 500 if hit.has_content else 100
            full_hit_tokens = hit_abstract_tokens + hit_content_tokens + 100  # +100 for formatting

            if total_tokens + full_hit_tokens <= available_for_memory:
                # Full content tier
                content_excerpt = f"[Content available] {hit.abstract}"
                messages.append({
                    "role": "user",
                    "content": f"[Memory] {hit.abstract}\n\n{content_excerpt}",
                })
                total_tokens += full_hit_tokens
                stats["included_hits"] += 1
                stats["full_content"] += 1
            else:
                # Check if we can fit abstract only
                abstract_only_tokens = hit_abstract_tokens + 50  # +50 for formatting
                if total_tokens + abstract_only_tokens <= available_for_memory:
                    messages.append({
                        "role": "user",
                        "content": f"[Memory] {hit.abstract}",
                    })
                    total_tokens += abstract_only_tokens
                    stats["included_hits"] += 1
                    stats["abstract_only"] += 1
                else:
                    stats["skipped"] += 1

        return AssembleResult(
            messages=messages,
            estimated_tokens=total_tokens,
            archive_count=0,
            archive_included=False,
            stats=stats,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (approximately 4 chars per token).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4

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
