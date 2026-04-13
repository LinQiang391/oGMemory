"""Stage 2 — Seed retriever.

Performs a global vector search across all levels (L0/L1/L2), then splits
results into starting_points (L0/L1) and initial_candidates (L2).
"""

from __future__ import annotations

import logging
from typing import Any

from core.enums import ContextType
from core.interfaces import VectorIndex, Embedder
from core.models import (
    RetrievalConfig,
    RequestContext,
    RetrieverMode,
    SeedResult,
    TypedQuery,
    SeedHit,
)

logger = logging.getLogger(__name__)


class SeedRetriever:
    """Global L0+L1+L2 search -> split into starting_points + initial_candidates."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    def search(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
        *,
        mode: str = RetrieverMode.QUICK,
    ) -> SeedResult:
        query_vector = self.embedder.embed_texts([typed_query.text])[0]
        root_uris = self._get_root_uris(typed_query.context_type, ctx)

        global_results = self._global_vector_search(query_vector, typed_query, ctx)

        starting_points, initial_candidates = self._merge_starting_points(
            root_uris, global_results,
        )

        return SeedResult(
            starting_points=starting_points,
            initial_candidates=initial_candidates,
            query_vector=query_vector,
            root_uris=root_uris,
        )

    @staticmethod
    def _get_root_uris(context_type: str | None, ctx: RequestContext) -> list[str]:
        # SECURITY: URI format per CLAUDE.md §1:
        # - ctx://{account}/users/{user}/memories/
        # - ctx://{account}/agents/{agent}/memories/
        # - ctx://{account}/agents/{agent}/skills/
        account = ctx.account_id
        visible_spaces = list(getattr(ctx, "visible_owner_spaces", ()) or [])
        visible_agent_ids = sorted(
            {
                space.split(":", 1)[1]
                for space in visible_spaces
                if isinstance(space, str) and space.startswith("agent:") and space.split(":", 1)[1]
            }
        )
        if not visible_agent_ids and ctx.agent_id:
            visible_agent_ids = [ctx.agent_id]
        if not context_type:
            roots = [
                f"ctx://{account}/users/{ctx.user_id}/memories/",
                f"ctx://{account}/resources/",
            ]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in visible_agent_ids)
            roots.extend(f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in visible_agent_ids)
            return roots
        ct = context_type.upper()
        if ct == ContextType.MEMORY.value:
            roots = [f"ctx://{account}/users/{ctx.user_id}/memories/"]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in visible_agent_ids)
            return roots
        if ct == ContextType.RESOURCE.value:
            return [f"ctx://{account}/resources/"]
        if ct == ContextType.SKILL.value:
            return [f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in visible_agent_ids]
        return []

    def _global_vector_search(
        self,
        query_vector: list[float],
        typed_query: TypedQuery,
        ctx: RequestContext,
    ) -> list[SeedHit]:
        filters: dict[str, Any] = {
            "level": [0, 1, 2],
            "account_id": ctx.account_id,
        }
        if typed_query.context_type:
            filters["context_type"] = typed_query.context_type
        if typed_query.owner_space:
            filters["owner_space"] = typed_query.owner_space
        if typed_query.categories:
            filters["category"] = typed_query.categories

        return self.vector_index.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=self.config.global_search_topk,
        )

    @staticmethod
    def _merge_starting_points(
        root_uris: list[str],
        global_results: list[SeedHit],
    ) -> tuple[list[SeedHit], list[SeedHit]]:
        initial_candidates: list[SeedHit] = []
        points: list[SeedHit] = []
        seen: set[str] = set()

        for hit in global_results:
            if hit.level == 2:
                initial_candidates.append(hit)
            else:
                if hit.uri not in seen:
                    points.append(hit)
                    seen.add(hit.uri)

        for uri in root_uris:
            if uri not in seen:
                points.append(SeedHit(uri=uri, score=0.0, level=0))
                seen.add(uri)

        return points, initial_candidates
