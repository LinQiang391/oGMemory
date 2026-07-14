"""Stage 2 — Seed retriever.

Performs a global vector search across all levels (L0/L2), optionally
augmented with BM25 keyword search via Vector-Anchored Fusion.
Splits results into starting_points (L0) and initial_candidates (L2).
"""

from __future__ import annotations

import logging
from typing import Any

from core.enums import ContextType
from core.interfaces import Embedder, VectorIndex
from core.models import (
    RequestContext,
    RetrievalConfig,
    RetrieverMode,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.candidate_pool import candidate_top_k
from retrieval.owner_scope import allowed_owner_space_filter, visible_agent_ids
from retrieval.trace import StageTrace, TraceTimer, trace_capable

logger = logging.getLogger(__name__)


class SeedRetriever:
    """Global L0+L2 search with optional BM25 hybrid fusion."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        bm25_index: Any | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.bm25_index = bm25_index

    @trace_capable
    def search(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
        *,
        mode: str = RetrieverMode.QUICK,
        trace_stage: StageTrace | None = None,
    ) -> SeedResult:
        with TraceTimer("embed_query", trace_stage) as child:
            child.input_count = 1
            query_vector = self.embedder.embed_texts([typed_query.text])[0]
            child.output_count = 1
            child.metrics["embedding_dim"] = len(query_vector)
            child.metrics["embedding_calls"] = 1
        with TraceTimer("build_root_uris", trace_stage) as child:
            child.input_count = 1
            root_uris = self._get_root_uris(typed_query.context_type, ctx)
            child.output_count = len(root_uris)
        global_results = self._global_vector_search(
            query_vector, typed_query, ctx, trace_stage=trace_stage
        )

        # Hybrid: merge BM25 results via Vector-Anchored Fusion
        bm25_results = []
        if self.bm25_index and self.bm25_index.doc_count > 0:
            with TraceTimer("bm25_search", trace_stage) as child:
                child.input_count = 1
                bm25_results = self._bm25_search(typed_query, ctx)
                child.output_count = len(bm25_results)
            if bm25_results:
                with TraceTimer("fusion", trace_stage) as child:
                    child.input_count = len(global_results) + len(bm25_results)
                    global_results = self._fuse_results(global_results, bm25_results)
                    child.output_count = len(global_results)

        with TraceTimer("split_starting_points", trace_stage) as child:
            child.input_count = len(global_results)
            starting_points, initial_candidates = self._merge_starting_points(
                root_uris,
                global_results,
            )
            child.output_count = len(starting_points) + len(initial_candidates)
            child.metrics["starting_points"] = len(starting_points)
            child.metrics["initial_candidates"] = len(initial_candidates)
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
        agent_ids = visible_agent_ids(ctx)
        if not agent_ids and ctx.agent_id:
            agent_ids = [ctx.agent_id]
        if not context_type:
            roots = [
                f"ctx://{account}/users/{ctx.user_id}/memories/",
                f"ctx://{account}/resources/",
            ]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in agent_ids)
            roots.extend(f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in agent_ids)
            return roots
        ct = context_type.upper()
        if ct == ContextType.MEMORY.value:
            roots = [f"ctx://{account}/users/{ctx.user_id}/memories/"]
            roots.extend(f"ctx://{account}/agents/{agent_id}/memories/" for agent_id in agent_ids)
            return roots
        if ct == ContextType.RESOURCE.value:
            return [f"ctx://{account}/resources/"]
        if ct == ContextType.SKILL.value:
            return [f"ctx://{account}/agents/{agent_id}/skills/" for agent_id in agent_ids]
        return []

    def _global_vector_search(
        self,
        query_vector: list[float],
        typed_query: TypedQuery,
        ctx: RequestContext,
        trace_stage: StageTrace | None = None,
    ) -> list[SeedHit]:
        with TraceTimer("build_vector_filters", trace_stage) as child:
            filters: dict[str, Any] = {
                "level": [0, 1, 2],  # Include directory L0/L1 for hierarchical expansion
                "account_id": ctx.account_id,
            }
            if typed_query.context_type:
                filters["context_type"] = typed_query.context_type
            owner_space = allowed_owner_space_filter(
                typed_query,
                ctx,
                enforce_visible_scope=getattr(self.config, "owner_scope_filter_enabled", False),
            )
            if owner_space:
                filters["owner_space"] = owner_space
            if typed_query.categories:
                filters["category"] = typed_query.categories
            child.input_count = 1
            child.output_count = len(filters)
            child.metrics["filter_keys"] = sorted(filters)
        search_top_k = candidate_top_k(
            self.config,
            typed_query.top_k,
            fallback_top_k=self.config.global_search_topk,
        )
        with TraceTimer("vector_search", trace_stage) as child:
            child.input_count = 1
            child.metrics["candidate_top_k"] = search_top_k
            child.metrics["filter_keys"] = sorted(filters)
            results = self.vector_index.search_by_vector(
                query_vector=query_vector,
                filters=filters,
                top_k=search_top_k,
            )
            child.output_count = len(results)
            if hasattr(self.vector_index, "last_search_metrics"):
                child.metrics.update(getattr(self.vector_index, "last_search_metrics") or {})
        return results

    def _bm25_search(
        self,
        typed_query: TypedQuery,
        ctx: RequestContext,
    ) -> list[SeedHit]:
        """BM25 keyword search with metadata filtering."""
        filters: dict[str, Any] = {
            "account_id": ctx.account_id,
        }
        if typed_query.context_type:
            filters["context_type"] = typed_query.context_type
        owner_space = allowed_owner_space_filter(
            typed_query,
            ctx,
            enforce_visible_scope=getattr(self.config, "owner_scope_filter_enabled", False),
        )
        if owner_space:
            filters["owner_space"] = owner_space
        if typed_query.categories:
            filters["category"] = typed_query.categories

        bm25_hits = self.bm25_index.search(
            query=typed_query.text,
            top_k=candidate_top_k(
                self.config,
                typed_query.top_k,
                fallback_top_k=self.config.global_search_topk,
            ),
            filters=filters,
        )
        hits: list[SeedHit] = []
        for h in bm25_hits:
            meta = h.metadata
            metadata = dict(meta or {})
            metadata["_bm25_raw_score"] = h.score
            hits.append(
                SeedHit(
                    uri=meta.get("uri", ""),
                    score=0.0,  # score set by _fuse_results
                    level=int(meta.get("level", 2)),
                    parent_uri=meta.get("parent_uri"),
                    context_type=meta.get("context_type", ""),
                    category=meta.get("category", ""),
                    owner_space=meta.get("owner_space", ""),
                    abstract=meta.get("abstract", ""),
                    has_overview=bool(meta.get("has_overview")),
                    has_content=bool(meta.get("has_content")),
                    active_count=int(meta.get("active_count", 0)),
                    updated_at=meta.get("updated_at"),
                    metadata=metadata,
                )
            )
        return hits

    @staticmethod
    def _fuse_results(
        vector_hits: list[SeedHit],
        bm25_hits: list[SeedHit],
        alpha: float = 0.7,
        saturation_k: float = 5.0,
    ) -> list[SeedHit]:
        """Vector-Anchored Fusion: combine vector and BM25 results.

        Formula:
            sat_bm25 = raw_bm25 / (raw_bm25 + saturation_k)
            final_score = alpha * vec_score + (1 - alpha) * sat_bm25

        Args:
            vector_hits: Results from vector search (score = cosine similarity)
            bm25_hits: Results from BM25 search (score = raw BM25 score)
            alpha: Weight for vector score (default 0.7 = 70% vector)
            saturation_k: BM25 saturation constant (default 5.0)
        """
        # Index vector hits by URI
        fused: dict[str, SeedHit] = {}
        vec_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for hit in vector_hits:
            fused[hit.uri] = hit
            vec_scores[hit.uri] = hit.score

        for hit in bm25_hits:
            bm25_scores[hit.uri] = hit.metadata.get("_bm25_raw_score", 0.0)
            if hit.uri not in fused:
                fused[hit.uri] = hit

        if not fused:
            return []

        # Compute floor defaults (minimum score from each path)
        min_vec = min(vec_scores.values()) if vec_scores else 0.0
        min_bm25_sat = (
            min(s / (s + saturation_k) for s in bm25_scores.values()) if bm25_scores else 0.0
        )

        # Compute fused scores
        results: list[SeedHit] = []
        for uri, hit in fused.items():
            vec_s = vec_scores.get(uri, min_vec)
            raw_bm25 = bm25_scores.get(uri, 0.0)
            sat_bm25 = raw_bm25 / (raw_bm25 + saturation_k) if raw_bm25 > 0 else min_bm25_sat
            fused_score = alpha * vec_s + (1 - alpha) * sat_bm25

            results.append(
                SeedHit(
                    uri=hit.uri,
                    score=fused_score,
                    level=hit.level,
                    parent_uri=hit.parent_uri,
                    context_type=hit.context_type,
                    category=hit.category,
                    owner_space=hit.owner_space,
                    abstract=hit.abstract,
                    has_overview=hit.has_overview,
                    has_content=hit.has_content,
                    active_count=hit.active_count,
                    updated_at=hit.updated_at,
                    metadata=hit.metadata,
                )
            )

        results.sort(key=lambda h: h.score, reverse=True)
        return results

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
