"""Retrieval pipeline orchestrator.

Runs 4 stages sequentially with graceful degradation:
  Stage 1  QueryPlanner        — classify query into TypedQuery[]
  Stage 2  SeedRetriever       — global L0/L1/L2 mixed vector search
  Stage 3  HierarchicalSearcher — auto-expand L0/L1 to L2 leaves
  Stage 4  AssemblyService     — keep only L2, sort, top-k -> RetrievedBlock[]

Returns a pure structured SearchMemoryResult.
"""

from __future__ import annotations

import logging
import time

from core.enums import ContextType
from core.errors import RetrievalError
from core.models import (
    LeafHit,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
    SeedResult,
    TypedQuery,
    SeedHit,
)
from retrieval.assembly_service import AssemblyService
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.trace import RetrievalTrace, TraceTimer

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        planner: QueryPlanner,
        seed_retriever: SeedRetriever,
        hierarchical_searcher: HierarchicalSearcher | None,
        assembly: AssemblyService,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.planner = planner
        self.seed_retriever = seed_retriever
        self.hierarchical_searcher = hierarchical_searcher
        self.assembly = assembly
        self.cfg = config or RetrievalConfig()

    def run(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        session_archive: dict | None = None,
        hints: dict | None = None,
        score_threshold: float | None = None,
        mode: str = RetrieverMode.QUICK,
    ) -> SearchMemoryResult:
        t0 = time.monotonic()
        trace = RetrievalTrace()

        # Stage 1
        typed_queries = self._run_planner(
            query, ctx, trace,
            top_k=top_k, categories=categories,
            hints=hints, session_archive=session_archive,
        )

        # Stage 2
        all_seed_results: list[SeedResult] = []
        for tq in typed_queries:
            sr = self._run_seed(tq, ctx, trace, mode=mode)
            all_seed_results.append(sr)

        # Stage 3
        all_leaf_hits: list[LeafHit] = []
        for tq, sr in zip(typed_queries, all_seed_results):
            leaves = self._run_expand(
                tq, sr, ctx, trace,
                limit=tq.top_k, mode=mode,
                score_threshold=score_threshold,
            )
            all_leaf_hits.extend(leaves)

        # Stage 4
        merged_seed = self._merge_seeds(all_seed_results)
        blocks = self._run_assembly(typed_queries, all_leaf_hits, merged_seed, trace, ctx)

        trace.total_ms = (time.monotonic() - t0) * 1000

        return SearchMemoryResult(
            request_id=trace.request_id,
            query=query,
            typed_queries=typed_queries,
            hits=blocks,
            trace=trace,
        )

    # -- Stage runners with degradation ------------------------------------

    def _run_planner(
        self, query, ctx, trace, *, top_k, categories, hints, session_archive
    ) -> list[TypedQuery]:
        with TraceTimer("planner") as st:
            st.input_count = 1
            try:
                queries = self.planner.plan(
                    query, ctx,
                    session_archive=session_archive,
                    hints=hints, categories=categories, top_k=top_k,
                )
            except RetrievalError:
                raise
            except Exception as exc:
                st.warnings.append(f"[planner_fallback] {exc}")
                queries = [
                    TypedQuery(
                        text=query, context_type=ct.value,
                        categories=[], account_id=ctx.account_id,
                        owner_space=ctx.user_space_name(),
                    )
                    for ct in ContextType
                ]
            st.output_count = len(queries)
        trace.add_stage(st)
        return queries

    def _run_seed(self, tq, ctx, trace, *, mode) -> SeedResult:
        with TraceTimer("seed_retrieval") as st:
            st.input_count = 1
            try:
                result = self.seed_retriever.search(tq, ctx, mode=mode)
            except Exception as exc:
                st.warnings.append(f"[vector_search_failed] {exc}")
                result = SeedResult()
            st.output_count = len(result.starting_points) + len(result.initial_candidates)
        trace.add_stage(st)
        return result

    def _run_expand(self, tq, sr, ctx, trace, *, limit, mode, score_threshold) -> list[LeafHit]:
        if not sr.starting_points and not sr.initial_candidates:
            return []

        has_dirs = any(v.level < 2 for v in sr.starting_points)

        if has_dirs and self.hierarchical_searcher:
            with TraceTimer("hierarchical") as st:
                st.input_count = len(sr.starting_points) + len(sr.initial_candidates)
                try:
                    leaves = self.hierarchical_searcher.expand(
                        tq, sr, ctx,
                        limit=limit, mode=mode, score_threshold=score_threshold,
                    )
                except Exception as exc:
                    st.warnings.append(f"[expand_failed] {exc}")
                    leaves = self._fallback_l2_from_seed(sr)
                st.output_count = len(leaves)
            trace.add_stage(st)
            return leaves

        return self._fallback_l2_from_seed(sr)

    def _run_assembly(self, typed_queries, leaf_hits, merged_seed, trace, ctx=None) -> list[RetrievedBlock]:
        with TraceTimer("assembly") as st:
            st.input_count = len(leaf_hits)
            try:
                tq = typed_queries[0] if typed_queries else TypedQuery(text="", context_type="")
                blocks = self.assembly.assemble(tq, leaf_hits, merged_seed, ctx=ctx)
            except Exception as exc:
                st.warnings.append(f"[assemble_failed] {exc}")
                blocks = []
            st.output_count = len(blocks)
        trace.add_stage(st)
        return blocks

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _fallback_l2_from_seed(sr: SeedResult) -> list[LeafHit]:
        out: list[LeafHit] = []
        for v in sr.initial_candidates:
            if v.level == 2:
                out.append(LeafHit(
                    uri=v.uri, score=v.score, level=v.level,
                    parent_uri=v.parent_uri, category=v.category,
                    owner_space=v.owner_space, abstract=v.abstract,
                    has_overview=v.has_overview, has_content=v.has_content,
                    active_count=v.active_count, updated_at=v.updated_at,
                ))
        return out

    @staticmethod
    def _merge_seeds(seed_results: list[SeedResult]) -> SeedResult | None:
        if not seed_results:
            return None
        if len(seed_results) == 1:
            return seed_results[0]
        merged = SeedResult()
        seen_sp: set[str] = set()
        seen_ic: set[str] = set()
        for sr in seed_results:
            for v in sr.starting_points:
                if v.uri not in seen_sp:
                    merged.starting_points.append(v)
                    seen_sp.add(v.uri)
            for v in sr.initial_candidates:
                if v.uri not in seen_ic:
                    merged.initial_candidates.append(v)
                    seen_ic.add(v.uri)
            if not merged.query_vector and sr.query_vector:
                merged.query_vector = sr.query_vector
            merged.root_uris.extend(sr.root_uris)
        return merged
