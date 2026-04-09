"""Integration-level unit tests for retrieval.pipeline.RetrievalPipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.models import (
    IndexRecord,
    RetrievalConfig,
    RequestContext,
    SearchMemoryResult,
)
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder
from retrieval.result_ranker import ResultRanker
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from tests.unit.retrieval.conftest import make_ctx


def _build_pipeline(
    idx: InMemoryVectorIndex,
    emb: MockEmbedder,
    cfg: RetrievalConfig,
    *,
    with_expand: bool = True,
) -> RetrievalPipeline:
    planner = QueryPlanner(cfg)
    seed = SeedRetriever(idx, emb, cfg)
    expand = HierarchicalSearcher(idx, cfg) if with_expand else None
    assembly = ResultRanker(cfg)
    return RetrievalPipeline(planner, seed, expand, assembly, cfg)


def _seed_records(idx: InMemoryVectorIndex) -> None:
    records = [
        IndexRecord(
            id="a0", uri="ctx://acct-1/users/u1/memories", level=0,
            text="user memory root with profile and preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile"},
        ),
        IndexRecord(
            id="a1", uri="ctx://acct-1/users/u1/memories/profile", level=1,
            text="user profile overview with coding preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories"},
        ),
        IndexRecord(
            id="a2", uri="ctx://acct-1/users/u1/memories/profile/detail", level=2,
            text="detailed user profile about coding habits and preference for Python",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories/profile"},
        ),
        IndexRecord(
            id="a3", uri="ctx://acct-1/users/u1/memories/entities/openai", level=2,
            text="OpenAI is a leading AI research company",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "entities", "parent_uri": "ctx://acct-1/users/u1/memories/entities"},
        ),
    ]
    idx.upsert(records)


@pytest.fixture()
def cfg():
    return RetrievalConfig(
        global_search_topk=20,
        default_top_k=5,
        max_convergence_rounds=2,
        score_propagation_alpha=0.5,
        hotness_alpha=0.0,
    )


@pytest.fixture()
def ctx():
    return make_ctx()


@pytest.fixture()
def idx():
    vi = InMemoryVectorIndex(dimension=8)
    _seed_records(vi)
    return vi


@pytest.fixture()
def emb():
    return MockEmbedder(dimension=8)


class TestPipeline:
    def test_returns_search_memory_result(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("用户的偏好是什么", ctx)
        assert isinstance(result, SearchMemoryResult)

    def test_hits_are_l2_only(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("用户的偏好是什么", ctx)
        for block in result.hits:
            assert block.level_hit == "L2"

    def test_without_expand(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg, with_expand=False)
        result = pipe.run("用户的偏好是什么", ctx)
        assert isinstance(result, SearchMemoryResult)

    def test_trace_contains_stages(self, idx, emb, cfg, ctx):
        pipe = _build_pipeline(idx, emb, cfg)
        result = pipe.run("tell me about coding", ctx)
        assert result.trace is not None
        stages = [s.stage for s in result.trace.stages]
        assert "planner" in stages
        assert "seed_retrieval" in stages

    def test_tenant_isolation(self, emb, cfg):
        vi = InMemoryVectorIndex(dimension=8)
        vi.upsert([
            IndexRecord(
                id="other", uri="ctx://other/users/x/memories/y", level=2,
                text="secret data",
                filters={"account_id": "other-acct", "owner_space": "user_x"},
                metadata={"context_type": "MEMORY"},
            ),
        ])
        pipe = _build_pipeline(vi, emb, cfg)
        ctx_mine = make_ctx(account_id="acct-1")
        result = pipe.run("secret data", ctx_mine)
        for block in result.hits:
            assert "other" not in block.uri
