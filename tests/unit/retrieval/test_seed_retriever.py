"""Unit tests for retrieval.seed_retriever.SeedRetriever."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.models import IndexRecord, RetrievalConfig, RequestContext, TypedQuery
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder
from retrieval.seed_retriever import SeedRetriever


def _make_ctx():
    return RequestContext(
        account_id="acct-1", user_id="u1", agent_id="a1",
        session_id="s1", trace_id="t1",
    )


def _make_cfg():
    return RetrievalConfig(
        global_search_topk=20, default_top_k=5,
        max_convergence_rounds=2, score_propagation_alpha=0.5, hotness_alpha=0.0,
    )


def _make_index() -> InMemoryVectorIndex:
    idx = InMemoryVectorIndex(dimension=8)
    records = [
        IndexRecord(
            id="r0", uri="ctx://acct-1/users/u1/memories", level=0,
            text="user memory root",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r1", uri="ctx://acct-1/users/u1/memories/profile", level=1,
            text="user profile overview",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "parent_uri": "ctx://acct-1/users/u1/memories",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r2", uri="ctx://acct-1/users/u1/memories/profile/detail", level=2,
            text="user profile full content about coding preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile",
                       "parent_uri": "ctx://acct-1/users/u1/memories/profile",
                       "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r3", uri="ctx://acct-1/users/u1/memories/entities/openai", level=2,
            text="OpenAI is an AI company",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "entities",
                       "parent_uri": "ctx://acct-1/users/u1/memories/entities",
                       "has_overview": True, "has_content": True},
        ),
    ]
    idx.upsert(records)
    return idx


@pytest.fixture()
def cfg():
    return _make_cfg()


@pytest.fixture()
def ctx():
    return _make_ctx()


@pytest.fixture()
def embedder():
    return MockEmbedder(dimension=8)


@pytest.fixture()
def index_with_data():
    return _make_index()


@pytest.fixture()
def retriever(index_with_data, embedder, cfg):
    return SeedRetriever(index_with_data, embedder, cfg)


@pytest.fixture()
def tq():
    return TypedQuery(
        text="user profile",
        context_type="MEMORY",
        categories=[],
        top_k=10,
        account_id="acct-1",
        owner_space="user_u1",
    )


class TestSeedRetriever:
    def test_returns_seed_result(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        total = len(result.starting_points) + len(result.initial_candidates)
        assert total > 0

    def test_query_vector_populated(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        assert len(result.query_vector) > 0

    def test_split_levels(self, retriever, tq, ctx):
        result = retriever.search(tq, ctx)
        for sp in result.starting_points:
            assert sp.level < 2 or sp.score == 0.0
        for ic in result.initial_candidates:
            assert ic.level == 2

    def test_empty_collection(self, embedder, cfg, ctx):
        empty_idx = InMemoryVectorIndex(dimension=8)
        r = SeedRetriever(empty_idx, embedder, cfg)
        tq = TypedQuery(text="anything", context_type="MEMORY", categories=[], top_k=5, account_id="acct-1")
        result = r.search(tq, ctx)
        assert len(result.initial_candidates) == 0
