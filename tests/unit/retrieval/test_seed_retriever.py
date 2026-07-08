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

    def test_owner_space_list_filters_visible_spaces(self, retriever, ctx):
        tq = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=10,
            account_id="acct-1",
            owner_space=["user_u1"],
        )

        result = retriever.search(tq, ctx)

        assert result.initial_candidates
        assert all(hit.owner_space == "user_u1" for hit in result.initial_candidates)

    def test_owner_space_defaults_to_visible_spaces(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="visible",
                    uri="ctx://acct-1/users/u1/memories/visible",
                    level=2,
                    text="shared visible support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u1"},
                    metadata={"context_type": "MEMORY", "category": "event"},
                ),
                IndexRecord(
                    id="hidden",
                    uri="ctx://acct-1/users/u2/memories/hidden",
                    level=2,
                    text="shared visible support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u2"},
                    metadata={"context_type": "MEMORY", "category": "event"},
                ),
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )
        query = TypedQuery(
            text="support group",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space=None,
        )

        result = retriever.search(query, ctx)

        assert {hit.owner_space for hit in result.initial_candidates} == {"user:u1"}

    def test_explicit_owner_space_is_intersected_with_visible_spaces(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="visible",
                    uri="ctx://acct-1/users/u1/memories/visible",
                    level=2,
                    text="support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u1"},
                    metadata={"context_type": "MEMORY"},
                ),
                IndexRecord(
                    id="hidden",
                    uri="ctx://acct-1/users/u2/memories/hidden",
                    level=2,
                    text="support group",
                    filters={"account_id": "acct-1", "owner_space": "user:u2"},
                    metadata={"context_type": "MEMORY"},
                ),
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )

        hidden_result = retriever.search(
            TypedQuery(
                text="support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space="user:u2",
            ),
            ctx,
        )
        mixed_result = retriever.search(
            TypedQuery(
                text="support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space=["user:u1", "user:u2"],
            ),
            ctx,
        )

        assert hidden_result.initial_candidates == []
        assert {hit.owner_space for hit in mixed_result.initial_candidates} == {"user:u1"}

    def test_owner_space_intersection_accepts_legacy_alias(self, embedder):
        idx = InMemoryVectorIndex(dimension=8)
        idx.upsert(
            [
                IndexRecord(
                    id="legacy",
                    uri="ctx://acct-1/users/u1/memories/legacy",
                    level=2,
                    text="legacy support group",
                    filters={"account_id": "acct-1", "owner_space": "user_u1"},
                    metadata={"context_type": "MEMORY"},
                )
            ]
        )
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1",),
        )
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )

        result = retriever.search(
            TypedQuery(
                text="legacy support group",
                context_type="MEMORY",
                categories=[],
                top_k=5,
                account_id="acct-1",
                owner_space="user_u1",
            ),
            ctx,
        )

        assert {hit.owner_space for hit in result.initial_candidates} == {"user_u1"}

    def test_resource_query_applies_visible_owner_space_filter(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.filters = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.filters = filters
                return []

        idx = RecordingIndex()
        retriever = SeedRetriever(
            idx,
            embedder,
            RetrievalConfig(global_search_topk=10, owner_scope_filter_enabled=True),
        )
        query = TypedQuery(
            text="resource reference",
            context_type="RESOURCE",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space=["user:u1", "agent:a1"],
        )

        retriever.search(query, ctx)

        assert idx.filters["owner_space"] == ["user:u1", "user_u1", "agent:a1", "agent_a1"]

    def test_root_uris_include_visible_shared_agents(self, retriever, ctx):
        shared_ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:u1", "agent:a1", "agent:a2"),
        )

        result = retriever.search(
            TypedQuery(
                text="user profile",
                context_type="MEMORY",
                categories=[],
                top_k=10,
                account_id="acct-1",
                owner_space=["user_u1"],
            ),
            shared_ctx,
        )

        assert "ctx://acct-1/agents/a1/memories/" in result.root_uris
        assert "ctx://acct-1/agents/a2/memories/" in result.root_uris

    def test_root_uris_include_legacy_visible_shared_agents(self, retriever):
        shared_ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user_u1", "agent_a2"),
        )

        result = retriever.search(
            TypedQuery(
                text="user profile",
                context_type="MEMORY",
                categories=[],
                top_k=10,
                account_id="acct-1",
                owner_space=["user_u1"],
            ),
            shared_ctx,
        )

        assert "ctx://acct-1/agents/a2/memories/" in result.root_uris

    def test_vector_search_uses_internal_candidate_pool(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(
            global_search_topk=30,
            default_top_k=5,
            candidate_pool_enabled=True,
            candidate_pool_multiplier=3,
            min_candidate_pool_size=15,
        )
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 15

    def test_candidate_pool_does_not_cap_below_requested_top_k(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(
            global_search_topk=30,
            default_top_k=5,
            candidate_pool_enabled=True,
            candidate_pool_multiplier=3,
            min_candidate_pool_size=15,
        )
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="large result set",
            context_type="MEMORY",
            categories=[],
            top_k=40,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 40

    def test_vector_search_uses_global_topk_when_candidate_pool_disabled(self, embedder, ctx):
        class RecordingIndex:
            def __init__(self) -> None:
                self.top_k: int | None = None

            def search_by_vector(self, query_vector, filters, top_k):
                self.top_k = top_k
                return []

        idx = RecordingIndex()
        cfg = RetrievalConfig(global_search_topk=30, default_top_k=5)
        retriever = SeedRetriever(idx, embedder, cfg)
        query = TypedQuery(
            text="user profile",
            context_type="MEMORY",
            categories=[],
            top_k=5,
            account_id="acct-1",
            owner_space="user_u1",
        )

        retriever.search(query, ctx)

        assert idx.top_k == 30
