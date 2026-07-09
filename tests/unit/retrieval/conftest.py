"""Shared fixtures for retrieval unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.models import (
    IndexRecord,
    RetrievalConfig,
    RequestContext,
    SeedHit,
)
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(
    account_id: str = "acct-1",
    user_id: str = "u1",
    agent_id: str = "a1",
) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id="s1",
        trace_id="t1",
    )


def seed_index(idx: InMemoryVectorIndex, records: list[IndexRecord]) -> None:
    """Insert records and their mock vectors into the in-memory index."""
    idx.upsert(records)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg() -> RetrievalConfig:
    return RetrievalConfig(
        global_search_topk=20,
        default_top_k=5,
        max_convergence_rounds=2,
        score_propagation_alpha=0.5,
        hotness_alpha=0.0,
    )


@pytest.fixture()
def ctx() -> RequestContext:
    return make_ctx()


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder(dimension=8)


@pytest.fixture()
def vector_index() -> InMemoryVectorIndex:
    return InMemoryVectorIndex(dimension=8)


@pytest.fixture()
def index_with_data(vector_index: InMemoryVectorIndex, ctx: RequestContext) -> InMemoryVectorIndex:
    """Index pre-loaded with a small hierarchy."""
    records = [
        IndexRecord(
            id="r0", uri="ctx://acct-1/users/u1/memories", level=0,
            text="user memory root",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r1", uri="ctx://acct-1/users/u1/memories/profile", level=1,
            text="user profile overview",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories", "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r2", uri="ctx://acct-1/users/u1/memories/profile/detail", level=2,
            text="user profile full content about coding preferences",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "profile", "parent_uri": "ctx://acct-1/users/u1/memories/profile", "has_overview": True, "has_content": True},
        ),
        IndexRecord(
            id="r3", uri="ctx://acct-1/users/u1/memories/entities/openai", level=2,
            text="OpenAI is an AI company",
            filters={"account_id": "acct-1", "owner_space": "user_u1"},
            metadata={"context_type": "MEMORY", "category": "entities", "parent_uri": "ctx://acct-1/users/u1/memories/entities", "has_overview": True, "has_content": True},
        ),
    ]
    seed_index(vector_index, records)
    return vector_index
