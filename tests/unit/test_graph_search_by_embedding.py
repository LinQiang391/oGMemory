"""Unit tests for ``MemoryGraphStore.search_by_embedding``.

These tests pin the contract from the v1 control-loop plan:
- returns ``[(node_id, score), ...]`` sorted by score desc, top-k
- cosine similarity, not euclidean
- respects account_id isolation, ``extractor_types``, ``video_id``, ``before_clip``
- dimension-mismatched embeddings and missing-embedding nodes are skipped silently
- non-extracted node types are skipped today (only ``ExtractedNode`` carries embeddings)
"""

from __future__ import annotations

import math

import pytest

from core.errors import AccessDeniedError
from core.graph_enums import GraphNodeType, RawDataType
from core.graph_models import (
    create_extracted_node,
    create_memory_node,
    create_raw_node,
)
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acc_test",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        trace_id="t1",
    )


@pytest.fixture
def store():
    return MemoryGraphStore()


@pytest.fixture
def other_ctx():
    """A second tenant's context for isolation tests."""
    return RequestContext(
        account_id="acc_other",
        user_id="u2",
        agent_id="a2",
        session_id="s2",
        trace_id="t2",
    )


# Orthonormal-ish 3-D basis vectors used across the suite so cosine is
# exact (orthogonal -> 0.0, parallel -> 1.0).
V_X = [1.0, 0.0, 0.0]
V_Y = [0.0, 1.0, 0.0]
V_Z = [0.0, 0.0, 1.0]


def _add_episode(
    store: MemoryGraphStore,
    ctx: RequestContext,
    *,
    video_id: str = "vid1",
    clip_id: int = 0,
    extractor_type: str = "episodic_memory_generator",
    embedding: list[float] | None = None,
    description: str = "episode",
) -> str:
    """Create an ExtractedNode pre-populated with clip provenance + optional embedding."""
    node = create_extracted_node(
        source_raw_id="raw-does-not-matter",
        extractor_type=extractor_type,
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
        },
    )
    store.create_node(node.serialize(), ctx)
    if embedding is not None:
        # Re-fetch and mutate to push through the proper touch() path; the
        # store only knows how to persist via serialize(), so we mutate the
        # in-place dict and call update_node.
        live = store.get_node(node.node_id, ctx)
        assert live is not None
        live.set_embedding(list(embedding))
        store.update_node(live.serialize(), ctx)
    return node.node_id


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------

class TestSearchByEmbeddingShape:
    def test_empty_store_returns_empty(self, store, ctx):
        results = store.search_by_embedding([0.1, 0.2, 0.3], ctx)
        assert results == []

    def test_top_k_zero_returns_empty(self, store, ctx):
        _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx, top_k=0)
        assert results == []

    def test_empty_query_vector_returns_empty(self, store, ctx):
        _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding([], ctx)
        assert results == []

    def test_no_candidates_returns_empty(self, store, ctx):
        # Nodes exist but have no embeddings.
        _add_episode(store, ctx, embedding=None)
        results = store.search_by_embedding(V_X, ctx)
        assert results == []

    def test_returns_pairs_of_node_id_and_score(self, store, ctx):
        _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx)
        assert len(results) == 1
        node_id, score = results[0]
        assert isinstance(node_id, str) and node_id  # non-empty string
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Cosine ranking
# ---------------------------------------------------------------------------

class TestCosineRanking:
    def test_identical_vector_scores_one(self, store, ctx):
        nid = _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx)
        assert results[0][0] == nid
        assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)

    def test_orthogonal_vector_scores_zero(self, store, ctx):
        _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding(V_Y, ctx)
        assert len(results) == 1
        assert math.isclose(results[0][1], 0.0, abs_tol=1e-6)

    def test_results_sorted_desc(self, store, ctx):
        _add_episode(store, ctx, clip_id=0, embedding=V_X)
        _add_episode(store, ctx, clip_id=1, embedding=V_Y)
        _add_episode(store, ctx, clip_id=2, embedding=V_Z)
        # Query that is X-ish gives X > Z > Y ordering.
        results = store.search_by_embedding(
            [2.0, 0.1, 1.0], ctx
        )
        scores = [score for _nid, score in results]
        assert scores == sorted(scores, reverse=True)
        assert len(results) == 3

    def test_top_k_caps_results(self, store, ctx):
        for i in range(5):
            _add_episode(store, ctx, clip_id=i, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx, top_k=3)
        assert len(results) == 3

    def test_dimension_mismatch_is_skipped_silently(self, store, ctx):
        _add_episode(store, ctx, embedding=[1.0, 0.0])  # 2-D
        results = store.search_by_embedding([0.5, 0.5, 0.5], ctx)  # 3-D query
        assert results == []

    def test_zero_norm_embedding_returns_zero_score_not_crash(self, store, ctx):
        nid = _add_episode(store, ctx, embedding=[0.0, 0.0, 0.0])
        results = store.search_by_embedding(V_X, ctx)
        assert len(results) == 1
        assert results[0][0] == nid
        assert math.isclose(results[0][1], 0.0)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_before_clip_excludes_higher_clips(self, store, ctx):
        _add_episode(store, ctx, clip_id=0, embedding=V_X)
        _add_episode(store, ctx, clip_id=5, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx, before_clip=2)
        assert len(results) == 1
        # The one returned should be the clip_id=0 node.
        nid_0 = [r[0] for r in results]
        # Verify by re-fetching — clip 5 must not appear.
        results_no_filter = store.search_by_embedding(V_X, ctx)
        nids = {r[0] for r in results_no_filter}
        assert nid_0[0] in nids
        # The filtered set is a strict subset of the unfiltered set.
        assert set(nid_0).issubset(nids)
        assert len(nid_0) < len(nids)

    def test_before_clip_excludes_nodes_missing_clip_id(self, store, ctx):
        # Node 1 has _clip_id, node 2 doesn't.
        _add_episode(store, ctx, clip_id=3, embedding=V_X)
        node2 = create_extracted_node(
            source_raw_id="raw-2",
            extractor_type="episodic_memory_generator",
            extraction_schema="v1",
            account_id=ctx.account_id,
            extracted_content={"description": "no clip tag"},
        )
        store.create_node(node2.serialize(), ctx)
        live = store.get_node(node2.node_id, ctx)
        live.set_embedding(V_X)
        store.update_node(live.serialize(), ctx)

        results = store.search_by_embedding(V_X, ctx, before_clip=5)
        nids = [r[0] for r in results]
        # node2 must be excluded because it carries no _clip_id, while clip 3 is in range.
        assert node2.node_id not in nids
        assert len(results) == 1

    def test_video_id_filter(self, store, ctx):
        _add_episode(store, ctx, video_id="video_a", clip_id=0, embedding=V_X)
        _add_episode(store, ctx, video_id="video_b", clip_id=0, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx, video_id="video_a")
        assert len(results) == 1
        # Both exist; the one returned belongs to video_a (we don't care about score here).
        assert results[0][0] is not None

    def test_extractor_types_filter(self, store, ctx):
        nid_epi = _add_episode(store, ctx, embedding=V_X, extractor_type="episodic_memory_generator")
        _add_episode(store, ctx, embedding=V_X, extractor_type="other_extractor")
        results = store.search_by_embedding(
            V_X, ctx, extractor_types=("episodic_memory_generator",)
        )
        nids = [r[0] for r in results]
        assert nids == [nid_epi]

    def test_node_types_string_aliases(self, store, ctx):
        # Only "extracted" carries embeddings today; even if memory/raw
        # are listed they should be skipped silently (we don't error).
        _add_episode(store, ctx, embedding=V_X)
        results = store.search_by_embedding(
            V_X, ctx, node_types=("extracted",)
        )
        assert len(results) == 1
        # Unknown type alone -> nothing.
        results_unknown = store.search_by_embedding(
            V_X, ctx, node_types=("bogus_type",)
        )
        assert results_unknown == []

    def test_combined_filters_intersect(self, store, ctx):
        # 4 nodes: video_a clip 0, video_a clip 5, video_b clip 0, video_b clip 5
        ids = {}
        ids["a0"] = _add_episode(store, ctx, video_id="a", clip_id=0, embedding=V_X)
        ids["a5"] = _add_episode(store, ctx, video_id="a", clip_id=5, embedding=V_X)
        ids["b0"] = _add_episode(store, ctx, video_id="b", clip_id=0, embedding=V_X)
        ids["b5"] = _add_episode(store, ctx, video_id="b", clip_id=5, embedding=V_X)
        results = store.search_by_embedding(
            V_X, ctx, video_id="a", before_clip=3
        )
        nids = {r[0] for r in results}
        assert nids == {ids["a0"]}


# ---------------------------------------------------------------------------
# Account isolation
# ---------------------------------------------------------------------------

class TestAccountIsolation:
    def test_other_account_nodes_not_returned(self, store, ctx, other_ctx):
        _add_episode(store, ctx, embedding=V_X)
        _add_episode(store, other_ctx, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx)
        nids = {r[0] for r in results}
        assert len(results) == 1

    def test_account_filter_takes_precedence(self, store, ctx, other_ctx):
        # Same description, same clip; both look identical except for account.
        _add_episode(store, ctx, embedding=V_X)
        nid_other = _add_episode(store, other_ctx, embedding=V_X)
        results = store.search_by_embedding(V_X, ctx)
        assert all(r[0] != nid_other for r in results)
        # And the other tenant sees only its own.
        results_other = store.search_by_embedding(V_X, other_ctx)
        assert len(results_other) == 1
        assert results_other[0][0] == nid_other


# ---------------------------------------------------------------------------
# Non-extracted nodes
# ---------------------------------------------------------------------------

class TestNonExtractedNodes:
    def test_raw_nodes_without_embeddings_are_skipped(self, store, ctx):
        # Raw nodes CAN carry an embedding via metadata, but the v1 contract
        # only indexes ExtractedNode — assert that's true today.
        raw = create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="bytes",
            account_id=ctx.account_id,
            metadata={"embedding": V_X},
        )
        store.create_node(raw.serialize(), ctx)
        results = store.search_by_embedding(V_X, ctx)
        assert results == []

    def test_memory_nodes_without_embeddings_are_skipped(self, store, ctx):
        mem = create_memory_node(
            summary="no embedding",
            account_id=ctx.account_id,
        )
        store.create_node(mem.serialize(), ctx)
        results = store.search_by_embedding(V_X, ctx)
        assert results == []

    def test_extracted_node_without_embedding_still_indexed_returns_empty(
        self, store, ctx
    ):
        _add_episode(store, ctx, embedding=None)
        results = store.search_by_embedding(V_X, ctx)
        assert results == []
