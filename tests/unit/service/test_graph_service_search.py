"""Unit tests for ``GraphService`` control-loop primitives.

Covers:
- ``ClipScope`` / ``EvidenceHit`` dataclass invariants.
- ``search_by_embedding`` accepts both ``str`` (via embedder) and
  ``list[float]`` (raw) queries, returns ``EvidenceHit``s, and
  threads a ``ClipScope`` through to the store-level filter.
- ``get_clips_summary`` groups ExtractedNodes by clip ordinal and
  surfaces the episodic_memory_generator node when present.

All tests run against the in-memory ``MemoryGraphStore`` so the
``GraphService`` layer can be exercised without external services.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.graph_enums import GraphNodeType
from core.graph_models import (
    create_extracted_node,
    create_memory_node,
    create_raw_node,
)
from core.graph_enums import RawDataType
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import (
    ClipScope,
    EvidenceHit,
    GraphService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acc_service",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        trace_id="t1",
    )


@pytest.fixture
def store():
    return MemoryGraphStore()


@pytest.fixture
def service(store):
    return GraphService(store)


# Orthonormal 3-D basis vectors.
V_X = [1.0, 0.0, 0.0]
V_Y = [0.0, 1.0, 0.0]
V_Z = [0.0, 0.0, 1.0]


def _add_episode(
    service: GraphService,
    ctx: RequestContext,
    *,
    video_id: str = "vid1",
    clip_id: int = 0,
    embedding: list[float] | None = None,
    extractor_type: str = "episodic_memory_generator",
    description: str = "episode",
) -> str:
    """Add one ExtractedNode through the service layer with clip provenance."""
    import uuid

    node = create_extracted_node(
        source_raw_id=str(uuid.uuid4()),
        extractor_type=extractor_type,
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
        },
    )
    service._store.create_node(node.serialize(), ctx)  # type: ignore[attr-defined]
    if embedding is not None:
        live = service._store.get_node(node.node_id, ctx)  # type: ignore[attr-defined]
        assert live is not None
        live.set_embedding(list(embedding))
        service._store.update_node(live.serialize(), ctx)  # type: ignore[attr-defined]
    return node.node_id


# ---------------------------------------------------------------------------
# Tiny Embedder stub for the str-query path. Deterministic and small.
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Maps each unique text to a fixed orthogonal basis vector.

    Returns 3-D vectors so they match the V_X / V_Y / V_Z fixtures above
    and the cosine math stays exact. Distinct texts map to distinct
    basis vectors via a tiny hash dispatch.
    """

    def __init__(self) -> None:
        self._slot: dict[str, list[float]] = {
            "x": V_X,
            "y": V_Y,
            "z": V_Z,
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            key = t.strip().lower()[:1]
            out.append(list(self._slot.get(key, V_X)))
        return out


# ---------------------------------------------------------------------------
# ClipScope
# ---------------------------------------------------------------------------

class TestClipScope:
    def test_defaults_are_none(self):
        scope = ClipScope()
        assert scope.video_id is None
        assert scope.before_clip is None

    def test_partial_before_clip_only(self):
        # `before_clip` without a `video_id` is the M3-Agent control-loop
        # default — pin that the partial form is supported.
        scope = ClipScope(before_clip=3)
        assert scope.video_id is None
        assert scope.before_clip == 3

    def test_partial_video_id_only(self):
        # And the other partial form — the M3 agent sometimes wants to
        # scope by video without bounding by clip.
        scope = ClipScope(video_id="vid1")
        assert scope.video_id == "vid1"
        assert scope.before_clip is None

    def test_frozen(self):
        scope = ClipScope(video_id="v1", before_clip=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.video_id = "v2"  # type: ignore[misc]

    def test_equality(self):
        # Frozen dataclasses are equal-by-value (good for caching / set use).
        assert ClipScope("v1", 5) == ClipScope("v1", 5)
        assert ClipScope("v1", 5) != ClipScope("v1", 6)
        assert ClipScope("v1", 5) != ClipScope("v2", 5)


# ---------------------------------------------------------------------------
# EvidenceHit
# ---------------------------------------------------------------------------

class TestEvidenceHit:
    def test_carries_node_and_score(self):
        node = create_extracted_node(
            source_raw_id="raw",
            extractor_type="t",
            extraction_schema="v1",
            account_id="acc",
        )
        hit = EvidenceHit(node=node, score=0.5)
        assert hit.node is node
        assert hit.score == 0.5
        assert hit.video_id is None
        assert hit.clip_id is None

    def test_frozen(self):
        node = create_extracted_node(
            source_raw_id="raw",
            extractor_type="t",
            extraction_schema="v1",
            account_id="acc",
        )
        hit = EvidenceHit(node=node, score=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# search_by_embedding — list[float] path
# ---------------------------------------------------------------------------

class TestSearchByEmbeddingVectorPath:
    def test_returns_evidence_hits(self, service, ctx):
        nid = _add_episode(service, ctx, embedding=V_X)
        hits = service.search_by_embedding(V_X, ctx)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, EvidenceHit)
        assert hit.node.node_id == nid
        assert hit.score == pytest.approx(1.0, abs=1e-6)

    def test_lifts_video_and_clip_provenance(self, service, ctx):
        _add_episode(service, ctx, video_id="vid_a", clip_id=7, embedding=V_X)
        hits = service.search_by_embedding(V_X, ctx)
        assert len(hits) == 1
        assert hits[0].video_id == "vid_a"
        assert hits[0].clip_id == 7

    def test_results_sorted_desc(self, service, ctx):
        _add_episode(service, ctx, clip_id=0, embedding=V_X)
        _add_episode(service, ctx, clip_id=1, embedding=V_Y)
        _add_episode(service, ctx, clip_id=2, embedding=V_Z)
        hits = service.search_by_embedding([2.0, 0.1, 1.0], ctx)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_results(self, service, ctx):
        for i in range(5):
            _add_episode(service, ctx, clip_id=i, embedding=V_X)
        hits = service.search_by_embedding(V_X, ctx, top_k=3)
        assert len(hits) == 3

    def test_accepts_tuple_query(self, service, ctx):
        _add_episode(service, ctx, embedding=V_X)
        hits = service.search_by_embedding(tuple(V_X), ctx)
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# search_by_embedding — str path
# ---------------------------------------------------------------------------

class TestSearchByEmbeddingStringPath:
    def test_embeds_via_embedder(self, service, ctx):
        nid = _add_episode(service, ctx, embedding=V_X)
        embedder = _StubEmbedder()
        # StubEmbedder maps "x*" -> V_X so this should match.
        hits = service.search_by_embedding("x marks the spot", ctx, embedder=embedder)
        assert len(hits) == 1
        assert hits[0].node.node_id == nid

    def test_str_without_embedder_raises(self, service, ctx):
        with pytest.raises(ValueError, match="embedder is required"):
            service.search_by_embedding("hello", ctx)

    def test_empty_embedder_response_returns_empty(self, service, ctx):
        _add_episode(service, ctx, embedding=V_X)

        class _EmptyEmbedder:
            def embed_texts(self, texts):
                return [[]]  # empty vector

        hits = service.search_by_embedding(
            "anything", ctx, embedder=_EmptyEmbedder()
        )
        assert hits == []


# ---------------------------------------------------------------------------
# search_by_embedding — scope / type errors
# ---------------------------------------------------------------------------

class TestSearchByEmbeddingScope:
    def test_scope_applies_video_filter(self, service, ctx):
        _add_episode(service, ctx, video_id="vid_a", embedding=V_X)
        _add_episode(service, ctx, video_id="vid_b", embedding=V_X)
        scope = ClipScope(before_clip=99, video_id="vid_a")
        hits = service.search_by_embedding(V_X, ctx, scope=scope)
        assert len(hits) == 1
        assert hits[0].video_id == "vid_a"

    def test_scope_applies_before_clip_filter(self, service, ctx):
        _add_episode(service, ctx, clip_id=0, embedding=V_X)
        _add_episode(service, ctx, clip_id=5, embedding=V_X)
        scope = ClipScope(before_clip=2, video_id="vid1")
        hits = service.search_by_embedding(V_X, ctx, scope=scope)
        # One node (clip_id=0) survives; clip_id=5 is excluded.
        assert len(hits) == 1
        assert hits[0].clip_id == 0

    def test_extractor_types_filter(self, service, ctx):
        _add_episode(
            service, ctx, embedding=V_X, extractor_type="episodic_memory_generator"
        )
        _add_episode(service, ctx, embedding=V_X, extractor_type="other_extractor")
        hits = service.search_by_embedding(
            V_X, ctx, extractor_types=("episodic_memory_generator",)
        )
        assert len(hits) == 1

    def test_invalid_query_type_raises(self, service, ctx):
        with pytest.raises(TypeError, match="query must be str or list"):
            service.search_by_embedding(42, ctx)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_clips_summary
# ---------------------------------------------------------------------------

class TestGetClipsSummary:
    def test_empty_account_returns_empty(self, service, ctx):
        assert service.get_clips_summary("any_video", ctx) == []

    def test_groups_by_clip_and_sorts(self, service, ctx):
        # Two videos, three clips in the target video, plus other-video noise.
        for clip in (3, 0, 1):
            _add_episode(service, ctx, video_id="target", clip_id=clip, embedding=V_X)
        _add_episode(service, ctx, video_id="other", clip_id=0, embedding=V_Y)
        summary = service.get_clips_summary("target", ctx)
        assert [row["clip_id"] for row in summary] == [0, 1, 3]
        assert all(row["extracted_count"] == 1 for row in summary)

    def test_surfaces_episodic_id(self, service, ctx):
        epi_id = _add_episode(
            service,
            ctx,
            video_id="vid1",
            clip_id=2,
            extractor_type="episodic_memory_generator",
            embedding=V_X,
        )
        # Add a non-episodic node for the same clip — episodic_id should
        # still resolve to the episodic_memory_generator node.
        _add_episode(
            service, ctx, video_id="vid1", clip_id=2,
            extractor_type="face_detection", embedding=V_Y,
        )
        summary = service.get_clips_summary("vid1", ctx)
        assert len(summary) == 1
        assert summary[0]["clip_id"] == 2
        assert summary[0]["extracted_count"] == 2
        assert summary[0]["episodic_id"] == epi_id

    def test_episodic_id_none_when_no_episodic_node(self, service, ctx):
        _add_episode(
            service, ctx, video_id="vid1", clip_id=0,
            extractor_type="face_detection", embedding=V_X,
        )
        summary = service.get_clips_summary("vid1", ctx)
        assert summary[0]["episodic_id"] is None

    def test_skips_nodes_without_clip_id(self, service, ctx):
        # Manually add a node with no _clip_id provenance; should be
        # skipped (and not crash).
        from core.graph_models import create_extracted_node
        import uuid

        node = create_extracted_node(
            source_raw_id=str(uuid.uuid4()),
            extractor_type="episodic_memory_generator",
            extraction_schema="v1",
            account_id=ctx.account_id,
            extracted_content={"description": "no clip tag", "_video_id": "vid1"},
        )
        service._store.create_node(node.serialize(), ctx)  # type: ignore[attr-defined]
        # Should be skipped: no _clip_id.
        assert service.get_clips_summary("vid1", ctx) == []
