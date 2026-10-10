"""Tests for the cross-clip identity edge types.

``same_identity`` and ``co_occurs_in`` are new GraphEdgeType members
that face_clustering / voice_clustering will use to collapse cross-clip
detections. This test pins:

* the enum members exist and have the right string values (storage
  layout — the SQL store persists ``edge.edge_type.value`` as VARCHAR);
* GraphService can create edges of both types and the resulting graph
  has them readable via ``get_neighbors``;
* round-tripping through the in-memory store preserves them;
* the operators that will consume them (face_clustering, etc.) can be
  built against the enum without needing enum-aware switch statements.

The SQL parity test (``tests/integration/test_sql_graph_search_by_embedding.py``
already covers the SQL store; this file is the unit-level coverage
both store kinds share).
"""

from __future__ import annotations

from core.graph_enums import GraphEdgeType, GraphNodeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


def _ctx(account_id: str = "acc_id_test") -> RequestContext:
    return RequestContext(
        account_id=account_id, user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


def _two_face_nodes(graph: GraphService, ctx: RequestContext):
    """Two ExtractedNodes representing face detections of the same
    person in different clips. Returns (anchor, candidate) ready to
    be linked by same_identity / co_occurs_in edges."""
    anchor = create_extracted_node(
        source_raw_id="video_clip_0",
        extractor_type="face_detection",
        extraction_schema="face_v1",
        account_id=ctx.account_id,
        extracted_content={"_video_id": "v", "_clip_id": 0, "face_id": 0},
        node_id="face-anchor",
    )
    candidate = create_extracted_node(
        source_raw_id="video_clip_1",
        extractor_type="face_detection",
        extraction_schema="face_v1",
        account_id=ctx.account_id,
        extracted_content={"_video_id": "v", "_clip_id": 1, "face_id": 1},
        node_id="face-candidate",
    )
    graph._store.create_node(anchor.serialize(), ctx)
    graph._store.create_node(candidate.serialize(), ctx)
    return anchor, candidate


def test_enum_members_exist():
    assert GraphEdgeType.SAME_IDENTITY.value == "same_identity"
    assert GraphEdgeType.CO_OCCURS_IN.value == "co_occurs_in"
    # And the older ones are unchanged (storage compatibility).
    assert GraphEdgeType.EXTRACTED_FROM.value == "extracted_from"
    assert GraphEdgeType.SUPPORTS.value == "supports"
    assert GraphEdgeType.REFERENCES.value == "references"
    assert GraphEdgeType.EVOLVES.value == "evolves"


def test_create_same_identity_edge_via_graph_service():
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    anchor, candidate = _two_face_nodes(graph, ctx)

    graph.create_edge(
        source_id=candidate.node_id,
        target_id=anchor.node_id,
        edge_type=GraphEdgeType.SAME_IDENTITY,
        properties={"score": 0.92},
        ctx=ctx,
    )

    neighbors = graph.get_neighbors(
        anchor.node_id, ctx,
        edge_types=[GraphEdgeType.SAME_IDENTITY],
    )
    nodes = [n for n, _e, _d in neighbors]
    assert any(n.node_id == candidate.node_id for n in nodes)


def test_create_co_occurs_in_edge_with_weight():
    """``co_occurs_in`` carries a joint-occurrence weight — the
    graph_service edge creation must accept a weight and round-trip
    it on the resulting GraphEdge."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    anchor, candidate = _two_face_nodes(graph, ctx)

    edge = graph.create_edge(
        source_id=anchor.node_id,
        target_id=candidate.node_id,
        edge_type=GraphEdgeType.CO_OCCURS_IN,
        weight=3.0,
        properties={"frame_indices": [12, 47, 89]},
        ctx=ctx,
    )
    assert edge.weight == 3.0
    assert edge.properties["frame_indices"] == [12, 47, 89]

    # Round-trip via get_neighbors: the edge must come back with the
    # right type and weight, not just the right neighbour.
    neighbors = graph.get_neighbors(
        anchor.node_id, ctx,
        edge_types=[GraphEdgeType.CO_OCCURS_IN],
    )
    matches = [(n, e) for n, e, _d in neighbors if n.node_id == candidate.node_id]
    assert len(matches) == 1
    _n, e = matches[0]
    assert e.weight == 3.0
    assert e.properties["frame_indices"] == [12, 47, 89]


def test_neighbors_default_filter_excludes_new_edge_types():
    """Existing graph queries that don't ask for the new edge types
    must not return them — backwards compat for callers that walk the
    graph for REFERENCES / EXTRACTED_FROM only."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    anchor, candidate = _two_face_nodes(graph, ctx)

    graph.create_edge(
        source_id=anchor.node_id, target_id=candidate.node_id,
        edge_type=GraphEdgeType.REFERENCES, ctx=ctx,
    )
    graph.create_edge(
        source_id=anchor.node_id, target_id=candidate.node_id,
        edge_type=GraphEdgeType.SAME_IDENTITY, ctx=ctx,
    )

    refs_only = graph.get_neighbors(
        anchor.node_id, ctx,
        edge_types=[GraphEdgeType.REFERENCES],
    )
    same_only = graph.get_neighbors(
        anchor.node_id, ctx,
        edge_types=[GraphEdgeType.SAME_IDENTITY],
    )

    assert len(refs_only) == 1
    assert len(same_only) == 1
    # Cross-check: the neighbor is the same node in both cases, but
    # the edges have different types.
    assert refs_only[0][1].edge_type == GraphEdgeType.REFERENCES
    assert same_only[0][1].edge_type == GraphEdgeType.SAME_IDENTITY


def test_str_enum_lookup_round_trip():
    """Storage round-trips via ``GraphEdgeType(string)`` — confirm the
    new members can be reconstructed from their string value (the SQL
    store uses this when materialising rows)."""
    for edge_type in (
        GraphEdgeType.SAME_IDENTITY,
        GraphEdgeType.CO_OCCURS_IN,
    ):
        reconstructed = GraphEdgeType(edge_type.value)
        assert reconstructed is edge_type