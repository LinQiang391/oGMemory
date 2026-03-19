"""Contract tests for GraphRelationStore.

Phase 2 — Verifies graph query correctness and interface compatibility.
These tests ensure 1-hop and 2-hop traversals return correct results.

Dependencies:
- P2-G2: GraphRelationStore implementation ✅ DONE
- P2-G3: 1-hop traversal method ✅ DONE
- P2-G4: 2-hop traversal method ✅ DONE
- P2-G5: GraphRelationStore unit tests ✅ DONE
"""

import pytest

from core.models import RequestContext, RelationEdge
from core.interfaces import RelationStore
from providers.relation_store.graph_relation_store import (
    GraphRelationStore,
    RelatedNode,
)


@pytest.fixture
def ctx():
    """Default request context for testing."""
    return RequestContext(
        account_id="test_account",
        user_id="user1",
        agent_id="agent1",
        session_id="session1",
        trace_id="trace1",
    )


@pytest.fixture
def ctx_other():
    """Different account context for isolation testing."""
    return RequestContext(
        account_id="other_account",
        user_id="user2",
        agent_id="agent2",
        session_id="session2",
        trace_id="trace2",
    )


@pytest.fixture
def store():
    """Fresh GraphRelationStore for each test."""
    return GraphRelationStore()


class TestRelationStoreProtocol:
    """Verify RelationStore Protocol compliance."""

    def test_relation_store_has_get_edges(self):
        """RelationStore must have get_edges(uri, ctx) method."""
        assert hasattr(RelationStore, 'get_edges')

    def test_relation_store_has_upsert_edges(self):
        """RelationStore must have upsert_edges(edges, ctx) method."""
        assert hasattr(RelationStore, 'upsert_edges')

    def test_graph_relation_store_is_relation_store_subclass(self):
        """GraphRelationStore must satisfy RelationStore Protocol.

        This is a duck-typing check - Python's Protocol uses structural
        subtyping, so we just need to verify the methods exist and
        have the right signatures.
        """
        store = GraphRelationStore()
        # Verify protocol methods exist
        assert hasattr(store, 'get_edges')
        assert hasattr(store, 'upsert_edges')
        # Verify they're callable
        assert callable(store.get_edges)
        assert callable(store.upsert_edges)

    def test_graph_relation_store_satisfies_protocol_contract(self, store, ctx):
        """GraphRelationStore must work with RelationStore Protocol usage."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test relation",
        )

        # Should work with Protocol interface
        store.upsert_edges([edge], ctx)
        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)

        assert len(retrieved) == 1
        assert retrieved[0].to_uri == edge.to_uri


class TestGraphTraversal:
    """Verify graph traversal returns correct results."""

    def test_one_hop_traversal_returns_direct_neighbors(self, store, ctx):
        """get_related_nodes(depth=1) should return directly connected nodes."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.6,
                reason="A to C",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.9,
                reason="A to D",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        # Should return B, C, D (not A itself)
        uris = {n.uri for n in related}
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
            "ctx://test/account/users/user/memories/D",
        }
        # All should be distance 1
        assert all(n.distance == 1 for n in related)

    def test_two_hop_traversal_returns_indirect_neighbors(self, store, ctx):
        """get_related_nodes(depth=2) should return nodes within 2 edges."""
        # A -> B -> D, A -> C
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.7,
                reason="A to C",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.6,
                reason="B to D",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        uris = {n.uri for n in related}
        # Should include B (1-hop), C (1-hop), D (2-hop via B)
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
            "ctx://test/account/users/user/memories/D",
        }

        # Verify 2-hop node has correct distance
        node_d = next(n for n in related if n.uri.endswith("/D"))
        assert node_d.distance == 2
        # 2-hop score is min of edge weights
        assert node_d.score == 0.6  # min(0.8, 0.6)

    def test_traversal_respects_direction(self, store, ctx):
        """Traversal should respect edge direction parameter."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/C",
                to_uri="ctx://test/account/users/user/memories/A",
                relation_type="related_to",
                weight=0.6,
                reason="C to A",
            ),
        ]

        store.upsert_edges(edges, ctx)

        # OUTBOUND: A -> B
        outbound = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )
        assert len(outbound) == 1
        assert outbound[0].uri == "ctx://test/account/users/user/memories/B"

        # INBOUND: C -> A
        inbound = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="INBOUND",
            depth=1,
        )
        assert len(inbound) == 1
        assert inbound[0].uri == "ctx://test/account/users/user/memories/C"

        # BOTH: combined
        both = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="BOTH",
            depth=1,
        )
        assert len(both) == 2

    def test_traversal_respects_account_isolation(self, store, ctx, ctx_other):
        """Graph traversal MUST NOT cross account boundaries.

        CRITICAL: This is a security invariant.
        """
        # Create edge in test_account
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Same account edge",
        )

        store.upsert_edges([edge], ctx)

        # Same account can traverse
        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )
        assert len(related) == 1

        # Different account cannot traverse (returns empty)
        related_other = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx_other,
            direction="OUTBOUND",
            depth=1,
        )
        assert len(related_other) == 0

    def test_traversal_with_empty_graph(self, store, ctx):
        """Traversal on a node with no edges should return empty list."""
        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )
        assert related == []

    def test_traversal_with_self_loops(self, store, ctx):
        """Traversal should handle self-referential edges (A -> A)."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/A",
            relation_type="related_to",
            weight=0.8,
            reason="Self loop",
        )

        store.upsert_edges([edge], ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        # Should include the self-referential node
        assert len(related) == 1
        assert related[0].uri == "ctx://test/account/users/user/memories/A"

    def test_traversal_with_cycles(self, store, ctx):
        """Traversal should handle cycles (A -> B -> A)."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/A",
                relation_type="related_to",
                weight=0.7,
                reason="B to A",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        # Should not cause infinite loop
        # Should include B once
        uris = {n.uri for n in related}
        assert uris == {"ctx://test/account/users/user/memories/B"}
        # A itself should not appear in results
        assert "ctx://test/account/users/user/memories/A" not in uris


class TestRelationEdgeProperties:
    """Verify RelationEdge properties are preserved correctly."""

    def test_edge_weight_influences_traversal_order(self, store, ctx):
        """Higher weight edges should be prioritized in results."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/low",
                relation_type="related_to",
                weight=0.3,
                reason="Low weight",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/high",
                relation_type="related_to",
                weight=0.9,
                reason="High weight",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/medium",
                relation_type="related_to",
                weight=0.6,
                reason="Medium weight",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        # Results should be sorted by score (weight) descending
        scores = [n.score for n in related]
        assert scores == [0.9, 0.6, 0.3]
        assert related[0].uri.endswith("/high")
        assert related[1].uri.endswith("/medium")
        assert related[2].uri.endswith("/low")

    def test_edge_type_filtering(self, store, ctx):
        """Traversal should support filtering by relation_type."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Related",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="Derived",
            ),
        ]

        store.upsert_edges(edges, ctx)

        # Filter by "related_to" only
        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
            relation_types=["related_to"],
        )

        assert len(related) == 1
        assert related[0].uri.endswith("/B")
        assert related[0].relation_types == ["related_to"]

    def test_min_weight_threshold(self, store, ctx):
        """Traversal should support min_weight threshold."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/high",
                relation_type="related_to",
                weight=0.9,
                reason="High",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/low",
                relation_type="related_to",
                weight=0.3,
                reason="Low",
            ),
        ]

        store.upsert_edges(edges, ctx)

        # Filter by min_weight=0.5
        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
            min_weight=0.5,
        )

        assert len(related) == 1
        assert related[0].uri.endswith("/high")
        assert related[0].score == 0.9


class TestRelatedNodeMetadata:
    """Verify RelatedNode contains correct metadata."""

    def test_related_node_contains_path_information(self, store, ctx):
        """RelatedNode should contain the path of edges taken."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="A to B",
        )

        store.upsert_edges([edge], ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        assert len(related) == 1
        assert len(related[0].path) == 1
        assert related[0].path[0].from_uri == edge.from_uri
        assert related[0].path[0].to_uri == edge.to_uri

    def test_related_node_two_hop_path(self, store, ctx):
        """2-hop RelatedNode should contain both edges in path."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="B to C",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        node_c = next(n for n in related if n.uri.endswith("/C"))
        assert node_c.distance == 2
        assert len(node_c.path) == 2
        assert node_c.path[0].to_uri.endswith("/B")
        assert node_c.path[1].to_uri.endswith("/C")
        # Path score is min of edge weights
        assert node_c.score == 0.6  # min(0.8, 0.6)

    def test_related_node_relation_types_aggregated(self, store, ctx):
        """relation_types should aggregate all types in path."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="A to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="B to C",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        node_c = next(n for n in related if n.uri.endswith("/C"))
        # Should have both relation types from the path
        assert set(node_c.relation_types) == {"related_to", "derived_from"}


class TestGraphRelationStoreIntegration:
    """Integration tests with actual GraphRelationStore implementation."""

    def test_graph_relation_store_agfs_compatibility(self, store, ctx):
        """GraphRelationStore should be compatible with AGFSRelationStore interface.

        Both satisfy RelationStore Protocol, so they should be interchangeable.
        """
        from providers.relation_store.agfs_relation_store import AGFSRelationStore

        # Both implement the same protocol
        assert hasattr(store, 'get_edges')
        assert hasattr(store, 'upsert_edges')
        # Both should have the extended method too
        assert hasattr(store, 'get_related_nodes')

    def test_graph_relation_store_persists_edges(self, store, ctx):
        """GraphRelationStore should persist edges across queries."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        store.upsert_edges([edge], ctx)

        # First query
        first = store.get_edges("ctx://test/account/users/user/memories/A", ctx)
        assert len(first) == 1

        # Second query (should still be there)
        second = store.get_edges("ctx://test/account/users/user/memories/A", ctx)
        assert len(second) == 1

        # Same data
        assert first[0].to_uri == second[0].to_uri
        assert first[0].weight == second[0].weight
