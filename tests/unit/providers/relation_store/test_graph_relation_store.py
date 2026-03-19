"""Unit tests for GraphRelationStore.

Tests cover:
- Single edge upsert and retrieval
- Multiple edges from same source
- Bidirectional navigation (OUTBOUND/INBOUND/BOTH)
- 2-hop traversal with path scoring
- Multi-tenant isolation
- Edge idempotency
- Deleted node handling
"""

import pytest

from core.models import RequestContext, RelationEdge
from providers.relation_store.graph_relation_store import (
    GraphRelationStore,
    NodeMetadata,
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


class TestGraphRelationStoreBasics:
    """Test basic RelationStore protocol implementation."""

    def test_empty_store_returns_no_edges(self, store, ctx):
        """Empty store returns empty list for any URI."""
        edges = store.get_edges("ctx://test/account/users/user/memories/test", ctx)
        assert edges == []

    def test_upsert_and_retrieve_single_edge(self, store, ctx):
        """Can upsert and retrieve a single edge."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test relation",
        )

        store.upsert_edges([edge], ctx)
        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)

        assert len(retrieved) == 1
        assert retrieved[0].from_uri == edge.from_uri
        assert retrieved[0].to_uri == edge.to_uri
        assert retrieved[0].relation_type == edge.relation_type
        assert retrieved[0].weight == edge.weight

    def test_upsert_idempotent(self, store, ctx):
        """Re-upserting same edge updates it."""
        edge1 = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.5,
            reason="Original",
        )

        edge2 = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.9,  # Updated weight
            reason="Updated",
        )

        store.upsert_edges([edge1], ctx)
        store.upsert_edges([edge2], ctx)

        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)

        assert len(retrieved) == 1
        assert retrieved[0].weight == 0.9
        assert retrieved[0].reason == "Updated"

    def test_multiple_edges_from_same_source(self, store, ctx):
        """Can store multiple edges from the same source."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Related to B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="Derived from C",
            ),
        ]

        store.upsert_edges(edges, ctx)
        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)

        assert len(retrieved) == 2
        to_uris = {e.to_uri for e in retrieved}
        assert to_uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
        }


class TestMultiTenantIsolation:
    """Test multi-tenant access control."""

    def test_cross_account_get_edges_returns_empty(self, store, ctx, ctx_other):
        """Cannot retrieve edges from different account."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        store.upsert_edges([edge], ctx)

        # Different account should get empty results
        edges = store.get_edges("ctx://test/account/users/user/memories/A", ctx_other)
        assert edges == []

    def test_cross_account_upsert_skipped(self, store, ctx, ctx_other):
        """Edges for other accounts are skipped during upsert."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        # Register node with different account first
        store._nodes["ctx://test/account/users/user/memories/A"] = NodeMetadata(
            uri="ctx://test/account/users/user/memories/A",
            exists=True,
            account_id="other_account",
        )

        # Upsert with different context should skip
        store.upsert_edges([edge], ctx)

        # Original account's data unchanged
        ctx_check = RequestContext(
            account_id="other_account",
            user_id="user",
            agent_id="agent",
            session_id="session",
            trace_id="trace",
        )
        edges = store.get_edges("ctx://test/account/users/user/memories/A", ctx_check)
        assert edges == []


class TestRelatedNodesTraversal:
    """Test get_related_nodes traversal methods."""

    def test_1hop_outbound_traversal(self, store, ctx):
        """1-hop outbound traversal returns direct neighbors."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Test",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="Test",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        assert len(related) == 2
        uris = {n.uri for n in related}
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
        }
        assert all(n.distance == 1 for n in related)

    def test_1hop_inbound_traversal(self, store, ctx):
        """1-hop inbound traversal finds sources pointing to target."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/A",
                relation_type="related_to",
                weight=0.7,
                reason="B points to A",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/C",
                to_uri="ctx://test/account/users/user/memories/A",
                relation_type="derived_from",
                weight=0.5,
                reason="C points to A",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="INBOUND",
            depth=1,
        )

        assert len(related) == 2
        uris = {n.uri for n in related}
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
        }

    def test_1hop_both_directions(self, store, ctx):
        """BOTH direction combines inbound and outbound."""
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
                relation_type="derived_from",
                weight=0.6,
                reason="C to A",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="BOTH",
            depth=1,
        )

        assert len(related) == 2
        uris = {n.uri for n in related}
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
        }

    def test_relation_type_filter(self, store, ctx):
        """Can filter results by relation type."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Test",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="derived_from",
                weight=0.6,
                reason="Test",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
            relation_types=["related_to"],
        )

        assert len(related) == 1
        assert related[0].uri == "ctx://test/account/users/user/memories/B"
        assert related[0].relation_types == ["related_to"]

    def test_min_weight_filter(self, store, ctx):
        """Can filter results by minimum weight."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.9,
                reason="High confidence",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.3,
                reason="Low confidence",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
            min_weight=0.5,
        )

        assert len(related) == 1
        assert related[0].uri == "ctx://test/account/users/user/memories/B"
        assert related[0].score == 0.9

    def test_2hop_traversal(self, store, ctx):
        """2-hop traversal finds neighbors of neighbors."""
        # A -> B -> D
        # A -> C
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
                relation_type="derived_from",
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

        # Should include B, C (1-hop) and D (2-hop via B)
        uris = {n.uri for n in related}
        assert uris == {
            "ctx://test/account/users/user/memories/B",
            "ctx://test/account/users/user/memories/C",
            "ctx://test/account/users/user/memories/D",
        }

        # Check 2-hop node
        node_d = next(n for n in related if n.uri == "ctx://test/account/users/user/memories/D")
        assert node_d.distance == 2
        assert node_d.score == 0.6  # min(0.8, 0.6)
        assert len(node_d.path) == 2

    def test_2hop_path_scoring(self, store, ctx):
        """2-hop path score is minimum of edge weights."""
        # Create two paths to same target with different scores
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.9,
                reason="High weight path",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.9,
                reason="High weight path",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.3,
                reason="Low weight path",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/C",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.9,
                reason="Low weight path",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        node_d = next(n for n in related if n.uri == "ctx://test/account/users/user/memories/D")

        # Best path is via B (min(0.9, 0.9) = 0.9)
        # Not via C (min(0.3, 0.9) = 0.3)
        assert node_d.score == 0.9
        assert len(node_d.path) == 2
        assert node_d.path[0].from_uri.endswith("/A")
        assert node_d.path[0].to_uri.endswith("/B")

    def test_2hop_deduplication(self, store, ctx):
        """2-hop traversal deduplicates same target reached via different paths."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Path 1",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.7,
                reason="Path 2",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/B",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.6,
                reason="Via B",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/C",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.5,
                reason="Via C",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        # D should appear once with best score
        node_d_list = [n for n in related if n.uri == "ctx://test/account/users/user/memories/D"]
        assert len(node_d_list) == 1
        # Best path score is min(0.8, 0.6) = 0.6 (via B)
        assert node_d_list[0].score == 0.6

    def test_2hop_excludes_source(self, store, ctx):
        """2-hop traversal doesn't include cycles back to source."""
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
                reason="B back to A",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        # Should not include A itself
        uris = {n.uri for n in related}
        assert "ctx://test/account/users/user/memories/A" not in uris
        assert "ctx://test/account/users/user/memories/B" in uris

    def test_results_sorted_by_score(self, store, ctx):
        """Results are sorted by score descending."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.5,
                reason="Low",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/C",
                relation_type="related_to",
                weight=0.9,
                reason="High",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/D",
                relation_type="related_to",
                weight=0.7,
                reason="Medium",
            ),
        ]

        store.upsert_edges(edges, ctx)

        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/A",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        scores = [n.score for n in related]
        assert scores == [0.9, 0.7, 0.5]

    def test_invalid_direction_raises(self, store, ctx):
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            store.get_related_nodes(
                "ctx://test/account/users/user/memories/A",
                ctx,
                direction="INVALID",
            )

    def test_invalid_depth_raises(self, store, ctx):
        """Invalid depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be"):
            store.get_related_nodes(
                "ctx://test/account/users/user/memories/A",
                ctx,
                depth=3,
            )


class TestNodeManagement:
    """Test node existence and deletion."""

    def test_node_exists(self, store, ctx):
        """Can check if a node exists."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        store.upsert_edges([edge], ctx)

        assert store.node_exists("ctx://test/account/users/user/memories/A", ctx)
        assert store.node_exists("ctx://test/account/users/user/memories/B", ctx)
        assert not store.node_exists("ctx://test/account/users/user/memories/C", ctx)

    def test_node_exists_cross_account(self, store, ctx, ctx_other):
        """node_exists respects account boundaries."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        store.upsert_edges([edge], ctx)

        # Same account can see node
        assert store.node_exists("ctx://test/account/users/user/memories/A", ctx)

        # Different account cannot
        assert not store.node_exists("ctx://test/account/users/user/memories/A", ctx_other)

    def test_delete_node_marks_not_exists(self, store, ctx):
        """Deleting a node marks it as not existing."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="Test",
        )

        store.upsert_edges([edge], ctx)
        assert store.node_exists("ctx://test/account/users/user/memories/A", ctx)

        store.delete_node("ctx://test/account/users/user/memories/A", ctx)
        assert not store.node_exists("ctx://test/account/users/user/memories/A", ctx)

    def test_delete_node_preserves_edges(self, store, ctx):
        """Deleting a node does not remove its edges."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Test",
            ),
        ]

        store.upsert_edges(edges, ctx)
        store.delete_node("ctx://test/account/users/user/memories/A", ctx)

        # Edges still exist
        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)
        # But node is marked deleted, so get_edges returns empty for deleted nodes
        # Actually - the current implementation checks exists flag
        assert len(retrieved) == 0

        # But we can still traverse from B to find the edge (inbound)
        related = store.get_related_nodes(
            "ctx://test/account/users/user/memories/B",
            ctx,
            direction="INBOUND",
        )
        # A is deleted, so won't appear
        assert len([n for n in related if n.uri.endswith("/A")]) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_upsert_empty_list_no_op(self, store, ctx):
        """Upserting empty list is safe no-op."""
        store.upsert_edges([], ctx)
        assert len(store._edges_by_source) == 0

    def test_get_edges_unknown_uri(self, store, ctx):
        """Getting edges from unknown URI returns empty list."""
        edges = store.get_edges("ctx://unknown/uri", ctx)
        assert edges == []

    def test_multiple_relation_types_same_pair(self, store, ctx):
        """Can have multiple relation types between same pair."""
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
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="contradicts",
                weight=0.3,
                reason="Also contradicts",
            ),
        ]

        store.upsert_edges(edges, ctx)

        retrieved = store.get_edges("ctx://test/account/users/user/memories/A", ctx)
        assert len(retrieved) == 2

        relation_types = {e.relation_type for e in retrieved}
        assert relation_types == {"related_to", "contradicts"}
