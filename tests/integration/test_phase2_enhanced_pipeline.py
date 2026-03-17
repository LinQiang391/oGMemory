"""Phase 2 End-to-End Integration Test.

Tests the complete enhanced pipeline:
    Write → Relation Retrieval → Hierarchical Search

This builds on Phase 1's baseline pipeline and adds Phase 2 capabilities:
    - Graph-based hierarchical retrieval (P2-R1 ✅ DONE)
    - Relations field in AssemblyService output (P2-R2 ✅ DONE)
    - GraphRelationStore with traversal (P2-G2/P2-G3/P2-G4 ✅ DONE)

Note: AI Functions (chunking, filtering) not yet implemented (P2-A* pending).
"""

import pytest
from typing import Optional

from core.models import (
    RequestContext,
    ContextNode,
    RelationEdge,
    TypedQuery,
    SeedHit,
)
from core.enums import ContextType
from core.interfaces import RelationStore
from providers.relation_store.graph_relation_store import GraphRelationStore
from retrieval.hierarchical_searcher import HierarchicalSearcher, ExpandedHit
from retrieval.assembly_service import AssemblyService, RetrievedBlock


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
def relation_store():
    """Fresh GraphRelationStore for each test."""
    return GraphRelationStore()


@pytest.fixture
def hierarchical_searcher(relation_store):
    """HierarchicalSearcher with relation store."""
    return HierarchicalSearcher(relation_store)


@pytest.fixture
def mock_fs():
    """Mock ContextFS for testing."""
    class MockFS:
        def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
            # Return mock node data
            return ContextNode(
                uri=uri,
                context_type=ContextType.MEMORY,
                category="profile",
                level=0,
                owner_space="user_space_user1",
                abstract=f"Abstract for {uri}",
                overview="## Overview\nMock overview content",
                content="Full content here",
            )

    return MockFS()


class TestGraphRelationStoreIntegration:
    """Test GraphRelationStore as foundation for Phase 2 features."""

    def test_relation_store_basic_operations(self, relation_store, ctx):
        """GraphRelationStore should support basic upsert and retrieval."""
        edge = RelationEdge(
            from_uri="ctx://test/account/users/user/memories/profile",
            to_uri="ctx://test/account/users/user/memories/entities/coffee",
            relation_type="related_to",
            weight=0.85,
            reason="User profile mentions coffee preference",
        )

        relation_store.upsert_edges([edge], ctx)

        retrieved = relation_store.get_edges(
            "ctx://test/account/users/user/memories/profile",
            ctx
        )

        assert len(retrieved) == 1
        assert retrieved[0].to_uri.endswith("/entities/coffee")
        assert retrieved[0].weight == 0.85

    def test_relation_store_1hop_traversal(self, relation_store, ctx):
        """GraphRelationStore should support 1-hop traversal."""
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/preferences/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="Profile mentions coffee",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/events/cafe_visit",
                relation_type="derived_from",
                weight=0.7,
                reason="Cafe visit informed profile",
            ),
        ]

        relation_store.upsert_edges(edges, ctx)

        related = relation_store.get_related_nodes(
            "ctx://test/account/users/user/memories/profile",
            ctx,
            direction="OUTBOUND",
            depth=1,
        )

        uris = {n.uri for n in related}
        assert len(uris) == 2
        assert any("coffee" in uri for uri in uris)
        assert any("cafe_visit" in uri for uri in uris)

    def test_relation_store_2hop_traversal(self, relation_store, ctx):
        """GraphRelationStore should support 2-hop traversal."""
        # profile -> coffee -> cafe_visit
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=0.8,
                reason="Likes coffee",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/entities/coffee",
                to_uri="ctx://test/account/users/user/memories/events/cafe_visit",
                relation_type="related_to",
                weight=0.7,
                reason="Visited cafe",
            ),
        ]

        relation_store.upsert_edges(edges, ctx)

        related = relation_store.get_related_nodes(
            "ctx://test/account/users/user/memories/profile",
            ctx,
            direction="OUTBOUND",
            depth=2,
        )

        uris = {n.uri for n in related}
        # Should include coffee (1-hop) and cafe_visit (2-hop)
        assert any("coffee" in uri for uri in uris)
        assert any("cafe_visit" in uri for uri in uris)

        # Check 2-hop node
        cafe = next((n for n in related if "cafe_visit" in n.uri), None)
        assert cafe is not None
        assert cafe.distance == 2
        assert cafe.score == 0.7  # min(0.8, 0.7)


class TestHierarchicalSearcherIntegration:
    """Test HierarchicalSearcher with graph expansion."""

    def test_expand_returns_original_hits(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher should include original seed hits."""
        relation_store.upsert_edges([], ctx)  # No edges

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        expanded = hierarchical_searcher.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=1,
        )

        assert len(expanded) == 1
        assert expanded[0].uri == seed_hits[0].uri
        assert expanded[0].distance == 0  # Original hit
        assert expanded[0].final_score == 0.92

    def test_expand_includes_related_nodes(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher should expand with related nodes."""
        # Setup graph: profile -> coffee
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=0.85,
                reason="Profile mentions coffee",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        expanded = hierarchical_searcher.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=1,
            min_relation_weight=0.3,
        )

        uris = {h.uri for h in expanded}
        assert len(uris) == 2
        assert any("profile" in uri for uri in uris)
        assert any("coffee" in uri for uri in uris)

        # Check expanded node
        coffee_hit = next((h for h in expanded if "coffee" in h.uri), None)
        assert coffee_hit is not None
        assert coffee_hit.distance == 1
        assert coffee_hit.source_uri.endswith("/profile")

    def test_expand_2hop_returns_indirect_relations(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher should support 2-hop expansion."""
        # Setup: profile -> coffee -> cafe_visit
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="Likes coffee",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/entities/coffee",
                to_uri="ctx://test/account/users/user/memories/events/cafe_visit",
                relation_type="derived_from",
                weight=0.8,
                reason="Cafe visit",
            ),
        ]
        relation_store.upsert_edges(edges, ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        expanded = hierarchical_searcher.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=2,
        )

        uris = {h.uri for h in expanded}
        assert any("profile" in uri for uri in uris)
        assert any("coffee" in uri for uri in uris)
        assert any("cafe_visit" in uri for uri in uris)

        # Check 2-hop node
        cafe_hit = next((h for h in expanded if "cafe_visit" in h.uri), None)
        assert cafe_hit is not None
        assert cafe_hit.distance == 2

    def test_expand_respects_account_isolation(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher must NOT cross account boundaries."""
        other_ctx = RequestContext(
            account_id="other_account",
            user_id="user2",
            agent_id="agent2",
            session_id="session2",
            trace_id="trace2",
        )

        # Create edge in test_account
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=0.85,
                reason="Test",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        # Same account can expand
        expanded = hierarchical_searcher.expand(seed_hits, ctx, depth=1)
        assert len(expanded) >= 1

        # Different account cannot expand (only original hit, no relations)
        # Since the node doesn't exist in other_account, get_related_nodes returns empty
        # But the original seed hit is still included
        expanded_other = hierarchical_searcher.expand(seed_hits, other_ctx, depth=1)
        # The node doesn't exist for other_account, so expansion returns nothing
        # Actually, let me check - the seed_hits would still be included
        # So we should get 1 result (the original) but no expanded nodes
        # Actually, looking at the implementation, if the node doesn't exist
        # in the relation_store for that account, get_related_nodes returns []
        # So we'd get just the original seed hit
        assert len(expanded_other) == 1  # Only original, no expansions

    def test_expand_applies_score_decay(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher should decay scores for expanded nodes."""
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=1.0,  # Max relation weight
                reason="Test",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        expanded = hierarchical_searcher.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=1,
        )

        # Original hit: score unchanged
        original = next(h for h in expanded if h.distance == 0)
        assert original.final_score == 0.92

        # Expanded hit: score decayed by 0.8 factor
        expanded_hit = next(h for h in expanded if h.distance == 1)
        # final_score = base_score * relation_weight * 0.8
        # final_score = 0.92 * 1.0 * 0.8 = 0.736
        assert expanded_hit.final_score == 0.92 * 1.0 * 0.8

    def test_expand_with_content_fetches_abstracts(self, hierarchical_searcher, relation_store, ctx):
        """expand_with_content should fetch abstracts for expanded nodes."""
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/entities/coffee",
                relation_type="related_to",
                weight=0.85,
                reason="Test",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        def mock_get_abstract(uri: str) -> str:
            return f"Abstract for {uri}"

        expanded = hierarchical_searcher.expand_with_content(
            seed_hits=seed_hits,
            ctx=ctx,
            get_abstract_fn=mock_get_abstract,
            depth=1,
        )

        # Check that abstracts are populated
        for hit in expanded:
            if hit.distance > 0:
                # Expanded nodes should have abstract from fetch function
                assert "Abstract for" in hit.abstract


class TestAssemblyServiceWithRelations:
    """Test AssemblyService Phase 2 enhancements."""

    def test_assembly_service_accepts_relation_store(self, mock_fs):
        """AssemblyService should accept optional RelationStore."""
        # Without relation store
        service = AssemblyService(mock_fs)
        assert service._relation_store is None

        # With relation store
        store = GraphRelationStore()
        service_with_store = AssemblyService(mock_fs, relation_store=store)
        assert service_with_store._relation_store is store

    def test_retrieved_block_has_relations_field(self):
        """RetrievedBlock should have relations field."""
        block = RetrievedBlock(
            uri="ctx://test/account/users/user/memories/profile",
            score=0.92,
            abstract="User profile",
            relations=[
                RelationEdge(
                    from_uri="ctx://test/account/users/user/memories/profile",
                    to_uri="ctx://test/account/users/user/memories/entities/coffee",
                    relation_type="related_to",
                    weight=0.85,
                    reason="Profile mentions coffee",
                ),
            ],
        )

        assert block.has_relations
        assert len(block.relations) == 1
        assert block.relations[0].relation_type == "related_to"

    def test_assemble_creates_retrieved_blocks(self, mock_fs):
        """AssemblyService.assemble should create RetrievedBlock objects."""
        service = AssemblyService(mock_fs)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        blocks = service.assemble(
            seed_hits=seed_hits,
            include_overview=False,
            include_content=False,
            ctx=None,  # No context needed for L0-only
        )

        assert len(blocks) == 1
        assert blocks[0].uri == seed_hits[0].uri
        assert blocks[0].score == seed_hits[0].score
        assert blocks[0].abstract == seed_hits[0].abstract


class TestPhase2FullPipeline:
    """End-to-end pipeline tests with Phase 2 features."""

    def test_write_to_relation_to_hierarchical_search(self, relation_store, hierarchical_searcher, ctx):
        """Full pipeline: Write relations → Hierarchical retrieval."""
        # 1. Create relation edges (as if from write operations)
        edges = [
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/preferences/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="Profile mentions coffee preference",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/events/cafe_visit",
                relation_type="derived_from",
                weight=0.7,
                reason="Cafe visit",
            ),
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/preferences/coffee",
                to_uri="ctx://test/account/users/user/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.85,
                reason="Likes coffee shops",
            ),
        ]

        relation_store.upsert_edges(edges, ctx)

        # 2. Simulate vector search returning profile
        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile and preferences",
                has_overview=True,
                has_content=True,
            ),
        ]

        # 3. Hierarchical expansion
        expanded = hierarchical_searcher.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=2,  # 2-hop to get coffee_shop
        )

        # 4. Verify we get all related nodes
        uris = {h.uri for h in expanded}
        assert any("profile" in uri for uri in uris)  # Original
        assert any("coffee" in uri and "preferences" in uri for uri in uris)  # 1-hop
        assert any("cafe_visit" in uri for uri in uris)  # 1-hop
        assert any("coffee_shop" in uri for uri in uris)  # 2-hop via coffee preference

    def test_cross_account_isolation_enforced(self, relation_store, hierarchical_searcher, ctx):
        """Graph traversal MUST NOT cross account boundaries (CRITICAL)."""
        other_ctx = RequestContext(
            account_id="other_account",
            user_id="user2",
            agent_id="agent2",
            session_id="session2",
            trace_id="trace2",
        )

        # Create edge in test_account
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/profile",
                to_uri="ctx://test/account/users/user/memories/preferences/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="Test",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        # Same account: can expand
        expanded = hierarchical_searcher.expand(seed_hits, ctx, depth=1)
        assert len(expanded) > 1  # Original + at least one expansion

        # Different account: only original hit, no expansions
        expanded_other = hierarchical_searcher.expand(seed_hits, other_ctx, depth=1)
        # Since the node doesn't exist in other_account, we only get the original seed hit
        # Actually, looking at the code again...
        # If the node doesn't exist in the relation_store, get_edges returns []
        # But the seed_hits are still processed, so we get at least the original
        # Let me verify: get_related_nodes checks node existence first
        # If node doesn't exist, it returns []
        # So expand would return just the original seed hit
        assert len(expanded_other) == 1
        assert expanded_other[0].distance == 0


class TestPhase2ContractCompliance:
    """Verify Phase 2 implementations satisfy contracts."""

    def test_graph_relation_store_satisfies_protocol(self, relation_store):
        """GraphRelationStore must satisfy RelationStore Protocol."""
        # Check all protocol methods exist
        assert hasattr(relation_store, 'get_edges')
        assert hasattr(relation_store, 'upsert_edges')
        # Check they're callable
        assert callable(relation_store.get_edges)
        assert callable(relation_store.upsert_edges)

    def test_hierarchical_searcher_expands_results(self, hierarchical_searcher, relation_store, ctx):
        """HierarchicalSearcher should expand results via relations."""
        relation_store.upsert_edges([
            RelationEdge(
                from_uri="ctx://test/account/users/user/memories/A",
                to_uri="ctx://test/account/users/user/memories/B",
                relation_type="related_to",
                weight=0.8,
                reason="Test",
            ),
        ], ctx)

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/A",
                score=0.9,
                abstract="Node A",
                has_overview=True,
                has_content=True,
            ),
        ]

        expanded = hierarchical_searcher.expand(seed_hits, ctx, depth=1)

        # Should have more results than original
        assert len(expanded) > len(seed_hits)
        # Should include original
        assert any(h.uri.endswith("/A") for h in expanded)
        # Should include expansion
        assert any(h.uri.endswith("/B") for h in expanded)


class TestPhase2BackwardCompatibility:
    """Verify Phase 2 maintains Phase 1 compatibility."""

    def test_assembly_service_without_relation_store(self, mock_fs):
        """AssemblyService should work without RelationStore (Phase 1 mode)."""
        service = AssemblyService(mock_fs)  # No relation_store

        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        blocks = service.assemble(
            seed_hits=seed_hits,
            include_overview=False,
            include_content=False,
            ctx=None,
        )

        # Should work fine without relations
        assert len(blocks) == 1
        assert blocks[0].relations == []  # Empty relations list

    def test_hierarchical_searcher_with_empty_graph(self, hierarchical_searcher, ctx):
        """HierarchicalSearcher should work with empty relation graph."""
        seed_hits = [
            SeedHit(
                uri="ctx://test/account/users/user/memories/profile",
                score=0.92,
                abstract="User profile",
                has_overview=True,
                has_content=True,
            ),
        ]

        # No relations in store, should still return original hits
        expanded = hierarchical_searcher.expand(seed_hits, ctx, depth=1)

        assert len(expanded) == 1
        assert expanded[0].uri == seed_hits[0].uri
        assert expanded[0].distance == 0


class TestSmokeTests:
    """Quick smoke tests to verify Phase 2 components are importable."""

    def test_import_graph_relation_store(self):
        """GraphRelationStore should be importable."""
        from providers.relation_store.graph_relation_store import GraphRelationStore
        assert GraphRelationStore is not None

    def test_import_hierarchical_searcher(self):
        """HierarchicalSearcher should be importable."""
        from retrieval.hierarchical_searcher import HierarchicalSearcher
        assert HierarchicalSearcher is not None

    def test_import_assembly_service(self):
        """AssemblyService should be importable."""
        from retrieval.assembly_service import AssemblyService
        assert AssemblyService is not None

    def test_graph_relation_store_has_get_related_nodes(self):
        """GraphRelationStore should have get_related_nodes method."""
        from providers.relation_store.graph_relation_store import GraphRelationStore
        store = GraphRelationStore()
        assert hasattr(store, 'get_related_nodes')
        assert callable(store.get_related_nodes)
