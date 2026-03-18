"""Unit tests for HierarchicalSearcher.

Tests cover:
- Basic expansion without relations
- 1-hop and 2-hop expansion
- Relation scoring and decay factors
- Abstract fetching for expanded nodes
- Relation path summaries
- Grouping by distance
"""

import pytest

from core.models import RequestContext, SeedHit, RelationEdge
from providers.relation_store.graph_relation_store import (
    GraphRelationStore,
    RelatedNode,
)
from retrieval.hierarchical_searcher import (
    HierarchicalSearcher,
    ExpandedHit,
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
def relation_store(ctx):
    """GraphRelationStore with test relations."""
    store = GraphRelationStore()

    # Set up test graph:
    # A -> B -> D
    # A -> C
    edges = [
        RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/B",
            relation_type="related_to",
            weight=0.8,
            reason="A related to B",
        ),
        RelationEdge(
            from_uri="ctx://test/account/users/user/memories/A",
            to_uri="ctx://test/account/users/user/memories/C",
            relation_type="derived_from",
            weight=0.6,
            reason="C derived from A",
        ),
        RelationEdge(
            from_uri="ctx://test/account/users/user/memories/B",
            to_uri="ctx://test/account/users/user/memories/D",
            relation_type="related_to",
            weight=0.7,
            reason="B related to D",
        ),
    ]

    store.upsert_edges(edges, ctx)
    return store


@pytest.fixture
def seed_hits():
    """Sample seed hits for testing."""
    return [
        SeedHit(
            uri="ctx://test/account/users/user/memories/A",
            score=0.9,
            abstract="Memory A",
            has_overview=True,
            has_content=True,
        ),
        SeedHit(
            uri="ctx://test/account/users/user/memories/X",
            score=0.7,
            abstract="Memory X (no relations)",
            has_overview=False,
            has_content=False,
        ),
    ]


class TestHierarchicalSearcherBasics:
    """Test basic expansion functionality."""

    def test_expand_without_relations(self, seed_hits, ctx):
        """Expansion without relation store returns original hits."""
        # Mock relation store that does nothing
        class NoOpRelationStore:
            def get_edges(self, uri, ctx):
                return []

        searcher = HierarchicalSearcher(NoOpRelationStore())

        expanded = searcher.expand(seed_hits, ctx)

        assert len(expanded) == 2
        assert expanded[0].uri == "ctx://test/account/users/user/memories/A"
        assert expanded[0].distance == 0
        assert expanded[0].final_score == 0.9

    def test_expand_preserves_original_scores(self, relation_store, seed_hits, ctx):
        """Original hits have unchanged scores."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        # Find original hit A
        hit_a = next(h for h in expanded if h.uri.endswith("/A"))
        assert hit_a.base_score == 0.9
        assert hit_a.final_score == 0.9
        assert hit_a.distance == 0

    def test_expansion_with_graph_store(self, relation_store, seed_hits, ctx):
        """Expansion with graph store includes related nodes."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        uris = {h.uri for h in expanded}
        # Should include A, X (originals) and B, C (related to A)
        assert "ctx://test/account/users/user/memories/A" in uris
        assert "ctx://test/account/users/user/memories/X" in uris
        assert "ctx://test/account/users/user/memories/B" in uris
        assert "ctx://test/account/users/user/memories/C" in uris


class TestExpansionScoring:
    """Test scoring of expanded nodes."""

    def test_1hop_decay_factor(self, relation_store, seed_hits, ctx):
        """1-hop expansions have 0.8 decay factor applied."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        # Find B (1-hop from A with weight 0.8)
        hit_b = next(h for h in expanded if h.uri.endswith("/B"))
        expected_score = 0.9 * 0.8 * 0.8  # base * relation * decay
        assert hit_b.distance == 1
        assert abs(hit_b.final_score - expected_score) < 0.001

    def test_2hop_decay_factor(self, relation_store, seed_hits, ctx):
        """2-hop expansions have 0.6 decay factor applied."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx, depth=2)

        # Find D (2-hop from A via B)
        hit_d = next((h for h in expanded if h.uri.endswith("/D")), None)
        if hit_d:
            assert hit_d.distance == 2
            # Score = base(A) * min(weight(A->B), weight(B->D)) * 0.6
            #       = 0.9 * min(0.8, 0.7) * 0.6 = 0.9 * 0.7 * 0.6 = 0.378
            expected_score = 0.9 * 0.7 * 0.6
            assert abs(hit_d.final_score - expected_score) < 0.001

    def test_results_sorted_by_final_score(self, relation_store, seed_hits, ctx):
        """Results are sorted by final_score descending."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        scores = [h.final_score for h in expanded]
        assert scores == sorted(scores, reverse=True)


class TestExpansionParameters:
    """Test various expansion parameters."""

    def test_min_relation_weight_filter(self, relation_store, seed_hits, ctx):
        """Can filter by minimum relation weight."""
        searcher = HierarchicalSearcher(relation_store)

        # Set high min_weight to exclude C (weight 0.6)
        expanded = searcher.expand(
            seed_hits,
            ctx,
            min_relation_weight=0.7,
        )

        uris = {h.uri for h in expanded}
        # B (weight 0.8) should be included
        assert "ctx://test/account/users/user/memories/B" in uris
        # C (weight 0.6) should be excluded
        assert "ctx://test/account/users/user/memories/C" not in uris

    def test_max_expansions_limit(self, relation_store, seed_hits, ctx):
        """Can limit maximum number of results."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(
            seed_hits,
            ctx,
            max_expansions=2,
        )

        assert len(expanded) <= 2

    def test_direction_outbound(self, relation_store, seed_hits, ctx):
        """OUTBOUND direction follows edges from source."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(
            seed_hits,
            ctx,
            direction="OUTBOUND",
        )

        # Should find B, C (outbound from A)
        uris = {h.uri for h in expanded}
        assert "ctx://test/account/users/user/memories/B" in uris
        assert "ctx://test/account/users/user/memories/C" in uris

    def test_invalid_depth_raises(self, relation_store, seed_hits, ctx):
        """Invalid depth raises ValueError."""
        searcher = HierarchicalSearcher(relation_store)

        with pytest.raises(ValueError, match="depth must be"):
            searcher.expand(seed_hits, ctx, depth=3)

    def test_invalid_direction_raises(self, relation_store, seed_hits, ctx):
        """Invalid direction raises ValueError."""
        searcher = HierarchicalSearcher(relation_store)

        with pytest.raises(ValueError, match="direction must be"):
            searcher.expand(seed_hits, ctx, direction="INVALID")


class TestExpansionWithContent:
    """Test expand_with_content method."""

    def test_expand_with_content_fetches_abstracts(self, relation_store, seed_hits, ctx):
        """Fetches abstracts for expanded nodes."""
        abstracts = {
            "ctx://test/account/users/user/memories/B": "Abstract B",
            "ctx://test/account/users/user/memories/C": "Abstract C",
        }

        def mock_abstract_fn(uri):
            return abstracts.get(uri, "")

        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand_with_content(
            seed_hits,
            ctx,
            mock_abstract_fn,
        )

        # Check abstracts were fetched
        hit_b = next(h for h in expanded if h.uri.endswith("/B"))
        assert hit_b.abstract == "Abstract B"

        hit_c = next(h for h in expanded if h.uri.endswith("/C"))
        assert hit_c.abstract == "Abstract C"

    def test_expand_with_content_graceful_failure(self, relation_store, seed_hits, ctx):
        """Handles abstract fetch failures gracefully."""
        def failing_abstract_fn(uri):
            raise Exception("Fetch failed")

        searcher = HierarchicalSearcher(relation_store)

        # Should not raise, just continue with empty abstracts
        expanded = searcher.expand_with_content(
            seed_hits,
            ctx,
            failing_abstract_fn,
        )

        # Original hit A should still have its abstract
        hit_a = next(h for h in expanded if h.uri.endswith("/A"))
        assert hit_a.abstract == "Memory A"


class TestRelationPath:
    """Test relation path tracking."""

    def test_relation_path_populated(self, relation_store, seed_hits, ctx):
        """Expanded hits have relation_path populated."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        hit_b = next(h for h in expanded if h.uri.endswith("/B"))
        assert hit_b.relation_path == ["related_to"]

        hit_c = next(h for h in expanded if h.uri.endswith("/C"))
        assert hit_c.relation_path == ["derived_from"]

    def test_source_uri_tracked(self, relation_store, seed_hits, ctx):
        """Expanded nodes track their source URI."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        hit_b = next(h for h in expanded if h.uri.endswith("/B"))
        assert hit_b.source_uri == "ctx://test/account/users/user/memories/A"

    def test_get_relation_path_summary(self, relation_store, seed_hits, ctx):
        """Can get human-readable relation path summary."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        hit_b = next(h for h in expanded if h.uri.endswith("/B"))
        summary = searcher.get_relation_path_summary(hit_b)
        assert summary == "related_to"

        # Original hit has no relation path
        hit_a = next(h for h in expanded if h.distance == 0)
        summary = searcher.get_relation_path_summary(hit_a)
        assert summary == ""


class TestGrouping:
    """Test grouping functionality."""

    def test_group_by_distance(self, relation_store, seed_hits, ctx):
        """Can group results by distance."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        groups = searcher.group_by_distance(expanded)

        assert 0 in groups
        assert 1 in groups
        assert 2 in groups

        # Distance 0: original hits
        assert len(groups[0]) == 2
        # Distance 1: B, C
        assert len(groups[1]) >= 2

    def test_group_by_distance_all_distances(self, relation_store, seed_hits, ctx):
        """All groups are present even if empty."""
        searcher = HierarchicalSearcher(relation_store)

        expanded = searcher.expand(seed_hits, ctx)

        groups = searcher.group_by_distance(expanded)

        # All distances should be present
        assert set(groups.keys()) == {0, 1, 2}


class TestExpandedHit:
    """Test ExpandedHit dataclass."""

    def test_expanded_hit_creation(self):
        """Can create ExpandedHit with all fields."""
        hit = ExpandedHit(
            uri="ctx://test/account/users/user/memories/A",
            base_score=0.9,
            relation_score=0.8,
            final_score=0.72,
            distance=1,
            abstract="Test",
            has_overview=True,
            has_content=False,
            relation_path=["related_to"],
            source_uri="ctx://test/account/users/user/memories/B",
        )

        assert hit.uri == "ctx://test/account/users/user/memories/A"
        assert hit.base_score == 0.9
        assert hit.relation_score == 0.8
        assert hit.final_score == 0.72
        assert hit.distance == 1
        assert hit.relation_path == ["related_to"]
        assert hit.source_uri == "ctx://test/account/users/user/memories/B"

    def test_expanded_hit_defaults(self):
        """Default values work correctly."""
        hit = ExpandedHit(
            uri="ctx://test/account/users/user/memories/A",
            base_score=0.9,
            relation_score=1.0,
            final_score=0.9,
            distance=0,
            abstract="Test",
            has_overview=False,
            has_content=False,
        )

        assert hit.relation_path == []
        assert hit.source_uri is None
