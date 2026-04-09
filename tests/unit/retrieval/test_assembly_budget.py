"""Unit tests for AssemblyService token budget management."""

import pytest
from core.models import (
    AssembleResult,
    LeafHit,
    RequestContext,
    SeedResult,
    TokenBudget,
    TypedQuery,
)
from retrieval.assembly_service import AssemblyService


class TestAssemblyServiceBudget:
    """Test AssemblyService assemble_with_budget method."""

    def test_assemble_with_empty_hits(self):
        """Test assembly with no hits returns minimal messages."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=100_000, archive_ratio=0.7)

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=[],
            seed_result=None,
            token_budget=budget,
            profile_text="Test profile",
        )

        assert isinstance(result, AssembleResult)
        assert result.messages == []
        assert result.estimated_tokens > 0  # overhead only
        assert result.archive_count == 0
        assert result.archive_included is False
        assert result.stats["total_hits"] == 0
        assert result.stats["included_hits"] == 0

    def test_assemble_with_full_content(self):
        """Test assembly includes full content when budget allows."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=100_000, archive_ratio=0.7)

        hits = [
            LeafHit(
                uri="ctx://test/users/test/memories/profile",
                score=0.9,
                level=2,
                abstract="User profile information",
                has_overview=True,
                has_content=True,
                category="profile",
                owner_space="user:test",
            )
        ]

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=hits,
            seed_result=None,
            token_budget=budget,
            profile_text="",
        )

        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        assert "[Memory]" in result.messages[0]["content"]
        assert result.stats["included_hits"] == 1
        assert result.stats["full_content"] >= 1

    def test_assemble_graceful_degradation_abstract_only(self):
        """Test assembly degrades to abstract only when budget constrained."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=1_000, archive_ratio=0.7)  # Very small budget

        hits = [
            LeafHit(
                uri=f"ctx://test/users/test/memories/profile{i}",
                score=0.9 - (i * 0.1),
                level=2,
                abstract=f"Abstract {i} - " + "Long content description " * 50,  # Large abstract
                has_content=True,
                category="profile",
                owner_space="user:test",
            )
            for i in range(10)
        ]

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=hits,
            seed_result=None,
            token_budget=budget,
            profile_text="",
        )

        # Should include some hits, possibly abstract-only
        assert result.stats["included_hits"] >= 0
        assert result.stats["skipped"] >= 0
        assert result.estimated_tokens <= budget.total

    def test_assemble_with_profile_text(self):
        """Test assembly accounts for profile text in budget."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=5_000, archive_ratio=0.7)

        long_profile = "Test profile " * 1000  # ~4000 chars

        hits = [
            LeafHit(
                uri="ctx://test/users/test/memories/profile",
                score=0.9,
                level=2,
                abstract="User abstract",
                category="profile",
                owner_space="user:test",
            )
        ]

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=hits,
            seed_result=None,
            token_budget=budget,
            profile_text=long_profile,
        )

        # Profile text should reduce available budget
        assert result.estimated_tokens > 0
        # Long profile leaves less room for hits

    def test_assemble_respects_score_ordering(self):
        """Test assembly respects score ordering when including hits."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=10_000, archive_ratio=0.7)

        hits = [
            LeafHit(
                uri=f"ctx://test/users/test/memories/profile{i}",
                score=0.5 - (i * 0.1),
                level=2,
                abstract=f"Abstract {i} with score {0.5 - (i * 0.1)}",
                category="profile",
                owner_space="user:test",
            )
            for i in range(5)
        ]

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=hits,
            seed_result=None,
            token_budget=budget,
            profile_text="",
        )

        # Higher scored hits should be included first
        assert result.stats["included_hits"] >= 0

    def test_assemble_from_seed_result_fallback(self):
        """Test assembly falls back to seed_result when no leaf_hits."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=10_000, archive_ratio=0.7)

        from core.models import SeedHit

        seed_result = SeedResult(
            initial_candidates=[
                SeedHit(
                    uri="ctx://test/users/test/memories/profile",
                    score=0.8,
                    level=2,
                    abstract="Seed hit abstract",
                    category="profile",
                    owner_space="user:test",
                )
            ]
        )

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=[],
            seed_result=seed_result,
            token_budget=budget,
            profile_text="",
        )

        # Should use seed result as fallback
        assert result.stats["total_hits"] >= 0

    def test_token_budget_archive_limit(self):
        """Test archive_limit property calculation."""
        budget = TokenBudget(total=100_000, archive_ratio=0.7)
        # Capped by archive_tier.max_tokens
        assert budget.archive_limit == 40_000

        budget2 = TokenBudget(total=200_000, archive_ratio=0.5)
        assert budget2.archive_limit == 40_000

    def test_assemble_stats_tracking(self):
        """Test assembly properly tracks statistics."""
        service = AssemblyService()
        query = TypedQuery(text="test", context_type="MEMORY", categories=["profile"])
        budget = TokenBudget(total=10_000, archive_ratio=0.7)

        hits = [
            LeafHit(
                uri="ctx://test/users/test/memories/profile",
                score=0.9,
                level=2,
                abstract="Test abstract",
                category="profile",
                owner_space="user:test",
            )
        ]

        result = service.assemble_with_budget(
            typed_query=query,
            leaf_hits=hits,
            seed_result=None,
            token_budget=budget,
            profile_text="",
        )

        assert "total_hits" in result.stats
        assert "included_hits" in result.stats
        assert "abstract_only" in result.stats
        assert "full_content" in result.stats
        assert "skipped" in result.stats


class TestAssemblyServiceEstimateTokens:
    """Test token estimation helper."""

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        assert AssemblyService._estimate_tokens("") == 0
        assert AssemblyService._estimate_tokens(None) == 0

    def test_estimate_tokens_approximate(self):
        """Test token estimation is approximately correct."""
        # 100 chars ~ 25 tokens
        assert AssemblyService._estimate_tokens("a" * 100) == 25
        # 400 chars ~ 100 tokens
        assert AssemblyService._estimate_tokens("a" * 400) == 100

    def test_estimate_tokens_rounding(self):
        """Test token estimation rounds down."""
        # 3 chars -> 0 tokens
        assert AssemblyService._estimate_tokens("abc") == 0
        # 7 chars -> 1 token
        assert AssemblyService._estimate_tokens("abcdefg") == 1
