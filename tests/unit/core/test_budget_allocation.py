"""Unit tests for priority-tiered budget allocation."""

import pytest
from core.models import BudgetTier, TokenBudget


class TestBudgetTier:
    """Test BudgetTier dataclass."""

    def test_budgettier_creation(self):
        """Test creating a BudgetTier."""
        tier = BudgetTier(
            name="test",
            priority=5,
            min_tokens=100,
            max_tokens=1000,
            degradable=True,
            expandable=False,
        )
        assert tier.name == "test"
        assert tier.priority == 5
        assert tier.min_tokens == 100
        assert tier.max_tokens == 1000
        assert tier.degradable is True
        assert tier.expandable is False

    def test_budgettier_defaults(self):
        """Test BudgetTier default values."""
        tier = BudgetTier(
            name="test",
            priority=5,
            min_tokens=100,
            max_tokens=1000,
        )
        assert tier.degradable is True  # default
        assert tier.expandable is False  # default


class TestTokenBudgetAllocation:
    """Test priority-tiered budget allocation."""

    def test_allocate_all_tiers_with_sufficient_budget(self):
        """Test allocation with sufficient budget for all tiers."""
        budget = TokenBudget(total=128_000)
        allocation = budget.allocate()

        # All tiers should get their max or close to it
        assert allocation["identity"] >= 500
        assert allocation["session_state"] >= 1000
        assert allocation["archive"] >= 500
        assert allocation["working_set"] >= 0

        # Total should not exceed budget
        assert sum(allocation.values()) <= 128_000

    def test_non_degradable_tiers_always_get_minimum(self):
        """Test that identity and session_state always get their minimum."""
        # Very small budget
        budget = TokenBudget(total=2000)
        allocation = budget.allocate()

        # Identity (priority 0, non-degradable) should get its minimum
        assert allocation["identity"] >= 500

        # Session state (priority 1, non-degradable) should get its minimum
        assert allocation["session_state"] >= 1000

    def test_small_budget_only_high_priority_survive(self):
        """Test with very small budget (1000 tokens) — only identity + session_state survive."""
        budget = TokenBudget(total=1000)
        allocation = budget.allocate()

        # Identity gets its min first
        assert allocation["identity"] == 500

        # Session state gets remaining
        assert allocation["session_state"] == 500

        # Archive and working set get nothing
        assert allocation.get("archive", 0) == 0
        assert allocation.get("working_set", 0) == 0

    def test_large_budget_working_set_expands(self):
        """Test with large budget (200K tokens) — working_set expands."""
        budget = TokenBudget(total=200_000)
        allocation = budget.allocate()

        # All tiers should get at least their max
        assert allocation["identity"] >= 5000
        assert allocation["session_state"] >= 10000
        assert allocation["archive"] >= 40000
        assert allocation["working_set"] >= 20000

        # Working set should expand beyond its max
        assert allocation["working_set"] > 20000

    def test_archive_degrades_when_budget_tight(self):
        """Test that archive degrades when budget is tight."""
        # Budget that covers identity + session_state but not archive max
        budget = TokenBudget(total=20_000)
        allocation = budget.allocate()

        # Identity and session_state get their minimums
        assert allocation["identity"] >= 500
        assert allocation["session_state"] >= 1000

        # Archive gets less than its max (degradable)
        assert allocation["archive"] < 40000

    def test_legacy_compat_archive_limit(self):
        """Test legacy archive_limit property."""
        budget = TokenBudget(total=128_000)
        # Should use allocation result
        limit = budget.archive_limit
        assert limit > 0
        assert limit <= 128_000

    def test_legacy_compat_session_state_limit(self):
        """Test legacy session_state_limit property."""
        budget = TokenBudget(total=128_000)
        # Should use allocation result
        limit = budget.session_state_limit
        assert limit > 0
        assert limit <= 128_000

    def test_allocation_priority_order(self):
        """Test that allocation respects priority order."""
        budget = TokenBudget(total=5000)
        allocation = budget.allocate()

        # Identity (priority 0) gets first claim
        assert allocation["identity"] >= 500

        # Session state (priority 1) gets second claim
        assert allocation["session_state"] >= 1000

        # Remaining goes to archive (priority 5) then working_set (priority 8)
        # Archive may get 0 if budget is too tight after satisfying minimums
        remaining_after_mins = 5000 - 500 - 1000
        archive_alloc = allocation.get("archive", 0)
        working_set_alloc = allocation.get("working_set", 0)
        assert archive_alloc + working_set_alloc <= remaining_after_mins

    def test_expandable_tier_gets_leftover(self):
        """Test that expandable tier gets all leftover budget."""
        budget = TokenBudget(total=100_000)
        allocation = budget.allocate()

        # Calculate total of non-expandable tiers
        non_expandable_total = (
            allocation["identity"] +
            allocation["session_state"] +
            allocation["archive"]
        )

        # Working set should get the rest
        expected_working_set = 100_000 - non_expandable_total
        assert allocation["working_set"] == expected_working_set

    def test_custom_tiers(self):
        """Test with custom tier configurations."""
        custom_budget = TokenBudget(
            total=50_000,
            identity_tier=BudgetTier(
                name="identity", priority=0, min_tokens=1000, max_tokens=10000,
                degradable=False, expandable=False,
            ),
            session_state_tier=BudgetTier(
                name="session_state", priority=1, min_tokens=2000, max_tokens=15000,
                degradable=False, expandable=False,
            ),
            archive_tier=BudgetTier(
                name="archive", priority=5, min_tokens=500, max_tokens=20000,
                degradable=True, expandable=False,
            ),
            working_set_tier=BudgetTier(
                name="working_set", priority=8, min_tokens=0, max_tokens=15000,
                degradable=True, expandable=True,
            ),
        )
        allocation = custom_budget.allocate()

        # Should respect custom minimums
        assert allocation["identity"] >= 1000
        assert allocation["session_state"] >= 2000
