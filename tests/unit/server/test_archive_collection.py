"""Unit tests for archive collection and distance-based graduated compression.

Tests build_archive_refs() and _collect_archives() logic:
- Empty archive list → ([], [])
- Single archive → latest with overview, pre empty
- Multiple archives → latest gets overview, rest get abstract
- Budget truncation → oldest abstracts dropped first
- Sorting by created_at descending
"""

import pytest

from core.models import TokenBudget, ArchiveRef
from session.models import ArchiveEntry
from server.memory_service import build_archive_refs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_entry(archive_id: str, session_id: str = "s1",
               overview: str = "overview", abstract: str = "abstract",
               created_at: str = "2024-01-01T00:00:00") -> ArchiveEntry:
    """Helper to create ArchiveEntry instances for tests."""
    return ArchiveEntry(
        archive_id=archive_id,
        session_id=session_id,
        overview=overview,
        abstract=abstract,
        messages=[],
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# build_archive_refs tests
# ---------------------------------------------------------------------------

class TestBuildArchiveRefsEmpty:
    """Empty input returns empty lists."""

    def test_empty_entries(self):
        latest, pre = build_archive_refs([], TokenBudget())
        assert latest == []
        assert pre == []


class TestBuildArchiveRefsSingle:
    """Single archive becomes latest with overview."""

    def test_single_entry_has_overview(self):
        entry = make_entry("arc1", overview="Full overview of session")
        latest, pre = build_archive_refs([entry], TokenBudget())

        assert len(latest) == 1
        assert len(pre) == 0
        assert latest[0].archive_id == "arc1"
        assert latest[0].overview == "Full overview of session"
        assert latest[0].abstract == "abstract"

    def test_single_entry_tokens_estimated(self):
        overview = "A" * 40  # 40 chars → 10 tokens
        entry = make_entry("arc1", overview=overview)
        latest, pre = build_archive_refs([entry], TokenBudget())

        assert latest[0].tokens == 10

    def test_single_entry_with_empty_overview(self):
        entry = make_entry("arc1", overview="", abstract="some abstract")
        latest, pre = build_archive_refs([entry], TokenBudget())

        assert len(latest) == 1
        assert latest[0].overview == ""
        assert latest[0].tokens == 0


class TestBuildArchiveRefsMultiple:
    """Multiple archives: latest gets overview, rest get abstract only."""

    def test_three_entries_tiered(self):
        entries = [
            make_entry("arc1", overview="First overview", abstract="First abstract",
                       created_at="2024-01-01T00:00:00"),
            make_entry("arc2", overview="Second overview", abstract="Second abstract",
                       created_at="2024-01-02T00:00:00"),
            make_entry("arc3", overview="Third overview", abstract="Third abstract",
                       created_at="2024-01-03T00:00:00"),
        ]
        latest, pre = build_archive_refs(entries, TokenBudget())

        # Newest (arc3) should be latest with full overview
        assert len(latest) == 1
        assert latest[0].archive_id == "arc3"
        assert latest[0].overview == "Third overview"

        # arc2 and arc1 should be pre with abstract only
        assert len(pre) == 2
        assert pre[0].archive_id == "arc2"
        assert pre[0].overview is None
        assert pre[0].abstract == "Second abstract"
        assert pre[1].archive_id == "arc1"
        assert pre[1].overview is None
        assert pre[1].abstract == "First abstract"

    def test_pre_archives_sorted_newest_first(self):
        entries = [
            make_entry("old", abstract="Old abstract", created_at="2024-01-01T00:00:00"),
            make_entry("mid", abstract="Mid abstract", created_at="2024-01-02T00:00:00"),
            make_entry("new", abstract="New abstract", created_at="2024-01-03T00:00:00"),
        ]
        latest, pre = build_archive_refs(entries, TokenBudget())

        assert latest[0].archive_id == "new"
        assert pre[0].archive_id == "mid"
        assert pre[1].archive_id == "old"

    def test_entries_with_same_timestamp(self):
        entries = [
            make_entry("arc_a", created_at="2024-01-01T00:00:00"),
            make_entry("arc_b", created_at="2024-01-01T00:00:00"),
        ]
        latest, pre = build_archive_refs(entries, TokenBudget())

        assert len(latest) == 1
        assert len(pre) == 1

    def test_entries_with_empty_created_at(self):
        entries = [
            make_entry("arc1", created_at=""),
            make_entry("arc2", created_at="2024-01-02T00:00:00"),
        ]
        latest, pre = build_archive_refs(entries, TokenBudget())

        # arc2 has a date, arc1 has empty → arc2 is newest
        assert latest[0].archive_id == "arc2"


class TestBuildArchiveRefsBudgetTruncation:
    """Budget truncation drops oldest archives first."""

    def test_tight_budget_drops_all_pre(self):
        """When budget is too small, no pre-archives are included."""
        entries = [
            make_entry("arc1", abstract="A" * 10000, created_at="2024-01-01T00:00:00"),
            make_entry("arc2", abstract="B" * 10000, created_at="2024-01-02T00:00:00"),
            make_entry("arc3", abstract="C" * 100, created_at="2024-01-03T00:00:00"),
        ]
        budget = TokenBudget(total=1000, archive_ratio=0.5)
        latest, pre = build_archive_refs(entries, budget)

        assert len(latest) == 1
        assert latest[0].archive_id == "arc3"
        # Both pre-archives have 2500 tokens each, way over remaining budget
        assert len(pre) == 0

    def test_budget_keeps_newest_pre_drops_oldest(self):
        """Budget fits newest pre-archive but drops oldest."""
        entries = [
            make_entry("arc1", abstract="A" * 2000, created_at="2024-01-01T00:00:00"),
            make_entry("arc2", abstract="B" * 100, created_at="2024-01-02T00:00:00"),
            make_entry("arc3", abstract="C" * 100, created_at="2024-01-03T00:00:00"),
        ]
        budget = TokenBudget(total=1000, archive_ratio=0.5)
        # archive_limit = 500
        # latest "arc3" overview="overview" = 8 chars → 2 tokens, used=2
        # pre "arc2" abstract "B"*100 = 25 tokens → 2+25=27 ≤ 500 → included
        # pre "arc1" abstract "A"*2000 = 500 tokens → 27+500=527 > 500 → DROPPED
        latest, pre = build_archive_refs(entries, budget)

        assert len(latest) == 1
        assert latest[0].archive_id == "arc3"
        assert len(pre) == 1
        assert pre[0].archive_id == "arc2"  # Newest pre-archive kept

    def test_generous_budget_includes_all(self):
        """Large budget includes all archives."""
        entries = [
            make_entry("arc1", abstract="Short", created_at="2024-01-01T00:00:00"),
            make_entry("arc2", abstract="Short", created_at="2024-01-02T00:00:00"),
            make_entry("arc3", abstract="Short", created_at="2024-01-03T00:00:00"),
        ]
        budget = TokenBudget(total=1_000_000, archive_ratio=0.7)
        latest, pre = build_archive_refs(entries, budget)

        assert len(latest) == 1
        assert len(pre) == 2

    def test_default_budget_includes_reasonable_content(self):
        """Default 128k budget with 70% archive ratio works for typical archives."""
        entries = [
            make_entry(f"arc{i}", overview=f"Overview {i}" * 20,
                       abstract=f"Abstract {i}" * 10,
                       created_at=f"2024-01-{i+1:02d}T00:00:00")
            for i in range(5)
        ]
        budget = TokenBudget()
        latest, pre = build_archive_refs(entries, budget)

        assert len(latest) == 1
        assert len(pre) == 4  # All fit in default budget


class TestBuildArchiveRefsTokenEstimation:
    """Token estimation accuracy."""

    def test_overview_token_count(self):
        entry = make_entry("arc1", overview="A" * 100)
        latest, _ = build_archive_refs([entry], TokenBudget(total=10000))
        assert latest[0].tokens == 25  # 100 / 4

    def test_abstract_token_count_in_pre(self):
        entries = [
            make_entry("arc1", abstract="B" * 80, created_at="2024-01-01T00:00:00"),
            make_entry("arc2", abstract="C" * 80, created_at="2024-01-02T00:00:00"),
        ]
        _, pre = build_archive_refs(entries, TokenBudget(total=10000))
        assert pre[0].tokens == 20  # 80 / 4

    def test_empty_text_zero_tokens(self):
        entry = make_entry("arc1", overview="", abstract="")
        latest, _ = build_archive_refs([entry], TokenBudget())
        assert latest[0].tokens == 0

    def test_archive_uri_format(self):
        entry = make_entry("arc42", session_id="sess7")
        latest, _ = build_archive_refs([entry], TokenBudget())
        assert latest[0].archive_uri == "archive://sess7/arc42"
