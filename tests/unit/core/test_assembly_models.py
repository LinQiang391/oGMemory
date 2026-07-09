"""Unit tests for unified assembly models."""

import pytest
from core.models import TokenBudget, ArchiveRef, SessionArchiveMeta, ComposedContext


class TestTokenBudget:
    """Test TokenBudget model."""

    def test_default_budget(self):
        """Test default token budget values."""
        budget = TokenBudget()
        assert budget.total == 128_000
        assert budget.archive_ratio == 0.7
        # Priority-based allocation caps archive at archive_tier.max_tokens
        assert budget.archive_limit == 40_000

    def test_custom_budget(self):
        """Test custom token budget values."""
        budget = TokenBudget(total=100_000, archive_ratio=0.5)
        assert budget.total == 100_000
        assert budget.archive_ratio == 0.5
        # Capped by archive_tier.max_tokens, not total * archive_ratio
        assert budget.archive_limit == 40_000

    def test_archive_limit_calculation(self):
        """Test archive_limit property calculation."""
        budget = TokenBudget(total=200_000, archive_ratio=0.8)
        # Capped by archive_tier.max_tokens regardless of total/ratio
        assert budget.archive_limit == 40_000


class TestArchiveRef:
    """Test ArchiveRef model."""

    def test_minimal_archive_ref(self):
        """Test ArchiveRef with minimal required fields."""
        ref = ArchiveRef(
            archive_id="archive-123",
            archive_uri="ctx://acme/users/alice/archives/archive-123",
            abstract="Session about Python debugging"
        )
        assert ref.archive_id == "archive-123"
        assert ref.archive_uri == "ctx://acme/users/alice/archives/archive-123"
        assert ref.abstract == "Session about Python debugging"
        assert ref.overview is None
        assert ref.tokens == 0

    def test_full_archive_ref(self):
        """Test ArchiveRef with all fields."""
        ref = ArchiveRef(
            archive_id="archive-456",
            archive_uri="ctx://acme/users/alice/archives/archive-456",
            abstract="Discussion about Rust memory safety",
            overview="Detailed conversation about ownership, borrowing, and lifetimes",
            tokens=5000
        )
        assert ref.archive_id == "archive-456"
        assert ref.overview == "Detailed conversation about ownership, borrowing, and lifetimes"
        assert ref.tokens == 5000


class TestSessionArchiveMeta:
    """Test SessionArchiveMeta model."""

    def test_session_archive_meta(self):
        """Test SessionArchiveMeta fields."""
        meta = SessionArchiveMeta(
            session_id="session-789",
            archive_id="archive-789",
            created_at="2025-04-01T12:00:00Z",
            message_count=15,
            overview="User asked about AGFS integration",
            abstract="AGFS integration planning session"
        )
        assert meta.session_id == "session-789"
        assert meta.archive_id == "archive-789"
        assert meta.created_at == "2025-04-01T12:00:00Z"
        assert meta.message_count == 15
        assert meta.overview == "User asked about AGFS integration"
        assert meta.abstract == "AGFS integration planning session"


class TestComposedContext:
    """Test ComposedContext model."""

    def test_minimal_result(self):
        """Test ComposedContext with minimal fields."""
        result = ComposedContext(
            messages=[{"role": "user", "content": "test"}],
            estimated_tokens=100
        )
        assert result.messages == [{"role": "user", "content": "test"}]
        assert result.estimated_tokens == 100
        assert result.archive_count == 0
        assert result.archive_included is False
        assert result.stats == {}

    def test_result_with_archives(self):
        """Test ComposedContext with archive information."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "system", "content": "[Archive Index] 2 archives"}
        ]
        result = ComposedContext(
            messages=messages,
            estimated_tokens=1500,
            archive_count=2,
            archive_included=True,
            stats={"archives_loaded": 2, "tokens_from_archives": 800}
        )
        assert len(result.messages) == 2
        assert result.estimated_tokens == 1500
        assert result.archive_count == 2
        assert result.archive_included is True
        assert result.stats["archives_loaded"] == 2

    def test_result_with_stats(self):
        """Test ComposedContext with detailed statistics."""
        result = ComposedContext(
            messages=[{"role": "user", "content": "test"}],
            estimated_tokens=500,
            stats={
                "memory_hits": 5,
                "archives_scanned": 3,
                "archives_included": 1,
                "truncated": False
            }
        )
        assert result.stats["memory_hits"] == 5
        assert result.stats["archives_scanned"] == 3
        assert result.stats["truncated"] is False
