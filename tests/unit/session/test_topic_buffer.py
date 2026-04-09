"""Tests for session.topic_buffer module."""

import pytest
from datetime import datetime, timezone
from session.topic_buffer import SessionTopicBuffer, SlotContent
from core.models import SeedHit


class TestSlotContent:
    """Tests for SlotContent dataclass."""

    def test_slot_content_creation(self):
        """Test creating SlotContent with minimal fields."""
        content = SlotContent(content="Test content")
        assert content.content == "Test content"
        assert content.uris == []
        assert content.tokens == 0
        assert content.cached_at != ""

    def test_slot_content_with_all_fields(self):
        """Test creating SlotContent with all fields."""
        now = datetime.now(timezone.utc).isoformat()
        content = SlotContent(
            content="Full content",
            uris=["ctx://test/uri1", "ctx://test/uri2"],
            tokens=100,
            cached_at=now,
        )
        assert content.content == "Full content"
        assert len(content.uris) == 2
        assert content.tokens == 100
        assert content.cached_at == now

    def test_slot_content_auto_timestamp(self):
        """Test that cached_at is auto-generated if not provided."""
        content1 = SlotContent(content="Test")
        assert content1.cached_at != ""

        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(content1.cached_at)


class TestSessionTopicBuffer:
    """Tests for SessionTopicBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initialization with session_id."""
        buffer = SessionTopicBuffer("test-session-123")
        assert buffer.session_id == "test-session-123"
        assert buffer.get_slot_names() == []

    def test_stable_slot_caching(self):
        """Test caching and retrieving stable slots."""
        buffer = SessionTopicBuffer("test-session")

        # Initially no slots
        assert buffer.get_cached_slot("identity") is None

        # Add a slot
        content = SlotContent(
            content="User is a software engineer",
            uris=["ctx://acme/users/alice/memories/profile"],
            tokens=50,
        )
        buffer.set_cached_slot("identity", content)

        # Retrieve it
        retrieved = buffer.get_cached_slot("identity")
        assert retrieved is not None
        assert retrieved.content == "User is a software engineer"
        assert len(retrieved.uris) == 1
        assert retrieved.tokens == 50

    def test_multiple_slots(self):
        """Test caching multiple different slots."""
        buffer = SessionTopicBuffer("test-session")

        # Add multiple slots
        identity = SlotContent(content="Profile data")
        skills = SlotContent(content="Skills data")
        archives = SlotContent(content="Archive data")

        buffer.set_cached_slot("identity", identity)
        buffer.set_cached_slot("skills_summary", skills)
        buffer.set_cached_slot("archive_history", archives)

        # Verify all are cached
        assert len(buffer.get_slot_names()) == 3
        assert buffer.get_cached_slot("identity").content == "Profile data"
        assert buffer.get_cached_slot("skills_summary").content == "Skills data"
        assert buffer.get_cached_slot("archive_history").content == "Archive data"

    def test_slot_invalidation(self):
        """Test invalidating a cached slot."""
        buffer = SessionTopicBuffer("test-session")

        # Add and verify slot
        content = SlotContent(content="Test content")
        buffer.set_cached_slot("identity", content)
        assert buffer.get_cached_slot("identity") is not None

        # Invalidate and verify
        buffer.invalidate("identity")
        assert buffer.get_cached_slot("identity") is None

        # Invalidating non-existent slot should not raise
        buffer.invalidate("nonexistent")  # Should not raise

    def test_clear_all_slots(self):
        """Test clearing all cached slots."""
        buffer = SessionTopicBuffer("test-session")

        # Add multiple slots
        buffer.set_cached_slot("identity", SlotContent(content="Profile"))
        buffer.set_cached_slot("skills", SlotContent(content="Skills"))
        assert len(buffer.get_slot_names()) == 2

        # Clear all
        buffer.clear_all_slots()
        assert len(buffer.get_slot_names()) == 0
        assert buffer.get_cached_slot("identity") is None
        assert buffer.get_cached_slot("skills") is None

    def test_prefetch_results(self):
        """Test storing and retrieving prefetch results."""
        buffer = SessionTopicBuffer("test-session")

        # Initially no prefetch
        assert buffer.get_pending_injection() is None

        # Set prefetch results
        hits = [
            SeedHit(
                uri="ctx://acme/users/alice/memories/preferences/coding_style",
                score=0.85,
                level=0,
                category="preference",
            ),
            SeedHit(
                uri="ctx://acme/users/alice/memories/entities/python",
                score=0.78,
                level=0,
                category="entity",
            ),
        ]
        buffer.set_pending_injection(hits)

        # Retrieve prefetch
        retrieved = buffer.get_pending_injection()
        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0].uri == "ctx://acme/users/alice/memories/preferences/coding_style"
        assert retrieved[1].score == 0.78

    def test_clear_prefetch_results(self):
        """Test clearing prefetch results."""
        buffer = SessionTopicBuffer("test-session")

        # Set and verify prefetch
        hits = [SeedHit(uri="ctx://test/uri", score=0.9, level=0)]
        buffer.set_pending_injection(hits)
        assert buffer.get_pending_injection() is not None

        # Clear prefetch
        buffer.clear_pending_injection()
        assert buffer.get_pending_injection() is None

    def test_injection_tracking(self):
        """Test tracking which URIs have been injected."""
        buffer = SessionTopicBuffer("test-session")

        # Initially nothing injected
        assert not buffer.was_injected("ctx://test/uri1")
        assert not buffer.was_injected("ctx://test/uri2")

        # Mark some URIs as injected
        buffer.mark_injected(["ctx://test/uri1", "ctx://test/uri2"])

        # Verify tracking
        assert buffer.was_injected("ctx://test/uri1")
        assert buffer.was_injected("ctx://test/uri2")
        assert not buffer.was_injected("ctx://test/uri3")

    def test_injection_tracking_accumulation(self):
        """Test that injection tracking accumulates across calls."""
        buffer = SessionTopicBuffer("test-session")

        # First batch
        buffer.mark_injected(["ctx://test/uri1", "ctx://test/uri2"])
        assert buffer.was_injected("ctx://test/uri1")

        # Second batch
        buffer.mark_injected(["ctx://test/uri3", "ctx://test/uri4"])
        assert buffer.was_injected("ctx://test/uri1")  # Still tracked
        assert buffer.was_injected("ctx://test/uri3")  # New

    def test_clear_injection_tracking(self):
        """Test clearing injected URI tracking."""
        buffer = SessionTopicBuffer("test-session")

        # Mark some as injected
        buffer.mark_injected(["ctx://test/uri1", "ctx://test/uri2"])
        assert buffer.was_injected("ctx://test/uri1")

        # Clear tracking
        buffer.clear_injected_tracking()
        assert not buffer.was_injected("ctx://test/uri1")
        assert not buffer.was_injected("ctx://test/uri2")

    def test_buffer_repr(self):
        """Test buffer string representation."""
        buffer = SessionTopicBuffer("test-session")
        repr_str = repr(buffer)

        assert "test-session" in repr_str
        assert "SessionTopicBuffer" in repr_str

        # Add slots and prefetch
        buffer.set_cached_slot("identity", SlotContent(content="Test"))
        buffer.set_pending_injection([SeedHit(uri="ctx://test", score=0.9, level=0)])

        repr_str = repr(buffer)
        assert "slots=1" in repr_str
        assert "pending=1" in repr_str

    def test_slot_update_replaces_content(self):
        """Test that updating a slot replaces the previous content."""
        buffer = SessionTopicBuffer("test-session")

        # Set initial content
        content1 = SlotContent(content="Original content")
        buffer.set_cached_slot("identity", content1)
        assert buffer.get_cached_slot("identity").content == "Original content"

        # Update with new content
        content2 = SlotContent(content="Updated content", tokens=100)
        buffer.set_cached_slot("identity", content2)
        assert buffer.get_cached_slot("identity").content == "Updated content"
        assert buffer.get_cached_slot("identity").tokens == 100

    def test_empty_pending_injection(self):
        """Test setting empty list as pending injection."""
        buffer = SessionTopicBuffer("test-session")

        # Set empty list
        buffer.set_pending_injection([])
        retrieved = buffer.get_pending_injection()
        assert retrieved is not None
        assert len(retrieved) == 0

    def test_complex_workflow(self):
        """Test a realistic workflow combining all features."""
        buffer = SessionTopicBuffer("session-123")

        # S1: Cache stable slots
        identity = SlotContent(
            content="Alice is a Python developer",
            uris=["ctx://acme/users/alice/memories/profile"],
            tokens=45,
        )
        buffer.set_cached_slot("identity", identity)

        # S2: Prefetch arrives
        prefetch_hits = [
            SeedHit(uri="ctx://test/1", score=0.9, level=0),
            SeedHit(uri="ctx://test/2", score=0.8, level=0),
        ]
        buffer.set_pending_injection(prefetch_hits)

        # S3: Assembly consumes prefetch and tracks injection
        working_set = buffer.get_pending_injection()
        assert working_set is not None
        injected_uris = [hit.uri for hit in working_set]
        buffer.mark_injected(injected_uris)
        buffer.clear_pending_injection()

        # S4: Next turn - prefetch still empty, identity cached
        assert buffer.get_pending_injection() is None
        cached_identity = buffer.get_cached_slot("identity")
        assert cached_identity is not None
        assert cached_identity.content == "Alice is a Python developer"

        # S5: New prefetch avoids duplicates
        new_hits = [
            SeedHit(uri="ctx://test/1", score=0.95, level=0),  # Duplicate
            SeedHit(uri="ctx://test/3", score=0.75, level=0),  # New
        ]
        buffer.set_pending_injection(new_hits)

        # Filter out already-injected URIs
        fresh_hits = [h for h in new_hits if not buffer.was_injected(h.uri)]
        assert len(fresh_hits) == 1
        assert fresh_hits[0].uri == "ctx://test/3"
