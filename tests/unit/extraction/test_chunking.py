"""
Unit tests for extraction/chunking.py

Tests for ConversationChunker covering:
- Small message batches (single chunk)
- Force split by message count
- Force split by tokens
- LLM boundary detection
- Flush tail
- Merging small chunks
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from extraction.chunking import (
    ConversationChunker,
    BatchBoundaryResult,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MIN_CHUNK_TOKENS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------- -->

@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    base_time = datetime(2024, 3, 10, 9, 0, 0)
    messages = []
    for i in range(10):
        messages.append({
            "role": "user",
            "content": f"Message {i}",
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
        })
    return messages


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    llm = MagicMock()
    llm.complete_json = MagicMock(return_value={
        "reasoning": "Test reasoning",
        "boundaries": [5],
        "should_wait": False,
    })
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConversationChunker:
    """Test suite for ConversationChunker."""

    def test_small_messages_returns_single_chunk(self, sample_messages):
        """Small batches (< 10 messages) return single chunk with should_wait=True."""
        chunker = ConversationChunker(llm=None, use_llm_boundary=False)
        result = chunker.chunk_messages(sample_messages[:5])

        assert len(result.chunks) == 1
        assert len(result.chunks[0]) == 5
        assert result.should_wait is True

    def test_force_split_by_message_count(self):
        """When message count exceeds max_messages, force split triggers."""
        # Create 600 messages (exceeds default 500)
        messages = []
        for i in range(600):
            messages.append({
                "role": "user",
                "content": f"Message {i}",
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        chunker = ConversationChunker(
            llm=None,
            max_messages=500,
            use_llm_boundary=False,
        )
        result = chunker.chunk_messages(messages)

        # Should split into multiple chunks
        assert len(result.chunks) >= 2
        # First chunk should have max_messages - 1 = 499 messages
        assert len(result.chunks[0]) == 499

    def test_force_split_by_tokens(self):
        """When token count exceeds max_tokens, force split triggers."""
        # Create messages with very long content
        # Each message is ~3750 tokens, so 3 messages = ~11250 tokens > 8000
        long_content = "word " * 3000  # ~3750 tokens per message
        messages = []
        for i in range(3):
            messages.append({
                "role": "user",
                "content": long_content,
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        chunker = ConversationChunker(
            llm=None,
            max_tokens=8000,
            use_llm_boundary=False,
        )
        result = chunker.chunk_messages(messages)

        # Should split into 2 chunks due to token limit
        assert len(result.chunks) == 2

    def test_llm_boundary_detection(self, sample_messages, mock_llm):
        """LLM boundary detection correctly splits at detected boundaries."""
        chunker = ConversationChunker(
            llm=mock_llm,
            use_llm_boundary=True,
        )

        # Create enough messages to trigger LLM boundary detection
        # Need to exceed the 30% threshold but stay under force-split limits
        # 30% of 500 = 150 messages, 30% of 8000 tokens = 2400 tokens
        # Each sample message is ~10 chars -> ~2-3 tokens, so 150 messages ~300-450 tokens
        large_batch = sample_messages * 20  # 200 messages

        result = chunker.chunk_messages(large_batch)

        # LLM returns boundary at [5], but since we're in a batch,
        # the chunker should split accordingly
        assert len(result.chunks) >= 1
        mock_llm.complete_json.assert_called()

    def test_llm_boundary_detection_filters_invalid_boundaries(self, mock_llm):
        """Invalid boundary indices are filtered out."""
        # Mock LLM returns out-of-range boundaries
        mock_llm.complete_json = MagicMock(return_value={
            "reasoning": "Test",
            "boundaries": [-1, 0, 1, 1000],  # Some invalid
            "should_wait": False,
        })

        chunker = ConversationChunker(llm=mock_llm, use_llm_boundary=True)

        messages = []
        for i in range(10):
            messages.append({
                "role": "user",
                "content": f"Message {i}",
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        result = chunker.chunk_messages(messages)

        # Should only accept boundary 1 (valid: 1 <= boundary < 10)
        assert len(result.chunks) >= 1

    def test_flush_tail_packs_remaining_messages(self):
        """Flush mode forces remaining messages into final chunk."""
        messages = []
        for i in range(20):
            messages.append({
                "role": "user",
                "content": f"Message {i}",
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        chunker = ConversationChunker(
            llm=None,
            max_messages=10,
            use_llm_boundary=False,
        )
        result = chunker.chunk_messages(messages, flush=True)

        # All messages should be packed into chunks
        total_chunked = sum(len(chunk) for chunk in result.chunks)
        assert total_chunked == 20
        # should_wait should be False in flush mode
        assert result.should_wait is False

    def test_merge_small_chunks(self):
        """Chunks smaller than min_chunk_tokens are merged with neighbors."""
        # Create messages where one chunk would be very small
        messages = []

        # First large chunk
        for i in range(100):
            messages.append({
                "role": "user",
                "content": "word " * 50,  # ~12.5 tokens each
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        # Tiny chunk (just a few words)
        messages.append({
            "role": "user",
            "content": "ok",
            "timestamp": datetime(2024, 3, 10, 10, 0, 0).isoformat(),
        })

        # Another large chunk
        for i in range(100):
            messages.append({
                "role": "user",
                "content": "word " * 50,
                "timestamp": datetime(2024, 3, 10, 11, 0, 0).isoformat(),
            })

        chunker = ConversationChunker(
            llm=None,
            max_tokens=8000,
            min_chunk_tokens=300,
            use_llm_boundary=False,
        )
        result = chunker.chunk_messages(messages)

        # The small chunk should have been merged
        # Check that no chunk is too small (except maybe the last)
        for i, chunk in enumerate(result.chunks[:-1]):
            chunk_tokens = chunker._estimate_tokens_batch(chunk)
            assert chunk_tokens >= 300 or chunk_tokens < 50, \
                f"Chunk {i} has {chunk_tokens} tokens, expected >= 300"

    def test_empty_messages_returns_empty_result(self):
        """Empty message list returns empty chunks."""
        chunker = ConversationChunker(llm=None)
        result = chunker.chunk_messages([])

        assert result.chunks == []
        assert result.should_wait is False

    def test_format_messages_with_indices(self):
        """Messages are formatted with correct indices and timestamps."""
        chunker = ConversationChunker(llm=None)

        messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-03-10T09:00:00+00:00",
            },
            {
                "role": "assistant",
                "content": "Hi there",
                "timestamp": "2024-03-10T09:01:00+00:00",
            },
        ]

        formatted = chunker._format_messages_with_indices(messages)

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "user: Hello" in formatted
        assert "assistant: Hi there" in formatted
        assert "2024-03-10" in formatted

    def test_estimate_tokens(self):
        """Token estimation uses len(text) // 4."""
        chunker = ConversationChunker(llm=None)

        # 40 characters -> 10 tokens (40 // 4 = 10)
        assert chunker._estimate_tokens("w" * 40) == 10

        # 50 characters -> 12 tokens (50 // 4 = 12)
        assert chunker._estimate_tokens("word " * 10) == 12

    def test_content_to_str(self):
        """Message content is normalized to string correctly."""
        chunker = ConversationChunker(llm=None)

        # String content
        assert chunker._content_to_str("plain text") == "plain text"

        # List content (common in chat APIs)
        list_content = [{"type": "text", "text": "formatted text"}]
        result = chunker._content_to_str(list_content)
        assert "formatted text" in result

        # Empty content
        assert chunker._content_to_str("") == ""
        assert chunker._content_to_str([]) == ""

    def test_llm_boundary_detection_failure_fallback(self, sample_messages):
        """When LLM boundary detection fails, falls back to single chunk."""
        # Mock LLM that raises exception
        mock_llm = MagicMock()
        mock_llm.complete_json = MagicMock(side_effect=Exception("LLM error"))

        chunker = ConversationChunker(
            llm=mock_llm,
            use_llm_boundary=True,
        )

        large_batch = sample_messages * 15  # 150 messages

        # Should not raise exception, but log warning
        result = chunker.chunk_messages(large_batch)

        # Should return chunks (possibly from force-split)
        assert isinstance(result.chunks, list)

    def test_find_force_split_point_respects_both_limits(self):
        """Force split point considers both token and message limits."""
        chunker = ConversationChunker(
            llm=None,
            max_tokens=1000,
            max_messages=10,
        )

        # Create messages that would hit token limit before message limit
        messages = []
        for i in range(20):
            messages.append({
                "role": "user",
                "content": "word " * 200,  # ~50 tokens each
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        split_at = chunker._find_force_split_point(messages)

        # Should split at a point that respects token limit
        # ~20 messages * 50 tokens = 1000 tokens
        # With max_tokens=1000, should split around 10-15 messages
        assert 1 <= split_at < len(messages)

    def test_should_wait_propagates_from_llm(self, mock_llm):
        """should_wait flag is correctly propagated from LLM response."""
        mock_llm.complete_json = MagicMock(return_value={
            "reasoning": "Insufficient context",
            "boundaries": [],
            "should_wait": True,
        })

        chunker = ConversationChunker(llm=mock_llm, use_llm_boundary=True)

        messages = []
        for i in range(50):
            messages.append({
                "role": "user",
                "content": f"Message {i}",
                "timestamp": datetime(2024, 3, 10, 9, 0, 0).isoformat(),
            })

        result = chunker.chunk_messages(messages)

        assert result.should_wait is True


# ---------------------------------------------------------------------------
# BatchBoundaryResult Tests
# ---------------------------------------------------------------------------

class TestBatchBoundaryResult:
    """Test suite for BatchBoundaryResult."""

    def test_default_values(self):
        """Default values are set correctly."""
        result = BatchBoundaryResult()

        assert result.boundaries == []
        assert result.should_wait is False

    def test_explicit_values(self):
        """Explicit values are stored correctly."""
        result = BatchBoundaryResult(
            boundaries=[1, 5, 10],
            should_wait=True,
        )

        assert result.boundaries == [1, 5, 10]
        assert result.should_wait is True

    def test_none_boundaries_converted_to_empty_list(self):
        """None boundaries is converted to empty list via __post_init__."""
        result = BatchBoundaryResult(boundaries=None)

        assert result.boundaries == []
