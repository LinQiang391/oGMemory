"""Tests for SessionCompressor."""

from unittest.mock import Mock

import pytest

from session.compressor import SessionCompressor


def test_compress_with_llm(compressor, sample_messages):
    """Test compression with LLM available."""
    overview, abstract = compressor.compress(sample_messages)

    assert overview == "Test session overview"
    assert abstract == "Test abstract"
    compressor.llm.complete_json.assert_called_once()


def test_compress_empty_messages(compressor_no_llm):
    """Test compression with empty message list."""
    overview, abstract = compressor_no_llm.compress([])

    assert overview == ""
    assert abstract == "Empty session"


def test_compress_fallback_no_llm(compressor_no_llm, sample_messages):
    """Test fallback compression without LLM."""
    overview, abstract = compressor_no_llm.compress(sample_messages)

    assert overview is not None
    assert len(overview) > 0
    assert "4 messages" in overview  # Should mention message count
    assert abstract is not None
    assert len(abstract) <= 100  # Should respect max length


def test_compress_fallback_abstract_length(compressor_no_llm):
    """Test that fallback abstract is truncated to 100 chars."""
    long_messages = [
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
    ]

    overview, abstract = compressor_no_llm.compress(long_messages)

    assert len(abstract) <= 100


def test_compress_llm_error_fallback(compressor, sample_messages):
    """Test that LLM errors fall back to simple compression."""
    compressor.llm.complete_json.side_effect = Exception("LLM failed")

    overview, abstract = compressor.compress(sample_messages)

    # Should fall back to simple compression
    assert overview is not None
    assert len(overview) > 0
    assert abstract is not None
    assert len(abstract) <= 100


def test_compress_llm_abstract_truncation(compressor, sample_messages):
    """Test that LLM abstract is truncated if too long."""
    compressor.llm.complete_json.return_value = {
        "overview": "Test overview",
        "abstract": "x" * 150,  # Too long
    }

    overview, abstract = compressor.compress(sample_messages)

    assert abstract == "x" * 100  # Should be truncated


def test_compress_fallback_role_counts(compressor_no_llm):
    """Test that fallback compression counts roles correctly."""
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
    ]

    overview, abstract = compressor_no_llm.compress(messages)

    assert "3 from user" in overview
    assert "2 from assistant" in overview


def test_compress_fallback_first_message_preview(compressor_no_llm):
    """Test that fallback compression includes first message preview."""
    messages = [
        {"role": "user", "content": "This is a very long first message that should be truncated"},
        {"role": "assistant", "content": "Response"},
    ]

    overview, abstract = compressor_no_llm.compress(messages)

    assert "Started with:" in overview
    assert "This is a very long first message" in overview


def test_compress_fallback_abstract_from_conversation(compressor_no_llm):
    """Test that fallback abstract derives from conversation."""
    messages = [
        {"role": "user", "content": "How do I write Python code?"},
        {"role": "assistant", "content": "Here's how you write Python code..."},
    ]

    overview, abstract = compressor_no_llm.compress(messages)

    assert "Python" in abstract or "write" in abstract


def test_compress_only_user_messages(compressor_no_llm):
    """Test compression with only user messages (no assistant)."""
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"},
    ]

    overview, abstract = compressor_no_llm.compress(messages)

    assert "2 messages" in overview
    assert abstract is not None


def test_compress_only_assistant_messages(compressor_no_llm):
    """Test compression with only assistant messages (no user)."""
    messages = [
        {"role": "assistant", "content": "Response 1"},
        {"role": "assistant", "content": "Response 2"},
    ]

    overview, abstract = compressor_no_llm.compress(messages)

    assert "2 messages" in overview
    assert abstract is not None
