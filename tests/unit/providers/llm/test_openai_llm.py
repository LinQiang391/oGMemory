"""Tests for OpenAILLM.

These tests require valid API credentials and will make real API calls.
Set the following environment variables before running:
- OPENAI_API_KEY
- OPENAI_BASE_URL (optional, for custom endpoints)
"""

import os
import pytest

from providers.llm import OpenAILLM, CachedOpenAILLM, GPT_4O_MINI
from core.interfaces import LLM


# Skip these tests if no API key is provided
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def llm():
    """Create OpenAI LLM for testing."""
    return OpenAILLM(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_LLM_MODEL", GPT_4O_MINI),
    )


@pytest.fixture
def cached_llm():
    """Create cached OpenAI LLM for testing."""
    return CachedOpenAILLM(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_LLM_MODEL", GPT_4O_MINI),
    )


class TestOpenAILLM:
    """Test OpenAI LLM functionality."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        llm = OpenAILLM(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        assert isinstance(llm, LLM)
        assert llm.model == os.environ.get("OPENAI_LLM_MODEL", GPT_4O_MINI)

    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key"):
                OpenAILLM(api_key=None, base_url=None)
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_complete_json_simple_schema(self, llm):
        """Test JSON completion with simple schema."""
        prompt = "Extract a user's name from: 'My name is John Doe'"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        result = llm.complete_json(prompt, schema)

        assert isinstance(result, dict)
        assert "name" in result
        assert isinstance(result["name"], str)

    def test_complete_json_array_response(self, llm):
        """Test JSON completion returning an array."""
        prompt = "List 3 programming languages"
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }

        result = llm.complete_json(prompt, schema)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, str) for item in result)

    def test_complete_json_nested_schema(self, llm):
        """Test JSON completion with nested schema."""
        prompt = "Extract user profile from: 'John is a 30-year-old software engineer'"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "age", "occupation"],
        }

        result = llm.complete_json(prompt, schema)

        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["occupation"] == "software engineer"

    def test_complete_json_with_candidates_structure(self, llm):
        """Test JSON completion matching CandidateMemory structure."""
        prompt = "Extract user preferences from: 'I like coffee and use VS Code'"
        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "abstract": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["category", "abstract", "confidence"],
                    },
                },
            },
            "required": ["candidates"],
        }

        result = llm.complete_json(prompt, schema)

        assert isinstance(result, dict)
        assert "candidates" in result
        assert isinstance(result["candidates"], list)
        # At least one candidate should be extracted
        assert len(result["candidates"]) > 0

    def test_response_matches_model(self, llm):
        """Test that the specified model is used."""
        assert llm.model == os.environ.get("OPENAI_LLM_MODEL", GPT_4O_MINI)


class TestCachedOpenAILLM:
    """Test cached OpenAI LLM functionality."""

    def test_cache_hit(self, cached_llm):
        """Test that cached responses are returned without API call."""
        prompt = "Repeat the word: test"
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        # First call
        result1 = cached_llm.complete_json(prompt, schema)
        stats1 = cached_llm.cache_stats

        # Second call (should hit cache)
        result2 = cached_llm.complete_json(prompt, schema)
        stats2 = cached_llm.cache_stats

        assert result1 == result2
        assert stats2["hits"] > stats1["hits"]

    def test_cache_stats(self, cached_llm):
        """Test cache statistics tracking."""
        prompt = "Say: hello"
        schema = {"type": "object", "properties": {"message": {"type": "string"}}}

        # First call (miss)
        cached_llm.complete_json(prompt, schema)
        stats = cached_llm.cache_stats
        assert stats["misses"] > 0
        assert stats["hits"] == 0

        # Second call (hit)
        cached_llm.complete_json(prompt, schema)
        stats = cached_llm.cache_stats
        assert stats["hits"] > 0
        assert stats["hit_rate"] > 0

    def test_clear_cache(self, cached_llm):
        """Test clearing the cache."""
        cached_llm.complete_json("test", {"type": "object", "properties": {}})
        assert cached_llm.cache_stats["size"] > 0

        cached_llm.clear_cache()

        assert cached_llm.cache_stats["size"] == 0
        assert cached_llm.cache_stats["hits"] == 0
        assert cached_llm.cache_stats["misses"] == 0
