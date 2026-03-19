"""Tests for OpenAIEmbedder.

These tests require valid API credentials and will make real API calls.
Set the following environment variables before running:
- OPENAI_API_KEY
- OPENAI_BASE_URL (optional, for custom endpoints)
"""

import os
import pytest

from core.interfaces import Embedder

# Lazy import OpenAIEmbedder
OpenAIEmbedder, CachedOpenAIEmbedder = None, None


def _get_openai_embedder():
    global OpenAIEmbedder, CachedOpenAIEmbedder
    if OpenAIEmbedder is None:
        from providers.embedder import get_openai_embedder
        OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
    return OpenAIEmbedder, CachedOpenAIEmbedder


# Skip these tests if no API key is provided
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def embedder():
    """Create OpenAI embedder for testing."""
    OpenAIEmbedder, _ = _get_openai_embedder()
    return OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
    )


@pytest.fixture
def cached_embedder():
    """Create cached OpenAI embedder for testing."""
    _, CachedOpenAIEmbedder = _get_openai_embedder()
    return CachedOpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
    )


class TestOpenAIEmbedder:
    """Test OpenAI embedder functionality."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        OpenAIEmbedder, _ = _get_openai_embedder()
        embedder = OpenAIEmbedder(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        assert isinstance(embedder, Embedder)
        assert embedder.model == "text-embedding-ada-002"

    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        OpenAIEmbedder, _ = _get_openai_embedder()
        import os
        # Temporarily unset env var
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key"):
                OpenAIEmbedder(api_key=None, base_url=None)
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        results = embedder.embed_texts(["Hello, world!"])
        assert len(results) == 1
        assert len(results[0]) == embedder.dimension
        assert all(isinstance(x, float) for x in results[0])

    def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts."""
        texts = ["Hello", "World", "Test"]
        results = embedder.embed_texts(texts)
        assert len(results) == len(texts)
        for result in results:
            assert len(result) == embedder.dimension

    def test_embeddings_are_deterministic(self, embedder):
        """Test that same text produces same embedding."""
        text = "deterministic test"
        results1 = embedder.embed_texts([text])
        results2 = embedder.embed_texts([text])

        assert len(results1) == len(results2) == 1
        assert results1[0] == results2[0]

    def test_embeddings_are_different_for_different_texts(self, embedder):
        """Test that different texts produce different embeddings."""
        results = embedder.embed_texts(["apple", "banana", "cherry"])

        # Calculate cosine similarity between pairs
        def cosine_similarity(a, b):
            import math
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(y * y for y in b))
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0

        # Similar texts should be more similar, but all should be distinct
        sim_01 = cosine_similarity(results[0], results[1])
        sim_02 = cosine_similarity(results[0], results[2])
        sim_12 = cosine_similarity(results[1], results[2])

        # None should be identical
        assert sim_01 < 1.0
        assert sim_02 < 1.0
        assert sim_12 < 1.0

    def test_empty_text_list_returns_empty(self, embedder):
        """Test that empty input returns empty list."""
        results = embedder.embed_texts([])
        assert results == []

    def test_batch_size_limit(self, embedder):
        """Test that batch size limit is enforced."""
        # Try to embed more than 2048 texts
        large_batch = ["text"] * 2500

        with pytest.raises(ValueError, match="Batch size too large"):
            embedder.embed_texts(large_batch)


class TestCachedOpenAIEmbedder:
    """Test cached OpenAI embedder functionality."""

    def test_cache_hit(self, cached_embedder):
        """Test that cached embeddings are returned without API call."""
        text = "cache test"

        # First call should hit API
        results1 = cached_embedder.embed_texts([text])
        assert len(results1) == 1
        cache_size_1 = cached_embedder.cache_size

        # Second call should use cache
        results2 = cached_embedder.embed_texts([text])
        assert results2 == results1
        assert cached_embedder.cache_size == cache_size_1  # No new cache entry

    def test_cache_miss(self, cached_embedder):
        """Test that uncached texts hit the API."""
        # Embed different texts
        results1 = cached_embedder.embed_texts(["text1"])
        cache_size_1 = cached_embedder.cache_size

        results2 = cached_embedder.embed_texts(["text2"])
        assert cached_embedder.cache_size == cache_size_1 + 1  # New cache entry

    def test_clear_cache(self, cached_embedder):
        """Test clearing the cache."""
        cached_embedder.embed_texts(["test"])
        assert cached_embedder.cache_size > 0

        cached_embedder.clear_cache()
        assert cached_embedder.cache_size == 0

    def test_cache_max_size_eviction(self, cached_embedder):
        """Test that oldest entries are evicted when cache is full."""
        _, CachedOpenAIEmbedder = _get_openai_embedder()
        # Create embedder with small cache
        small_cache_embedder = CachedOpenAIEmbedder(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            cache_max_size=3,
        )

        # Fill cache
        for i in range(5):
            small_cache_embedder.embed_texts([f"text{i}"])

        # Cache should not exceed max size
        assert small_cache_embedder.cache_size <= 3
