"""Unit tests for MockEmbedder."""

import pytest

from providers.embedder.mock_embedder import MockEmbedder, ZeroEmbedder, MOCK_EMBEDDING_DIM


class TestMockEmbedder:
    """Tests for MockEmbedder deterministic vector generation."""

    def test_embed_texts_returns_correct_count(self):
        embedder = MockEmbedder()
        texts = ["hello", "world", "test"]
        results = embedder.embed_texts(texts)

        assert len(results) == len(texts)

    def test_embed_texts_returns_correct_dimension(self):
        embedder = MockEmbedder()
        texts = ["hello"]
        results = embedder.embed_texts(texts)

        assert len(results[0]) == MOCK_EMBEDDING_DIM

    def test_embed_texts_is_deterministic(self):
        """Same text should produce same vector across calls."""
        embedder = MockEmbedder()
        text = "consistent text"

        vector1 = embedder.embed_texts([text])[0]
        vector2 = embedder.embed_texts([text])[0]

        assert vector1 == vector2

    def test_embed_texts_different_texts_different_vectors(self):
        """Different texts should produce different vectors."""
        embedder = MockEmbedder()

        vector1 = embedder.embed_texts(["text one"])[0]
        vector2 = embedder.embed_texts(["text two"])[0]

        assert vector1 != vector2

    def test_custom_dimension(self):
        embedder = MockEmbedder(dimension=128)
        texts = ["test"]
        results = embedder.embed_texts(texts)

        assert len(results[0]) == 128

    def test_empty_text(self):
        embedder = MockEmbedder()
        results = embedder.embed_texts([""])

        assert len(results[0]) == MOCK_EMBEDDING_DIM


class TestZeroEmbedder:
    """Tests for ZeroEmbedder."""

    def test_returns_zero_vectors(self):
        embedder = ZeroEmbedder()
        texts = ["hello", "world"]
        results = embedder.embed_texts(texts)

        assert len(results) == 2
        assert all(all(v == 0.0 for v in vec) for vec in results)

    def test_all_identical(self):
        """All texts get identical zero vectors."""
        embedder = ZeroEmbedder()
        results = embedder.embed_texts(["a", "b", "c"])

        # All vectors should be identical
        assert results[0] == results[1] == results[2]
