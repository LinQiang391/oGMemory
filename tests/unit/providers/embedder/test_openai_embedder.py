"""Unit tests for the OpenAI embedder providers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import providers.embedder as embedder_package
import providers.embedder.openai_embedder as openai_embedder_module

# Lazy import OpenAIEmbedder
OpenAIEmbedder, CachedOpenAIEmbedder = None, None


def _get_openai_embedder():
    global OpenAIEmbedder, CachedOpenAIEmbedder
    if OpenAIEmbedder is None:
        from providers.embedder import get_openai_embedder
        OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
    return OpenAIEmbedder, CachedOpenAIEmbedder


def _embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


@pytest.fixture
def fake_client() -> Mock:
    return Mock()


@pytest.fixture(autouse=True)
def fake_openai_ctor(monkeypatch: pytest.MonkeyPatch, fake_client: Mock) -> Mock:
    ctor = Mock(return_value=fake_client)
    monkeypatch.setattr(openai_embedder_module, "OpenAI", ctor)
    return ctor


class TestOpenAIEmbedder:
    def test_lazy_loader_returns_classes(self):
        openai_embedder, cached_embedder = embedder_package.get_openai_embedder()
        assert openai_embedder is openai_embedder_module.OpenAIEmbedder
        assert cached_embedder is openai_embedder_module.CachedOpenAIEmbedder

    def test_init_uses_explicit_credentials_and_dimension(self, fake_openai_ctor: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="text-embedding-3-large",
        )

        fake_openai_ctor.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.com/v1",
        )
        assert embedder.dimension == openai_embedder_module.EMBEDDING_3_LARGE_DIM
        assert embedder.model == "text-embedding-3-large"

    def test_init_falls_back_to_environment_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_openai_ctor: Mock,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        openai_embedder_module.OpenAIEmbedder()

        fake_openai_ctor.assert_called_once_with(api_key="env-key")

    def test_init_without_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            openai_embedder_module.OpenAIEmbedder(api_key=None)

    def test_embed_texts_returns_empty_for_empty_input(self, fake_client: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        assert embedder.embed_texts([]) == []
        fake_client.embeddings.create.assert_not_called()

    def test_embed_texts_calls_client_and_returns_vectors(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1, 0.2], [0.3, 0.4]])
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            model="custom-model",
            dimension=2,
        )

        result = embedder.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        fake_client.embeddings.create.assert_called_once_with(
            input=["hello", "world"],
            model="custom-model",
        )

    def test_embed_texts_rejects_batches_larger_than_limit(self, fake_client: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        with pytest.raises(ValueError, match="Batch size too large"):
            embedder.embed_texts(["x"] * 2049)

        fake_client.embeddings.create.assert_not_called()

    def test_embed_texts_validates_embedding_dimension(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1]])
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key", dimension=2)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            embedder.embed_texts(["hello"])

    def test_unknown_model_dimension_defaults_to_ada_size(self):
        embedder = object.__new__(openai_embedder_module.OpenAIEmbedder)

        assert embedder._get_model_dimension("unknown") == openai_embedder_module.ADA_002_DIM


class TestCachedOpenAIEmbedder:
    def test_cached_embedder_reuses_cached_vectors_and_preserves_order(self, fake_client: Mock):
        fake_client.embeddings.create.side_effect = [
            _embedding_response([[1.0, 1.1], [2.0, 2.1]]),
            _embedding_response([[3.0, 3.1]]),
        ]
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=2,
            cache_max_size=3,
        )

        first = embedder.embed_texts(["alpha", "beta"])
        second = embedder.embed_texts(["beta", "gamma", "alpha"])

        assert first == [[1.0, 1.1], [2.0, 2.1]]
        assert second == [[2.0, 2.1], [3.0, 3.1], [1.0, 1.1]]
        assert fake_client.embeddings.create.call_count == 2
        assert embedder.cache_size == 3

    def test_cache_eviction_removes_oldest_entry(self, fake_client: Mock):
        fake_client.embeddings.create.side_effect = [
            _embedding_response([[1.0]]),
            _embedding_response([[2.0]]),
            _embedding_response([[3.0]]),
            _embedding_response([[4.0]]),
        ]
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=1,
            cache_max_size=2,
        )

        embedder.embed_texts(["first"])
        embedder.embed_texts(["second"])
        embedder.embed_texts(["third"])
        embedder.embed_texts(["first"])

        assert fake_client.embeddings.create.call_count == 4
        assert embedder.cache_size == 2

    def test_clear_cache_removes_all_entries(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[1.0]])
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=1,
        )

        embedder.embed_texts(["value"])
        embedder.clear_cache()

        assert embedder.cache_size == 0
