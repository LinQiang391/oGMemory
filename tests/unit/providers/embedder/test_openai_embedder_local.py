"""Tests for OpenAIEmbedder compatibility with keyless local servers."""

from __future__ import annotations

import pytest

from providers.embedder import get_openai_embedder


def test_openai_embedder_accepts_empty_key_when_base_url_provided():
    OpenAIEmbedder, _ = get_openai_embedder()
    emb = OpenAIEmbedder(
        api_key="",
        base_url="http://127.0.0.1:8345/v1/",
        model="text-embedding-ada-002",
    )
    assert emb._api_key == "no-key-required"
    assert emb._base_url == "http://127.0.0.1:8345/v1/"


def test_openai_embedder_rejects_missing_key_without_base_url():
    OpenAIEmbedder, _ = get_openai_embedder()
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        OpenAIEmbedder(api_key=None, base_url=None)
