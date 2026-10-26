"""Compatibility checks for the original Ark text-embedding configuration."""

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from cli.commands.onboard import _step3_embedding
from providers.embedder import volcengine_embedder
from providers.unified_config import OgMemConfig


@pytest.mark.parametrize("data_shape", ["object", "list"])
def test_original_text_configuration_builds_and_embeds(monkeypatch, data_shape):
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    data = [SimpleNamespace(embedding=vector) for vector in vectors]
    if data_shape == "list":
        data = [[item] for item in data]
    client = SimpleNamespace(multimodal_embeddings=SimpleNamespace(create=Mock(side_effect=[
        SimpleNamespace(data=item, usage=SimpleNamespace(total_tokens=3)) for item in data
    ])))
    ark = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "volcenginesdkarkruntime", SimpleNamespace(Ark=ark))
    tracker = Mock()
    monkeypatch.setattr(volcengine_embedder, "TokenTracker", Mock(return_value=tracker))
    cfg = OgMemConfig(
        provider="openai", embedding_provider="volcengine",
        openai_api_key="chat-key", openai_embedding_api_key="embedding-key",
        openai_embedding_base_url="https://ark.example.test/api/v3",
        openai_embedding_model="doubao-embedding-vision-250615", opengauss_dimension=2,
    )

    embedder = cfg.to_provider_config().create_embedder()
    assert embedder.embed_texts(["first document", "second document"]) == vectors
    assert embedder.model == "doubao-embedding-vision-250615"
    assert embedder.dimension == 2
    ark.assert_called_once_with(api_key="embedding-key", base_url="https://ark.example.test/api/v3")
    assert client.multimodal_embeddings.create.call_args_list == [
        call(input=[{"type": "text", "text": text}], model=embedder.model, dimensions=2)
        for text in ["first document", "second document"]
    ]
    assert tracker.record_embed.call_args_list == [
        call(3, usage_known=True), call(3, usage_known=True)
    ]


def test_original_onboarding_reuses_ark_text_embedding_defaults():
    args = Namespace(non_interactive=True, embedding_provider=None, embedding_model=None,
                     embedding_dimension=None, embedding_base_url=None, embedding_api_key=None)
    result = _step3_embedding(args, {"provider": "volcengine", "api_key": "original-key",
                                    "base_url": "https://ark.example.test/api/v3"})
    assert result["provider"] == "volcengine"
    assert result["model"] == "doubao-embedding-vision-250615"
    assert result["dimension"] == 1024
    assert result["base_url"] == "https://ark.example.test/api/v3"
    assert result["api_key"] == "original-key"
