"""Tests for provider configuration and factory helpers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import providers.config as config_module


def test_from_env_normalizes_base_url_and_parses_numeric_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXTENGINE_PROVIDER", "openai-cached")
    monkeypatch.setenv("OGMEM_API_KEY", "secret")
    monkeypatch.setenv("OGMEM_BASE_URL", "https://example.com/api")
    monkeypatch.setenv("OGMEM_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OGMEM_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("VECTOR_DB_TYPE", "opengauss")
    monkeypatch.setenv("OPENGAUSS_CONNECTION_STRING", "postgres://db")
    monkeypatch.setenv("OPENGAUSS_DIMENSION", "3072")
    monkeypatch.setenv("OPENGAUSS_TABLE_NAME", "memories")
    monkeypatch.setenv("OPENGAUSS_POOL_SIZE", "7")

    cfg = config_module.ProviderConfig.from_env()

    assert cfg.provider == "openai-cached"
    assert cfg.openai_base_url == "https://example.com/api/v1"
    assert cfg.openai_embedding_model == "text-embedding-3-small"
    assert cfg.openai_llm_model == "gpt-4o"
    assert cfg.vector_db_type == "opengauss"
    assert cfg.opengauss_dimension == 3072
    assert cfg.opengauss_table_name == "memories"
    assert cfg.opengauss_pool_size == 7


def test_from_env_keeps_existing_v1_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OGMEM_BASE_URL", "https://example.com/v1")

    cfg = config_module.ProviderConfig.from_env()

    assert cfg.openai_base_url == "https://example.com/v1"


def test_create_embedder_returns_mock_embedder_for_mock_provider():
    cfg = config_module.ProviderConfig(provider="mock")

    embedder = cfg.create_embedder()

    assert isinstance(embedder, config_module.MockEmbedder)


def test_create_embedder_uses_cached_openai_factory(monkeypatch: pytest.MonkeyPatch):
    openai_embedder = Mock(name="OpenAIEmbedder")
    cached_embedder = Mock(name="CachedOpenAIEmbedder")
    monkeypatch.setattr(
        config_module,
        "get_openai_embedder",
        Mock(return_value=(openai_embedder, cached_embedder)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_embedding_model="text-embedding-3-small",
        cache_max_size=123,
    )

    cfg.create_embedder()

    cached_embedder.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="text-embedding-3-small",
        cache_max_size=123,
    )


def test_create_embedder_uses_non_cached_openai_factory_when_cache_disabled(monkeypatch: pytest.MonkeyPatch):
    openai_embedder = Mock(name="OpenAIEmbedder")
    cached_embedder = Mock(name="CachedOpenAIEmbedder")
    monkeypatch.setattr(
        config_module,
        "get_openai_embedder",
        Mock(return_value=(openai_embedder, cached_embedder)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        enable_cache=False,
    )

    cfg.create_embedder()

    openai_embedder.assert_called_once()
    cached_embedder.assert_not_called()


def test_create_vector_index_returns_in_memory_index():
    cfg = config_module.ProviderConfig(vector_db_type="memory", opengauss_dimension=8)

    index = cfg.create_vector_index()

    assert isinstance(index, config_module.InMemoryVectorIndex)
    assert index._dimension == 8


def test_create_vector_index_requires_connection_string_for_opengauss():
    cfg = config_module.ProviderConfig(vector_db_type="opengauss", opengauss_connection_string=None)

    with pytest.raises(ValueError, match="OPENGAUSS_CONNECTION_STRING"):
        cfg.create_vector_index()


def test_create_vector_index_builds_opengauss_index(monkeypatch: pytest.MonkeyPatch):
    ctor = Mock(name="OpenGaussVectorIndex")
    monkeypatch.setattr(config_module, "OpenGaussVectorIndex", ctor)
    cfg = config_module.ProviderConfig(
        vector_db_type="opengauss",
        opengauss_connection_string="postgres://db",
        opengauss_dimension=16,
        opengauss_table_name="vectors",
        opengauss_pool_size=9,
    )

    cfg.create_vector_index()

    ctor.assert_called_once_with(
        connection_string="postgres://db",
        dimension=16,
        table_name="vectors",
        pool_size=9,
    )


def test_create_llm_uses_openai_factory(monkeypatch: pytest.MonkeyPatch):
    openai_llm = Mock(name="OpenAILLM")
    cached_llm = Mock(name="CachedOpenAILLM")
    monkeypatch.setattr(
        config_module,
        "get_openai_llm",
        Mock(return_value=(openai_llm, cached_llm)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_llm_model="gpt-4o",
        llm_temperature=0.1,
        llm_max_tokens=42,
    )

    cfg.create_llm()

    openai_llm.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="gpt-4o",
        temperature=0.1,
        max_tokens=42,
    )


def test_create_llm_uses_cached_factory_when_enabled(monkeypatch: pytest.MonkeyPatch):
    openai_llm = Mock(name="OpenAILLM")
    cached_llm = Mock(name="CachedOpenAILLM")
    monkeypatch.setattr(
        config_module,
        "get_openai_llm",
        Mock(return_value=(openai_llm, cached_llm)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        cache_max_size=12,
    )

    cfg.create_llm()

    cached_llm.assert_called_once()
    openai_llm.assert_not_called()


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (config_module.ProviderConfig(provider="mystery"), "Unknown provider type"),
        (config_module.ProviderConfig(vector_db_type="mystery"), "Unknown vector_db_type"),
    ],
)
def test_invalid_configuration_branches_raise(cfg, message):
    target = cfg.create_llm if cfg.provider == "mystery" else cfg.create_vector_index

    with pytest.raises(ValueError, match=message):
        target()


def test_helper_functions_use_default_config(monkeypatch: pytest.MonkeyPatch):
    default_config = Mock()
    default_config.create_embedder.return_value = "embedder"
    default_config.create_llm.return_value = "llm"
    default_config.create_vector_index.return_value = "index"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", default_config)

    assert config_module.get_embedder() == "embedder"
    assert config_module.get_llm() == "llm"
    assert config_module.get_vector_index() == "index"
