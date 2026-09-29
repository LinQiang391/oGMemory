"""Request timeout must survive config loading and provider projection."""

from providers import unified_config
from providers.config import ProviderConfig


def test_timeout_yaml_and_provider_roundtrip(monkeypatch):
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(
        unified_config, "_load_yaml",
        lambda _: {"llm": {"request_timeout_seconds": 300}},
    )
    cfg = unified_config.OgMemConfig.load("ignored.yaml")
    assert cfg.llm_request_timeout_seconds == 300
    provider = ProviderConfig.from_ogmem_config(cfg)
    assert provider.llm_request_timeout_seconds == 300
    assert provider.to_ogmem_config().llm_request_timeout_seconds == 300


def test_timeout_env_and_default(monkeypatch):
    monkeypatch.setattr(unified_config, "_load_yaml", lambda _: {})
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    assert unified_config.OgMemConfig.load("ignored.yaml").llm_request_timeout_seconds == 120
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "301.5")
    assert unified_config.OgMemConfig.load("ignored.yaml").llm_request_timeout_seconds == 301.5
