"""Focused tests for MemoryService LLM parameter forwarding."""

from unittest.mock import Mock

import pytest

import server.memory_service as memory_service_module
from providers.unified_config import OgMemConfig


@pytest.mark.parametrize("provider", ["openai", "mock"])
def test_get_llm_forwards_all_existing_model_parameters(monkeypatch, provider):
    llm_ctor = Mock(name="OpenAILLM")
    monkeypatch.setattr(
        memory_service_module,
        "get_openai_llm",
        Mock(return_value=(llm_ctor, Mock())),
    )
    cfg = OgMemConfig(
        provider=provider,
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_llm_model="test-model",
        llm_temperature=0.2,
        llm_max_tokens=321,
        llm_request_timeout_seconds=300.0,
        llm_json_mode=True,
        llm_reasoning_effort="xhigh",
        llm_is_reasoning_model=True,
        llm_extra_body={"thinking": {"type": "enabled"}},
    )
    service = memory_service_module.MemoryService(config=cfg)
    usage_store = object()
    service._model_usage_store = usage_store

    service.get_llm()

    llm_ctor.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=321,
        request_timeout_seconds=300.0,
        json_mode=True,
        reasoning_effort="xhigh",
        is_reasoning_model=True,
        extra_body={"thinking": {"type": "enabled"}},
    )
    llm_ctor.return_value.token_tracker.set_usage_store.assert_called_once_with(usage_store)


@pytest.mark.parametrize("provider", ["openai", "mock"])
def test_service_constructs_sdk_client_with_configured_timeout(provider):
    """Check the real service constructor, not just ProviderConfig.create_llm.

    No model call or database access: an explicit no-op usage store keeps
    construction local. The mock-with-key path deliberately promotes to OpenAI.
    """
    cfg = OgMemConfig(
        provider=provider,
        openai_api_key="unit-test-key",
        openai_base_url="https://example.invalid/v1",
        llm_request_timeout_seconds=301.5,
    )
    service = memory_service_module.MemoryService(config=cfg)
    service._model_usage_store = Mock()
    llm = service.get_llm()
    try:
        assert llm._client.timeout == 301.5
        assert service.get_llm() is llm
    finally:
        llm._client.close()
