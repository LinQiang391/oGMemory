"""Focused tests for MemoryService LLM parameter forwarding."""

from unittest.mock import Mock

from providers.config import ProviderConfig
import server.memory_service as memory_service_module


def test_get_llm_forwards_all_existing_model_parameters(monkeypatch):
    llm_ctor = Mock(name="OpenAILLM")
    monkeypatch.setattr(
        memory_service_module,
        "get_openai_llm",
        Mock(return_value=(llm_ctor, Mock())),
    )
    service = object.__new__(memory_service_module.MemoryService)
    service._llm = None
    service._provider_cfg = ProviderConfig(
        provider="openai",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_llm_model="test-model",
        llm_temperature=0.2,
        llm_max_tokens=321,
        llm_json_mode=True,
    )

    service.get_llm()

    llm_ctor.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=321,
        json_mode=True,
    )
