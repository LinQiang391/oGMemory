"""Focused tests for MemoryService LLM parameter forwarding."""

import threading
from unittest.mock import Mock

import server.memory_service as memory_service_module
from providers.config import ProviderConfig


def test_get_llm_forwards_all_existing_model_parameters(monkeypatch):
    llm_ctor = Mock(name="OpenAILLM")
    monkeypatch.setattr(
        memory_service_module,
        "get_openai_llm",
        Mock(return_value=(llm_ctor, Mock())),
    )
    service = object.__new__(memory_service_module.MemoryService)
    service._init_lock = threading.RLock()
    service._llm = None
    service._provider_cfg = ProviderConfig(
        provider="openai",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_llm_model="test-model",
        llm_temperature=0.2,
        llm_max_tokens=321,
        llm_json_mode=True,
        llm_reasoning_effort="xhigh",
        llm_is_reasoning_model=True,
        llm_extra_body={"thinking": {"type": "enabled"}},
    )

    service.get_llm()

    llm_ctor.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="test-model",
        temperature=0.2,
        max_tokens=321,
        json_mode=True,
        reasoning_effort="xhigh",
        is_reasoning_model=True,
        extra_body={"thinking": {"type": "enabled"}},
    )
