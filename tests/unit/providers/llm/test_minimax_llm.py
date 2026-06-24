"""Tests for :class:`MiniMaxLLM`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import providers.llm.minimax_llm as minimax_llm_module


@pytest.fixture(autouse=True)
def _isolate_openai_ctor():
    """Patch the OpenAI ctor (in openai_llm, where MiniMaxLLM imports
    it from) so init doesn't try to hit the network."""
    from providers.llm import openai_llm as openai_llm_module
    with patch.object(openai_llm_module, "OpenAI") as ctor:
        ctor.return_value = None
        yield ctor


class TestMiniMaxLLM:
    def test_lazy_loader_returns_class(self):
        from providers.llm import get_minimax_llm
        cls = get_minimax_llm()
        assert cls is minimax_llm_module.MiniMaxLLM

    def test_default_base_url(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_BASE_URL", raising=False)
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._base_url == "https://api.minimaxi.com/v1"
        # OpenAI client constructed with the expected base_url.
        _isolate_openai_ctor.assert_called_once()
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["base_url"] == "https://api.minimaxi.com/v1"
        assert kwargs["api_key"] == "test-key"

    def test_env_base_url_override(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("OGMEM_MINIMAX_BASE_URL", "https://proxy.example/v1")
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._base_url == "https://proxy.example/v1"

    def test_explicit_args_override_env(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "env-key")
        llm = minimax_llm_module.MiniMaxLLM(
            api_key="arg-key", base_url="https://explicit.example/v1",
        )
        assert llm._base_url == "https://explicit.example/v1"
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["api_key"] == "arg-key"

    def test_minimax_api_key_env_var(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-only-key")
        minimax_llm_module.MiniMaxLLM()
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["api_key"] == "minimax-only-key"

    def test_missing_api_key_raises(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            minimax_llm_module.MiniMaxLLM()

    def test_default_model(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        monkeypatch.delenv("OGMEM_MINIMAX_MODEL", raising=False)
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._model == "MiniMax-M3"

    def test_inherits_complete_multimodal(self, _isolate_openai_ctor, monkeypatch):
        """MiniMaxLLM must satisfy the LLM Protocol including
        complete_multimodal — the multimodal path is the whole point."""
        from core.interfaces import LLM
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        llm = minimax_llm_module.MiniMaxLLM()
        assert isinstance(llm, LLM)
        assert hasattr(llm, "complete_multimodal")
        assert callable(llm.complete_multimodal)


def test_normalise_base_url_strips_chat_completions_endpoint():
    from providers.llm.minimax_llm import _normalise_base_url
    assert _normalise_base_url(
        "https://api.minimaxi.com/v1/chat/completions"
    ) == "https://api.minimaxi.com/v1"
