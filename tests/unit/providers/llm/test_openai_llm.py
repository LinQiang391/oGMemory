"""Unit tests for the OpenAI LLM providers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import providers.llm as llm_package
import providers.llm.openai_llm as openai_llm_module
from core.errors import ToolCallProtocolError


def _chat_response(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _bad_request(message: str) -> openai_llm_module.BadRequestError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai_llm_module.BadRequestError(
        message,
        response=response,
        body={"error": {"message": message}},
    )


@pytest.fixture
def fake_client() -> Mock:
    return Mock()


@pytest.fixture(autouse=True)
def fake_openai_ctor(monkeypatch: pytest.MonkeyPatch, fake_client: Mock) -> Mock:
    ctor = Mock(return_value=fake_client)
    monkeypatch.setattr(openai_llm_module, "OpenAI", ctor)
    return ctor


class TestOpenAILLM:
    def test_lazy_loader_returns_classes(self):
        openai_llm, cached_llm = llm_package.get_openai_llm()
        assert openai_llm is openai_llm_module.OpenAILLM
        assert cached_llm is openai_llm_module.CachedOpenAILLM

    def test_init_uses_explicit_credentials_and_base_url(self, fake_openai_ctor: Mock):
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key",
            base_url="https://example.com/v1",
            model=openai_llm_module.GPT_4O,
            temperature=0.3,
            max_tokens=99,
        )

        fake_openai_ctor.assert_called_once_with(
            timeout=120.0,
            api_key="test-key",
            base_url="https://example.com/v1",
        )
        assert llm.model == openai_llm_module.GPT_4O

    def test_init_falls_back_to_environment_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_openai_ctor: Mock,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        openai_llm_module.OpenAILLM()

        fake_openai_ctor.assert_called_once_with(timeout=120.0, api_key="env-key")

    def test_init_without_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            openai_llm_module.OpenAILLM(api_key=None)

    def test_complete_json_uses_json_object_mode_for_object_schema(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": "ok"}')
        llm = openai_llm_module.OpenAILLM(api_key="test-key", json_mode=True)

        result = llm.complete_json(
            "Return JSON",
            {"type": "object", "properties": {"value": {"type": "string"}}},
        )

        assert result == {"value": "ok"}
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["messages"][0]["role"] == "system"

    def test_complete_json_uses_text_mode_for_array_schema(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='["a", "b"]')
        llm = openai_llm_module.OpenAILLM(api_key="test-key", json_mode=True)

        result = llm.complete_json(
            "Return list",
            {"type": "array", "items": {"type": "string"}},
        )

        assert result == ["a", "b"]
        # Array schemas don't use json_object format even with json_mode=True
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs

    def test_complete_json_raises_for_invalid_json(self, fake_client: Mock):
        # Use truly unparseable JSON that can't be recovered by any fallback
        fake_client.chat.completions.create.return_value = _chat_response(content="not even trying")
        # json_mode=False means it will use text fallback
        llm = openai_llm_module.OpenAILLM(api_key="test-key", json_mode=False)

        with pytest.raises(ValueError, match="parse LLM response as JSON"):
            llm.complete_json("broken", {"type": "object"})

    def test_complete_json_validates_required_properties(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"other": 1}')
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(ValueError, match="Missing required property"):
            llm.complete_json(
                "missing field",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )

    def test_complete_json_wraps_client_errors(self, fake_client: Mock):
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(RuntimeError, match="OpenAI API error: boom"):
            llm.complete_json("broken", {"type": "object"})

    def test_complete_with_tools_parses_tool_calls(self, fake_client: Mock):
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="search_memories",
                    arguments='{"query":"coffee"}',
                )
            )
        ]
        fake_client.chat.completions.create.return_value = _chat_response(tool_calls=tool_calls)
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        result = llm.complete_with_tools(
            prompt="Find coffee",
            tools=[
                {
                    "name": "search_memories",
                    "description": "Search memories",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
        )

        assert result == [{
            "tool": "search_memories",
            "input": {"query": "coffee"},
            "_llm_usage": {},
        }]
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"][0]["function"]["name"] == "search_memories"

    def test_complete_with_tools_required_success_keeps_required(self, fake_client: Mock):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="extract_event", arguments='{"abstract":"ok"}')
        )
        fake_client.chat.completions.create.return_value = _chat_response(
            tool_calls=[tool_call]
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        llm.complete_with_tools("prompt", tools=[{
            "name": "extract_event",
            "description": "Extract event",
            "input_schema": {"type": "object"},
        }], tool_choice="required")

        assert fake_client.chat.completions.create.call_count == 1
        assert fake_client.chat.completions.create.call_args.kwargs["tool_choice"] == "required"
        assert llm._required_tool_choice_capability == "supported"

    def test_required_empty_retries_once_without_marking_unsupported(
        self,
        fake_client: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_client.chat.completions.create.return_value = _chat_response(tool_calls=None)
        monkeypatch.setattr("time.sleep", lambda _delay: None)
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(ToolCallProtocolError):
            llm.complete_with_tools("prompt", tools=[{
                "name": "extract_event",
                "description": "Extract event",
                "input_schema": {"type": "object"},
            }], tool_choice="required")

        assert fake_client.chat.completions.create.call_count == 2
        assert [
            call.kwargs["tool_choice"]
            for call in fake_client.chat.completions.create.call_args_list
        ] == ["required", "required"]
        assert llm._required_tool_choice_capability == "unknown"

    def test_required_unsupported_falls_back_to_auto_and_caches(
        self,
        fake_client: Mock,
    ):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="extract_event", arguments='{"abstract":"ok"}')
        )
        fake_client.chat.completions.create.side_effect = [
            _bad_request("tool_choice=required is not supported; only auto is supported"),
            _chat_response(tool_calls=[tool_call]),
            _chat_response(tool_calls=[tool_call]),
        ]
        llm = openai_llm_module.OpenAILLM(api_key="test-key")
        tools = [{
            "name": "extract_event",
            "description": "Extract event",
            "input_schema": {"type": "object"},
        }]

        llm.complete_with_tools("first", tools=tools, tool_choice="required")
        llm.complete_with_tools("second", tools=tools, tool_choice="required")

        assert [
            call.kwargs["tool_choice"]
            for call in fake_client.chat.completions.create.call_args_list
        ] == ["required", "auto", "auto"]
        assert llm._required_tool_choice_capability == "unsupported"

    def test_unrelated_bad_request_does_not_downgrade_required(self, fake_client: Mock):
        fake_client.chat.completions.create.side_effect = _bad_request(
            "Invalid tool schema: missing properties"
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(RuntimeError, match="Invalid tool schema"):
            llm.complete_with_tools("prompt", tools=[{
                "name": "extract_event",
                "description": "Extract event",
                "input_schema": {"type": "object"},
            }], tool_choice="required")

        assert fake_client.chat.completions.create.call_count == 1
        assert llm._required_tool_choice_capability == "unknown"

    def test_complete_with_tools_wraps_client_errors(self, fake_client: Mock):
        fake_client.chat.completions.create.side_effect = RuntimeError("tool boom")
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(RuntimeError, match="OpenAI API error: tool boom"):
            llm.complete_with_tools("prompt", tools=[])

    def test_complete_with_tools_empty_required_tool_calls_does_not_log_text_content(
        self,
        fake_client: Mock,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        sensitive_response = "User private memory: Caroline went to the support group on May 7."
        fake_client.chat.completions.create.return_value = _chat_response(
            content=sensitive_response,
            tool_calls=None,
        )
        monkeypatch.setattr("time.sleep", lambda _delay: None)
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with caplog.at_level(logging.WARNING, logger="providers.llm.openai_llm"):
            with pytest.raises(RuntimeError, match="Required tool call returned no tool_calls"):
                llm.complete_with_tools(
                    prompt="Extract private memory",
                    tools=[
                        {
                            "name": "extract_event",
                            "description": "Extract event",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    tool_choice="required",
                )

        assert sensitive_response not in caplog.text
        assert "Caroline went to the support group" not in caplog.text
        assert "text_response=" not in caplog.text
        assert "text_response_len=" in caplog.text
        assert "text_response_sha=" in caplog.text

    def test_build_system_message_contains_schema(self):
        llm = object.__new__(openai_llm_module.OpenAILLM)

        system_message = llm._build_system_message({"type": "object", "properties": {"x": {}}})

        assert "Respond ONLY with valid JSON" in system_message
        assert '"properties"' in system_message

    @pytest.mark.parametrize(
        ("result", "schema", "expected_message"),
        [
            ("not-a-dict", {"type": "object"}, "Expected object"),
            ({"ok": True}, {"type": "array"}, "Expected array"),
        ],
    )
    def test_validate_result_checks_schema_type(self, result, schema, expected_message):
        llm = object.__new__(openai_llm_module.OpenAILLM)

        with pytest.raises(ValueError, match=expected_message):
            llm._validate_result(result, schema)


class TestCachedOpenAILLM:
    def test_cache_hit_and_stats(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": 1}')
        llm = openai_llm_module.CachedOpenAILLM(api_key="test-key", cache_max_size=2)

        first = llm.complete_json("repeat", {"type": "object"})
        second = llm.complete_json("repeat", {"type": "object"})

        assert first == second
        assert fake_client.chat.completions.create.call_count == 1
        assert llm.cache_stats == {"size": 1, "hits": 1, "misses": 1, "hit_rate": 50.0}

    def test_cache_eviction_removes_oldest_entry(self, fake_client: Mock):
        fake_client.chat.completions.create.side_effect = [
            _chat_response(content='{"value": 1}'),
            _chat_response(content='{"value": 2}'),
            _chat_response(content='{"value": 3}'),
            _chat_response(content='{"value": 4}'),
        ]
        llm = openai_llm_module.CachedOpenAILLM(api_key="test-key", cache_max_size=2)

        llm.complete_json("first", {"type": "object"})
        llm.complete_json("second", {"type": "object"})
        llm.complete_json("third", {"type": "object"})
        llm.complete_json("first", {"type": "object"})

        assert fake_client.chat.completions.create.call_count == 4
        assert llm.cache_stats["size"] == 2

    def test_clear_cache_resets_entries_and_stats(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": 1}')
        llm = openai_llm_module.CachedOpenAILLM(api_key="test-key")

        llm.complete_json("prompt", {"type": "object"})
        llm.clear_cache()

        assert llm.cache_stats == {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0}
