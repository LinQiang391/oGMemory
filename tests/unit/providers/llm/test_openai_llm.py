"""Unit tests for the OpenAI LLM providers."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from core.errors import LLMCallError
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
    def test_custom_request_timeout(self, fake_openai_ctor):
        openai_llm_module.OpenAILLM(api_key="test-key", request_timeout_seconds=300)
        fake_openai_ctor.assert_called_once_with(timeout=300, api_key="test-key")

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_nonpositive_timeout(self, timeout):
        with pytest.raises(ValueError, match="must be positive"):
            openai_llm_module.OpenAILLM(api_key="test-key", request_timeout_seconds=timeout)

    @pytest.mark.parametrize("use_messages", [False, True])
    @pytest.mark.parametrize("tool_calls", [
        None,
        [SimpleNamespace(function=SimpleNamespace(
            name="extract_event", arguments='{"private":"partial',
        ))],
    ])
    def test_truncated_tool_output_fails_without_retry_or_partial_repair(
        self, fake_client, tool_calls, caplog, use_messages,
    ):
        fake_client.chat.completions.create.return_value = _chat_response(
            content="private text", tool_calls=tool_calls, finish_reason="length",
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")
        with pytest.raises(LLMCallError, match="output truncated"):
            if use_messages:
                llm.complete_with_tools_messages(
                    [{"role": "user", "content": "prompt"}], [],
                    tool_choice="required",
                )
            else:
                llm.complete_with_tools("prompt", [], tool_choice="required")
        assert fake_client.chat.completions.create.call_count == 1
        assert "private text" not in caplog.text
        assert "max_tokens=4096" in caplog.text

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

    def test_complete_json_forwards_reasoning_configuration(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": "ok"}')
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key",
            reasoning_effort="max",
            is_reasoning_model=True,
            extra_body={"thinking": {"type": "enabled"}},
        )

        llm.complete_json("Return JSON", {"type": "object"})

        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["max_completion_tokens"] == 4096
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_complete_json_omits_unconfigured_reasoning(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": "ok"}')
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        llm.complete_json("Return JSON", {"type": "object"})

        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    def test_custom_model_can_force_reasoning_parameters(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": "ok"}')
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key", model="5.6luna", is_reasoning_model=True,
            reasoning_effort="xhigh",
        )

        llm.complete_json("Return JSON", {"type": "object"})

        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == "xhigh"
        assert kwargs["max_completion_tokens"] == 4096
        assert "temperature" not in kwargs

    def test_reasoning_model_can_be_forced_to_standard_parameters(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(content='{"value": "ok"}')
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key", model="gpt-5-custom", is_reasoning_model=False,
            reasoning_effort="high",
        )

        llm.complete_json("Return JSON", {"type": "object"})

        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096
        assert kwargs["temperature"] == 0.7
        assert "reasoning_effort" not in kwargs

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

    def test_complete_json_skips_prefixed_json_with_wrong_schema_type(self, fake_client: Mock):
        fake_client.chat.completions.create.return_value = _chat_response(
            content='See [1]. Result: {"value": "ok"}'
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        result = llm.complete_json(
            "Return JSON",
            {"type": "object", "required": ["value"]},
        )

        assert result == {"value": "ok"}

    def test_extract_json_from_text_preserves_prefixed_top_level_array(self):
        result = openai_llm_module.OpenAILLM._extract_json_from_text(
            'Result: [{"name": "one"}, {"name": "two"}]'
        )

        assert result == [{"name": "one"}, {"name": "two"}]

    @pytest.mark.parametrize("prefix", ["Note [unfinished ", "Note {unfinished "])
    def test_complete_json_skips_unmatched_prefix(self, fake_client: Mock, prefix):
        fake_client.chat.completions.create.return_value = _chat_response(
            content=prefix + '{"value": "ok"}'
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")
        assert llm.complete_json(
            "Extract text memory", {"type": "object", "required": ["value"]}
        ) == {"value": "ok"}

    def test_json_recovery_does_not_extract_from_unfinished_reasoning(self):
        with pytest.raises(ValueError):
            openai_llm_module.OpenAILLM._extract_json_from_text(
                '<think>unfinished [{"value": "not a final answer"}'
            )

    def test_json_recovery_uses_final_answer_not_reasoning(self):
        assert openai_llm_module.OpenAILLM._extract_json_from_text(
            '<think>{"value": "scratch"}</think>'
            'Note [unfinished {"value": "final"}'
        ) == {"value": "final"}

    def test_failed_json_reports_truncation_without_logging_private_text(
        self, fake_client: Mock, caplog,
    ):
        response = _chat_response(
            content="<think>private conversation", finish_reason="length",
        )
        response.usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=4096,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=4096),
        )
        fake_client.chat.completions.create.return_value = response
        llm = openai_llm_module.OpenAILLM(api_key="test-key")
        with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
            llm.complete_json("Extract text memory", {"type": "object"})
        assert "finish_reason=length" in caplog.text
        assert "reasoning_tokens=4096" in caplog.text
        assert "visible_len=0" in caplog.text
        assert "private conversation" not in caplog.text

    @pytest.mark.parametrize(
        "content",
        [
            '{"text": "literal,}"}',
            '{"text": "literal,}",}',
        ],
    )
    def test_extract_json_from_text_preserves_comma_in_string(self, content):
        result = openai_llm_module.OpenAILLM._extract_json_from_text(content)

        assert result == {"text": "literal,}"}

    def test_complete_json_raises_for_invalid_json(self, fake_client: Mock):
        # Use truly unparsable JSON that can't be recovered by any fallback
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

    def test_waiting_required_calls_leave_probe_lock_after_capability_is_known(
        self,
    ):
        class TrackingLock:
            def __init__(self):
                self._lock = threading.Lock()
                self._guard = threading.Lock()
                self._attempts = 0
                self.second_attempted = threading.Event()

            def __enter__(self):
                with self._guard:
                    self._attempts += 1
                    if self._attempts == 2:
                        self.second_attempted.set()
                self._lock.acquire()
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                self._lock.release()

            def locked(self):
                return self._lock.locked()

        llm = openai_llm_module.OpenAILLM(api_key="test-key")
        probe_lock = TrackingLock()
        llm._required_tool_choice_lock = probe_lock
        first_started = threading.Event()
        release_first = threading.Event()
        invocation_lock_states = []
        invocation_guard = threading.Lock()

        def complete_impl(**_kwargs):
            with invocation_guard:
                invocation_index = len(invocation_lock_states)
                invocation_lock_states.append(probe_lock.locked())
            if invocation_index == 0:
                first_started.set()
                assert release_first.wait(timeout=1)
                llm._required_tool_choice_capability = "supported"
            return []

        llm._complete_with_tools_impl = Mock(side_effect=complete_impl)
        tools = [{
            "name": "extract_event",
            "description": "Extract event",
            "input_schema": {"type": "object"},
        }]
        errors = []

        def call_required():
            try:
                llm.complete_with_tools("prompt", tools, tool_choice="required")
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        first = threading.Thread(target=call_required)
        second = threading.Thread(target=call_required)
        first.start()
        assert first_started.wait(timeout=1)
        second.start()
        assert probe_lock.second_attempted.wait(timeout=1)
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert invocation_lock_states == [True, False]

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

        with pytest.raises(LLMCallError, match="Invalid tool schema"):
            llm.complete_with_tools("prompt", tools=[{
                "name": "extract_event",
                "description": "Extract event",
                "input_schema": {"type": "object"},
            }], tool_choice="required")

        assert fake_client.chat.completions.create.call_count == 1
        assert llm._required_tool_choice_capability == "unknown"

    def test_complete_with_tools_wraps_client_errors(self, fake_client: Mock):
        from core.errors import LLMCallError

        fake_client.chat.completions.create.side_effect = RuntimeError("tool boom")
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        # Tool-calling paths wrap runtime errors in LLMCallError so the
        # ReactLoop can detect them and finish with
        # finish_reason="llm_error" instead of silently swallowing.
        with pytest.raises(LLMCallError, match="OpenAI API error: tool boom"):
            llm.complete_with_tools("prompt", tools=[])

    def test_complete_with_tools_raises_llm_call_error_on_400(self, fake_client: Mock):
        """A 400 (BadRequestError) is surface as LLMCallError(400) so
        the driver can distinguish a 400 protocol mismatch from a 500
        upstream failure."""
        from openai import BadRequestError

        from core.errors import LLMCallError

        fake_response = Mock(status_code=400)
        fake_client.chat.completions.create.side_effect = BadRequestError(
            message="tool schema rejected",
            response=fake_response,
            body={"error": {"message": "no such tool"}},
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(LLMCallError) as exc_info:
            llm.complete_with_tools(
                "prompt",
                tools=[{"name": "missing", "description": "x",
                        "input_schema": {"type": "object"}}],
            )
        assert exc_info.value.status_code == 400
        assert "BadRequestError" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, BadRequestError)

    def test_complete_with_tools_messages_raises_llm_call_error_on_400(
        self, fake_client: Mock,
    ):
        """The messages-based tool path must surface the same typed
        signal — this is the path the ReactLoop drives."""
        from openai import BadRequestError

        from core.errors import LLMCallError

        fake_response = Mock(status_code=400)
        fake_client.chat.completions.create.side_effect = BadRequestError(
            message="tool schema rejected",
            response=fake_response,
            body={"error": {"message": "no such tool"}},
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(LLMCallError) as exc_info:
            llm.complete_with_tools_messages(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "missing", "description": "x",
                        "input_schema": {"type": "object"}}],
            )
        assert exc_info.value.status_code == 400
        assert "BadRequestError" in str(exc_info.value)

    def test_complete_with_tools_messages_raises_llm_call_error_on_500(
        self, fake_client: Mock,
    ):
        """A 500 (APIStatusError) surfaces as LLMCallError with
        status_code=500 so the driver can react."""
        from openai import APIStatusError

        from core.errors import LLMCallError

        fake_response = Mock(status_code=500)
        fake_client.chat.completions.create.side_effect = APIStatusError(
            message="upstream down",
            response=fake_response,
            body={"error": {"message": "boom"}},
        )
        llm = openai_llm_module.OpenAILLM(api_key="test-key")

        with pytest.raises(LLMCallError) as exc_info:
            llm.complete_with_tools_messages(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "x", "description": "x",
                        "input_schema": {"type": "object"}}],
            )
        assert exc_info.value.status_code == 500

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
            # A successful HTTP response without the required tool call is
            # a protocol failure, not a transport failure.
            from core.errors import ToolCallProtocolError
            with pytest.raises(ToolCallProtocolError, match="Required tool call returned no tool_calls"):
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


# ---------------------------------------------------------------------------
# complete_multimodal
# ---------------------------------------------------------------------------

class TestOpenAILLMCompleteMultimodal:
    def test_complete_multimodal_with_json_mode(self, fake_client, fake_openai_ctor):
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key", model="gpt-4o", json_mode=True,
        )
        fake_client.chat.completions.create.return_value = _chat_response(
            content='{"description": "robot picks cup", "characters": [{"name": "robot_arm"}]}',
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                {"type": "input_audio", "input_audio": {"data": "BBB", "format": "wav"}},
            ],
        }]
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "characters": {"type": "array"},
            },
        }
        result = llm.complete_multimodal(messages, schema)

        assert result["description"] == "robot picks cup"
        # The OpenAI client received the multimodal messages + json_object.
        call = fake_client.chat.completions.create.call_args
        sent_messages = call.kwargs["messages"]
        # System message is schema-driven; user message is verbatim.
        assert sent_messages[0]["role"] == "system"
        assert "description" in sent_messages[0]["content"]  # schema embedded
        assert sent_messages[1] == messages[0]  # user multimodal content forwarded
        assert call.kwargs["response_format"] == {"type": "json_object"}

    def test_complete_multimodal_drops_caller_system_messages(self, fake_client, fake_openai_ctor):
        """Caller-supplied system messages are dropped — the schema-derived
        system message is authoritative for JSON-mode guarantees."""
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key", model="gpt-4o", json_mode=True,
        )
        fake_client.chat.completions.create.return_value = _chat_response(
            content='{"description": "ok"}',
        )
        messages = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "describe"},
        ]
        llm.complete_multimodal(messages, {"type": "object"})
        sent = fake_client.chat.completions.create.call_args.kwargs["messages"]
        # Only one system message (the schema-driven one), no caller system.
        assert all(m["role"] != "system" or "JSON" not in m["content"]
                   for m in sent if m.get("content") == "ignore me")

    def test_complete_multimodal_text_fallback_on_bad_request(
        self, fake_client, fake_openai_ctor,
    ):
        from openai import BadRequestError
        llm = openai_llm_module.OpenAILLM(
            api_key="test-key", model="gpt-4o", json_mode=True,
        )
        # First call: json_object rejected. Second call (text fallback):
        # JSON-extracted from free text. BadRequestError.__init__ accesses
        # several httpx-style attributes on response; using Mock() with
        # spec avoids the SimpleNamespace attribute tax.
        from unittest.mock import Mock
        fake_response = Mock(status_code=400)
        fake_client.chat.completions.create.side_effect = [
            BadRequestError(
                message="json_object not supported",
                response=fake_response,
                body={"error": {"message": "no json"}},
            ),
            _chat_response(content='{"description": "fallback ok"}'),
        ]
        result = llm.complete_multimodal(
            [{"role": "user", "content": "describe"}],
            {"type": "object", "properties": {"description": {"type": "string"}}},
        )
        assert result["description"] == "fallback ok"
        # Two calls: structured + text fallback.
        assert fake_client.chat.completions.create.call_count == 2
        # Second call has no response_format (text mode).
        second_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" not in second_kwargs
