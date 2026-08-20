"""Unit tests for ReactLoop base class.

Tests the generic loop logic using a mock subclass.
"""

import json
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from core.react_loop import (
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
    ReactIterationTrace,
    ReactTrace,
)


class MockReactLoop(ReactLoop):
    """Mock subclass for testing base class logic."""

    def __init__(self, llm, **kw):
        super().__init__(llm, **kw)
        self._tools = [{"name": "mock_tool", "description": "A mock tool", "input_schema": {"type": "object"}}]
        self._execute_results = {}
        self._parse_result = None
        self._should_continue = False

    def _get_tools(self) -> list[dict]:
        return self._tools

    def _build_messages(self, ctx, task_ctx) -> list[dict]:
        return [{"role": "system", "content": "mock prompt"}]

    def _execute_tool(self, name, input, ctx) -> dict | str:
        key = f"{name}:{json.dumps(input, sort_keys=True)}"
        return self._execute_results.get(key, {"result": "ok"})

    def _parse_output(self, content: str) -> list | None:
        return self._parse_result

    def _should_continue_after_output(self, parsed, messages, ctx, iter_trace=None) -> bool:
        return self._should_continue


def _ctx():
    return RequestContext(account_id="acme", user_id="alice",
                          agent_id="bob", session_id="s1", trace_id="t1")


def _task_ctx():
    return ReactTaskContext()


def _make_llm():
    llm = Mock()
    llm._queue = []

    def _call(messages, tools=None, tool_choice="auto"):
        return llm._queue.pop(0) if llm._queue else ([], "")

    llm.complete_with_tools_messages = _call
    return llm


class TestReactTaskContext:

    def test_empty_context(self):
        ctx = ReactTaskContext()
        assert ctx is not None

    def test_extensible_via_subclass(self):
        @dataclass
        class MyTaskContext(ReactTaskContext):
            custom_field: str = "value"

        ctx = MyTaskContext()
        assert ctx.custom_field == "value"


class TestReactIterationTrace:

    def test_default_values(self):
        trace = ReactIterationTrace(iteration=1, latency_seconds=0.5)
        assert trace.tool_calls_count == 0
        assert trace.tool_call_names == []
        assert trace.content_length == 0

    def test_with_tool_calls(self):
        trace = ReactIterationTrace(
            iteration=2,
            latency_seconds=1.0,
            tool_calls_count=3,
            tool_call_names=["read", "list", "extract"],
            content_length=0,
        )
        assert trace.tool_calls_count == 3
        assert "read" in trace.tool_call_names


class TestReactTrace:

    def test_aggregates_iterations(self):
        it1 = ReactIterationTrace(iteration=1, latency_seconds=0.5)
        it2 = ReactIterationTrace(iteration=2, latency_seconds=1.0)
        trace = ReactTrace(
            total_iterations=2,
            total_latency_seconds=1.5,
            iterations=[it1, it2],
            total_read_uris=5,
        )
        assert trace.total_iterations == 2
        assert len(trace.iterations) == 2
        assert trace.total_read_uris == 5


class TestReactLoopResult:

    def test_completed_flag(self):
        result = ReactLoopResult(
            output=[{"key": "value"}],
            tools_used=[],
            iterations=2,
            read_uris=set(),
            trace=None,
            completed=True,
            finish_reason="done",
        )
        assert result.completed
        assert result.finish_reason == "done"

    def test_timeout_flag(self):
        result = ReactLoopResult(
            output=[],
            iterations=3,
            completed=False,
            finish_reason="timeout",
        )
        assert not result.completed
        assert result.finish_reason == "timeout"


class TestLoopControl:

    def test_tool_call_then_output(self):
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "x"}, "id": "1"}], ""))
        llm._queue.append(([], json.dumps([{"key": "val"}])))

        loop = MockReactLoop(llm, max_iterations=3)
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())

        assert result.output == [{"key": "val"}]
        assert result.iterations == 2
        assert result.completed
        assert result.finish_reason == "done"

    def test_max_iterations_exhausted(self):
        llm = _make_llm()
        for _ in range(5):
            llm._queue.append(([{"name": "read", "input": {"uri": "x"}, "id": "1"}], ""))

        loop = MockReactLoop(llm, max_iterations=3)
        result = loop.run(_ctx(), _task_ctx())

        assert result.output == []
        assert result.iterations >= 3
        assert not result.completed
        assert result.finish_reason == "max_iterations"

    def test_empty_response_disables_tools(self):
        llm = _make_llm()
        seen_choices = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            seen_choices.append(tool_choice)
            if n == 1:
                return ([], "")  # neither tools nor content
            return ([], json.dumps([{"key": "val"}]))

        llm.complete_with_tools_messages = _track
        loop = MockReactLoop(llm, max_iterations=5)
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())

        assert "none" in seen_choices
        assert result.completed

    def test_content_but_parse_fails(self):
        llm = _make_llm()
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            if n == 1:
                return ([], "unparseable content")
            return ([], json.dumps([{"key": "val"}]))

        llm.complete_with_tools_messages = _track
        loop = MockReactLoop(llm, max_iterations=5)

        def _parse_side_effect(content):
            if "unparseable" in content:
                return None
            return [{"key": "val"}]

        loop._parse_output = _parse_side_effect

        result = loop.run(_ctx(), _task_ctx())

        assert result.iterations >= 2


class TestShouldContinueAfterOutput:

    def test_default_returns_false(self):
        llm = _make_llm()
        llm._queue.append(([], json.dumps([{"key": "val"}])))

        loop = MockReactLoop(llm)
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())
        assert result.completed
        assert result.iterations == 1

    def test_override_continues_loop(self):
        llm = _make_llm()
        llm._queue.append(([], json.dumps([{"key": "val"}])))  # first output
        llm._queue.append(([], json.dumps([{"key": "val2"}])))  # second output

        loop = MockReactLoop(llm, max_iterations=5)
        loop._parse_result = [{"key": "val"}]

        def _parse_dynamic(content):
            if "val" in content and "val2" not in content:
                return [{"key": "val"}]
            return [{"key": "val2"}]

        loop._parse_output = _parse_dynamic
        loop._should_continue_after_output = lambda p, m, c, it=None: (
            p[0]["key"] == "val" if p else False
        )

        result = loop.run(_ctx(), _task_ctx())
        assert result.iterations >= 2


class TestToolExecution:

    def test_tool_result_appended_to_messages(self):
        llm = _make_llm()
        llm._queue.append(([{"name": "mock_tool", "input": {"x": 1}, "id": "t1"}], ""))
        llm._queue.append(([], json.dumps([{"key": "val"}])))

        loop = MockReactLoop(llm)
        loop._execute_results["mock_tool:{\"x\": 1}"] = {"data": "result"}
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())

        assert len(result.tools_used) == 1
        assert result.tools_used[0]["tool_name"] == "mock_tool"

    def test_unknown_tool_handling(self):
        llm = _make_llm()
        llm._queue.append(([{"name": "unknown", "input": {}, "id": "1"}], ""))
        llm._queue.append(([], json.dumps([{"key": "val"}])))

        loop = MockReactLoop(llm)
        loop._execute_results["unknown:{}"] = {"error": "Unknown tool: unknown"}
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())

        assert result.tools_used[0]["result"]["error"].startswith("Unknown tool")
        assert loop._disable_tools_for_iteration


class TestTimeout:

    def test_timeout_triggers_graceful_exit(self):
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "x"}, "id": "1"}], ""))
        llm._queue.append(([{"name": "read", "input": {"uri": "y"}, "id": "2"}], ""))

        loop = MockReactLoop(llm, max_iterations=10, timeout_seconds=0.01)
        result = loop.run(_ctx(), _task_ctx())

        assert result.finish_reason in ("timeout", "max_iterations")
        assert not result.completed


class TestLLMException:

    def test_llm_exception_returns_empty(self):
        """LLM throws on first call → empty result, no crash.

        The base loop catches :class:`LLMCallError` raised from
        :meth:`_call_llm` and finishes with
        ``finish_reason="llm_error"`` instead of falling through to
        ``_on_invalid_response``'s "disable tools and continue" path.
        The previous silent-swallow contract (``return None, None``)
        was responsible for turning hard server errors into
        ``[Search] … [Answer] …`` protocol garbage.
        """
        from core.errors import LLMCallError

        llm = Mock()
        llm.complete_with_tools_messages = Mock(side_effect=RuntimeError("API down"))
        result = _make_loop(llm=llm).run("conv", _ctx())
        assert result.candidates == []
        assert result.finish_reason == "llm_error"
        assert result.completed is False

    def test_call_llm_wraps_unknown_transport_error(self):
        """``_call_llm`` wraps ``RuntimeError`` from the LLM in
        :class:`LLMCallError` so the driver has a typed signal.

        Previously the base method returned ``(None, None)`` and the
        loop fell through to "no tools, continue"; that turned 400
        errors into ``[Search] … [Answer] …`` protocol garbage.
        """
        from core.errors import LLMCallError

        llm = Mock()
        llm.complete_with_tools_messages = Mock(side_effect=RuntimeError("API down"))
        loop = _make_loop(llm=llm)
        with pytest.raises(LLMCallError) as exc_info:
            loop._call_llm([{"role": "user", "content": "hi"}])
        assert "API down" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, RuntimeError)

    def test_call_llm_propagates_already_classified_error(self):
        """If the LLM provider already raises ``LLMCallError`` (e.g.
        ``OpenAILLM`` on 400), ``_call_llm`` re-raises it unchanged so
        the status code is preserved end-to-end."""
        from core.errors import LLMCallError

        original = LLMCallError("BadRequestError (400)", status_code=400)
        llm = Mock()
        llm.complete_with_tools_messages = Mock(side_effect=original)
        loop = _make_loop(llm=llm)
        with pytest.raises(LLMCallError) as exc_info:
            loop._call_llm([{"role": "user", "content": "hi"}])
        assert exc_info.value is original
        assert exc_info.value.status_code == 400

    def test_run_does_not_retry_with_tools_disabled_after_llm_error(self):
        """After a transport failure the loop must NOT fall through to
        ``_on_invalid_response``'s "disable tools and continue" path.

        We assert by counting LLM calls: with the buggy silent-swallow
        behaviour the loop would have retried with tools disabled; the
        hardened contract terminates after the first failure.
        """
        llm = Mock()
        llm.complete_with_tools_messages = Mock(
            side_effect=RuntimeError("500 Internal Server Error"),
        )
        loop = _make_loop(llm=llm, max_iterations=5)
        result = loop.run("conv", _ctx())
        assert result.finish_reason == "llm_error"
        assert result.iterations == 1
        assert llm.complete_with_tools_messages.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Timeout
# ---------------------------------------------------------------------------

class TestTimeout:

    def test_last_iteration_adds_message(self):
        llm = _make_llm()
        captured_messages = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            captured_messages.append(list(messages))
            if n < 3:
                return ([{"name": "read", "input": {"uri": "x"}, "id": str(n)}], "")
            return ([], json.dumps([{"key": "val"}]))

        llm.complete_with_tools_messages = _track
        loop = MockReactLoop(llm, max_iterations=3)
        loop._parse_result = [{"key": "val"}]

        result = loop.run(_ctx(), _task_ctx())

        last_iteration_messages = captured_messages[-1]
        termination_msg = None
        for msg in last_iteration_messages:
            if msg.get("role") == "user" and "maximum number" in msg.get("content", ""):
                termination_msg = msg
                break

        assert termination_msg is not None
        assert "maximum number" in termination_msg.get("content", "")
