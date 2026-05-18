"""Unit tests for ExtractionReActLoop — core loop logic only.

Tests the loop control flow, not the individual tool implementations
or JSON parsing (those belong to their own modules).
"""

import json
from unittest.mock import Mock, patch

import pytest

from core.models import CandidateMemory, ContextNode, RequestContext
from extraction.prefetch import PrefetchResult
from extraction.react_loop import ExtractionReActLoop, ReActResult, ReActTrace

# ---------------------------------------------------------------------------
# Patch parse_tool_call — we're testing loop logic, not schema parsing
# ---------------------------------------------------------------------------

_PARSE_PATCH = patch(
    "extraction.react_loop.parse_tool_call",
    side_effect=lambda name, inp, _reg: (
        (cat, inp.get("owner_scope", "user"), CandidateMemory(
            category=cat,
            owner_scope=inp.get("owner_scope", "user"),
            routing_key=inp.get("routing_key", "k"),
            abstract=inp.get("abstract", ""),
            overview=inp.get("overview", ""),
            content=inp.get("content", ""),
            confidence=inp.get("confidence", 0.9),
        )) if (name.startswith("extract_")) and (cat := name.replace("extract_", ""))
        else None
    ),
)


@pytest.fixture(autouse=True)
def _mock_parse():
    with _PARSE_PATCH:
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx():
    return RequestContext(account_id="acme", user_id="alice",
                         agent_id="bob", session_id="s1", trace_id="t1")


def _node(content="hello", metadata=None):
    return ContextNode(uri="ctx://acme/x", context_type="MEMORY",
                       category="profile", level=0, owner_space="user:alice",
                       abstract="a", overview="o", content=content,
                       metadata=metadata or {})


def _extract_input(routing_key="k", **kw):
    """Shorthand for extract_profile tool input."""
    return {"routing_key": routing_key, "abstract": "a", "overview": "o",
            "content": "c", "confidence": 0.9, "owner_scope": "user", **kw}


def _extract_json(**kw):
    """JSON string for a single extract_profile operation."""
    return json.dumps([{"name": "extract_profile", "input": _extract_input(**kw)}])


def _make_llm():
    llm = Mock()
    llm._queue = []
    def _call(messages, tools=None, tool_choice="auto"):
        return llm._queue.pop(0) if llm._queue else ([], "")
    llm.complete_with_tools_messages = _call
    return llm


def _make_fs(exists=False, content="hello"):
    fs = Mock()
    fs.read_node = Mock(return_value=_node(content=content))
    fs.list_children = Mock(return_value=[])
    fs.exists = Mock(return_value=exists)
    return fs


def _make_registry(is_add_only=False):
    schema = Mock(is_add_only=is_add_only)
    reg = Mock()
    reg.get = Mock(return_value=schema)
    reg.list_enabled = Mock(return_value=[])
    return reg


def _make_uri_resolver():
    r = Mock()
    r.resolve = Mock(return_value="ctx://acme/test/uri")
    r.validate_uri = Mock(return_value=True)
    return r


def _make_loop(llm=None, fs=None, **kw):
    return ExtractionReActLoop(
        llm=llm or _make_llm(),
        fs=fs or _make_fs(),
        registry=kw.pop("registry", _make_registry()),
        uri_resolver=kw.pop("uri_resolver", _make_uri_resolver()),
        max_iterations=kw.get("max_iterations", 3),
        timeout_seconds=kw.get("timeout_seconds", 30.0),
    )


# ---------------------------------------------------------------------------
# Tests: Loop control flow
# ---------------------------------------------------------------------------

class TestLoopControl:
    """Core iteration, termination, and state management."""

    def test_tool_call_then_content(self):
        """LLM calls read → gets result → outputs extraction in next iteration."""
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://acme/p"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json(routing_key="java-dev")))

        result = _make_loop(llm=llm).run("I switched to Java", _ctx())

        assert len(result.candidates) == 1
        assert result.candidates[0].routing_key == "java-dev"
        assert result.iterations == 2
        assert "ctx://acme/p" in result.read_uris

    def test_max_iterations_exhausted(self):
        """Loop keeps getting tool calls → exhausts max_iterations → empty."""
        llm = _make_llm()
        for _ in range(5):
            llm._queue.append(([{"name": "read", "input": {"uri": "ctx://x"}, "id": "1"}], ""))

        result = _make_loop(llm=llm, max_iterations=3).run("conv", _ctx())
        assert result.candidates == []
        assert result.iterations >= 3

    def test_state_reset_between_runs(self):
        """run() resets read_files so second run doesn't leak first run's state."""
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://a"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))
        llm._queue.append(([], _extract_json(routing_key="k2")))

        loop = _make_loop(llm=llm)
        r1 = loop.run("first", _ctx())
        assert "ctx://a" in r1.read_uris

        r2 = loop.run("second", _ctx())
        assert "ctx://a" not in r2.read_uris

    def test_prefetch_uris_included(self):
        """PrefetchResult.read_uris are included in final read_uris."""
        llm = _make_llm()
        llm._queue.append(([], _extract_json()))
        prefetch = PrefetchResult(messages=["ctx"], read_uris={"ctx://acme/p"}, listed_uris=set())

        result = _make_loop(llm=llm).run("conv", _ctx(), prefetch)
        assert "ctx://acme/p" in result.read_uris

    def test_content_present_but_parse_fails(self):
        """LLM returns non-JSON content → _parse_operations returns None → falls to Case 3 → disables tools."""
        llm = _make_llm()
        seen_choices = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            seen_choices.append(tool_choice)
            if n == 1:
                return [], "I think the user likes Python"  # content but not JSON
            return [], _extract_json()

        llm.complete_with_tools_messages = _track
        result = _make_loop(llm=llm, max_iterations=5).run("conv", _ctx())

        # First iteration: content present but unparseable → tools disabled for next
        # Second iteration: tools off, LLM returns valid extraction
        assert len(result.candidates) == 1
        assert "none" in seen_choices


# ---------------------------------------------------------------------------
# Tests: Tool disable mechanism
# ---------------------------------------------------------------------------

class TestToolDisable:
    """_disable_tools_for_iteration flips tool_choice to 'none'."""

    def test_unknown_tool_disables_next(self):
        llm = _make_llm()
        seen = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            seen.append(tool_choice)
            if n == 1:
                return [{"name": "bogus", "input": {}, "id": "1"}], ""
            return [], _extract_json()

        llm.complete_with_tools_messages = _track
        _make_loop(llm=llm).run("conv", _ctx())
        assert seen == ["auto", "none"]

    def test_empty_response_disables_next(self):
        llm = _make_llm()
        seen = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            seen.append(tool_choice)
            if n <= 2:
                return ([], "")  # neither tools nor content
            return [], _extract_json()

        llm.complete_with_tools_messages = _track
        _make_loop(llm=llm, max_iterations=5).run("conv", _ctx())
        assert "none" in seen


# ---------------------------------------------------------------------------
# Tests: Safety refetch
# ---------------------------------------------------------------------------

class TestSafetyRefetch:
    """_check_unread_existing_files + _did_safety_reread guard."""

    def test_refetch_triggered_for_unread_existing(self):
        """Candidate targets existing unread file → auto-refetch → extra iteration."""
        llm = _make_llm()
        fs = _make_fs(exists=True, content="old data")
        llm._queue.append(([], _extract_json()))          # targets unread existing
        llm._queue.append(([], _extract_json(routing_key="k2")))  # after refetch

        result = _make_loop(llm=llm, fs=fs).run("conv", _ctx())
        assert any(it.safety_check_triggered for it in result.trace.iterations)

    def test_refetch_only_once(self):
        """_did_safety_reread prevents refetch from firing repeatedly."""
        llm = _make_llm()
        fs = _make_fs(exists=True)
        for _ in range(4):
            llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs, max_iterations=5).run("conv", _ctx())
        safety_count = sum(1 for it in result.trace.iterations if it.safety_check_triggered)
        assert safety_count <= 1

    def test_add_only_schema_skips_refetch(self):
        """is_add_only=True means no conflict risk → no refetch."""
        llm = _make_llm()
        fs = _make_fs(exists=True)  # exists but shouldn't matter
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs, registry=_make_registry(is_add_only=True)).run("conv", _ctx())
        assert all(not it.safety_check_triggered for it in result.trace.iterations)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_unknown_tool_returns_error(self):
        """Unknown tool → error dict, loop continues."""
        llm = _make_llm()
        llm._queue.append(([{"name": "fly", "input": {}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm).run("conv", _ctx())
        assert result.tools_used[0]["result"]["error"] == "Unknown tool: fly"

    def test_tool_exception_continues_loop(self):
        """Tool throws → error dict, loop doesn't crash."""
        llm = _make_llm()
        fs = _make_fs()
        fs.read_node.side_effect = RuntimeError("boom")
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://x"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs).run("conv", _ctx())
        assert "error" in result.tools_used[0]["result"]
        assert len(result.candidates) == 1

    def test_llm_exception_returns_empty(self):
        """LLM throws on first call → empty result, no crash."""
        llm = _make_llm()
        llm.complete_with_tools_messages = Mock(side_effect=RuntimeError("API down"))
        result = _make_loop(llm=llm).run("conv", _ctx())
        assert result.candidates == []
