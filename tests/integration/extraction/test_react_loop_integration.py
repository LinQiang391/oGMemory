# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Integration test for ExtractionReActLoop with real SchemaRegistry and URIResolver.

Uses actual YAML schemas and parse_tool_call — only mocks LLM and ContextFS.
"""

import json
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.react_loop import ExtractionReActLoop
from extraction.schemas.registry import SchemaRegistry


# ---------------------------------------------------------------------------
# Real dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return SchemaRegistry()  # loads from extraction/schemas/definitions/


@pytest.fixture
def uri_resolver(registry):
    return URIResolver(registry)


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acme", user_id="alice",
        agent_id="assistant", session_id="s1", trace_id="t1",
    )


# ---------------------------------------------------------------------------
# Mock external dependencies (LLM, filesystem)
# ---------------------------------------------------------------------------

def _make_llm():
    llm = Mock()
    llm._queue = []

    def _call(messages, tools=None, tool_choice="auto"):
        return llm._queue.pop(0) if llm._queue else ([], "")

    llm.complete_with_tools_messages = _call
    return llm


def _make_fs(existing_nodes=None):
    """Mock ContextFS with pre-populated nodes."""
    existing = existing_nodes or {}

    fs = Mock()

    def _read_node(uri, ctx):
        if uri in existing:
            return existing[uri]
        raise FileNotFoundError(uri)

    fs.read_node = Mock(side_effect=_read_node)
    fs.list_children = Mock(return_value=list(existing.keys()))
    fs.exists = Mock(side_effect=lambda uri, ctx: uri in existing)
    return fs


def _node(uri, content="existing content", abstract="existing abstract",
          overview="existing overview", metadata=None):
    return ContextNode(
        uri=uri, context_type="MEMORY", category="preference",
        level=0, owner_space="user:alice",
        abstract=abstract, overview=overview, content=content,
        metadata=metadata or {},
    )


def _preference_input(topic, abstract, overview, content, confidence=0.9):
    """Valid input for extract_preference tool."""
    return {
        "topic": topic,
        "abstract": abstract,
        "overview": overview,
        "content": content,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Integration test: realistic multi-turn extraction
# ---------------------------------------------------------------------------

class TestReActLoopIntegration:
    """End-to-end ReAct loop with real schema parsing and URI resolution."""

    def test_read_existing_then_extract_preference(self, registry, uri_resolver, ctx):
        """LLM reads existing coding_style preference → decides to update it.

        Flow:
        1. LLM lists preferences directory
        2. LLM reads coding_style.md (existing: "likes Java")
        3. LLM extracts updated preference ("switched to Python")
        """
        llm = _make_llm()
        pref_uri = "ctx://acme/users/alice/memories/preferences/coding_style"
        fs = _make_fs(existing_nodes={
            pref_uri: _node(pref_uri, content="User prefers Java for backend",
                            abstract="Java backend developer",
                            overview="Prefers Java"),
        })

        # Iteration 1: LLM lists the preferences directory
        llm._queue.append(([
            {"name": "list",
             "input": {"uri": "ctx://acme/users/alice/memories/preferences"},
             "id": "tc-1"},
        ], ""))

        # Iteration 2: LLM reads the existing coding_style preference
        llm._queue.append(([
            {"name": "read",
             "input": {"uri": pref_uri},
             "id": "tc-2"},
        ], ""))

        # Iteration 3: LLM returns the updated extraction
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                topic="coding_style",
                abstract="Switched from Java to Python",
                overview="Now prefers Python for backend development",
                content="User switched from Java to Python for backend. Likes type hints and fastapi.",
                confidence=0.95,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm, fs=fs, registry=registry, uri_resolver=uri_resolver,
            max_iterations=5,
        )
        result = loop.run(
            "User: I've been using Java for years but recently switched to Python. "
            "The type hints are great and I love fastapi.",
            ctx,
        )

        # Verify candidates
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.category == "preference"
        assert c.routing_key == "coding_style"
        assert "Python" in c.content
        assert c.confidence == 0.95

        # Verify loop execution
        assert result.iterations == 3
        assert len(result.tools_used) == 2  # list + read
        assert result.tools_used[0]["tool_name"] == "list"
        assert result.tools_used[1]["tool_name"] == "read"

        # Verify read_uris tracking
        assert pref_uri in result.read_uris

        # Verify trace data
        trace = result.trace
        assert trace is not None
        assert trace.total_iterations == 3
        assert trace.final_candidate_count == 1
        assert trace.total_read_uris == 1
        assert len(trace.iterations) == 3

        # Verify iteration-level trace
        iter0 = trace.iterations[0]
        assert iter0.tool_calls_count == 1
        assert iter0.tool_call_names == ["list"]

        iter1 = trace.iterations[1]
        assert iter1.tool_call_names == ["read"]

        iter2 = trace.iterations[2]
        assert iter2.tool_calls_count == 0  # content returned, no tool calls
        assert iter2.content_length > 0

    def test_safety_refetch_with_real_schema(self, registry, uri_resolver, ctx):
        """LLM extracts without reading existing file → refetch triggered.

        Verifies the refetch mechanism works with real URI resolution:
        preference URI resolved via Jinja2 template from YAML schema.
        """
        llm = _make_llm()
        pref_uri = "ctx://acme/users/alice/memories/preferences/coffee"
        fs = _make_fs(existing_nodes={
            pref_uri: _node(pref_uri, content="User drinks black coffee",
                            abstract="Black coffee drinker"),
        })

        # Iteration 1: LLM extracts directly WITHOUT reading first
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                topic="coffee",
                abstract="Now prefers latte",
                overview="Switched from black coffee to latte",
                content="User now prefers latte over black coffee",
                confidence=0.85,
            ),
        }])))

        # Iteration 2: After refetch, LLM returns merged result
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                topic="coffee",
                abstract="Prefers latte (was black coffee)",
                overview="Evolved from black coffee to latte",
                content="User used to drink black coffee, now prefers latte",
                confidence=0.9,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm, fs=fs, registry=registry, uri_resolver=uri_resolver,
            max_iterations=5,
        )
        result = loop.run("User: I switched to lattes recently", ctx)

        # Refetch should have triggered
        assert any(it.safety_check_triggered for it in result.trace.iterations)
        assert len(result.candidates) == 1
        assert result.candidates[0].routing_key == "coffee"
        assert result.iterations == 2

        # Verify the prefetched URI is now in read_uris
        assert pref_uri in result.read_uris

    def test_extract_new_preference_no_existing(self, registry, uri_resolver, ctx):
        """No existing file → no refetch → direct extraction.

        Verifies URI resolution produces correct URI for new preference.
        """
        llm = _make_llm()
        fs = _make_fs()  # no existing nodes

        # Single iteration: LLM extracts directly
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                topic="music",
                abstract="Enjoys jazz music",
                overview="Likes jazz and blues",
                content="User enjoys listening to jazz and blues in their free time",
                confidence=0.8,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm, fs=fs, registry=registry, uri_resolver=uri_resolver,
        )
        result = loop.run("User: I love jazz and blues", ctx)

        assert len(result.candidates) == 1
        assert result.candidates[0].category == "preference"
        assert result.candidates[0].routing_key == "music"
        assert result.iterations == 1

        # No safety check should have triggered (file doesn't exist)
        assert all(not it.safety_check_triggered for it in result.trace.iterations)
