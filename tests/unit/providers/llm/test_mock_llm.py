"""Tests for the mock LLM provider."""

from __future__ import annotations

import pytest

from providers.llm.mock_llm import MockLLM


def test_custom_fixtures_take_priority():
    llm = MockLLM(fixtures={"coffee": {"custom": True}})

    result = llm.complete_json("Coffee preference", {"type": "object"})

    assert result == {"custom": True}
    assert llm.call_count == 1


@pytest.mark.parametrize(
    ("prompt", "expected_category", "expected_scope"),
    [
        ("profile details", "profile", "user"),
        ("preference details", "preference", "user"),
        ("entity details", "entity", "user"),
        ("event details", "event", "user"),
        ("pattern details", "pattern", "agent"),
        ("skill details", "skill", "agent"),
    ],
)
def test_builtin_prompt_routing_returns_expected_fixture(prompt, expected_category, expected_scope):
    llm = MockLLM()

    result = llm.complete_json(prompt, {"type": "object"})

    assert result["candidates"][0]["category"] == expected_category
    assert result["candidates"][0]["owner_scope"] == expected_scope


def test_unknown_prompt_returns_empty_candidates():
    llm = MockLLM()

    assert llm.complete_json("no known keywords", {"type": "object"}) == {"candidates": []}


def test_complete_with_tools_returns_mock_calls_and_tracks_prompt():
    llm = MockLLM()
    llm._mock_tool_calls = [{"tool": "search", "input": {"query": "coffee"}}]

    result = llm.complete_with_tools("Find coffee", tools=[{"name": "search"}])

    assert result == [{"tool": "search", "input": {"query": "coffee"}}]
    assert llm.call_count == 1
    assert llm._last_prompt == "Find coffee"


def test_reset_clears_call_count():
    llm = MockLLM()
    llm.complete_json("profile", {"type": "object"})

    llm.reset()

    assert llm.call_count == 0
