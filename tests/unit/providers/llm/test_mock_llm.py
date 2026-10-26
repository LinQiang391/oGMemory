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


# ---------------------------------------------------------------------------
# complete_multimodal
# ---------------------------------------------------------------------------

class TestCompleteMultimodal:
    def test_returns_canned_response_when_set(self):
        llm = MockLLM()
        llm._mock_multimodal_response = {
            "description": "A robot picks up a red cup.",
            "characters": [{"name": "robot_arm"}],
        }
        schema = {"type": "object", "properties": {"description": {"type": "string"}}}
        result = llm.complete_multimodal(
            [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]}],
            schema,
        )
        assert result["description"] == "A robot picks up a red cup."
        assert result["characters"] == [{"name": "robot_arm"}]

    def test_tracks_last_multimodal_messages(self):
        llm = MockLLM()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        llm.complete_multimodal(msgs, {"type": "object"})
        assert llm._last_multimodal_messages == msgs

    def test_deterministic_response_when_no_canned(self):
        llm = MockLLM()
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "characters": {"type": "array"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
            },
        }
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "describe the clip"}],
        }]
        result = llm.complete_multimodal(messages, schema)
        # String fields get the first text content as a hint.
        assert result["description"] == "describe the clip"
        # Other types get typed defaults.
        assert result["characters"] == []
        assert result["score"] == 0
        assert result["active"] is True

    def test_string_field_picks_first_text_content(self):
        llm = MockLLM()
        messages = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe this clip"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]},
        ]
        result = llm.complete_multimodal(
            messages,
            {"type": "object", "properties": {"description": {"type": "string"}}},
        )
        assert result["description"] == "describe this clip"

    def test_increments_call_count(self):
        llm = MockLLM()
        llm.complete_multimodal(
            [{"role": "user", "content": "x"}], {"type": "object"},
        )
        assert llm.call_count == 1
