"""Unit tests for Extractor tool-use implementation.

Tests verify:
- Empty results when no tool calls
- Single preference extraction
- Multiple types extraction
- Skill extraction with agent scope
- Pattern extraction with agent scope
- Low confidence filtering
- Tool name to category mapping
- Profile routing key defaults to category
"""

from unittest.mock import Mock

import pytest

from core.models import RequestContext, CandidateMemory
from extraction.tools import Extractor, EXTRACTION_TOOLS, TOOL_TO_CATEGORY
from providers.llm.mock_llm import MockLLM


class TestExtractorEmptyResults:
    """Tests for Extractor with no tool calls."""

    def test_extract_empty_when_no_tool_calls(self):
        """LLM returns no tool calls → empty result."""
        llm = MockLLM()
        llm._mock_tool_calls = []

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = extractor.extract(messages, ctx)

        assert result == []

    def test_extract_filters_low_confidence(self):
        """Candidates with confidence < 0.5 are filtered out."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.3,  # Below threshold
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee"}]
        result = extractor.extract(messages, ctx)

        assert result == []


class TestExtractorSinglePreference:
    """Tests for single preference extraction."""

    def test_extract_single_preference(self):
        """LLM returns extract_preference → CandidateMemory with category=preference."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "User's coffee preferences",
                    "overview": "Likes dark roast, drinks 2-3 cups daily",
                    "content": "Prefers dark roast coffee, drinks 2-3 cups per day",
                    "confidence": 0.85,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "preference"
        assert candidate.owner_scope == "user"
        assert candidate.routing_key == "coffee"
        assert candidate.confidence == 0.85


class TestExtractorMultipleTypes:
    """Tests for extracting multiple candidate types."""

    def test_extract_multiple_types(self):
        """LLM returns preference + entity → 2 candidates with correct owner_scope."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                },
            },
            {
                "tool": "extract_entity",
                "input": {
                    "routing_key": "alice",
                    "abstract": "Person named Alice",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.8,
                },
            },
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee with Alice"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 2

        # Check preference
        preference = [c for c in result if c.category == "preference"][0]
        assert preference.owner_scope == "user"
        assert preference.routing_key == "coffee"

        # Check entity
        entity = [c for c in result if c.category == "entity"][0]
        assert entity.owner_scope == "user"
        assert entity.routing_key == "alice"


class TestExtractorSkillAndPattern:
    """Tests for skill and pattern extraction (agent scope)."""

    def test_extract_skill(self):
        """extract_skill → CandidateMemory with category=skill, owner_scope=agent."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_skill",
                "input": {
                    "abstract": "Code review skill",
                    "overview": "Best practices for code review",
                    "content": "Full skill documentation",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "How to do code review"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "skill"
        assert candidate.owner_scope == "agent"
        assert candidate.routing_key == "skill"  # Default for skill

    def test_extract_pattern_owner_scope(self):
        """extract_pattern → CandidateMemory with owner_scope=agent."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_pattern",
                "input": {
                    "routing_key": "error_handling",
                    "abstract": "Common error handling pattern",
                    "overview": "Retry with exponential backoff",
                    "content": "When errors occur, implement retry with exponential backoff",
                    "confidence": 0.85,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I noticed a pattern"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "pattern"
        assert candidate.owner_scope == "agent"
        assert candidate.routing_key == "error_handling"


class TestExtractorToolMapping:
    """Tests for tool name to category mapping."""

    def test_all_tool_names_mapped(self):
        """TOOL_TO_CATEGORY keys match EXTRACTION_TOOLS names exactly."""
        tool_names = {t["name"] for t in EXTRACTION_TOOLS}
        mapped_names = set(TOOL_TO_CATEGORY.keys())

        assert tool_names == mapped_names, (
            f"Mismatch between tools and mapping. "
            f"Tools: {tool_names}, Mapped: {mapped_names}"
        )

    def test_profile_routing_key_defaults_to_category(self):
        """extract_profile has no routing_key field → defaults to 'profile'."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_profile",
                "input": {
                    "abstract": "User profile",
                    "overview": "Name and background",
                    "content": "Full profile",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I'm a developer"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "profile"
        assert candidate.routing_key == "profile"  # Default

    def test_skill_routing_key_defaults_to_category(self):
        """extract_skill has no routing_key field → defaults to 'skill'."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_skill",
                "input": {
                    "abstract": "Debugging skill",
                    "overview": "How to debug",
                    "content": "Full debugging guide",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "How to debug"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "skill"
        assert candidate.routing_key == "skill"  # Default


class TestExtractorUnknownTool:
    """Tests for unknown tool handling."""

    def test_unknown_tool_is_ignored(self):
        """Tool calls with unknown names are ignored (not converted)."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "unknown_tool",
                "input": {
                    "abstract": "Unknown",
                    "confidence": 0.9,
                },
            },
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                },
            },
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "Test"}]
        result = extractor.extract(messages, ctx)

        # Only extract_preference should be converted
        assert len(result) == 1
        assert result[0].category == "preference"
