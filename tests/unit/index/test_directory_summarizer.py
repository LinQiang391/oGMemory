"""Unit tests for DirectorySummarizer.

Tests the directory summary generation logic with mock ContextFS and LLM.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from core.models import ContextNode, RequestContext
from index.directory_summarizer import (
    DirectorySummarizer,
    DirectorySummary,
    is_directory_uri,
)


class MockContextFS:
    """Mock ContextFS for testing."""

    def __init__(self):
        self.nodes: dict[str, ContextNode] = {}
        self.children: dict[str, list[str]] = {}

    def add_node(self, node: ContextNode):
        self.nodes[node.uri] = node

    def set_children(self, parent_uri: str, child_uris: list[str]):
        self.children[parent_uri] = child_uris

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        if uri not in self.nodes:
            raise FileNotFoundError(f"Node not found: {uri}")
        return self.nodes[uri]

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        return self.children.get(uri, [])


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: dict = None):
        self._response = response or {
            "abstract": "Test directory abstract",
            "overview": "## Overview\n\nTest overview content",
        }
        self.call_count = 0
        self.last_prompt = None

    def complete_json(self, prompt: str, schema: dict) -> dict:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response

    def set_response(self, response: dict):
        self._response = response


class FailingMockLLM(MockLLM):
    """Mock LLM that always fails."""

    def complete_json(self, prompt: str, schema: dict) -> dict:
        self.call_count += 1
        raise RuntimeError("LLM API error")


class TestIsDirectoryUri:
    """Tests for is_directory_uri function."""

    def test_directory_uri_ends_with_slash(self):
        assert is_directory_uri("ctx://acme/users/alice/memories/preferences/") is True

    def test_leaf_uri_does_not_end_with_slash(self):
        assert is_directory_uri("ctx://acme/users/alice/memories/profile") is False

    def test_root_directory(self):
        assert is_directory_uri("ctx://acme/") is True

    def test_empty_string(self):
        assert is_directory_uri("") is False


class TestDirectorySummary:
    """Tests for DirectorySummary dataclass."""

    def test_creation(self):
        summary = DirectorySummary(
            abstract="Test abstract",
            overview="Test overview",
            child_count=5,
            categories=["profile", "preference"],
        )
        assert summary.abstract == "Test abstract"
        assert summary.overview == "Test overview"
        assert summary.child_count == 5
        assert summary.categories == ["profile", "preference"]

    def test_default_categories(self):
        summary = DirectorySummary(
            abstract="Test",
            overview="Overview",
            child_count=1,
        )
        assert summary.categories == []


class TestDirectorySummarizer:
    """Tests for DirectorySummarizer class."""

    @pytest.fixture
    def mock_fs(self):
        return MockContextFS()

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    def test_init(self, mock_fs, mock_llm):
        """Test initialization."""
        summarizer = DirectorySummarizer(mock_fs, mock_llm)
        assert summarizer._fs is mock_fs
        assert summarizer._llm is mock_llm
        assert summarizer._max_children == 50

    def test_init_with_custom_max_children(self, mock_fs, mock_llm):
        """Test initialization with custom max_children."""
        summarizer = DirectorySummarizer(mock_fs, mock_llm, max_children=10)
        assert summarizer._max_children == 10


class TestFallbackMethods:
    """Tests for fallback summary generation methods."""

    @pytest.fixture
    def summarizer(self):
        fs = MockContextFS()
        llm = MockLLM()
        return DirectorySummarizer(fs, llm)

    def test_fallback_abstract(self, summarizer):
        """Test fallback abstract generation."""
        child_summaries = [
            {"uri": "uri1", "abstract": "Abstract 1", "category": "preference"},
            {"uri": "uri2", "abstract": "Abstract 2", "category": "event"},
        ]

        abstract = summarizer._fallback_abstract(child_summaries)

        assert "2" in abstract
        assert "preference" in abstract or "event" in abstract

    def test_fallback_overview(self, summarizer):
        """Test fallback overview generation."""
        child_summaries = [
            {"uri": "uri1", "abstract": "Abstract 1", "category": "preference"},
            {"uri": "uri2", "abstract": "Abstract 2", "category": "preference"},
        ]

        overview = summarizer._fallback_overview(child_summaries)

        assert "# Directory Overview" in overview
        assert "preference" in overview

    def test_fallback_overview_multiple_categories(self, summarizer):
        """Test fallback overview with multiple categories."""
        child_summaries = [
            {"uri": "uri1", "abstract": "Abstract 1", "category": "preference"},
            {"uri": "uri2", "abstract": "Abstract 2", "category": "event"},
            {"uri": "uri3", "abstract": "Abstract 3", "category": "event"},
        ]

        overview = summarizer._fallback_overview(child_summaries)

        assert "## preference" in overview
        assert "## event" in overview


class TestGenerateSummary:
    """Tests for _generate_summary method."""

    def test_generates_summary_with_llm(self):
        """Test that summary is generated using LLM."""
        fs = MockContextFS()
        llm = MockLLM({
            "abstract": "Custom abstract",
            "overview": "Custom overview",
        })

        summarizer = DirectorySummarizer(fs, llm)
        child_summaries = [
            {"uri": "uri1", "abstract": "Abstract 1", "category": "preference"},
        ]

        summary = summarizer._generate_summary("test_directory/", child_summaries)

        assert summary.abstract == "Custom abstract"
        assert summary.overview == "Custom overview"
        assert summary.child_count == 1

    def test_abstract_is_truncated(self):
        """Test that abstract is truncated to 200 chars."""
        fs = MockContextFS()
        llm = MockLLM({
            "abstract": "A" * 300,
            "overview": "Overview",
        })

        summarizer = DirectorySummarizer(fs, llm)
        child_summaries = [
            {"uri": "uri1", "abstract": "Abstract 1", "category": "preference"},
        ]

        summary = summarizer._generate_summary("test_directory/", child_summaries)

        assert len(summary.abstract) == 200

    def test_categories_are_deduplicated(self):
        """Test that categories are deduplicated."""
        fs = MockContextFS()
        llm = MockLLM({
            "abstract": "Abstract",
            "overview": "Overview",
        })

        summarizer = DirectorySummarizer(fs, llm)
        child_summaries = [
            {"uri": "uri1", "abstract": "A1", "category": "preference"},
            {"uri": "uri2", "abstract": "A2", "category": "preference"},
            {"uri": "uri3", "abstract": "A3", "category": "event"},
        ]

        summary = summarizer._generate_summary("test_directory/", child_summaries)

        assert set(summary.categories) == {"preference", "event"}
        assert len(summary.categories) == 2

    def test_llm_failure_uses_fallback(self):
        """Test that LLM failure triggers fallback summary."""
        fs = MockContextFS()
        failing_llm = FailingMockLLM()

        summarizer = DirectorySummarizer(fs, failing_llm)
        child_summaries = [
            {"uri": "uri1", "abstract": "Style preference abstract", "category": "preference"},
        ]

        summary = summarizer._generate_summary("test_directory/", child_summaries)

        assert summary is not None
        assert summary.child_count == 1
        assert "preference" in summary.abstract
