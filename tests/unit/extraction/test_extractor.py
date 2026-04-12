"""Unit tests for Extractor tool-use implementation.

Tests verify:
- Empty results when no tool calls
- Single preference extraction
- Multiple types extraction
- Skill extraction with agent scope
- Pattern extraction with agent scope
- Case extraction with agent scope
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
                    "routing_key": "code_review_checklist",
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
        assert candidate.routing_key == "code_review_checklist"

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


class TestExtractorCase:
    """Tests for case extraction (agent scope, problem-resolution pairs)."""

    def test_extract_case_with_agent_scope(self):
        """extract_case → CandidateMemory with category=case, owner_scope=agent."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_case",
                "input": {
                    "routing_key": "debug_api_timeout",
                    "abstract": "Fixed API timeout issue by increasing timeout and adding retry",
                    "overview": "Problem: API calls timing out. Solution: Increased timeout to 30s and added exponential backoff retry.",
                    "content": "Full case: API was timing out after 5s. Diagnosed as network latency. Increased timeout to 30s, added retry with exponential backoff.",
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

        messages = [{"role": "user", "content": "The API is timing out"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.category == "case"
        assert candidate.owner_scope == "agent"
        assert candidate.routing_key == "debug_api_timeout"

    def test_extract_case_requires_routing_key(self):
        """extract_case requires routing_key (no default like profile/skill)."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_case",
                "input": {
                    "routing_key": "fix_memory_leak",
                    "abstract": "Fixed memory leak in batch processing",
                    "overview": "Problem: Memory growing. Solution: Added explicit cleanup after each batch.",
                    "content": "Full case details",
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

        messages = [{"role": "user", "content": "Memory is leaking"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.routing_key == "fix_memory_leak"


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

    def test_profile_routing_key_is_explicit(self):
        """extract_profile requires routing_key for field-level storage."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_profile",
                "input": {
                    "routing_key": "occupation",
                    "abstract": "User is a developer",
                    "overview": "## Occupation\n- Developer",
                    "content": "User is a developer.",
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
        assert candidate.routing_key == "occupation"

    def test_skill_routing_key_is_explicit(self):
        """extract_skill requires routing_key to identify distinct skills."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_skill",
                "input": {
                    "routing_key": "debug_protocol",
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
        assert candidate.routing_key == "debug_protocol"


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


class TestExtractorLanguageDetection:
    """Tests for language detection in Extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.llm._mock_tool_calls = []
        self.extractor = Extractor(self.llm)

    def test_detect_language_chinese(self):
        """_detect_language returns 'zh-CN' for Chinese user messages."""
        messages = [{"role": "user", "content": "你好，我是张三"}]
        assert self.extractor._detect_language(messages) == "zh-CN"

    def test_detect_language_english(self):
        """_detect_language returns 'en' for English user messages."""
        messages = [{"role": "user", "content": "Hello, I need help"}]
        assert self.extractor._detect_language(messages) == "en"

    def test_language_detection_chinese(self):
        """_detect_language returns 'zh-CN' for Chinese user messages."""
        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "你好，我是开发工程师"},
            {"role": "assistant", "content": "Hi"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "zh-CN"

    def test_language_detection_default_english(self):
        """_detect_language returns 'en' for pure English messages."""
        messages = [
            {"role": "user", "content": "Hello, I am a software engineer"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_japanese(self):
        """_detect_language returns 'ja' for Japanese user messages."""
        messages = [
            {"role": "user", "content": "こんにちは、私はエンジニアです"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ja"

    def test_language_detection_korean(self):
        """_detect_language returns 'ko' for Korean user messages."""
        messages = [
            {"role": "user", "content": "안녕하세요, 저는 엔지니어입니다"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ko"

    def test_language_detection_russian(self):
        """_detect_language returns 'ru' for Russian user messages."""
        messages = [
            {"role": "user", "content": "Привет, я инженер"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ru"

    def test_language_detection_empty_messages(self):
        """_detect_language returns 'en' when no user messages."""
        messages = []

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_no_user_role(self):
        """_detect_language returns 'en' when no user role messages."""
        messages = [
            {"role": "assistant", "content": "Hello there"},
            {"role": "system", "content": "You are helpful"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_mixed_content_uses_first_detected(self):
        """_detect_language returns the first detected language in priority order."""
        # Japanese has higher priority now (checked before Chinese)
        # because Hiragana/Katakana are unique to Japanese
        messages = [
            {"role": "user", "content": "你好こんにちは안녕"},  # Mixed CJK
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ja"  # Japanese detected first (Hiragana present)

    def test_extract_includes_language_in_prompt(self):
        """extract() includes detected language in the LLM prompt."""
        messages = [
            {"role": "user", "content": "你好，我是开发工程师"},
        ]

        # The MockLLM should receive a prompt containing the language
        result = self.extractor.extract(messages, RequestContext(
            account_id="test",
            user_id="user",
            agent_id="agent",
            session_id="session",
            trace_id="trace",
        ))

        # Verify the language was detected
        assert self.extractor._detect_language(messages) == "zh-CN"

        # Verify the prompt sent to LLM contains the language code
        assert "zh-CN" in self.llm._last_prompt
        # The prompt instructs to output fields "in zh-CN" or similar
        assert "zh-cn" in self.llm._last_prompt.lower()
