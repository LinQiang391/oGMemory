"""Unit tests for CandidateExtractor implementations."""

import pytest

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor
from extraction.profile_extractor import ProfileExtractor
from extraction.preference_extractor import PreferenceExtractor
from extraction.entity_event_extractor import EntityEventExtractor
from extraction.procedure_extractor import ProcedureExtractor
from providers.llm.mock_llm import MockLLM


class TestProfileExtractor:
    """Tests for ProfileExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.extractor = ProfileExtractor(self.llm)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_extract_returns_profile_candidates(self):
        """Test that extract returns profile candidates."""
        messages = [
            {"role": "user", "content": "Hi, I'm Alice"},
            {"role": "assistant", "content": "Hello Alice!"},
        ]

        candidates = self.extractor.extract(messages, self.ctx)

        assert len(candidates) > 0
        assert all(c.category == "profile" for c in candidates)
        assert all(c.owner_scope == "user" for c in candidates)

    def test_extract_filters_low_confidence(self):
        """Test that low confidence candidates are filtered."""
        # Custom fixture with low confidence
        fixtures = {
            "profile": {
                "candidates": [
                    {
                        "routing_key": "profile",
                        "abstract": "Test",
                        "overview": "Test",
                        "content": "Test",
                        "confidence": 0.3,  # Below threshold
                    }
                ]
            }
        }
        llm = MockLLM(fixtures)
        extractor = ProfileExtractor(llm)

        candidates = extractor.extract([], self.ctx)

        assert len(candidates) == 0


class TestPreferenceExtractor:
    """Tests for PreferenceExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.extractor = PreferenceExtractor(self.llm)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_extract_returns_preference_candidates(self):
        """Test that extract returns preference candidates."""
        candidates = self.extractor.extract([], self.ctx)

        assert all(c.category == "preference" for c in candidates)
        assert all(c.owner_scope == "user" for c in candidates)


class TestEntityEventExtractor:
    """Tests for EntityEventExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.extractor = EntityEventExtractor(self.llm)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_extract_returns_entity_and_event_candidates(self):
        """Test that extract returns entity and event candidates."""
        candidates = self.extractor.extract([], self.ctx)

        assert all(c.category in ("entity", "event") for c in candidates)
        assert all(c.owner_scope == "user" for c in candidates)

    def test_extract_filters_invalid_categories(self):
        """Test that invalid categories are filtered."""
        fixtures = {
            "entity": {
                "candidates": [
                    {
                        "category": "invalid",
                        "routing_key": "test",
                        "abstract": "Test",
                        "overview": "Test",
                        "content": "Test",
                        "confidence": 0.9,
                    }
                ]
            }
        }
        llm = MockLLM(fixtures)
        extractor = EntityEventExtractor(llm)

        candidates = extractor.extract([], self.ctx)

        # Invalid category should be filtered
        assert all(c.category in ("entity", "event") for c in candidates)


class TestProcedureExtractor:
    """Tests for ProcedureExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.extractor = ProcedureExtractor(self.llm)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_extract_returns_pattern_candidates(self):
        """Test that extract returns pattern candidates."""
        candidates = self.extractor.extract([], self.ctx)

        assert all(c.category == "pattern" for c in candidates)
        assert all(c.owner_scope == "agent" for c in candidates)


class TestMockLLM:
    """Tests for MockLLM."""

    def test_complete_json_returns_profile(self):
        """Test that MockLLM returns profile fixture."""
        llm = MockLLM()
        schema = {"type": "object"}
        result = llm.complete_json("Extract profile information", schema)

        assert "candidates" in result
        assert llm.call_count == 1

    def test_complete_json_returns_preference(self):
        """Test that MockLLM returns preference fixture."""
        llm = MockLLM()
        schema = {"type": "object"}
        result = llm.complete_json("What are my preferences?", schema)

        assert "candidates" in result
        candidates = result["candidates"]
        if candidates:
            assert candidates[0]["category"] == "preference"

    def test_complete_json_custom_fixtures(self):
        """Test that MockLLM uses custom fixtures."""
        fixtures = {
            "custom": {
                "custom_field": "custom_value",
            }
        }
        llm = MockLLM(fixtures)
        schema = {"type": "object"}
        result = llm.complete_json("This is a custom prompt", schema)

        assert result.get("custom_field") == "custom_value"

    def test_reset_clears_call_count(self):
        """Test that reset clears the call counter."""
        llm = MockLLM()
        llm.complete_json("test", {})
        assert llm.call_count == 1

        llm.reset()
        assert llm.call_count == 0
