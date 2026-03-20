"""Unit tests for OpenClaw plugin write_memory.py."""

import pytest
from unittest.mock import Mock, patch

from core.models import RequestContext, CandidateMemory
from openclaw_plugin.bridge.write_memory import (
    write_memory,
    init_context_engine,
    CATEGORY_OWNER_SCOPE,
    DEFAULT_OWNER_SCOPE,
)


class TestCategoryOwnerScopeMapping:
    """Tests for category to owner_scope mapping."""

    def test_user_categories_map_to_user(self):
        """User categories should map to 'user' owner_scope."""
        user_categories = ["profile", "preference", "entity", "event"]
        for category in user_categories:
            assert CATEGORY_OWNER_SCOPE.get(category, DEFAULT_OWNER_SCOPE) == "user"

    def test_agent_categories_map_to_agent(self):
        """Agent categories should map to 'agent' owner_scope."""
        agent_categories = ["case", "pattern", "skill"]
        for category in agent_categories:
            assert CATEGORY_OWNER_SCOPE.get(category, DEFAULT_OWNER_SCOPE) == "agent"

    def test_unknown_category_defaults_to_user(self):
        """Unknown categories should default to 'user' owner_scope."""
        assert CATEGORY_OWNER_SCOPE.get("unknown", DEFAULT_OWNER_SCOPE) == "user"


class TestWriteMemoryBugFixes:
    """Tests for the three bug fixes in write_memory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_api = Mock()

        # Mock the write_memory method to return a WritePlan
        self.mock_api.write_memory.return_value = type('obj', (object,), {
            'action': 'merge',
            'target_uri': 'ctx://test-account/users/user-123/memories/preferences/coffee',
            'merged_fields': {'usage_count': 5},
        })()

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug1_routing_key_uses_input_not_uuid(self, mock_get_api):
        """Bug 1 fix: routing_key should use input parameter, not generate uuid."""
        mock_get_api.return_value = self.mock_api

        # Call with explicit routing_key
        result = write_memory(
            content="User likes dark roast coffee",
            category="preference",
            account_id="test-account",
            user_id="user-123",
            agent_id="bot",
            session_id="sess-123",
            routing_key="coffee",  # Explicit routing key
        )

        # Verify write_memory was called with routing_key="coffee"
        candidate = self.mock_api.write_memory.call_args[0][0]
        assert candidate.routing_key == "coffee"
        assert candidate.routing_key != "preference_12345678"  # Not uuid-based

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug1_routing_key_defaults_to_category(self, mock_get_api):
        """Bug 1 fix: routing_key should default to category when not provided."""
        mock_get_api.return_value = self.mock_api

        # Call without routing_key
        result = write_memory(
            content="User likes coffee",
            category="preference",
            account_id="test-account",
            user_id="user-123",
            agent_id="bot",
            session_id="sess-123",
            # No routing_key provided
        )

        # Verify routing_key defaults to category
        candidate = self.mock_api.write_memory.call_args[0][0]
        assert candidate.routing_key == "preference"

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug2_owner_scope_agent_categories(self, mock_get_api):
        """Bug 2 fix: case/pattern/skill should use 'agent' owner_scope."""
        mock_get_api.return_value = self.mock_api

        for category in ["case", "pattern", "skill"]:
            result = write_memory(
                content=f"Content for {category}",
                category=category,
                account_id="test-account",
                user_id="user-123",
                agent_id="bot",
                session_id="sess-123",
            )

            # Verify owner_scope is "agent" for these categories
            candidate = self.mock_api.write_memory.call_args[0][0]
            assert candidate.owner_scope == "agent"

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug2_owner_scope_user_categories(self, mock_get_api):
        """Bug 2 fix: profile/preference/entity/event should use 'user' owner_scope."""
        mock_get_api.return_value = self.mock_api

        for category in ["profile", "preference", "entity", "event"]:
            result = write_memory(
                content=f"Content for {category}",
                category=category,
                account_id="test-account",
                user_id="user-123",
                agent_id="bot",
                session_id="sess-123",
            )

            # Verify owner_scope is "user" for these categories
            candidate = self.mock_api.write_memory.call_args[0][0]
            assert candidate.owner_scope == "user"

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug3_abstract_uses_first_sentence(self, mock_get_api):
        """Bug 3 fix: abstract should extract first sentence, not simple truncation."""
        mock_get_api.return_value = self.mock_api

        # Content with Chinese and English punctuation
        content = "User prefers dark roast coffee. Usually drinks 2-3 cups daily. 每天早上必喝。"

        result = write_memory(
            content=content,
            category="preference",
            account_id="test-account",
            user_id="user-123",
            agent_id="bot",
            session_id="sess-123",
        )

        # Verify abstract is first sentence (period stripped by split logic)
        candidate = self.mock_api.write_memory.call_args[0][0]
        assert candidate.abstract == "User prefers dark roast coffee"
        # Should NOT be "User prefers dark roast coffee. Usually dr" (simple truncation)

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_bug3_overview_structured(self, mock_get_api):
        """Bug 3 fix: overview should be structured with category header."""
        mock_get_api.return_value = self.mock_api

        content = "This is a long content that needs proper overview. " * 20

        result = write_memory(
            content=content,
            category="preference",
            account_id="test-account",
            user_id="user-123",
            agent_id="bot",
            session_id="sess-123",
        )

        # Verify overview has category header
        candidate = self.mock_api.write_memory.call_args[0][0]
        assert "## Preference" in candidate.overview
        assert candidate.overview.startswith("## Preference")

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_abstract_and_overview_provided_skip_auto_generation(self, mock_get_api):
        """When abstract and overview are provided, use them as-is."""
        mock_get_api.return_value = self.mock_api

        custom_abstract = "Custom abstract"
        custom_overview = "Custom overview"

        result = write_memory(
            content="Some content",
            category="preference",
            account_id="test-account",
            user_id="user-123",
            agent_id="bot",
            session_id="sess-123",
            abstract=custom_abstract,
            overview=custom_overview,
        )

        # Verify provided abstract and overview are used
        candidate = self.mock_api.write_memory.call_args[0][0]
        assert candidate.abstract == custom_abstract
        assert candidate.overview == custom_overview

    @patch('openclaw_plugin.bridge.write_memory.get_write_api')
    def test_write_memory_without_initialized_api_raises_error(self, mock_get_api):
        """write_memory should raise error when API not initialized."""
        mock_get_api.return_value = None

        with pytest.raises(RuntimeError, match="MemoryWriteAPI not initialized"):
            write_memory(
                content="Test content",
                category="preference",
                account_id="test-account",
                user_id="user-123",
                agent_id="bot",
                session_id="sess-123",
            )


class TestInitContextEngine:
    """Tests for init_context_engine function."""

    @patch('service.api.init_write_api')
    def test_init_context_engine_delegates_to_init_write_api(self, mock_init):
        """init_context_engine should delegate to service.init_write_api."""
        mock_fs = Mock()
        mock_llm = Mock()
        mock_outbox = Mock()

        result = init_context_engine(mock_fs, mock_llm, mock_outbox)

        mock_init.assert_called_once_with(mock_fs, mock_llm, mock_outbox)
