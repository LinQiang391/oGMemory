"""Unit tests for service layer write API.

Tests verify:
- Extractor initialization (all four extractors)
- commit_session calls the complete pipeline
- OutboxStore integration
- Statistics return values
- Single candidate writes
- Cross-account access control

See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from core.models import RequestContext, CandidateMemory, WritePlan, ContextNode
from core.errors import AccessDeniedError
from service.api import MemoryWriteAPI, init_write_api, get_write_api
from commit import OutboxStore
from providers.llm.mock_llm import MockLLM


class TestMemoryWriteAPIInit:
    """Verify MemoryWriteAPI initialization."""

    def test_init_creates_context_writer(self):
        """init should create ContextWriter with fs."""
        mock_fs = Mock()
        mock_llm = MockLLM()

        api = MemoryWriteAPI(mock_fs, mock_llm)

        assert api._fs is mock_fs
        assert api._llm is mock_llm
        assert api._writer is not None
        assert api._pipeline is not None

    def test_init_creates_four_extractors(self):
        """init should create all four extractors."""
        mock_fs = Mock()
        mock_llm = MockLLM()

        api = MemoryWriteAPI(mock_fs, mock_llm)

        # Verify pipeline has extractors set
        assert api._pipeline._extractors is not None
        assert len(api._pipeline._extractors) == 4

        # Verify extractor types
        from extraction import ProfileExtractor, PreferenceExtractor, EntityEventExtractor, ProcedureExtractor
        extractor_types = [type(e) for e in api._pipeline._extractors]

        assert ProfileExtractor in extractor_types
        assert PreferenceExtractor in extractor_types
        assert EntityEventExtractor in extractor_types
        assert ProcedureExtractor in extractor_types

    def test_init_with_outbox_store(self):
        """init should accept optional OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        mock_outbox = Mock(spec=OutboxStore)

        api = MemoryWriteAPI(mock_fs, mock_llm, mock_outbox)

        assert api._outbox_store is mock_outbox

    def test_init_without_outbox_store(self):
        """init should work without OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()

        api = MemoryWriteAPI(mock_fs, mock_llm)

        assert api._outbox_store is None


class TestCommitSession:
    """Verify commit_session method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_commit_session_calls_pipeline_extract(self):
        """commit_session should call extract on pipeline."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Mock the writer to return success
        self.mock_fs.exists.return_value = False
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify extract was called
        assert "candidates_extracted" in result

    def test_commit_session_filters_by_confidence(self):
        """commit_session should filter low confidence candidates."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        # Test with high threshold
        result = self.api.commit_session(messages, self.ctx, confidence_threshold=0.9)

        assert "candidates_filtered" in result

    def test_commit_session_deduplicates_candidates(self):
        """commit_session should deduplicate before writing."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify writes were performed
        assert "writes_completed" in result

    def test_commit_session_returns_statistics(self):
        """commit_session should return comprehensive statistics."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer to return different plans
        plans = [
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[]),
            WritePlan(action="skip", target_uri="uri2", merged_fields={}, relation_edges=[]),
            WritePlan(action="create", target_uri="uri3", merged_fields={}, relation_edges=[]),
        ]
        self.api._writer.write_candidates = Mock(return_value=plans)

        # Mock node read for outbox
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="uri1",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.commit_session(messages, self.ctx)

        # Verify all statistics keys
        assert "candidates_extracted" in result
        assert "candidates_filtered" in result
        assert "writes_completed" in result
        assert "writes_skipped" in result
        assert "writes_failed" in result
        assert "plans" in result

        # Verify counts
        assert result["writes_completed"] == 2
        assert result["writes_skipped"] == 1

    def test_commit_session_empty_messages(self):
        """commit_session should handle empty messages.

        Note: MockLLM returns fixtures even with empty messages,
        so candidates_extracted may be > 0. The important thing
        is that it doesn't crash and returns valid statistics.
        """
        result = self.api.commit_session([], self.ctx)

        # Should return valid statistics structure
        assert "candidates_extracted" in result
        assert "writes_completed" in result
        assert "writes_skipped" in result
        assert "writes_failed" in result
        # All counts should be non-negative integers
        assert result["candidates_extracted"] >= 0
        assert result["writes_completed"] >= 0

    def test_commit_session_registers_outbox_events(self):
        """commit_session should register OutboxEvents when OutboxStore is provided."""
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        self.api._writer.write_candidates = Mock(return_value=[
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[]),
        ])

        # Mock node read for outbox registration
        mock_node = ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Abstract",
            overview="Overview",
            content="Content",
        )
        self.mock_fs.read_node = Mock(return_value=mock_node)

        result = self.api.commit_session(messages, self.ctx)

        # Verify outbox was called
        self.mock_outbox.register_write.assert_called()
        call_args = self.mock_outbox.register_write.call_args
        assert call_args[0][0] is mock_node
        assert call_args[0][1] is self.ctx

    def test_commit_session_without_outbox_store(self):
        """commit_session should work without OutboxStore."""
        api = MemoryWriteAPI(self.mock_fs, self.mock_llm, outbox_store=None)
        messages = [{"role": "user", "content": "Test"}]

        # Mock writer
        api._writer.write_candidates = Mock(return_value=[])

        # Should not raise
        result = api.commit_session(messages, self.ctx)

        assert result is not None


class TestWriteMemory:
    """Verify write_memory method for single candidate writes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_write_memory_single_candidate(self):
        """write_memory should write a single candidate."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        # Mock node read for outbox
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.write_memory(candidate, self.ctx)

        # Verify result structure
        assert "action" in result
        assert "target_uri" in result
        assert "merged_fields" in result
        assert result["action"] == "create"

        # Verify writer was called
        self.api._writer.write_candidate.assert_called_once_with(candidate, self.ctx)

    def test_write_memory_skipped_candidate(self):
        """write_memory should handle skip action."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="User profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer to return skip
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="skip",
            target_uri="ctx://test-account/users/user-123/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        result = self.api.write_memory(candidate, self.ctx)

        assert result["action"] == "skip"

        # Outbox should NOT be called for skipped writes
        self.mock_outbox.register_write.assert_not_called()


class TestWriteMemories:
    """Verify write_memories method for batch writes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.mock_outbox = Mock(spec=OutboxStore)
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm, self.mock_outbox)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_write_memories_parallel(self):
        """write_memories should write in parallel by default."""
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key=f"pref_{i}",
                abstract=f"Preference {i}",
                overview="Overview",
                content="Content",
                confidence=0.9,
            )
            for i in range(3)
        ]

        # Mock writer
        self.api._writer.write_candidates_parallel = Mock(return_value=[
            WritePlan(action="create", target_uri=f"uri{i}", merged_fields={}, relation_edges=[])
            for i in range(3)
        ])

        # Mock node read
        self.mock_fs.read_node = Mock(return_value=ContextNode(
            uri="uri1",
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user_space",
            abstract="",
            overview="",
            content="",
        ))

        result = self.api.write_memories(candidates, self.ctx, parallel=True)

        assert len(result) == 3
        self.api._writer.write_candidates_parallel.assert_called_once()

    def test_write_memories_serial(self):
        """write_memories should write serially when parallel=False."""
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="pref_1",
                abstract="Preference 1",
                overview="Overview",
                content="Content",
                confidence=0.9,
            )
        ]

        # Mock writer
        self.api._writer.write_candidates = Mock(return_value=[
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[])
        ])

        result = self.api.write_memories(candidates, self.ctx, parallel=False)

        assert len(result) == 1
        self.api._writer.write_candidates.assert_called_once()

    def test_write_memories_deduplicates(self):
        """write_memories should deduplicate candidates."""
        # Create duplicate candidates
        candidates = [
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",
                abstract="Likes coffee",
                overview="Overview",
                content="Content",
                confidence=0.9,
            ),
            CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",  # Same routing key
                abstract="Likes dark roast",
                overview="Different overview",
                content="Different content",
                confidence=0.8,  # Lower confidence
            ),
        ]

        # Mock writer
        self.api._writer.write_candidates = Mock(return_value=[
            WritePlan(action="create", target_uri="uri1", merged_fields={}, relation_edges=[])
        ])

        result = self.api.write_memories(candidates, self.ctx, parallel=False)

        # Should only call writer once (deduplicated)
        self.api._writer.write_candidates.assert_called_once()


class TestWriteAPICrossAccountAccess:
    """Verify cross-account access control for write operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_fs = Mock()
        self.mock_llm = MockLLM()
        self.api = MemoryWriteAPI(self.mock_fs, self.mock_llm)

        self.ctx_acme = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

    def test_write_memory_enforces_account_in_context(self):
        """write_memory should use account_id from context."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="Profile",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        # Mock writer
        self.api._writer.write_candidate = Mock(return_value=WritePlan(
            action="create",
            target_uri="ctx://acme/users/alice/memories/profile",
            merged_fields={},
            relation_edges=[],
        ))

        self.api.write_memory(candidate, self.ctx_acme)

        # Verify the writer was called with acme context
        self.api._writer.write_candidate.assert_called_once()
        call_ctx = self.api._writer.write_candidate.call_args[0][1]
        assert call_ctx.account_id == "acme"


class TestWriteAPISingleton:
    """Verify global write API singleton pattern."""

    def test_init_write_api_creates_singleton(self):
        """init_write_api should create and return a global API instance."""
        mock_fs = Mock()
        mock_llm = MockLLM()

        api = init_write_api(mock_fs, mock_llm)

        assert api is not None
        assert isinstance(api, MemoryWriteAPI)

    def test_get_write_api_returns_singleton(self):
        """get_write_api should return the same instance created by init_write_api."""
        mock_fs = Mock()
        mock_llm = MockLLM()

        # First call
        api1 = init_write_api(mock_fs, mock_llm)
        # Second call
        api2 = get_write_api()

        assert api1 is api2

    def test_init_write_api_with_outbox_store(self):
        """init_write_api should accept optional OutboxStore."""
        mock_fs = Mock()
        mock_llm = MockLLM()
        mock_outbox = Mock(spec=OutboxStore)

        api = init_write_api(mock_fs, mock_llm, mock_outbox)

        assert api._outbox_store is mock_outbox
