"""End-to-end verification tests for recent fixes.

Verifies:
1. Directory events are written to mount_prefix path
2. IndexService with fs/llm processes directory events
3. Agent skill queries recall from agent space
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, TypedQuery
from commit import OutboxStore
from fs.agfs_adapter import AGFSContextFS
from index import OutboxWorker, WorkerConfig
from retrieval.query_planner import QueryPlanner
from providers.llm import MockLLM
from providers.embedder import MockEmbedder
from providers.vector_index import InMemoryVectorIndex
from core.enums import ContextType


@pytest.mark.integration
class TestFixesVerification:
    """Verification tests for recent bug fixes."""

    def test_directory_event_uses_mount_prefix(self):
        """Verify directory events are written to correct mount_prefix path."""
        mock_client = Mock()
        mock_client.mkdir.return_value = None
        mock_client.write.return_value = None

        mock_fs = Mock()
        mock_fs.list_children.return_value = []

        # Use custom mount_prefix
        store = OutboxStore(client=mock_client, fs=mock_fs, mount_prefix="/custom")

        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        # Create a mock node
        from core.models import ContextNode
        node = ContextNode(
            uri="ctx://test-acct/users/user-1/memories/preferences/coffee",
            context_type="MEMORY",
            category="preference",
            level=3,
            owner_space="user:user-1",
            abstract="Likes coffee",
            overview="## Coffee Preference\n\nDark roast",
            content="Full content here",
            metadata={"status": "ACTIVE"},
        )

        event = store.register_directory(node, ctx)

        # Verify event was created
        assert event.event_type == "UPSERT_DIRECTORY"
        assert event.uri == "ctx://test-acct/users/user-1/memories/preferences/"

        # Verify write path includes mount_prefix
        write_calls = [call[0][0] for call in mock_client.write.call_args_list]
        outbox_writes = [p for p in write_calls if ".outbox" in p]
        assert len(outbox_writes) > 0
        assert any("/custom/accounts/" in p for p in outbox_writes), \
            f"Expected /custom/accounts/ in paths: {outbox_writes}"

    def test_index_service_with_fs_llm_processes_directory(self):
        """Verify IndexService with fs/llm can process UPSERT_DIRECTORY events."""
        mock_client = Mock()
        mock_fs = Mock()
        mock_fs.list_children.return_value = []
        mock_llm = MockLLM()

        embedder = MockEmbedder()
        vector_index = InMemoryVectorIndex()

        # Create IndexService with fs and llm
        from service.index_service import IndexService
        service = IndexService(
            outbox_store=OutboxStore(client=mock_client, fs=mock_fs),
            embedder=embedder,
            vector_index=vector_index,
            get_account_ids=lambda: ["test-acct"],
            fs=mock_fs,
            llm=mock_llm,
        )

        # Start service - should not crash
        service.start(worker_count=1)

        # Verify worker was created with fs and llm
        assert service._worker is not None
        assert service._worker._fs is mock_fs
        assert service._worker._llm is mock_llm

        service.stop(wait=True)

    def test_agent_skill_query_uses_agent_space(self):
        """Verify SKILL queries use agent_space_name()."""
        planner = QueryPlanner()

        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        # SKILL query should use agent space
        queries = planner.plan("how to deploy a service", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == ContextType.SKILL.value
        assert queries[0].owner_space == ctx.agent_space_name(), \
            f"Expected {ctx.agent_space_name()}, got {queries[0].owner_space}"

        # MEMORY query should not filter by owner_space (search both)
        queries = planner.plan("用户偏好是什么", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == ContextType.MEMORY.value
        assert queries[0].owner_space is None, \
            f"Expected None for MEMORY, got {queries[0].owner_space}"

    def test_overview_append_preserves_existing_content(self):
        """Verify overview_append merges instead of overwriting."""
        from commit import ArchiveBuilder
        from core.models import WritePlan, CandidateMemory

        builder = ArchiveBuilder(llm=MockLLM())

        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee",
            abstract="Likes dark roast",
            overview="Prefers dark roast coffee",
            content="Full content",
            confidence=0.9,
        )

        existing_overview = "Alice's coffee preference"
        new_overview = "She likes dark roast"

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-acct/users/user-1/memories/preferences/coffee",
            merged_fields={
                "existing_overview": existing_overview,
                "overview_append": new_overview,
            },
            relation_edges=[],
        )

        node = builder.build(candidate, plan, ctx)

        # Both overviews should be present
        assert existing_overview in node.overview, \
            f"Existing overview '{existing_overview}' not in '{node.overview}'"
        assert new_overview in node.overview, \
            f"New overview '{new_overview}' not in '{node.overview}'"

    def test_content_append_preserves_existing_content(self):
        """Verify content_append merges instead of overwriting."""
        from commit import ArchiveBuilder
        from core.models import WritePlan, CandidateMemory

        builder = ArchiveBuilder(llm=MockLLM())

        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding",
            abstract="Coding style",
            overview="## Coding Style",
            content="New info",
            confidence=0.9,
        )

        existing_content = "## Coding Style\n\n- Uses 4 spaces"
        new_content = "- Prefers snake_case"

        plan = WritePlan(
            action="merge",
            target_uri="ctx://test-acct/users/user-1/memories/preferences/coding",
            merged_fields={
                "existing_content": existing_content,
                "content_append": new_content,
            },
            relation_edges=[],
        )

        node = builder.build(candidate, plan, ctx)

        # Both content parts should be present
        assert "Uses 4 spaces" in node.content
        assert "Prefers snake_case" in node.content
