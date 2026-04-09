"""Unit tests for AGFSRelationStore."""

import json
from unittest.mock import Mock

import pytest

from core.models import RequestContext, RelationEdge
from core.errors import AccessDeniedError
from providers.relation_store.agfs_relation_store import AGFSRelationStore


class TestAGFSRelationStore:
    """Tests for AGFSRelationStore."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.store = AGFSRelationStore(self.mock_client)
        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_get_edges_returns_empty_list_for_missing_file(self):
        """Test that missing .relations.json returns empty list."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        edges = self.store.get_edges("ctx://test-account/users/user-123/memories/profile", self.ctx)

        assert edges == []

    def test_get_edges_returns_parsed_edges(self):
        """Test that existing .relations.json is parsed correctly."""
        test_edges = [
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/coffee_shop",
                "relation_type": "related_to",
                "weight": 0.85,
                "reason": "Alice visits this coffee shop",
            }
        ]

        self.mock_client.read.return_value = json.dumps(test_edges).encode()

        edges = self.store.get_edges("ctx://test-account/users/user-123/memories/profile", self.ctx)

        assert len(edges) == 1
        assert edges[0].relation_type == "related_to"
        assert edges[0].weight == 0.85

    def test_upsert_edges_creates_new_file(self):
        """Test that upsert creates .relations.json if missing."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.85,
                reason="Alice visits this coffee shop",
            )
        ]

        self.store.upsert_edges(edges, self.ctx)

        # Verify write was called
        self.mock_client.write.assert_called_once()
        written_content = self.mock_client.write.call_args[0][1].decode('utf-8')
        written_data = json.loads(written_content)
        assert len(written_data) == 1
        assert written_data[0]["relation_type"] == "related_to"

    def test_upsert_edges_updates_existing(self):
        """Test that upsert updates existing edges."""
        existing_edges = [
            {
                "from_uri": "ctx://test-account/users/user-123/memories/profile",
                "to_uri": "ctx://test-account/users/user-123/memories/entities/coffee_shop",
                "relation_type": "related_to",
                "weight": 0.5,
                "reason": "Old reason",
            }
        ]

        self.mock_client.read.return_value = json.dumps(existing_edges).encode()

        # Upsert with same edge but different weight/reason
        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.9,
                reason="Updated reason",
            )
        ]

        self.store.upsert_edges(edges, self.ctx)

        # Verify write was called with updated edge
        written_content = self.mock_client.write.call_args[0][1].decode('utf-8')
        written_data = json.loads(written_content)
        assert len(written_data) == 1
        assert written_data[0]["weight"] == 0.9
        assert written_data[0]["reason"] == "Updated reason"

    def test_upsert_edges_handles_multiple_sources(self):
        """Test that upsert handles edges from multiple source URIs."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.85,
                reason="Alice visits this coffee shop",
            ),
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/events/visit_001",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.9,
                reason="Event at coffee shop",
            ),
        ]

        self.store.upsert_edges(edges, self.ctx)

        # Should have written to two different files
        assert self.mock_client.write.call_count == 2

    def test_get_edges_cross_account_denied(self):
        """Test that get_edges denies cross-account access."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        # Try to access edges from different account
        with pytest.raises(AccessDeniedError, match="belongs to account"):
            self.store.get_edges("ctx://other-account/users/alice/memories/profile", self.ctx)

    def test_upsert_edges_cross_account_denied(self):
        """Test that upsert_edges denies cross-account access."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        # Try to upsert edges for different account
        edges = [
            RelationEdge(
                from_uri="ctx://other-account/users/alice/memories/profile",
                to_uri="ctx://other-account/users/alice/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.85,
                reason="Cross-account attempt",
            )
        ]

        with pytest.raises(AccessDeniedError, match="belongs to account"):
            self.store.upsert_edges(edges, self.ctx)

    def test_uses_mount_prefix(self):
        """Test that mount_prefix is used in file paths."""
        from pyagfs.exceptions import AGFSClientError
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        # Create store with custom mount_prefix
        store = AGFSRelationStore(self.mock_client, mount_prefix="/custom")

        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/entities/coffee_shop",
                relation_type="related_to",
                weight=0.85,
                reason="Test",
            )
        ]

        store.upsert_edges(edges, self.ctx)

        # Verify write path includes custom mount_prefix
        write_path = self.mock_client.write.call_args[0][0]
        assert write_path.startswith("/custom/accounts/")
        assert write_path.endswith(".relations.json")
