"""Unit tests for OutboxStore."""

import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

import pytest

from core.models import RequestContext, ContextNode, OutboxEvent
from core.enums import EventType
from commit.outbox_store import OutboxStore


class TestOutboxStoreInit:
    """Verify OutboxStore initialization."""

    def test_init_with_defaults(self):
        """init should accept default mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs)

        assert store._client is mock_client
        assert store._fs is mock_fs
        assert store._mount_prefix == "/local"

    def test_init_with_custom_mount_prefix(self):
        """init should accept custom mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs, mount_prefix="/custom")

        assert store._mount_prefix == "/custom"

    def test_init_trailing_slash_removed(self):
        """init should remove trailing slash from mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        store = OutboxStore(mock_client, mock_fs, mount_prefix="/custom/")

        assert store._mount_prefix == "/custom"

    def test_init_with_path_traversal_raises_error(self):
        """init should reject mount_prefix with path traversal patterns."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="../../../etc")

    def test_init_with_double_dot_raises_error(self):
        """init should reject mount_prefix containing '..'."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="/local/../test")

    def test_init_with_relative_path_raises_error(self):
        """init should reject relative mount_prefix."""
        mock_client = Mock()
        mock_fs = Mock()

        with pytest.raises(ValueError, match="Invalid mount_prefix"):
            OutboxStore(mock_client, mock_fs, mount_prefix="local")


class TestUriToAgfsPath:
    """Verify _uri_to_agfs_path method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/local")

    def test_uri_to_path_with_default_prefix(self):
        """_uri_to_agfs_path should prepend default mount_prefix."""
        uri = "ctx://test-account/users/user-123/memories/profile"
        result = self.store._uri_to_agfs_path(uri)

        assert result.startswith("/local/")
        assert "accounts/test-account/users/user-123/memories/profile" in result

    def test_uri_to_path_with_custom_prefix(self):
        """_uri_to_agfs_path should use custom mount_prefix."""
        store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/custom")
        uri = "ctx://test-account/users/user-123/memories/profile"
        result = store._uri_to_agfs_path(uri)

        assert result.startswith("/custom/")
        assert "accounts/test-account/users/user-123/memories/profile" in result


class TestRegisterWrite:
    """Verify register_write method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs, mount_prefix="/local")

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        self.node = ContextNode(
            uri="ctx://test-account/users/user-123/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Test profile",
            overview="Overview",
            content="Full content",
        )

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_builds_index_records(self, mock_build):
        """register_write should build IndexRecords from ContextNode."""
        mock_build.return_value = []

        self.store.register_write(self.node, self.ctx)

        mock_build.assert_called_once_with(self.node)

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_creates_upsert_event(self, mock_build):
        """register_write should create UPSERT_CONTEXT event."""
        from core.models import IndexRecord
        mock_build.return_value = [
            IndexRecord(
                id="test-id",
                uri=self.node.uri,
                level=0,
                text="abstract",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile"},
            )
        ]

        event = self.store.register_write(self.node, self.ctx)

        assert event.event_type == EventType.UPSERT_CONTEXT.value
        assert event.uri == self.node.uri
        assert event.status == "PENDING"
        assert "records" in event.payload

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_includes_all_record_fields(self, mock_build):
        """register_write should include all required IndexRecord fields in payload."""
        from core.models import IndexRecord
        mock_build.return_value = [
            IndexRecord(
                id="test-id-0",
                uri=self.node.uri,
                level=0,
                text="abstract text",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile", "has_content": True},
            ),
            IndexRecord(
                id="test-id-1",
                uri=self.node.uri,
                level=1,
                text="overview text",
                filters={"account_id": "test-account", "owner_space": "user_space"},
                metadata={"category": "profile", "has_content": True},
            ),
        ]

        event = self.store.register_write(self.node, self.ctx)

        records = event.payload["records"]
        assert len(records) == 2

        # Verify all required fields are present
        for record in records:
            assert "id" in record
            assert "uri" in record
            assert "level" in record
            assert "text" in record
            assert "filters" in record
            assert "metadata" in record

        # Verify filters contain account_id and owner_space
        assert records[0]["filters"]["account_id"] == "test-account"
        assert records[0]["filters"]["owner_space"] == "user_space"

    @patch('index.index_record_builder.build_index_records')
    @patch('commit.outbox_store.uuid.uuid4')
    def test_register_write_generates_unique_event_id(self, mock_uuid, mock_build):
        """register_write should generate unique event_id."""
        mock_uuid.return_value = "test-uuid-123"
        mock_build.return_value = []

        event = self.store.register_write(self.node, self.ctx)

        assert event.event_id == "test-uuid-123"

    @patch('index.index_record_builder.build_index_records')
    def test_register_write_calls_write_event(self, mock_build):
        """register_write should call _write_event to persist event."""
        mock_build.return_value = []
        self.store._write_event = Mock()

        self.store.register_write(self.node, self.ctx)

        self.store._write_event.assert_called_once()


class TestRegisterDelete:
    """Verify register_delete method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_register_delete_creates_delete_event(self):
        """register_delete should create DELETE_CONTEXT event."""
        uri = "ctx://test-account/users/user-123/memories/profile"

        event = self.store.register_delete(uri, self.ctx)

        assert event.event_type == EventType.DELETE_CONTEXT.value
        assert event.uri == uri
        assert event.status == "PENDING"
        assert event.payload == {}


class TestRegisterRelation:
    """Verify register_relation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_fs = Mock()
        self.store = OutboxStore(self.mock_client, self.mock_fs)

        self.ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    def test_register_relation_with_edges(self):
        """register_relation should create UPSERT_RELATION event with edges."""
        from core.models import RelationEdge

        edges = [
            RelationEdge(
                from_uri="ctx://test-account/users/user-123/memories/profile",
                to_uri="ctx://test-account/users/user-123/memories/preferences/coffee",
                relation_type="related_to",
                weight=0.9,
                reason="User preference",
            )
        ]

        event = self.store.register_relation(edges, self.ctx)

        assert event.event_type == EventType.UPSERT_RELATION.value
        assert "edges" in event.payload
        assert len(event.payload["edges"]) == 1

        edge_data = event.payload["edges"][0]
        assert edge_data["from_uri"] == edges[0].from_uri
        assert edge_data["to_uri"] == edges[0].to_uri
        assert edge_data["relation_type"] == edges[0].relation_type
        assert edge_data["weight"] == edges[0].weight

    def test_register_relation_with_empty_edges(self):
        """register_relation should handle empty edge list."""
        event = self.store.register_relation([], self.ctx)

        assert event.event_type == EventType.UPSERT_RELATION.value
        assert event.payload["edges"] == []
