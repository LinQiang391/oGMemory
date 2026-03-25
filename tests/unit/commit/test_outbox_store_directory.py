"""Tests for OutboxStore.register_directory."""

import json
from unittest.mock import Mock, patch

import pytest

from core.models import ContextNode, RequestContext, OutboxEvent
from commit.outbox_store import OutboxStore


class TestNodeUriToDirectoryUri:
    """Tests for _node_uri_to_directory_uri method."""

    @pytest.fixture
    def store(self, mock_client, mock_fs):
        """Create OutboxStore with mocks."""
        return OutboxStore(client=mock_client, fs=mock_fs)

    def test_leaf_node_to_directory(self, store):
        """Test converting leaf node URI to directory URI."""
        node_uri = "ctx://acme/users/u1/memories/preferences/coffee"
        result = store._node_uri_to_directory_uri(node_uri)
        assert result == "ctx://acme/users/u1/memories/preferences/"

    def test_node_without_trailing_slash(self, store):
        """Test node URI without trailing slash."""
        node_uri = "ctx://acme/users/u1/memories/preferences"
        result = store._node_uri_to_directory_uri(node_uri)
        assert result == "ctx://acme/users/u1/memories/"

    def test_deeply_nested_node(self, store):
        """Test deeply nested node URI."""
        node_uri = "ctx://acme/users/u1/memories/entities/companies/example"
        result = store._node_uri_to_directory_uri(node_uri)
        assert result == "ctx://acme/users/u1/memories/entities/companies/"


class TestCollectSiblingAbstracts:
    """Tests for _collect_sibling_abstracts method."""

    @pytest.fixture
    def store(self, mock_client, mock_fs):
        """Create OutboxStore with mocks."""
        return OutboxStore(client=mock_client, fs=mock_fs)

    @pytest.fixture
    def ctx(self):
        """Create test RequestContext."""
        return RequestContext(
            account_id="acme",
            user_id="u1",
            agent_id="bot",
            session_id="sess",
            trace_id="trace",
        )

    def test_collects_siblings_from_list_children(self, store, mock_fs, ctx):
        """Test collecting sibling abstracts via list_children."""
        node_uri = "ctx://acme/users/u1/memories/preferences/coffee"

        # Mock list_children to return sibling URIs
        mock_fs.list_children.return_value = [
            "ctx://acme/users/u1/memories/preferences/coffee",
            "ctx://acme/users/u1/memories/preferences/tea",
        ]

        # Mock read_node to return nodes with abstracts
        mock_fs.read_node.side_effect = [
            ContextNode(
                uri="ctx://acme/users/u1/memories/preferences/coffee",
                context_type="MEMORY",
                category="preference",
                level=3,
                owner_space="user:u1",
                abstract="User likes coffee",
                overview="",
                content="",
                metadata={},
            ),
            ContextNode(
                uri="ctx://acme/users/u1/memories/preferences/tea",
                context_type="MEMORY",
                category="preference",
                level=3,
                owner_space="user:u1",
                abstract="User prefers tea",
                overview="",
                content="",
                metadata={},
            ),
        ]

        result = store._collect_sibling_abstracts(node_uri, ctx)

        assert len(result) == 2
        assert "User likes coffee" in result
        assert "User prefers tea" in result

    def test_limits_to_20_siblings(self, store, mock_fs, ctx):
        """Test that collection is capped at 20 siblings."""
        node_uri = "ctx://acme/users/u1/memories/preferences/p1"

        # Mock 30 siblings
        mock_fs.list_children.return_value = [
            f"ctx://acme/users/u1/memories/preferences/p{i}" for i in range(30)
        ]

        # All reads return nodes with abstracts
        mock_fs.read_node.side_effect = [
            ContextNode(
                uri=f"ctx://acme/users/u1/memories/preferences/p{i}",
                context_type="MEMORY",
                category="preference",
                level=3,
                owner_space="user:u1",
                abstract=f"Preference {i}",
                overview="",
                content="",
                metadata={},
            )
            for i in range(30)
        ]

        result = store._collect_sibling_abstracts(node_uri, ctx)

        # Should only collect first 20
        assert len(result) == 20

    def test_silently_ignores_read_failures(self, store, mock_fs, ctx):
        """Test that individual read failures don't break collection."""
        node_uri = "ctx://acme/users/u1/memories/preferences/coffee"

        mock_fs.list_children.return_value = [
            "ctx://acme/users/u1/memories/preferences/coffee",
            "ctx://acme/users/u1/memories/preferences/tea",
        ]

        # First read succeeds, second fails
        mock_fs.read_node.side_effect = [
            ContextNode(
                uri="ctx://acme/users/u1/memories/preferences/coffee",
                context_type="MEMORY",
                category="preference",
                level=3,
                owner_space="user:u1",
                abstract="Valid abstract",
                overview="",
                content="",
                metadata={},
            ),
            Exception("Node not found"),
        ]

        result = store._collect_sibling_abstracts(node_uri, ctx)

        # Should return only the successful read
        assert len(result) == 1
        assert "Valid abstract" in result


class TestRegisterDirectory:
    """Tests for register_directory method."""

    @pytest.fixture
    def store(self, mock_client, mock_fs):
        """Create OutboxStore with mocks."""
        return OutboxStore(client=mock_client, fs=mock_fs)

    @pytest.fixture
    def ctx(self):
        """Create test RequestContext."""
        return RequestContext(
            account_id="acme",
            user_id="u1",
            agent_id="bot",
            session_id="sess",
            trace_id="trace",
        )

    @pytest.fixture
    def node(self):
        """Create test ContextNode."""
        return ContextNode(
            uri="ctx://acme/users/u1/memories/preferences/coffee",
            context_type="MEMORY",
            category="preference",
            level=3,
            owner_space="user:u1",
            abstract="Likes coffee",
            overview="",
            content="",
            metadata={},
        )

    def test_returns_upsert_directory_event(self, store, mock_client, mock_fs, ctx, node):
        """Test register_directory returns UPSERT_DIRECTORY event."""
        mock_fs.list_children.return_value = []

        event = store.register_directory(node, ctx)

        assert event.event_type == "UPSERT_DIRECTORY"
        assert event.uri == "ctx://acme/users/u1/memories/preferences/"
        assert event.status == "PENDING"

    def test_payload_has_directory_uri_with_trailing_slash(self, store, mock_client, mock_fs, ctx, node):
        """Test payload contains directory_uri with trailing slash."""
        mock_fs.list_children.return_value = []

        event = store.register_directory(node, ctx)

        assert event.payload["directory_uri"] == "ctx://acme/users/u1/memories/preferences/"

    def test_register_directory_payload_filters_colon_format(self, store, mock_client, mock_fs, ctx, node):
        """Test payload filters use colon format for owner_space."""
        mock_fs.list_children.return_value = []

        event = store.register_directory(node, ctx)

        assert event.payload["filters"]["account_id"] == "acme"
        assert event.payload["filters"]["owner_space"] == "user:u1"

    def test_register_directory_payload_child_abstracts_is_list(self, store, mock_client, mock_fs, ctx, node):
        """Test payload contains child_abstracts as list."""
        mock_fs.list_children.return_value = [
            "ctx://acme/users/u1/memories/preferences/coffee",
        ]

        mock_fs.read_node.return_value = ContextNode(
            uri="ctx://acme/users/u1/memories/preferences/coffee",
            context_type="MEMORY",
            category="preference",
            level=3,
            owner_space="user:u1",
            abstract="Abstract text",
            overview="",
            content="",
            metadata={},
        )

        event = store.register_directory(node, ctx)

        assert isinstance(event.payload["child_abstracts"], list)
        assert len(event.payload["child_abstracts"]) == 1
        assert event.payload["child_abstracts"][0] == "Abstract text"

    def test_register_directory_caps_siblings_at_20(self, store, mock_client, mock_fs, ctx, node):
        """Test that child_abstracts is capped at 20 items."""
        mock_fs.list_children.return_value = [f"ctx://acme/users/u1/memories/preferences/p{i}" for i in range(30)]

        mock_fs.read_node.side_effect = [
            ContextNode(
                uri=f"ctx://acme/users/u1/memories/preferences/p{i}",
                context_type="MEMORY",
                category="preference",
                level=3,
                owner_space="user:u1",
                abstract=f"Abstract {i}",
                overview="",
                content="",
                metadata={},
            )
            for i in range(30)
        ]

        event = store.register_directory(node, ctx)

        assert len(event.payload["child_abstracts"]) == 20

    def test_writes_to_directory_outbox(self, store, mock_client, mock_fs, ctx, node):
        """Test event is written to parent directory's .outbox/."""
        mock_fs.list_children.return_value = []

        store.register_directory(node, ctx)

        # Verify write to correct path
        expected_path = "/accounts/acme/users/u1/memories/preferences/.outbox/"
        write_calls = [call[0][0] for call in mock_client.write.call_args_list]

        # Find the .outbox write (not mkdir)
        outbox_writes = [p for p in write_calls if ".outbox" in p]
        assert any(expected_path in p for p in outbox_writes)


@pytest.fixture
def mock_client():
    """Create mock AGFSClient."""
    client = Mock()
    client.mkdir.return_value = None
    client.write.return_value = None
    return client


@pytest.fixture
def mock_fs():
    """Create mock ContextFS."""
    fs = Mock()
    return fs
