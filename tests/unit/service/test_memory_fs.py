"""Unit tests for MemoryFS - file system-like memory browsing interface."""

import pytest
from unittest.mock import Mock, MagicMock

from core.models import RequestContext, ContextNode
from core.enums import NodeStatus
from service.memory_fs import MemoryFS
from core.errors import AccessDeniedError, NodeNotFoundError


@pytest.fixture
def mock_fs():
    """Create a mock ContextFS."""
    fs = Mock()

    # Mock exists
    def exists_side_effect(uri, ctx):
        if ctx.account_id != "test-acct":
            return False
        return "/profile" in uri or "/preferences" in uri

    fs.exists.side_effect = exists_side_effect

    # Mock list_children
    def list_children_side_effect(uri, ctx):
        if ctx.account_id != "test-acct":
            raise AccessDeniedError(uri, ctx.account_id, "Account mismatch")

        # Check for full URI patterns
        if uri == "ctx://test-acct/users/alice/memories":
            return [
                "ctx://test-acct/users/alice/memories/profile",
                "ctx://test-acct/users/alice/memories/preferences",
                "ctx://test-acct/users/alice/memories/entities",
            ]
        elif uri == "ctx://test-acct/users/alice/memories/preferences":
            return [
                "ctx://test-acct/users/alice/memories/preferences/coding_style",
                "ctx://test-acct/users/alice/memories/preferences/tools",
            ]
        elif uri == "ctx://test-acct/users/alice/memories/entities":
            return [
                "ctx://test-acct/users/alice/memories/entities/company_a",
            ]
        return []

    fs.list_children.side_effect = list_children_side_effect

    # Mock read_node
    def read_node_side_effect(uri, ctx):
        if ctx.account_id != "test-acct":
            raise AccessDeniedError(uri, ctx.account_id, "Account mismatch")

        if "profile" in uri:
            return ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="profile",
                level=1,
                owner_space="user:alice",
                abstract="Alice is a software engineer",
                overview="## Profile\nAlice works at TechCorp",
                content="Full profile content here...",
                metadata={
                    "created_at": "2026-03-19T10:00:00Z",
                    "updated_at": "2026-03-19T10:00:00Z",
                    "version": 1,
                },
            )
        elif "coding_style" in uri:
            return ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="preference",
                level=1,
                owner_space="user:alice",
                abstract="Prefers Python and TypeScript",
                overview="## Coding Style\n- Python for scripts\n- TypeScript for web",
                content="Detailed coding preferences...",
                metadata={
                    "created_at": "2026-03-19T11:00:00Z",
                    "updated_at": "2026-03-19T11:00:00Z",
                    "version": 1,
                },
            )
        raise NodeNotFoundError(uri)

    fs.read_node.side_effect = read_node_side_effect

    return fs


@pytest.fixture
def context():
    """Create a test RequestContext."""
    return RequestContext(
        account_id="test-acct",
        user_id="alice",
        agent_id="agent-001",
        session_id="sess-001",
        trace_id="trace-001",
    )


class TestMemoryFSInit:
    """Tests for MemoryFS initialization."""

    def test_init_with_fs(self, mock_fs):
        """MemoryFS should initialize with a ContextFS."""
        fs = MemoryFS(mock_fs)
        assert fs._fs is mock_fs


class TestPathConversion:
    """Tests for path to URI conversion."""

    def test_user_path_to_uri(self, context):
        """Convert user memory path to URI."""
        fs = MemoryFS(Mock())
        uri = fs._path_to_uri("/users/alice/memories/profile", context)
        assert uri == "ctx://test-acct/users/alice/memories/profile"

    def test_agent_path_to_uri(self, context):
        """Convert agent memory path to URI."""
        fs = MemoryFS(Mock())
        uri = fs._path_to_uri("/agents/bob/memories/cases/bug123", context)
        assert uri == "ctx://test-acct/agents/bob/memories/cases/bug123"

    def test_full_uri_passthrough_same_account(self, context):
        """Full URI with same account should pass through unchanged."""
        fs = MemoryFS(Mock())
        uri = fs._path_to_uri("ctx://test-acct/users/alice/memories/profile", context)
        assert uri == "ctx://test-acct/users/alice/memories/profile"

    def test_full_uri_cross_account_rejected(self, context):
        """Full URI with different account should be rejected (security fix C-1)."""
        from core.errors import AccessDeniedError
        fs = MemoryFS(Mock())
        with pytest.raises(AccessDeniedError, match="does not match context account"):
            fs._path_to_uri("ctx://other-acct/users/alice/memories/profile", context)

    def test_uri_to_path(self):
        """Convert URI to display path."""
        fs = MemoryFS(Mock())
        path = fs._uri_to_path("ctx://test-acct/users/alice/memories/profile")
        assert path == "/test-acct/users/alice/memories/profile"


class TestMemoryFSList:
    """Tests for list operation."""

    def test_list_root_categories(self, mock_fs, context):
        """List user's memory categories."""
        fs = MemoryFS(mock_fs)
        result = fs.list("/users/alice/memories", context)

        assert len(result) == 3
        categories = [r["name"] for r in result]
        assert "profile" in categories
        assert "preferences" in categories
        assert "entities" in categories

    def test_list_category_items(self, mock_fs, context):
        """List items within a category."""
        fs = MemoryFS(mock_fs)
        result = fs.list("/users/alice/memories/preferences", context)

        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "coding_style" in names
        assert "tools" in names

    def test_list_memories_default_path(self, mock_fs, context):
        """list_memories should default to user's memories."""
        fs = MemoryFS(mock_fs)
        result = fs.list_memories(context)

        assert len(result) == 3
        assert any(r["category"] == "profile" for r in result)


class TestMemoryFSStat:
    """Tests for stat operation."""

    def test_stat_profile(self, mock_fs, context):
        """Get metadata for profile node."""
        fs = MemoryFS(mock_fs)
        info = fs.stat("/users/alice/memories/profile", context)

        assert info["category"] == "profile"
        assert info["size"] > 0
        assert info["has_overview"] is True
        assert info["has_content"] is True
        assert "created_at" in info

    def test_stat_returns_path(self, mock_fs, context):
        """stat should return both URI and display path."""
        fs = MemoryFS(mock_fs)
        info = fs.stat("/users/alice/memories/profile", context)

        assert "uri" in info
        assert "path" in info
        assert info["path"].startswith("/")


class TestMemoryFSRead:
    """Tests for read operations."""

    def test_read_abstract(self, mock_fs, context):
        """Read just the abstract."""
        fs = MemoryFS(mock_fs)
        abstract = fs.read_abstract("/users/alice/memories/profile", context)

        assert abstract == "Alice is a software engineer"

    def test_read_overview(self, mock_fs, context):
        """Read the overview."""
        fs = MemoryFS(mock_fs)
        overview = fs.read_overview("/users/alice/memories/profile", context)

        assert "## Profile" in overview

    def test_read_abstract_by_uri(self, mock_fs, context):
        """Can read abstract using full URI."""
        fs = MemoryFS(mock_fs)
        abstract = fs.read_abstract("ctx://test-acct/users/alice/memories/profile", context)

        assert abstract == "Alice is a software engineer"


class TestMemoryFSExists:
    """Tests for exists operation."""

    def test_exists_true(self, mock_fs, context):
        """Check existing node returns True."""
        fs = MemoryFS(mock_fs)
        assert fs.exists("/users/alice/memories/profile", context) is True

    def test_exists_false(self, mock_fs, context):
        """Check non-existent node returns False."""
        fs = MemoryFS(mock_fs)
        assert fs.exists("/users/alice/memories/nonexistent", context) is False


class TestMemoryFSAccessControl:
    """Tests for access control enforcement."""

    def test_cross_account_list_denied(self, mock_fs):
        """List with wrong account should be denied."""
        fs = MemoryFS(mock_fs)
        wrong_ctx = RequestContext(
            account_id="other-acct",
            user_id="alice",
            agent_id="agent-001",
            session_id="sess",
            trace_id="trace",
        )

        with pytest.raises(AccessDeniedError):
            fs.list("/users/alice/memories", wrong_ctx)

    def test_cross_account_stat_denied(self, mock_fs):
        """Stat with wrong account should be denied."""
        fs = MemoryFS(mock_fs)
        wrong_ctx = RequestContext(
            account_id="other-acct",
            user_id="alice",
            agent_id="agent-001",
            session_id="sess",
            trace_id="trace",
        )

        with pytest.raises(AccessDeniedError):
            fs.stat("/users/alice/memories/profile", wrong_ctx)


class TestMemoryFSSummary:
    """Tests for summary operations."""

    def test_get_summary(self, mock_fs, context):
        """Get memory summary."""
        fs = MemoryFS(mock_fs)
        summary = fs.get_summary(context)

        assert "user_memories" in summary
        assert "total_nodes" in summary
        assert summary["total_nodes"] > 0

    def test_get_categories(self, mock_fs, context):
        """Get available categories."""
        fs = MemoryFS(mock_fs)
        categories = fs.get_categories(context, owner_type="user")

        assert "profile" in categories
        assert "preferences" in categories
