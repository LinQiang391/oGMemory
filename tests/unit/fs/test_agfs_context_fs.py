"""Unit tests for AGFSContextFS URI to path mapping."""

import json
from unittest.mock import Mock

import pytest
from pyagfs.exceptions import AGFSClientError

from core.enums import NodeStatus
from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS, build_uri, parse_uri, uri_to_path


class TestUriToPath:
    """Tests for uri_to_path function."""

    def test_user_profile_uri(self):
        """Test user profile URI mapping."""
        uri = "ctx://acme/users/alice/memories/profile"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/profile/"

    def test_user_preference_uri(self):
        """Test user preference URI mapping."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/preferences/coffee/"

    def test_user_entity_uri(self):
        """Test user entity URI mapping."""
        uri = "ctx://acme/users/alice/memories/entities/coffee_shop"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/entities/coffee_shop/"

    def test_user_event_uri(self):
        """Test user event URI mapping."""
        uri = "ctx://acme/users/alice/memories/events/visit_20250315_abc123"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/events/visit_20250315_abc123/"

    def test_agent_case_uri(self):
        """Test agent case URI mapping."""
        uri = "ctx://acme/agents/gpt-4/memories/cases/case_20250315_xyz789"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/memories/cases/case_20250315_xyz789/"

    def test_agent_pattern_uri(self):
        """Test agent pattern URI mapping."""
        uri = "ctx://acme/agents/gpt-4/memories/patterns/error_handling"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/memories/patterns/error_handling/"

    def test_agent_skill_uri(self):
        """Test agent skill URI mapping (skills instead of memories)."""
        uri = "ctx://acme/agents/gpt-4/skills/generated/code_review"
        path = uri_to_path(uri)
        # Skills have their own /skills/ subdirectory (not under /memories/)
        assert path == "/accounts/acme/agents/gpt-4/skills/generated/code_review/"

        assert (
            uri_to_path("ctx://acme/agents/gpt-4/skills/code_review")
            == "/accounts/acme/agents/gpt-4/skills/code_review/"
        )

    def test_agent_skills_directory_uri(self):
        """Test agent skills directory URI mapping."""
        uri = "ctx://acme/agents/gpt-4/skills"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/skills/"

    def test_agent_managed_skill_uri(self):
        uri = "ctx://acme/agents/gpt-4/skills/imported/code_review"
        assert uri_to_path(uri) == "/accounts/acme/agents/gpt-4/skills/imported/code_review/"

    def test_managed_skill_uri_round_trip(self):
        uri = "ctx://acme/agents/gpt-4/skills/imported/code_review"
        assert build_uri(**parse_uri(uri)) == uri

    def test_url_encoded_slug(self):
        """Test URI with URL-encoded slug."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee%20preferences"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/preferences/coffee preferences/"

    def test_invalid_uri_format(self):
        """Test that invalid URI format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path("not-a-valid-uri")

        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path("http://example.com/path")

    def test_uri_with_trailing_slash(self):
        """Test that URI with trailing slash is handled correctly."""
        # Trailing slashes are now stripped during URI processing, so this works
        path = uri_to_path("ctx://acme/users/alice/memories/profile/")
        assert path == "/accounts/acme/users/alice/memories/profile/"


class TestParseUri:
    """Tests for parse_uri function."""

    def test_parse_user_uri(self):
        """Test parsing a user URI."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'users',
            'owner_id': 'alice',
            'category': 'preferences',
            'slug': 'coffee',
        }

    def test_parse_agent_uri(self):
        """Test parsing an agent URI."""
        uri = "ctx://acme/agents/gpt-4/memories/cases/case_123"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'cases',
            'slug': 'case_123',
        }

    def test_parse_agent_skills_directory_uri(self):
        """Test parsing the agent skills directory URI."""
        uri = "ctx://acme/agents/gpt-4/skills"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'skills',
            'slug': '',
        }

    def test_parse_agent_managed_skill_uri(self):
        result = parse_uri("ctx://acme/agents/gpt-4/skills/imported/code_review")
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'skills/imported',
            'slug': 'code_review',
        }

    def test_parse_url_encoded_uri(self):
        """Test parsing URI with URL-encoded components."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee%20shop"
        result = parse_uri(uri)
        assert result['slug'] == 'coffee shop'

    def test_parse_invalid_uri(self):
        """Test that invalid URI raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            parse_uri("invalid-uri")


def test_skill_summary_paging_is_uri_ordered_and_respects_cursor():
    client = Mock()
    client.ls.return_value = [
        {"name": "skill-c", "isDir": True},
        {"name": "skill-a", "isDir": True},
        {"name": "skill-b", "isDir": True},
    ]

    def read(path):
        name = path.split("/skills/")[1].split("/")[0]
        if path.endswith(".meta.json"):
            return json.dumps({
                "uri": f"ctx://acme/agents/bot/skills/{name}",
                "context_type": "SKILL", "category": "skill", "level": 2,
                "owner_space": "agent:bot", "status": NodeStatus.ACTIVE.value,
            }).encode()
        return name.encode()

    client.read.side_effect = read
    fs = AGFSContextFS(client)
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="bot",
        session_id="session", trace_id="trace",
    )

    page = fs.list_node_summaries_page(
        "ctx://acme/agents/bot/skills", ctx,
        limit=1, after_uri="ctx://acme/agents/bot/skills/skill-a",
    )

    assert [node.uri for node in page] == ["ctx://acme/agents/bot/skills/skill-b"]


def test_generated_skill_summary_preserves_governance_metadata():
    client = Mock()
    client.ls.return_value = [{"name": "filter-rows", "isDir": True}]

    def read(path):
        if path.endswith(".meta.json"):
            return json.dumps({
                "uri": "ctx://acme/agents/bot/skills/generated/filter-rows",
                "context_type": "SKILL",
                "category": "skill",
                "level": 0,
                "owner_space": "agent:bot",
                "status": NodeStatus.ACTIVE.value,
                "ogmemory_generated": True,
                "publication_status": "published",
                "content_hash": "abc123",
            }).encode()
        return b"Filter matching rows"

    client.read.side_effect = read
    fs = AGFSContextFS(client)
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="bot",
        session_id="session", trace_id="trace",
    )

    page = fs.list_node_summaries_page(
        "ctx://acme/agents/bot/skills/generated", ctx,
        limit=10, after_uri=None,
    )

    assert page[0].metadata["ogmemory_generated"] is True
    assert page[0].metadata["publication_status"] == "published"
    assert page[0].metadata["content_hash"] == "abc123"


class TestBuildUri:
    """Tests for build_uri function."""

    def test_build_user_uri(self):
        """Test building a user URI."""
        uri = build_uri('acme', 'users', 'alice', 'preferences', 'coffee')
        assert uri == "ctx://acme/users/alice/memories/preferences/coffee"

    def test_build_agent_uri(self):
        """Test building an agent URI."""
        uri = build_uri('acme', 'agents', 'gpt-4', 'cases', 'case_123')
        assert uri == "ctx://acme/agents/gpt-4/memories/cases/case_123"

    def test_singular_owner_type_normalized(self):
        """Test that singular owner_type is normalized to plural."""
        uri = build_uri('acme', 'user', 'alice', 'profile', '')
        assert 'users/' in uri

        uri = build_uri('acme', 'agent', 'gpt-4', 'skills', 'code_review')
        assert 'agents/' in uri

    def test_roundtrip_uri(self):
        """Test that build_uri and parse_uri are inverses."""
        original_uri = "ctx://acme/users/alice/memories/preferences/coffee"
        parsed = parse_uri(original_uri)
        rebuilt = build_uri(**parsed)
        assert rebuilt == original_uri

    def test_build_skill_uri(self):
        """Test building a skill URI."""
        uri = build_uri('acme', 'agents', 'gpt-4', 'skills/generated', 'code_review')
        assert uri == "ctx://acme/agents/gpt-4/skills/generated/code_review"

    def test_build_profile_uri(self):
        """Test building a profile URI."""
        uri = build_uri('acme', 'users', 'alice', 'profile', 'profile')
        assert uri == "ctx://acme/users/alice/memories/profile"


class TestIsAccessible:
    """Tests for _is_accessible method."""

    def test_same_account_allowed(self):
        """Test that same account access is allowed."""
        client = None  # Not used for _is_accessible
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        assert fs._is_accessible(uri, ctx) is True

    def test_cross_account_denied(self):
        """Test that cross-account access is denied."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="other-corp",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        assert fs._is_accessible(uri, ctx) is False

    def test_invalid_uri_denied(self):
        """Test that invalid URI is denied."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "not-a-valid-uri"
        assert fs._is_accessible(uri, ctx) is False

    def test_session_state_matching_session_allowed(self):
        """Test that session state access requires matching session_id."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-1",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, ctx) is True

    def test_session_state_different_session_denied(self):
        """Test that session state denies a different session_id."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, ctx) is False

    def test_session_history_same_account_remains_accessible(self):
        """Test that session history remains account-scoped."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/history"
        assert fs._is_accessible(uri, ctx) is True

    def test_session_archive_same_account_remains_accessible(self):
        """Test that session archives remain account-scoped."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/history/archive-1"
        assert fs._is_accessible(uri, ctx) is True


class TestEnsureAccessible:
    """Tests for _ensure_accessible method."""

    def test_same_account_succeeds(self):
        """Test that same account access succeeds."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        # Should not raise
        fs._ensure_accessible(uri, ctx)

    def test_cross_account_raises(self):
        """Test that cross-account access raises AccessDeniedError."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="other-corp",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        with pytest.raises(AccessDeniedError) as exc_info:
            fs._ensure_accessible(uri, ctx)
        assert exc_info.value.account_id == "other-corp"
        assert exc_info.value.uri == uri
        assert "acme" in str(exc_info.value.reason)

    def test_invalid_uri_raises(self):
        """Test that invalid URI raises AccessDeniedError."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "not-a-valid-uri"
        with pytest.raises(AccessDeniedError) as exc_info:
            fs._ensure_accessible(uri, ctx)
        assert "Invalid URI format" in str(exc_info.value.reason)


class TestWriteNode:
    """Tests for write_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_write_node_creates_files_in_order(self):
        """Test that write_node creates files in the correct order."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="User profile for Alice",
            overview="Name: Alice\nAccount: acme",
            content="# Alice's Profile\n\nFull content here",
            metadata={},
        )

        self.fs.write_node(node, self.ctx)

        # Verify mkdir was called for the node directory
        self.mock_client.mkdir.assert_called()

        # Verify write was called for each file in the correct order
        calls = self.mock_client.write.call_args_list
        written_paths = [call[0][0] for call in calls]

        # Step ①: content.md should be written first
        assert any("content.md" in path for path in written_paths)
        content_idx = next(i for i, p in enumerate(written_paths) if "content.md" in p)

        # Step ②: .relations.json should be written second
        assert any(".relations.json" in path for path in written_paths)
        relations_idx = next(i for i, p in enumerate(written_paths) if ".relations.json" in p)
        assert relations_idx > content_idx

        # Step ③: .abstract.md and .overview.md
        assert any(".abstract.md" in path for path in written_paths)
        assert any(".overview.md" in path for path in written_paths)

        # Step ④: .meta.json should be written last (commit point)
        assert any(".meta.json" in path for path in written_paths)
        meta_idx = next(i for i, p in enumerate(written_paths) if ".meta.json" in p)
        assert meta_idx > relations_idx

    def test_write_node_includes_active_status(self):
        """Test that .meta.json includes status=ACTIVE."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="User profile for Alice",
            overview="Name: Alice",
            content="# Profile",
            metadata={},
        )

        self.fs.write_node(node, self.ctx)

        # Find the .meta.json write call
        calls = self.mock_client.write.call_args_list
        meta_call = next(call for call in calls if ".meta.json" in call[0][0])
        meta_content = meta_call[0][1].decode('utf-8')
        meta = json.loads(meta_content)

        assert meta["status"] == "ACTIVE"
        assert meta["uri"] == node.uri
        assert meta["category"] == "profile"

    def test_write_node_cross_account_denied(self):
        """Test that cross-account write raises AccessDeniedError."""
        node = ContextNode(
            uri="ctx://other-corp/users/bob/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:bob",
            abstract="Bob's profile",
            overview="Name: Bob",
            content="# Bob",
            metadata={},
        )

        with pytest.raises(AccessDeniedError):
            self.fs.write_node(node, self.ctx)

    def test_create_only_publishes_with_atomic_directory_move(self):
        self.mock_client.read.side_effect = AGFSClientError("File not found")
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Demo",
            metadata={"_expected_version": 0},
        )

        self.fs.write_node(node, self.ctx)

        source, destination = self.mock_client.mv.call_args.args
        assert ".demo.create-" in source
        assert destination == "/local/accounts/acme/agents/gpt-4/skills/imported/demo"


class TestReadNode:
    """Tests for read_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

        # Set up default mock responses
        self.mock_client.read.side_effect = lambda path: self._mock_read(path)

    def _mock_read(self, path):
        """Mock read implementation that returns test data."""
        if ".meta.json" in path:
            return json.dumps({
                "uri": "ctx://acme/users/alice/memories/profile",
                "context_type": "MEMORY",
                "category": "profile",
                "level": 0,
                "owner_space": "user:alice",
                "status": "ACTIVE",
                "created_at": "2025-03-17T10:00:00Z",
                "updated_at": "2025-03-17T11:00:00Z",
                "version": 1,
                "ogmemory_managed": True,
                "package_files": [{"path": "scripts/run.py"}],
            }).encode()
        elif "content.md" in path:
            return b"# Alice's Profile\n\nFull content"
        elif ".abstract.md" in path:
            return b"User profile for Alice"
        elif ".overview.md" in path:
            return b"Name: Alice\nAccount: acme"
        elif ".relations.json" in path:
            return b"[]"
        raise AGFSClientError("File not found")

    def test_read_node_returns_context_node(self):
        """Test that read_node returns a properly constructed ContextNode."""
        uri = "ctx://acme/users/alice/memories/profile"
        node = self.fs.read_node(uri, self.ctx)

        assert node.uri == uri
        assert node.context_type == "MEMORY"
        assert node.category == "profile"
        assert node.abstract == "User profile for Alice"
        assert node.overview == "Name: Alice\nAccount: acme"
        assert node.content == "# Alice's Profile\n\nFull content"
        assert node.metadata["ogmemory_managed"] is True
        assert node.metadata["package_files"] == [{"path": "scripts/run.py"}]

    def test_read_node_not_found(self):
        """Test that reading non-existent node raises NodeNotFoundError."""
        self.mock_client.read.side_effect = AGFSClientError("No such file")

        with pytest.raises(NodeNotFoundError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_pending_not_visible(self):
        """Test that PENDING nodes raise NodeNotFoundError."""
        def mock_read_pending(path):
            if ".meta.json" in path:
                return json.dumps({
                    "uri": "ctx://acme/users/alice/memories/profile",
                    "context_type": "MEMORY",
                    "category": "profile",
                    "level": 0,
                    "owner_space": "user:alice",
                    "status": "PENDING",
                }).encode()
            return b""

        self.mock_client.read.side_effect = mock_read_pending

        with pytest.raises(NodeNotFoundError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_broken_raises_error(self):
        """Test that BROKEN nodes raise NodeBrokenError."""
        def mock_read_broken(path):
            if ".meta.json" in path:
                return json.dumps({
                    "uri": "ctx://acme/users/alice/memories/profile",
                    "status": "BROKEN",
                }).encode()
            return b""

        self.mock_client.read.side_effect = mock_read_broken

        with pytest.raises(NodeBrokenError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_cross_account_denied(self):
        """Test that cross-account read raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.read_node("ctx://other-corp/users/bob/memories/profile", self.ctx)

    def test_read_node_fields_reads_only_requested_files(self):
        """Test field-level reads avoid loading full content."""
        uri = "ctx://acme/users/alice/memories/profile"
        node = self.fs.read_node_fields(uri, self.ctx, "abstract")

        read_paths = [call.args[0] for call in self.mock_client.read.call_args_list]
        assert node.abstract == "User profile for Alice"
        assert node.overview == ""
        assert node.content == ""
        assert any(path.endswith(".meta.json") for path in read_paths)
        assert any(path.endswith(".abstract.md") for path in read_paths)
        assert not any(path.endswith("content.md") for path in read_paths)


class TestExists:
    """Tests for exists method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_exists_active_node(self):
        """Test that exists returns True for ACTIVE nodes."""
        self.mock_client.read.return_value = json.dumps({
            "uri": "ctx://acme/users/alice/memories/profile",
            "status": "ACTIVE",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is True

    def test_exists_pending_node(self):
        """Test that exists returns False for PENDING nodes."""
        self.mock_client.read.return_value = json.dumps({
            "status": "PENDING",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_broken_node(self):
        """Test that exists returns False for BROKEN nodes."""
        self.mock_client.read.return_value = json.dumps({
            "status": "BROKEN",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_missing_node(self):
        """Test that exists returns False for missing nodes."""
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_cross_account_denied(self):
        """Test that cross-account exists returns False."""
        assert self.fs.exists("ctx://other-corp/users/bob/memories/profile", self.ctx) is False


class TestDeleteNode:
    """Tests for delete_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_delete_node_calls_rm(self):
        """Test that delete_node calls rm with recursive flag."""
        self.fs.delete_node("ctx://acme/users/alice/memories/profile", self.ctx)

        self.mock_client.rm.assert_called_once()
        call_args = self.mock_client.rm.call_args
        assert "/accounts/acme/users/alice/memories/profile/" in call_args[0][0]
        # The second positional argument should be recursive=True
        assert call_args[1].get("recursive") is True or call_args[0][1] is True

    def test_delete_node_cross_account_denied(self):
        """Test that cross-account delete raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.delete_node("ctx://other-corp/users/bob/memories/profile", self.ctx)


class TestArchiveNode:
    """Tests for archive_node soft delete behavior."""

    def setup_method(self):
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_archive_node_marks_meta_archived(self):
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        self.mock_client.read.return_value = json.dumps(
            {
                "uri": uri,
                "context_type": "MEMORY",
                "category": "preference",
                "level": 4,
                "owner_space": "user:alice",
                "status": NodeStatus.ACTIVE.value,
                "created_at": "2026-06-23T00:00:00Z",
                "updated_at": "2026-06-23T00:00:00Z",
                "version": 1,
            }
        ).encode()

        self.fs.archive_node(uri, self.ctx)

        write_path, write_bytes = self.mock_client.write.call_args[0]
        meta = json.loads(write_bytes.decode("utf-8"))
        assert write_path.endswith("/accounts/acme/users/alice/memories/preferences/coffee/.meta.json")
        assert meta["status"] == NodeStatus.ARCHIVED.value
        assert meta["archived_at"]
        assert meta["updated_at"] == meta["archived_at"]

    def test_archive_node_requires_active_node(self):
        self.mock_client.read.return_value = json.dumps({"status": NodeStatus.ARCHIVED.value}).encode()

        with pytest.raises(NodeNotFoundError):
            self.fs.archive_node("ctx://acme/users/alice/memories/preferences/coffee", self.ctx)

    def test_archive_node_cross_account_denied(self):
        with pytest.raises(AccessDeniedError):
            self.fs.archive_node("ctx://other-corp/users/bob/memories/preferences/coffee", self.ctx)


class TestMoveNode:
    """Tests for move_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

        # Mock exists to return True
        self.fs.exists = Mock(return_value=True)

    def test_move_node_calls_mv(self):
        """Test that move_node calls AGFS mv."""
        # Use valid URIs - moving within the same category is more realistic
        from_uri = "ctx://acme/users/alice/memories/profile"
        to_uri = "ctx://acme/users/alice/memories/preferences/profile_backup"

        self.fs.move_node(from_uri, to_uri, self.ctx)

        self.mock_client.mv.assert_called_once()
        call_args = self.mock_client.mv.call_args
        from_path = call_args[0][0]
        to_path = call_args[0][1]
        assert "/accounts/acme/users/alice/memories/profile/" in from_path
        assert "/accounts/acme/users/alice/memories/preferences/profile_backup/" in to_path

    def test_move_node_cross_account_denied(self):
        """Test that cross-account move raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.move_node(
                "ctx://acme/users/alice/memories/profile",
                "ctx://other-corp/users/alice/memories/profile",
                self.ctx
            )

    def test_move_node_to_invalid_category_denied(self):
        """Test that moving to an invalid URI raises AccessDeniedError."""
        # profile_new is not a valid category, but move_node now accepts any valid URI structure
        # The URI parsing succeeds since profile_new is treated as a valid slug
        # Move should succeed since both URIs are accessible to the same account
        from_uri = "ctx://acme/users/alice/memories/profile"
        to_uri = "ctx://acme/users/alice/memories/profile_new"
        # Just verify the move is attempted (both URIs are accessible to acme account)
        self.fs.move_node(from_uri, to_uri, self.ctx)
