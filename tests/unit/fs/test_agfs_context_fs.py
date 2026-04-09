"""Unit tests for AGFSContextFS URI to path mapping."""

import pytest

from fs.agfs_adapter.agfs_context_fs import (
    uri_to_path,
    parse_uri,
    build_uri,
    AGFSContextFS,
)
from core.errors import AccessDeniedError
from core.models import RequestContext


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
        uri = "ctx://acme/agents/gpt-4/skills/code_review"
        path = uri_to_path(uri)
        # Skills have their own /skills/ subdirectory (not under /memories/)
        assert path == "/accounts/acme/agents/gpt-4/skills/code_review/"

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
        # Our regex doesn't allow trailing slash, so this should fail
        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path("ctx://acme/users/alice/memories/profile/")


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

    def test_parse_url_encoded_uri(self):
        """Test parsing URI with URL-encoded components."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee%20shop"
        result = parse_uri(uri)
        assert result['slug'] == 'coffee shop'

    def test_parse_invalid_uri(self):
        """Test that invalid URI raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            parse_uri("invalid-uri")


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
        uri = build_uri('acme', 'agents', 'gpt-4', 'skills', 'code_review')
        assert uri == "ctx://acme/agents/gpt-4/skills/code_review"

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
