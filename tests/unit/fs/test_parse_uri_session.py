"""Unit tests for session URI parsing support."""

import pytest

from fs.agfs_adapter.agfs_context_fs import parse_uri, uri_to_path


class TestParseSessionUri:
    """Tests for parse_uri function with session archive patterns."""

    def test_parse_session_archive_uri(self):
        """Test parsing a session archive URI (Pattern 6)."""
        uri = "ctx://acme/sessions/session-123/history/archive-456"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'sessions',
            'owner_id': 'session-123',
            'category': 'history',
            'slug': 'archive-456',
        }

    def test_parse_session_history_directory_uri(self):
        """Test parsing a session history directory URI (Pattern 7)."""
        uri = "ctx://acme/sessions/session-123/history"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'sessions',
            'owner_id': 'session-123',
            'category': 'history',
            'slug': '',
        }

    def test_parse_session_history_directory_uri_with_trailing_slash(self):
        """Test parsing session history directory URI with trailing slash."""
        uri = "ctx://acme/sessions/session-123/history/"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'sessions',
            'owner_id': 'session-123',
            'category': 'history',
            'slug': '',
        }

    def test_parse_session_archive_with_complex_ids(self):
        """Test parsing session archive with UUID-like IDs."""
        uri = "ctx://mycorp/sessions/abc-123-def-456/history/2025-04-03T10:30:00Z"
        result = parse_uri(uri)
        assert result == {
            'account': 'mycorp',
            'owner_type': 'sessions',
            'owner_id': 'abc-123-def-456',
            'category': 'history',
            'slug': '2025-04-03T10:30:00Z',
        }

    def test_parse_invalid_uri_still_raises(self):
        """Test that invalid URI format still raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            parse_uri("not-a-valid-uri")

        with pytest.raises(ValueError, match="Invalid URI format"):
            parse_uri("http://example.com/path")


class TestSessionUriToPath:
    """Tests for uri_to_path function with session archive patterns."""

    def test_session_archive_uri_to_path(self):
        """Test session archive URI to path mapping (Pattern 6)."""
        uri = "ctx://acme/sessions/session-123/history/archive-456"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/sessions/session-123/history/archive-456/"

    def test_session_history_directory_uri_to_path(self):
        """Test session history directory URI to path mapping (Pattern 7)."""
        uri = "ctx://acme/sessions/session-123/history"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/sessions/session-123/history/"

    def test_session_history_directory_uri_with_trailing_slash_to_path(self):
        """Test session history directory with trailing slash to path mapping."""
        uri = "ctx://acme/sessions/session-123/history/"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/sessions/session-123/history/"
