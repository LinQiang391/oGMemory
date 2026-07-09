"""Tests for directory URI trailing slash consistency.

This test file verifies that directory URIs are consistently handled
throughout the codebase to prevent the directory event processing
chain from breaking.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext
from retrieval.seed_retriever import SeedRetriever
from commit.outbox_store import OutboxStore
from index.directory_summarizer import is_directory_uri
from unittest.mock import Mock


@pytest.mark.integration
class TestDirectoryURIConsistency:
    """Tests for consistent directory URI handling."""

    def test_seed_retriever_root_uris_have_trailing_slash(self):
        """Root URIs from SeedRetriever should have trailing slashes."""
        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        # Test without context_type filter
        root_uris = SeedRetriever._get_root_uris(None, ctx)

        for uri in root_uris:
            assert uri.endswith("/"), \
                f"Root URI should end with '/': {uri}"
            assert is_directory_uri(uri), \
                f"Root URI should be recognized as directory: {uri}"

    def test_seed_retriever_memory_uris_have_trailing_slash(self):
        """MEMORY context type URIs should have trailing slashes."""
        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        root_uris = SeedRetriever._get_root_uris("MEMORY", ctx)

        for uri in root_uris:
            assert uri.endswith("/"), \
                f"MEMORY root URI should end with '/': {uri}"
            assert is_directory_uri(uri), \
                f"MEMORY root URI should be recognized as directory: {uri}"

    def test_seed_retriever_skill_uris_have_trailing_slash(self):
        """SKILL context type URIs should have trailing slashes."""
        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        root_uris = SeedRetriever._get_root_uris("SKILL", ctx)

        for uri in root_uris:
            assert uri.endswith("/"), \
                f"SKILL root URI should end with '/': {uri}"
            assert is_directory_uri(uri), \
                f"SKILL root URI should be recognized as directory: {uri}"

    def test_seed_retriever_resource_uris_have_trailing_slash(self):
        """RESOURCE context type URIs should have trailing slashes."""
        ctx = RequestContext(
            account_id="test-acct",
            user_id="user-1",
            agent_id="agent-1",
            session_id="sess-1",
            trace_id="trace-1",
        )

        root_uris = SeedRetriever._get_root_uris("RESOURCE", ctx)

        for uri in root_uris:
            assert uri.endswith("/"), \
                f"RESOURCE root URI should end with '/': {uri}"
            assert is_directory_uri(uri), \
                f"RESOURCE root URI should be recognized as directory: {uri}"

    def test_outbox_store_directory_uri_has_trailing_slash(self):
        """OutboxStore should preserve trailing slash in directory URIs."""
        from commit.outbox_store import OutboxStore

        mock_client = Mock()
        mock_client.mkdir.return_value = None
        mock_fs = Mock()
        mock_fs.list_children.return_value = []

        store = OutboxStore(client=mock_client, fs=mock_fs)
        """OutboxStore should preserve trailing slash in directory URIs."""
        from commit.outbox_store import OutboxStore

        mock_client = Mock()
        mock_fs = Mock()
        mock_fs.list_children.return_value = []

        store = OutboxStore(client=mock_client, fs=mock_fs)

        # Test node URI to directory URI conversion
        node_uri = "ctx://test-acct/users/user-1/memories/preferences/coffee"
        dir_uri = store._node_uri_to_directory_uri(node_uri)

        assert dir_uri.endswith("/"), \
            f"Directory URI should end with '/': {dir_uri}"
        assert is_directory_uri(dir_uri), \
            f"Directory URI should be recognized as directory: {dir_uri}"

    def test_is_directory_uri_consistent_detection(self):
        """is_directory_uri should consistently detect directory URIs."""
        # Directory URIs with trailing slash
        assert is_directory_uri("ctx://test/memories/")
        assert is_directory_uri("ctx://test/users/u1/memories/preferences/")
        assert is_directory_uri("/accounts/test/users/u1/memories/preferences/")

        # Non-directory URIs without trailing slash
        assert not is_directory_uri("ctx://test/memories")
        assert not is_directory_uri("ctx://test/users/u1/memories/preferences/coffee")
        assert not is_directory_uri("/accounts/test/users/u1/memories/preferences/coffee")
