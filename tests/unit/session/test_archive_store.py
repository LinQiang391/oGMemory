"""Tests for SessionArchiveStore."""

from __future__ import annotations

import json
import threading
from unittest.mock import Mock, call
from datetime import datetime, timezone

import pytest

from core.models import RequestContext, ContextNode, Role
from session.archive_store import SessionArchiveStore
from session.models import ArchiveEntry, ArchiveWriteResult


def test_archive_store_initializes_reentrant_metadata_lock(archive_store):
    lock = archive_store._metadata_lock

    assert isinstance(lock, threading.RLock().__class__)
    with lock:
        with lock:
            pass


def test_write_archive_success(archive_store, mock_context, sample_messages, mock_fs):
    """Test successful archive write."""
    result = archive_store.write_archive(
        session_id="test_session",
        overview="Test overview",
        abstract="Test abstract",
        messages=sample_messages,
        ctx=mock_context,
        archive_id="test_archive_001",
    )

    assert result.success is True
    assert result.archive_id == "test_archive_001"
    assert result.session_id == "test_session"
    assert "ctx://test_account/sessions/test_session/history/test_archive_001" in result.uri
    assert result.error is None

    # Verify ContextFS.write_node was called with correct ContextNode
    mock_fs.write_node.assert_called_once()
    node_arg = mock_fs.write_node.call_args[0][0]
    assert isinstance(node_arg, ContextNode)
    assert node_arg.context_type == "RESOURCE"
    assert node_arg.category == "history"
    assert node_arg.owner_space == "session:test_session"
    assert node_arg.abstract == "Test abstract"
    assert node_arg.overview == "Test overview"

    # Verify messages stored as JSON in content
    stored_messages = json.loads(node_arg.content)
    assert stored_messages == sample_messages

    # Verify metadata
    assert node_arg.metadata["session_id"] == "test_session"
    assert node_arg.metadata["archive_id"] == "test_archive_001"
    assert node_arg.metadata["message_count"] == 4


def test_write_archive_merges_extra_metadata(
    archive_store,
    mock_context,
    sample_messages,
    mock_fs,
):
    result = archive_store.write_archive(
        session_id="test_session",
        overview="overview",
        abstract="abstract",
        messages=sample_messages,
        ctx=mock_context,
        archive_id="archive_quality",
        metadata={
            "compression_quality": {"entity_retention_ratio": 1.0},
            "source_archive_ids": ["a1", "a2"],
        },
    )

    assert result.success is True
    node = mock_fs.write_node.call_args.args[0]
    assert node.metadata["archive_id"] == "archive_quality"
    assert node.metadata["compression_quality"] == {"entity_retention_ratio": 1.0}
    assert node.metadata["source_archive_ids"] == ["a1", "a2"]


def test_write_archive_auto_generates_id(archive_store, mock_context, sample_messages):
    """Test that archive_id is auto-generated if not provided."""
    result = archive_store.write_archive(
        session_id="test_session",
        overview="Test overview",
        abstract="Test abstract",
        messages=sample_messages,
        ctx=mock_context,
    )

    assert result.success is True
    assert result.archive_id is not None
    assert len(result.archive_id) > 0  # Should have generated an ID


def test_list_archives_empty(archive_store, mock_context, mock_fs):
    """Test listing archives when none exist."""
    mock_fs.list_children.return_value = []

    entries = archive_store.list_archives("test_session", mock_context)

    assert entries == []
    mock_fs.list_children.assert_called_once()


def test_list_archives_with_entries(archive_store, mock_context, mock_fs):
    """Test listing archives with existing entries."""
    mock_fs.list_children.return_value = [
        "ctx://test_account/sessions/test_session/history/archive_001",
        "ctx://test_account/sessions/test_session/history/archive_002",
    ]

    # Mock read_node to return archive data
    def mock_read_node(uri, ctx):
        archive_id = uri.split("/")[-1]
        return ContextNode(
            uri=uri,
            context_type="RESOURCE",
            category="history",
            level=0,
            owner_space="session:test_session",
            abstract=f"Abstract for {archive_id}",
            overview=f"Overview for {archive_id}",
            content='[]',
            metadata={"created_at": "2025-04-03T10:00:00Z"},
        )

    mock_fs.read_node = Mock(side_effect=mock_read_node)

    entries = archive_store.list_archives("test_session", mock_context)

    assert len(entries) == 2
    assert entries[0].archive_id == "archive_001"
    assert entries[0].abstract == "Abstract for archive_001"
    assert entries[0].overview == "Overview for archive_001"
    assert entries[0].messages == []  # List shouldn't include messages
    assert entries[1].archive_id == "archive_002"


def test_list_archives_excludes_merged_entries(archive_store, mock_context, mock_fs):
    mock_fs.list_children.return_value = [
        "ctx://test_account/sessions/test_session/history/archive_001",
        "ctx://test_account/sessions/test_session/history/archive_002",
    ]

    def mock_read_node(uri, ctx):
        archive_id = uri.split("/")[-1]
        metadata = {"created_at": "2025-04-03T10:00:00Z"}
        if archive_id == "archive_001":
            metadata["status"] = "MERGED"
        return ContextNode(
            uri=uri,
            context_type="RESOURCE",
            category="history",
            level=0,
            owner_space="session:test_session",
            abstract=f"Abstract for {archive_id}",
            overview=f"Overview for {archive_id}",
            content="[]",
            metadata=metadata,
        )

    mock_fs.read_node = Mock(side_effect=mock_read_node)

    entries = archive_store.list_archives("test_session", mock_context)

    assert [entry.archive_id for entry in entries] == ["archive_002"]


def test_read_archive_not_found(archive_store, mock_context, mock_fs):
    """Test reading a non-existent archive."""
    mock_fs.exists.return_value = False

    entry = archive_store.read_archive("test_session", "missing_archive", mock_context)

    assert entry is None


def test_read_archive_found(archive_store, mock_context, mock_fs):
    """Test reading an existing archive."""
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="Test abstract",
        overview="Test overview",
        content=json.dumps(test_messages),
        metadata={"created_at": "2025-04-03T10:00:00Z"},
    )

    entry = archive_store.read_archive("test_session", "archive_001", mock_context)

    assert entry is not None
    assert entry.archive_id == "archive_001"
    assert entry.session_id == "test_session"
    assert entry.abstract == "Test abstract"
    assert entry.overview == "Test overview"
    assert entry.messages == test_messages
    assert entry.created_at == "2025-04-03T10:00:00Z"


def test_read_archive_returns_merged_archive_for_direct_lookup(archive_store, mock_context, mock_fs):
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="Test abstract",
        overview="Test overview",
        content="[]",
        metadata={"status": "MERGED"},
    )

    entry = archive_store.read_archive("test_session", "archive_001", mock_context)

    assert entry is not None
    assert entry.archive_id == "archive_001"
    assert entry.metadata["status"] == "MERGED"


def test_read_archive_abstract_not_found(archive_store, mock_context, mock_fs):
    """Test reading abstract from non-existent archive."""
    mock_fs.exists.return_value = False

    abstract = archive_store.read_archive_abstract("test_session", "missing_archive", mock_context)

    assert abstract is None


def test_read_archive_abstract_found(archive_store, mock_context, mock_fs):
    """Test reading abstract from existing archive."""
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="Test abstract",
        overview="Test overview",
        content='[]',
        metadata={},
    )

    abstract = archive_store.read_archive_abstract("test_session", "archive_001", mock_context)

    assert abstract == "Test abstract"


def test_read_archive_abstract_returns_merged_archive_for_direct_lookup(
    archive_store,
    mock_context,
    mock_fs,
):
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="Test abstract",
        overview="Test overview",
        content="[]",
        metadata={"status": "MERGED"},
    )

    abstract = archive_store.read_archive_abstract(
        "test_session",
        "archive_001",
        mock_context,
    )

    assert abstract == "Test abstract"


def test_session_uri_to_path():
    """Test URI to path conversion function."""
    from session.archive_store import session_uri_to_path

    path = session_uri_to_path("acme", "sess123", "arc456")

    assert path == "/accounts/acme/sessions/sess123/history/arc456/"


def test_archive_entry_dataclass():
    """Test ArchiveEntry dataclass creation."""
    entry = ArchiveEntry(
        archive_id="arc001",
        session_id="sess001",
        overview="Test overview",
        abstract="Test abstract",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert entry.archive_id == "arc001"
    assert entry.session_id == "sess001"
    assert entry.overview == "Test overview"
    assert entry.abstract == "Test abstract"
    assert len(entry.messages) == 1
    assert entry.created_at is not None  # Should be auto-generated


def test_archive_write_result_dataclass():
    """Test ArchiveWriteResult dataclass creation."""
    result = ArchiveWriteResult(
        archive_id="arc001",
        session_id="sess001",
        uri="ctx://test/sessions/sess001/history/arc001",
        success=True,
    )

    assert result.success is True
    assert result.archive_id == "arc001"
    assert result.error is None
    assert result.created_at is not None  # Should be auto-generated


def test_archive_write_result_failure():
    """Test ArchiveWriteResult with failure."""
    result = ArchiveWriteResult(
        archive_id="arc001",
        session_id="sess001",
        uri="ctx://test/sessions/sess001/history/arc001",
        success=False,
        error="Storage unavailable",
    )

    assert result.success is False
    assert result.error == "Storage unavailable"


def test_delete_archive_calls_context_fs_delete_node(archive_store, mock_context, mock_fs):
    ok = archive_store.delete_archive(
        "test_session",
        "archive_001",
        mock_context,
    )

    assert ok is True
    mock_fs.delete_node.assert_called_once_with(
        "ctx://test_account/sessions/test_session/history/archive_001",
        mock_context,
    )


def test_delete_archive_returns_false_on_failure(archive_store, mock_context, mock_fs):
    mock_fs.delete_node.side_effect = RuntimeError("delete failed")

    ok = archive_store.delete_archive(
        "test_session",
        "archive_001",
        mock_context,
    )

    assert ok is False


def test_admin_session_cleanup_deletes_ownerless_legacy_archives(mock_fs):
    store = SessionArchiveStore(mock_fs)
    ctx = RequestContext(
        "test_account", "test_user", "test_agent", "test_session", "trace", Role.ADMIN,
    )
    uris = [
        "ctx://test_account/sessions/test_session/history/legacy",
        "ctx://test_account/sessions/test_session/history/owned",
        "ctx://test_account/sessions/test_session/history/other",
    ]
    metadata = {
        uris[0]: {},
        uris[1]: {"owner_user_id": "test_user", "owner_agent_id": "test_agent"},
        uris[2]: {"owner_user_id": "other", "owner_agent_id": "other-agent"},
    }
    mock_fs.list_children.return_value = uris
    mock_fs.read_node.side_effect = lambda uri, _ctx: ContextNode(
        uri=uri,
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="",
        overview="",
        content="[]",
        metadata=metadata[uri],
    )

    assert store.delete_session_archives("test_session", ctx) == 2
    assert mock_fs.delete_node.call_args_list == [call(uris[0], ctx), call(uris[1], ctx)]


def test_member_session_cleanup_does_not_delete_ownerless_legacy_archives(mock_fs):
    store = SessionArchiveStore(mock_fs)
    ctx = RequestContext(
        "test_account", "test_user", "test_agent", "test_session", "trace", Role.MEMBER,
    )
    uri = "ctx://test_account/sessions/test_session/history/legacy"
    mock_fs.list_children.return_value = [uri]
    mock_fs.read_node.return_value = ContextNode(
        uri=uri,
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="",
        overview="",
        content="[]",
        metadata={},
    )

    assert store.delete_session_archives("test_session", ctx) == 0
    mock_fs.delete_node.assert_not_called()


def test_session_owner_lookup_propagates_backend_failure(mock_fs):
    store = SessionArchiveStore(mock_fs)
    mock_fs.list_children.side_effect = OSError("archive backend unavailable")

    with pytest.raises(OSError, match="archive backend unavailable"):
        store.get_session_owner("test_session", RequestContext(
            "test_account", "test_user", "test_agent", "test_session", "trace",
        ))


def test_mark_archive_merged_rewrites_metadata(archive_store, mock_context, mock_fs):
    node = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="old abstract",
        overview="old overview",
        content="[]",
        metadata={"archive_id": "archive_001", "session_id": "test_session"},
    )
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = node

    ok = archive_store.mark_archive_merged(
        "test_session",
        "archive_001",
        mock_context,
        merged_into="merged_001",
    )

    assert ok is True
    rewritten = mock_fs.write_node.call_args.args[0]
    assert rewritten.metadata["merged_into"] == "merged_001"
    assert rewritten.metadata["archive_status"] == "MERGED"
    assert rewritten.metadata["merged_at"]


def test_mark_archive_merged_restores_required_metadata(
    archive_store,
    mock_context,
    mock_fs,
):
    node = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="old abstract",
        overview="old overview",
        content="[]",
        metadata={"message_count": 7},
    )
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = node

    ok = archive_store.mark_archive_merged(
        "test_session",
        "archive_001",
        mock_context,
        merged_into="merged_001",
    )

    assert ok is True
    rewritten = mock_fs.write_node.call_args.args[0]
    assert rewritten.metadata["archive_id"] == "archive_001"
    assert rewritten.metadata["session_id"] == "test_session"
    assert rewritten.metadata["message_count"] == 7
    assert rewritten.metadata["archive_status"] == "MERGED"
    assert rewritten.metadata["merged_into"] == "merged_001"
    assert rewritten.metadata["merged_at"]


def test_unmark_archive_merged_restores_visible_metadata(
    archive_store,
    mock_context,
    mock_fs,
):
    node = ContextNode(
        uri="ctx://test_account/sessions/test_session/history/archive_001",
        context_type="RESOURCE",
        category="history",
        level=0,
        owner_space="session:test_session",
        abstract="old abstract",
        overview="old overview",
        content="[]",
        metadata={
            "archive_id": "archive_001",
            "status": "MERGED",
            "merged_into": "merged_001",
            "merged_at": "now",
        },
    )
    mock_fs.exists.return_value = True
    mock_fs.read_node.return_value = node

    ok = archive_store.unmark_archive_merged(
        "test_session",
        "archive_001",
        mock_context,
        "merged_001",
    )

    assert ok is True
    rewritten = mock_fs.write_node.call_args.args[0]
    assert "status" not in rewritten.metadata
    assert "merged_into" not in rewritten.metadata
    assert "merged_at" not in rewritten.metadata
