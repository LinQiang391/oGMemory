"""Tests for MemoryArchiver."""

from datetime import datetime, timezone, timedelta

import pytest
from unittest.mock import Mock

from core.models import RequestContext, SeedHit
from index.memory_archiver import (
    MemoryArchiver,
    ArchivalCandidate,
    ArchivalResult,
    _build_archive_uri,
    _build_restore_uri,
)


@pytest.fixture
def mock_context():
    """Mock RequestContext for testing."""
    return RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )


@pytest.fixture
def mock_fs():
    """Mock ContextFS for testing."""
    fs = Mock()
    fs.move_node = Mock()
    fs.exists = Mock(return_value=True)
    return fs


@pytest.fixture
def mock_vector_index():
    """Mock VectorIndex for testing."""
    index = Mock()
    return index


@pytest.fixture
def memory_archiver(mock_fs, mock_vector_index):
    """MemoryArchiver instance with mocked dependencies."""
    return MemoryArchiver(
        fs=mock_fs,
        vector_index=mock_vector_index,
        threshold=0.1,
        min_age_days=7,
    )


@pytest.fixture
def sample_seed_hits():
    """Sample SeedHit objects for testing."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=30)  # 30 days old

    return [
        SeedHit(
            uri="ctx://test_account/users/test_user/memories/preferences/coding_style",
            score=0.5,
            active_count=0,  # Low activity
            updated_at=old_time,
            context_type="MEMORY",
            category="preference",
            owner_space="user:test_user",
        ),
        SeedHit(
            uri="ctx://test_account/users/test_user/memories/entities/python",
            score=0.6,
            active_count=5,  # Higher activity
            updated_at=now - timedelta(days=1),  # Recent
            context_type="MEMORY",
            category="entity",
            owner_space="user:test_user",
        ),
    ]


def test_hotness_score_static_method():
    """Test the static hotness_score method."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=30)

    # Low activity, old memory = low hotness
    score = MemoryArchiver.hotness_score(
        active_count=0,
        updated_at=old_time,
        now=now,
        half_life_days=7.0,
    )
    assert score < 0.1

    # High activity, recent memory = high hotness
    score = MemoryArchiver.hotness_score(
        active_count=10,
        updated_at=now - timedelta(hours=1),
        now=now,
        half_life_days=7.0,
    )
    assert score > 0.5


def test_scan_finds_cold_memories(memory_archiver, mock_vector_index, sample_seed_hits):
    """Test that scan identifies cold memories."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    mock_vector_index.search_by_vector.return_value = sample_seed_hits

    scope_uri = "ctx://test_account/users/test_user/memories/"
    candidates = memory_archiver.scan(scope_uri, ctx)

    # Should find the cold memory (low activity, old)
    assert len(candidates) >= 1
    # Should be sorted by score ascending (coldest first)
    if len(candidates) > 1:
        assert candidates[0].score <= candidates[1].score


def test_scan_respects_min_age(memory_archiver, mock_vector_index, mock_context):
    now = datetime.now(timezone.utc)
    recent_hit = SeedHit(
        uri="ctx://test_account/users/test_user/memories/recent",
        score=0.5,
        active_count=0,
        updated_at=now - timedelta(days=1),  # Too recent
        context_type="MEMORY",
        category="preference",
        owner_space="user:test_user",
    )

    mock_vector_index.search_by_vector.return_value = [recent_hit]

    scope_uri = "ctx://test_account/users/test_user/memories/"
    candidates = memory_archiver.scan(scope_uri, mock_context)

    # Should skip the recent memory
    assert len(candidates) == 0


def test_scan_skips_archived(memory_archiver, mock_vector_index):
    """Test that scan skips already archived memories."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    archived_hit = SeedHit(
        uri="ctx://test_account/users/test_user/memories/_archive/old_pref",
        score=0.1,
        active_count=0,
        updated_at=datetime.now(timezone.utc) - timedelta(days=30),
        context_type="MEMORY",
        category="preference",
        owner_space="user:test_user",
    )

    mock_vector_index.search_by_vector.return_value = [archived_hit]

    scope_uri = "ctx://test_account/users/test_user/memories/"
    candidates = memory_archiver.scan(scope_uri, ctx)

    # Should skip the archived memory
    assert len(candidates) == 0


def test_scan_filters_by_scope(memory_archiver, mock_vector_index):
    """Test that scan only returns memories in the requested scope."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    other_scope_hit = SeedHit(
        uri="ctx://other_account/users/other_user/memories/pref",
        score=0.1,
        active_count=0,
        updated_at=datetime.now(timezone.utc) - timedelta(days=30),
        context_type="MEMORY",
        category="preference",
        owner_space="user:other_user",
    )

    mock_vector_index.search_by_vector.return_value = [other_scope_hit]

    scope_uri = "ctx://test_account/users/test_user/memories/"
    candidates = memory_archiver.scan(scope_uri, ctx)

    # Should skip memories from other scopes
    assert len(candidates) == 0


def test_archive_dry_run(memory_archiver, mock_fs):
    """Test archive with dry_run=True."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    candidates = [
        ArchivalCandidate(
            uri="ctx://test_account/users/test_user/memories/pref1",
            active_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
            score=0.05,
        )
    ]

    result = memory_archiver.archive(candidates, ctx, dry_run=True)

    assert result.archived == 0
    assert result.skipped == 1
    assert result.errors == 0
    mock_fs.move_node.assert_not_called()


def test_archive_success(memory_archiver, mock_fs):
    """Test successful archive operation."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    candidates = [
        ArchivalCandidate(
            uri="ctx://test_account/users/test_user/memories/pref1",
            active_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
            score=0.05,
        )
    ]

    result = memory_archiver.archive(candidates, ctx, dry_run=False)

    assert result.archived == 1
    assert result.skipped == 0
    assert result.errors == 0
    mock_fs.move_node.assert_called_once()


def test_archive_error_handling(memory_archiver, mock_fs):
    """Test archive with move_node errors."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    mock_fs.move_node.side_effect = Exception("Move failed")

    candidates = [
        ArchivalCandidate(
            uri="ctx://test_account/users/test_user/memories/pref1",
            active_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
            score=0.05,
        )
    ]

    result = memory_archiver.archive(candidates, ctx, dry_run=False)

    assert result.archived == 0
    assert result.skipped == 0
    assert result.errors == 1


def test_restore_success(memory_archiver, mock_fs):
    """Test successful restore operation."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    archived_uri = "ctx://test_account/users/test_user/memories/_archive/pref1"

    result = memory_archiver.restore(archived_uri, ctx)

    assert result is True
    mock_fs.move_node.assert_called_once()


def test_restore_not_archived(memory_archiver, mock_fs):
    """Test restore when URI is not in archive directory."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    normal_uri = "ctx://test_account/users/test_user/memories/pref1"

    result = memory_archiver.restore(normal_uri, ctx)

    assert result is False
    mock_fs.move_node.assert_not_called()


def test_restore_error_handling(memory_archiver, mock_fs):
    """Test restore with move_node errors."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    mock_fs.move_node.side_effect = Exception("Restore failed")

    archived_uri = "ctx://test_account/users/test_user/memories/_archive/pref1"

    result = memory_archiver.restore(archived_uri, ctx)

    assert result is False


def test_scan_and_archive_convenience(memory_archiver, mock_vector_index, sample_seed_hits):
    """Test the scan_and_archive convenience method."""
    ctx = RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )
    mock_vector_index.search_by_vector.return_value = sample_seed_hits

    scope_uri = "ctx://test_account/users/test_user/memories/"
    result = memory_archiver.scan_and_archive(scope_uri, ctx, dry_run=True)

    assert isinstance(result, ArchivalResult)
    assert result.scanned >= 0


def test_build_archive_uri():
    """Test _build_archive_uri helper function."""
    uri = "ctx://test_account/users/test_user/memories/pref1"
    archive_uri = _build_archive_uri(uri)

    expected = "ctx://test_account/users/test_user/memories/_archive/pref1"
    assert archive_uri == expected


def test_build_archive_uri_no_slash():
    """Test _build_archive_uri with URI without slash."""
    uri = "simple_uri"
    archive_uri = _build_archive_uri(uri)

    assert archive_uri == "_archive/simple_uri"


def test_build_restore_uri():
    """Test _build_restore_uri helper function."""
    archived_uri = "ctx://test_account/users/test_user/memories/_archive/pref1"
    original_uri = _build_restore_uri(archived_uri)

    expected = "ctx://test_account/users/test_user/memories/pref1"
    assert original_uri == expected


def test_build_restore_uri_not_archived():
    """Test _build_restore_uri with non-archived URI."""
    normal_uri = "ctx://test_account/users/test_user/memories/pref1"
    result = _build_restore_uri(normal_uri)

    assert result is None


def test_archival_candidate_dataclass():
    """Test ArchivalCandidate dataclass creation."""
    candidate = ArchivalCandidate(
        uri="ctx://test/memories/pref1",
        active_count=5,
        updated_at=datetime.now(timezone.utc),
        score=0.5,
        context_type="MEMORY",
        parent_uri="ctx://test/memories/",
        category="preference",
        owner_space="user:test_user",
    )

    assert candidate.uri == "ctx://test/memories/pref1"
    assert candidate.active_count == 5
    assert candidate.score == 0.5
    assert candidate.context_type == "MEMORY"


def test_archival_result_dataclass():
    """Test ArchivalResult dataclass creation."""
    result = ArchivalResult(
        scanned=10,
        archived=5,
        skipped=3,
        errors=2,
        candidates=[],
    )

    assert result.scanned == 10
    assert result.archived == 5
    assert result.skipped == 3
    assert result.errors == 2
    assert result.candidates == []


def test_default_threshold_and_min_age():
    """Test default threshold and min_age values."""
    archiver = MemoryArchiver(
        fs=Mock(),
        vector_index=Mock(),
    )

    assert archiver.threshold == MemoryArchiver.DEFAULT_THRESHOLD
    assert archiver.min_age_days == MemoryArchiver.DEFAULT_MIN_AGE_DAYS
