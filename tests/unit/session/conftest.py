"""Fixtures for session package tests."""

from unittest.mock import Mock

import pytest

from core.models import RequestContext
from session.archive_store import SessionArchiveStore
from session.compressor import SessionCompressor


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
    fs.write_node = Mock()
    fs.read_node = Mock()
    fs.exists = Mock(return_value=True)
    fs.list_children = Mock(return_value=[])
    fs.delete_node = Mock()
    fs.move_node = Mock()
    return fs


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.complete_json = Mock(
        return_value={
            "overview": "Test session overview",
            "abstract": "Test abstract",
        }
    )
    return llm


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What do you need?"},
    ]


@pytest.fixture
def archive_store(mock_fs):
    """SessionArchiveStore instance with mocked dependencies."""
    return SessionArchiveStore(fs=mock_fs)


@pytest.fixture
def compressor(mock_llm):
    """SessionCompressor instance with mocked dependencies."""
    return SessionCompressor(llm=mock_llm)


@pytest.fixture
def compressor_no_llm():
    """SessionCompressor instance without LLM (fallback mode)."""
    return SessionCompressor(llm=None)
