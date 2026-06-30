"""Shared pytest fixtures for unit tests."""

import pytest

from core.models import RequestContext, ContextNode
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.embedder.mock_embedder import MockEmbedder


@pytest.fixture(scope="function")
def sample_context():
    """Create a sample RequestContext for testing."""
    import uuid
    return RequestContext(
        account_id="test-account",
        user_id="test-user",
        agent_id="test-agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def sample_context_a():
    """Context for user A in multi-tenant tests."""
    import uuid
    return RequestContext(
        account_id="account-a",
        user_id="alice",
        agent_id="agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def sample_context_b():
    """Context for user B in multi-tenant tests."""
    import uuid
    return RequestContext(
        account_id="account-b",
        user_id="bob",
        agent_id="agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )


@pytest.fixture(scope="function")
def vector_index():
    """Create InMemoryVectorIndex for testing."""
    # Use dimension compatible with text-embedding-ada-002
    return InMemoryVectorIndex(dimension=1536)


@pytest.fixture(scope="function")
def mock_embedder():
    """Create mock embedder for testing."""
    return MockEmbedder(dimension=1536)


# ============================================================================
# Legacy fixture aliases (for backward compatibility)
# ============================================================================

@pytest.fixture(scope="function")
def mock_vector_index(vector_index):
    """Alias for vector_index (legacy name)."""
    return vector_index


# ============================================================================
# Test data helpers
# ============================================================================

@pytest.fixture(scope="function")
def sample_profile_node():
    """Create a sample profile ContextNode."""
    import uuid
    return ContextNode(
        uri="ctx://test-account/users/test-user/memories/profile",
        context_type="MEMORY",
        category="profile",
        level=0,
        owner_space="user:test-user",
        abstract="Test user is a software engineer",
        overview="## Profile\n\nName: Test User\nRole: Software Engineer",
        content="Full profile content here...",
        metadata={"status": "ACTIVE"},
    )


@pytest.fixture(scope="function")
def sample_preference_node():
    """Create a sample preference ContextNode."""
    import uuid
    return ContextNode(
        uri="ctx://test-account/users/test-user/memories/preferences/coding",
        context_type="MEMORY",
        category="preferences",
        level=0,
        owner_space="user:test-user",
        abstract="Prefers PEP8 coding style",
        overview="## Coding Preferences\n\n- PEP8 compliant\n- Type hints",
        content="Detailed coding preferences...",
        metadata={"status": "ACTIVE"},
    )


@pytest.fixture(scope="function")
def mock_llm():
    """Create a mock LLM for testing."""
    from providers.llm import MockLLM
    return MockLLM()
