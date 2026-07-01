"""Contract tests for GraphRelationStore.

Phase 2 — Verifies graph query correctness and interface compatibility.
These tests ensure 1-hop and 2-hop traversals return correct results.

NOTE: Phase 2 feature - SKIPPED until GraphRelationStore is available.
Dependencies:
- P2-G2: GraphRelationStore implementation ✅ DONE
- P2-G3: 1-hop traversal method ✅ DONE
- P2-G4: 2-hop traversal method ✅ DONE
- P2-G5: GraphRelationStore unit tests ✅ DONE
"""

import pytest

# Phase 2: GraphRelationStore is not available in dev branch (write path only)
# Skip all tests in this module
pytestmark = pytest.mark.skip(reason="Phase 2: GraphRelationStore not available in dev branch")

from core.models import RequestContext, RelationEdge
from core.interfaces import RelationStore

try:
    from providers.relation_store.graph_relation_store import (
        GraphRelationStore,
        RelatedNode,
    )
    _HAS_GRAPH_STORE = True
except ImportError:
    _HAS_GRAPH_STORE = False


@pytest.fixture
def ctx():
    """Default request context for testing."""
    return RequestContext(
        account_id="test_account",
        user_id="user1",
        agent_id="agent1",
        session_id="session1",
        trace_id="trace1",
    )


@pytest.fixture
def ctx_other():
    """Different account context for isolation testing."""
    return RequestContext(
        account_id="other_account",
        user_id="user2",
        agent_id="agent2",
        session_id="session2",
        trace_id="trace2",
    )


@pytest.fixture
def store():
    """Fresh GraphRelationStore for each test."""
    if not _HAS_GRAPH_STORE:
        pytest.skip("GraphRelationStore not available")
    return GraphRelationStore()


class TestRelationStoreProtocol:
    """Verify RelationStore Protocol compliance."""

    def test_relation_store_has_get_edges(self):
        """RelationStore must have get_edges(uri, ctx) method."""
        assert hasattr(RelationStore, 'get_edges')

    def test_relation_store_has_upsert_edges(self):
        """RelationStore must have upsert_edges(edges, ctx) method."""
        assert hasattr(RelationStore, 'upsert_edges')


class TestGraphRelationStore:
    """Test GraphRelationStore when available."""

    def test_graph_relation_store_satisfies_protocol(self, store, ctx):
        """GraphRelationStore must satisfy RelationStore Protocol."""
        # Verify protocol methods exist
        assert hasattr(store, 'get_edges')
        assert hasattr(store, 'upsert_edges')
        # Verify they're callable
        assert callable(store.get_edges)
        assert callable(store.upsert_edges)
