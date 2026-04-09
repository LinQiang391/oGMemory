"""Contract tests for core schema.

Phase 0 frozen — verifies all dataclasses can be instantiated and serialized.
These tests catch accidental schema changes that break cross-layer contracts.
"""

import json
from dataclasses import asdict

import pytest

from core.models import (
    RequestContext,
    ContextNode,
    RelationEdge,
    CandidateMemory,
    WritePlan,
    IndexRecord,
    OutboxEvent,
    TypedQuery,
)
from core.enums import NodeStatus, ContextType, EventType


class TestRequestContext:
    """RequestContext validation."""

    def test_create_request_context(self):
        """RequestContext should create with all required fields."""
        ctx = RequestContext(
            account_id="acme-corp",
            user_id="user-123",
            agent_id="agent-gpt4",
            session_id="session-abc",
            trace_id="trace-xyz",
        )
        assert ctx.account_id == "acme-corp"
        assert ctx.user_id == "user-123"
        assert ctx.agent_id == "agent-gpt4"
        assert ctx.session_id == "session-abc"
        assert ctx.trace_id == "trace-xyz"

    def test_request_context_is_frozen(self):
        """RequestContext should be immutable (frozen dataclass)."""
        ctx = RequestContext(
            account_id="acme",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
        )
        with pytest.raises(Exception):  # frozen=True raises FrozenInstanceError (subclass of Exception)
            ctx.account_id = "other"


class TestContextNode:
    """ContextNode validation."""

    def test_create_context_node(self):
        """ContextNode should create with all required fields."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type=ContextType.MEMORY,
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="User profile and preferences",
            overview="Alice is a software engineer...",
            content="Full profile content here...",
        )
        assert node.uri == "ctx://acme/users/alice/memories/profile"
        assert node.context_type == "MEMORY"
        assert node.category == "profile"
        assert node.level == 0
        assert node.owner_space == "user:alice"

    def test_context_node_metadata_default(self):
        """ContextNode metadata should default to empty dict."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type=ContextType.MEMORY,
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Summary",
            overview="Overview",
            content="Content",
        )
        assert node.metadata == {}

    def test_context_node_with_metadata(self):
        """ContextNode should accept custom metadata."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type=ContextType.MEMORY,
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Summary",
            overview="Overview",
            content="Content",
            metadata={"status": "ACTIVE", "tags": ["verified"]},
        )
        assert node.metadata["status"] == "ACTIVE"
        assert node.metadata["tags"] == ["verified"]

    def test_context_node_parent_uri_property(self):
        """ContextNode should have parent_uri computed from uri."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/preferences/coding_style",
            context_type=ContextType.MEMORY,
            category="preference",
            level=0,
            owner_space="user_space_alice",
            abstract="Summary",
            overview="Overview",
            content="Content",
        )
        assert node.parent_uri == "ctx://acme/users/alice/memories/preferences"

    def test_context_node_parent_uri_for_root_node(self):
        """ContextNode parent_uri for root-level node."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type=ContextType.MEMORY,
            category="profile",
            level=0,
            owner_space="user_space_alice",
            abstract="Summary",
            overview="Overview",
            content="Content",
        )
        assert node.parent_uri == "ctx://acme/users/alice/memories"


class TestRelationEdge:
    """RelationEdge validation."""

    def test_create_relation_edge(self):
        """RelationEdge should create with all required fields."""
        edge = RelationEdge(
            from_uri="ctx://acme/users/alice/memories/entities/coffee",
            to_uri="ctx://acme/users/alice/memories/events/cafe_visit",
            relation_type="related_to",
            weight=0.85,
            reason="Alice visited a cafe serving coffee",
        )
        assert edge.from_uri == "ctx://acme/users/alice/memories/entities/coffee"
        assert edge.to_uri == "ctx://acme/users/alice/memories/events/cafe_visit"
        assert edge.relation_type == "related_to"
        assert edge.weight == 0.85
        assert edge.reason == "Alice visited a cafe serving coffee"

    def test_relation_edge_weight_range(self):
        """RelationEdge weight should be 0.0-1.0 (test verifies we can set extremes)."""
        edge_min = RelationEdge(
            from_uri="a", to_uri="b", relation_type="test", weight=0.0, reason="min"
        )
        edge_max = RelationEdge(
            from_uri="a", to_uri="b", relation_type="test", weight=1.0, reason="max"
        )
        assert edge_min.weight == 0.0
        assert edge_max.weight == 1.0


class TestCandidateMemory:
    """CandidateMemory validation."""

    def test_create_candidate_memory(self):
        """CandidateMemory should create with all required fields."""
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coffee_preferences",
            abstract="Likes dark roast coffee",
            overview="Prefers dark roast, medium roast...",
            content="Detailed coffee preferences...",
            confidence=0.9,
        )
        assert candidate.category == "preference"
        assert candidate.owner_scope == "user"
        assert candidate.routing_key == "coffee_preferences"
        assert candidate.confidence == 0.9


class TestWritePlan:
    """WritePlan validation."""

    def test_create_write_plan(self):
        """WritePlan should create with all required fields."""
        plan = WritePlan(
            action="create",
            target_uri="ctx://acme/users/alice/memories/preferences/coffee",
        )
        assert plan.action == "create"
        assert plan.target_uri == "ctx://acme/users/alice/memories/preferences/coffee"
        assert plan.merged_fields == {}
        assert plan.relation_edges == []

    def test_write_plan_with_merge(self):
        """WritePlan should support merge action with fields."""
        plan = WritePlan(
            action="merge",
            target_uri="ctx://acme/users/alice/memories/profile",
            merged_fields={"overview": "Updated overview"},
        )
        assert plan.action == "merge"
        assert plan.merged_fields == {"overview": "Updated overview"}

    def test_write_plan_with_relations(self):
        """WritePlan should include relation edges."""
        edge = RelationEdge(
            from_uri="a", to_uri="b", relation_type="related_to", weight=0.5, reason="test"
        )
        plan = WritePlan(
            action="create", target_uri="ctx://acme/users/alice/memories/test", relation_edges=[edge]
        )
        assert len(plan.relation_edges) == 1
        assert plan.relation_edges[0].from_uri == "a"

    def test_write_plan_valid_actions(self):
        """WritePlan action should be one of: create, merge, append, skip."""
        valid_actions = ["create", "merge", "append", "skip"]
        for action in valid_actions:
            plan = WritePlan(action=action, target_uri="test://uri")
            assert plan.action == action


class TestIndexRecord:
    """IndexRecord validation."""

    def test_create_index_record(self):
        """IndexRecord should create with all required fields."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/users/alice/memories/profile",
            level=0,
            text="User profile abstract",
            filters={"account_id": "acme", "owner_space": "user:alice"},
            metadata={"category": "profile"},
        )
        assert record.id == "abc123"
        assert record.uri == "ctx://acme/users/alice/memories/profile"
        assert record.level == 0
        assert record.text == "User profile abstract"

    def test_index_record_generate_id(self):
        """IndexRecord.generate_id should produce stable IDs."""
        uri = "ctx://acme/users/alice/memories/profile"
        id1 = IndexRecord.generate_id(uri, 0)
        id2 = IndexRecord.generate_id(uri, 0)
        id3 = IndexRecord.generate_id(uri, 1)

        assert id1 == id2  # Same input → same ID
        assert len(id1) == 16  # SHA256[:16]
        assert id1 != id3  # Different level → different ID

    def test_index_record_three_levels(self):
        """Each ContextNode expands to 3 IndexRecords (L0, L1, L2)."""
        uri = "ctx://acme/users/alice/memories/profile"
        l0_id = IndexRecord.generate_id(uri, 0)
        l1_id = IndexRecord.generate_id(uri, 1)
        l2_id = IndexRecord.generate_id(uri, 2)

        assert l0_id != l1_id != l2_id
        assert len({l0_id, l1_id, l2_id}) == 3  # All unique


class TestOutboxEvent:
    """OutboxEvent validation."""

    def test_create_outbox_event(self):
        """OutboxEvent should create with all required fields."""
        event = OutboxEvent(
            event_id="550e8400-e29b-41d4-a716-446655440000",
            event_type=EventType.UPSERT_CONTEXT,
            uri="ctx://acme/users/alice/memories/profile",
            payload={"test": "data"},
            status="PENDING",
        )
        assert event.event_id == "550e8400-e29b-41d4-a716-446655440000"
        assert event.event_type == "UPSERT_CONTEXT"
        assert event.uri == "ctx://acme/users/alice/memories/profile"
        assert event.payload == {"test": "data"}
        assert event.status == "PENDING"
        assert event.retry_count == 0

    def test_outbox_event_auto_timestamp(self):
        """OutboxEvent should auto-generate created_at if empty."""
        event = OutboxEvent(
            event_id="test-id",
            event_type=EventType.UPSERT_CONTEXT,
            uri="test://uri",
            payload={},
            status="PENDING",
        )
        assert event.created_at != ""  # Should be set by __post_init__

    def test_outbox_event_status_values(self):
        """OutboxEvent status should be one of: PENDING, PROCESSING, DONE, FAILED."""
        valid_statuses = ["PENDING", "PROCESSING", "DONE", "FAILED"]
        for status in valid_statuses:
            event = OutboxEvent(
                event_id="test", event_type=EventType.UPSERT_CONTEXT, uri="test", payload={}, status=status
            )
            assert event.status == status


class TestTypedQuery:
    """TypedQuery validation."""

    def test_create_typed_query(self):
        """TypedQuery should create with all required fields."""
        query = TypedQuery(
            text="What are Alice's coffee preferences?",
            context_type=ContextType.MEMORY,
            categories=["preference"],
            account_id="acme",
            owner_space="user:alice",
            top_k=10,
        )
        assert query.text == "What are Alice's coffee preferences?"
        assert query.context_type == "MEMORY"
        assert query.categories == ["preference"]
        assert query.account_id == "acme"
        assert query.owner_space == "user:alice"
        assert query.top_k == 10

    def test_typed_query_with_target_uri(self):
        """TypedQuery should support direct lookup via target_uri."""
        query = TypedQuery(
            text="Get profile",
            context_type=ContextType.MEMORY,
            categories=["profile"],
            target_uri="ctx://acme/users/alice/memories/profile",
            account_id="acme",
        )
        assert query.target_uri == "ctx://acme/users/alice/memories/profile"

    def test_typed_query_defaults(self):
        """TypedQuery should have sensible defaults."""
        query = TypedQuery(
            text="test query",
            context_type=ContextType.MEMORY,
            categories=["test"],
        )
        assert query.target_uri is None
        assert query.top_k == 10
        assert query.account_id == ""
        assert query.owner_space is None


class TestSerialization:
    """Test all models can be serialized to JSON for storage/transmission."""

    def test_models_serializable(self):
        """All core models should be JSON-serializable."""
        ctx = RequestContext(
            account_id="acme", user_id="u1", agent_id="a1", session_id="s1", trace_id="t1"
        )
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type=ContextType.MEMORY,
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="Summary",
            overview="Overview",
            content="Content",
            metadata={"status": "ACTIVE"},
        )

        # RequestContext is frozen, convert to dict manually
        ctx_dict = {
            "account_id": ctx.account_id,
            "user_id": ctx.user_id,
            "agent_id": ctx.agent_id,
            "session_id": ctx.session_id,
            "trace_id": ctx.trace_id,
        }

        # ContextNode should be dict-convertible
        node_dict = asdict(node)

        # Both should be JSON-serializable
        json.dumps(ctx_dict)
        json.dumps(node_dict)

    def test_index_record_filters_serializable(self):
        """IndexRecord.filters should be JSON-serializable."""
        record = IndexRecord(
            id="abc123",
            uri="test://uri",
            level=0,
            text="test",
            filters={"account_id": "acme", "owner_space": "user_space"},
            metadata={},
        )
        json.dumps(record.filters)

    def test_outbox_event_payload_serializable(self):
        """OutboxEvent.payload should be JSON-serializable."""
        event = OutboxEvent(
            event_id="test",
            event_type=EventType.UPSERT_CONTEXT,
            uri="test://uri",
            payload={"node_uri": "test://uri", "data": {"key": "value"}},
            status="PENDING",
        )
        json.dumps(event.payload)
