"""Tests for PolicyRouter.

Tests the routing of CandidateMemory to appropriate MergePolicy.
"""

import pytest
from unittest.mock import Mock

from commit.policy_router import PolicyRouter
from commit.merge_policies import (
    ProfilePolicy,
    AggregateTopicPolicy,
    AppendOnlyPolicy,
    SkillToolPolicy,
)
from core.models import RequestContext, CandidateMemory


def create_mock_fs():
    """Create a mock ContextFS for testing."""
    mock_fs = Mock()
    mock_fs.exists.return_value = False
    mock_fs.read_node.return_value = None
    return mock_fs


class TestPolicyRouterInit:
    """Test PolicyRouter initialization."""

    def test_init_with_fs(self):
        """PolicyRouter requires ContextFS."""
        mock_fs = create_mock_fs()
        router = PolicyRouter(fs=mock_fs)
        assert router._profile_policy is not None
        assert router._aggregate_topic_policy is not None
        assert router._append_only_policy is not None
        assert router._skill_tool_policy is not None

    def test_init_creates_all_policies(self):
        """PolicyRouter should create all policy instances."""
        mock_fs = create_mock_fs()
        router = PolicyRouter(fs=mock_fs)

        assert isinstance(router._profile_policy, ProfilePolicy)
        assert isinstance(router._aggregate_topic_policy, AggregateTopicPolicy)
        assert isinstance(router._append_only_policy, AppendOnlyPolicy)
        assert isinstance(router._skill_tool_policy, SkillToolPolicy)


class TestPolicyRouterRoute:
    """Test PolicyRouter.route() method."""

    @pytest.fixture
    def router(self):
        """Create PolicyRouter with Mock ContextFS."""
        return PolicyRouter(fs=create_mock_fs())

    def test_route_profile_returns_profile_policy(self, router):
        """route() should return ProfilePolicy for profile category."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, ProfilePolicy)

    def test_route_preference_returns_aggregate_policy(self, router):
        """route() should return AggregateTopicPolicy for preference category."""
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding_style",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, AggregateTopicPolicy)

    def test_route_entity_returns_aggregate_policy(self, router):
        """route() should return AggregateTopicPolicy for entity category."""
        candidate = CandidateMemory(
            category="entity",
            owner_scope="user",
            routing_key="openai",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, AggregateTopicPolicy)

    def test_route_event_returns_append_policy(self, router):
        """route() should return AppendOnlyPolicy for event category."""
        candidate = CandidateMemory(
            category="event",
            owner_scope="user",
            routing_key="20250325_event",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, AppendOnlyPolicy)

    def test_route_case_returns_append_policy(self, router):
        """route() should return AppendOnlyPolicy for case category."""
        candidate = CandidateMemory(
            category="case",
            owner_scope="agent",
            routing_key="20250325_case",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, AppendOnlyPolicy)

    def test_route_pattern_returns_aggregate_policy(self, router):
        """route() should return AggregateTopicPolicy for pattern category."""
        candidate = CandidateMemory(
            category="pattern",
            owner_scope="agent",
            routing_key="debugging_workflow",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, AggregateTopicPolicy)

    def test_route_skill_returns_skill_policy(self, router):
        """route() should return SkillToolPolicy for skill category."""
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="code_review",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert isinstance(policy, SkillToolPolicy)

    def test_route_unknown_category_returns_none(self, router):
        """route() should return None for unknown category."""
        candidate = CandidateMemory(
            category="unknown_category",
            owner_scope="user",
            routing_key="test",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert policy is None


class TestPolicyRouterPlan:
    """Test PolicyRouter.plan() method."""

    @pytest.fixture
    def router(self):
        """Create PolicyRouter with Mock ContextFS."""
        return PolicyRouter(fs=create_mock_fs())

    @pytest.fixture
    def ctx(self):
        """Create RequestContext."""
        return RequestContext(
            account_id="test-account",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            trace_id="trace-001",
        )

    def test_plan_returns_write_plan(self, router, ctx):
        """plan() should return WritePlan."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert plan.action in ("create", "merge", "append", "skip")
        assert plan.target_uri is not None

    def test_plan_profile_creates_or_merges(self, router, ctx):
        """plan() for profile should create or merge."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",
            abstract="User profile",
            overview="Profile overview",
            content="Profile content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert plan.action in ("create", "merge")
        assert "profile" in plan.target_uri

    def test_plan_preference_uses_routing_key(self, router, ctx):
        """plan() for preference should use routing_key in URI."""
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding_style",
            abstract="Prefers Python",
            overview="User prefers Python",
            content="Python preference",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert "coding_style" in plan.target_uri

    def test_plan_event_creates_unique_uri(self, router, ctx):
        """plan() for event should create unique URI."""
        candidate = CandidateMemory(
            category="event",
            owner_scope="user",
            routing_key="20250325_event",
            abstract="Event abstract",
            overview="Event overview",
            content="Event content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert plan.action == "create"
        assert "event" in plan.target_uri  # Note: singular "event" not "events"

    def test_plan_unknown_category_raises_error(self, router, ctx):
        """plan() should raise ValueError for unknown category."""
        candidate = CandidateMemory(
            category="unknown",
            owner_scope="user",
            routing_key="test",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        with pytest.raises(ValueError, match="Unknown candidate category"):
            router.plan(candidate, ctx)


class TestPolicyRouterRegisterPolicy:
    """Test PolicyRouter.register_policy() method."""

    @pytest.fixture
    def router(self):
        """Create PolicyRouter with Mock ContextFS."""
        return PolicyRouter(fs=create_mock_fs())

    def test_register_policy_adds_custom_policy(self, router):
        """register_policy() should add custom policy for category."""
        custom_policy = ProfilePolicy(create_mock_fs())
        custom_category = "custom_category"

        router.register_policy(custom_category, custom_policy)

        assert router._policy_map[custom_category] == custom_policy

    def test_register_policy_overrides_existing(self, router):
        """register_policy() should override existing policy."""
        new_policy = ProfilePolicy(create_mock_fs())

        router.register_policy("profile", new_policy)

        assert router._policy_map["profile"] == new_policy

    def test_registered_policy_is_used_in_route(self, router):
        """Registered policy should be used by route()."""
        custom_policy = ProfilePolicy(create_mock_fs())

        router.register_policy("custom_category", custom_policy)

        candidate = CandidateMemory(
            category="custom_category",
            owner_scope="user",
            routing_key="test",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        policy = router.route(candidate)

        assert policy == custom_policy


class TestPolicyRouterEdgeCases:
    """Test PolicyRouter edge cases."""

    @pytest.fixture
    def router(self):
        """Create PolicyRouter with Mock ContextFS."""
        return PolicyRouter(fs=create_mock_fs())

    @pytest.fixture
    def ctx(self):
        """Create RequestContext."""
        return RequestContext(
            account_id="test-account",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            trace_id="trace-001",
        )

    def test_plan_with_empty_routing_key(self, router, ctx):
        """plan() should handle empty routing_key for profile."""
        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="",  # Empty for profile
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert plan.target_uri.endswith("/profile")

    def test_plan_with_special_characters_in_routing_key(self, router, ctx):
        """plan() should handle special characters in routing_key."""
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding_style_2025",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert "coding_style_2025" in plan.target_uri

    def test_plan_with_user_scope(self, router, ctx):
        """plan() should create URI with user scope."""
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="test",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert f"users/{ctx.user_id}" in plan.target_uri

    def test_plan_with_agent_scope(self, router, ctx):
        """plan() should create URI with agent scope."""
        candidate = CandidateMemory(
            category="skill",
            owner_scope="agent",
            routing_key="test_skill",
            abstract="Abstract",
            overview="Overview",
            content="Content",
            confidence=0.9,
        )

        plan = router.plan(candidate, ctx)

        assert f"agents/{ctx.agent_id}" in plan.target_uri
