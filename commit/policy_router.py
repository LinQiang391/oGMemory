"""Policy router for selecting the appropriate merge policy.

Routes CandidateMemory to the correct MergePolicy based on category.
"""

from typing import Optional

from core.models import RequestContext, CandidateMemory, WritePlan
from core.interfaces import ContextFS, MergePolicy
from commit.merge_policies import (
    ProfilePolicy,
    AggregateTopicPolicy,
    AppendOnlyPolicy,
    SkillToolPolicy,
)


class PolicyRouter:
    """Routes candidates to the appropriate merge policy.

    Category to policy mapping (CLAUDE.md §5):
    - profile → AggregateTopicPolicy (field-level nodes: profile/name, profile/location)
    - preference → AggregateTopicPolicy (merge by slug)
    - entity → AggregateTopicPolicy (merge by slug)
    - event → AppendOnlyPolicy (always create, unique ID)
    - case → AppendOnlyPolicy (always create, unique ID)
    - pattern → AggregateTopicPolicy (merge by slug)
    - skill → SkillToolPolicy (fixed URI, accumulate)
    - tool → SkillToolPolicy (fixed URI, accumulate, tools/ path)
    """

    def __init__(self, fs: ContextFS):
        """Initialize PolicyRouter with all policies.

        Args:
            fs: ContextFS instance for policy operations
        """
        self._profile_policy = ProfilePolicy(fs)
        self._aggregate_topic_policy = AggregateTopicPolicy(fs)
        self._append_only_policy = AppendOnlyPolicy(fs)
        self._skill_tool_policy = SkillToolPolicy(fs)

        # Category → Policy mapping
        self._policy_map = {
            "profile": self._aggregate_topic_policy,
            "preference": self._aggregate_topic_policy,
            "entity": self._aggregate_topic_policy,
            "event": self._append_only_policy,
            "case": self._append_only_policy,
            "pattern": self._aggregate_topic_policy,
            "skill": self._skill_tool_policy,
            "tool": self._skill_tool_policy,
        }

    def route(self, candidate: CandidateMemory) -> Optional[MergePolicy]:
        """Get the appropriate policy for a candidate.

        Args:
            candidate: CandidateMemory to route

        Returns:
            MergePolicy for this category, or None if unknown category
        """
        return self._policy_map.get(candidate.category)

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan using the appropriate policy.

        Args:
            candidate: CandidateMemory to plan for
            ctx: RequestContext for this operation

        Returns:
            WritePlan from the appropriate policy

        Raises:
            ValueError: If candidate category is unknown
        """
        policy = self.route(candidate)
        if policy is None:
            raise ValueError(f"Unknown candidate category: {candidate.category}")

        return policy.plan(candidate, ctx)

    def register_policy(self, category: str, policy: MergePolicy) -> None:
        """Register a custom policy for a category.

        Args:
            category: Memory category
            policy: MergePolicy to use for this category
        """
        self._policy_map[category] = policy
