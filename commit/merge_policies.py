"""Merge policies for determining write actions.

Each policy implements the MergePolicy Protocol and decides whether
to create, merge, append, or skip a candidate memory.
"""

import re
from datetime import datetime, timezone
from typing import Optional
import uuid as uuid_lib

from core.models import RequestContext, CandidateMemory, WritePlan, RelationEdge
from core.interfaces import MergePolicy, ContextFS
from core.errors import ConcurrentModificationError


def _generate_event_id() -> str:
    """Generate a unique event ID with timestamp.

    Uses full UUID4 to avoid collision risk in high-concurrency scenarios.
    Format: {iso_timestamp}_{full_uuid}
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    unique = uuid_lib.uuid4().hex
    return f"{timestamp}_{unique}"


class ProfilePolicy(MergePolicy):
    """Merge policy for user profile nodes.

    Behavior (CLAUDE.md §5):
    - Target URI: ctx://{account}/users/{user}/memories/profile
    - Always merge (never create new, never skip)
    - New information replaces old (user state changes)
    - Conflict resolution: new information wins

    Profile is a fixed URI that gets continuously updated.
    """

    def __init__(self, fs: ContextFS):
        """Initialize ProfilePolicy.

        Args:
            fs: ContextFS for checking existing nodes
        """
        self._fs = fs

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for profile candidate.

        Args:
            candidate: CandidateMemory (must be category="profile")
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="merge" and target URI for profile
        """
        target_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"

        # Check if profile exists
        exists = self._fs.exists(target_uri, ctx)

        if exists:
            action = "merge"
            # Read existing node to get version for optimistic locking
            existing_node = self._fs.read_node(target_uri, ctx)
            current_version = existing_node.metadata.get("version", 0)

            # Merge strategy: new info replaces old (user state changes)
            merged_fields = {
                "abstract": candidate.abstract,
                "overview": candidate.overview,
                "content": candidate.content,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "expected_version": current_version,  # For optimistic locking
            }
        else:
            action = "create"
            merged_fields = {}

        return WritePlan(
            action=action,
            target_uri=target_uri,
            merged_fields=merged_fields,
            relation_edges=[],
        )


class AggregateTopicPolicy(MergePolicy):
    """Merge policy for topic-based memories.

    Applies to: preference, entity, pattern

    Behavior (CLAUDE.md §5):
    - Target URI: ctx://.../{category}/{slug}
    - If exists: merge content into overview/content
    - If not exists: create new node
    - Similar slug handling: edit distance < 2 or semantic > 0.9 → merge

    Uses routing_key as slug for URI construction.
    """

    def __init__(self, fs: ContextFS, similarity_threshold: float = 0.9):
        """Initialize AggregateTopicPolicy.

        Args:
            fs: ContextFS for checking existing nodes
            similarity_threshold: Semantic similarity threshold for slug merging
        """
        self._fs = fs
        self._similarity_threshold = similarity_threshold

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for topic-based candidate.

        Args:
            candidate: CandidateMemory (preference, entity, or pattern)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" or "merge"
        """
        # Build target URI from routing_key (slug)
        owner_type = f"{candidate.owner_scope}s"  # user→users, agent→agents
        target_uri = f"ctx://{ctx.account_id}/{owner_type}/"
        if candidate.owner_scope == "user":
            target_uri += f"{ctx.user_id}/memories/{candidate.category}/{candidate.routing_key}"
        else:
            target_uri += f"{ctx.agent_id}/memories/{candidate.category}/{candidate.routing_key}"

        # Check if exact URI exists
        if self._fs.exists(target_uri, ctx):
            # Read existing node to get version for optimistic locking
            existing_node = self._fs.read_node(target_uri, ctx)
            current_version = existing_node.metadata.get("version", 0)

            return WritePlan(
                action="merge",
                target_uri=target_uri,
                merged_fields={
                    "abstract": candidate.abstract,
                    "existing_overview": existing_node.overview,
                    "overview_append": candidate.overview,
                    "existing_content": existing_node.content,
                    "content_append": candidate.content,
                    "expected_version": current_version,  # For optimistic locking
                },
                relation_edges=[],
            )

        # Check for similar slugs (simplified - exact match only for now)
        # TODO: P2 - Add semantic similarity and edit distance checking

        return WritePlan(
            action="create",
            target_uri=target_uri,
            merged_fields={},
            relation_edges=[],
        )


class AppendOnlyPolicy(MergePolicy):
    """Merge policy for time-series events.

    Applies to: event, case

    Behavior (CLAUDE.md §5):
    - event_id/case_id globally unique (timestamp_uuid)
    - Always create, never overwrite history
    - Establish SEQUENCE relations for same-session events

    Each event is a new node - historical record preserved.
    """

    def __init__(self, fs: ContextFS):
        """Initialize AppendOnlyPolicy.

        Args:
            fs: ContextFS for checking existing nodes
        """
        self._fs = fs

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for event/case candidate.

        Args:
            candidate: CandidateMemory (event or case)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" and unique event ID
        """
        # Generate unique event ID
        event_id = _generate_event_id()

        owner_type = f"{candidate.owner_scope}s"
        target_uri = f"ctx://{ctx.account_id}/{owner_type}/"
        if candidate.owner_scope == "user":
            target_uri += f"{ctx.user_id}/memories/{candidate.category}/{event_id}"
        else:
            target_uri += f"{ctx.agent_id}/memories/{candidate.category}/{event_id}"

        # Always create new event nodes (never merge)
        return WritePlan(
            action="create",
            target_uri=target_uri,
            merged_fields={},
            relation_edges=[],
        )


class SkillToolPolicy(MergePolicy):
    """Merge policy for agent skills and tools.

    Applies to: skill, tool

    Behavior (CLAUDE.md §5):
    - skill URI: ctx://{account}/agents/{agent}/skills/{skill_name}
    - tool URI:  ctx://{account}/agents/{agent}/memories/tools/{tool_name}
    - Fixed URI, always exists or is created
    - Cumulative: best practices, failures, parameters appended
    - Usage stats tracked in metadata

    Skills and tools accumulate knowledge over time.
    """

    def __init__(self, fs: ContextFS):
        """Initialize SkillToolPolicy.

        Args:
            fs: ContextFS for checking existing nodes
        """
        self._fs = fs

    def _build_uri(self, candidate: CandidateMemory, ctx: RequestContext) -> str:
        """Build target URI based on category."""
        if candidate.category == "tool":
            return f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tools/{candidate.routing_key}"
        return f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/{candidate.routing_key}"

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for skill/tool candidate.

        Args:
            candidate: CandidateMemory (skill or tool)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" or "merge" (accumulate)
        """
        target_uri = self._build_uri(candidate, ctx)

        exists = self._fs.exists(target_uri, ctx)

        if exists:
            # Read existing node to get version for optimistic locking
            existing_node = self._fs.read_node(target_uri, ctx)
            current_version = existing_node.metadata.get("version", 0)

            return WritePlan(
                action="merge",
                target_uri=target_uri,
                merged_fields={
                    "skill_merge": True,
                    "existing_content": existing_node.content,
                    "existing_overview": existing_node.overview,
                    "existing_abstract": existing_node.abstract,
                    "new_content": candidate.content,
                    "new_overview": candidate.overview,
                    "new_abstract": candidate.abstract,
                    "usage_count": existing_node.metadata.get("usage_count", 0) + 1,
                    "expected_version": current_version,  # For optimistic locking
                },
                relation_edges=[],
            )
        else:
            return WritePlan(
                action="create",
                target_uri=target_uri,
                merged_fields={
                    "usage_count": 1,
                },
                relation_edges=[],
            )
