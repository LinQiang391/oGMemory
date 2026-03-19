"""Archive builder for constructing ContextNode from candidates.

Takes CandidateMemory and WritePlan to produce ContextNode ready for writing.
"""

from datetime import datetime, timezone

from core.models import RequestContext, CandidateMemory, WritePlan, ContextNode
from core.enums import ContextType


class ArchiveBuilder:
    """Builds ContextNode from candidate and write plan.

    The builder handles:
    - Constructing proper URI from WritePlan
    - Merging content for merge actions
    - Setting appropriate metadata
    - Building relations
    """

    def build(
        self,
        candidate: CandidateMemory,
        plan: WritePlan,
        ctx: RequestContext
    ) -> ContextNode:
        """Build ContextNode from candidate and write plan.

        Args:
            candidate: CandidateMemory with extracted content
            plan: WritePlan with action and target URI
            ctx: RequestContext for this operation

        Returns:
            ContextNode ready for ContextFS.write_node()
        """
        # Determine context type from category
        if candidate.category in ("skill", "pattern"):
            context_type = ContextType.SKILL.value
        else:
            context_type = ContextType.MEMORY.value

        # Calculate level (depth in directory tree)
        # profile/ = level 3, others = level 4
        if candidate.category == "profile":
            level = 3
        else:
            level = 4

        # Determine owner space
        if candidate.owner_scope == "user":
            owner_space = f"user_space_{ctx.user_id}"
        else:
            owner_space = f"agent_space_{ctx.agent_id}"

        # Build content based on action
        if plan.action == "merge" and plan.merged_fields:
            content = self._merge_content(candidate, plan.merged_fields)
        else:
            content = candidate.content

        # Build metadata
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add any merged metadata
        if "usage_count" in plan.merged_fields:
            metadata["usage_count"] = plan.merged_fields["usage_count"]
        if "usage_count_increment" in plan.merged_fields:
            metadata["usage_count_increment"] = plan.merged_fields["usage_count_increment"]

        # Store relations in metadata for write_node
        if plan.relation_edges:
            metadata["_relations"] = plan.relation_edges

        return ContextNode(
            uri=plan.target_uri,
            context_type=context_type,
            category=candidate.category,
            level=level,
            owner_space=owner_space,
            abstract=candidate.abstract,
            overview=candidate.overview,
            content=content,
            metadata=metadata,
        )

    def _merge_content(self, candidate: CandidateMemory, merged_fields: dict) -> str:
        """Merge candidate content with existing data.

        Args:
            candidate: CandidateMemory with new content
            merged_fields: Fields to merge from existing node

        Returns:
            Merged content string
        """
        # For append operations
        if "content_append" in merged_fields:
            existing = merged_fields.get("existing_content", "")
            new = candidate.content
            return f"{existing}\n\n{new}"

        # For replace operations (profile, etc.)
        if "abstract" in merged_fields:
            # New content replaces old
            return candidate.content

        # Default: use candidate content
        return candidate.content

    def build_relations(
        self,
        candidate: CandidateMemory,
        plan: WritePlan,
        existing_node: ContextNode | None = None
    ) -> list:
        """Build relation edges for this write.

        Args:
            candidate: CandidateMemory
            plan: WritePlan with relation_edges
            existing_node: Existing node if merging

        Returns:
            List of RelationEdge objects
        """
        relations = []

        # Add edges from plan
        relations.extend(plan.relation_edges)

        # TODO: Add automatic relation inference
        # - Same-session events → SEQUENCE relations
        # - Entity mentions → related_to relations

        return relations
