"""Archive builder for constructing ContextNode from candidates.

Takes CandidateMemory and WritePlan to produce ContextNode ready for writing.
"""

from datetime import datetime, timezone
from typing import Optional

from core.models import (
    RequestContext,
    CandidateMemory,
    WritePlan,
    ContextNode,
)
from core.enums import ContextType
from core.logging_config import get_logger

logger = get_logger(__name__)


class ArchiveBuilder:
    """Builds ContextNode from candidate and write plan.

    The builder handles:
    - Constructing proper URI from WritePlan
    - Merging content for merge actions (with optional LLM semantic merge)
    - Setting appropriate metadata
    - Building relations
    """

    def __init__(self, llm=None):
        """Initialize ArchiveBuilder.

        Args:
            llm: Optional LLM instance for semantic merge of skill content.
                 When provided, skill content will be semantically merged
                 (deduplicated, refined) instead of simple append.
        """
        self._llm = llm

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

        # Determine owner space using RequestContext methods for consistent format
        if candidate.owner_scope == "user":
            owner_space = ctx.user_space_name()
        else:
            owner_space = ctx.agent_space_name()

        # Build content based on action
        if plan.action == "merge" and plan.merged_fields:
            content = self._merge_content(candidate, plan.merged_fields)
        else:
            content = candidate.content

        # Build overview based on action (for skill_merge)
        if plan.action == "merge" and plan.merged_fields.get("skill_merge"):
            overview = self._merge_overview(candidate, plan.merged_fields)
        else:
            overview = candidate.overview

        # Build abstract based on action (for skill_merge)
        if plan.action == "merge" and plan.merged_fields.get("skill_merge"):
            abstract = plan.merged_fields.get("new_abstract", candidate.abstract)
        else:
            abstract = candidate.abstract

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
            abstract=abstract,
            overview=overview,
            content=content,
            metadata=metadata,
        )

    def _merge_overview(self, candidate: CandidateMemory, merged_fields: dict) -> str:
        """Merge overview for skill_merge operations.

        Args:
            candidate: CandidateMemory with new overview
            merged_fields: Fields to merge from existing node

        Returns:
            Merged overview string
        """
        existing = merged_fields.get("existing_overview", "")
        new = merged_fields.get("new_overview", candidate.overview)
        if not existing:
            return new
        return f"{existing}\n\n{new}"

    def _merge_content(self, candidate: CandidateMemory, merged_fields: dict) -> str:
        """Merge candidate content with existing data.

        Args:
            candidate: CandidateMemory with new content
            merged_fields: Fields to merge from existing node

        Returns:
            Merged content string
        """
        # For skill_merge operations (structured skill accumulation)
        if merged_fields.get("skill_merge"):
            existing = merged_fields.get("existing_content", "")
            new_content = merged_fields.get("new_content", candidate.content)
            if not existing:
                return new_content
            # Use LLM semantic merge when available
            if self._llm is not None:
                return self._llm_merge_skill_content(
                    existing=existing,
                    new=new_content,
                )
            # Fallback: simple append (保持原有行为)
            return f"{existing}\n\n---\n\n## 更新\n\n{new_content}"

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

    def _llm_merge_skill_content(self, existing: str, new: str) -> str:
        """Use LLM to semantically merge skill content (deduplication, refinement).

        Args:
            existing: Existing skill content
            new: New skill content to merge in

        Returns:
            Merged skill content with duplicates removed and structure refined.
            Falls back to simple append on any error.
        """
        prompt = f"""你是一个技能知识库维护助手。
将以下两段技能说明合并为一段，要求：
- 去除重复内容
- 保留所有有价值的步骤、注意事项、失败案例
- 如果新内容与旧内容矛盾，保留新内容
- 保持结构清晰，用 Markdown 格式

现有内容：
{existing}

新增内容：
{new}

返回合并后的完整技能说明。"""

        schema = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "合并后的技能说明内容"
                }
            },
            "required": ["content"]
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            merged = result.get("content", "").strip()
            if merged:
                logger.debug("LLM semantic merge successful: %d chars → %d chars",
                            len(existing) + len(new), len(merged))
                return merged
        except Exception as e:
            logger.warning("LLM semantic merge failed, falling back to append: %s", e)

        # Fallback: simple append (保持原有行为)
        return f"{existing}\n\n---\n\n## 更新\n\n{new}"

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
