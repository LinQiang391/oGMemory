"""Query Planner - infers query type and categories from natural language.

Phase 1: Rule-based inference (no ML).
account_id is injected by service layer, never set by Planner.

See CLAUDE.md §7 for retrieval chain rules.
"""

import re
from typing import Optional

from core.models import RequestContext, TypedQuery, ContextType


# Memory-related keywords for context_type inference
_MEMORY_KEYWORDS = [
    "偏好", "习惯", "背景", "记忆",
    "preference", "habit", "background", "memory", "remember",
    "recall", "知道", "了解", "记得"
]

# Skill-related keywords
_SKILL_KEYWORDS = [
    "如何", "步骤", "方法", "怎么做",
    "how to", "how do", "step", "method", "tutorial",
    "实现", "写", "代码", "code", "implement"
]

# Resource-related keywords
_RESOURCE_KEYWORDS = [
    "文档", "资料", "参考",
    "document", "documentation", "reference", "resource",
    "read", "reading", "资料"
]

# Category-specific keywords
_CATEGORY_KEYWORDS = {
    "profile": ["简介", "介绍", "概况", "profile", "bio", "about", "简介"],
    "preference": ["喜欢", "偏好", "习惯", "preference", "like", "favor", "习惯"],
    "entity": ["公司", "人", "组织", "company", "person", "organization", "实体"],
    "event": ["事件", "会议", "讨论", "event", "meeting", "discussion", "发生"],
    "case": ["案例", "情况", "问题", "case", "issue", "bug", "debug", "故障"],
    "pattern": ["模式", "规律", "经验", "pattern", "practice", "workflow", "模式"],
    "skill": ["技能", "能力", "会", "skill", "ability", "can", "技能"],
}


class Planner:
    """Rule-based query planning for retrieval.

    Infers context_type and categories from query text.
    Passes through account_id without modification (service layer responsibility).
    """

    def plan(self, query_text: str, ctx: RequestContext) -> TypedQuery:
        """Create a TypedQuery from natural language.

        Args:
            query_text: Raw query text from user
            ctx: RequestContext containing account_id (NOT set here)

        Returns:
            TypedQuery with inferred context_type and categories.
            account_id is passed through from ctx, never set here.

        NOTE: account_id comes from ctx, NOT inferred or set by Planner.
        The service layer is responsible for ensuring ctx.account_id is set.
        """
        # Determine context_type from keywords
        context_type = self._infer_context_type(query_text)

        # Determine relevant categories
        categories = self._infer_categories(query_text, context_type)

        return TypedQuery(
            text=query_text,
            context_type=context_type,
            categories=categories,
            account_id=ctx.account_id,  # PASSED THROUGH, NOT SET
            owner_space=ctx.user_id if context_type == ContextType.MEMORY else ctx.agent_id,
            top_k=10,
        )

    def _infer_context_type(self, query_text: str) -> str:
        """Infer context_type from query keywords.

        Priority: SKILL > RESOURCE > MEMORY (default)
        """
        query_lower = query_text.lower()

        # Check for skill keywords first (highest priority)
        for keyword in _SKILL_KEYWORDS:
            if keyword in query_lower:
                return ContextType.SKILL

        # Check for resource keywords
        for keyword in _RESOURCE_KEYWORDS:
            if keyword in query_lower:
                return ContextType.RESOURCE

        # Default to MEMORY
        return ContextType.MEMORY

    def _infer_categories(self, query_text: str, context_type: str) -> list[str]:
        """Infer relevant categories from query text.

        Returns list of categories that might contain relevant results.
        Empty list means search all categories.
        """
        query_lower = query_text.lower()
        matched_categories = []

        for category, keywords in _CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_categories.append(category)
                    break

        return matched_categories

    def infer_owner_space(
        self,
        query_text: str,
        ctx: RequestContext,
        context_type: str,
    ) -> Optional[str]:
        """Infer owner_space filter based on query type.

        Returns None for broader search (both user and agent).
        Returns specific owner_space for targeted queries.

        Phase 1: Simple heuristic - explicit "agent" or "skill" queries target agent space.
        """
        query_lower = query_text.lower()

        if context_type == ContextType.SKILL:
            # Skill queries default to agent space
            return f"agent_space:{ctx.agent_id}"

        if any(word in query_lower for word in ["agent", "助手", "ai", "模型"]):
            return f"agent_space:{ctx.agent_id}"

        if any(word in query_lower for word in ["user", "用户", "我"]):
            return f"user_space:{ctx.user_id}"

        # No specific filter - search both
        return None
