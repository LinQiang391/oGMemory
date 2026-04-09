"""Stage 1 — Rule-based query planner.

Classifies a natural-language query into one or more TypedQuery objects
using keyword matching with a deterministic fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from core.enums import ContextType
from core.errors import ValidationError
from core.models import RetrievalConfig, RequestContext, TypedQuery

logger = logging.getLogger(__name__)

_MEMORY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"偏好|背景|身份|记忆|回顾|profile|preference|background|remember",
    ]
]

_SKILL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"怎么做|步骤|命令|流程|how\s*to|workflow|procedure|instruction|tutorial",
    ]
]

_RESOURCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"文档|资料|规范|链接|document|spec|reference|link|paper|article",
    ]
]


def _match_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```[\s\S]*?```"),                          # markdown code blocks (sender metadata)
    re.compile(r"Sender\s*\(untrusted metadata\)\s*:\s*"),  # OpenClaw sender prefix
    re.compile(r"\[[\w\s:\-]+UTC\]\s*"),                    # timestamp like [Fri 2026-03-27 06:16 UTC]
]


def sanitize_query(raw: str) -> str:
    """Strip upstream metadata noise, keeping only user intent."""
    text = raw
    for pat in _NOISE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


class QueryPlanner:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()

    def plan(
        self,
        query: str,
        ctx: RequestContext,
        *,
        session_archive: dict | None = None,
        hints: dict | None = None,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[TypedQuery]:
        query = sanitize_query(query or "")
        if not query:
            raise ValidationError("query", "query must not be empty")

        effective_top_k = top_k or self.config.default_top_k
        if effective_top_k > self.config.max_top_k:
            raise ValidationError(
                "top_k",
                f"top_k={effective_top_k} exceeds maximum {self.config.max_top_k}",
            )

        matched_types = self._classify(query)

        if len(matched_types) == 1:
            return [
                self._make_query(
                    query, matched_types[0], ctx,
                    categories=categories, top_k=effective_top_k,
                )
            ]

        logger.info("[QueryPlanner] matched %d types, falling back to triple-query", len(matched_types))
        return [
            self._make_query(query, ct.value, ctx, categories=categories, top_k=effective_top_k)
            for ct in ContextType
        ]

    @staticmethod
    def _classify(text: str) -> list[str]:
        hits: list[str] = []
        if _match_any(text, _MEMORY_PATTERNS):
            hits.append(ContextType.MEMORY.value)
        if _match_any(text, _SKILL_PATTERNS):
            hits.append(ContextType.SKILL.value)
        if _match_any(text, _RESOURCE_PATTERNS):
            hits.append(ContextType.RESOURCE.value)
        return hits

    @staticmethod
    def _make_query(
        text: str,
        context_type: str,
        ctx: RequestContext,
        *,
        categories: list[str] | None,
        top_k: int,
    ) -> TypedQuery:
        # Determine owner_space based on context_type:
        # - SKILL: agent space (skills/patterns/cases are agent-scoped)
        # - MEMORY: None (search both user and agent spaces)
        # - RESOURCE: None (can be in either space)
        if context_type == ContextType.SKILL.value:
            owner_space = ctx.agent_space_name()
        else:
            owner_space = None

        return TypedQuery(
            text=text,
            context_type=context_type,
            categories=categories or [],
            top_k=top_k,
            account_id=ctx.account_id,
            owner_space=owner_space,
        )
