"""Unit tests for retrieval.query_planner.QueryPlanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.errors import ValidationError
from core.models import RetrievalConfig, RequestContext, Role
from retrieval.query_planner import QueryPlanner
from tests.unit.retrieval.conftest import make_ctx


@pytest.fixture()
def planner():
    return QueryPlanner(RetrievalConfig(default_top_k=5, max_top_k=20))


@pytest.fixture()
def ctx():
    return make_ctx()


class TestQueryPlanner:
    def test_empty_query_raises(self, planner, ctx):
        with pytest.raises(ValidationError):
            planner.plan("", ctx)

    def test_topk_exceeds_max(self, planner, ctx):
        with pytest.raises(ValidationError):
            planner.plan("hello", ctx, top_k=999)

    def test_memory_keyword(self, planner, ctx):
        queries = planner.plan("用户的偏好是什么", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == "MEMORY"
        assert queries[0].account_id == ctx.account_id

    def test_skill_keyword(self, planner, ctx):
        queries = planner.plan("how to deploy a service", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == "SKILL"

    def test_resource_keyword(self, planner, ctx):
        queries = planner.plan("请提供相关文档", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == "RESOURCE"

    def test_ambiguous_falls_back_to_triple(self, planner, ctx):
        queries = planner.plan("hello world", ctx)
        assert len(queries) == 3
        types = {q.context_type for q in queries}
        assert types == {"MEMORY", "SKILL", "RESOURCE"}

    def test_categories_passed_through(self, planner, ctx):
        queries = planner.plan("偏好", ctx, categories=["profile"])
        assert queries[0].categories == ["profile"]

    def test_skill_uses_agent_owner_space(self, planner, ctx):
        """SKILL context_type should use agent_space_name()."""
        queries = planner.plan("how to fix the bug", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == "SKILL"
        assert queries[0].owner_space == ctx.agent_space_name()

    def test_memory_uses_no_owner_space(self, planner, ctx):
        """MEMORY context_type should search both user and agent spaces."""
        queries = planner.plan("用户偏好是什么", ctx)
        assert len(queries) == 1
        assert queries[0].context_type == "MEMORY"
        assert queries[0].owner_space is None  # Search both spaces

    def test_memory_uses_visible_owner_spaces_when_present(self, planner):
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            role=Role.MEMBER,
            visible_owner_spaces=("user:u1", "agent:a1"),
        )

        queries = planner.plan("用户偏好是什么", ctx)

        assert queries[0].owner_space == ["user:u1", "agent:a1"]

    def test_skill_becomes_inaccessible_when_agent_not_visible(self, planner):
        ctx = RequestContext(
            account_id="acct-1",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            trace_id="t1",
            role=Role.MEMBER,
            visible_owner_spaces=("user:u1",),
        )

        queries = planner.plan("how to fix the bug", ctx)

        assert queries[0].owner_space == "__inaccessible__"
