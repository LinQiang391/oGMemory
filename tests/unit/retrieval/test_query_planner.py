"""Unit tests for retrieval.query_planner.QueryPlanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.errors import ValidationError
from core.models import RetrievalConfig
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
