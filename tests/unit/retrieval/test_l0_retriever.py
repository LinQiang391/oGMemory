"""Tests for retrieval.l0_retriever."""

from __future__ import annotations

from unittest.mock import Mock

from core.models import SeedHit, TypedQuery
from retrieval.l0_retriever import L0Retriever
from tests.unit.retrieval.conftest import make_ctx


def test_retrieve_overrides_mismatched_account_id_before_search():
    vector_index = Mock()
    expected = [SeedHit(uri="ctx://acct-1/users/u1/memories/profile", score=0.9)]
    vector_index.search.return_value = expected
    retriever = L0Retriever(vector_index)
    ctx = make_ctx(account_id="acct-1")
    query = TypedQuery(
        text="coffee",
        context_type="MEMORY",
        categories=["preferences"],
        account_id="wrong-acct",
    )

    result = retriever.retrieve(query, ctx)

    assert result == expected
    assert query.account_id == "acct-1"
    vector_index.search.assert_called_once_with(query)


def test_retrieve_with_filters_replaces_categories_when_explicit_filter_passed():
    vector_index = Mock()
    vector_index.search.return_value = []
    retriever = L0Retriever(vector_index)
    ctx = make_ctx()
    query = TypedQuery(
        text="coffee",
        context_type="MEMORY",
        categories=["profile"],
        top_k=3,
        account_id=ctx.account_id,
        owner_space="user:u1",
    )

    retriever.retrieve_with_filters(query, ctx, categories=["preferences"])

    called_query = vector_index.search.call_args.args[0]
    assert called_query.categories == ["preferences"]
    assert called_query.text == "coffee"
    assert called_query.owner_space == "user:u1"


def test_retrieve_with_filters_uses_original_query_categories_when_filter_missing():
    vector_index = Mock()
    vector_index.search.return_value = []
    retriever = L0Retriever(vector_index)
    ctx = make_ctx()
    query = TypedQuery(
        text="coffee",
        context_type="MEMORY",
        categories=["profile"],
        account_id=ctx.account_id,
    )

    retriever.retrieve_with_filters(query, ctx)

    called_query = vector_index.search.call_args.args[0]
    assert called_query is query
