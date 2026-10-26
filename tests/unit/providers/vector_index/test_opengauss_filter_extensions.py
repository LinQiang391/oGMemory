"""SQL tenant and space filter compatibility tests."""

from __future__ import annotations

from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_build_where_empty_filters_returns_true():
    """Edge case: an empty filter dict must still produce a valid
    WHERE clause (we use ``TRUE`` so the caller doesn't have to
    special-case "no filters"). A regression to ``AND `` (empty
    join) would emit a SQL syntax error on the first SELECT."""
    where, params = OpenGaussVectorIndex._build_where({})
    assert where == "TRUE"
    assert params == {}


def test_legacy_ownership_fallback_preserves_explicit_grants():
    where, params = OpenGaussVectorIndex._build_where({
        "account_id": "acme",
        "owner_space": ["user:alice", "agent:main"],
        "space_id": ["private-space"],
    })
    assert "COALESCE(account_id, filters->>'account_id') = %(f_account_id)s" in where
    assert "COALESCE(owner_space, filters->>'owner_space') = ANY(%(f_owner_space)s)" in where
    assert "space_id = ANY(%(f_space_id)s)" in where
    assert "space_id IS NULL" not in where
    assert params["f_owner_space"] == ["user:alice", "agent:main"]
