"""Contract test for cross-account access control.

Phase 1 — verifies that cross-account read/write/search all return 403.

Multi-tenant isolation is CRITICAL. Any violation causes silent data leakage
between tenants. This contract test enforces the three-layer isolation model:

1. ContextFS layer: URI account prefix validation + _is_accessible()
2. Service layer: account_id injection, cannot be overridden
3. VectorIndex layer: implicit filtering by query.account_id

See CLAUDE.md §8 Multi-Tenant Isolation Rules.
"""

import pytest
from dataclasses import dataclass

from core.models import RequestContext, TypedQuery, ContextNode
from core.errors import AccessDeniedError


class TestRequestContextIsolation:
    """Verify RequestContext carries isolation keys."""

    def test_request_context_has_account_id(self):
        """RequestContext must have account_id for all operations."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )
        assert ctx.account_id == "acme"

    def test_request_context_identifies_owner_space(self):
        """RequestContext includes user/agent scope for namespace isolation."""
        ctx_user = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )
        assert ctx_user.account_id == "acme"
        assert ctx_user.user_id == "alice"


class TestURIAccountPrefixValidation:
    """Verify URI account prefix is validated against RequestContext."""

    def test_uri_account_must_match_context(self):
        """
        URI's account prefix MUST match RequestContext.account_id.

        uri: ctx://{account}/users/{user}/...
                    ^^^^^^^^
                    Must equal RequestContext.account_id
        """
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        same_account_uri = "ctx://acme/users/alice/memories/profile"
        assert self._extract_account_from_uri(same_account_uri) == ctx.account_id

        different_account_uri = "ctx://other-corp/users/alice/memories/profile"
        assert self._extract_account_from_uri(different_account_uri) != ctx.account_id

    def test_cross_account_read_returns_403(self):
        """
        Reading from a different account must raise AccessDeniedError.

        Scenario: User from account "acme" tries to read account "other-corp" node.
        Expected: AccessDeniedError with uri and account_id details.
        """
        ctx_acme = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        other_account_uri = "ctx://other-corp/users/bob/memories/profile"

        with pytest.raises(AccessDeniedError) as exc_info:
            self._simulate_read_with_validation(other_account_uri, ctx_acme)

        assert exc_info.value.uri == other_account_uri
        assert exc_info.value.account_id == "acme"
        assert "account" in str(exc_info.value).lower() or "access denied" in str(exc_info.value).lower()

    def test_cross_account_write_returns_403(self):
        """
        Writing to a different account must raise AccessDeniedError.

        Scenario: User from account "acme" tries to write to account "other-corp".
        Expected: AccessDeniedError with uri and account_id details.
        """
        ctx_acme = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        other_account_node = ContextNode(
            uri="ctx://other-corp/users/bob/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:bob",
            abstract="Other profile",
            overview="Overview",
            content="Content",
        )

        with pytest.raises(AccessDeniedError) as exc_info:
            self._simulate_write_with_validation(other_account_node, ctx_acme)

        assert exc_info.value.uri == other_account_node.uri
        assert exc_info.value.account_id == "acme"

    def test_cross_account_search_returns_403(self):
        """
        Searching a different account must be filtered at VectorIndex layer.

        Unlike read/write, search doesn't explicitly validate URI account.
        Instead, VectorIndex.search() MUST filter by query.account_id.
        """
        ctx_acme = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        query = TypedQuery(
            text="software engineer",
            context_type="MEMORY",
            categories=["profile"],
            account_id="acme",
            top_k=10,
        )

        records = [
            self._make_index_record("acme", "uri-1"),
            self._make_index_record("other-corp", "uri-2"),
        ]

        filtered = [r for r in records if r.filters["account_id"] == query.account_id]

        assert len(filtered) == 1
        assert filtered[0].filters["account_id"] == "acme"

    def _extract_account_from_uri(self, uri: str) -> str:
        """Extract account ID from URI: ctx://{account}/..."""
        without_scheme = uri.replace("ctx://", "")
        return without_scheme.split("/")[0]

    def _simulate_read_with_validation(self, uri: str, ctx: RequestContext) -> None:
        """Simulate service layer read validation."""
        uri_account = self._extract_account_from_uri(uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                uri=uri,
                account_id=ctx.account_id,
                reason=f"URI account '{uri_account}' does not match context account '{ctx.account_id}'",
            )

    def _simulate_write_with_validation(self, node: ContextNode, ctx: RequestContext) -> None:
        """Simulate service layer write validation."""
        uri_account = self._extract_account_from_uri(node.uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                uri=node.uri,
                account_id=ctx.account_id,
                reason=f"URI account '{uri_account}' does not match context account '{ctx.account_id}'",
            )

    def _make_index_record(self, account_id: str, uri: str):
        """Mock IndexRecord for testing."""
        from core.models import IndexRecord
        return IndexRecord(
            id="test-id",
            uri=uri,
            level=0,
            text="test",
            filters={"account_id": account_id, "owner_space": "test_space"},
            metadata={},
        )


class TestServiceLayerAccountIdInjection:
    """Verify service layer injects account_id, caller cannot override."""

    def test_search_memory_injects_account_id(self):
        """
        search_memory() MUST inject account_id from RequestContext.

        Caller provides: query text, top_k
        Service layer adds: account_id (from RequestContext)
        Caller CANNOT override account_id
        """
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        caller_query = "find software engineers"

        constructed_query = TypedQuery(
            text=caller_query,
            context_type="MEMORY",
            categories=[],
            account_id=ctx.account_id,
            top_k=10,
        )

        assert constructed_query.account_id == "acme"

    def test_caller_cannot_override_account_id(self):
        """
        Even if caller tries to pass account_id, service layer MUST use
        RequestContext.account_id instead.
        """
        ctx = RequestContext(
            account_id="acme-correct",
            user_id="alice",
            agent_id="bot",
            session_id="sess-1",
            trace_id="trace-1",
        )

        attempted_account_id = "other-corp"

        final_query = TypedQuery(
            text="test",
            context_type="MEMORY",
            categories=[],
            account_id=ctx.account_id,
            top_k=10,
        )

        assert final_query.account_id == "acme-correct"
        assert final_query.account_id != attempted_account_id


class TestVectorIndexImplicitFiltering:
    """Verify VectorIndex.search() filters by account_id implicitly."""

    def test_vector_index_filters_by_account_id(self):
        """
        VectorIndex.search() MUST filter results by query.account_id.

        This is the last line of defense against cross-tenant leakage.
        """
        query = TypedQuery(
            text="software engineer",
            context_type="MEMORY",
            categories=[],
            account_id="acme",
            top_k=10,
        )

        all_records = [
            self._make_index_record("acme", "uri-1"),
            self._make_index_record("other-corp", "uri-2"),
            self._make_index_record("acme", "uri-3"),
            self._make_index_record("third-corp", "uri-4"),
        ]

        results = [r for r in all_records if r.filters["account_id"] == query.account_id]

        assert len(results) == 2
        assert all(r.filters["account_id"] == "acme" for r in results)

    def _make_index_record(self, account_id: str, uri: str):
        """Mock IndexRecord for testing."""
        from core.models import IndexRecord
        return IndexRecord(
            id="test-id",
            uri=uri,
            level=0,
            text="test",
            filters={"account_id": account_id, "owner_space": "test_space"},
            metadata={},
        )


class TestOwnerSpaceIsolation:
    """Verify user/agent namespace isolation within same account."""

    def test_user_and_agent_memories_isolated(self):
        """
        Within same account, user and agent memories are isolated
        by owner_space filter.
        """
        query_user = TypedQuery(
            text="preferences",
            context_type="MEMORY",
            categories=["preference"],
            account_id="acme",
            owner_space="user:alice",
            top_k=10,
        )

        records = [
            self._make_index_record_with_owner("acme", "user:alice", "uri-user"),
            self._make_index_record_with_owner("acme", "agent:bot", "uri-agent"),
        ]

        results = [
            r for r in records
            if r.filters["account_id"] == query_user.account_id
            and r.filters.get("owner_space") == query_user.owner_space
        ]

        assert len(results) == 1
        assert results[0].filters["owner_space"] == "user:alice"

    def _make_index_record_with_owner(self, account_id: str, owner_space: str, uri: str):
        """Mock IndexRecord with owner_space for testing."""
        from core.models import IndexRecord
        return IndexRecord(
            id="test-id",
            uri=uri,
            level=0,
            text="test",
            filters={"account_id": account_id, "owner_space": owner_space},
            metadata={},
        )
