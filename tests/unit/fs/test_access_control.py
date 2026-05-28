"""Unit tests for unified access control helper."""

from __future__ import annotations

import pytest

from core.models import RequestContext, Role
from fs.access_control import check_uri_access


class TestCheckUriAccessAccountIsolation:
    """Account-level isolation tests."""

    def test_account_mismatch_always_denied(self):
        """Cross-account access must always fail."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        components = {
            "account": "other-corp",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "profile",
            "slug": "",
        }

        assert not check_uri_access(ctx, components, strict_mode=False)
        assert not check_uri_access(ctx, components, strict_mode=True)

    def test_account_match_required_for_all_modes(self):
        """Account match is prerequisite for both modes."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared"),
        )
        comps_wrong = {
            "account": "other",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "",
            "slug": "",
        }
        assert not check_uri_access(ctx, comps_wrong, strict_mode=False)
        assert not check_uri_access(ctx, comps_wrong, strict_mode=True)


class TestCheckUriAccessStrictMode:
    """AGFS strict mode tests - ignores visible_owner_spaces."""

    def test_own_user_space_allowed(self):
        """User can access their own user space."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared"),  # Ignored
        )
        comps = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "profile",
            "slug": "",
        }

        assert check_uri_access(ctx, comps, strict_mode=True)

    def test_own_agent_space_allowed(self):
        """User can access their own agent space."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "bot",
            "category": "skill",
            "slug": "test",
        }

        assert check_uri_access(ctx, comps, strict_mode=True)

    def test_other_user_denied(self):
        """Strict mode denies other user's space."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "bob",
            "category": "profile",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps, strict_mode=True)

    def test_other_agent_denied(self):
        """Strict mode denies other agent's space."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("agent:shared"),  # Ignored in strict mode
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "shared",
            "category": "skill",
            "slug": "test",
        }

        assert not check_uri_access(ctx, comps, strict_mode=True)

    def test_visible_owner_spaces_ignored_in_strict_mode(self):
        """visible_owner_spaces is not used in strict mode."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared"),
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "shared",
            "category": "",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps, strict_mode=True)


class TestCheckUriAccessSqlRelaxedMode:
    """SQL relaxed mode tests - uses visible_owner_spaces."""

    def test_visible_owner_spaces_enables_shared_agent(self):
        """User can access agent in visible_owner_spaces."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared", "agent:bot"),
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "shared",
            "category": "skill",
            "slug": "test",
        }

        assert check_uri_access(ctx, comps, strict_mode=False)

    def test_visible_owner_spaces_enables_own_user(self):
        """User space in visible_owner_spaces allows access."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared"),
        )
        comps = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "profile",
            "slug": "",
        }

        assert check_uri_access(ctx, comps, strict_mode=False)

    def test_agent_not_in_visible_spaces_denied(self):
        """Agent not in visible_owner_spaces is denied."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice", "agent:shared"),
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "other-bot",
            "category": "",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps, strict_mode=False)

    def test_user_not_in_visible_spaces_denied(self):
        """User not in visible_owner_spaces is denied."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("agent:shared"),  # No user:alice
        )
        comps = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps, strict_mode=False)


class TestCheckUriAccessFallback:
    """Fallback to strict check when visible_owner_spaces is empty."""

    def test_empty_visible_spaces_fallback_to_strict_user(self):
        """Empty visible_owner_spaces uses strict check for user."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=(),
        )
        # Own user: allowed
        comps_own = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "",
            "slug": "",
        }
        assert check_uri_access(ctx, comps_own, strict_mode=False)

        # Other user: denied
        comps_other = {
            "account": "acme",
            "owner_type": "users",
            "owner_id": "bob",
            "category": "",
            "slug": "",
        }
        assert not check_uri_access(ctx, comps_other, strict_mode=False)

    def test_empty_visible_spaces_fallback_to_strict_agent(self):
        """Empty visible_owner_spaces uses strict check for agent."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=(),
        )
        # Own agent: allowed
        comps_own = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "bot",
            "category": "",
            "slug": "",
        }
        assert check_uri_access(ctx, comps_own, strict_mode=False)

        # Other agent: denied
        comps_other = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "shared",
            "category": "",
            "slug": "",
        }
        assert not check_uri_access(ctx, comps_other, strict_mode=False)

    def test_none_visible_spaces_fallback_to_strict(self):
        """Empty visible_owner_spaces is treated as empty tuple."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=(),
        )
        comps = {
            "account": "acme",
            "owner_type": "agents",
            "owner_id": "bot",
            "category": "",
            "slug": "",
        }
        assert check_uri_access(ctx, comps, strict_mode=False)


class TestCheckUriAccessNoOwnerRestriction:
    """URIs without owner_id allow access after account check."""

    def test_no_owner_id_allows_access(self):
        """Session-level URIs (no owner_id) allow access."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "acme",
            "owner_type": "",
            "owner_id": "",
            "category": "history",
            "slug": "session-1",
        }

        assert check_uri_access(ctx, comps, strict_mode=False)
        assert check_uri_access(ctx, comps, strict_mode=True)

    def test_resource_uri_no_owner_allowed(self):
        """Resources (no owner restriction) allow access."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "acme",
            "owner_type": "",
            "owner_id": "",
            "category": "",
            "slug": "",
        }

        assert check_uri_access(ctx, comps)


class TestCheckUriAccessEdgeCases:
    """Edge case tests."""

    def test_unknown_owner_type_denied_in_strict_mode(self):
        """Unknown owner_type is denied by default in strict mode."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "acme",
            "owner_type": "unknown",
            "owner_id": "someone",
            "category": "",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps, strict_mode=True)

    def test_unknown_owner_type_denied_in_relaxed_mode(self):
        """Unknown owner_type is denied in relaxed mode (not in visible_owner_spaces)."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=("user:alice"),
        )
        comps = {
            "account": "acme",
            "owner_type": "unknown",
            "owner_id": "someone",
            "category": "",
            "slug": "",
        }
        # "unknown:someone" not in visible_owner_spaces
        assert not check_uri_access(ctx, comps, strict_mode=False)

    def test_unknown_owner_type_denied_in_fallback(self):
        """Unknown owner_type is denied in fallback (empty visible_owner_spaces)."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
            visible_owner_spaces=(),
        )
        comps = {
            "account": "acme",
            "owner_type": "unknown",
            "owner_id": "someone",
            "category": "",
            "slug": "",
        }
        assert not check_uri_access(ctx, comps, strict_mode=False)

    def test_empty_account_in_components_denied(self):
        """Empty account in components is denied."""
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="bot",
            session_id="s1",
            trace_id="t1",
        )
        comps = {
            "account": "",
            "owner_type": "users",
            "owner_id": "alice",
            "category": "",
            "slug": "",
        }

        assert not check_uri_access(ctx, comps)
