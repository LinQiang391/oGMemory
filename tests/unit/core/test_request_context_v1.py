"""Tests for V1 RequestContext extensions."""

from core.models import RequestContext, Role


def test_request_context_defaults_role_and_visible_owner_spaces():
    ctx = RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )

    assert ctx.role == Role.MEMBER
    assert ctx.visible_owner_spaces == ()


def test_request_context_accepts_explicit_role():
    ctx = RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
        role=Role.ADMIN,
        visible_owner_spaces=("user:user", "agent:agent"),
    )

    assert ctx.role == Role.ADMIN
    assert ctx.visible_owner_spaces == ("user:user", "agent:agent")
