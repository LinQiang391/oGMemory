from __future__ import annotations

import pytest

from core.models import RequestContext
from server.mcp_resources import (
    external_to_internal_uri,
    internal_to_external_uri,
    parse_ogmemory_uri,
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id="bot",
        session_id="s1",
        trace_id="t1",
    )


def test_parse_status_and_types_resources():
    assert parse_ogmemory_uri("ogmemory://status").kind == "status"
    assert parse_ogmemory_uri("ogmemory://types").kind == "types"


def test_external_profile_uri_maps_to_current_user_context():
    assert (
        external_to_internal_uri("ogmemory://memories/profile/self", _ctx())
        == "ctx://acct-1/users/alice/memories/profile"
    )


def test_external_preference_uri_maps_to_current_user_context():
    assert (
        external_to_internal_uri("ogmemory://memories/preference/coding_style", _ctx())
        == "ctx://acct-1/users/alice/memories/preferences/coding_style"
    )


def test_external_tool_memory_uri_maps_to_current_agent_context():
    assert (
        external_to_internal_uri("ogmemory://memories/tool_memory/grep", _ctx())
        == "ctx://acct-1/agents/bot/memories/tools/grep"
    )


def test_external_agent_pattern_uri_maps_to_legacy_agent_context():
    assert (
        external_to_internal_uri("ogmemory://memories/pattern/agent/review", _ctx())
        == "ctx://acct-1/agents/bot/memories/patterns/review"
    )


def test_internal_uri_maps_to_external_without_leaking_ctx_uri():
    assert (
        internal_to_external_uri("ctx://acct-1/users/alice/memories/preferences/coding")
        == "ogmemory://memories/preference/coding"
    )


def test_internal_skill_uri_maps_to_public_skill_type():
    assert (
        internal_to_external_uri("ctx://acct-1/agents/bot/skills/debug_tests")
        == "ogmemory://memories/skill/debug_tests"
    )


def test_legacy_agent_pattern_uri_maps_to_scoped_public_uri():
    assert (
        internal_to_external_uri("ctx://acct-1/agents/bot/memories/patterns/review")
        == "ogmemory://memories/pattern/agent/review"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "ctx://acct-1/users/alice/memories/profile",
        "file:///tmp/memory",
        "ogmemory://memories/preference/../secret",
        "ogmemory://memories/preference/a%2Fb",
        "ogmemory://memories/preference/a\\b",
    ],
)
def test_public_uri_validation_rejects_unsafe_inputs(uri):
    with pytest.raises(ValueError):
        parse_ogmemory_uri(uri)
