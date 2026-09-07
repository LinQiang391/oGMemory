"""Plan input revision tracking scenarios."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.input_versions import collect_input_versions, find_changed_inputs
from core.interfaces import ContextFS
from core.models import ContextNode, RequestContext


@pytest.fixture
def ctx():
    return RequestContext("revision-account", "alice", "assistant", "session", "trace")


@pytest.fixture
def node(ctx):
    return ContextNode(
        uri=f"ctx://{ctx.account_id}/users/alice/memories/events/release",
        context_type="MEMORY",
        category="events",
        level=4,
        owner_space=ctx.user_space_name(),
        abstract="项目 A 周五发布",
        overview="发布安排",
        content="项目 A 于 2026-09-04 发布。",
        metadata={"version": 1, "provenance_ids": ["session:1:msg:2"]},
    )


def test_collect_versions_from_planning_inputs(node):
    versions = collect_input_versions([node])
    assert versions == {node.uri: 1}
    assert json.loads(json.dumps(versions)) == versions


def test_collect_sorts_and_deduplicates_inputs(node):
    other = replace(node, uri=node.uri + "-next")
    versions = collect_input_versions([other, node, other])
    assert list(versions.items()) == [(node.uri, 1), (other.uri, 1)]


def test_recorded_versions_survive_changes_to_planning_nodes(node):
    versions = collect_input_versions([node])
    node.metadata["version"] = 2
    assert versions == {node.uri: 1}
    assert collect_input_versions([node]) == {node.uri: 2}


@pytest.mark.parametrize("owner", ["user", "agent"])
def test_recheck_current_inputs(node, ctx, owner):
    if owner == "agent":
        node.uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/events/release"
        node.owner_space = ctx.agent_space_name()
    fs = Mock(spec=ContextFS)
    fs.read_node.return_value = node
    assert find_changed_inputs(fs, collect_input_versions([node]), ctx) == {}
    fs.read_node.assert_called_once_with(node.uri, ctx)


def test_recheck_reports_current_revision_for_changed_input(node, ctx):
    versions = collect_input_versions([node])
    node.metadata["version"] = 2
    fs = Mock(spec=ContextFS)
    fs.read_node.return_value = node
    assert find_changed_inputs(fs, versions, ctx) == {node.uri: 2}
    assert find_changed_inputs(fs, collect_input_versions([node]), ctx) == {}


def test_recheck_identifies_changed_member_in_batch(node, ctx):
    other = replace(node, uri=node.uri + "-next", metadata={"version": 1})
    versions = collect_input_versions([node, other])
    other.metadata["version"] = 3
    nodes = {node.uri: node, other.uri: other}
    fs = Mock(spec=ContextFS)
    fs.read_node.side_effect = lambda uri, _ctx: nodes[uri]
    assert find_changed_inputs(fs, versions, ctx) == {other.uri: 3}
