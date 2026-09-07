"""Evidence capture and refresh scenarios."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.interfaces import ContextFS
from core.models import ContextNode, RelationEdge, RequestContext
from service.memory_evidence import MemoryEvidenceService


@pytest.fixture
def ctx():
    return RequestContext("evidence-account", "alice", "assistant", "session", "trace")


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
        metadata={"version": 1, "when": "2026-09-04", "provenance_ids": ["session:1:msg:2"]},
    )


@pytest.fixture
def fs(node):
    store = Mock(spec=ContextFS)
    store.read_node.return_value = node
    return store


def test_capture_full_node_and_context(fs, node, ctx):
    snapshot = MemoryEvidenceService(fs).capture([node.uri], ctx, owner_space=node.owner_space)
    data = snapshot.to_dict()

    assert data["schema_version"] == 1
    assert data["account_id"] == ctx.account_id
    assert data["owner_space"] == ctx.user_space_name()
    assert data["nodes"][0]["version"] == 1
    assert data["nodes"][0]["node"]["content"] == node.content
    assert data["nodes"][0]["node"]["metadata"]["provenance_ids"] == ["session:1:msg:2"]
    assert len(data["fingerprint"]) == 64
    fs.read_node.assert_called_once_with(node.uri, ctx)


def test_capture_has_stable_order_and_deduplicates_uris(node, ctx):
    other = replace(node, uri=node.uri + "-next")
    nodes = {node.uri: node, other.uri: other}
    fs = Mock(spec=ContextFS)
    fs.read_node.side_effect = lambda uri, _ctx: nodes[uri]
    service = MemoryEvidenceService(fs)

    first = service.capture([other.uri, node.uri, other.uri], ctx, owner_space=node.owner_space)
    second = service.capture([node.uri, other.uri], ctx, owner_space=node.owner_space)

    assert first == second
    assert [item.uri for item in first.nodes] == sorted(nodes)
    assert fs.read_node.call_count == 4


def test_capture_detaches_nested_metadata_and_export(fs, node, ctx):
    snapshot = MemoryEvidenceService(fs).capture([node.uri], ctx, owner_space=node.owner_space)
    node.metadata["provenance_ids"].append("session:2:msg:1")
    exported = snapshot.to_dict()
    exported["nodes"][0]["node"]["metadata"]["provenance_ids"].append("exported-copy")

    assert snapshot.to_dict()["nodes"][0]["node"]["metadata"]["provenance_ids"] == [
        "session:1:msg:2"
    ]


def test_capture_serializes_relation_evidence(fs, node, ctx):
    node.metadata["_relations"] = [
        RelationEdge(node.uri, node.uri + "-next", "related_to", 1.0, "same project")
    ]
    snapshot = MemoryEvidenceService(fs).capture([node.uri], ctx, owner_space=node.owner_space)

    assert (
        json.loads(snapshot.nodes[0].node_json)["metadata"]["_relations"][0]["reason"]
        == "same project"
    )
    assert json.loads(json.dumps(snapshot.to_dict())) == snapshot.to_dict()


@pytest.mark.parametrize(
    "field", ["abstract", "overview", "content", "version", "when", "provenance_ids"]
)
def test_refresh_identifies_changed_evidence(field, fs, node, ctx):
    service = MemoryEvidenceService(fs)
    snapshot = service.capture([node.uri], ctx, owner_space=node.owner_space)
    if field == "version":
        node.metadata[field] = 2
    elif field == "when":
        node.metadata[field] = "2026-09-05"
    elif field == "provenance_ids":
        node.metadata[field] = ["session:2:msg:1"]
    else:
        setattr(node, field, "更新后的发布安排")

    result = service.check(snapshot, ctx)
    refreshed = service.capture([node.uri], ctx, owner_space=node.owner_space)

    assert result.changed_uris == (node.uri,)
    assert refreshed.fingerprint != snapshot.fingerprint
    assert service.check(refreshed, ctx).is_current


def test_repeated_read_and_metadata_key_order_are_stable(fs, node, ctx):
    service = MemoryEvidenceService(fs)
    snapshot = service.capture([node.uri], ctx, owner_space=node.owner_space)
    node.metadata = dict(reversed(list(node.metadata.items())))

    assert service.capture([node.uri], ctx, owner_space=node.owner_space) == snapshot
    assert service.check(snapshot, ctx).is_current


def test_capture_agent_owned_memory(fs, node, ctx):
    node.uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/events/release"
    node.owner_space = ctx.agent_space_name()
    snapshot = MemoryEvidenceService(fs).capture([node.uri], ctx, owner_space=node.owner_space)

    assert snapshot.owner_space == ctx.agent_space_name()
    assert snapshot.to_dict()["nodes"][0]["node"]["uri"] == node.uri
