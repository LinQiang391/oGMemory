from __future__ import annotations

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from providers.unified_config import OgMemConfig
from server.mcp_resources import render_resource


class _FakeContextFS:
    def __init__(self, nodes):
        self.nodes = {node.uri: node for node in nodes}

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return self.nodes[uri]

    def exists(self, uri, ctx):
        return uri in self.nodes

    def list_children(self, uri, ctx):
        prefix = uri.rstrip("/") + "/"
        children = []
        for node_uri in sorted(self.nodes):
            if not node_uri.startswith(prefix):
                continue
            suffix = node_uri[len(prefix) :]
            if suffix and "/" not in suffix:
                children.append(node_uri)
        return children


class _FakeService:
    def __init__(self, cfg, nodes):
        self._cfg = cfg
        self.fs = _FakeContextFS(nodes)

    def _get_context_fs(self):
        return self.fs


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id="bot",
        session_id="s1",
        trace_id="t1",
    )


def _node(uri, *, category="preference", content="content"):
    return ContextNode(
        uri=uri,
        context_type="MEMORY",
        category=category,
        level=0,
        owner_space="user:alice",
        abstract="abstract",
        overview="overview",
        content=content,
        metadata={"created_at": "2026-06-23T00:00:00Z", "updated_at": "2026-06-23T01:00:00Z"},
    )


def test_status_resource_reports_implemented_toolsets_only():
    service = _FakeService(
        OgMemConfig(
            mcp_enable_write_tools=True,
            mcp_enable_delete_tools=True,
            mcp_enable_runtime_tools=True,
            mcp_enable_diagnostic_tools=True,
        ),
        [],
    )

    payload = render_resource(service, "ogmemory://status", _ctx())

    assert payload["toolsets"]["read_only"] is True
    assert payload["toolsets"]["write"] is True
    assert payload["toolsets"]["delete"] is True
    assert payload["toolsets"]["runtime"] is False
    assert payload["toolsets"]["diagnostic"] is False


def test_types_resource_exposes_public_memory_types_without_internal_types():
    service = _FakeService(OgMemConfig(mcp_enable_write_tools=True), [])

    payload = render_resource(service, "ogmemory://types", _ctx())
    types = {row["type"]: row for row in payload["types"]}

    assert {"profile", "preference", "entity", "event", "case", "pattern", "skill", "tool_memory"} <= set(types)
    assert "session_archive" not in types
    assert "session_summary" not in types
    assert types["preference"]["write_tool_enabled"] is True
    assert types["preference"]["delete_tool_enabled"] is False

    enabled_service = _FakeService(OgMemConfig(mcp_enable_delete_tools=True), [])
    enabled_types = {
        row["type"]: row
        for row in render_resource(enabled_service, "ogmemory://types", _ctx())["types"]
    }
    assert enabled_types["preference"]["delete_tool_enabled"] is True
    assert enabled_types["profile"]["deletable_type"] is False


def test_collection_resource_lists_external_summaries_only():
    service = _FakeService(
        OgMemConfig(),
        [
            _node("ctx://acct-1/users/alice/memories/preferences/coding_style"),
        ],
    )

    payload = render_resource(service, "ogmemory://memories/preference", _ctx())

    assert payload["uri"] == "ogmemory://memories/preference"
    assert payload["memories"] == [
        {
            "uri": "ogmemory://memories/preference/coding_style",
            "type": "preference",
            "id": "coding_style",
            "abstract": "abstract",
            "updated_at": "2026-06-23T01:00:00Z",
            "has_content": True,
        }
    ]
    assert "ctx://" not in str(payload)


def test_item_resource_reads_external_uri_detail():
    service = _FakeService(
        OgMemConfig(),
        [
            _node("ctx://acct-1/users/alice/memories/preferences/coding_style"),
        ],
    )

    payload = render_resource(
        service,
        "ogmemory://memories/preference/coding_style",
        _ctx(),
    )

    assert payload["uri"] == "ogmemory://memories/preference/coding_style"
    assert payload["abstract"] == "abstract"
    assert payload["overview"] == "overview"
    assert payload["content"] == "content"
    assert "ctx://" not in str(payload)
