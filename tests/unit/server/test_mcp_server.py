from __future__ import annotations

import anyio

from providers.unified_config import OgMemConfig
from server.mcp_server import _auth_kwargs_from_context, create_mcp_server


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers


class _FakeRequestContext:
    def __init__(self, request):
        self.request = request


class _FakeMCPContext:
    def __init__(self, request):
        self.request_context = _FakeRequestContext(request)


class _FakeService:
    _cfg = OgMemConfig()


def test_create_mcp_server_excludes_context_from_tool_schema():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    tools = anyio.run(mcp.list_tools)
    query_tool = next(tool for tool in tools if tool.name == "ogm.query_memory")
    read_tool = next(tool for tool in tools if tool.name == "ogm.read_memory")

    assert "ctx" not in query_tool.inputSchema.get("properties", {})
    assert "ctx" not in read_tool.inputSchema.get("properties", {})


def test_create_mcp_server_defaults_to_read_only_tools():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == {"ogm.query_memory", "ogm.read_memory"}


def test_create_mcp_server_static_resources_are_readable():
    mcp = create_mcp_server(service=_FakeService(), config=OgMemConfig())

    resources = anyio.run(mcp.list_resources)
    payloads = anyio.run(mcp.read_resource, "ogmemory://status")

    assert "ogmemory://status" in {str(resource.uri) for resource in resources}
    assert '"status": "ok"' in payloads[0].content


def test_create_mcp_server_registers_write_tool_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_write_tools=True),
    )

    tools = anyio.run(mcp.list_tools)
    write_tool = next(tool for tool in tools if tool.name == "ogm.write_memory")

    assert {tool.name for tool in tools} == {
        "ogm.query_memory",
        "ogm.read_memory",
        "ogm.write_memory",
    }
    assert "ctx" not in write_tool.inputSchema.get("properties", {})
    assert "session_id" in write_tool.inputSchema.get("properties", {})


def test_create_mcp_server_registers_delete_tool_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_delete_tools=True),
    )

    tools = anyio.run(mcp.list_tools)
    delete_tool = next(tool for tool in tools if tool.name == "ogm.delete_memory")

    assert {tool.name for tool in tools} == {
        "ogm.query_memory",
        "ogm.read_memory",
        "ogm.delete_memory",
    }
    assert "ctx" not in delete_tool.inputSchema.get("properties", {})
    assert "uri" in delete_tool.inputSchema.get("properties", {})


def test_create_mcp_server_registers_read_write_and_delete_tools_when_enabled():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_enable_write_tools=True, mcp_enable_delete_tools=True),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == {
        "ogm.query_memory",
        "ogm.read_memory",
        "ogm.write_memory",
        "ogm.delete_memory",
    }


def test_create_mcp_server_applies_streamable_http_settings():
    cfg = OgMemConfig(mcp_host="127.0.0.1", mcp_port=8092, mcp_path="/custom-mcp")

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8092
    assert mcp.settings.streamable_http_path == "/custom-mcp"


def test_create_mcp_server_applies_transport_security_allowlists():
    cfg = OgMemConfig(
        mcp_host="127.0.0.1",
        mcp_allowed_hosts=["memory.local"],
        mcp_allowed_origins=["https://agent.example.com"],
    )

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    security = mcp.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["memory.local", "memory.local:*"]
    assert security.allowed_origins == ["https://agent.example.com"]


def test_create_mcp_server_defaults_transport_security_origins_from_hosts():
    cfg = OgMemConfig(mcp_allowed_hosts=["127.0.0.1", "localhost"])

    mcp = create_mcp_server(service=_FakeService(), config=cfg)

    security = mcp.settings.transport_security
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "localhost:*" in security.allowed_hosts
    assert "http://127.0.0.1:*" in security.allowed_origins
    assert "http://localhost:*" in security.allowed_origins


def test_http_context_disables_stdio_env_auth_fallback():
    ctx = _FakeMCPContext(_FakeRequest({"x-api-key": "header-key"}))

    kwargs = _auth_kwargs_from_context(ctx)

    assert kwargs["transport_headers"]["x-api-key"] == "header-key"
    assert kwargs["allow_stdio_env_auth"] is False


def test_missing_context_allows_stdio_env_auth_fallback():
    kwargs = _auth_kwargs_from_context(None)

    assert kwargs["transport_headers"] == {}
    assert kwargs["allow_stdio_env_auth"] is True
