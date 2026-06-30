from __future__ import annotations

import asyncio
import anyio
import pytest
from starlette.testclient import TestClient

from core.models import Role
from providers.unified_config import OgMemConfig
from server.auth import ResolvedIdentity
from server.mcp_server import _auth_kwargs_from_context, create_mcp_http_app, create_mcp_server, run_mcp_server


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


class _FakeAuth:
    def __init__(self):
        self.headers = None

    def role_control_active(self):
        return True

    def resolve_identity(self, headers):
        self.headers = headers
        return ResolvedIdentity(role=Role.ROOT, account_id="acct-auth", user_id="user-auth")


class _FakeDisposeService:
    _cfg = OgMemConfig()

    def __init__(self):
        self.auth = _FakeAuth()
        self.build_context_params = None
        self.build_context_identity = None
        self.dispose_params = None
        self.idle_params = None

    def get_auth_service(self):
        return self.auth

    def build_context(self, params, identity=None):
        self.build_context_params = dict(params)
        self.build_context_identity = identity
        return "ctx-from-same-service"

    def dispose(self, params):
        self.dispose_params = dict(params)
        return {"disposed": True, "ctx": params["_ctx"]}

    def wait_until_idle(self, params):
        self.idle_params = dict(params)
        return {"idle": True, "ctx": params["_ctx"]}


class _FakeThreadCheckingDisposeService(_FakeDisposeService):
    def wait_until_idle(self, params):
        self.idle_params = dict(params)
        try:
            asyncio.get_running_loop()
            running_loop = True
        except RuntimeError:
            running_loop = False
        return {"idle": True, "running_loop": running_loop, "ctx": params["_ctx"]}


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


def test_create_mcp_server_read_write_toolset_registers_write_only():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="read-write"),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == {
        "ogm.query_memory",
        "ogm.read_memory",
        "ogm.write_memory",
    }


def test_create_mcp_server_all_toolset_registers_write_and_delete():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="all"),
    )

    tools = anyio.run(mcp.list_tools)

    assert {tool.name for tool in tools} == {
        "ogm.query_memory",
        "ogm.read_memory",
        "ogm.write_memory",
        "ogm.delete_memory",
    }


def test_create_mcp_server_instructions_describe_generic_memory_timing():
    mcp = create_mcp_server(
        service=_FakeService(),
        config=OgMemConfig(mcp_default_toolset="all"),
    )

    tools = {tool.name: tool for tool in anyio.run(mcp.list_tools)}
    instructions = mcp.instructions.lower()

    assert "persistent memory tools" in instructions
    assert "before answering" in instructions
    assert "not present in the current prompt" in instructions
    assert "information that may be needed later" in instructions
    assert "durable context" in instructions
    assert "interaction text or messages" in instructions
    assert "decide what to persist" in instructions
    assert "ogm.query_memory" in instructions
    assert "ogm.read_memory" in instructions
    assert "ogm.write_memory" in instructions
    assert "ogm.delete_memory" in instructions
    assert "people" not in instructions
    assert "preferences" not in instructions
    assert "relationships" not in instructions
    assert "locomo" not in instructions
    assert "benchmark" not in instructions
    assert "chat transcript" not in instructions
    assert "before answering" in tools["ogm.query_memory"].description.lower()
    assert "user-provided information" in tools["ogm.write_memory"].description.lower()
    assert "raw text/messages" in tools["ogm.write_memory"].description.lower()


def test_run_mcp_server_rejects_disabled_config(monkeypatch):
    monkeypatch.setattr(
        "server.mcp_server.get_config",
        lambda config_path=None: OgMemConfig(mcp_enabled=False),
    )

    with pytest.raises(RuntimeError, match="MCP server is disabled"):
        run_mcp_server()


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


def test_mcp_http_app_exposes_dispose_on_same_service():
    service = _FakeDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/dispose",
        json={"sessionId": "s1", "userId": "eval-user", "agentId": "eval-agent"},
        headers={"Authorization": "Bearer root-key", "X-Account-ID": "acct-1", "X-User-ID": "eval-user"},
    )

    assert response.status_code == 200
    assert response.json() == {"disposed": True, "ctx": "ctx-from-same-service"}
    assert service.auth.headers["authorization"] == "Bearer root-key"
    assert service.build_context_params == {"sessionId": "s1", "userId": "eval-user", "agentId": "eval-agent"}
    assert service.build_context_identity.role == Role.ROOT
    assert service.dispose_params["_ctx"] == "ctx-from-same-service"


def test_mcp_http_app_exposes_idle_wait_on_same_service():
    service = _FakeDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "userId": "eval-user", "agentId": "eval-agent"},
        headers={"Authorization": "Bearer root-key", "X-Account-ID": "acct-1", "X-User-ID": "eval-user"},
    )

    assert response.status_code == 200
    assert response.json() == {"idle": True, "ctx": "ctx-from-same-service"}
    assert service.idle_params["_ctx"] == "ctx-from-same-service"


def test_mcp_http_app_runs_idle_wait_outside_event_loop():
    service = _FakeThreadCheckingDisposeService()
    mcp = create_mcp_server(service=service, config=OgMemConfig())
    app = create_mcp_http_app(mcp, service)

    response = TestClient(app).post(
        "/api/v1/idle",
        json={"sessionId": "s1", "userId": "eval-user", "agentId": "eval-agent"},
        headers={"Authorization": "Bearer root-key", "X-Account-ID": "acct-1", "X-User-ID": "eval-user"},
    )

    assert response.status_code == 200
    assert response.json()["running_loop"] is False


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
