from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
_ENV_PASSTHROUGH = {
    "COMSPEC",
    "HOME",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
}


@pytest.mark.integration
def test_mcp_streamable_http_protocol_exposes_tools_and_resources():
    anyio.run(_run_streamable_http_protocol_check)


@pytest.mark.integration
def test_mcp_sse_protocol_exposes_tools_and_resources():
    anyio.run(_run_sse_protocol_check)


@pytest.mark.integration
def test_mcp_sse_rejects_missing_auth_for_tools_and_resources():
    anyio.run(_run_sse_auth_rejection_check)


@pytest.mark.integration
def test_mcp_sse_cli_path_overrides_config():
    anyio.run(_run_sse_cli_path_override_check)


@pytest.mark.integration
def test_mcp_sse_rejects_oversized_post():
    anyio.run(_run_sse_oversized_post_check)


@pytest.mark.integration
def test_mcp_streamable_http_protocol_exposes_all_tools_when_enabled():
    anyio.run(_run_streamable_http_all_tools_check)


@pytest.mark.integration
def test_mcp_streamable_http_protocol_can_call_all_tools_when_enabled():
    anyio.run(_run_streamable_http_all_tools_call_check)


async def _run_streamable_http_protocol_check() -> None:
    port = _free_port()
    proc = _start_http_server(port)
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    init = await session.initialize()
                    assert init.serverInfo.name == "oGMemory"

                    tools = await session.list_tools()
                    assert {tool.name for tool in tools.tools} == {
                        "ogm.query_memory",
                        "ogm.read_memory",
                    }
                    resources = await session.list_resources()
                    assert "ogmemory://status" in {
                        str(resource.uri) for resource in resources.resources
                    }
                    templates = await session.list_resource_templates()
                    assert "ogmemory://memories/{memory_type}" in {
                        template.uriTemplate for template in templates.resourceTemplates
                    }

                    read_result = await session.call_tool(
                        "ogm.read_memory",
                        {"uri": "ogmemory://types"},
                    )
                    assert "types" in _tool_json(read_result)

                    status_payload = _resource_json(
                        await session.read_resource("ogmemory://status")
                    )
                    assert status_payload["status"] == "ok"
    finally:
        _terminate(proc)


async def _run_sse_protocol_check() -> None:
    port = _free_port()
    path = "/custom-sse"
    proc = _start_http_server(port, transport="sse", env_path=path)
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        async with sse_client(
            f"http://127.0.0.1:{port}{path}",
            headers=headers,
            timeout=30,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "oGMemory"

                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "ogm.query_memory",
                    "ogm.read_memory",
                }
                resources = await session.list_resources()
                assert "ogmemory://status" in {
                    str(resource.uri) for resource in resources.resources
                }

                read_result = await session.call_tool(
                    "ogm.read_memory",
                    {"uri": "ogmemory://types"},
                )
                assert not read_result.isError
                assert "types" in _tool_json(read_result)
    finally:
        _terminate(proc)


async def _run_sse_auth_rejection_check() -> None:
    port = _free_port()
    proc = _start_http_server(port, transport="sse", env_path="/custom-sse")
    try:
        _wait_for_port(port, proc)
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "GET",
                f"http://127.0.0.1:{port}/custom-sse",
            ) as response:
                assert response.status_code == 401
                assert await response.aread() == b'{"error":"authentication_required"}'
    finally:
        _terminate(proc)


async def _run_sse_cli_path_override_check() -> None:
    port = _free_port()
    proc = _start_http_server(
        port,
        transport="sse",
        path="/cli-sse",
        env_path="/configured-sse",
    )
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        async with sse_client(
            f"http://127.0.0.1:{port}/cli-sse",
            headers=headers,
            timeout=30,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                read_result = await session.call_tool(
                    "ogm.read_memory",
                    {"uri": "ogmemory://types"},
                )
                assert not read_result.isError
                assert "types" in _tool_json(read_result)
    finally:
        _terminate(proc)


async def _run_sse_oversized_post_check() -> None:
    port = _free_port()
    max_request_bytes = 4096
    proc = _start_http_server(
        port,
        transport="sse",
        env_path="/custom-sse",
        max_request_bytes=max_request_bytes,
    )
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        session_ids: list[str] = []
        async with sse_client(
            f"http://127.0.0.1:{port}/custom-sse",
            headers=headers,
            timeout=30,
            on_session_created=session_ids.append,
        ):
            assert len(session_ids) == 1
            async with httpx.AsyncClient(headers=headers, timeout=30) as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/messages/",
                    params={"session_id": session_ids[0]},
                    content=b"x" * (max_request_bytes + 1),
                )

            assert response.status_code == 413
            assert response.json() == {"error": "request_too_large"}
    finally:
        _terminate(proc)


async def _run_streamable_http_all_tools_check() -> None:
    port = _free_port()
    proc = _start_http_server(port, default_toolset="all")
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    assert {tool.name for tool in tools.tools} == {
                        "ogm.query_memory",
                        "ogm.read_memory",
                        "ogm.write_memory",
                        "ogm.delete_memory",
                    }

                    status_payload = _resource_json(
                        await session.read_resource("ogmemory://status")
                    )
                    assert status_payload["toolsets"] == {
                        "read_only": True,
                        "write": True,
                        "delete": True,
                        "runtime": False,
                        "diagnostic": False,
                    }
    finally:
        _terminate(proc)


async def _run_streamable_http_all_tools_call_check() -> None:
    port = _free_port()
    proc = _start_http_server(port, default_toolset="all")
    try:
        _wait_for_port(port, proc)
        headers = {
            "X-API-Key": "root-key",
            "X-Account-ID": "acct-1",
            "X-User-ID": "root",
        }
        async with httpx.AsyncClient(headers=headers, timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    query_result = await session.call_tool(
                        "ogm.query_memory",
                        {"query": "nothing has been written for this account", "top_k": 1},
                    )
                    assert not query_result.isError
                    assert _tool_json(query_result)["hits"] == []

                    read_result = await session.call_tool(
                        "ogm.read_memory",
                        {"uri": "ogmemory://types"},
                    )
                    assert not read_result.isError
                    assert "types" in _tool_json(read_result)

                    write_result = await session.call_tool(
                        "ogm.write_memory",
                        {
                            "session_id": "mcp-streamable-http-all-tools",
                            "text": "Remember that OpenClaw must be able to call every oGMemory MCP tool.",
                            "wait": False,
                            "client_request_id": "mcp-e2e-write-1",
                        },
                    )
                    assert not write_result.isError
                    write_payload = _tool_json(write_result)
                    assert write_payload["ok"] is True
                    assert write_payload["accepted_messages"] == 1
                    assert write_payload["client_request_id"] == "mcp-e2e-write-1"

                    delete_result = await session.call_tool(
                        "ogm.delete_memory",
                        {
                            "uri": "ogmemory://memories/preference",
                            "reason": "verify delete tool validation through Streamable HTTP",
                            "client_request_id": "mcp-e2e-delete-1",
                        },
                    )
                    assert delete_result.isError is True
                    assert "single ogmemory" in delete_result.content[0].text
    finally:
        _terminate(proc)


@pytest.mark.integration
def test_mcp_streamable_http_rejects_missing_auth_for_tools_and_resources():
    anyio.run(_run_streamable_http_auth_rejection_check)


async def _run_streamable_http_auth_rejection_check() -> None:
    port = _free_port()
    proc = _start_http_server(port)
    try:
        _wait_for_port(port, proc)
        async with httpx.AsyncClient(timeout=30) as http_client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tool_result = await session.call_tool(
                        "ogm.read_memory",
                        {"uri": "ogmemory://types"},
                    )
                    assert tool_result.isError is True
                    assert "Missing API Key" in tool_result.content[0].text

                    with pytest.raises(Exception, match="Missing API Key"):
                        await session.read_resource("ogmemory://memories/preference")
    finally:
        _terminate(proc)


def _start_http_server(
    port: int,
    *,
    transport: str = "streamable-http",
    path: str | None = None,
    env_path: str | None = None,
    max_request_bytes: int | None = None,
    default_toolset: str = "read-only",
    enable_write_tools: bool | None = None,
    enable_delete_tools: bool | None = None,
) -> subprocess.Popen:
    env = _clean_subprocess_env()
    if enable_write_tools is None:
        enable_write_tools = default_toolset in {"read-write", "all", "full"}
    if enable_delete_tools is None:
        enable_delete_tools = default_toolset in {"all", "full"}
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": default_toolset,
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": str(enable_write_tools).lower(),
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": str(enable_delete_tools).lower(),
            "OG_ROLE_CONTROL_ENABLED": "true",
            "OG_ROOT_API_KEY": "root-key",
            "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false",
            "OGMEM_AFTER_TURN_THRESHOLD": "100000",
            "OGMEM_CONFIG": str(ROOT / "__missing_mcp_test_config__.yaml"),
        }
    )
    if env_path is not None:
        env["OGMEM_MCP_PATH"] = env_path
    if max_request_bytes is not None:
        env["OGMEM_MCP_MAX_REQUEST_BYTES"] = str(max_request_bytes)
    args = [
        sys.executable,
        "-m",
        "cli.main",
        "mcp",
        "--transport",
        transport,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if path is not None:
        args.extend(["--path", path])
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _clean_subprocess_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() in _ENV_PASSTHROUGH}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(
                f"MCP HTTP server exited early with {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"MCP HTTP server did not listen on port {port}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    try:
        return json.loads(content.text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool result is not JSON: {content.text!r}") from exc


def _resource_json(result: Any) -> dict[str, Any]:
    content = result.contents[0]
    return json.loads(content.text)
