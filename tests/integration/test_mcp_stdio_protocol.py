from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

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
def test_mcp_stdio_protocol_exposes_tools_and_resources():
    anyio.run(_run_stdio_protocol_check)


async def _run_stdio_protocol_check() -> None:
    env = _clean_subprocess_env()
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": "all",
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": "true",
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": "true",
            "OGMEM_CONFIG": str(ROOT / "__missing_mcp_test_config__.yaml"),
            "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false",
        }
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cli.main", "mcp", "--transport", "stdio"],
        cwd=str(ROOT),
        env=env,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "oGMemory"

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert tool_names == {
                "ogm_query_memory",
                "ogm_read_memory",
                "ogm_write_memory",
                "ogm_delete_memory",
            }
            assert all("." not in name for name in tool_names)

            read_result = await session.call_tool("ogm_read_memory", {"uri": "ogmemory://types"})
            read_payload = _tool_json(read_result)
            assert "types" in read_payload

            query_result = await session.call_tool("ogm_query_memory", {"query": "anything", "top_k": 1})
            query_payload = _tool_json(query_result)
            assert query_payload["hits"] == []

            legacy_calls = (
                ("ogm.query_memory", {"query": "legacy alias must be rejected", "top_k": 1}),
                ("ogm.read_memory", {"uri": "ogmemory://types"}),
                (
                    "ogm.write_memory",
                    {
                        "session_id": "legacy-name-rejection",
                        "text": "legacy alias must be rejected",
                        "wait": False,
                        "client_request_id": "mcp-e2e-legacy-write-1",
                    },
                ),
                (
                    "ogm.delete_memory",
                    {
                        "uri": "ogmemory://memories/preference/legacy-name-rejection",
                        "reason": "legacy alias must be rejected",
                        "client_request_id": "mcp-e2e-legacy-delete-1",
                    },
                ),
            )
            for tool_name, arguments in legacy_calls:
                legacy_result = await session.call_tool(tool_name, arguments)
                assert legacy_result.isError is True
                error_text = "\n".join(
                    content.text
                    for content in legacy_result.content
                    if getattr(content, "type", None) == "text"
                )
                assert "Unknown tool" in error_text
                assert tool_name in error_text

            resources = await session.list_resources()
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert {"ogmemory://status", "ogmemory://types"} <= resource_uris

            templates = await session.list_resource_templates()
            template_uris = {template.uriTemplate for template in templates.resourceTemplates}
            assert "ogmemory://memories/{memory_type}" in template_uris
            assert "ogmemory://memories/{memory_type}/{memory_id}" in template_uris
            assert "ogmemory://memories/pattern/agent/{memory_id}" in template_uris

            status_payload = _resource_json(await session.read_resource("ogmemory://status"))
            types_payload = _resource_json(await session.read_resource("ogmemory://types"))
            assert status_payload["status"] == "ok"
            assert "types" in types_payload


def _clean_subprocess_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENV_PASSTHROUGH
    }


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    return json.loads(content.text)


def _resource_json(result: Any) -> dict[str, Any]:
    content = result.contents[0]
    return json.loads(content.text)
