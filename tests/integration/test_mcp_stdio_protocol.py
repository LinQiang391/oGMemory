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


@pytest.mark.integration
def test_mcp_stdio_protocol_exposes_tools_and_resources():
    anyio.run(_run_stdio_protocol_check)


async def _run_stdio_protocol_check() -> None:
    env = os.environ.copy()
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": "false",
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": "false",
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
            assert tool_names == {"ogm.query_memory", "ogm.read_memory"}

            read_result = await session.call_tool("ogm.read_memory", {"uri": "ogmemory://types"})
            read_payload = _tool_json(read_result)
            assert "types" in read_payload

            query_result = await session.call_tool("ogm.query_memory", {"query": "anything", "top_k": 1})
            query_payload = _tool_json(query_result)
            assert query_payload["hits"] == []

            resources = await session.list_resources()
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert {"ogmemory://status", "ogmemory://types"} <= resource_uris

            templates = await session.list_resource_templates()
            template_uris = {template.uriTemplate for template in templates.resourceTemplates}
            assert "ogmemory://memories/{memory_type}" in template_uris
            assert "ogmemory://memories/{memory_type}/{memory_id}" in template_uris

            status_payload = _resource_json(await session.read_resource("ogmemory://status"))
            types_payload = _resource_json(await session.read_resource("ogmemory://types"))
            assert status_payload["status"] == "ok"
            assert "types" in types_payload


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    return json.loads(content.text)


def _resource_json(result: Any) -> dict[str, Any]:
    content = result.contents[0]
    return json.loads(content.text)
