"""ogmem mcp command - run the oGMemory MCP server."""

from __future__ import annotations

from server.mcp_server import run_mcp_server


def run(args) -> int:
    run_mcp_server(
        transport=args.transport,
        config_path=args.config,
        host=args.host,
        port=args.port,
        path=args.path,
    )
    return 0
