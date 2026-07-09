"""Unified CLI for oG-Memory — main entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from cli import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogmem",
        description="oG-Memory — unified management CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- onboard --
    p_onboard = sub.add_parser("onboard", help="Interactive setup wizard")
    p_onboard.add_argument("--non-interactive", action="store_true", help="Run without prompts")
    p_onboard.add_argument("--mode", choices=["docker", "plugin", "headless"], help="Deployment mode")
    p_onboard.add_argument("--provider", help="LLM provider (openai/volcengine/dashscope/zhipu/mock)")
    p_onboard.add_argument("--api-key", help="LLM API key")
    p_onboard.add_argument("--base-url", help="LLM API base URL")
    p_onboard.add_argument("--model", help="LLM model name")
    p_onboard.add_argument("--embedding-model", help="Embedding model name")
    p_onboard.add_argument("--embedding-provider", help="Embedding provider")
    p_onboard.add_argument("--vector-db", choices=["chroma", "opengauss"], help="Vector DB type")
    p_onboard.add_argument("--port", type=int, default=8090, help="HTTP port")
    p_onboard.add_argument("--storage-backend", choices=["agfs", "sql"], default="sql", help="Storage backend (default: sql)")
    p_onboard.add_argument("--db-connection", help="PostgreSQL connection string (SQL backend)")

    # -- start --
    p_start = sub.add_parser("start", help="Start services")
    p_start.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["headless", "local", "docker", "plugin", "index"],
        help="What to start",
    )
    p_start.add_argument("-d", "--daemon", action="store_true", help="Run in background")
    p_start.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_start.add_argument("-c", "--config", help="Path to ogmem.yaml")
    p_start.add_argument("--agfs-bin", help="Override AGFS binary path")
    p_start.add_argument("--health-timeout", type=int, default=30, help="Health check timeout (seconds)")
    # docker-specific
    p_start.add_argument("--env-file", help="Path to deploy.env (docker mode)")
    p_start.add_argument("--skip-pull", action="store_true", help="Skip image pull (docker mode)")
    p_start.add_argument("--password", "-p", help="openGauss password (docker mode)")

    # -- stop --
    p_stop = sub.add_parser("stop", help="Stop services")
    p_stop.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["headless", "local", "docker", "plugin"],
        help="What to stop",
    )

    # -- status --
    sub.add_parser("status", help="Show service status")

    # -- mcp --
    p_mcp = sub.add_parser("mcp", help="Run MCP server")
    p_mcp.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    p_mcp.add_argument("-c", "--config", help="Path to ogmem.yaml")
    p_mcp.add_argument("--host", default=None, help="Host for streamable-http transport")
    p_mcp.add_argument("--port", type=int, default=None, help="Port for streamable-http transport")
    p_mcp.add_argument("--path", default=None, help="Path for streamable-http transport")

    # -- check --
    sub.add_parser("check", help="Environment diagnostics")

    # -- logs --
    p_logs = sub.add_parser("logs", help="Tail service logs")
    p_logs.add_argument("--service", choices=["agfs", "ogmem", "openclaw"], help="Which service logs")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")

    # -- eval --
    p_eval = sub.add_parser("eval", help="Run LoCoMo eval pipeline")
    p_eval.add_argument("eval_mode", choices=["ingest", "qa", "judge", "stat"], help="Eval step")
    p_eval.add_argument("input", help="Input file (JSON or CSV)")
    p_eval.add_argument("--output", "-o", help="Output path")
    p_eval.add_argument("--base-url", default="http://127.0.0.1:18789")
    p_eval.add_argument("--token", help="Auth token")
    p_eval.add_argument("--sample", type=int, help="Sample index")
    p_eval.add_argument("--count", type=int, help="Number of questions")
    p_eval.add_argument("-p", "--parallel", type=int, help="Parallel workers")

    # -- config --
    p_config = sub.add_parser("config", help="Configuration management")
    config_sub = p_config.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="Show resolved config (masked)")
    config_sub.add_parser("init", help="Generate starter ogmem.yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch
    cmd = args.command

    if cmd == "check":
        from cli.commands.check import run
        return run(args)
    elif cmd == "status":
        from cli.commands.status import run
        return run(args)
    elif cmd == "mcp":
        from cli.commands.mcp import run
        return run(args)
    elif cmd == "start":
        from cli.commands.start import run
        return run(args)
    elif cmd == "stop":
        from cli.commands.stop import run
        return run(args)
    elif cmd == "onboard":
        from cli.commands.onboard import run
        return run(args)
    elif cmd == "logs":
        from cli.commands.logs import run
        return run(args)
    elif cmd == "eval":
        from cli.commands.eval_cmd import run
        return run(args)
    elif cmd == "config":
        from cli.commands.config import run
        return run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
