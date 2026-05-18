"""ogmem start command - launch services."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from cli.lib.colors import CYAN, NC, YELLOW, fail, ok, warn
from cli.lib.health import check_agfs, check_ogmem
from cli.lib.process import ensure_data_dirs, find_project_root, spawn_process, start_and_wait, write_pid

def _load_env(root: Path) -> dict[str, str]:
    env = {}
    if (root / "config" / ".env").exists():
        for line in (root / "config" / ".env").read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def start_headless(args) -> int:
    root = find_project_root()
    ensure_data_dirs(root)
    os.environ.update(_load_env(root))
    env, d = os.environ, root / ".ogmem_data"
    agfs_dir = Path(env.get("AGFS_SERVER_DIR", str(root / "agfs")))
    agfs_bin = Path(env.get("AGFS_SERVER_BIN", str(agfs_dir / "build" / "agfs-server")))
    agfs_cfg = Path(env.get("AGFS_SERVER_CONFIG", str(agfs_dir / "config.yaml")))
    url = env.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
    agfs_port = int(env.get("AGFS_PORT", url.rsplit(":", 1)[-1].rstrip("/")))
    try:
        from providers.unified_config import get_config
        ce_port = get_config().http_port
    except Exception:
        ce_port = int(env.get("OGMEM_HTTP_PORT", "8090"))
    dry, daemon = getattr(args, "dry_run", False), getattr(args, "daemon", False)
    def start(name, cmd, pidf, logf, port, kind, cwd):
        if dry:
            print(f"Would start: {' '.join(cmd)}")
            return True
        try:
            pid = start_and_wait(cmd, pidf, logf, "127.0.0.1", port, kind, cwd=cwd)
            print(ok(f"{name} up (PID {pid})"))
            return True
        except RuntimeError as e:
            print(fail(str(e)))
            return False
    if check_agfs(port=agfs_port):
        print(ok(f"AGFS already running on :{agfs_port}"))
    elif dry:
        print(f"Would start AGFS: {agfs_bin} -c {agfs_cfg} -addr :{agfs_port}")
    elif not (agfs_bin.exists() and os.access(agfs_bin, os.X_OK)):
        print(fail(f"AGFS binary not found: {agfs_bin}"))
        print(f"  Set AGFS_SERVER_BIN to override, or install AGFS first")
        return 1
    elif not start("AGFS", [str(agfs_bin), "-c", str(agfs_cfg), "-addr", f":{agfs_port}"], d / "agfs.pid", d / "logs" / "agfs.log", agfs_port, "agfs", agfs_dir):
        return 1
    if check_ogmem(port=ce_port):
        print(ok(f"ContextEngine already running on :{ce_port}"))
    elif not start("ContextEngine", [sys.executable, "server/app.py"], d / "ce.pid", d / "logs" / "context_engine.log", ce_port, "ogmem", root):
        return 1
    print(f"\n  AGFS            http://127.0.0.1:{agfs_port}  UP\n  ContextEngine   http://127.0.0.1:{ce_port}  UP\n")
    if not dry:
        print(f"  Logs: {CYAN}.ogmem_data/logs/agfs.log{NC}  {CYAN}.ogmem_data/logs/context_engine.log{NC}\n  Stop: {YELLOW}ogmem stop local{NC}")
    if daemon or dry:
        print(ok("Started in daemon mode" if daemon else "Dry run complete"))
        return 0
    print(f"\n  {YELLOW}Ctrl+C to stop{NC}")
    def shutdown(s, f):
        print("\nShutting down...")
        from cli.lib.process import stop_service
        stop_service(d / "agfs.pid", agfs_port)
        stop_service(d / "ce.pid", ce_port)
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)
    return 0


def start_docker(args) -> int:
    """Start services using Docker."""
    from cli.lib.docker_ops import docker_available, docker_pull
    from cli.lib.colors import fail, ok

    root = find_project_root()
    deploy_env = root / "deploy.env"

    if not deploy_env.exists():
        print(fail(f"deploy.env not found: {deploy_env}"))
        return 1

    # Parse deploy.env
    env = {}
    for line in deploy_env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    # Validate required vars
    required = ["LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OGMEM_EMBEDDING_MODEL"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(fail(f"Missing required vars in deploy.env: {', '.join(missing)}"))
        return 1

    if not docker_available():
        print(fail("Docker is not available or not running"))
        return 1

    skip_pull = getattr(args, "skip_pull", False)

    # Print what would be deployed
    print(ok(f"Docker deploy from: {deploy_env}"))
    print(f"  LLM_PROVIDER: {env['LLM_PROVIDER']}")
    print(f"  LLM_MODEL: {env['LLM_MODEL']}")

    og_enabled = env.get("ENABLE_OPENGAUSS", "false").lower() == "true"
    print(f"  openGauss: {'enabled' if og_enabled else 'disabled'}")

    if not skip_pull:
        print(f"  Would pull images...")
    else:
        print(f"  Skip pull (--skip-pull)")

    print(f"\n  Would deploy:")
    if og_enabled:
        print(f"    1. openGauss container")
    print(f"    {'1' if not og_enabled else '2'}. ogmemory container")
    print(f"    {'2' if not og_enabled else '3'}. openclaw-ogmemory container")

    print(ok("\nDocker mode structure ready (full deployment pending)"))
    return 0
def start_plugin(args) -> int:
    """Start CE server for plugin mode (OpenClaw or Claude Code hooks).

    Reads config/ogmem.yaml for plugin.type and service.base_url.
    Starts CE if needed, then runs plugin-type-specific setup.
    """
    from urllib.parse import urlparse
    from cli.lib.health import check_ogmem, wait_for_healthy

    root = find_project_root()
    ensure_data_dirs(root)
    os.environ.update(_load_env(root))
    dry = getattr(args, "dry_run", False)

    # Step 1: Load config — read YAML directly for plugin section
    # (get_config() returns flat OgMemConfig which doesn't include plugin fields)
    import yaml
    plugin_cfg = {}
    config_file = root / "config" / "ogmem.yaml"
    if config_file.exists():
        try:
            raw = yaml.safe_load(config_file.read_text()) or {}
            plugin_cfg = raw.get("plugin", {})
        except Exception:
            pass
    plugin_type = plugin_cfg.get("type", "")

    try:
        from providers.unified_config import get_config
        cfg = get_config()
        base_url = f"http://127.0.0.1:{cfg.http_port}" if hasattr(cfg, "http_port") else "http://127.0.0.1:8090"
    except Exception:
        base_url = os.environ.get("OGMEM_API_URL", "http://127.0.0.1:8090")

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8090

    print(f"Plugin mode ({CYAN}{plugin_type or 'unknown'}{NC}) — ContextEngine API: {CYAN}{base_url}{NC}\n")

    # Step 2: Start CE if not running (local only)
    if check_ogmem(host, port):
        print(ok(f"ContextEngine already running at {base_url}"))
    else:
        if host not in ("127.0.0.1", "localhost"):
            print(fail(f"Remote CE at {base_url} not reachable"))
            print(f"  Ensure the oG-Memory server is running at that address")
            return 1
        if dry:
            print(f"Would start ContextEngine on :{port}")
        else:
            d = root / ".ogmem_data"
            try:
                pid = start_and_wait(
                    [sys.executable, "server/app.py"],
                    d / "ce.pid", d / "logs" / "context_engine.log",
                    host, port, "ogmem", cwd=root,
                )
                print(ok(f"ContextEngine started (PID {pid})"))
            except RuntimeError as e:
                print(fail(str(e)))
                return 1

    # Step 3: Health check
    health_timeout = getattr(args, "health_timeout", 30)
    if not dry and not wait_for_healthy(host, port, "ogmem", timeout=health_timeout):
        print(fail("ContextEngine health check failed"))
        return 1
    if not dry:
        health = check_ogmem(host, port) or {}
        print(ok(f"ContextEngine healthy at {base_url}"))
        if health.get("storage"):
            print(f"  Backend: {health.get('storage', '?')}, SQL: {health.get('sql', '?')}")

    # Step 4: Plugin-type-specific flow
    print()
    if plugin_type.startswith("openclaw"):
        _start_openclaw_plugin(root, plugin_cfg, dry)
    elif plugin_type == "claude_hooks":
        _start_claude_hooks_plugin(root, dry)
    else:
        # Fallback: detect from files
        _start_plugin_autodetect(root, dry)

    return 0


def _start_openclaw_plugin(root: Path, plugin_cfg: dict, dry: bool) -> None:
    """OpenClaw plugin flow: validate config, start gateway if available."""
    from cli.lib.colors import info as _info
    import shutil

    openclaw_config = plugin_cfg.get("openclaw_config", "")
    config_path = Path(openclaw_config) if openclaw_config else Path.cwd() / "openclaw.plugin.json"

    if not config_path.exists():
        print(warn(f"OpenClaw config not found: {config_path}"))
        print(f"  Run {CYAN}ogmem onboard --mode plugin{NC} to generate it")
        return

    # Validate config
    import json
    try:
        gw_cfg = json.loads(config_path.read_text())
        gw_port = gw_cfg.get("gateway", {}).get("port", "18789")
        plugins = gw_cfg.get("plugins", {}).get("entries", {})
        ce_slot = next(iter(plugins.values()), {})
        ce_url = ce_slot.get("config", {}).get("memoryApiBaseUrl", "?")
        print(ok(f"OpenClaw gateway config: {config_path}"))
        print(f"  Gateway port: {gw_port}, CE URL: {ce_url}")
    except (json.JSONDecodeError, KeyError) as e:
        print(warn(f"Could not parse gateway config: {e}"))
        return

    # Try to start OpenClaw gateway
    openclaw_bin = shutil.which("openclaw")
    if openclaw_bin:
        if dry:
            print(f"Would start OpenClaw: {openclaw_bin} --config {config_path}")
        else:
            print(f"\n  Starting OpenClaw gateway...")
            proc = spawn_process(
                [openclaw_bin, "--config", str(config_path)],
                log_file=root / ".ogmem_data" / "logs" / "openclaw.log",
                cwd=root,
            )
            write_pid(root / ".ogmem_data" / "openclaw.pid", proc.pid)
            print(ok(f"OpenClaw started (PID {proc.pid})"))
            print(f"  Log: {CYAN}.ogmem_data/logs/openclaw.log{NC}")
    else:
        print(f"\n  OpenClaw binary not found in PATH")
        print(f"  Start manually:")
        print(f"    {CYAN}openclaw --config {config_path}{NC}")


def _start_claude_hooks_plugin(root: Path, dry: bool) -> None:
    """Claude Code hooks flow: install hooks into project settings."""
    import json
    claude_plugin = root / "claude-plugin"
    hooks_json = claude_plugin / "hooks" / "hooks.json"

    if not claude_plugin.exists():
        print(warn(f"Claude plugin directory not found: {claude_plugin}"))
        return

    if not hooks_json.exists():
        print(warn(f"hooks.json not found: {hooks_json}"))
        return

    # Read hooks template and resolve ${CLAUDE_PLUGIN_ROOT} to absolute paths
    hooks_cfg = json.loads(hooks_json.read_text())
    scripts_dir = (claude_plugin / "scripts").resolve()
    for event_name, matchers in hooks_cfg.get("hooks", {}).items():
        for matcher in matchers:
            for hook in matcher.get("hooks", []):
                if "command" in hook:
                    hook["command"] = hook["command"].replace(
                        "${CLAUDE_PLUGIN_ROOT}", str(claude_plugin.resolve())
                    )

    # Write into project .claude/settings.json
    settings_dir = root / ".claude"
    settings_file = settings_dir / "settings.json"
    settings = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}

    # Merge hooks — don't overwrite existing hooks, merge per-event
    existing_hooks = settings.get("hooks", {})
    for event_name, matchers in hooks_cfg["hooks"].items():
        if event_name not in existing_hooks:
            existing_hooks[event_name] = matchers
        else:
            # Append new matchers that aren't already registered
            existing_commands = set()
            for m in existing_hooks[event_name]:
                for h in m.get("hooks", []):
                    existing_commands.add(h.get("command", ""))
            for matcher in matchers:
                matcher_cmds = {h.get("command", "") for h in matcher.get("hooks", [])}
                if not matcher_cmds.intersection(existing_commands):
                    existing_hooks[event_name].append(matcher)

    settings["hooks"] = existing_hooks

    if dry:
        print(f"Would write hooks to {settings_file}")
    else:
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
        print(ok(f"Claude Code hooks installed to {settings_file}"))
        hook_count = sum(len(m) for ms in hooks_cfg["hooks"].values() for m in ms)
        print(f"  {hook_count} hooks registered ({', '.join(hooks_cfg['hooks'].keys())})")

    print(f"\n  Launch Claude Code:")
    print(f"    {CYAN}claude{NC}  (hooks auto-loaded from .claude/settings.json)")
    print(f"  Or with plugin dir:")
    print(f"    {CYAN}claude --plugin-dir {claude_plugin}{NC}")


def _start_plugin_autodetect(root: Path, dry: bool) -> None:
    """Fallback: detect plugin type from existing files and run appropriate flow."""
    openclaw_json = Path.cwd() / "openclaw.plugin.json"
    claude_plugin = root / "claude-plugin"
    hooks_json = claude_plugin / "hooks" / "hooks.json"

    if openclaw_json.exists():
        _start_openclaw_plugin(root, {"openclaw_config": str(openclaw_json)}, dry)
    if hooks_json.exists():
        _start_claude_hooks_plugin(root, dry)
    if not openclaw_json.exists() and not hooks_json.exists():
        print(warn("No plugin config detected"))
        print(f"  Run {CYAN}ogmem onboard{NC} with plugin mode to configure")
def start_index(args) -> int:
    root = find_project_root()
    ensure_data_dirs(root)
    dry, daemon = getattr(args, "dry_run", False), getattr(args, "daemon", False)
    idx_script = root / "scripts" / "run_index_service.py"
    if not idx_script.exists():
        print(fail(f"Index script not found: {idx_script}"))
        return 1
    if dry:
        print(f"Would start: {sys.executable} {idx_script}")
        return 0
    log_file, pid_file = root / ".ogmem_data" / "logs" / "index.log", root / ".ogmem_data" / "index.pid"
    proc = spawn_process([sys.executable, str(idx_script)], log_file=log_file, cwd=root)
    write_pid(pid_file, proc.pid)
    print(ok(f"Index worker started (PID {proc.pid})"))
    print(f"  Log: {CYAN}{log_file}{NC}")
    if daemon:
        return 0
    print(f"\n  {YELLOW}Ctrl+C to stop{NC}")
    signal.signal(signal.SIGINT, lambda s, f: os.kill(proc.pid, signal.SIGTERM))
    signal.signal(signal.SIGTERM, lambda s, f: os.kill(proc.pid, signal.SIGTERM))
    proc.wait()
    return 0
def run(args) -> int:
    mode = getattr(args, "mode", None) or "headless"
    f = {"headless": start_headless, "local": start_headless, "docker": start_docker, "plugin": start_plugin, "index": start_index}.get(mode)
    if f:
        return f(args)
    print(fail(f"Unknown mode: {mode}"))
    return 1
