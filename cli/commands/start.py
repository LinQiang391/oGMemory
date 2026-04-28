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
    if (root / ".env").exists():
        for line in (root / ".env").read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def start_local(args) -> int:
    root = find_project_root()
    ensure_data_dirs(root)
    os.environ.update(_load_env(root))
    env, d = os.environ, root / ".ogmem_data"
    agfs_dir = Path(env.get("AGFS_SERVER_DIR", str(root / "agfs")))
    agfs_bin = Path(env.get("AGFS_SERVER_BIN", str(agfs_dir / "build" / "agfs-server")))
    agfs_cfg = Path(env.get("AGFS_SERVER_CONFIG", str(agfs_dir / "config.yaml")))
    url = env.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
    agfs_port = int(env.get("AGFS_PORT", url.rsplit(":", 1)[-1].rstrip("/")))
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
    elif not (agfs_bin.exists() and os.access(agfs_bin, os.X_OK)):
        print(fail(f"AGFS binary not found: {agfs_bin}"))
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
    print(warn("Plugin mode not yet implemented"))
    return 1
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
    mode = getattr(args, "mode", "local")
    f = {"local": start_local, "docker": start_docker, "plugin": start_plugin, "index": start_index}.get(mode)
    if f:
        return f(args)
    print(fail(f"Unknown mode: {mode}"))
    return 1
