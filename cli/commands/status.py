"""ogmem status command - show health of all oG-Memory services."""

import json
import os
from urllib.parse import urlsplit
import subprocess
from pathlib import Path

from cli.lib.colors import GREEN, YELLOW, NC, ok, fail
from cli.lib.health import check_agfs, check_ogmem
from cli.lib.process import find_project_root
from cli.lib.local_config import load_local_config
from cli.lib.local_runtime import running_state


def _detect_docker_mode() -> bool:
    """Check if ogmem/openclaw containers are running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ogmem", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_local_mode(root: Path) -> dict:
    """Use the managed process's port and the live backend when available."""
    state = running_state(root)
    try:
        cfg = load_local_config(root)
    except Exception:
        if not state or "port" not in state:
            raise ValueError("Cannot resolve local configuration or recorded service port") from None
        cfg = None
    port = state.get("port", cfg.http_port if cfg else 8090) if state else cfg.http_port
    health = check_ogmem(port=port) if state else None
    backend = (health or {}).get("storage_backend") or (cfg.storage_backend if cfg else None)
    required = backend == "agfs"
    agfs_health = None
    agfs_url = (health or {}).get("agfs_url") or os.environ.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
    if required:
        url = urlsplit(agfs_url)
        agfs_port = url.port or (443 if url.scheme == "https" else 80)
        if not (health or {}).get("agfs_url"):
            agfs_port = int(os.environ.get("AGFS_PORT", str(agfs_port)))
        agfs_url = f"http://{url.hostname}:{agfs_port}"
        agfs_health = check_agfs(host=url.hostname or "127.0.0.1", port=agfs_port)
    healthy = bool(state and health and health.get("backend") == "og-memory" and health.get("status") == "ok")
    return {"ogmem": bool(state), "ogmem_health": health,
            "agfs": bool(agfs_health), "agfs_health": agfs_health,
            "ogmem_port": port, "agfs_url": agfs_url, "backend": backend,
            "agfs_required": required,
            "healthy": healthy and backend in {"sql", "agfs"} and (not required or bool(agfs_health))}


def _check_docker_mode() -> dict:
    """Check services via docker ps and health endpoints."""
    status = {"agfs": False, "ogmem": False, "agfs_health": None, "ogmem_health": None}

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ogmem", "--filter", "name=openclaw",
             "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        containers = result.stdout.strip().split("\n")
        status["agfs"] = any("openclaw" in c for c in containers)
        status["ogmem"] = any("ogmem" in c for c in containers)

        if status["agfs"]:
            status["agfs_health"] = check_agfs()
        if status["ogmem"]:
            status["ogmem_health"] = check_ogmem()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return status


def run(args) -> int:
    """Show status of all oG-Memory services."""
    root = find_project_root()
    # A project's local PID takes precedence over unrelated containers on the host.
    local_record = root / ".ogmem_data"
    is_docker = not any((local_record / name).exists() for name in ("ce.pid", "local-runtime.json")) and _detect_docker_mode()

    if is_docker:
        status = _check_docker_mode()
        mode_str = f"({YELLOW}Docker{NC})"
    else:
        try:
            status = _check_local_mode(root)
        except (OSError, ValueError, RuntimeError) as exc:
            print(fail(str(exc)))
            return 1
        mode_str = f"({YELLOW}Local{NC})"

    # Print status table
    agfs_status = ok("UP") if status["agfs"] else fail("DOWN")
    ogmem_status = (ok("UP") if status.get("healthy", True) else fail("UNHEALTHY")) if status["ogmem"] else fail("DOWN")

    print(f"\noG-Memory Status {mode_str}\n")
    if status.get("agfs_required", True):
        print(f"  AGFS            {status.get('agfs_url', 'http://127.0.0.1:1833')}  {agfs_status}")
    elif status.get("backend") == "sql":
        print("  Storage         SQL (AGFS not required)")
    else:
        print("  Storage         unknown (health unavailable)")
    print(f"  ContextEngine   http://127.0.0.1:{status.get('ogmem_port', 8090)}  {ogmem_status}")

    # Show health details
    if status["agfs_health"]:
        health_json = json.dumps(status["agfs_health"], separators=(",", ":"))
        print(f"  {GREEN}AGFS Health{NC}     {health_json}")
    elif status["agfs"]:
        print(f"  {YELLOW}AGFS Health{NC}     (unreachable)")

    if status["ogmem_health"]:
        health_json = json.dumps(status["ogmem_health"], separators=(",", ":"))
        print(f"  {GREEN}Health{NC}          {health_json}")
    elif status["ogmem"]:
        print(f"  {YELLOW}Health{NC}          (unreachable)")

    print()

    return 0 if status.get("healthy", status["agfs"] and status["ogmem"]) else 1
