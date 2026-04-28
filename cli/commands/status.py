"""ogmem status command - show health of all oG-Memory services."""

import json
import subprocess
from pathlib import Path

from cli.lib.colors import GREEN, RED, YELLOW, NC, ok, fail, warn
from cli.lib.health import check_agfs, check_ogmem, port_alive
from cli.lib.process import find_project_root, read_pid, is_process_alive


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
    """Check services via PID files and health endpoints."""
    status = {"agfs": False, "ogmem": False, "agfs_health": None, "ogmem_health": None}

    # Check AGFS
    agfs_pid = read_pid(root / ".ogmem_data" / "agfs.pid")
    if agfs_pid and is_process_alive(agfs_pid):
        status["agfs"] = True
        status["agfs_health"] = check_agfs()

    # Check oG-Memory
    ogmem_pid = read_pid(root / ".ogmem_data" / "ce.pid")
    if ogmem_pid and is_process_alive(ogmem_pid):
        status["ogmem"] = True
        status["ogmem_health"] = check_ogmem()

    return status


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
    is_docker = _detect_docker_mode()

    if is_docker:
        status = _check_docker_mode()
        mode_str = f"({YELLOW}Docker{NC})"
    else:
        status = _check_local_mode(root)
        mode_str = f"({YELLOW}Local{NC})"

    # Print status table
    agfs_status = ok("UP") if status["agfs"] else fail("DOWN")
    ogmem_status = ok("UP") if status["ogmem"] else fail("DOWN")

    print(f"\noG-Memory Status {mode_str}\n")
    print(f"  AGFS            http://127.0.0.1:1833  {agfs_status}")
    print(f"  ContextEngine   http://127.0.0.1:8090  {ogmem_status}")

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

    return 0 if (status["agfs"] and status["ogmem"]) else 1
