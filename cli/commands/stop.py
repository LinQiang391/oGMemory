"""ogmem stop command - stop services."""

from pathlib import Path

from cli.lib.colors import ok
from cli.lib.process import find_project_root, stop_service


def stop_local(args) -> int:
    """Stop all local services."""
    root = find_project_root()

    # Get ports from env or defaults
    import os
    agfs_url = os.environ.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
    if ":" in agfs_url.rsplit(":", 1)[-1]:
        agfs_port = int(agfs_url.rsplit(":", 1)[-1].rstrip("/"))
    else:
        agfs_port = 1833
    agfs_port = int(os.environ.get("AGFS_PORT", str(agfs_port)))
    ce_port = int(os.environ.get("OGMEM_HTTP_PORT", "8090"))

    # Stop AGFS
    stop_service(root / ".ogmem_data" / "agfs.pid", agfs_port)

    # Stop ContextEngine
    stop_service(root / ".ogmem_data" / "ce.pid", ce_port)

    # Stop index worker
    stop_service(root / ".ogmem_data" / "index.pid", 0)  # Port 0 = skip port kill

    print(ok("All services stopped"))
    return 0


def stop_docker(args) -> int:
    """Stop Docker services."""
    from cli.lib.colors import ok

    print("Would stop Docker containers:")
    print("  - ogmemory")
    print("  - openclaw-ogmemory")
    print("  - openGauss (if enabled)")

    print(ok("\nDocker stop structure ready (full implementation pending)"))
    return 0


def stop_plugin(args) -> int:
    """Stop plugin service (not yet implemented)."""
    print("Plugin mode not yet implemented")
    return 1


def run(args) -> int:
    """ogmem stop [local|docker|plugin]"""
    mode = getattr(args, "mode", "local")

    if mode == "local":
        return stop_local(args)
    elif mode == "docker":
        return stop_docker(args)
    elif mode == "plugin":
        return stop_plugin(args)
    else:
        print(f"Unknown mode: {mode}")
        return 1
