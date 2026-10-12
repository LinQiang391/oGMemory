"""ogmem stop command - stop services."""

from cli.lib.colors import fail, ok
from cli.lib.process import find_project_root, stop_service


def stop_local(args) -> int:
    """Stop all local services."""
    root = find_project_root()
    from cli.lib.local_runtime import stop_recorded_service
    try:
        stop_recorded_service(root)
    except (RuntimeError, OSError, ValueError) as exc:
        print(fail(str(exc)))
        return 1
    # Tracked processes can still be stopped when the saved YAML is invalid.
    # Never kill an arbitrary listener based on a newly configured port.
    if (root / ".ogmem_data" / "agfs.pid").exists():
        stop_service(root / ".ogmem_data" / "agfs.pid", 0)

    # Stop index worker
    stop_service(root / ".ogmem_data" / "index.pid", 0)  # Port 0 = skip port kill

    print(ok("All services stopped"))
    return 0


def run(args) -> int:
    """Stop services managed by the local project checkout."""
    mode = getattr(args, "mode", None) or "local"

    if mode in {"headless", "local"}:
        return stop_local(args)
    print(f"Unknown mode: {mode}")
    return 1
