"""ogmem eval command - run evaluation tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(args) -> int:
    """Run evaluation tests via tests/e2e/eval.py."""
    eval_script = Path(__file__).parent.parent.parent / "tests" / "e2e" / "eval.py"

    if not eval_script.exists():
        print(f"Error: eval script not found: {eval_script}")
        return 1

    # Build command args
    cmd = [sys.executable, str(eval_script), args.eval_mode, args.input]

    if args.output:
        cmd.extend(["--output", args.output])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if hasattr(args, "sample") and args.sample is not None:
        cmd.extend(["--sample", str(args.sample)])
    if hasattr(args, "sessions") and args.sessions:
        cmd.extend(["--sessions", args.sessions])
    if hasattr(args, "count") and args.count is not None:
        cmd.extend(["--count", str(args.count)])
    if hasattr(args, "parallel") and args.parallel:
        cmd.extend(["--parallel", str(args.parallel)])
    if hasattr(args, "user") and args.user:
        cmd.extend(["--user", args.user])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error running eval: {e}")
        return 1
