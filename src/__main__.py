"""Entry point for python -m ogmemory: run the CLI in project root."""

import sys
from pathlib import Path

# Project root = parent of package dir (src when package-dir ogmemory=src)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Run the same CLI as ogmemory_cli.py
import ogmemory_cli

ogmemory_cli.main()
