#!/usr/bin/env python3
"""CLI wrapper for the M3-Bench evaluation harness.

Thin entry point that delegates to ``m3.eval``. Lives in ``scripts/``
so operators can run the eval as a standalone command, mirroring the
existing ``scripts/cv_service.py`` / ``scripts/perf_test.py`` /
``scripts/run_e2e_long_context_*.py`` pattern.

Usage::

    ./scripts/m3_eval.py \\
        --annotations data/annotations/robot.json \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --embedder openai --llm minimax --judge-llm openai \\
        --output-dir data/m3_eval/$(date +%Y-%m-%d)

The actual evaluation logic (adapter invocation, judge scoring,
per-type aggregation, resumable runs) lives in :mod:`m3.eval`. This
script is a thin shim that sets up logging and forwards CLI args.
"""

from __future__ import annotations

import sys

from m3.eval import main


if __name__ == "__main__":
    sys.exit(main())