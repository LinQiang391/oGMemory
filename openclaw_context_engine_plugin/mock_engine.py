#!/usr/bin/env python3
"""
mock_engine.py — Mock Context Engine for OpenClaw plugin development.

Usage (called automatically by index.js):
    python3 mock_engine.py <method> '<json_params>'

Every lifecycle method prints its received parameters to stdout for
debugging, then returns a minimal valid JSON response on the last line.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from bridge.memory_api import (
    memory_store,
    memory_search,
    memory_delete,
)

LOG_LEVEL = os.environ.get("MOCK_ENGINE_LOG_LEVEL", "debug")

# ── helpers ──────────────────────────────────────────────────────────

def _log(level: str, msg: str) -> None:
    """Print a timestamped log line to stderr for visibility in gateway logs."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[mock-engine {ts}] [{level.upper()}] {msg}", file=sys.stderr, flush=True)


def _dump_params(method: str, params: dict) -> None:
    """Pretty-print the incoming params dict."""
    _log("debug", f"──── {method}() called ────")
    _log("debug", f"params = {json.dumps(params, indent=2, default=str)}")


def _respond(data: dict) -> None:
    """Print the JSON return value as the last line of stdout."""
    print(json.dumps(data, default=str), flush=True)


# ── lifecycle methods ────────────────────────────────────────────────

def ingest(params: dict) -> dict:
    """Called when a new message is added to the session."""
    _dump_params("ingest", params)

    session_id = params.get("sessionId", "unknown")
    message = params.get("message", {})
    is_heartbeat = params.get("isHeartbeat", False)

    _log("info", f"ingest  sessionId={session_id}  role={message.get('role', '?')}  heartbeat={is_heartbeat}")

    # 模拟存储到 memory bridge
    memory_store(
        key=f"session:{session_id}:msg:{message.get('id', 'none')}",
        value=message,
    )

    return {"ingested": True}


def assemble(params: dict) -> dict:
    """Called before each model run. Return messages + token estimate."""
    _dump_params("assemble", params)

    session_id = params.get("sessionId", "unknown")
    messages = params.get("messages", [])
    token_budget = params.get("tokenBudget", 128_000)

    _log("info", f"assemble  sessionId={session_id}  input_msgs={len(messages)}  budget={token_budget}")

    # Mock: 模拟一次 memory search
    memory_search(query=f"session:{session_id}", top_k=5)

    # Mock: 直接透传所有 messages，不做裁剪
    return {
        "messages": messages,
        "estimatedTokens": len(messages) * 200,  # 粗略估算
        "systemPromptAddition": (
            "[mock-context-engine] This context was assembled by the mock engine. "
            "All original messages are passed through without modification."
        ),
    }


def compact(params: dict) -> dict:
    """Called when context window is full or user runs /compact."""
    _dump_params("compact", params)

    session_id = params.get("sessionId", "unknown")
    force = params.get("force", False)

    _log("info", f"compact  sessionId={session_id}  force={force}")

    return {"ok": True, "compacted": True, "method": "mock-noop"}


def bootstrap(params: dict) -> dict:
    """Initialize engine state for a session (called once per session)."""
    _dump_params("bootstrap", params)
    _log("info", f"bootstrap  sessionId={params.get('sessionId', 'unknown')}")
    return {"bootstrapped": True}


def ingest_batch(params: dict) -> dict:
    """Ingest a completed turn as a batch."""
    _dump_params("ingest_batch", params)

    messages = params.get("messages", [])
    _log("info", f"ingest_batch  count={len(messages)}")

    for msg in messages:
        memory_store(
            key=f"batch:{msg.get('id', 'none')}",
            value=msg,
        )

    return {"ingested": True, "count": len(messages)}


def after_turn(params: dict) -> dict:
    """Post-run lifecycle work."""
    _dump_params("after_turn", params)
    _log("info", f"after_turn  sessionId={params.get('sessionId', 'unknown')}")
    return {"ok": True}


def prepare_subagent_spawn(params: dict) -> dict:
    """Set up shared state for a child session (future use)."""
    _dump_params("prepare_subagent_spawn", params)
    _log("info", f"prepare_subagent_spawn  parentSession={params.get('parentSessionId', '?')}")
    return {"prepared": True}


def on_subagent_ended(params: dict) -> dict:
    """Clean up after a subagent ends."""
    _dump_params("on_subagent_ended", params)

    child_session = params.get("childSessionId", "unknown")
    _log("info", f"on_subagent_ended  childSession={child_session}")

    # 模拟清理 memory
    memory_delete(key=f"session:{child_session}:*")

    return {"cleaned": True}


def dispose(params: dict) -> dict:
    """Release resources during gateway shutdown."""
    _dump_params("dispose", params)
    _log("info", "dispose  releasing resources")
    return {"disposed": True}


# ── CLI entry point ──────────────────────────────────────────────────

METHODS = {
    "ingest": ingest,
    "assemble": assemble,
    "compact": compact,
    "bootstrap": bootstrap,
    "ingest_batch": ingest_batch,
    "after_turn": after_turn,
    "prepare_subagent_spawn": prepare_subagent_spawn,
    "on_subagent_ended": on_subagent_ended,
    "dispose": dispose,
}


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: mock_engine.py <method> '<json_params>'", file=sys.stderr)
        sys.exit(1)

    method_name = sys.argv[1]
    raw_params = sys.argv[2]

    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        _log("error", f"Invalid JSON params: {exc}")
        _respond({"error": str(exc)})
        sys.exit(1)

    handler = METHODS.get(method_name)
    if handler is None:
        _log("error", f"Unknown method: {method_name}")
        _respond({"error": f"unknown method: {method_name}"})
        sys.exit(1)

    result = handler(params)
    _respond(result)


if __name__ == "__main__":
    main()
