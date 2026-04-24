"""
bridge/memory_api.py — OpenClaw Plugin Bridge to oG-Memory

CLI entry point called by index.js (execFileSync).
Delegates all business logic to server.memory_service.MemoryService.

Architecture (all-in-one mode):
    index.js (Node.js) → execFileSync → bridge/memory_api.py → MemoryService → oG-Memory internals

Architecture (standalone mode):
    index.js (Node.js) → HTTP fetch → server/app.py → MemoryService → oG-Memory internals
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Python path setup (needed when called as subprocess by index.js)
# ---------------------------------------------------------------------------
_file_dir = Path(__file__).parent.parent  # openclaw_context_engine_plugin/
_project_root = _file_dir.parent          # ContextEngine/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ['no_proxy'] = "localhost,127.0.0.1"

# Force HuggingFace offline mode so cached models don't hit network
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Set OGMEM_CONFIG so MemoryService picks up ogmem.yaml
_cfg_file = Path(__file__).parent.parent.parent / "ogmem.yaml"
if _cfg_file.exists() and not os.environ.get("OGMEM_CONFIG"):
    os.environ["OGMEM_CONFIG"] = str(_cfg_file)

from server.memory_service import MemoryService

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_service: MemoryService | None = None


def _get_service() -> MemoryService:
    global _service
    if _service is None:
        from providers.unified_config import get_config
        cfg = get_config()
        _service = MemoryService(config=cfg)
        _log("info", f"MemoryService initialized  agfs={cfg.agfs_base_url}  "
             f"mount={cfg.agfs_mount_prefix}  acct={cfg.account_id}")
    return _service


# ---------------------------------------------------------------------------
# Logging / CLI helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _log(level: str, msg: str) -> None:
    print(f"[memory-api {_ts()}] [{level.upper()}] {msg}", file=sys.stderr, flush=True)


def _respond(data: dict) -> None:
    """Print the JSON return value as the last line of stdout."""
    print(json.dumps(data, default=str), flush=True)


# ============================================================================
# OpenClaw Lifecycle Methods
# ============================================================================

def ingest(params: dict) -> dict:
    _log("info", f"ingest  sessionId={params.get('sessionId', '?')}")
    return _get_service().ingest(params)


def compose(params: dict) -> dict:
    session_id = params.get("sessionId", "unknown")
    _log("info", f"compose  sessionId={session_id}  msgs={len(params.get('messages', []))}")
    result = _get_service().compose(params)
    hits_count = result.get("systemPromptAddition", "").count("相关度:")
    _log("info", f"assemble  memory search returned {hits_count} results")
    return result


def compact(params: dict) -> dict:
    _log("info", f"compact  sessionId={params.get('sessionId', '?')}")
    return _get_service().compact(params)


def prepare_compaction(params: dict) -> dict:
    _log("info", f"prepare_compaction  sessionId={params.get('sessionId', '?')}")
    result = _get_service().prepare_compaction(params)
    return {
        "ok": True,
        "prepared": True,
        "prepareToken": result.get("prepareToken", ""),
        "messagesPrepared": len(result.get("messages", [])),
    }


def bootstrap(params: dict) -> dict:
    _log("info", f"bootstrap  sessionId={params.get('sessionId', '?')}")
    return _get_service().bootstrap(params)


def ingest_batch(params: dict) -> dict:
    _log("info", f"ingest_batch  count={len(params.get('messages', []))}")
    return _get_service().ingest_batch(params)


def after_turn(params: dict) -> dict:
    """Delegate to MemoryService.after_turn().

    Unified write entry point: accumulates session buffer, extracts memories
    and writes to AGFS + vector index when token threshold is reached.
    Same logic for both HTTP and subprocess modes.
    """
    session_id = params.get("sessionId", "unknown")
    _log("info", f"after_turn  sessionId={session_id}  msgs={len(params.get('messages', []))}")
    return _get_service().after_turn(params)


def prepare_subagent_spawn(params: dict) -> dict:
    _log("info", f"prepare_subagent_spawn  parent={params.get('parentSessionId', '?')}")
    return _get_service().prepare_subagent_spawn(params)


def on_subagent_ended(params: dict) -> dict:
    _log("info", f"on_subagent_ended  child={params.get('childSessionId', '?')}")
    return _get_service().on_subagent_ended(params)


def dispose(params: dict) -> dict:
    _log("info", "dispose  releasing resources")
    return _get_service().dispose(params)


def memory_health() -> dict:
    return _get_service().health()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

METHODS = {
    "ingest": ingest,
    "compose": compose,
    "compact": compact,
    "prepare_compaction": prepare_compaction,
    "bootstrap": bootstrap,
    "ingest_batch": ingest_batch,
    "after_turn": after_turn,
    "prepare_subagent_spawn": prepare_subagent_spawn,
    "on_subagent_ended": on_subagent_ended,
    "dispose": dispose,
    "memory_health": lambda params: memory_health(),
}


def main() -> None:
    """CLI entry point called by index.js via execFileSync."""
    if len(sys.argv) < 3:
        print("Usage: memory_api.py <method> '<json_params>'", file=sys.stderr)
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

    try:
        result = handler(params)
        _respond(result)
    except Exception as e:
        _log("error", f"Handler {method_name} failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        _respond({"error": str(e), "method": method_name})
        sys.exit(1)


if __name__ == "__main__":
    main()
