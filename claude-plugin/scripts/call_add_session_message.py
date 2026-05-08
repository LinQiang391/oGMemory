#!/usr/bin/env python3
"""Claude Code hook: PostToolUse — append tool I/O to oG-Memory session buffer.

After each successful tool call, serializes tool_name / tool_input / tool_response
into one message with role ``tool`` and POSTs to /api/v1/sessions/<id>/messages.
Complements call_after_turn (user/assistant from transcript) with tool context.

HTTP headers and identity: :mod:`ogm_plugin_request`.

Input  (stdin):  PostToolUse JSON (Claude Code hooks)
Output (stdout):  none (exit 0); errors logged to stderr only
"""

import json
import sys
from pathlib import Path
import urllib.error
import urllib.request

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ogm_plugin_request import base_api_url, base_ctx, http_plugin_headers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Must stay below hooks.json PostToolUse timeout (10s) so HTTP fails before the hook is killed.
POST_TIMEOUT = 8

# Only mutating / side-effect tools — keep in sync with hooks.json PostToolUse matcher.
_SIDE_EFFECT_TOOLS = frozenset(
    {"Write", "Edit", "MultiEdit", "Bash", "NotebookEdit"}
)
# Truncate very large tool payloads
_MAX_CHUNK = 10_000
_PREVIEW = 400


def log(msg: str) -> None:
    print(f"[call_add_session_message] {msg}", file=sys.stderr)


def _truncate(text: str, max_len: int = _MAX_CHUNK) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n… [{len(text) - max_len} more chars omitted]"


def _json_chunk(obj: object) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(obj)
    return _truncate(raw)


def _build_tool_text(data: dict) -> str:
    name = data.get("tool_name") or "unknown_tool"
    t_in = data.get("tool_input")
    t_out = data.get("tool_response")
    lines = [
        f"[PostToolUse] {name}",
        f"tool_input: {_json_chunk(t_in)}",
        f"tool_response: {_json_chunk(t_out)}",
    ]
    return "\n".join(lines)


def post_session_message(session_id: str, role: str, content: str) -> bool:
    url = f"{base_api_url()}/api/v1/sessions/{session_id}/messages"
    body = {**base_ctx(session_id), "role": role, "content": content}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers=http_plugin_headers(),
        method="POST",
    )
    log(f"POST {url} role={role} content_len={len(content)}")
    try:
        with urllib.request.urlopen(req, timeout=POST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            log(f"Response {resp.status}: {raw[:200]}")
            return 200 <= resp.status < 300
    except urllib.error.URLError as e:
        log(f"HTTP error: {e}")
        return False
    except Exception as e:
        log(f"Request failed: {e}")
        return False


def _preview_val(obj: object) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        s = str(obj)
    if len(s) > _PREVIEW:
        s = s[:_PREVIEW] + f"…(+{len(s) - _PREVIEW})"
    return s


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        log("Invalid JSON on stdin")
        sys.exit(0)

    session_id = str(data.get("session_id") or "unknown")
    if session_id == "unknown":
        log("No session_id, skip")
        sys.exit(0)

    tool_name = data.get("tool_name") or "unknown_tool"
    if tool_name not in _SIDE_EFFECT_TOOLS:
        log(f"skip non-side-effect tool (matcher widened or stale config): {tool_name}")
        sys.exit(0)

    t_in = data.get("tool_input")
    t_out = data.get("tool_response")
    log(
        f"tool={tool_name}  input: {_preview_val(t_in)}  output: {_preview_val(t_out)}"
    )

    text = _build_tool_text(data)
    if not text.strip():
        sys.exit(0)

    ok = post_session_message(session_id, "tool", text)
    if not ok:
        log(f"Failed to post tool message for session={session_id}, tool={tool_name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
