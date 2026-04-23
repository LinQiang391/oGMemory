#!/usr/bin/env python3
"""Claude Code hook: call oG-Memory compose on UserPromptSubmit.

Calls oG-Memory /api/v1/compose with the user's prompt, extracts
identityContext (profile) and retrievedEvidence (working set), and injects
them as additionalContext into Claude's context.

Usage: called by Claude Code UserPromptSubmit hook (stdin = hook JSON).
"""

import json
import os
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config (reuse OG_MEMORY_URL from call_after_turn.py, no new env vars)
# ---------------------------------------------------------------------------
OG_MEMORY_URL = os.environ.get("OG_MEMORY_URL", "http://localhost:8090")
COMPOSE_URL = f"{OG_MEMORY_URL}/api/v1/compose"
COMPOSE_TIMEOUT = 5          # seconds
MAX_CONTEXT_CHARS = 9500     # leave 500 for JSON wrapper
MIN_PROMPT_LEN = 4           # skip queries shorter than this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[call_compose] {msg}", file=sys.stderr)


def call_compose(prompt, session_id):
    """POST to oG-Memory compose endpoint. Returns parsed JSON or None."""
    payload = json.dumps({
        "prompt": prompt,
        "sessionId": session_id,
    }).encode("utf-8")

    req = urllib.request.Request(
        COMPOSE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    log(f"POST {COMPOSE_URL} session={session_id} prompt_len={len(prompt)}")

    try:
        with urllib.request.urlopen(req, timeout=COMPOSE_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            log(f"Response {resp.status}: {body[:200]}")
            return json.loads(body)
    except urllib.error.URLError as e:
        log(f"HTTP error: {e}")
        return None
    except Exception as e:
        log(f"Request failed: {e}")
        return None


def build_additional_context(data):
    """Extract identityContext + retrievedEvidence, truncate to MAX_CONTEXT_CHARS."""
    parts = []

    identity = (data.get("identityContext") or "").strip()
    if identity:
        parts.append(identity)

    evidence = (data.get("retrievedEvidence") or "").strip()
    if evidence:
        parts.append(evidence)

    if not parts:
        return None

    # Header + body
    body = "\n\n".join(parts)

    total = body
    if len(total) > MAX_CONTEXT_CHARS:
        total = total[:MAX_CONTEXT_CHARS]

    return f"[oG-Memory]\n{total}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Read hook input from stdin
    hook_input = sys.stdin.read()
    try:
        hook = json.loads(hook_input)
    except json.JSONDecodeError:
        log("Failed to parse hook input JSON")
        sys.exit(0)

    prompt = (hook.get("prompt") or "").strip()
    session_id = hook.get("session_id", "unknown")

    # 2. Skip short / empty / slash commands
    if len(prompt) < MIN_PROMPT_LEN:
        log(f"Prompt too short ({len(prompt)} chars), skipping")
        sys.exit(0)

    if prompt.startswith("/"):
        log("Slash command, skipping")
        sys.exit(0)

    # 3. Call compose
    data = call_compose(prompt, session_id)
    if data is None:
        sys.exit(0)

    # 4. Build additionalContext
    additional = build_additional_context(data)
    if additional is None:
        log("No relevant context returned")
        sys.exit(0)

    # 5. Output JSON (method B: additionalContext)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional,
        }
    }
    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
