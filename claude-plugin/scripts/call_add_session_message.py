#!/usr/bin/env python3
"""Capture Claude Code lifecycle traces and append mutating tool I/O.

Every registered hook event is posted to ``/api/v1/trace/events``. Successful
side-effect tool calls are also appended to the existing session message buffer.

HTTP headers and identity: :mod:`ogm_plugin_request`.

Input  (stdin):  Claude Code hook JSON
Output (stdout):  none (exit 0); errors logged to stderr only
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ogm_plugin_request import base_api_url, base_ctx, http_plugin_headers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Combined worst case stays below hooks.json PostToolUse timeout (10s).
TRACE_POST_TIMEOUT = 0.25
MESSAGE_POST_TIMEOUT = 6
TRACE_CIRCUIT_BREAK_SECONDS = 30

# Only mutating / side-effect tools are also copied to the session message buffer.
_SIDE_EFFECT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "Bash", "NotebookEdit"})
# Truncate very large tool payloads
_MAX_CHUNK = 10_000
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.IGNORECASE
)
_SECRET_NAME = (
    r"((?:[a-z0-9]+[_-])*(?:api[_-]?key|password|secret"
    r"(?:[_-]access[_-]key)?|token)|authorization|cookie)"
)
_QUOTED_SECRET = re.compile(
    rf"(?i)\b{_SECRET_NAME}(\s*[:=]\s*)(['\"])(.*?)(\3)"
)
_INLINE_SECRET = re.compile(
    rf"(?i)\b{_SECRET_NAME}"
    r"(\s*[:=]\s*)(?:(?:bearer|basic)\s+)?([^\s,;\"']+)"
)
_QUOTED_CLI_SECRET = re.compile(
    r"(?i)(--(?:api[-_]?key|password|secret|token)\s+)(['\"])(.*?)(\2)"
)
_CLI_SECRET = re.compile(
    r"(?i)(--(?:api[-_]?key|password|secret|token)\s+)([^\s,;\"']+)"
)
_TRACE_SPOOL_OWNER = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
_TRACE_SPOOL_ROOT = Path(tempfile.gettempdir()) / f"ogmem-trace-hook-spool-{_TRACE_SPOOL_OWNER}"
_TRACE_SPOOL_LIMIT = 1024
_TRACE_SPOOL_TTL_SECONDS = 3600
_TRACE_LOCK_LEASE_SECONDS = 10

_TRACE_EVENT_TYPES = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "user_message",
    "PreToolUse": "tool_start",
    "PostToolUse": "tool_success",
    "PostToolUseFailure": "tool_failure",
    "Stop": "assistant_message",
    "SubagentStop": "assistant_message",
    "PreCompact": "pre_compact",
    "SessionEnd": "session_end",
}


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


def _capture_trace_content() -> bool:
    return os.environ.get("OGMEM_TRACE_CAPTURE_CONTENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    if isinstance(value, str):
        value = _QUOTED_SECRET.sub(r"\1\2\3[REDACTED]\3", value)
        value = _INLINE_SECRET.sub(r"\1\2[REDACTED]", value)
        value = _QUOTED_CLI_SECRET.sub(r"\1\2[REDACTED]\2", value)
        return _CLI_SECRET.sub(r"\1[REDACTED]", value)
    return value


def _content_size(value: object) -> int:
    return len(_json_chunk(value))


def _trace_timeout() -> float:
    try:
        return max(
            0.05,
            min(2.0, float(os.environ.get("OGMEM_TRACE_POST_TIMEOUT", TRACE_POST_TIMEOUT))),
        )
    except ValueError:
        return TRACE_POST_TIMEOUT


def _trace_circuit_path(url: str, account_id: str) -> Path:
    key = hashlib.sha256(f"{url}\0{account_id}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"ogmem-trace-hook-circuit-{key}"


def _trace_spool_partition(
    url: str,
    account_id: str,
    user_id: str,
    agent_id: str,
    auth_fingerprint: str,
) -> Path:
    identity = "\0".join(
        (url, account_id, user_id, agent_id, auth_fingerprint)
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return _TRACE_SPOOL_ROOT / key


def _acquire_drain_lock(path: Path) -> str | None:
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    try:
        os.write(descriptor, token.encode("ascii"))
    finally:
        os.close(descriptor)
    return token


def _release_drain_lock(path: Path, token: str) -> None:
    try:
        if path.read_text(encoding="ascii") == token:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _owns_drain_lock(path: Path, token: str) -> bool:
    try:
        return path.read_text(encoding="ascii") == token
    except OSError:
        return False


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise OSError(f"refusing symlinked trace spool directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        if path.stat().st_uid != os.getuid():
            raise OSError(f"trace spool directory has unexpected owner: {path}")
        path.chmod(0o700)


def _write_private_spool(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _spawn_detached(command: list[str]) -> None:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _start_trace_watchdog(lock_path: Path) -> bool:
    """Ensure an orphaned drain lease is recovered without a future hook."""
    watchdog_path = lock_path.with_name(".watchdog.lock")
    if (
        watchdog_path.exists()
        and time.time() - watchdog_path.stat().st_mtime > _TRACE_LOCK_LEASE_SECONDS
    ):
        watchdog_path.unlink(missing_ok=True)
    watchdog_token = _acquire_drain_lock(watchdog_path)
    if watchdog_token is None:
        return True
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--watch-traces",
        str(lock_path),
        str(watchdog_path),
        watchdog_token,
    ]
    try:
        _spawn_detached(command)
        return True
    except OSError:
        _release_drain_lock(watchdog_path, watchdog_token)
        return False


def _watch_spooled_traces(
    lock_path: Path,
    watchdog_path: Path,
    watchdog_token: str,
) -> int:
    """Wait for the active lease, then take over if its worker disappeared."""
    spool_dir = lock_path.parent
    try:
        while any(spool_dir.glob("*.json")):
            if not _owns_drain_lock(watchdog_path, watchdog_token):
                return 0
            if lock_path.exists():
                try:
                    lease_age = time.time() - lock_path.stat().st_mtime
                except OSError:
                    continue
                if lease_age <= _TRACE_LOCK_LEASE_SECONDS:
                    if not _wait_with_lock_heartbeat(
                        watchdog_path, watchdog_token, 1.0
                    ):
                        return 0
                    continue
                lock_path.unlink(missing_ok=True)
            drain_token = _acquire_drain_lock(lock_path)
            if drain_token is not None:
                return _drain_spooled_traces(lock_path, drain_token)
        return 0
    finally:
        _release_drain_lock(watchdog_path, watchdog_token)


def _send_spool_path(path: Path) -> str:
    """Send one spool file and remove it only on success/permanent rejection."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        send_result = _send_trace_event(
            payload["session_id"],
            payload["event"],
            str(payload.get("event_id") or path.stem),
            str(
                payload.get("event_time")
                or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log(f"Invalid trace spool file: {exc}")
        path.unlink(missing_ok=True)
        return "drop"
    if send_result in {"sent", "drop"}:
        path.unlink(missing_ok=True)
    return send_result


def _spool_sort_key(path: Path) -> tuple[str, str]:
    """Order by persisted occurrence time, with mtime fallback for old spools."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        event_time = str(payload.get("event_time") or "")
        modified_time = path.stat().st_mtime
    except OSError:
        return "\uffff", path.name
    except (TypeError, ValueError, json.JSONDecodeError):
        event_time = ""
        try:
            modified_time = path.stat().st_mtime
        except OSError:
            return "\uffff", path.name
    if not event_time:
        event_time = datetime.fromtimestamp(modified_time, UTC).isoformat()
    return event_time, path.name


def _send_spooled_prefix(spool_dir: Path, max_events: int = 16) -> bool:
    """Bounded synchronous fallback that never jumps ahead of older events."""
    paths = sorted(spool_dir.glob("*.json"), key=_spool_sort_key)
    for path in paths[:max_events]:
        if _send_spool_path(path) not in {"sent", "drop"}:
            return False
    return not any(spool_dir.glob("*.json"))


def _wait_with_lock_heartbeat(path: Path, token: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _owns_drain_lock(path, token):
            return False
        path.touch()
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return _owns_drain_lock(path, token)


def _trace_circuit_open(path: Path) -> bool:
    try:
        return float(path.read_text(encoding="ascii")) > time.time()
    except (OSError, ValueError):
        return False


def _set_trace_circuit(path: Path, open_: bool) -> None:
    try:
        if open_:
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                str(time.time() + TRACE_CIRCUIT_BREAK_SECONDS), encoding="ascii"
            )
            temporary.replace(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        log(f"Trace circuit state update failed: {exc}")


def _build_tool_text(data: dict) -> str:
    name = data.get("tool_name") or "unknown_tool"
    t_in = data.get("tool_input")
    t_out = data.get("tool_response")
    lines = [
        f"[PostToolUse] {name}",
        f"tool_input: {_json_chunk(_redact(t_in))}",
        f"tool_response: {_json_chunk(_redact(t_out))}",
    ]
    return "\n".join(lines)


def _hook_name(data: dict) -> str:
    explicit = str(data.get("hook_event_name") or "").strip()
    if explicit:
        return explicit
    if data.get("tool_name") is not None:
        return "PostToolUse"
    return "unknown"


def _build_trace_event(data: dict) -> dict:
    """Build a bounded, structured event from Claude Code hook input."""
    hook_type = _hook_name(data)
    event = {
        "event_type": _TRACE_EVENT_TYPES.get(hook_type, "hook_event"),
        "hook_type": hook_type,
    }

    capture_content = _capture_trace_content()
    if data.get("prompt") is not None:
        event["prompt_size"] = _content_size(data.get("prompt"))
        if capture_content:
            event["prompt"] = _truncate(str(_redact(data.get("prompt") or "")))

    if data.get("tool_name") is not None:
        event.update(
            {
                "tool_name": str(data.get("tool_name") or "unknown_tool"),
                "tool_use_id": str(data.get("tool_use_id") or ""),
                "tool_input_size": _content_size(data.get("tool_input")),
            }
        )
        if capture_content:
            event["tool_input"] = _json_chunk(_redact(data.get("tool_input")))
        if hook_type == "PostToolUseFailure":
            error = data.get("error") or data.get("tool_response")
            event["error_size"] = _content_size(error)
            if capture_content:
                event["error"] = _json_chunk(_redact(error))
        elif data.get("tool_response") is not None:
            event["tool_response_size"] = _content_size(data.get("tool_response"))
            if capture_content:
                event["tool_response"] = _json_chunk(
                    _redact(data.get("tool_response"))
                )

    if data.get("last_assistant_message") is not None:
        assistant_message = data.get("last_assistant_message") or ""
        event["assistant_message_size"] = _content_size(assistant_message)
        if capture_content:
            event["assistant_message"] = _truncate(str(_redact(assistant_message)))
    if data.get("agent_id") is not None:
        event["subagent_id"] = str(data.get("agent_id") or "")
    if data.get("reason") is not None:
        event["reason"] = _truncate(str(data.get("reason") or ""))
    if data.get("source") is not None:
        event["source"] = _truncate(str(data.get("source") or ""))
    return event


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
        with urllib.request.urlopen(req, timeout=MESSAGE_POST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            log(f"Response {resp.status}: {raw[:200]}")
            return 200 <= resp.status < 300
    except urllib.error.URLError as e:
        log(f"HTTP error: {e}")
        return False
    except Exception as e:
        log(f"Request failed: {e}")
        return False


def _send_trace_event(
    session_id: str,
    event: dict,
    event_id: str | None = None,
    event_time: str | None = None,
) -> str:
    """Post one trace event. Failures are logged and never block the hook."""
    url = f"{base_api_url()}/api/v1/trace/events"
    ctx = base_ctx(session_id)
    circuit_path = _trace_circuit_path(url, str(ctx["accountId"]))
    if _trace_circuit_open(circuit_path):
        log("Trace circuit open; skip HTTP post")
        return "retry"
    event_id = event_id or uuid.uuid4().hex
    body = {
        "request_id": event_id,
        "trace_id": str(event.get("tool_use_id") or event_id),
        "operation": "claude_code_hook",
        "status": "error" if event.get("event_type") == "tool_failure" else "ok",
        "error_type": (
            "tool_failure" if event.get("event_type") == "tool_failure" else None
        ),
        "timestamp": event_time or datetime.now(UTC).isoformat(),
        "account_id": ctx["accountId"],
        "user_id": ctx["userId"],
        "agent_id": ctx["agentId"],
        "session_id": session_id,
        "event": event,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers=http_plugin_headers(),
        method="POST",
    )
    log(f"POST {url} hook={event.get('hook_type')} event={event.get('event_type')}")
    try:
        with urllib.request.urlopen(req, timeout=_trace_timeout()) as resp:
            resp.read()
            if 200 <= resp.status < 300:
                _set_trace_circuit(circuit_path, False)
                return "sent"
            if 400 <= resp.status < 500 and resp.status not in {408, 429}:
                return "drop"
            _set_trace_circuit(circuit_path, True)
            return "retry"
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500 and exc.code not in {408, 429}:
            log(f"Trace rejected permanently: HTTP {exc.code}")
            return "drop"
        log(f"Trace HTTP error: {exc}")
        _set_trace_circuit(circuit_path, True)
        return "retry"
    except urllib.error.URLError as exc:
        log(f"Trace HTTP error: {exc}")
        _set_trace_circuit(circuit_path, True)
        return "retry"
    except Exception as exc:
        log(f"Trace request failed: {exc}")
        _set_trace_circuit(circuit_path, True)
        return "retry"


def post_trace_event(session_id: str, event: dict) -> bool:
    """Spool an event and ensure at most one detached drain worker is running."""
    spool_path: Path | None = None
    url = f"{base_api_url()}/api/v1/trace/events"
    ctx = base_ctx(session_id)
    auth_headers = http_plugin_headers()
    auth_fingerprint = hashlib.sha256(
        json.dumps(auth_headers, sort_keys=True).encode()
    ).hexdigest()
    spool_dir = _trace_spool_partition(
        url,
        str(ctx["accountId"]),
        str(ctx["userId"]),
        str(ctx["agentId"]),
        auth_fingerprint,
    )
    lock_path = spool_dir / ".drain.lock"
    try:
        _ensure_private_directory(_TRACE_SPOOL_ROOT)
        _ensure_private_directory(spool_dir)
        existing = sorted(
            spool_dir.glob("*.json"), key=_spool_sort_key
        )
        for stale in existing[: max(0, len(existing) - _TRACE_SPOOL_LIMIT + 1)]:
            stale.unlink(missing_ok=True)

        event_id = uuid.uuid4().hex
        event_time = datetime.now(UTC).isoformat()
        spool_path = spool_dir / f"{event_id}.json"
        _write_private_spool(
            spool_path,
            {
                "event_id": event_id,
                "event_time": event_time,
                "session_id": session_id,
                "event": event,
            },
        )

        if (
            lock_path.exists()
            and time.time() - lock_path.stat().st_mtime > _TRACE_LOCK_LEASE_SECONDS
        ):
            lock_path.unlink(missing_ok=True)
        lock_token = _acquire_drain_lock(lock_path)
        if lock_token is None:
            if not _start_trace_watchdog(lock_path):
                return _send_spooled_prefix(spool_dir)
            return True

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--drain-traces",
            str(lock_path),
            lock_token,
        ]
        try:
            _spawn_detached(command)
            if not _start_trace_watchdog(lock_path):
                return _send_spooled_prefix(spool_dir)
        except OSError as exc:
            log(f"Trace drain launch failed: {exc}")
            _release_drain_lock(lock_path, lock_token)
            if _start_trace_watchdog(lock_path):
                return True
            return _send_spooled_prefix(spool_dir)
        return True
    except (OSError, TypeError, ValueError) as exc:
        log(f"Trace spool failed: {exc}")
        if "lock_token" in locals():
            _release_drain_lock(lock_path, lock_token)
        return False


def _drain_spooled_traces(lock_path: Path, lock_token: str) -> int:
    spool_dir = lock_path.parent
    try:
        while True:
            retry_required = False
            paths = sorted(
                spool_dir.glob("*.json"), key=_spool_sort_key
            )
            for path in paths:
                if not _owns_drain_lock(lock_path, lock_token):
                    return 0
                lock_path.touch()
                try:
                    spool_age = time.time() - path.stat().st_mtime
                except OSError:
                    continue
                if spool_age > _TRACE_SPOOL_TTL_SECONDS:
                    path.unlink(missing_ok=True)
                    continue
                send_result = _send_spool_path(path)
                if _owns_drain_lock(lock_path, lock_token):
                    lock_path.touch()
                if send_result == "drop":
                    continue
                if send_result not in {True, "sent"}:
                    remaining_ttl = _TRACE_SPOOL_TTL_SECONDS - (
                        time.time() - path.stat().st_mtime
                    )
                    if remaining_ttl <= 0:
                        path.unlink(missing_ok=True)
                        continue
                    if not _wait_with_lock_heartbeat(
                        lock_path,
                        lock_token,
                        min(TRACE_CIRCUIT_BREAK_SECONDS, remaining_ttl),
                    ):
                        return 1
                    retry_required = True
                    break

            if retry_required:
                continue

            _release_drain_lock(lock_path, lock_token)
            if not any(spool_dir.glob("*.json")):
                return 0
            next_token = _acquire_drain_lock(lock_path)
            if next_token is None:
                return 0
            lock_token = next_token
    finally:
        _release_drain_lock(lock_path, lock_token)


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

    event = _build_trace_event(data)
    if not post_trace_event(session_id, event):
        log(f"Failed to post trace event for session={session_id}")

    if _hook_name(data) != "PostToolUse":
        sys.exit(0)

    tool_name = data.get("tool_name") or "unknown_tool"
    if tool_name not in _SIDE_EFFECT_TOOLS:
        log(f"skip non-side-effect tool (matcher widened or stale config): {tool_name}")
        sys.exit(0)

    t_in = data.get("tool_input")
    t_out = data.get("tool_response")
    log(
        f"tool={tool_name} input_len={_content_size(t_in)} "
        f"output_len={_content_size(t_out)}"
    )

    text = _build_tool_text(data)
    if not text.strip():
        sys.exit(0)

    ok = post_session_message(session_id, "tool", text)
    if not ok:
        log(f"Failed to post tool message for session={session_id}, tool={tool_name}")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--drain-traces":
        sys.exit(_drain_spooled_traces(Path(sys.argv[2]), sys.argv[3]))
    if len(sys.argv) == 5 and sys.argv[1] == "--watch-traces":
        sys.exit(
            _watch_spooled_traces(
                Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
            )
        )
    main()
