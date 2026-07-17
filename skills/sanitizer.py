"""Deterministic trust boundary for trace-derived skill generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class UnsafeTraceError(ValueError):
    """Raised when a trace contains instruction-like or unverifiable input."""


_REDACTED = "[REDACTED_SECRET]"
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|passwd|private[_-]?key|"
    r"secret(?:[_-]access[_-]key)?|token)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    # PEM payloads must be removed as a unit, not merely have their header hidden.
    re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----.*?"
        r"-----END(?: [A-Z0-9]+)* PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Provider credentials are often pasted without a label.
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # JWTs are credentials even when no surrounding field identifies them as such.
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\bAuthorization\s*:\s*(?:Bearer|Basic)\s+[^\s,'\";]+"),
    re.compile(
        r"(?i)\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|password|passwd|private[_-]?key|"
        r"secret(?:[_-]access[_-]key)?|token|authorization|cookie)"
        r"\s*[:=]\s*(?:(?:bearer|basic)\s+)?(?:['\"][^'\"]+['\"]|[^\s,;]+)"
    ),
    re.compile(
        r"(?i)--(?:api[-_]?key|password|secret|token)"
        r"(?:=|\s+)(?:['\"][^'\"]+['\"]|[^\s,;]+)"
    ),
)
_ABSOLUTE_PATHS = (
    re.compile(r"(?<![\w])(?:[A-Za-z]:\\|[A-Za-z]:/)(?:[^\s<>:\"|?*]+[\\/])*[^\s<>:\"|?*]*"),
    # UNC paths, including administrative shares. Require both a server and
    # share so ordinary escaped text is not treated as a path.
    re.compile(r"(?<![\w\\])\\\\[^\s\\/'\"<>]+\\[^\s\\/'\"<>]+(?:\\[^\s'\"<>]+)*"),
    # Home-relative paths are machine/user specific even without a username.
    re.compile(r"(?<![\w])~[\\/][^\s'\"<>]+"),
    # General POSIX absolute paths. The lookbehind and (?!/) deliberately
    # exclude the // in URL schemes and protocol-relative URLs.
    re.compile(r"(?<![\w:/])/(?!/)(?:[^\s/'\"<>]+/)*[^\s/'\"<>]+"),
)
_INJECTION = re.compile(
    r"(?i)(ignore|disregard|override|forget)\s+"
    r"(?:(?:all|the|your|these|those|above|previous|prior|system|developer)\s+){0,3}instructions|"
    r"do\s+not\s+follow\s+"
    r"(?:(?:the|these|those|above|previous|prior|system|developer)\s+){0,3}instructions|"
    r"reveal\s+(the\s+)?(system prompt|credentials|secrets)|"
    r"you\s+are\s+now\s+(?:a|an)|<\s*(?:system|developer)\s*>"
)
_SESSION_VALUE = re.compile(r"\b(?:session|trace|request)[-_ ]?id\s*[:=]\s*[A-Za-z0-9._:-]+", re.I)
_FAILED_OUTCOME = re.compile(
    r"(?i)(?:\b(?:i|we)\s+(?:could(?:n't| not)|can(?:'t| not)|was unable|were unable|failed)\b|"
    r"\b(?:task|operation|request|work)\s+(?:is|was|remains)?\s*(?:failed|incomplete|unfinished)\b|"
    r"\b(?:did not|didn't|have not|haven't)\s+(?:finish|complete|succeed)\b|"
    r"\b(?:timed out|unhandled exception|fatal error)\b)"
)
_MESSAGE_FIELDS = {
    "user_message": ("prompt", "message", "content"),
    "assistant_message": ("assistant_message", "message", "content"),
}
_KNOWN_FILE_SUFFIXES = {
    ".css", ".csv", ".gif", ".html", ".ini", ".jpeg", ".jpg", ".json",
    ".md", ".pdf", ".png", ".py", ".toml", ".ts", ".txt", ".xml", ".yaml", ".yml",
}
_IDENTIFIER = r"[A-Za-z][A-Za-z0-9_.-]{0,63}"
_TOOL_LIKE_IDENTIFIER = r"[A-Za-z][A-Za-z0-9_.-]*[_.-][A-Za-z0-9_.-]{0,63}"
_STRUCTURED_TOOL_PATTERNS = (
    re.compile(rf"(?i)[`'\"]?(?:tool|tool_name)[`'\"]?\s*[:=]\s*[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)\b(?:run|call|invoke|execute)\s+(?:the\s+)?tool\s+[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)\b(?:run|call|invoke|execute)\s+[`'\"]?({_TOOL_LIKE_IDENTIFIER})"),
    re.compile(rf"(?i)\buse\s+(?:the\s+)?tool\s+[`'\"]?({_IDENTIFIER})"),
    re.compile(rf"(?i)[`'\"]({_IDENTIFIER})[`'\"]\s+tool\b"),
)


@dataclass(frozen=True)
class SanitizedTrace:
    events: list[dict[str, Any]]
    tools: frozenset[str]


def sanitize_text(value: str, *, reject_instructions: bool = False) -> str:
    """Remove secrets and machine/session-specific values from text."""
    if reject_instructions and _INJECTION.search(value):
        raise UnsafeTraceError("trace contains prompt-injection-like instructions")
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    for pattern in _ABSOLUTE_PATHS:
        result = pattern.sub("[LOCAL_PATH]", result)
    result = _SESSION_VALUE.sub("session_id=[SESSION]", result)
    return result


def sanitize_trace(events: list[dict[str, Any]]) -> SanitizedTrace:
    """Sanitize a JSON trace and verify messages and tool outcomes are complete."""
    if not isinstance(events, list) or not events:
        raise UnsafeTraceError("trace is empty")
    cleaned: list[dict[str, Any]] = []
    started: dict[str, int] = {}
    completed: dict[str, int] = {}
    has_user = False
    has_assistant = False
    for raw in events:
        if not isinstance(raw, dict):
            raise UnsafeTraceError("trace contains a malformed event")
        event_type = raw.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise UnsafeTraceError("trace event has no event_type")
        event_type = event_type.strip()
        if event_type in _MESSAGE_FIELDS:
            message = _message_text(raw, _MESSAGE_FIELDS[event_type])
            if not message:
                raise UnsafeTraceError(f"{event_type} has no meaningful content")
            if event_type == "user_message":
                has_user = True
            else:
                if _FAILED_OUTCOME.search(message):
                    raise UnsafeTraceError("trace contains an explicit failed or incomplete outcome")
                has_assistant = True
        if event_type == "tool_failure":
            raise UnsafeTraceError("trace contains a failed tool call")
        tool_name_value = raw.get("tool_name")
        tool_name = tool_name_value.strip() if isinstance(tool_name_value, str) else ""
        if event_type in {"tool_start", "tool_success"} and not tool_name:
            raise UnsafeTraceError(f"{event_type} has no tool_name")
        if event_type == "tool_start":
            started[tool_name] = started.get(tool_name, 0) + 1
        if event_type == "tool_success":
            completed[tool_name] = completed.get(tool_name, 0) + 1
            if completed[tool_name] > started.get(tool_name, 0):
                raise UnsafeTraceError(f"tool success has no observed start: {tool_name}")
        cleaned.append(_sanitize_value(raw))
    if not has_user or not has_assistant:
        raise UnsafeTraceError("trace must contain user and assistant messages")
    for tool_name, count in started.items():
        if completed.get(tool_name, 0) < count:
            raise UnsafeTraceError(f"tool call has no observed success: {tool_name}")
    return SanitizedTrace(cleaned, frozenset(completed))


def validate_skill_markdown(content: str, observed_tools: set[str] | frozenset[str]) -> str:
    """Apply the final publication boundary and reject invented agent tools."""
    sanitized = sanitize_text(content, reject_instructions=True)
    if len(sanitized.splitlines()) > 500:
        raise UnsafeTraceError("SKILL.md exceeds 500 lines")
    if len(sanitized.split()) > 5000:
        raise UnsafeTraceError("SKILL.md exceeds 5000 words")
    allowed = {tool.casefold() for tool in observed_tools}
    for token in _referenced_tools(sanitized):
        if token.casefold() not in allowed:
            raise UnsafeTraceError(f"skill references an unobserved tool: {token}")
    return sanitized


def prompt_payload(events: list[dict[str, Any]]) -> str:
    """Serialize untrusted evidence as JSON data, never as prompt instructions."""
    return json.dumps(events, ensure_ascii=False, sort_keys=True)


def _message_text(event: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _referenced_tools(content: str) -> set[str]:
    references: set[str] = set()
    for pattern in _STRUCTURED_TOOL_PATTERNS:
        references.update(match.group(1) for match in pattern.finditer(content))

    # Snake/kebab/namespace identifiers in code spans or call syntax are strong tool
    # references. A dotted filename such as README.md is deliberately not one.
    for match in re.finditer(rf"`({_IDENTIFIER})`", content):
        token = match.group(1)
        if _looks_like_tool_identifier(token):
            references.add(token)
    for match in re.finditer(rf"\b({_IDENTIFIER})\s*\(", content):
        token = match.group(1)
        if "_" in token or "-" in token or "." in token:
            references.add(token)
    return references


def _looks_like_tool_identifier(token: str) -> bool:
    lowered = token.casefold()
    if any(lowered.endswith(suffix) for suffix in _KNOWN_FILE_SUFFIXES):
        return False
    return "_" in token or "-" in token or "." in token


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, reject_instructions=True)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            secret_key = _SECRET_KEY.search(str(key)) is not None
            clean_key = _sanitize_key(key)
            if clean_key in cleaned:
                suffix = 2
                candidate = f"{clean_key}_{suffix}"
                while candidate in cleaned:
                    suffix += 1
                    candidate = f"{clean_key}_{suffix}"
                clean_key = candidate
            cleaned[clean_key] = _REDACTED if secret_key else _sanitize_value(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value), reject_instructions=True)


def _sanitize_key(key: Any) -> str:
    value = str(key)
    if _SECRET_KEY.search(value):
        return _REDACTED
    return sanitize_text(value, reject_instructions=True)
