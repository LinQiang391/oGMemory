from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins" / "ogmemory"
SCRIPT = PLUGIN / "scripts" / "ogmemory_codex_hook.py"

HOOK_EVENTS = {
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("ogmemory_codex_hook", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hook_timeout(hooks, event):
    return hooks["hooks"][event][0]["hooks"][0]["timeout"]


def test_plugin_package_contract():
    module = _load_module()
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())
    marketplace = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text())
    initialization_budget = (
        module.TRACE_TIMEOUT_SECONDS + module.WRITE_TIMEOUT_SECONDS + module.RECALL_TIMEOUT_SECONDS
    )

    assert manifest["name"] == "ogmemory"
    assert set(hooks["hooks"]) == HOOK_EVENTS
    assert _hook_timeout(hooks, "SessionStart") > initialization_budget
    assert _hook_timeout(hooks, "SubagentStart") > initialization_budget
    assert _hook_timeout(hooks, "SessionEnd") == 3
    assert marketplace["plugins"][0]["source"]["path"] == "./plugins/ogmemory"


def test_user_prompt_traces_recalls_and_caches_turn(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/compose":
            return {"identityContext": "Prefers Python", "retrievedEvidence": "Use uv"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    output = module.handle_event(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "prompt": "How should we package this project?",
        }
    )

    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/compose"]
    assert calls[0][1]["event"]["event_type"] == "user_message"
    assert calls[1][1]["agentId"] == "codex"
    assert "Prefers Python" in output["hookSpecificOutput"]["additionalContext"]
    assert module._load_prompt("session-1", "turn-1")[1] == "How should we package this project?"


def test_stop_traces_and_persists_completed_turn(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    module._save_prompt("session-1", "turn-1", "Remember this decision")
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    output = module.handle_event(
        {
            "hook_event_name": "Stop",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "last_assistant_message": "Saved.",
        }
    )

    assert output == {"continue": True}
    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/after_turn"]
    assert calls[1][1]["messages"] == [
        {"role": "user", "content": "Remember this decision"},
        {"role": "assistant", "content": "Saved."},
    ]
    assert calls[1][1]["forceExtract"] is True
    assert module._load_prompt("session-1", "turn-1") is None


def test_user_config_applies_to_http_calls(monkeypatch, tmp_path):
    config_path = tmp_path / "codex-plugin.json"
    config_path.write_text(
        json.dumps(
            {
                "url": "https://memory.example.com",
                "api_key": "tenant-key",
                "account_id": "account-1",
                "user_id": "user-1",
                "agent_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OG_MEMORY_CODEX_CONFIG", str(config_path))
    for name in (
        "OG_MEMORY_URL",
        "OGMEM_API_URL",
        "OG_AUTH_API_KEY",
        "OG_MEMORY_API_KEY",
        "OG_MEMORY_ACCOUNT_ID",
        "OG_ACCOUNT_ID",
        "OG_MEMORY_USER_ID",
        "OG_USER_ID",
        "OG_MEMORY_AGENT_ID",
        "OG_AGENT_ID",
    ):
        monkeypatch.setenv(name, "")
    module = _load_module()

    assert module._base_url() == "https://memory.example.com"
    assert module._identity("session-1") == {
        "accountId": "account-1",
        "userId": "user-1",
        "agentId": "agent-1",
        "sessionId": "session-1",
    }
    assert module._headers() == {
        "Content-Type": "application/json",
        "X-API-Key": "tenant-key",
        "X-Account-ID": "account-1",
        "X-User-ID": "user-1",
    }


def test_session_end_uses_three_second_hook_budget(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    module.handle_event(
        {
            "hook_event_name": "SessionEnd",
            "session_id": "session-1",
            "reason": "other",
        }
    )

    assert [call[0] for call in calls] == ["/api/v1/trace/events", "/api/v1/dispose"]
    assert [call[2] for call in calls] == [1, 1]


def test_compaction_prepare_token_is_committed(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/prepare_compaction":
            return {"ok": True, "prepareToken": "prepare-1"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    event = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "trigger": "auto",
    }
    module.handle_event({"hook_event_name": "PreCompact", **event})
    module.handle_event({"hook_event_name": "PostCompact", **event})

    compact_call = next(call for call in calls if call[0] == "/api/v1/compact")
    assert compact_call[1]["prepareToken"] == "prepare-1"


def test_tool_trace_redacts_credentials(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    module.handle_event(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "tool_name": "Bash",
            "tool_use_id": "tool-1",
            "tool_input": {"api_key": "secret-value"},
            "tool_response": "Authorization: Bearer secret-token",
        }
    )

    event = calls[0][1]["event"]
    assert event["tool_input"] == {"api_key": "[REDACTED]"}
    assert event["tool_response"] == "Authorization: Bearer [REDACTED]"


def test_permission_requests_use_distinct_trace_ids(monkeypatch):
    module = _load_module()
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    base = {
        "hook_event_name": "PermissionRequest",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "permission_mode": "default",
    }
    module.handle_event(
        {
            **base,
            "tool_name": "Bash",
            "tool_input": {"command": "pwd"},
        }
    )
    module.handle_event(
        {
            **base,
            "tool_name": "apply_patch",
            "tool_input": {"patch": "*** Begin Patch"},
        }
    )

    assert calls[0][1]["request_id"] != calls[1][1]["request_id"]


@pytest.mark.parametrize(
    ("event", "extra", "expected_paths"),
    [
        (
            "SessionStart",
            {"source": "startup"},
            ["/api/v1/trace/events", "/api/v1/bootstrap", "/api/v1/compose"],
        ),
        ("SessionEnd", {"reason": "other"}, ["/api/v1/trace/events", "/api/v1/dispose"]),
        (
            "PreToolUse",
            {"tool_name": "Bash", "tool_input": {"command": "pwd"}},
            ["/api/v1/trace/events", "/api/v1/compose"],
        ),
        ("PermissionRequest", {"tool_name": "Bash"}, ["/api/v1/trace/events"]),
        ("PostToolUse", {"tool_name": "Bash", "tool_response": "done"}, ["/api/v1/trace/events"]),
        ("PreCompact", {"trigger": "auto"}, ["/api/v1/trace/events", "/api/v1/prepare_compaction"]),
        ("PostCompact", {"trigger": "auto"}, ["/api/v1/trace/events", "/api/v1/compact"]),
        (
            "SubagentStart",
            {"agent_id": "agent-1", "agent_type": "explorer"},
            ["/api/v1/trace/events", "/api/v1/prepare_subagent_spawn", "/api/v1/compose"],
        ),
        (
            "SubagentStop",
            {"agent_id": "agent-1", "agent_type": "explorer", "last_assistant_message": "Found it"},
            ["/api/v1/trace/events", "/api/v1/on_subagent_ended"],
        ),
    ],
)
def test_lifecycle_event_routes_to_ogmemory(
    monkeypatch,
    tmp_path,
    event,
    extra,
    expected_paths,
):
    module = _load_module()
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path))
    calls = []

    def post(path, body, timeout):
        calls.append((path, body, timeout))
        if path == "/api/v1/compose":
            return {"retrievedEvidence": "Relevant project memory"}
        if path == "/api/v1/prepare_compaction":
            return {"ok": True, "prepareToken": "prepare-1"}
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json", post)
    module.handle_event(
        {
            "hook_event_name": event,
            "session_id": "session-1",
            "turn_id": "turn-1",
            "cwd": "/workspace",
            **extra,
        }
    )

    assert [call[0] for call in calls] == expected_paths
    assert calls[0][1]["event"]["event_type"] == module.EVENT_TYPES[event]
