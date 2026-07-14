"""Unit tests for claude-plugin hook scripts.

Covers: ogm_plugin_request, call_add_session_message,
        call_after_turn (parse/extract), call_compose (context builder).
"""

from __future__ import annotations

import importlib
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make claude-plugin/scripts importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "claude-plugin" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _reimport(name: str):
    """Force-reload a scripts module so monkeypatched env vars take effect."""
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


# ===========================================================================
# ogm_plugin_request
# ===========================================================================


class TestPluginApiKey:
    def test_returns_og_auth_api_key_when_set(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "secret-root")
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() == "secret-root"

    def test_returns_none_when_no_key_set(self, monkeypatch):
        monkeypatch.delenv("OG_AUTH_API_KEY", raising=False)
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "  trimmed  ")
        mod = _reimport("ogm_plugin_request")
        assert mod._plugin_api_key() == "trimmed"


class TestHttpPluginHeaders:
    def test_no_key_returns_content_type_only(self, monkeypatch):
        monkeypatch.delenv("OG_AUTH_API_KEY", raising=False)
        monkeypatch.delenv("OGMEM_PLUGIN_API_KEY", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert mod.http_plugin_headers() == {"Content-Type": "application/json"}

    def test_with_key_adds_auth_headers(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "my-key")
        monkeypatch.setenv("OG_ACCOUNT_ID", "acct-test")
        monkeypatch.setenv("OG_USER_ID", "u-test")
        mod = _reimport("ogm_plugin_request")
        monkeypatch.setattr(mod, "_try_ogmem_config", lambda: None)
        headers = mod.http_plugin_headers()
        assert headers["X-API-Key"] == "my-key"
        assert headers["X-Account-ID"] == "acct-test"
        assert headers["X-User-ID"] == "u-test"
        assert headers["Content-Type"] == "application/json"

    def test_no_agentid_in_headers(self, monkeypatch):
        monkeypatch.setenv("OG_AUTH_API_KEY", "k")
        mod = _reimport("ogm_plugin_request")
        assert "X-Agent-ID" not in mod.http_plugin_headers()


class TestBaseCtx:
    def test_returns_required_keys(self, monkeypatch):
        monkeypatch.setenv("OG_ACCOUNT_ID", "acct-demo")
        monkeypatch.setenv("OG_USER_ID", "u-alice")
        monkeypatch.setenv("OG_AGENT_ID", "claude-code")
        mod = _reimport("ogm_plugin_request")
        monkeypatch.setattr(mod, "_try_ogmem_config", lambda: None)
        ctx = mod.base_ctx("sess-123")
        assert ctx["accountId"] == "acct-demo"
        assert ctx["userId"] == "u-alice"
        assert ctx["agentId"] == "claude-code"
        assert ctx["sessionId"] == "sess-123"

    def test_session_id_passed_through(self, monkeypatch):
        mod = _reimport("ogm_plugin_request")
        assert mod.base_ctx("my-session")["sessionId"] == "my-session"


class TestBaseApiUrl:
    def test_uses_og_memory_url_env(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://remote:9000")
        mod = _reimport("ogm_plugin_request")
        assert mod.base_api_url() == "http://remote:9000"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://remote:9000/")
        mod = _reimport("ogm_plugin_request")
        assert mod.base_api_url() == "http://remote:9000"

    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("OG_MEMORY_URL", raising=False)
        mod = _reimport("ogm_plugin_request")
        assert "8090" in mod.base_api_url()


# ===========================================================================
# call_add_session_message
# ===========================================================================


class TestBuildToolText:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_add_session_message")

    def test_includes_tool_name(self):
        text = self.mod._build_tool_text(
            {"tool_name": "Bash", "tool_input": {}, "tool_response": {}}
        )
        assert "[PostToolUse] Bash" in text

    def test_includes_input_and_response(self):
        text = self.mod._build_tool_text(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/foo"},
                "tool_response": {"content": "bar"},
            }
        )
        assert "tool_input:" in text
        assert "tool_response:" in text
        assert "/foo" in text

    def test_handles_none_values(self):
        text = self.mod._build_tool_text(
            {"tool_name": "Edit", "tool_input": None, "tool_response": None}
        )
        assert "[PostToolUse] Edit" in text
        assert "null" in text

    def test_truncates_large_input(self):
        big = "x" * 20_000
        text = self.mod._build_tool_text(
            {"tool_name": "Bash", "tool_input": big, "tool_response": "ok"}
        )
        assert len(text) < 25_000

    def test_redacts_secrets_from_session_message(self):
        text = self.mod._build_tool_text(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "OPENAI_API_KEY=env-secret "
                        "AWS_SECRET_ACCESS_KEY='quoted-secret' "
                        "password=\"hunter2\" "
                        "tool --api-key cli-secret --token 'quoted-cli-secret' "
                        "curl -H 'Authorization: Bearer sk-secret'"
                    ),
                    "api_key": "raw-secret",
                },
                "tool_response": "ok",
            }
        )

        assert "sk-secret" not in text
        assert "env-secret" not in text
        assert "quoted-secret" not in text
        assert "hunter2" not in text
        assert "cli-secret" not in text
        assert "quoted-cli-secret" not in text
        assert "raw-secret" not in text
        assert text.count("[REDACTED]") == 7


class TestBuildTraceEvent:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_add_session_message")

    @pytest.mark.parametrize(
        ("hook_name", "expected_type"),
        [
            ("SessionStart", "session_start"),
            ("UserPromptSubmit", "user_message"),
            ("PreToolUse", "tool_start"),
            ("PostToolUse", "tool_success"),
            ("PostToolUseFailure", "tool_failure"),
            ("Stop", "assistant_message"),
            ("SubagentStop", "assistant_message"),
            ("PreCompact", "pre_compact"),
            ("SessionEnd", "session_end"),
        ],
    )
    def test_maps_supported_hook_events(self, hook_name, expected_type):
        event = self.mod._build_trace_event(
            {
                "hook_event_name": hook_name,
                "prompt": "hello",
                "tool_name": "Read",
                "last_assistant_message": "done",
            }
        )

        assert event["hook_type"] == hook_name
        assert event["event_type"] == expected_type

    def test_bounds_tool_payloads(self, monkeypatch):
        monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "1")
        event = self.mod._build_trace_event(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": "x" * 20_000,
                "tool_response": "y" * 20_000,
            }
        )

        assert len(event["tool_input"]) < 11_000
        assert len(event["tool_response"]) < 11_000

    def test_content_is_disabled_by_default(self):
        event = self.mod._build_trace_event(
            {
                "hook_event_name": "PostToolUse",
                "prompt": "private prompt",
                "tool_name": "Read",
                "tool_input": {"file_path": "/secret"},
                "tool_response": {"content": "private content"},
            }
        )

        assert "prompt" not in event
        assert "tool_input" not in event
        assert "tool_response" not in event
        assert event["prompt_size"] > 0
        assert event["tool_input_size"] > 0
        assert event["tool_response_size"] > 0

    def test_opted_in_content_redacts_secrets(self, monkeypatch):
        monkeypatch.setenv("OGMEM_TRACE_CAPTURE_CONTENT", "true")
        event = self.mod._build_trace_event(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "curl -H 'authorization: bearer-value'",
                    "api_key": "raw-secret",
                },
            }
        )

        assert "raw-secret" not in event["tool_input"]
        assert "bearer-value" not in event["tool_input"]
        assert event["tool_input"].count("[REDACTED]") == 2

    def test_circuit_state_is_isolated_by_endpoint_and_account(self):
        first = self.mod._trace_circuit_path("https://one.test/events", "acct-a")
        other_endpoint = self.mod._trace_circuit_path(
            "https://two.test/events", "acct-a"
        )
        other_account = self.mod._trace_circuit_path(
            "https://one.test/events", "acct-b"
        )

        assert len({first, other_endpoint, other_account}) == 3


class TestPostSessionMessage:
    def test_returns_true_on_200(self, monkeypatch):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = mod.post_session_message("sess-1", "tool", "content")
        assert result is True

    def test_returns_false_on_url_error(self, monkeypatch):
        import urllib.error

        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = mod.post_session_message("sess-1", "tool", "content")
        assert result is False


class TestPostTraceEvent:
    def test_spools_and_starts_detached_sender(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        with patch.object(mod.subprocess, "Popen") as popen:
            assert mod.post_trace_event("sess-1", {"event_type": "tool_start"})

        spool_files = list(tmp_path.rglob("*.json"))
        assert len(spool_files) == 1
        payload = json.loads(spool_files[0].read_text(encoding="utf-8"))
        assert payload == {
            "session_id": "sess-1",
            "event": {"event_type": "tool_start"},
        }
        command = popen.call_args.args[0]
        assert command[-3] == "--drain-traces"
        assert Path(command[-2]).name == ".drain.lock"
        assert command[-1]
        if os.name != "nt":
            assert stat.S_IMODE(spool_files[0].stat().st_mode) == 0o600
            assert stat.S_IMODE(spool_files[0].parent.stat().st_mode) == 0o700

    def test_drain_retains_event_when_send_fails(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)
        spool = tmp_path / "event.json"
        spool.write_text(
            json.dumps({"session_id": "sess-1", "event": {"event_type": "tool_start"}}),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        with (
            patch.object(mod, "_send_trace_event", return_value=False),
            patch.object(mod, "_wait_with_lock_heartbeat", return_value=False),
        ):
            assert mod._drain_spooled_traces(lock, "owner-token") == 1

        assert spool.exists()
        assert not lock.exists()

    def test_drain_rescans_events_added_during_send(self, tmp_path):
        mod = _reimport("call_add_session_message")
        first = tmp_path / "first.json"
        first.write_text(
            json.dumps({"session_id": "s1", "event": {"event_type": "first"}}),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")
        sent = []

        def send(session_id, event):
            sent.append(event["event_type"])
            if event["event_type"] == "first":
                (tmp_path / "second.json").write_text(
                    json.dumps(
                        {"session_id": "s2", "event": {"event_type": "second"}}
                    ),
                    encoding="utf-8",
                )
            return True

        with patch.object(mod, "_send_trace_event", side_effect=send):
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        assert sent == ["first", "second"]
        assert list(tmp_path.glob("*.json")) == []

    def test_drain_retries_without_a_later_hook(self, tmp_path):
        mod = _reimport("call_add_session_message")
        spool = tmp_path / "event.json"
        spool.write_text(
            json.dumps({"session_id": "s1", "event": {"event_type": "first"}}),
            encoding="utf-8",
        )
        lock = tmp_path / ".drain.lock"
        lock.write_text("owner-token", encoding="ascii")

        with (
            patch.object(mod, "_send_trace_event", side_effect=[False, True]),
            patch.object(mod, "_wait_with_lock_heartbeat", return_value=True),
        ):
            assert mod._drain_spooled_traces(lock, "owner-token") == 0

        assert not spool.exists()

    def test_spool_partition_isolated_by_endpoint_and_identity(self, tmp_path, monkeypatch):
        mod = _reimport("call_add_session_message")
        monkeypatch.setattr(mod, "_TRACE_SPOOL_ROOT", tmp_path)

        partitions = {
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://two.test/events", "acct-a", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-b", "user-a", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-b", "agent-a", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-b", "auth-a"
            ),
            mod._trace_spool_partition(
                "https://one.test/events", "acct-a", "user-a", "agent-a", "auth-b"
            ),
        }

        assert len(partitions) == 6


class TestAddSessionMessageMain:
    def test_skips_when_no_session_id(self, capsys):
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps({"tool_name": "Bash", "tool_input": {}, "tool_response": {}})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_read_only_tools_are_traced_but_not_added_to_messages(self, monkeypatch, capsys):
        """Read-only tools belong in traces but not the session message buffer."""
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps(
            {
                "session_id": "sess-1",
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x"},
                "tool_response": {"content": "y"},
            }
        )
        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "post_session_message") as mock_post,
            patch.object(mod, "post_trace_event", return_value=True) as mock_trace,
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0
        mock_post.assert_not_called()
        mock_trace.assert_called_once()
        assert "skip non-side-effect" in capsys.readouterr().err

    def test_logs_failure_when_post_fails(self, monkeypatch, capsys):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_add_session_message")
        hook_input = json.dumps(
            {
                "session_id": "sess-x",
                "tool_name": "Bash",
                "tool_input": {"command": "ls"},
                "tool_response": {"stdout": "file.txt"},
            }
        )
        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "post_session_message", return_value=False),
            patch.object(mod, "post_trace_event", return_value=True),
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit):
                mod.main()
        captured = capsys.readouterr()
        assert "Failed" in captured.err


class TestTraceHookRegistration:
    def test_registers_complete_trace_lifecycle(self):
        hooks_path = _SCRIPTS_DIR.parent / "hooks" / "hooks.json"
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))["hooks"]

        expected = {
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SubagentStop",
            "PreCompact",
            "SessionEnd",
        }
        assert expected <= hooks.keys()
        assert "matcher" not in hooks["PostToolUse"][0]

        for hook_name in expected:
            commands = [entry["command"] for entry in hooks[hook_name][0]["hooks"]]
            assert any("call_add_session_message.py" in command for command in commands)


# ===========================================================================
# call_after_turn — parse_messages / extract_text
# ===========================================================================


class TestExtractText:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_after_turn")

    def test_string_content_returned_as_is(self):
        assert self.mod.extract_text("hello") == "hello"

    def test_extracts_text_blocks_from_list(self):
        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert self.mod.extract_text(content) == "part1\npart2"

    def test_skips_non_text_blocks(self):
        content = [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "visible"}]
        assert self.mod.extract_text(content) == "visible"

    def test_empty_list_returns_empty(self):
        assert self.mod.extract_text([]) == ""


class TestParseMessages:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_after_turn")

    def _line(self, **kwargs) -> str:
        return json.dumps(kwargs)

    def test_parses_user_and_assistant(self):
        lines = "\n".join(
            [
                self._line(type="user", message={"role": "user", "content": "hi"}),
                self._line(
                    type="assistant",
                    message={"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                ),
            ]
        )
        msgs = self.mod.parse_messages(lines)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1] == {"role": "assistant", "content": "hello"}

    def test_filters_sidechain(self):
        lines = self._line(
            type="user", isSidechain=True, message={"role": "user", "content": "sub"}
        )
        assert self.mod.parse_messages(lines) == []

    def test_filters_api_error(self):
        lines = self._line(
            type="user", isApiErrorMessage=True, message={"role": "user", "content": "err"}
        )
        assert self.mod.parse_messages(lines) == []

    def test_filters_non_user_assistant_types(self):
        lines = self._line(type="tool_result", message={"role": "tool", "content": "data"})
        assert self.mod.parse_messages(lines) == []

    def test_filters_empty_content(self):
        lines = self._line(type="user", message={"role": "user", "content": [{"type": "tool_use"}]})
        assert self.mod.parse_messages(lines) == []

    def test_skips_malformed_json(self):
        lines = "not-json\n" + self._line(type="user", message={"role": "user", "content": "ok"})
        msgs = self.mod.parse_messages(lines)
        assert len(msgs) == 1


class TestOffsetReadWrite:
    def test_read_returns_zero_when_missing(self, tmp_path):
        mod = _reimport("call_after_turn")
        assert mod.read_offset(str(tmp_path / "missing.offset")) == 0

    def test_write_then_read(self, tmp_path):
        mod = _reimport("call_after_turn")
        path = str(tmp_path / "test.offset")
        mod.write_offset(path, 1234)
        assert mod.read_offset(path) == 1234


# ===========================================================================
# call_compose — build_additional_context
# ===========================================================================


class TestBuildAdditionalContext:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _reimport("call_compose")

    def test_includes_identity_and_evidence(self):
        data = {"identityContext": "I am Alice", "retrievedEvidence": "Past: X"}
        result = self.mod.build_additional_context(data)
        assert "I am Alice" in result
        assert "Past: X" in result
        assert result.startswith("[oG-Memory]")

    def test_returns_none_when_both_empty(self):
        assert self.mod.build_additional_context({}) is None
        assert (
            self.mod.build_additional_context({"identityContext": "", "retrievedEvidence": ""})
            is None
        )

    def test_works_with_only_identity(self):
        result = self.mod.build_additional_context({"identityContext": "profile data"})
        assert result is not None
        assert "profile data" in result

    def test_truncates_to_max_chars(self):
        data = {"identityContext": "x" * 20_000}
        result = self.mod.build_additional_context(data)
        assert len(result) <= self.mod.MAX_CONTEXT_CHARS + len("[oG-Memory]\n")


class TestCallComposeMain:
    def test_skips_short_prompt(self):
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "hi", "session_id": "s1"})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_skips_slash_command(self):
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "/og-compose test", "session_id": "s1"})
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit) as exc:
                mod.main()
        assert exc.value.code == 0

    def test_outputs_additional_context_on_success(self, monkeypatch, capsys):
        monkeypatch.setenv("OG_MEMORY_URL", "http://localhost:8090")
        mod = _reimport("call_compose")
        hook_input = json.dumps({"prompt": "what did we discuss", "session_id": "s1"})
        compose_response = {"identityContext": "Alice is a dev", "retrievedEvidence": ""}

        with (
            patch("sys.stdin") as mock_stdin,
            patch.object(mod, "call_compose", return_value=compose_response),
        ):
            mock_stdin.read.return_value = hook_input
            with pytest.raises(SystemExit):
                mod.main()

        captured = capsys.readouterr()
        out = json.loads(captured.out)
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "Alice is a dev" in out["hookSpecificOutput"]["additionalContext"]
