"""LoCoMO evaluation through real agent hosts and native oGMemory plugins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "tests" / "e2e"
SUPPORTED_AGENTS = ("openclaw", "hermes", "opencode", "claude-code", "dsh", "codex")
CONTEXT_KEYS = (
    "identityContext",
    "episodicContext",
    "sessionContext",
    "taskContext",
    "retrievedEvidence",
)
QA_WRITE_PATHS = ("/after_turn", "/commit", "/compact", "/dispose")

sys.path.insert(0, str(E2E))
import eval as locomo_eval  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    return ""


class ReadOnlyQaProxy:
    """Forward oGMemory traffic, suppressing lifecycle writes during QA."""

    def __init__(self, target_url: str) -> None:
        self.target_url = target_url.rstrip("/")
        self.read_only = False
        self.intercepted: list[str] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = self.path.split("?", 1)[0].rstrip("/")
                if proxy.read_only and any(path.endswith(suffix) for suffix in QA_WRITE_PATHS):
                    proxy.intercepted.append(path)
                    payload = json.dumps({"ok": True, "readOnly": True}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in {"host", "content-length", "connection"}
                }
                response = requests.request(
                    self.command,
                    f"{proxy.target_url}{self.path}",
                    headers=headers,
                    data=body,
                    timeout=1200,
                )
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def do_PUT(self) -> None:  # noqa: N802
                self._handle()

            def do_DELETE(self) -> None:  # noqa: N802
                self._handle()

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class AgentHost:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = (
            args.runtime_root.resolve()
            if args.runtime_root
            else args.output_dir / "runtime" / args.agent
        )
        self.logs = args.output_dir / "host-logs"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        invocation_path = args.output_dir / "host-invocations.json"
        if args.resume and invocation_path.exists():
            existing = json.loads(invocation_path.read_text(encoding="utf-8"))
            self.invocations = existing if isinstance(existing, list) else []
        else:
            self.invocations = []
        self.env = os.environ.copy()
        self.env.update(
            {
                "OGMEM_AGENT_LLM_API_KEY": args.llm_api_key,
                "OG_MEMORY_URL": args.ogmemory_url,
                "OG_AUTH_API_KEY": args.ogmemory_api_key,
                "OG_AUTH_ACCOUNT_ID": args.account_id,
                "OG_AUTH_USER_ID": args.user_id,
                "OG_AUTH_AGENT_ID": args.memory_agent_id,
                "OGMEM_TRACE_CAPTURE_CONTENT": "true",
            }
        )

    def prepare(self) -> None:
        pass

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        raise NotImplementedError

    def close(self) -> None:
        _write_json(self.args.output_dir / "host-invocations.json", self.invocations)

    def _run(
        self,
        command: list[str],
        label: str,
        *,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        check: bool = True,
        record: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=self.runtime,
                env=env or self.env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout or self.args.agent_timeout,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            log_path = self.logs / f"{label}.json"
            _write_json(
                log_path,
                {
                    "command": command[:1] + ["<arguments-redacted>"],
                    "timed_out": True,
                    "duration_seconds": duration,
                },
            )
            if record:
                self.invocations.append(
                    {
                        "label": label,
                        "executable": command[0],
                        "timed_out": True,
                        "duration_seconds": duration,
                        "log": str(log_path.relative_to(self.args.output_dir)),
                    }
                )
            raise
        duration = time.monotonic() - started
        log_path = self.logs / f"{label}.json"
        _write_json(
            log_path,
            {
                "command": command[:1] + ["<arguments-redacted>"],
                "exit_code": result.returncode,
                "duration_seconds": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if record:
            self.invocations.append(
                {
                    "label": label,
                    "executable": command[0],
                    "exit_code": result.returncode,
                    "duration_seconds": duration,
                    "log": str(log_path.relative_to(self.args.output_dir)),
                }
            )
        if check and result.returncode != 0:
            detail = "\n".join(part for part in (result.stdout, result.stderr) if part)[-2000:]
            raise RuntimeError(f"{self.args.agent} host failed ({result.returncode}): {detail}")
        return result

    def _require(self, executable: str) -> None:
        if not shutil.which(executable):
            raise RuntimeError(f"required agent executable not found: {executable}")


class OpenClawHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.openclaw_command)
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", self.args.openclaw_port)) == 0:
                raise RuntimeError(f"OpenClaw gateway port is already in use: {self.args.openclaw_port}")
        self.home = self.runtime / "home"
        self.profile = f"locomo-{self.args.run_id}"
        self.state = self.home / f".openclaw-{self.profile}"
        self.workspace = self.runtime / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)
        self.plugin_dir = self.runtime / "plugin"
        _copy_tree(ROOT / "plugins/openclaw", self.plugin_dir)
        for path in [self.runtime, self.home, self.state, self.workspace, self.plugin_dir]:
            path.chmod(0o700)
        for path in self.plugin_dir.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.token = uuid.uuid4().hex
        self.env["HOME"] = str(self.home)
        self.env["OPENCLAW_GATEWAY_TOKEN"] = self.token
        config = {
            "agents": {
                "defaults": {
                    "workspace": str(self.workspace),
                    "model": {"primary": f"locomo/{self.args.llm_model}"},
                    "timeoutSeconds": self.args.agent_timeout,
                    "params": {"thinking": "off", "reasoning": "off"},
                },
                "list": [{"id": "main", "model": f"locomo/{self.args.llm_model}"}],
            },
            "gateway": {
                "mode": "local",
                "bind": "loopback",
                "port": self.args.openclaw_port,
                "auth": {"mode": "token", "token": self.token},
                "http": {"endpoints": {"responses": {"enabled": True}}},
            },
            "models": {
                "providers": {
                    "locomo": {
                        "baseUrl": self.args.llm_base_url,
                        "apiKey": "${OGMEM_AGENT_LLM_API_KEY}",
                        "api": self.args.openclaw_provider_api,
                        "models": [{"id": self.args.llm_model, "name": self.args.llm_model}],
                    }
                }
            },
            "tools": {"profile": "minimal"},
            "plugins": {
                "load": {"paths": [str(self.plugin_dir)]},
                "slots": {"contextEngine": "og-memory-context-engine"},
                "entries": {
                    "og-memory-context-engine": {
                        "enabled": True,
                        "hooks": {"allowConversationAccess": True},
                        "config": {
                            "mode": "remote",
                            "memoryApiBaseUrl": self.args.ogmemory_url,
                            "authApiKey": self.args.ogmemory_api_key,
                            "authAccountId": self.args.account_id,
                            "accountId": self.args.account_id,
                            "userId": self.args.user_id,
                            "agentId": self.args.memory_agent_id,
                            "prefetchEnabled": True,
                            "prefetchTopK": 5,
                            "compactTakeoverEnabled": False,
                            "shortTermIndexMode": "sync",
                        },
                    }
                },
            },
        }
        _write_json(self.state / "openclaw.json", config)
        command = [self.args.openclaw_command, "--profile", self.profile]
        self._run(command + ["config", "validate"], "prepare-validate", record=False)
        self.gateway_log = (self.logs / "gateway.log").open("w", encoding="utf-8")
        self.gateway = subprocess.Popen(
            command
            + [
                "gateway",
                "run",
                "--port",
                str(self.args.openclaw_port),
                "--bind",
                "loopback",
                "--auth",
                "token",
                "--token",
                self.token,
            ],
            cwd=self.runtime,
            env=self.env,
            text=True,
            stdout=self.gateway_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.gateway.poll() is not None:
                raise RuntimeError("OpenClaw gateway exited during startup")
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.args.openclaw_port}/health",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=2,
                )
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError("OpenClaw gateway did not become healthy")

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        started = time.monotonic()
        response = requests.post(
            f"http://127.0.0.1:{self.args.openclaw_port}/v1/responses",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "model": "openclaw",
                "input": prompt,
                "stream": False,
                "user": f"{self.args.user_id}-{label}",
            },
            timeout=self.args.agent_timeout,
        )
        duration = time.monotonic() - started
        if not response.ok:
            log_path = self.logs / f"{label}.json"
            _write_json(
                log_path,
                {
                    "duration_seconds": duration,
                    "status_code": response.status_code,
                    "response": response.text,
                },
            )
            response.raise_for_status()
        body = response.json()
        text = locomo_eval.extract_response_text(body)
        usage = body.get("usage") or {}
        log_path = self.logs / f"{label}.json"
        _write_json(log_path, {"duration_seconds": duration, "response": body})
        self.invocations.append(
            {
                "label": label,
                "executable": self.args.openclaw_command,
                "transport": "/v1/responses",
                "exit_code": 0,
                "duration_seconds": duration,
                "log": str(log_path.relative_to(self.args.output_dir)),
            }
        )
        return text, {key: int(usage.get(key, 0) or 0) for key in ("input_tokens", "output_tokens", "total_tokens")}

    def close(self) -> None:
        if hasattr(self, "gateway") and self.gateway.poll() is None:
            self.gateway.send_signal(signal.SIGTERM)
            try:
                self.gateway.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.gateway.kill()
                self.gateway.wait(timeout=5)
        if hasattr(self, "profile"):
            self._run(
                [self.args.openclaw_command, "--profile", self.profile, "gateway", "stop"],
                "gateway-stop",
                check=False,
                record=False,
            )
        if hasattr(self, "gateway_log"):
            self.gateway_log.close()
        super().close()


class OpenCodeHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.opencode_command)
        self.env["HOME"] = str(self.runtime / "home")
        self.env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {
                "plugin": [str(ROOT / "plugins/opencode" / "index.mjs")],
                "provider": {
                    "locomo": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "LoCoMO evaluator",
                        "options": {
                            "baseURL": self.args.llm_base_url,
                            "apiKey": "{env:OGMEM_AGENT_LLM_API_KEY}",
                        },
                        "models": {self.args.llm_model: {"name": self.args.llm_model}},
                    }
                },
            }
        )
        (self.runtime / "workspace").mkdir(parents=True, exist_ok=True)
        result = self._run([self.args.opencode_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.opencode_command,
                "run",
                "--format",
                "json",
                "--model",
                f"locomo/{self.args.llm_model}",
                "--dir",
                str(self.runtime / "workspace"),
                prompt,
            ],
            label,
        )
        answer: list[str] = []
        usage = defaultdict(int)
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part") or event.get("properties", {}).get("part") or {}
            if event.get("type") == "text" or part.get("type") == "text":
                text = part.get("text") or event.get("text")
                if text:
                    answer.append(str(text))
            tokens = part.get("tokens") or event.get("tokens") or {}
            usage["input_tokens"] += int(tokens.get("input", 0) or 0)
            usage["output_tokens"] += int(tokens.get("output", 0) or 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if not answer:
            raise RuntimeError("OpenCode returned no assistant text event")
        return "".join(answer).strip(), dict(usage)


class ClaudeCodeHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.claude_command)
        self.env["HOME"] = str(self.runtime / "home")
        self.env.pop("ANTHROPIC_API_KEY", None)
        self.env.pop("ANTHROPIC_AUTH_TOKEN", None)
        auth_env = "ANTHROPIC_AUTH_TOKEN" if self.args.claude_auth_token else "ANTHROPIC_API_KEY"
        self.env[auth_env] = self.args.llm_api_key
        self.env["ANTHROPIC_BASE_URL"] = (
            self.args.claude_base_url or self.args.llm_base_url.removesuffix("/v1")
        )
        self.env.update(
            {
                "OG_MEMORY_URL": self.args.ogmemory_url,
                "OG_MEMORY_ACCOUNT_ID": self.args.account_id,
                "OG_MEMORY_USER_ID": self.args.user_id,
                "OG_MEMORY_AGENT_ID": self.args.memory_agent_id,
                "OG_AUTH_API_KEY": self.args.ogmemory_api_key,
            }
        )
        self.plugin_dir = self.runtime / "plugin"
        _copy_tree(ROOT / "plugins/claude-code", self.plugin_dir)
        result = self._run([self.args.claude_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.claude_command,
                "-p",
                "--output-format",
                "json",
                "--plugin-dir",
                str(self.plugin_dir),
                "--model",
                self.args.llm_model,
                prompt,
            ],
            label,
        )
        try:
            body = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Claude Code returned invalid JSON") from error
        answer = str(body.get("result") or "").strip()
        if not answer:
            raise RuntimeError(f"Claude Code returned no result: {body.get('subtype', '')}")
        raw_usage = body.get("usage") or {}
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return answer, usage


class HermesHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.hermes_command)
        self.home = self.runtime / "home"
        self.env["HOME"] = str(self.home)
        self.env["HERMES_HOME"] = str(self.home)
        self.env["OPENAI_API_KEY"] = self.args.llm_api_key
        self.env["OGMEM_AGENT_LLM_API_KEY"] = self.args.llm_api_key
        self.env.update(
            {
                "OGMEMORY_ENDPOINT": self.args.ogmemory_url,
                "OGMEMORY_API_KEY": self.args.ogmemory_api_key,
                "OGMEMORY_ACCOUNT_ID": self.args.account_id,
                "OGMEMORY_USER_ID": self.args.user_id,
                "OGMEMORY_AGENT_ID": self.args.memory_agent_id,
                "OGMEMORY_TIMEOUT_SECONDS": str(self.args.agent_timeout),
            }
        )
        _copy_tree(ROOT / "plugins/hermes" / "ogmemory", self.home / "plugins" / "ogmemory")
        _write_yaml(
            self.home / "config.yaml",
            {
                "model": {
                    "provider": "locomo",
                    "default": self.args.llm_model,
                    "base_url": self.args.llm_base_url,
                    "api_mode": "chat_completions",
                },
                "custom_providers": [
                    {
                        "name": "locomo",
                        "base_url": self.args.llm_base_url,
                        "key_env": "OGMEM_AGENT_LLM_API_KEY",
                        "api_mode": "chat_completions",
                        "model": self.args.llm_model,
                    }
                ],
                "memory": {"provider": "ogmemory"},
            },
        )
        result = self._run([self.args.hermes_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()
        self._run([self.args.hermes_command, "memory", "status"], "prepare-memory", record=False)

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [self.args.hermes_command, "--ignore-rules", "-z", prompt],
            label,
        )
        answer = result.stdout.strip()
        if not answer:
            raise RuntimeError("Hermes returned no assistant text")
        return answer, {}


class DshHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.dsh_command)
        self._require("pnpm")
        self.home = self.runtime / "home"
        self.env["HOME"] = str(self.home)
        self.env["DSH_HOME"] = str(self.home / ".dsh")
        self.env["DEEPSEEK_API_KEY"] = self.args.llm_api_key
        self.env.update(
            {
                "OGMEM_URL": self.args.ogmemory_url,
                "OGMEM_DSH_API_KEY": self.args.ogmemory_api_key,
                "OGMEM_ACCOUNT_ID": self.args.account_id,
                "OGMEM_USER_ID": self.args.user_id,
                "OGMEM_AGENT_ID": self.args.memory_agent_id,
                "OGMEM_DSH_TOOLS": "false",
                "OGMEM_DSH_FORCE_EXTRACT": "true",
                "DSH_PERMISSION_MODE": "read-only",
            }
        )
        self._run(
            [
                self.args.dsh_command,
                "plugin",
                "--profile",
                "headless",
                "add",
                str(ROOT / "plugins/deepseek-harness"),
            ],
            "prepare-plugin",
            timeout=max(self.args.agent_timeout, 600),
            record=False,
        )
        _write_yaml(
            Path(self.env["DSH_HOME"]) / "settings.yaml",
            {
                "agent-default-model": {"provider": "locomo", "model": self.args.llm_model},
                "llm-pi-ai": {
                    "providers": {
                        "locomo": {
                            "apiKeyEnv": "OGMEM_AGENT_LLM_API_KEY",
                            "api": "openai-completions",
                            "baseURL": self.args.llm_base_url,
                            "models": [{"id": self.args.llm_model}],
                        }
                    }
                },
            },
        )
        result = self._run([self.args.dsh_command, "--version"], "prepare-version", record=False)
        self.version = (result.stdout or result.stderr).strip()
        self._run(
            [self.args.dsh_command, "--profile", "headless", "--dump-config"],
            "prepare-config",
            record=False,
        )

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [self.args.dsh_command, "--profile", "headless", prompt],
            label,
        )
        answer = result.stdout.strip()
        if not answer:
            raise RuntimeError("DSH returned no assistant text")
        return answer, {}


class CodexHost(AgentHost):
    def prepare(self) -> None:
        self._require(self.args.codex_command)
        self.home = self.runtime / "home"
        self.codex_home = self.home / ".codex"
        self.workspace = self.runtime / "workspace"
        self.codex_home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.env["HOME"] = str(self.home)
        self.env["CODEX_HOME"] = str(self.codex_home)
        self.env.update(
            {
                "OG_MEMORY_URL": self.args.ogmemory_url,
                "OG_MEMORY_ACCOUNT_ID": self.args.account_id,
                "OG_MEMORY_USER_ID": self.args.user_id,
                "OG_MEMORY_AGENT_ID": self.args.memory_agent_id,
                "OG_AUTH_API_KEY": self.args.ogmemory_api_key,
            }
        )
        config_path = self.codex_home / "config.toml"
        provider_config = (
            f"model = {json.dumps(self.args.llm_model)}\n"
            'model_provider = "locomo"\n\n'
            "[model_providers.locomo]\n"
            'name = "LoCoMO evaluator"\n'
            f"base_url = {json.dumps(self.args.llm_base_url)}\n"
            'env_key = "OGMEM_AGENT_LLM_API_KEY"\n'
            'wire_api = "responses"\n'
        )
        config_path.write_text(provider_config, encoding="utf-8")
        result = self._run([self.args.codex_command, "--version"], "prepare-version", record=False)
        self.version = result.stdout.strip()
        self._run(
            [
                self.args.codex_command,
                "plugin",
                "marketplace",
                "add",
                str(ROOT),
                "--json",
            ],
            "prepare-marketplace",
            record=False,
        )
        self._run(
            [self.args.codex_command, "plugin", "add", "ogmemory@ogmemory", "--json"],
            "prepare-plugin",
            record=False,
        )

    def invoke(self, prompt: str, label: str) -> tuple[str, dict[str, int]]:
        result = self._run(
            [
                self.args.codex_command,
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--dangerously-bypass-hook-trust",
                "--sandbox",
                "read-only",
                "--cd",
                str(self.workspace),
                prompt,
            ],
            label,
        )
        answer: list[str] = []
        usage = defaultdict(int)
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text")
                if text:
                    answer.append(str(text))
            if event.get("type") == "turn.completed":
                raw_usage = event.get("usage") or {}
                for key in ("input_tokens", "output_tokens"):
                    usage[key] += int(raw_usage.get(key, 0) or 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if not answer:
            raise RuntimeError("Codex returned no assistant text event")
        return "".join(answer).strip(), dict(usage)


def create_host(args: argparse.Namespace) -> AgentHost:
    hosts = {
        "openclaw": OpenClawHost,
        "opencode": OpenCodeHost,
        "claude-code": ClaudeCodeHost,
        "hermes": HermesHost,
        "dsh": DshHost,
        "codex": CodexHost,
    }
    return hosts[args.agent](args)


def is_retryable_host_error(error: Exception) -> bool:
    if isinstance(error, (requests.ConnectionError, requests.Timeout, subprocess.TimeoutExpired)):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code == 429 or error.response.status_code >= 500
    detail = str(error).lower()
    return any(
        marker in detail
        for marker in (
            "429",
            "rate limit",
            "too many requests",
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "502",
            "503",
            "504",
        )
    )


def is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code == 429
    detail = str(error).lower()
    return "429" in detail or "rate limit" in detail or "too many requests" in detail


def invoke_with_retries(
    host: AgentHost, prompt: str, label: str
) -> tuple[str, dict[str, int]]:
    for attempt in range(1, host.args.agent_attempts + 1):
        attempt_label = label if attempt == 1 else f"{label}-retry-{attempt}"
        try:
            return host.invoke(prompt, attempt_label)
        except Exception as error:
            rate_limited = is_rate_limit_error(error)
            if host.args.stop_on_rate_limit and rate_limited:
                raise
            if attempt == host.args.agent_attempts or not is_retryable_host_error(error):
                raise
            delay = (
                host.args.rate_limit_retry_delay
                if rate_limited
                else host.args.agent_retry_delay * attempt
            )
            print(
                f"{label} host attempt {attempt}/{host.args.agent_attempts} failed; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable host retry state")


def verify_plugin_write(args: argparse.Namespace, prompt: str) -> None:
    headers = {"Content-Type": "application/json"}
    if args.ogmemory_api_key:
        headers.update(
            {
                "X-API-Key": args.ogmemory_api_key,
                "X-Account-ID": args.account_id,
                "X-User-ID": args.user_id,
            }
        )
    deadline = time.monotonic() + args.write_wait_timeout
    while time.monotonic() < deadline:
        response = requests.post(
            f"{args.ogmemory_url}/api/v1/compose",
            headers=headers,
            json={
                "accountId": args.account_id,
                "userId": args.user_id,
                "agentId": args.memory_agent_id,
                "sessionId": f"verify-{args.run_id}-{uuid.uuid4().hex[:8]}",
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "tokenBudget": 6000,
            },
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        context = "\n".join(_flatten_text(body.get(key)) for key in CONTEXT_KEYS).strip()
        if context:
            _write_json(args.output_dir / "plugin-write-verification.json", body)
            return
        time.sleep(2)
    raise RuntimeError("plugin write verification found no recalled oGMemory context")


def write_summary(path: Path, rows: list[dict[str, Any]], timings: dict[str, float]) -> dict[str, Any]:
    correct = sum(row.get("is_correct") == "CORRECT" for row in rows)
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tokens = defaultdict(int)
    for row in rows:
        bucket = by_category[str(row.get("category", ""))]
        bucket[1] += 1
        bucket[0] += row.get("is_correct") == "CORRECT"
        usage = json.loads(row.get("usage") or "{}")
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            tokens[key] += int(usage.get(key, 0))
    summary = {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0,
        "by_category": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(by_category.items())
        },
        "tokens": dict(tokens),
        "timing_seconds": timings,
    }
    _write_json(path, summary)
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def evaluation_signature(
    item: dict[str, Any], sessions: list[dict[str, Any]], qas: list[dict[str, Any]]
) -> str:
    payload = {
        "sample_id": item.get("sample_id"),
        "sessions": sessions,
        "qas": qas,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser(agent: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"LoCoMO evaluation through real {agent} host")
    parser.set_defaults(agent=agent)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sessions", help="Ingest session range, for example 1-4")
    parser.add_argument("--count", type=positive_int)
    parser.add_argument("--ogmemory-url", default="http://127.0.0.1:8090")
    parser.add_argument("--account-id", default="locomo-agent-eval")
    parser.add_argument("--user-id", default=f"locomo-{agent}-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--memory-agent-id", default=agent)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    parser.add_argument("--ogmemory-api-key-env", default="OGMEM_LOCOMO_OGMEMORY_API_KEY")
    parser.add_argument("--llm-api-key-env", default="OGMEM_API_KEY")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--claude-base-url")
    parser.add_argument("--claude-auth-token", action="store_true")
    parser.add_argument("--judge-api-key-env", default="OGMEM_API_KEY")
    parser.add_argument("--judge-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-parallel", type=int, default=6)
    parser.add_argument("--judge-max-attempts", type=int, default=3)
    parser.add_argument("--qa-parallel", type=int, default=1)
    parser.add_argument("--agent-timeout", type=int, default=1200)
    parser.add_argument("--agent-attempts", type=int, default=5)
    parser.add_argument("--agent-retry-delay", type=int, default=5)
    parser.add_argument("--rate-limit-retry-delay", type=int, default=30)
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--allow-qa-writeback", action="store_true")
    parser.add_argument("--write-wait-timeout", type=int, default=1200)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--openclaw-command", default="openclaw")
    parser.add_argument("--openclaw-port", type=int, default=18791)
    parser.add_argument(
        "--openclaw-provider-api",
        choices=("openai-responses", "openai-completions"),
        default="openai-responses",
    )
    parser.add_argument("--opencode-command", default="opencode")
    parser.add_argument("--claude-command", default="claude")
    parser.add_argument("--hermes-command", default="hermes")
    parser.add_argument("--dsh-command", default="dsh")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main_for(agent: str, argv: list[str] | None = None) -> int:
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"unsupported agent: {agent}")
    args = build_parser(agent).parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output_dir / "run-manifest.json"
    previous_manifest = None
    if args.resume:
        if not manifest_path.exists():
            raise RuntimeError("cannot resume without run-manifest.json")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.allow_qa_writeback and previous_manifest.get("qa_writeback") is not False:
            raise RuntimeError(
                "cannot resume a run whose QA may have written to memory; use a fresh output directory"
            )
        args.account_id = str(previous_manifest["account_id"])
        args.user_id = str(previous_manifest["user_id"])
        args.memory_agent_id = str(previous_manifest["memory_agent_id"])
        args.run_id = str(previous_manifest["run_id"])
    args.ogmemory_url = args.ogmemory_url.rstrip("/")
    args.llm_base_url = args.llm_base_url.rstrip("/")
    if args.claude_base_url:
        args.claude_base_url = args.claude_base_url.rstrip("/")
    args.judge_base_url = args.judge_base_url.rstrip("/")
    if args.skip_ingest and args.skip_qa:
        raise ValueError("--skip-ingest and --skip-qa cannot be combined")
    if args.agent_attempts < 1 or args.agent_retry_delay < 0 or args.rate_limit_retry_delay < 0:
        raise ValueError("agent retry settings must be non-negative with at least one attempt")
    if args.qa_parallel < 1:
        raise ValueError("qa parallelism must be at least one")
    args.ogmemory_api_key = os.environ.get(args.ogmemory_api_key_env, "")
    args.llm_api_key = os.environ.get(args.llm_api_key_env, "")
    judge_key = os.environ.get(args.judge_api_key_env, "") or args.llm_api_key
    if not args.llm_api_key or not judge_key:
        raise RuntimeError("LLM and judge API key environment variables are required")

    samples = locomo_eval.load_locomo_data(str(args.dataset), args.sample)
    if len(samples) != 1:
        raise RuntimeError("exactly one LoCoMO sample must be selected")
    item = samples[0]
    session_range = locomo_eval.parse_session_range(args.sessions) if args.sessions else None
    sessions = locomo_eval.build_session_messages(item, session_range)
    qas = [qa for qa in item["qa"] if str(qa.get("category", "")) != "5"]
    if args.count is not None:
        qas = qas[: args.count]
    resume_contract = {
        "agent": agent,
        "dataset": str(args.dataset),
        "sample": args.sample,
        "sessions": args.sessions,
        "questions": len(qas),
        "evaluation_signature": evaluation_signature(item, sessions, qas),
        "ogmemory_url": args.ogmemory_url,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "claude_base_url": args.claude_base_url,
        "claude_auth": "auth_token" if args.claude_auth_token else "api_key",
        "openclaw_provider_api": args.openclaw_provider_api,
        "qa_writeback": args.allow_qa_writeback,
    }
    if previous_manifest is not None:
        mismatches = [
            key for key, value in resume_contract.items() if previous_manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(f"cannot resume incompatible run; changed: {', '.join(mismatches)}")
    question_time = locomo_eval.get_sample_question_time(item)
    qa_proxy = None
    if not args.allow_qa_writeback:
        qa_proxy = ReadOnlyQaProxy(args.ogmemory_url)
        interceptions_path = args.output_dir / "qa-writeback-interceptions.json"
        if args.resume and interceptions_path.exists():
            previous = json.loads(interceptions_path.read_text(encoding="utf-8"))
            qa_proxy.intercepted.extend(str(path) for path in previous.get("paths", []))
        qa_proxy.start()
        args.ogmemory_url = qa_proxy.url
    host = create_host(args)
    timings_path = args.output_dir / "run-timings.json"
    if args.resume and timings_path.exists():
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    else:
        timings: dict[str, float] = {}
    try:
        host.prepare()
        _write_json(
            manifest_path,
            {
                **resume_contract,
                "host_class": type(host).__name__,
                "account_id": args.account_id,
                "user_id": args.user_id,
                "memory_agent_id": args.memory_agent_id,
                "run_id": args.run_id,
                "stop_on_rate_limit": args.stop_on_rate_limit,
                "rate_limit_retry_delay": args.rate_limit_retry_delay,
                "qa_parallel": args.qa_parallel,
                "bm25": False,
                "rrf": False,
            },
        )
        if not args.skip_ingest:
            prior_ingest = float(timings.get("ingest", 0.0))
            started = time.monotonic()
            ingest_path = args.output_dir / "ingest.json"
            if args.resume and ingest_path.exists():
                ingest_rows = json.loads(ingest_path.read_text(encoding="utf-8"))
            else:
                ingest_rows = []
            completed_sessions = {str(row["session"]) for row in ingest_rows}
            for index, session in enumerate(sessions, start=1):
                if session["meta"]["session_key"] in completed_sessions:
                    continue
                reply, usage = invoke_with_retries(host, session["message"], f"ingest-{index:03d}")
                ingest_rows.append(
                    {
                        "session": session["meta"]["session_key"],
                        "reply": reply,
                        "usage": usage,
                    }
                )
                _write_json(ingest_path, ingest_rows)
                print(f"ingest {len(ingest_rows)}/{len(sessions)}", flush=True)
                timings["ingest"] = prior_ingest + time.monotonic() - started
                _write_json(timings_path, timings)
            verify_plugin_write(args, sessions[0]["message"][:1000])
            timings["ingest"] = time.monotonic() - started
            timings["ingest"] += prior_ingest
            _write_json(timings_path, timings)
        else:
            timings.setdefault("ingest", 0.0)
        if args.skip_qa:
            _write_json(args.output_dir / "summary.json", {"ingest_only": True, "timing_seconds": timings})
            return 0

        if qa_proxy is not None:
            qa_proxy.read_only = True

        qa_csv = args.output_dir / "qa.csv"
        rows: list[dict[str, Any]] = []
        if qa_csv.exists() and args.resume:
            with qa_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        completed_questions = {int(row["qi"]) for row in rows}
        prior_qa = float(timings.get("qa", 0.0))
        started = time.monotonic()

        def answer_question(qi: int, qa: dict[str, Any]) -> dict[str, Any]:
            date_prefix = f"Current date: {question_time}.\n" if question_time else ""
            prompt = (
                "Answer this question directly and concisely. Use recalled historical memory when relevant. "
                "If memory lacks the answer, say so.\n"
                f"{date_prefix}Question: {qa['question']}"
            )
            answer, usage = invoke_with_retries(host, prompt, f"qa-{qi:03d}")
            return {
                "sample_id": item["sample_id"],
                "sample_idx": args.sample,
                "qi": qi,
                "question": str(qa["question"]),
                "expected": str(qa["answer"]),
                "response": answer,
                "category": str(qa.get("category", "")),
                "evidence": json.dumps(qa.get("evidence", []), ensure_ascii=False),
                "usage": json.dumps(usage),
            }

        def checkpoint_question(row: dict[str, Any]) -> None:
            rows.append(row)
            rows.sort(key=lambda item: int(item["qi"]))
            with qa_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"qa {len(rows)}/{len(qas)}", flush=True)
            timings["qa"] = prior_qa + time.monotonic() - started
            _write_json(timings_path, timings)

        pending_questions = [
            (qi, qa)
            for qi, qa in enumerate(qas, start=1)
            if qi not in completed_questions
        ]
        if args.qa_parallel == 1:
            for qi, qa in pending_questions:
                checkpoint_question(answer_question(qi, qa))
        elif pending_questions:
            with ThreadPoolExecutor(max_workers=args.qa_parallel) as executor:
                futures = [
                    executor.submit(answer_question, qi, qa) for qi, qa in pending_questions
                ]
                try:
                    for future in as_completed(futures):
                        checkpoint_question(future.result())
                except Exception:
                    for future in futures:
                        future.cancel()
                    timings["qa"] = prior_qa + time.monotonic() - started
                    _write_json(timings_path, timings)
                    raise
        rows.sort(key=lambda row: int(row["qi"]))
        timings["qa"] = prior_qa + time.monotonic() - started
        _write_json(timings_path, timings)
        with qa_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        judged_csv = args.output_dir / "qa_judged.csv"
        judge_env = os.environ.copy()
        judge_env["OGMEM_API_KEY"] = judge_key
        prior_judge = float(timings.get("judge", 0.0))
        started = time.monotonic()
        judge_command = [
                sys.executable,
                str(E2E / "judge_parallel.py"),
                str(qa_csv),
                "--output",
                str(judged_csv),
                "--api-base",
                args.judge_base_url,
                "--judge-model",
                args.judge_model,
                "--parallel",
                str(args.judge_parallel),
                "--max-attempts",
                str(args.judge_max_attempts),
                "--retry-delay",
                str(args.rate_limit_retry_delay),
            ]
        if args.stop_on_rate_limit:
            judge_command.append("--fail-fast-rate-limit")
        try:
            subprocess.run(
                judge_command,
                cwd=ROOT,
                env=judge_env,
                check=True,
            )
        finally:
            timings["judge"] = prior_judge + time.monotonic() - started
            _write_json(timings_path, timings)
        with judged_csv.open(encoding="utf-8-sig", newline="") as handle:
            judged_rows = list(csv.DictReader(handle))
        if len(judged_rows) != len(qas):
            raise RuntimeError(f"judge output incomplete: {len(judged_rows)}/{len(qas)}")
        summary = write_summary(args.output_dir / "summary.json", judged_rows, timings)
        print(f"accuracy={summary['accuracy']:.2%} ({summary['correct']}/{summary['questions']})")
        return 0
    finally:
        host.close()
        if qa_proxy is not None:
            _write_json(
                args.output_dir / "qa-writeback-interceptions.json",
                {"count": len(qa_proxy.intercepted), "paths": qa_proxy.intercepted},
            )
            summary_path = args.output_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["qa_writes_intercepted"] = len(qa_proxy.intercepted)
                _write_json(summary_path, summary)
            qa_proxy.close()
