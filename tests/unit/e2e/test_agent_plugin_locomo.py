from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from agent_plugin_locomo import (  # noqa: E402
    ClaudeCodeHost,
    ReadOnlyQaProxy,
    build_parser,
    evaluation_signature,
    invoke_with_retries,
    is_rate_limit_error,
    main_for,
)
from eval import build_session_messages  # noqa: E402


def test_claude_supports_separate_anthropic_endpoint(tmp_path: Path, monkeypatch) -> None:
    args = build_parser("claude-code").parse_args(
        [
            "locomo.json",
            "--output-dir",
            str(tmp_path),
            "--llm-base-url",
            "https://provider.example/openai/v1",
            "--claude-base-url",
            "https://provider.example/anthropic",
            "--claude-auth-token",
        ]
    )
    args.llm_api_key = "test-key"
    args.ogmemory_api_key = ""
    host = ClaudeCodeHost(args)
    monkeypatch.setattr(host, "_require", lambda _executable: None)
    monkeypatch.setattr(
        host,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "claude 1.0\n", ""),
    )

    host.prepare()

    assert host.env["ANTHROPIC_BASE_URL"] == "https://provider.example/anthropic"
    assert host.env["ANTHROPIC_AUTH_TOKEN"] == "test-key"
    assert "ANTHROPIC_API_KEY" not in host.env


def test_rate_limit_detection() -> None:
    assert is_rate_limit_error(RuntimeError("provider returned 429 Too Many Requests"))


def test_host_rate_limit_uses_fixed_retry_delay(monkeypatch) -> None:
    delays: list[int] = []

    class Host:
        args = type(
            "Args",
            (),
            {
                "agent_attempts": 3,
                "stop_on_rate_limit": False,
                "rate_limit_retry_delay": 30,
                "agent_retry_delay": 5,
            },
        )()

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _prompt: str, _label: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 Too Many Requests")
            return "ok", {}

    monkeypatch.setattr("agent_plugin_locomo.time.sleep", delays.append)
    host = Host()

    assert invoke_with_retries(host, "prompt", "qa-001") == ("ok", {})
    assert delays == [30]


def test_qa_parallelism_defaults_to_one() -> None:
    args = build_parser("hermes").parse_args(["locomo.json", "--output-dir", "results"])

    assert args.qa_parallel == 1
    assert args.rate_limit_retry_delay == 30
    assert args.allow_qa_writeback is False


def test_count_must_be_positive() -> None:
    parser = build_parser("dsh")

    with pytest.raises(SystemExit):
        parser.parse_args(["locomo.json", "--output-dir", "results", "--count", "0"])


def test_resume_rejects_changed_question_count(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "locomo.json"
    item = {
        "sample_id": "sample-0",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [{"speaker": "Alice", "text": "hello"}],
            "session_1_date_time": "1:00 pm on 1 January, 2024",
        },
        "qa": [
            {"question": "Q1", "answer": "A1", "category": 1, "evidence": []},
            {"question": "Q2", "answer": "A2", "category": 1, "evidence": []},
        ],
    }
    dataset.write_text(json.dumps([item]), encoding="utf-8")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    sessions = build_session_messages(item)
    manifest = {
        "agent": "dsh",
        "dataset": str(dataset.resolve()),
        "sample": 0,
        "sessions": None,
        "questions": 2,
        "evaluation_signature": evaluation_signature(item, sessions, item["qa"]),
        "ogmemory_url": "http://127.0.0.1:8090",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
        "claude_base_url": None,
        "claude_auth": "api_key",
        "openclaw_provider_api": "openai-responses",
        "qa_writeback": False,
        "account_id": "account",
        "user_id": "user",
        "memory_agent_id": "dsh",
        "run_id": "run",
    }
    (output_dir / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("OGMEM_API_KEY", "dummy")

    with pytest.raises(RuntimeError, match=r"changed: questions, evaluation_signature"):
        main_for(
            "dsh",
            [
                str(dataset),
                "--output-dir",
                str(output_dir),
                "--resume",
                "--skip-ingest",
                "--count",
                "1",
            ],
        )


def test_read_only_qa_proxy_forwards_compose_and_suppresses_writeback() -> None:
    calls: list[str] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = b'{"retrievedEvidence":"memory"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    proxy = ReadOnlyQaProxy(f"http://127.0.0.1:{target.server_port}")
    proxy.start()
    proxy.read_only = True
    try:
        compose = requests.post(f"{proxy.url}/api/v1/compose", json={"prompt": "q"}, timeout=5)
        writeback = requests.post(f"{proxy.url}/api/v1/after_turn", json={"messages": []}, timeout=5)
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)

    assert compose.json()["retrievedEvidence"] == "memory"
    assert writeback.json() == {"ok": True, "readOnly": True}
    assert calls == ["/api/v1/compose"]
    assert proxy.intercepted == ["/api/v1/after_turn"]
