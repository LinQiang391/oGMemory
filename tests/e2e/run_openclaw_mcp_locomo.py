"""One-command OpenClaw + oGMemory MCP LoCoMo evaluation.

Usage:
    python tests/e2e/run_openclaw_mcp_locomo.py /path/to/locomo.json

The dataset is the only positional argument. Judge credentials must be
configured in the environment or an explicit YAML file.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "tests" / "e2e" / "eval.py"
JUDGE = ROOT / "tests" / "e2e" / "judge_parallel.py"
EXPECTED_TOOLS = ("query_memory", "read_memory", "write_memory")
OGMEMORY_TOOLS = (
    "ogmemory__ogm_query_memory",
    "ogmemory__ogm_read_memory",
    "ogmemory__ogm_write_memory",
    "ogmemory__resources_list",
    "ogmemory__resources_read",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a clean LoCoMO evaluation through OpenClaw and oGMemory MCP.",
        epilog=(
            "Optional env: OGMEMORY_MCP_URL, LOCOMO_OGMEM_CONFIG, LOCOMO_QA_PARALLEL, "
            "LOCOMO_JUDGE_PARALLEL, LOCOMO_JUDGE_API_KEY, LOCOMO_JUDGE_BASE_URL, "
            "LOCOMO_JUDGE_MODEL, LOCOMO_JUDGE_API_MODE, LOCOMO_MCP_TIMEOUT_SECONDS, "
            "LOCOMO_OPENCLAW_CONFIG, LOCOMO_OPENCLAW_STATE_DIR."
        ),
    )
    parser.add_argument("dataset", type=Path, help="Path to locomo.json")
    parser.add_argument(
        "--output-dir", type=Path, help="Result directory; defaults under tests/e2e/result"
    )
    parser.add_argument(
        "--sample", type=int, action="append", help="Run selected 0-based sample; repeatable"
    )
    parser.add_argument(
        "--parallel", type=int, default=int(os.environ.get("LOCOMO_QA_PARALLEL", "2"))
    )
    parser.add_argument(
        "--judge-parallel", type=int, default=int(os.environ.get("LOCOMO_JUDGE_PARALLEL", "6"))
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate MCP/OpenClaw without ingest or scoring",
    )
    return parser


def _run_command(
    *args: str,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}):\n{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"LoCoMo dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("LoCoMo dataset must be a non-empty JSON array")
    sample_ids: set[str] = set()
    for index, item in enumerate(data):
        prefix = f"Invalid LoCoMo item at index {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: expected an object")
        if not isinstance(item.get("sample_id"), str) or not item["sample_id"].strip():
            raise ValueError(f"{prefix}: sample_id must be a non-empty string")
        if item["sample_id"] in sample_ids:
            raise ValueError(f"{prefix}: duplicate sample_id {item['sample_id']!r}")
        sample_ids.add(item["sample_id"])
        conversation = item.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"{prefix}: conversation must be an object")
        for speaker_key in ("speaker_a", "speaker_b"):
            if not isinstance(conversation.get(speaker_key), str):
                raise ValueError(f"{prefix}: conversation.{speaker_key} must be a string")
        session_count = 0
        for session_key, messages in conversation.items():
            if not session_key.startswith("session_") or session_key.endswith("_date_time"):
                continue
            if not session_key.removeprefix("session_").isdigit() or not isinstance(messages, list):
                raise ValueError(f"{prefix}: conversation.{session_key} must be a message array")
            for message_index, message in enumerate(messages):
                message_prefix = (
                    f"{prefix}: conversation.{session_key} message at index {message_index}"
                )
                if not isinstance(message, dict):
                    raise ValueError(f"{message_prefix} must be an object")
                for field in ("speaker", "text"):
                    if not isinstance(message.get(field), str) or not message[field].strip():
                        raise ValueError(f"{message_prefix}: {field} must be a non-empty string")
            session_count += 1
        if not session_count:
            raise ValueError(f"{prefix}: conversation must include at least one session_N array")
        qas = item.get("qa")
        if not isinstance(qas, list):
            raise ValueError(f"{prefix}: qa must be an array")
        evaluated_qa_count = 0
        for qa_index, qa in enumerate(qas):
            qa_prefix = f"{prefix}, QA at index {qa_index}"
            if not isinstance(qa, dict):
                raise ValueError(f"{qa_prefix}: expected an object")
            if str(qa.get("category", "")) == "5":
                continue
            evaluated_qa_count += 1
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise ValueError(f"{qa_prefix}: question must be a non-empty string")
            answer = qa.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError(f"{qa_prefix}: answer must be a string or number")
            if isinstance(answer, str) and not answer.strip():
                raise ValueError(f"{qa_prefix}: answer must not be empty")
        if not evaluated_qa_count:
            raise ValueError(f"{prefix}: qa must include at least one evaluated question")
    return data


def _read_openclaw_config() -> dict[str, Any]:
    return json.loads(_openclaw_config_path().read_text(encoding="utf-8"))


def _openclaw_config_path() -> Path:
    configured = os.environ.get("LOCOMO_OPENCLAW_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".openclaw" / "openclaw.json"


def _openclaw_state_dir() -> Path:
    configured = os.environ.get("LOCOMO_OPENCLAW_STATE_DIR") or os.environ.get("OPENCLAW_STATE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".openclaw"


def _main_agent_dir(config: dict[str, Any]) -> Path:
    for agent in config.get("agents", {}).get("list", []):
        if isinstance(agent, dict) and agent.get("id") == "main" and agent.get("agentDir"):
            return Path(str(agent["agentDir"])).expanduser()
    return _openclaw_state_dir() / "agents" / "main" / "agent"


def _gateway_url(config: dict[str, Any]) -> str:
    port = int(config.get("gateway", {}).get("port", 18789))
    return f"http://127.0.0.1:{port}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_private_text(path: Path, content: str) -> None:
    json.loads(content)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o600)


def _copy_auth_state(source_agent_dir: Path, destination_agent_dir: Path) -> list[str]:
    destination_agent_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(destination_agent_dir, 0o700)
    copied: list[str] = []
    database = source_agent_dir / "openclaw-agent.sqlite"
    if database.is_file():
        uri = f"{database.resolve().as_uri()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                for table, column, key_column, filename in (
                    ("auth_profile_store", "store_json", "store_key", "auth-profiles.json"),
                    ("auth_profile_state", "state_json", "state_key", "auth-state.json"),
                ):
                    if table not in tables:
                        continue
                    row = connection.execute(
                        f"SELECT {column} FROM {table} WHERE {key_column} = ?", ("primary",)
                    ).fetchone()
                    if row and isinstance(row[0], str):
                        _write_private_text(destination_agent_dir / filename, row[0])
                        copied.append(filename)
        except sqlite3.Error as exc:
            raise RuntimeError(f"Unable to read OpenClaw auth profile store: {exc}") from exc
    for filename in ("auth-profiles.json", "auth-state.json", "auth.json"):
        if filename in copied:
            continue
        source = source_agent_dir / filename
        if source.is_file():
            _write_private_text(
                destination_agent_dir / filename, source.read_text(encoding="utf-8")
            )
            copied.append(filename)
    return copied


def _start_gateway(
    config: dict[str, Any], runtime_home: Path, log_path: Path, auth_source_dir: Path
) -> tuple[subprocess.Popen, dict[str, str]]:
    state_dir = runtime_home / ".openclaw"
    state_dir.mkdir(parents=True)
    destination_agent_dir = state_dir / "agents" / "main" / "agent"
    _copy_auth_state(auth_source_dir, destination_agent_dir)
    runtime_config = copy.deepcopy(config)
    models_path = auth_source_dir / "models.json"
    if models_path.is_file():
        stored_models = json.loads(models_path.read_text(encoding="utf-8"))
        stored_providers = stored_models.get("providers", {})
        if isinstance(stored_providers, dict):
            configured_providers = runtime_config.setdefault("models", {}).setdefault(
                "providers", {}
            )
            merged_providers = stored_providers
            for name, provider in configured_providers.items():
                if isinstance(provider, dict) and isinstance(
                    merged_providers.get(name), dict
                ):
                    merged_providers[name] = {**merged_providers[name], **provider}
                else:
                    merged_providers[name] = provider
            runtime_config["models"]["providers"] = merged_providers
    workspace = state_dir / "workspace"
    workspace.mkdir()
    runtime_config.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = str(
        workspace
    )
    for agent in runtime_config.get("agents", {}).get("list", []):
        if isinstance(agent, dict) and agent.get("id") == "main":
            agent["agentDir"] = str(destination_agent_dir)
    config_path = state_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.chmod(config_path, 0o600)
    env = os.environ.copy()
    env.update(
        {
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_REAPPLY_CONFIG": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["openclaw", "gateway", "--port", str(config["gateway"]["port"])],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    url = _gateway_url(config) + "/health"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Isolated OpenClaw gateway exited; inspect {log_path}")
        try:
            if requests.get(url, timeout=2).ok:
                return proc, env
        except requests.RequestException:
            pass
        time.sleep(0.5)
    _stop_gateway(proc)
    raise RuntimeError(f"Isolated OpenClaw gateway did not become healthy at {url}")


def _stop_gateway(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _mcp_headers(config: dict[str, Any]) -> dict[str, str]:
    server = config.get("mcp", {}).get("servers", {}).get("ogmemory", {})
    existing = server.get("headers", {}) if isinstance(server, dict) else {}
    return {
        "X-API-Key": os.environ.get(
            "OGMEMORY_MCP_API_KEY", str(existing.get("X-API-Key", "root-key"))
        ),
        "X-Account-ID": os.environ.get(
            "OGMEMORY_MCP_ACCOUNT_ID", str(existing.get("X-Account-ID", "acct-demo"))
        ),
        "X-User-ID": "preflight",
        "X-Agent-ID": "preflight-agent",
    }


def _probe_mcp_http(url: str, headers: dict[str, str]) -> bool:
    request_headers = {
        **headers,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "locomo-preflight", "version": "1"},
        },
    }
    try:
        response = requests.post(url, headers=request_headers, json=body, timeout=30)
        response.raise_for_status()
        if response.json().get("result", {}).get("serverInfo", {}).get("name") != "oGMemory":
            return False
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            request_headers["Mcp-Session-Id"] = session_id
            requests.post(
                url,
                headers=request_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=30,
            ).raise_for_status()
        tools_response = requests.post(
            url,
            headers=request_headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            timeout=30,
        )
        tools_response.raise_for_status()
        tools = [
            str(tool.get("name", ""))
            for tool in tools_response.json().get("result", {}).get("tools", [])
        ]
        return all(any(name in tool for tool in tools) for name in EXPECTED_TOOLS)
    except (requests.RequestException, ValueError):
        return False


def _discover_mcp(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    headers = _mcp_headers(config)
    configured = config.get("mcp", {}).get("servers", {}).get("ogmemory", {}).get("url")
    candidates = [
        os.environ.get("OGMEMORY_MCP_URL"),
        configured,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        url = str(candidate).rstrip("/")
        if not url.endswith("/mcp"):
            url += "/mcp"
        if url in seen:
            continue
        seen.add(url)
        if _probe_mcp_http(url, headers):
            return url, headers
    raise RuntimeError("The oGMemory MCP configured in OpenClaw is not reachable or authenticated")


def _validate_mcp_user_scope(mcp_url: str, headers: dict[str, str]) -> None:
    account_id = headers["X-Account-ID"]
    url = f"{mcp_url.removesuffix('/mcp')}/api/v1/admin/accounts/{account_id}/roles"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 503:
            return
        if response.status_code == 403:
            raise RuntimeError(
                "LoCoMO evaluation requires an Admin or Root oGMemory API key so "
                "X-User-ID can isolate each sample"
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to validate oGMemory API key privileges: {exc}") from exc


def _evaluation_config(
    original: dict[str, Any],
    *,
    mcp_url: str,
    headers: dict[str, str],
    user_id: str,
    memory_agent_id: str,
    gateway_token: str,
) -> dict[str, Any]:
    config = copy.deepcopy(original)
    gateway = config.setdefault("gateway", {})
    gateway["auth"] = {"mode": "token", "token": gateway_token}
    endpoints = gateway.setdefault("http", {}).setdefault("endpoints", {})
    endpoints.setdefault("responses", {})["enabled"] = True
    config.setdefault("mcp", {})["servers"] = {
        "ogmemory": {
            "url": mcp_url,
            "transport": "streamable-http",
            "connectTimeout": 30,
            "timeout": int(os.environ.get("LOCOMO_MCP_TIMEOUT_SECONDS", "1200")),
            "supportsParallelToolCalls": True,
            "toolFilter": {"include": ["ogm_*", "resources_*"]},
            "headers": {
                **headers,
                "X-User-ID": user_id,
                "X-Agent-ID": memory_agent_id,
            },
        }
    }

    plugins = config.setdefault("plugins", {})
    slots = plugins.setdefault("slots", {})
    memory_slot = slots.get("memory")
    slots["memory"] = "none"
    slots["contextEngine"] = "legacy"
    for name, entry in plugins.get("entries", {}).items():
        normalized_name = name.lower()
        if (
            name == memory_slot or "memory" in normalized_name or "mem0" in normalized_name
        ) and isinstance(entry, dict):
            entry["enabled"] = False

    config["tools"] = {
        "profile": "minimal",
        "alsoAllow": list(OGMEMORY_TOOLS),
        "deny": ["session_status"],
    }
    return config


def _sample_scope(run_id: str, index: int) -> tuple[str, str]:
    user_id = f"{run_id}-conv-{index}"
    return user_id, f"{user_id}-agent"


def _new_run_id() -> str:
    return f"openclaw_mcp_locomo_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


def _parse_first_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise RuntimeError("OpenClaw MCP probe returned no JSON")


def _probe_openclaw(env: dict[str, str]) -> None:
    result = _run_command(
        "openclaw",
        "mcp",
        "probe",
        "ogmemory",
        "--json",
        check=False,
        timeout=120,
        env=env,
    )
    payload = _parse_first_json(result.stdout + "\n" + result.stderr)
    diagnostics = payload.get("diagnostics", [])
    tools = [str(tool) for tool in payload.get("tools", [])]
    missing = [
        name
        for name in EXPECTED_TOOLS
        if not any(name.replace("_", "-") in tool or name in tool for tool in tools)
    ]
    if diagnostics or missing:
        raise RuntimeError(
            f"OpenClaw MCP probe failed: diagnostics={diagnostics}, missing_tools={missing}, tools={tools}"
        )


def _judge_settings() -> tuple[str, str, str, str]:
    key = os.environ.get("LOCOMO_JUDGE_API_KEY") or os.environ.get("OGMEM_API_KEY")
    base = os.environ.get("LOCOMO_JUDGE_BASE_URL")
    model = os.environ.get("LOCOMO_JUDGE_MODEL")
    mode = os.environ.get("LOCOMO_JUDGE_API_MODE")

    local_config_value = os.environ.get("LOCOMO_OGMEM_CONFIG")
    if local_config_value:
        local_config_path = Path(local_config_value).expanduser()
        if not local_config_path.is_file():
            raise RuntimeError(f"LOCOMO_OGMEM_CONFIG not found: {local_config_path}")
        local_config = yaml.safe_load(local_config_path.read_text(encoding="utf-8")) or {}
        llm = local_config.get("llm", {})
        key = key or llm.get("api_key")
        base = base or llm.get("base_url")
        model = model or llm.get("model")
        mode = mode or llm.get("api_mode") or "chat"
    if not all((key, base, model)):
        raise RuntimeError(
            "Judge credentials unavailable. Set LOCOMO_JUDGE_API_KEY, "
            "LOCOMO_JUDGE_BASE_URL, and LOCOMO_JUDGE_MODEL, or set LOCOMO_OGMEM_CONFIG."
        )
    mode = mode or "chat"
    if mode not in ("chat", "responses"):
        raise RuntimeError("LOCOMO_JUDGE_API_MODE must be chat or responses")
    return str(key), str(base), str(model), mode


def _run_logged(command: list[str], log_prefix: Path, *, env: dict[str, str]) -> None:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout,
        log_prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed; inspect {log_prefix}.stderr.log")


def _merge_csv(inputs: list[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in inputs:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_qa_csv(path: Path, data: list[dict[str, Any]], indices: list[int]) -> None:
    expected: Counter[tuple[str, str, str, str]] = Counter()
    for index in indices:
        item = data[index]
        qas = [qa for qa in item["qa"] if str(qa.get("category", "")) != "5"]
        expected.update(
            (
                str(item["sample_id"]),
                str(qi),
                str(qa["question"]),
                str(qa["answer"]),
            )
            for qi, qa in enumerate(qas, start=1)
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = Counter(
        (
            row.get("sample_id", ""),
            row.get("qi", ""),
            row.get("question", ""),
            row.get("expected", ""),
        )
        for row in rows
    )
    if actual != expected:
        missing = sum((expected - actual).values())
        unexpected = sum((actual - expected).values())
        raise RuntimeError(
            "QA output is incomplete or mismatched: "
            f"expected={sum(expected.values())}, actual={sum(actual.values())}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid_responses = [
        f"{row.get('sample_id', '')}/Q{row.get('qi', '')}"
        for row in rows
        if not (response := str(row.get("response") or "").strip())
        or response.startswith("[ERROR")
    ]
    if invalid_responses:
        raise RuntimeError(f"QA output contains empty or error responses: {invalid_responses}")


def _validate_ingest_result(path: Path, item: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"Ingest result missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid ingest result: {path}")
    expected = {
        key
        for key in item["conversation"]
        if key.startswith("session_") and not key.endswith("_date_time")
    }
    actual = {str(row.get("session", "")) for row in rows if isinstance(row, dict)}
    failures = [
        str(row.get("session", "")) if isinstance(row, dict) else "<invalid-row>"
        for row in rows
        if not isinstance(row, dict) or str(row.get("reply", "")).lstrip().startswith("[ERROR]")
    ]
    if actual != expected or failures:
        raise RuntimeError(
            "Ingest output is incomplete or failed: "
            f"expected={len(expected)}, actual={len(actual)}, failed={failures}"
        )


def _prepare_run_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_summary(run_dir: Path, judged_csv: Path, timings: dict[str, float]) -> dict[str, Any]:
    with judged_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row in rows:
        passed = row.get("is_correct") == "CORRECT"
        correct += int(passed)
        for bucket, key in (
            (by_sample, row.get("sample_id", "")),
            (by_category, row.get("category", "")),
        ):
            bucket[key][1] += 1
            bucket[key][0] += int(passed)
    summary = {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "by_sample": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(by_sample.items())
        },
        "by_category": {
            key: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(by_category.items())
        },
        "timing_seconds": timings,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OpenClaw oGMemory MCP LoCoMO",
        "",
        f"- Overall: {summary['accuracy']:.2%} ({correct}/{len(rows)})",
        f"- Ingest: {timings.get('ingest', 0):.1f}s",
        f"- QA: {timings.get('qa', 0):.1f}s",
        f"- Judge: {timings.get('judge', 0):.1f}s",
        "",
        "## By category",
    ]
    lines.extend(
        f"- {key}: {value['accuracy']:.2%} ({value['correct']}/{value['total']})"
        for key, value in summary["by_category"].items()
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    data = _load_dataset(dataset)
    indices = args.sample or list(range(len(data)))
    if any(index < 0 or index >= len(data) for index in indices):
        raise ValueError(f"Sample index out of range; dataset has {len(data)} items")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate --sample indices are not allowed")

    run_id = _new_run_id()
    run_dir = (args.output_dir or ROOT / "tests" / "e2e" / "result" / run_id).resolve()
    if not args.preflight_only:
        _prepare_run_dir(run_dir)
    original = _read_openclaw_config()
    gateway_token = secrets.token_urlsafe(32)
    mcp_url, headers = _discover_mcp(original)
    _validate_mcp_user_scope(mcp_url, headers)
    judge_settings = None if args.preflight_only else _judge_settings()
    auth_source_dir = _main_agent_dir(original)
    timings = {"ingest": 0.0, "qa": 0.0, "judge": 0.0}
    qa_csvs: list[Path] = []
    runtime_dir = tempfile.TemporaryDirectory(prefix="openclaw-locomo-")
    gateway_proc: subprocess.Popen | None = None
    try:
        for position, index in enumerate(indices):
            user_id, memory_agent_id = _sample_scope(run_id, index)
            sample_dir = run_dir / f"sample_{index}"
            if not args.preflight_only:
                sample_dir.mkdir(parents=True, exist_ok=True)
            config = _evaluation_config(
                original,
                mcp_url=mcp_url,
                headers=headers,
                user_id=user_id,
                memory_agent_id=memory_agent_id,
                gateway_token=gateway_token,
            )
            config.setdefault("gateway", {})["port"] = _free_port()
            runtime_home = Path(runtime_dir.name) / f"sample_{index}" / "home"
            gateway_log = (
                sample_dir / "openclaw-gateway.log"
                if not args.preflight_only
                else Path(runtime_dir.name) / "openclaw-gateway.log"
            )
            gateway_proc, gateway_env = _start_gateway(
                config, runtime_home, gateway_log, auth_source_dir
            )
            _probe_openclaw(gateway_env)
            if args.preflight_only:
                print(f"preflight ok: OpenClaw -> {mcp_url} ({len(EXPECTED_TOOLS)} required tools)")
                return 0

            eval_env = gateway_env.copy()
            eval_env.update(
                {
                    "HOME": str(runtime_home),
                    "USERPROFILE": str(runtime_home),
                    "OPENCLAW_GATEWAY_TOKEN": gateway_token,
                    "OGMEMORY_QA_ENABLE_READ": "1",
                    "LOCOMO_INCLUDE_CATEGORY5": "0",
                    "LOCOMO_INGEST_RECORD_PATH": str(run_dir / "ingest_record.json"),
                    "LOCOMO_API_LOG_DIR": str(run_dir / "api_logs"),
                    "OGMEMORY_MCP_API_KEY": headers["X-API-Key"],
                }
            )
            idle_url = mcp_url.removesuffix("/mcp")
            common = [
                "--base-url",
                _gateway_url(config),
                "--sample",
                str(index),
                "--user",
                user_id,
                "--agent-id",
                "main",
            ]
            started = time.monotonic()
            ingest_output = sample_dir / "ingest"
            _run_logged(
                [
                    sys.executable,
                    str(EVAL),
                    "ingest",
                    str(dataset),
                    "--output",
                    str(ingest_output),
                    *common,
                    "--ogmemory-url",
                    idle_url,
                    "--ogmemory-account-id",
                    headers["X-Account-ID"],
                    "--ogmemory-agent-id",
                    memory_agent_id,
                    "--ogmemory-idle-timeout",
                    os.environ.get("LOCOMO_IDLE_TIMEOUT", "1200"),
                    "--force-ingest",
                ],
                sample_dir / "ingest",
                env=eval_env,
            )
            _validate_ingest_result(ingest_output.with_suffix(".json"), data[index])
            timings["ingest"] += time.monotonic() - started

            qa_output = sample_dir / "qa"
            started = time.monotonic()
            _run_logged(
                [
                    sys.executable,
                    str(EVAL),
                    "qa",
                    str(dataset),
                    "--output",
                    str(qa_output),
                    *common,
                    "--parallel",
                    str(args.parallel),
                ],
                sample_dir / "qa",
                env=eval_env,
            )
            timings["qa"] += time.monotonic() - started
            qa_csvs.append(qa_output.with_suffix(".csv"))
            print(f"[{position + 1}/{len(indices)}] sample {index} complete")
            _stop_gateway(gateway_proc)
            gateway_proc = None
    finally:
        _stop_gateway(gateway_proc)
        runtime_dir.cleanup()

    qa_csv = run_dir / "qa.csv"
    judged_csv = run_dir / "qa_judged.csv"
    _merge_csv(qa_csvs, qa_csv)
    _validate_qa_csv(qa_csv, data, indices)
    if judge_settings is None:
        raise RuntimeError("Judge settings were not initialized")
    judge_key, judge_base, judge_model, judge_mode = judge_settings
    judge_env = os.environ.copy()
    judge_env["OGMEM_API_KEY"] = judge_key
    started = time.monotonic()
    _run_logged(
        [
            sys.executable,
            str(JUDGE),
            str(qa_csv),
            "--output",
            str(judged_csv),
            "--api-base",
            judge_base,
            "--judge-model",
            judge_model,
            "--api-mode",
            judge_mode,
            "--parallel",
            str(args.judge_parallel),
        ],
        run_dir / "judge",
        env=judge_env,
    )
    timings["judge"] = time.monotonic() - started

    with (
        (run_dir / "stat.txt").open("w", encoding="utf-8") as stdout,
        (run_dir / "stat.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            [sys.executable, str(EVAL), "stat", str(judged_csv)],
            cwd=ROOT,
            env=judge_env,
            stdout=stdout,
            stderr=stderr,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Stat failed; inspect {run_dir / 'stat.stderr.log'}")

    summary = _write_summary(run_dir, judged_csv, timings)
    print(f"accuracy={summary['accuracy']:.2%} ({summary['correct']}/{summary['questions']})")
    print(f"results={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
