"""Isolated OpenViking + OpenClaw LoCoMo runner.

This script is intentionally local-evaluation glue. It keeps OpenClaw's tool
surface limited to OpenViking MCP tools, clears the OpenViking workspace, and
throttles ingestion by waiting for OpenViking background commit tasks to finish.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import requests

import eval_openviking  # noqa: F401 - patches eval.py prompts for OpenViking
import eval as locomo_eval


BASE = pathlib.Path("/root/openviking-locomo")
WORKSPACE = BASE / "openviking-workspace"
DATASET = pathlib.Path("/root/hymemory-locomo-tests/work/locomo-test-kit/data/locomo10.json")
PYTHON = pathlib.Path("/root/.venvs/ogmemory-dev/bin/python")
OPENVIKING_PORT = 2033
LOCAL_NO_PROXY = "127.0.0.1,localhost,127.0.0.2"
OPENVIKING_TOOLS = [
    "openviking__remember",
    "openviking__search",
    "openviking__read",
    "openviking__find",
    "openviking__list",
    "openviking__add_resource",
    "openviking__health",
]


def wait_port(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(1)
    return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def choose_openclaw_port() -> int:
    for port in range(18802, 18880):
        if port_free(port):
            return port
    raise RuntimeError("no free OpenClaw port in 18802-18879")


def terminate_pid(pid: int, timeout: float = 20.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def stop_prior_openviking_eval_processes() -> None:
    for pid_path in sorted((BASE / "results").glob("ov_full_locomo10_*/openclaw-gateway.pid")):
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            continue
        terminate_pid(pid)

    pid_path = BASE / "openviking-server.pid"
    if pid_path.exists():
        try:
            terminate_pid(int(pid_path.read_text().strip()))
        except Exception:
            pass


def clean_openviking_workspace() -> None:
    target = WORKSPACE.resolve()
    expected = pathlib.Path("/root/openviking-locomo/openviking-workspace")
    if target != expected:
        raise RuntimeError(f"refusing to delete unexpected workspace: {target}")
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)


def load_setup_module() -> Any:
    spec = importlib.util.spec_from_file_location("ov_full_setup", BASE / "ov_full_setup.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ov_full_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_full_config() -> dict[str, Any]:
    setup = load_setup_module()
    return setup.write_full_config()


def start_openviking() -> int:
    logs = BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "openviking-server-full-locomo-runner.log"
    env = os.environ.copy()
    env["OPENVIKING_CONFIG_FILE"] = str(BASE / "ov.full.conf")
    env["OPENVIKING_PUBLIC_BASE_URL"] = f"http://127.0.0.1:{OPENVIKING_PORT}"
    env["NO_PROXY"] = LOCAL_NO_PROXY
    env["no_proxy"] = LOCAL_NO_PROXY
    proc = subprocess.Popen(
        [
            str(BASE / ".venv-openviking/bin/openviking-server"),
            "--config",
            str(BASE / "ov.full.conf"),
        ],
        stdout=open(log_path, "ab"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    (BASE / "openviking-server.pid").write_text(str(proc.pid), encoding="ascii")
    if not wait_port(OPENVIKING_PORT, 180):
        tail = log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else ""
        raise RuntimeError(f"OpenViking did not start on {OPENVIKING_PORT}\n{tail}")
    return proc.pid


def create_run(public_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    import secrets

    run_id = "ov_full_locomo10_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE / "results" / run_id
    home_dir = BASE / f"home-{run_id}"
    state_dir = home_dir / ".openclaw"
    workspace = state_dir / "workspace"
    port = choose_openclaw_port()
    account = "acct-openviking-" + run_id
    user = "user-openviking-" + run_id
    agent = "main"

    for path in [
        run_dir,
        run_dir / "logs",
        workspace,
        state_dir / "agents/main/agent",
        state_dir / "agents/main/sessions",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    global_cfg = json.load(open("/root/.openclaw/openclaw.json", encoding="utf-8-sig"))
    token = secrets.token_hex(24)
    config = {
        "agents": {
            "defaults": {
                "workspace": f"{home_dir}/.openclaw/workspace",
                "model": global_cfg["agents"]["defaults"]["model"],
                "models": global_cfg["agents"]["defaults"].get("models", {}),
            },
            "list": [{"id": agent}],
        },
        "gateway": {
            "mode": "local",
            "auth": {"mode": "token", "token": token},
            "port": port,
            "bind": "loopback",
            "tailscale": {"mode": "off", "resetOnExit": False},
            "http": {"endpoints": {"responses": {"enabled": True}}},
        },
        "session": global_cfg.get("session", {"dmScope": "per-channel-peer"}),
        "tools": {
            "profile": "minimal",
            "alsoAllow": OPENVIKING_TOOLS,
            "deny": ["session_status"],
        },
        "skills": global_cfg.get("skills", {}),
        "wizard": global_cfg.get("wizard", {}),
        "meta": global_cfg.get("meta", {}),
        "plugins": {"entries": {}, "slots": {}, "load": {"paths": []}},
        "auth": global_cfg.get("auth", {}),
        "models": global_cfg.get("models", {}),
        "mcp": {
            "servers": {
                "openviking": {
                    "url": f"http://127.0.0.1:{OPENVIKING_PORT}/mcp",
                    "transport": "streamable-http",
                    "headers": {
                        "X-OpenViking-Account": account,
                        "X-OpenViking-User": user,
                    },
                    "connectTimeout": 30,
                    "timeout": 300,
                    "supportsParallelToolCalls": False,
                    "toolFilter": {
                        "include": [
                            "find",
                            "search",
                            "read",
                            "list",
                            "remember",
                            "add_resource",
                            "health",
                        ]
                    },
                }
            }
        },
    }
    (state_dir / "openclaw.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "gateway-token.txt").write_text(token, encoding="ascii")

    src_dir = pathlib.Path("/root/.openclaw/agents/main/agent")
    dst_dir = state_dir / "agents/main/agent"
    for name in [
        "models.json",
        "openclaw-agent.sqlite",
        "openclaw-agent.sqlite-shm",
        "openclaw-agent.sqlite-wal",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)

    auth_count = 0
    db = dst_dir / "openclaw-agent.sqlite"
    if db.exists():
        con = sqlite3.connect(db, timeout=30)
        for table in [
            "cache_entries",
            "memory_embedding_cache",
            "memory_index_chunks",
            "memory_index_meta",
            "memory_index_sources",
            "memory_index_state",
        ]:
            try:
                con.execute(f"delete from {table}")
            except sqlite3.Error:
                pass
        con.commit()
        try:
            auth_count = con.execute("select count(*) from auth_profile_store").fetchone()[0]
        except sqlite3.Error:
            auth_count = 0
        con.close()

    meta = {
        "run_id": run_id,
        "linux_home": str(home_dir),
        "linux_run_dir": str(run_dir),
        "port": port,
        "account": account,
        "user": user,
        "agent_id": agent,
        "mcp_servers": list(config["mcp"]["servers"].keys()),
        "qa_enable_read": os.environ.get("OPENVIKING_QA_ENABLE_READ", "1") == "1",
        "qa_min_score": os.environ.get("OPENVIKING_QA_MIN_SCORE", ""),
        "qa_search_limit": 5,
        "qa_parallel": 2,
        "openviking_workspace_cleared_before_run": True,
        "ingest_input": str(DATASET),
        "qa_input": str(DATASET),
        "auth_profiles_copied": auth_count,
        "run_type": "openviking_full_locomo10_clean_optimized_mcp_only",
        "dataset_variant": "locomo10_small",
        "include_category5": os.environ.get("LOCOMO_INCLUDE_CATEGORY5") == "1",
        "optimized_config_public": public_config,
        "rerank_enabled": False,
        "metrics_to_report": [
            "accuracy",
            "input_tokens_total",
            "input_tokens_avg_per_qa",
            "qa_latency_avg_seconds",
        ],
        "openclaw_tools_profile": "minimal",
        "openclaw_tools_alsoAllow": OPENVIKING_TOOLS,
        "openclaw_tools_deny": ["session_status"],
        "strict_mcp_tool_surface_requested": True,
        "strict_mcp_tool_surface_note": (
            "OpenClaw core profile minimal; only OpenViking MCP tools added via tools.alsoAllow; "
            "session_status denied. OpenClaw default workspace context and configured skills are retained."
        ),
        "openclaw_default_workspace_context_retained": True,
        "openclaw_default_skills_prompt_retained": True,
    }
    write_json(run_dir / "run-meta.json", meta)
    (BASE / "current-full-run-id.txt").write_text(run_id + "\n", encoding="utf-8")
    (BASE / "current-full-run-dir.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    return meta, token


def start_openclaw(meta: dict[str, Any]) -> int:
    run_dir = pathlib.Path(meta["linux_run_dir"])
    home_dir = pathlib.Path(meta["linux_home"])
    log_path = run_dir / "logs" / "openclaw-gateway.log"
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["OPENCLAW_STATE_DIR"] = str(home_dir / ".openclaw")
    env["NO_PROXY"] = LOCAL_NO_PROXY
    env["no_proxy"] = LOCAL_NO_PROXY
    proc = subprocess.Popen(
        ["openclaw", "gateway", "--port", str(meta["port"])],
        stdout=open(log_path, "ab"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    (run_dir / "openclaw-gateway.pid").write_text(str(proc.pid), encoding="ascii")
    if not wait_port(int(meta["port"]), 180):
        tail = log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else ""
        raise RuntimeError(f"OpenClaw did not start on {meta['port']}\n{tail}")
    return proc.pid


def write_json(path: pathlib.Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_current_meta() -> tuple[dict[str, Any], str]:
    run_dir = pathlib.Path((BASE / "current-full-run-dir.txt").read_text().strip())
    meta = json.load(open(run_dir / "run-meta.json", encoding="utf-8"))
    token = (run_dir / "gateway-token.txt").read_text().strip()
    return meta, token


def setup_clean_run(_args: argparse.Namespace) -> None:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    stop_prior_openviking_eval_processes()
    clean_openviking_workspace()
    public = write_full_config()
    ov_pid = start_openviking()
    meta, token = create_run(public)
    oc_pid = start_openclaw(meta)
    meta["openviking_pid"] = ov_pid
    meta["openclaw_pid"] = oc_pid
    write_json(pathlib.Path(meta["linux_run_dir"]) / "run-meta.json", meta)
    print("RUN_ID=" + meta["run_id"])
    print("RUN_DIR=" + meta["linux_run_dir"])
    print("HOME=" + meta["linux_home"])
    print("PORT=" + str(meta["port"]))
    print("USER=" + meta["user"])
    print("AGENT=" + meta["agent_id"])
    print("OPENVIKING_PID=" + str(ov_pid))
    print("OPENCLAW_PID=" + str(oc_pid))
    print("TOKEN_LEN=" + str(len(token)))


def openviking_headers(meta: dict[str, Any]) -> dict[str, str]:
    return {
        "X-OpenViking-Account": meta["account"],
        "X-OpenViking-User": meta["user"],
    }


def list_openviking_tasks(meta: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
    resp = requests.get(
        f"http://127.0.0.1:{OPENVIKING_PORT}/api/v1/tasks",
        headers=openviking_headers(meta),
        params={"limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body.get("result") or []


def wait_for_openviking_idle(
    meta: dict[str, Any],
    expected_completed_at_least: int,
    timeout: float = 900.0,
    ignore_failed_task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    ignored = ignore_failed_task_ids or set()
    deadline = time.time() + timeout
    last_line = ""
    while time.time() < deadline:
        tasks = list_openviking_tasks(meta)
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.get("status", "?")] = counts.get(task.get("status", "?"), 0) + 1
        inflight = counts.get("pending", 0) + counts.get("running", 0)
        failed = counts.get("failed", 0)
        completed = counts.get("completed", 0)
        line = f"tasks completed={completed} running={counts.get('running', 0)} pending={counts.get('pending', 0)} failed={failed}"
        if line != last_line:
            print("    [ov] " + line, file=sys.stderr, flush=True)
            last_line = line
        failures = [
            t
            for t in tasks
            if t.get("status") == "failed" and t.get("task_id") not in ignored
        ]
        if failures:
            raise RuntimeError(f"OpenViking task failed: {failures[:3]}")
        ignored_failed_visible = sum(
            1
            for task in tasks
            if task.get("status") == "failed" and task.get("task_id") in ignored
        )
        visible_completed_capacity = max(0, len(tasks) - ignored_failed_visible)
        visible_expected = min(expected_completed_at_least, visible_completed_capacity)
        if inflight == 0 and completed >= visible_expected:
            return tasks
        time.sleep(5)
    raise TimeoutError(f"OpenViking tasks did not become idle within {timeout}s")


def clear_openclaw_sessions(meta: dict[str, Any]) -> None:
    sessions_file = pathlib.Path(meta["linux_home"]) / ".openclaw/agents/main/sessions/sessions.json"
    if sessions_file.exists():
        sessions_file.unlink()


def send_message_with_rate_backoff(
    base_url: str,
    token: str,
    user: str,
    message: str,
    agent_id: str,
    session_key: str,
    *,
    retries: int,
    base_sleep: float,
) -> tuple[str, dict[str, Any]]:
    for attempt in range(retries + 1):
        try:
            return locomo_eval.send_message(base_url, token, user, message, agent_id, session_key)
        except Exception as exc:
            text = str(exc)
            if "429" not in text or attempt >= retries:
                raise
            sleep_for = min(base_sleep * (2**attempt), 180.0)
            print(
                f"    [rate-limit] {text[:180]} ; sleeping {sleep_for:.0f}s before retry {attempt + 1}/{retries}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)
    raise RuntimeError("unreachable rate-limit retry state")


def build_neutral_ingest_input_message(conversation_text: str) -> str:
    return locomo_eval.build_ingest_input_message(conversation_text, "ingest-openviking")


def build_workload(dataset: pathlib.Path) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    samples = locomo_eval.load_locomo_data(str(dataset), None)
    workload = []
    for sample_idx, item in enumerate(samples):
        for sess in locomo_eval.build_session_messages(item, None):
            workload.append((sample_idx, item, sess))
    return workload


def load_ingest_overrides(run_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "ingest_overrides.json"
    if not path.exists():
        return {}
    return json.load(open(path, encoding="utf-8"))


def ingest_throttled(args: argparse.Namespace) -> None:
    meta, token = load_current_meta()
    run_dir = pathlib.Path(meta["linux_run_dir"])
    output_jsonl = run_dir / "ingest_throttled.jsonl"
    progress_path = run_dir / "ingest_throttled_progress.json"
    overrides = load_ingest_overrides(run_dir)
    workload = build_workload(pathlib.Path(args.dataset))
    done_keys: set[str] = set()
    if progress_path.exists() and not args.force:
        progress = json.load(open(progress_path, encoding="utf-8"))
        done_keys = set(progress.get("completed_keys", []))

    base_url = f"http://127.0.0.1:{meta['port']}"
    total = len(workload)
    completed = len(done_keys)
    ignored_failed_task_ids = {
        task.get("task_id", "")
        for task in list_openviking_tasks(meta)
        if task.get("status") == "failed"
    }
    started_at = time.perf_counter()
    clear_openclaw_sessions(meta)
    print(f"[ingest] total_sessions={total} already_completed={completed}", file=sys.stderr)

    for index, (_sample_idx, item, sess) in enumerate(workload, start=1):
        sample_id = item["sample_id"]
        session_key = sess["meta"]["session_key"]
        durable_key = f"{sample_id}:{session_key}"
        if durable_key in done_keys and not args.force:
            continue
        raw_msg = sess["message"]
        memory_session_id = f"ingest-{sample_id}"
        openclaw_session_key = f"{memory_session_id}-{session_key}"
        override = overrides.get(durable_key)
        if override:
            raw_msg = str(override["text"])
            openclaw_session_key = f"{openclaw_session_key}-override"
        print(f"[ingest] {index}/{total} {durable_key}", file=sys.stderr, flush=True)

        if override:
            attempts = [(raw_msg, openclaw_session_key, False, "override_summary")]
        else:
            attempts = [(raw_msg, openclaw_session_key, False, "default")]
            sanitized_msg, changed = locomo_eval.strip_image_urls_for_ingest_retry(raw_msg)
            if changed:
                attempts.append((sanitized_msg, openclaw_session_key + "-sanitized", True, "sanitized"))
            neutral_msg = sanitized_msg if changed else raw_msg
            attempts.append((neutral_msg, openclaw_session_key + "-neutral", changed, "neutral"))

        last_error = ""
        record = None
        for attempt_idx, (msg_text, attempt_session_key, sanitized, prompt_mode) in enumerate(attempts, start=1):
            msg = locomo_eval.build_ingest_input_message(msg_text, memory_session_id)
            try:
                reply, usage = send_message_with_rate_backoff(
                    base_url,
                    token,
                    meta["user"],
                    msg,
                    meta["agent_id"],
                    attempt_session_key,
                    retries=args.rate_retries,
                    base_sleep=args.rate_sleep,
                )
                if not locomo_eval.is_successful_ingest_reply(reply):
                    raise RuntimeError(f"Ingest did not confirm MCP write: {reply[:240]}")
                next_completed = completed + 1
                tasks = wait_for_openviking_idle(
                    meta,
                    next_completed,
                    timeout=args.task_timeout,
                    ignore_failed_task_ids=ignored_failed_task_ids,
                )
                completed = next_completed
                done_keys.add(durable_key)
                record = {
                    "sample_id": sample_id,
                    "session": session_key,
                    "key": durable_key,
                    "index": index,
                    "reply": reply,
                    "usage": usage,
                    "sanitized_retry": sanitized,
                    "prompt_mode": prompt_mode,
                    "task_count": len(tasks),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                write_json(
                    progress_path,
                    {
                        "total": total,
                        "completed": completed,
                        "completed_keys": sorted(done_keys),
                        "last_key": durable_key,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                break
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"    [attempt {attempt_idx}/{len(attempts)}] {last_error[:400]}",
                    file=sys.stderr,
                    flush=True,
                )
                for task in list_openviking_tasks(meta):
                    if task.get("status") == "failed":
                        ignored_failed_task_ids.add(task.get("task_id", ""))
                if attempt_idx < len(attempts):
                    clear_openclaw_sessions(meta)
                    continue
                raise
        if record is None:
            raise RuntimeError(f"failed ingest for {durable_key}: {last_error}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    elapsed = time.perf_counter() - started_at
    tasks = wait_for_openviking_idle(
        meta,
        total,
        timeout=args.task_timeout,
        ignore_failed_task_ids=ignored_failed_task_ids,
    )
    summary = {
        "total_sessions": total,
        "completed_sessions": len(done_keys),
        "openviking_task_count": len(tasks),
        "elapsed_seconds": elapsed,
        "output_jsonl": str(output_jsonl),
    }
    write_json(run_dir / "ingest_throttled_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def health_probe(_args: argparse.Namespace) -> None:
    meta, token = load_current_meta()
    base_url = f"http://127.0.0.1:{meta['port']}"
    response, usage = locomo_eval.send_message(
        base_url,
        token,
        meta["user"],
        "Call openviking__health exactly once. After it returns, answer only: health ok",
        meta["agent_id"],
        "surface-check",
    )
    run_dir = pathlib.Path(meta["linux_run_dir"])
    meta["strict_mcp_tool_surface_verified"] = True
    meta["strict_mcp_tool_surface_verified_tools"] = OPENVIKING_TOOLS
    meta["surface_check_response"] = response
    meta["surface_check_usage"] = usage
    write_json(run_dir / "run-meta.json", meta)
    print(response)
    print(json.dumps(usage, ensure_ascii=False))


def run_qa(args: argparse.Namespace) -> None:
    meta, token = load_current_meta()
    run_dir = pathlib.Path(meta["linux_run_dir"])
    env = os.environ.copy()
    env["HOME"] = meta["linux_home"]
    env["OPENCLAW_GATEWAY_TOKEN"] = token
    env["OPENVIKING_QA_ENABLE_READ"] = os.environ.get("OPENVIKING_QA_ENABLE_READ", "1")
    env["OPENVIKING_QA_MIN_SCORE"] = os.environ.get("OPENVIKING_QA_MIN_SCORE", "")
    env["NO_PROXY"] = LOCAL_NO_PROXY
    env["no_proxy"] = LOCAL_NO_PROXY
    cmd = [
        str(PYTHON),
        "eval_openviking.py",
        "--base-url",
        f"http://127.0.0.1:{meta['port']}",
        "--token",
        token,
        "--user",
        meta["user"],
        "--agent-id",
        meta["agent_id"],
        "--output",
        str(run_dir / "qa"),
        "-p",
        str(args.parallel),
        "qa",
        str(args.dataset),
    ]
    proc = subprocess.run(cmd, cwd="/mnt/d/code/oGMemory-dev/tests/e2e", env=env, text=True)
    raise SystemExit(proc.returncode)


def wait_then_ingest(args: argparse.Namespace) -> None:
    target = datetime.fromisoformat(args.until)
    delay = max(0.0, target.timestamp() - time.time())
    print(f"sleeping_until={target.isoformat()} seconds={delay:.0f}", flush=True)
    if delay > 0:
        time.sleep(delay)
    ingest_throttled(args)


def run_judge(args: argparse.Namespace) -> None:
    meta, _token = load_current_meta()
    run_dir = pathlib.Path(meta["linux_run_dir"])
    config = json.load(open(BASE / "ov.full.conf", encoding="utf-8-sig"))
    env = os.environ.copy()
    env["OGMEM_API_KEY"] = config["vlm"]["api_key"]
    cmd = [
        str(PYTHON),
        "eval_openviking.py",
        "judge",
        str(run_dir / "qa.csv"),
        "--output",
        str(run_dir / "qa_judged.csv"),
    ]
    proc = subprocess.run(cmd, cwd="/mnt/d/code/oGMemory-dev/tests/e2e", env=env, text=True)
    raise SystemExit(proc.returncode)


def summarize_metrics(args: argparse.Namespace) -> None:
    meta, _token = load_current_meta()
    run_dir = pathlib.Path(meta["linux_run_dir"])
    csv_path = pathlib.Path(args.csv) if args.csv else run_dir / "qa_judged.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig", newline="")))
    total = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct") == "CORRECT")
    wrong = sum(1 for row in rows if row.get("is_correct") == "WRONG")
    input_tokens = [int(row.get("input_tokens") or 0) for row in rows]
    qa_elapsed_ms = [int(row.get("qa_elapsed_ms") or 0) for row in rows]

    def pct(values: list[int], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[pos])

    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        cat = row.get("category", "?")
        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["total"] += 1
        if row.get("is_correct") == "CORRECT":
            by_category[cat]["correct"] += 1

    metrics = {
        "run_id": meta["run_id"],
        "csv": str(csv_path),
        "total_questions": total,
        "judged": correct + wrong,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else None,
        "input_tokens_total": sum(input_tokens),
        "input_tokens_avg_per_qa": (sum(input_tokens) / total) if total else None,
        "qa_latency_avg_seconds": (sum(qa_elapsed_ms) / total / 1000.0) if total else None,
        "qa_latency_p50_seconds": pct(qa_elapsed_ms, 0.50) / 1000.0,
        "qa_latency_p95_seconds": pct(qa_elapsed_ms, 0.95) / 1000.0,
        "by_category": {
            cat: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else None,
            }
            for cat, stats in sorted(by_category.items())
        },
    }
    write_json(run_dir / "final_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup-clean-run")
    setup.set_defaults(func=setup_clean_run)

    health = sub.add_parser("health-probe")
    health.set_defaults(func=health_probe)

    ingest = sub.add_parser("ingest-throttled")
    ingest.add_argument("--dataset", default=str(DATASET))
    ingest.add_argument("--task-timeout", type=float, default=900.0)
    ingest.add_argument("--sleep", type=float, default=0.0)
    ingest.add_argument("--rate-retries", type=int, default=8)
    ingest.add_argument("--rate-sleep", type=float, default=30.0)
    ingest.add_argument("--force", action="store_true")
    ingest.set_defaults(func=ingest_throttled)

    wait_ingest = sub.add_parser("wait-ingest")
    wait_ingest.add_argument("--until", required=True)
    wait_ingest.add_argument("--dataset", default=str(DATASET))
    wait_ingest.add_argument("--task-timeout", type=float, default=900.0)
    wait_ingest.add_argument("--sleep", type=float, default=0.0)
    wait_ingest.add_argument("--rate-retries", type=int, default=8)
    wait_ingest.add_argument("--rate-sleep", type=float, default=30.0)
    wait_ingest.add_argument("--force", action="store_true")
    wait_ingest.set_defaults(func=wait_then_ingest)

    qa = sub.add_parser("qa")
    qa.add_argument("--dataset", default=str(DATASET))
    qa.add_argument("-p", "--parallel", type=int, default=2)
    qa.set_defaults(func=run_qa)

    judge = sub.add_parser("judge")
    judge.set_defaults(func=run_judge)

    metrics = sub.add_parser("metrics")
    metrics.add_argument("--csv", default=None)
    metrics.set_defaults(func=summarize_metrics)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
