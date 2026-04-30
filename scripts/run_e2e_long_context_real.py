#!/usr/bin/env python3
"""Run a real-service (non-mock) long-context E2E for AGFS + OpenClaw bridge.

Flow:
1) Verify AGFS is reachable
2) Verify OpenAI credentials (unless --allow-mock-llm)
3) Drive the six lifecycle stages (bootstrap -> after_turn[*] -> compose ->
   compact -> dispose) through the OpenClaw bridge
4) Render the perf recorder's Markdown report when --record-perf is set
5) Print an execution summary

Fixture shapes supported:
    - Flat chat: ``[{"role": ..., "content": ...}, ...]`` — fed directly.
    - LoCoMo:    ``[{"conversation": {...}, "qa": [...], ...}, ...]`` — converted
      via :func:`_locomo_to_messages`, one ``user`` message per session (matching
      the ``tests/e2e/eval.py`` convention).  ``created_at`` is populated from
      ``session_N_date_time`` so the extractor grounds relative time references
      against the real LoCoMo timestamps.

Notes:
- This script exercises the real OpenClaw plugin bridge implementation in
  ``openclaw_context_engine_plugin/bridge/memory_api.py``.
- It does NOT require running the OpenClaw daemon; it invokes bridge lifecycle
  functions directly in-process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


# ---------------------------------------------------------------------------
# Bridge helper
# ---------------------------------------------------------------------------

def _load_bridge_module(repo_root: Path):
    """Import bridge module from local plugin path."""
    plugin_dir = repo_root / "openclaw_context_engine_plugin"
    bridge_file = plugin_dir / "bridge" / "memory_api.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("memory_api", bridge_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["memory_api"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _check_agfs(agfs_base_url: str) -> None:
    from pyagfs import AGFSClient

    client = AGFSClient(api_base_url=agfs_base_url)
    client.ls("/")


def _check_openclaw_cli() -> dict:
    """Best-effort check for OpenClaw CLI availability."""
    import shutil
    import subprocess

    result = {
        "openclaw_cli_found": False,
        "openclaw_version": None,
        "openclaw_doctor": None,
    }

    binary = shutil.which("openclaw")
    if not binary:
        return result

    result["openclaw_cli_found"] = True

    try:
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["openclaw_version"] = (version.stdout or version.stderr).strip()
    except Exception as exc:
        result["openclaw_version"] = f"error: {exc}"

    try:
        doctor = subprocess.run(
            [binary, "doctor"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        result["openclaw_doctor"] = (doctor.stdout or doctor.stderr).strip()
    except Exception as exc:
        result["openclaw_doctor"] = f"error: {exc}"

    return result


def _restart_ogmem_with_perf(container_name: str, perf_http_url: str) -> dict:
    """Stop, reconfigure, and restart the ogmem container with perf env vars.

    Returns a dict with container_name, ogmem_url on success, or error info.
    """
    import subprocess

    try:
        inspect_result = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        import json
        containers = json.loads(inspect_result.stdout)
        if not containers:
            return {"error": f"container {container_name} not found"}
        c = containers[0]
        if c["State"]["Status"] != "running":
            return {"error": f"container {container_name} is not running"}

        image = c["Config"]["Image"]
        network_mode = c["HostConfig"].get("NetworkMode", "bridge")
        port_bindings = c["HostConfig"].get("PortBindings") or {}
        binds = c["HostConfig"].get("Binds") or []
        env_list = c["Config"]["Env"] or []

        current_env = {}
        for e in env_list:
            if "=" in e:
                k, v = e.split("=", 1)
                current_env[k] = v

        perf_env = {
            "OGMEM_PERF_ENABLED": "1",
            "OGMEM_PERF_HTTP_URL": perf_http_url,
            "OGMEM_PERF_OUT": "/tmp/ogmem_perf.jsonl",
        }
        new_env = {**current_env, **perf_env}

        subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=30, check=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True, timeout=15, check=True)

        run_cmd = ["docker", "run", "-d", "--name", container_name, "--network", network_mode]
        for pb in port_bindings:
            host_port = port_bindings[pb][0]["HostPort"]
            run_cmd += ["-p", f"{host_port}:{pb.replace("/tcp", "")}"]
        for bind in binds:
            run_cmd += ["-v", bind]
        for k, v in new_env.items():
            run_cmd += ["-e", f"{k}={v}"]
        run_cmd.append(image)

        subprocess.run(run_cmd, capture_output=True, text=True, timeout=60, check=True)

        import time
        time.sleep(20)

        health = _http_get(f"http://127.0.0.1:8090", "health")
        if health.get("_http_call_failed"):
            return {"error": f"container restarted but health check failed: {health}"}

        return {"ok": True, "container_name": container_name}
    except subprocess.CalledProcessError as exc:
        return {"error": f"docker command failed: {exc.stderr}"}
    except Exception as exc:
        return {"error": str(exc)}


def _detect_docker_ogmem() -> str | None:
    """Check if the ogmem container is running in Docker.

    Returns the HTTP base URL of the container if found, else None.
    Matches containers whose name starts with "ogmem" but NOT "openclaw".
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None

        ogmem_containers = []
        for name in result.stdout.strip().split("\n"):
            if not name:
                continue
            if name.startswith("ogmem") and not name.startswith("openclaw"):
                ogmem_containers.append(name.strip())

        if not ogmem_containers:
            return None

        container_name = ogmem_containers[0]

        port_result = subprocess.run(
            ["docker", "inspect", container_name, "--format", "{{.NetworkSettings.Ports}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        ports_str = port_result.stdout.strip()
        port = "8090"
        if ports_str:
            import re
            m = re.search(r'8090/tcp -> (\d+)/tcp', ports_str)
            if m:
                port = m.group(1)
        return f"http://127.0.0.1:{port}"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def _http_call(base_url: str, method: str, params: dict) -> dict:
    """Make an HTTP POST to the oG-Memory API."""
    import urllib.request

    url = f"{base_url}/api/v1/{method}"
    payload = json.dumps(params, default=str).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc), "_http_call_failed": True}


def _http_get(base_url: str, path: str) -> dict:
    """Make an HTTP GET to the oG-Memory API."""
    import urllib.request

    url = f"{base_url}/api/v1/{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc), "_http_call_failed": True}


# ---------------------------------------------------------------------------
# Fixture shape detection + LoCoMo adapter
# ---------------------------------------------------------------------------

def _is_chat_payload(obj: Any) -> bool:
    """Return True if ``obj`` is a flat ``[{role, content}, ...]`` list."""
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def _is_locomo_payload(obj: Any) -> bool:
    """Return True if ``obj`` is a LoCoMo-style list of samples.

    Each sample must have a ``conversation`` dict — that is the minimal
    identifier used by the upstream LoCoMo dataset and by
    ``tests/e2e/eval.py``.
    """
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    return isinstance(first, dict) and isinstance(first.get("conversation"), dict)


def _parse_session_range(spec: str | None) -> tuple[int, int] | None:
    """Parse ``"A:B"`` into ``(A, B)`` inclusive. ``None`` means no restriction."""
    if not spec:
        return None
    try:
        lo_str, hi_str = spec.split(":", 1)
        lo = int(lo_str)
        hi = int(hi_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid --session-range {spec!r}: expected 'A:B'") from exc
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid --session-range {spec!r}: must satisfy 1 <= lo <= hi")
    return (lo, hi)


def _locomo_to_messages(
    sample: dict,
    *,
    session_range: tuple[int, int] | None = None,
    max_sessions: int = 0,
    ack_text: str = "Acknowledged.",
) -> tuple[list[dict], list[dict]]:
    """Convert a single LoCoMo sample into a flat chat payload.

    Each session is concatenated into a single ``user`` message (preserving
    the ``[group chat conversation: <date>]`` header + per-turn
    ``Speaker: text`` lines). A minimal ``assistant`` acknowledgement is
    interleaved so the extractor's user/assistant filter treats the trace
    normally.

    Args:
        sample: one entry from ``tests/e2e/data/locomo10.json``.
        session_range: optional ``(lo, hi)`` inclusive, matching
            ``session_N`` numbering used by ``tests/e2e/eval.py``.
        max_sessions: hard cap on the number of sessions (0 = no cap).
        ack_text: body of the synthesized assistant acknowledgement.

    Returns:
        A tuple ``(messages, session_metas)`` where ``messages`` is the flat
        chat list ready for ``bridge.after_turn`` and ``session_metas`` is a
        list of per-session metadata dicts (``session_key``, ``date_time``,
        ``created_at``, ``lines``) useful for driver-level logging and tests.
    """
    # Re-use the existing adapter so the rendering matches tests/e2e/eval.py.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.e2e.eval import (  # local import — avoids paying the cost on non-LoCoMo paths
        build_session_messages,
        parse_locomo_datetime,
    )

    bundles = build_session_messages(sample, session_range=session_range, tail="")
    if max_sessions and max_sessions > 0:
        bundles = bundles[:max_sessions]

    messages: list[dict] = []
    metas: list[dict] = []
    for bundle in bundles:
        meta = dict(bundle.get("meta") or {})
        created_at = None
        dt = parse_locomo_datetime(meta.get("date_time", ""))
        if dt is not None:
            # Preserve minute granularity; extractor only uses date portion.
            created_at = dt.isoformat()

        content = bundle["message"]
        user_msg = {"role": "user", "content": content}
        if created_at:
            user_msg["created_at"] = created_at

        messages.append(user_msg)
        messages.append({"role": "assistant", "content": ack_text})

        metas.append(
            {
                "session_key": meta.get("session_key"),
                "date_time": meta.get("date_time"),
                "created_at": created_at,
                "lines": len(content.splitlines()),
                "chars": len(content),
            }
        )

    return messages, metas


def _select_compose_query(
    sample: dict | None,
    mode: str,
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """Pick the prompt to feed ``compose``.

    Returns ``(query, tail_messages)``. ``tail_messages`` is used when
    ``mode == "tail"`` to preserve the pre-existing behaviour of sending the
    last 12 turns to ``compose``.
    """
    if mode == "first_qa" and sample and isinstance(sample.get("qa"), list):
        for qa in sample["qa"]:
            q = (qa or {}).get("question")
            if isinstance(q, str) and q.strip():
                return q.strip(), []
    if mode == "last_user":
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"], []
    # Fallback: tail
    tail = messages[-12:] if messages else []
    return "", tail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run real long-context E2E via AGFS + OpenClaw bridge")
    parser.add_argument(
        "--fixture",
        default="tests/e2e/data/locomo10.json",
        help="Path to fixture JSON (flat chat list OR LoCoMo sample list)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="LoCoMo sample index to use when the fixture is LoCoMo-shaped (default: 0)",
    )
    parser.add_argument(
        "--session-range",
        default=None,
        help="Restrict to LoCoMo sessions A:B inclusive (e.g. '1:5')",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Hard cap on sessions processed (0 = no cap)",
    )
    parser.add_argument(
        "--compose-query",
        choices=("first_qa", "last_user", "tail"),
        default="first_qa",
        help="Query source for the compose stage (default: first_qa)",
    )
    parser.add_argument(
        "--agfs-base-url",
        default=os.environ.get("AGFS_BASE_URL", "http://localhost:1833"),
        help="AGFS base URL",
    )
    parser.add_argument(
        "--session-id",
        default=f"e2e-real-{uuid4()}",
        help="Session id for this test run",
    )
    parser.add_argument(
        "--allow-mock-llm",
        action="store_true",
        help="Allow fallback to mock provider if OGMEM_API_KEY is not configured",
    )
    parser.add_argument(
        "--record-perf",
        action="store_true",
        help="Enable the perf.recorder to attribute time/tokens/$ to each lifecycle stage",
    )
    parser.add_argument(
        "--perf-out",
        default=None,
        help="Path to write perf JSONL events (default: perf_logs/{run_id}.jsonl)",
    )
    parser.add_argument(
        "--perf-report",
        default=None,
        help="Path to write the rendered Markdown perf report (default: perf_logs/{run_id}.md)",
    )
    parser.add_argument(
        "--perf-rate-card",
        default=None,
        help="Override rate card YAML (default: perf/rate_cards.yaml)",
    )
    parser.add_argument(
        "--docker-mode",
        action="store_true",
        help="Detect ogmem container running in Docker and call its HTTP API instead of loading the bridge module",
    )
    parser.add_argument(
        "--ogmem-url",
        default=None,
        help="oG-Memory HTTP API base URL (default: auto-detect from Docker or http://localhost:8090)",
    )
    parser.add_argument(
        "--perf-remote-sink",
        action="store_true",
        help="In docker-mode: restart ogmem container with OGMEM_PERF_ENABLED so it streams events to --ogmem-url",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    fixture_path = (repo_root / args.fixture).resolve()

    if not fixture_path.exists():
        print(json.dumps({"status": "error", "error": f"fixture not found: {fixture_path}"}, ensure_ascii=False))
        return 2

    if not args.allow_mock_llm and not os.environ.get("OGMEM_API_KEY"):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "OGMEM_API_KEY is required for non-mock full-chain run (or pass --allow-mock-llm)",
                },
                ensure_ascii=False,
            )
        )
        return 2

    # Force mock provider when --allow-mock-llm is set, regardless of API key presence
    if args.allow_mock_llm:
        os.environ["CONTEXTENGINE_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
    elif os.environ.get("OGMEM_API_KEY"):
        os.environ["CONTEXTENGINE_PROVIDER"] = "openai"

    os.environ["AGFS_BASE_URL"] = args.agfs_base_url

    # ------------------------------------------------------------------
    # Docker mode detection + ogmem URL resolution
    # ------------------------------------------------------------------
    docker_ogmem_url: str | None = None
    use_docker_mode = args.docker_mode

    if use_docker_mode or args.ogmem_url:
        if args.ogmem_url:
            docker_ogmem_url = args.ogmem_url.rstrip("/")
        else:
            docker_ogmem_url = _detect_docker_ogmem()

        if not docker_ogmem_url:
            print(
                json.dumps(
                    {"status": "error", "error": "docker mode requested but ogmem container not found or not running"},
                    ensure_ascii=False,
                )
            )
            return 2
        os.environ["OGMEM_BASE_URL"] = docker_ogmem_url

        if args.record_perf and args.perf_remote_sink:
            perf_http_url = f"{docker_ogmem_url}/api/v1/perf/events"
            print(
                json.dumps(
                    {"status": "note", "message": f"perf remote mode: restarting ogmem_auto with perf env vars, sink={perf_http_url}"},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            restart_result = _restart_ogmem_with_perf("ogmem_auto", perf_http_url)
            if not restart_result.get("ok"):
                print(
                    json.dumps(
                        {"status": "error", "error": f"failed to restart ogmem container with perf: {restart_result.get('error')}"},
                        ensure_ascii=False,
                    )
                )
                return 2
    else:
        detected = _detect_docker_ogmem()
        if detected:
            print(
                json.dumps(
                    {
                        "status": "note",
                        "message": f"ogmem container detected at {detected}; use --docker-mode to use HTTP API",
                        "detected_url": detected,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )

    try:
        _check_agfs(args.agfs_base_url)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"AGFS unreachable at {args.agfs_base_url}", "detail": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2

    # ------------------------------------------------------------------
    # Load fixture + adapt shape
    # ------------------------------------------------------------------
    raw_payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    try:
        session_range = _parse_session_range(args.session_range)
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2

    fixture_info: dict[str, Any] = {}
    session_metas: list[dict] = []
    selected_sample: dict | None = None

    if _is_locomo_payload(raw_payload):
        if args.sample < 0 or args.sample >= len(raw_payload):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": f"--sample {args.sample} out of range (0-{len(raw_payload) - 1})",
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        selected_sample = raw_payload[args.sample]
        messages, session_metas = _locomo_to_messages(
            selected_sample,
            session_range=session_range,
            max_sessions=args.max_sessions,
        )
        fixture_info = {
            "format": "locomo",
            "sample_index": args.sample,
            "sample_id": selected_sample.get("sample_id"),
            "qa_count": len(selected_sample.get("qa", []) or []),
            "session_count": len(session_metas),
            "session_range": args.session_range,
            "max_sessions": args.max_sessions,
        }
    elif _is_chat_payload(raw_payload):
        messages = raw_payload
        # Chat fixtures aren't session-structured; cap via max_sessions is a no-op.
        fixture_info = {"format": "chat", "sample_index": None, "session_count": None}
    else:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": (
                        "unrecognized fixture shape — expected either a flat "
                        "[{role, content}, ...] list or a LoCoMo list whose "
                        "entries carry a 'conversation' dict"
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 2

    # ------------------------------------------------------------------
    # Perf recorder wiring (opt-in, remote mode)
    # ------------------------------------------------------------------
    perf_info: dict | None = None
    _perf_remote_mode = bool(args.record_perf and args.perf_remote_sink and docker_ogmem_url)

    if args.record_perf:
        os.environ["OGMEM_PERF_ENABLED"] = "1"
        from perf import get_recorder
        from perf.sinks import JsonlSink, CompositeSink, HttpSink

        run_id = args.session_id
        out_path = args.perf_out or str(repo_root / "perf_logs" / f"{run_id}.jsonl")
        report_path = args.perf_report or str(repo_root / "perf_logs" / f"{run_id}.md")

        if _perf_remote_mode:
            perf_info = {
                "enabled": True,
                "out": out_path,
                "report": report_path,
                "run_id": run_id,
                "docker_mode": True,
                "remote_sink": True,
            }
        else:
            sinks: list[Any] = [JsonlSink(out_path)]
            if docker_ogmem_url:
                perf_http_url = f"{docker_ogmem_url}/api/v1/perf/events"
                sinks.append(HttpSink(perf_http_url))
                os.environ["OGMEM_PERF_HTTP_URL"] = perf_http_url

            sink: Any = CompositeSink(sinks) if len(sinks) > 1 else sinks[0]
            get_recorder().enable(
                sink=sink,
                run_id=run_id,
                rate_card_path=args.perf_rate_card,
            )
            perf_info = {"enabled": True, "out": out_path, "report": report_path, "run_id": run_id}
            if docker_ogmem_url:
                perf_info["docker_mode"] = True
                perf_info["perf_http_url"] = f"{docker_ogmem_url}/api/v1/perf/events"

    bridge = None
    if docker_ogmem_url:
        bridge = None
    else:
        bridge = _load_bridge_module(repo_root)

    # Attach the recorder to the just-created service, if enabled.
    if args.record_perf and bridge is not None and not _perf_remote_mode:
        from perf import get_recorder
        svc = bridge._get_service()
        get_recorder().attach(svc)

    # Optional visibility into OpenClaw CLI status
    openclaw_info = _check_openclaw_cli()

    # ------------------------------------------------------------------
    # Drive the six lifecycle stages
    # ------------------------------------------------------------------
    if docker_ogmem_url:
        bootstrap_result = _http_call(docker_ogmem_url, "bootstrap", {"sessionId": args.session_id})
    else:
        bootstrap_result = bridge.bootstrap({"sessionId": args.session_id})

    # For LoCoMo we feed one session per after_turn call so the perf report
    # captures per-session ingestion cost; a flat chat fixture is still sent
    # as a single batch (backward compatibility).
    after_turn_results: list[dict] = []
    if fixture_info["format"] == "locomo":
        pair_count = 2  # user + assistant
        for i, meta in enumerate(session_metas):
            start = i * pair_count
            session_msgs = messages[start : start + pair_count]
            params = {
                "sessionId": args.session_id,
                "messages": session_msgs,
                "sessionMeta": meta,
            }
            if docker_ogmem_url:
                res = _http_call(docker_ogmem_url, "after_turn", params)
            else:
                res = bridge.after_turn(params)
            after_turn_results.append(res)
    else:
        params = {"sessionId": args.session_id, "messages": messages}
        if docker_ogmem_url:
            after_turn_results.append(_http_call(docker_ogmem_url, "after_turn", params))
        else:
            after_turn_results.append(bridge.after_turn(params))

    # --- compose ---
    compose_query, compose_tail = _select_compose_query(
        selected_sample, args.compose_query, messages,
    )
    compose_params: dict[str, Any] = {
        "sessionId": args.session_id,
        "tokenBudget": 128_000,
    }
    if compose_query:
        compose_params["prompt"] = compose_query
        compose_params["messages"] = []
    else:
        compose_params["messages"] = compose_tail

    if docker_ogmem_url:
        assemble_result = _http_call(docker_ogmem_url, "compose", compose_params)
    else:
        assemble_result = bridge.compose(compose_params)

    # Exercise the remaining lifecycle stages so the perf report covers all 6.
    compact_params = {"sessionId": args.session_id, "tokenBudget": 128_000}
    if docker_ogmem_url:
        compact_result = _http_call(docker_ogmem_url, "compact", compact_params)
        dispose_result = _http_call(docker_ogmem_url, "dispose", {"sessionId": args.session_id})
        health_result = _http_get(docker_ogmem_url, "health")  # /api/v1/health
    else:
        compact_result = bridge.compact(compact_params)
        dispose_result = bridge.dispose({"sessionId": args.session_id})
        health_result = bridge.memory_health()

    summary: dict[str, Any] = {
        "status": "ok",
        "mode": "docker-http" if docker_ogmem_url else ("real" if os.environ.get("CONTEXTENGINE_PROVIDER") == "openai" else "mock-allowed"),
        "docker_mode": docker_ogmem_url is not None,
        "ogmem_url": docker_ogmem_url,
        "session_id": args.session_id,
        "fixture": str(fixture_path.relative_to(repo_root)),
        "fixture_info": fixture_info,
        "message_count": len(messages),
        "compose_query": {"mode": args.compose_query, "text": compose_query[:200] if compose_query else None},
        "bootstrap": bootstrap_result,
        "after_turn_calls": len(after_turn_results),
        "after_turn_last": after_turn_results[-1] if after_turn_results else None,
        "assemble_estimated_tokens": assemble_result.get("estimatedTokens"),
        "assemble_system_prompt_preview": (assemble_result.get("systemPromptAddition") or "")[:300],
        "compact": compact_result,
        "dispose": dispose_result,
        "memory_health": health_result,
        "openclaw_cli": openclaw_info,
    }

    # Render the Markdown report after the trace file is flushed.
    if args.record_perf and perf_info:
        from perf import get_recorder
        from perf.report import load_events, render

        get_recorder().disable()

        if _perf_remote_mode:
            import subprocess
            try:
                copy_result = subprocess.run(
                    ["docker", "cp", "ogmem_auto:/tmp/ogmem_perf.jsonl", perf_info["out"]],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if copy_result.returncode == 0 and Path(perf_info["out"]).exists():
                    events = load_events(perf_info["out"])
                    md = render(events)
                    Path(perf_info["report"]).parent.mkdir(parents=True, exist_ok=True)
                    Path(perf_info["report"]).write_text(md, encoding="utf-8")
                    perf_info["events"] = len(events)
                    summary["perf"] = perf_info
                else:
                    summary["perf"] = {**perf_info, "copy_error": copy_result.stderr or "copy failed", "report_note": "remote mode - events not yet available"}
            except Exception as exc:
                summary["perf"] = {**perf_info, "report_note": f"remote mode - could not fetch events: {exc}"}
        else:
            try:
                events = load_events(perf_info["out"])
                md = render(events)
                Path(perf_info["report"]).parent.mkdir(parents=True, exist_ok=True)
                Path(perf_info["report"]).write_text(md, encoding="utf-8")
                perf_info["events"] = len(events)
                summary["perf"] = perf_info
            except FileNotFoundError:
                summary["perf"] = {**perf_info, "report_note": "no local JSONL"}
            except Exception as exc:
                summary["perf"] = {**perf_info, "report_error": str(exc)}

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())