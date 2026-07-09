#!/usr/bin/env python3
"""LoCoMo perf concurrency test.

直连 ogmem native API（默认 http://127.0.0.1:8091/api/v1）跑 LoCoMo conversation：
- 每个 session: ingest 消息 → commit (等待) → compose 一次
- 支持 --workers N 并发跑 N 个 session
- 主要目的: 触发 perf 模块的 compose / after_turn / extract / commit 等阶段事件
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


DEFAULT_BASE = os.environ.get("OGMEM_BASE_URL_TEST", "http://127.0.0.1:8091/api/v1")
DEFAULT_DATA = str(Path(__file__).parent / "data" / "locomo10.json")


def http_post(base: str, path: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(base: str, path: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(f"{base}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_sessions(path: str, sample_idx: int, max_sessions: int, max_turns_per_session: int) -> list[tuple[str, list[tuple[str, str]]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    sample = data[sample_idx]
    conv = sample.get("conversation", {})
    keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    sessions: list[tuple[str, list[tuple[str, str]]]] = []
    for k in keys[:max_sessions]:
        turns: list[tuple[str, str]] = []
        for turn in conv[k][:max_turns_per_session]:
            text = turn.get("text", "").strip()
            if not text:
                continue
            speaker = (turn.get("speaker") or "").lower()
            role = "assistant" if speaker in ("assistant", "ai", "bot", "system") else "user"
            turns.append((role, text))
        if turns:
            sessions.append((k, turns))
    return sessions


def run_session(
    base: str,
    common: dict,
    session_id: str,
    turns: list[tuple[str, str]],
    do_compose: bool,
) -> dict:
    """Drive after_turn + compose to exercise perf instrumentation.

    after_turn 触发 perf stage="after_turn" + 后台 stage="extract"。
    compose 触发 perf stage="compose"。
    """
    stats = {
        "session_id": session_id,
        "turns": len(turns),
        "after_turn_ms": 0.0,
        "compose_ms": 0.0,
        "errors": [],
    }

    # Build the messages list for after_turn (simulate full conversation in one shot).
    # Each role/content pair becomes a message dict.
    messages_payload = [
        {"role": role, "content": content[:4000]}
        for role, content in turns
    ]

    t_at = time.perf_counter()
    try:
        result = http_post(
            base,
            "/after_turn",
            {
                **common,
                "sessionId": session_id,
                "messages": messages_payload,
            },
            timeout=600,
        )
        stats["after_turn_result"] = {
            k: result.get(k) for k in ("ok", "status", "extracted", "task_id")
        }
        if result.get("ok") is False:
            stats["errors"].append(f"after_turn:{result}")
    except Exception as exc:
        stats["errors"].append(f"after_turn_exc:{exc}")
    stats["after_turn_ms"] = (time.perf_counter() - t_at) * 1000

    if do_compose:
        t_p = time.perf_counter()
        try:
            compose = http_post(
                base,
                "/compose",
                {
                    **common,
                    "sessionId": session_id,
                    "messages": [
                        {"role": "user", "content": "Summarize what you know about me."}
                    ],
                    "tokenBudget": 4096,
                },
                timeout=180,
            )
            stats["compose_assembled_tokens"] = (
                compose.get("usage", {}).get("total_tokens")
                or compose.get("token_usage", {}).get("total")
                or compose.get("token_count")
            )
        except Exception as exc:
            stats["errors"].append(f"compose_exc:{exc}")
        stats["compose_ms"] = (time.perf_counter() - t_p) * 1000

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=4, help="number of sessions to ingest")
    parser.add_argument("--turns", type=int, default=12, help="max turns per session")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--account", default="acct-perfconc")
    parser.add_argument("--user", default="u-perfconc")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--prefix", default="perf-conc", help="session id prefix")
    parser.add_argument("--no-compose", action="store_true")
    parser.add_argument("--marker", default=None, help="optional run marker for output")
    args = parser.parse_args()

    sessions = load_sessions(args.data, args.sample, args.sessions, args.turns)
    print(f"[INFO] loaded {len(sessions)} sessions, total turns = {sum(len(t) for _, t in sessions)}")

    common = {
        "accountId": args.account,
        "userId": args.user,
        "agentId": args.agent,
    }

    # Sanity-check server
    health = http_get(args.base, "/health")
    print(f"[INFO] server health: {health}")

    run_id = args.marker or f"perf-{int(time.time())}"
    print(f"[INFO] run_id = {run_id}, workers = {args.workers}")
    started_at = time.time()
    print(f"[MARK] start_ts={started_at:.6f}")

    work = []
    for idx, (orig_key, turns) in enumerate(sessions):
        session_id = f"{args.prefix}-{run_id}-{idx:02d}"
        work.append((session_id, turns, orig_key))

    results: list[dict] = []
    lock = Lock()

    def task(item):
        session_id, turns, orig_key = item
        return run_session(args.base, common, session_id, turns, do_compose=not args.no_compose)

    if args.workers <= 1:
        for item in work:
            r = task(item)
            with lock:
                results.append(r)
                print(
                    f"[DONE] {r['session_id']} turns={r['turns']} "
                    f"after_turn={r['after_turn_ms']:.0f}ms "
                    f"compose={r['compose_ms']:.0f}ms err={len(r['errors'])}"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(task, item): item for item in work}
            for fut in as_completed(futures):
                r = fut.result()
                with lock:
                    results.append(r)
                    print(
                        f"[DONE] {r['session_id']} turns={r['turns']} "
                        f"after_turn={r['after_turn_ms']:.0f}ms "
                        f"compose={r['compose_ms']:.0f}ms err={len(r['errors'])}"
                    )

    finished_at = time.time()
    print(f"[MARK] end_ts={finished_at:.6f}")
    wall = finished_at - started_at

    summary = {
        "run_id": run_id,
        "workers": args.workers,
        "session_count": len(results),
        "wall_s": wall,
        "started_at": started_at,
        "finished_at": finished_at,
        "totals": {
            "turns": sum(r["turns"] for r in results),
            "after_turn_ms_sum": sum(r["after_turn_ms"] for r in results),
            "compose_ms_sum": sum(r["compose_ms"] for r in results),
            "errors": sum(len(r["errors"]) for r in results),
        },
        "per_session": results,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
