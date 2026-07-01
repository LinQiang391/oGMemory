#!/usr/bin/env python3
"""Stress test: 1000+ conversation turns through ContextEngine session API.

Uses LoCoMo dataset conversations to drive realistic multi-session traffic.

Prerequisites:
  - AGFS server on port 1833
  - ContextEngine server on port 8090

Run:
    PYTHONPATH=. python3 tests/e2e/test_stress_1000_turns.py
    PYTHONPATH=. python3 tests/e2e/test_stress_1000_turns.py --synthetic  # no dataset needed
"""

import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
import argparse
from pathlib import Path

BASE = "http://localhost:8090/api/v1"
DATASET_PATH = Path(__file__).parent / "data" / "locomo10.json"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(path, params, timeout=30):
    data = json.dumps(params).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def http_get(path, params=None, timeout=10):
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{BASE}{path}{query}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "_code": e.code}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# LoCoMo dataset parser
# ---------------------------------------------------------------------------

def parse_locomo_conversations(path):
    """Parse LoCoMo JSON into list of (session_id, turns).

    Each turn is (role, content).
    Returns list of conversations, each being a list of turns.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    conversations = []
    for sample in data:
        sample_id = sample.get("sample_id", "unknown")
        conv = sample.get("conversation", {})

        # Extract sessions
        all_turns = []
        session_idx = 1
        while f"session_{session_idx}" in conv:
            s_key = f"session_{session_idx}"
            dt_key = f"session_{session_idx}_date_time"
            session_dt = conv.get(dt_key, "")

            for turn in conv[s_key]:
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                if not text:
                    continue
                # Map speaker to role
                if speaker.lower() in ("user", "human"):
                    role = "user"
                elif speaker.lower() in ("assistant", "ai", "bot", "system"):
                    role = "assistant"
                else:
                    role = "user"  # Default to user
                all_turns.append((role, text))

            session_idx += 1

        if all_turns:
            conversations.append((sample_id, all_turns))

    return conversations


def generate_synthetic_conversations(num_turns=1000, avg_turn_len=100):
    """Generate synthetic conversation turns."""
    topics = [
        "Python programming", "machine learning", "web development",
        "database design", "system architecture", "API design",
        "testing strategies", "code review", "deployment pipelines",
        "monitoring and observability", "security best practices",
        "performance optimization", "data modeling", "microservices",
        "event-driven architecture", "caching strategies",
    ]

    user_templates = [
        "我想了解一下{topic}方面的最佳实践",
        "能给我讲讲{topic}的关键概念吗？",
        "{topic}在实际项目中怎么应用的？",
        "关于{topic}，有什么常见的坑需要避免？",
        "请推荐一些学习{topic}的资源",
        "我遇到一个关于{topic}的问题，能帮我看一下吗？",
        "{topic}和{alt}有什么区别，各自适用什么场景？",
        "在{topic}中，如何做好性能优化？",
    ]

    assistant_templates = [
        "关于{topic}，有几个要点需要注意。首先，{detail}。其次，{detail}。最后，{detail}。",
        "{topic}的核心概念包括：1) {detail} 2) {detail} 3) {detail}。让我详细解释一下。",
        "在{topic}的实践中，我建议从以下几个方面入手：{detail}。这样可以帮助你更好地理解和应用。",
        "这是一个很好的问题。关于{topic}，{detail}。同时也要注意{detail}。",
    ]

    details = [
        "要注重基础概念的建立", "多做实践项目来巩固理解",
        "关注社区最新的发展动态", "学会阅读和分析源代码",
        "建立系统的知识体系", "注意边界条件和异常处理",
        "合理设计模块间的接口", "保持代码的可测试性",
        "使用合适的工具和框架", "重视文档和注释",
    ]

    turns = []
    for i in range(num_turns):
        topic = random.choice(topics)
        alt = random.choice([t for t in topics if t != topic])
        if i % 2 == 0:
            template = random.choice(user_templates)
            text = template.format(topic=topic, alt=alt)
            role = "user"
        else:
            template = random.choice(assistant_templates)
            d1, d2, d3 = random.sample(details, 3)
            text = template.format(topic=topic, detail=d1, alt=d2) + " " * random.randint(0, avg_turn_len)
            role = "assistant"
        turns.append((role, text))

    return [("synthetic", turns)]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_stress_test(conversations, session_prefix="stress", commit_threshold=500):
    """Run conversations through ContextEngine session API.

    Tracks: turns processed, commits triggered, compactions, errors.
    """
    stats = {
        "total_turns": 0,
        "total_messages_added": 0,
        "commits_triggered": 0,
        "compacts_triggered": 0,
        "errors": [],
        "latencies": [],
        "session_stats": {},
    }

    session_counter = 0

    for conv_idx, (sample_id, turns) in enumerate(conversations):
        session_counter += 1
        session_id = f"{session_prefix}-{session_counter:04d}"
        common = {
            "accountId": "acct-stress",
            "userId": "u-stress",
            "agentId": "agent-stress",
        }

        print(f"\n[{conv_idx+1}/{len(conversations)}] Session {session_id}: {len(turns)} turns (sample: {sample_id})")

        for turn_idx, (role, content) in enumerate(turns):
            t0 = time.time()

            # Add message to session buffer
            result = http_post(f"/sessions/{session_id}/messages", {
                **common,
                "role": role,
                "content": content[:4000],  # Cap at 4000 chars
            })

            latency = time.time() - t0
            stats["latencies"].append(latency)
            stats["total_turns"] += 1

            if result.get("ok"):
                stats["total_messages_added"] += 1
            else:
                stats["errors"].append({
                    "session": session_id,
                    "turn": turn_idx,
                    "error": result.get("error", "unknown"),
                })

            # Check pending_tokens and maybe commit
            pending = result.get("pending_tokens", 0)
            if pending >= commit_threshold:
                print(f"  Threshold reached ({pending} >= {commit_threshold}), committing...")
                commit_result = http_post(f"/sessions/{session_id}/commit", {
                    **common, "wait": False,
                })
                if commit_result.get("task_id"):
                    stats["commits_triggered"] += 1
                    print(f"  Commit triggered: {commit_result['task_id']}")

            # Progress report every 100 turns
            if stats["total_turns"] % 100 == 0:
                print(f"  Progress: {stats['total_turns']} turns, "
                      f"{stats['commits_triggered']} commits, "
                      f"{len(stats['errors'])} errors, "
                      f"avg latency: {sum(stats['latencies'][-100:])/min(len(stats['latencies']),100):.3f}s")

        # End of conversation: compact
        if len(turns) > 5:
            compact_result = http_post("/compact", {
                **common,
                "sessionId": session_id,
                "tokenBudget": 128000,
            })
            if compact_result.get("compacted"):
                stats["compacts_triggered"] += 1

        # Get final session state
        session_state = http_get(f"/sessions/{session_id}", common)
        stats["session_stats"][session_id] = {
            "pending_tokens": session_state.get("pending_tokens", 0),
            "commit_count": session_state.get("commit_count", 0),
            "message_count": session_state.get("message_count", 0),
            "turns_input": len(turns),
        }

    return stats


def print_report(stats):
    """Print final test report."""
    latencies = stats["latencies"]
    latencies.sort()

    print("\n" + "=" * 70)
    print("STRESS TEST REPORT")
    print("=" * 70)
    print(f"  Total turns processed:     {stats['total_turns']}")
    print(f"  Messages added:            {stats['total_messages_added']}")
    print(f"  Commits triggered:         {stats['commits_triggered']}")
    print(f"  Compacts triggered:        {stats['compacts_triggered']}")
    print(f"  Errors:                    {len(stats['errors'])}")
    print()
    if latencies:
        print(f"  Latency P50:               {latencies[len(latencies)//2]:.3f}s")
        print(f"  Latency P90:               {latencies[int(len(latencies)*0.9)]:.3f}s")
        print(f"  Latency P99:               {latencies[int(len(latencies)*0.99)]:.3f}s")
        print(f"  Latency max:               {latencies[-1]:.3f}s")
    print()

    if stats["errors"]:
        print("  First 5 errors:")
        for e in stats["errors"][:5]:
            print(f"    {e['session']} turn {e['turn']}: {str(e['error'])[:100]}")
    else:
        print("  No errors!")

    # Session summary
    print()
    total_commits = sum(s["commit_count"] for s in stats["session_stats"].values())
    total_pending = sum(s["pending_tokens"] for s in stats["session_stats"].values())
    print(f"  Sessions:                  {len(stats['session_stats'])}")
    print(f"  Total session commits:     {total_commits}")
    print(f"  Total pending tokens:      {total_pending}")

    print("=" * 70)

    ok = len(stats["errors"]) == 0 and stats["total_turns"] >= 1000
    if ok:
        print("  RESULT: PASSED")
    elif stats["total_turns"] >= 1000:
        print(f"  RESULT: PASSED WITH {len(stats['errors'])} ERRORS")
    else:
        print("  RESULT: FAILED (not enough turns)")
    print("=" * 70)

    return ok


def main():
    parser = argparse.ArgumentParser(description="Stress test ContextEngine with 1000+ turns")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of LoCoMo")
    parser.add_argument("--num-turns", type=int, default=1200, help="Number of turns for synthetic mode")
    parser.add_argument("--threshold", type=int, default=500, help="Commit token threshold")
    args = parser.parse_args()

    print("=" * 70)
    print("ContextEngine Stress Test — 1000+ Turns")
    print("=" * 70)

    # Wait for server
    print("\nWaiting for ContextEngine server...")
    for _ in range(15):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=3)
            print("  Server ready!")
            break
        except Exception:
            time.sleep(1)
    else:
        print("  FAIL: Server not reachable")
        sys.exit(1)

    # Load conversations
    if args.synthetic:
        print(f"\nUsing synthetic data ({args.num_turns} turns)")
        conversations = generate_synthetic_conversations(args.num_turns)
    elif DATASET_PATH.exists():
        print(f"\nLoading LoCoMo dataset from {DATASET_PATH}")
        conversations = parse_locomo_conversations(DATASET_PATH)
        total = sum(len(t) for _, t in conversations)
        print(f"  Loaded {len(conversations)} conversations, {total} total turns")
    else:
        print(f"\nDataset not found at {DATASET_PATH}, falling back to synthetic")
        conversations = generate_synthetic_conversations(args.num_turns)

    # Run
    t_start = time.time()
    stats = run_stress_test(
        conversations,
        session_prefix="stress",
        commit_threshold=args.threshold,
    )
    elapsed = time.time() - t_start

    print(f"\nWall time: {elapsed:.1f}s ({stats['total_turns']/elapsed:.1f} turns/sec)")

    ok = print_report(stats)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
