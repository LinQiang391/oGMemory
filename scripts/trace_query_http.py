#!/usr/bin/env python3
"""Trace the full ContextEngine pipeline via HTTP API.

Uses the running oG-Memory server at localhost:8090.

Usage:
    python scripts/trace_query_http.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "http://localhost:8090"
LOCOMO_PATH = "tests/e2e/data/locomo10.json"
SAMPLE_INDEX = 0
TARGET_QUESTION = "What fields would Caroline be likely to pursue in her educaton?"
EXPECTED_ANSWER = "Psychology, counseling certification"
USER_ID = "trace-user-1"
AGENT_ID = "main"


def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def section(title):
    print(f"\n--- {title} ---")

def kv(key, value, indent=2):
    prefix = " " * indent
    val_str = str(value)
    if len(val_str) > 300:
        val_str = val_str[:300] + f"... ({len(val_str)} chars)"
    print(f"{prefix}{key}: {val_str}")


def api_call(method, endpoint, payload=None, label=""):
    """Make a timed HTTP call and return (result_dict, elapsed_ms)."""
    url = f"{BASE_URL}{endpoint}"
    t0 = time.monotonic()
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=600)
    else:
        resp = requests.get(url, timeout=60)
    elapsed = (time.monotonic() - t0) * 1000
    print(f"  ⏱  {label or endpoint}: {elapsed:.1f}ms (HTTP {resp.status_code})")
    resp.raise_for_status()
    return resp.json(), elapsed


def load_sample():
    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)
    return data[SAMPLE_INDEX]


def build_sessions(sample):
    conv = sample["conversation"]
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    sessions = []
    for sk in session_keys:
        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")
        parts = [f"[group chat conversation: {date_time}]"]
        for msg in conv[sk]:
            speaker = msg.get("speaker", "unknown")
            text = msg.get("text", "")
            parts.append(f"{speaker}: {text}")
        combined = "\n\n".join(parts)
        sessions.append({
            "session_key": sk,
            "date_time": date_time,
            "message": combined,
            "char_count": len(combined),
        })
    return sessions


def main():
    banner("ContextEngine Pipeline Trace (HTTP)")
    print(f"Query:    \"{TARGET_QUESTION}\"")
    print(f"Expected: \"{EXPECTED_ANSWER}\"")
    print(f"Server:   {BASE_URL}")
    print(f"Time:     {ts()}")

    # Step 0: Health check
    banner("Step 0: Health Check")
    health, _ = api_call("GET", "/api/v1/health", label="health")
    for k, v in health.items():
        kv(k, v)

    # ================================================================
    # Step 1: INGEST — Stage ④ afterTurn
    # ================================================================
    banner("Step 1: INGEST conversations (Stage ④ afterTurn)")
    sample = load_sample()
    sessions = build_sessions(sample)
    kv("sample_id", sample["sample_id"])
    kv("total_sessions", len(sessions))

    total_ingest_ms = 0
    total_candidates = 0
    total_writes = 0

    for sess in sessions[:4]:  # First 4 sessions (contains evidence D1:9, D1:11)
        section(f"Ingesting {sess['session_key']} ({sess['date_time']})")
        kv("message_chars", sess["char_count"])

        session_id = f"trace-ingest-{sess['session_key']}"
        payload = {
            "sessionId": session_id,
            "messages": [{"role": "user", "content": sess["message"]}],
            "prePromptMessageCount": 0,
            "userId": USER_ID,
            "agentId": AGENT_ID,
        }

        result, elapsed = api_call("POST", "/api/v1/after_turn", payload, f"after_turn({sess['session_key']})")
        total_ingest_ms += elapsed

        kv("status", result.get("status"))
        extracted = result.get("candidates_extracted", 0)
        writes = result.get("writes_completed", 0)
        kv("candidates_extracted", extracted)
        kv("writes_completed", writes)
        kv("writes_skipped", result.get("writes_skipped", 0))
        kv("writes_failed", result.get("writes_failed", 0))
        total_candidates += extracted
        total_writes += writes

        # Print write plans (what was actually written)
        plans = result.get("plans", [])
        for p in plans[:5]:
            action = p.get("action", "?")
            uri = p.get("target_uri", "?")
            print(f"    [{action}] {uri}")
        if len(plans) > 5:
            print(f"    ... and {len(plans)-5} more")

        token_usage = result.get("token_usage", {})
        if token_usage:
            llm_u = token_usage.get("llm", {})
            emb_u = token_usage.get("embedding", {})
            kv("llm_tokens", f"in={llm_u.get('input_tokens',0)} out={llm_u.get('output_tokens',0)} calls={llm_u.get('calls',0)}")
            kv("embed_tokens", f"total={emb_u.get('total_tokens',0)} calls={emb_u.get('calls',0)}")

    section("Ingest Summary")
    kv("total_ingest_time_ms", f"{total_ingest_ms:.0f}")
    kv("total_candidates_extracted", total_candidates)
    kv("total_writes_completed", total_writes)

    # ================================================================
    # Step 2: COMPOSE — Stages ①②
    # ================================================================
    banner("Step 2: COMPOSE — Context assembly for query (Stages ①②)")

    question_time = "2023-08-23"
    input_msg = f"Current date: {question_time}. Answer the question directly: {TARGET_QUESTION}"

    section("Stage ①: message_received")
    kv("input_message", input_msg)
    kv("user_id", USER_ID)

    section("Stage ②: compose (bootstrap + assemble + before_prompt_build)")
    compose_payload = {
        "messages": [{"role": "user", "content": input_msg}],
        "sessionId": f"trace-qa-{int(time.time())}",
        "userId": USER_ID,
        "agentId": AGENT_ID,
        "tokenBudget": 128000,
    }

    result, elapsed = api_call("POST", "/api/v1/compose", compose_payload, "compose()")

    # Parse all output fields
    kv("estimatedTokens", result.get("estimatedTokens"))
    kv("archiveCount", result.get("archiveCount"))
    kv("archiveIncluded", result.get("archiveIncluded"))

    identity = result.get("identityContext", "")
    kv("identityContext_len", len(identity))
    if identity:
        kv("identityContext (first 200)", identity[:200])

    episodic = result.get("episodicContext", "")
    kv("episodicContext_len", len(episodic))

    session_ctx = result.get("sessionContext", "")
    kv("sessionContext_len", len(session_ctx))

    retrieved = result.get("retrievedEvidence", "")
    kv("retrievedEvidence_len", len(retrieved))

    kv("budgetUsedBySlot", result.get("budgetUsedBySlot"))
    kv("openLoops", result.get("openLoops"))
    kv("uncertainties", result.get("uncertainties"))

    # Print working set / retrieved evidence
    section("Retrieved Evidence (working set)")
    if retrieved:
        lines = retrieved.split("\n")
        for line in lines[:30]:
            print(f"  {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines)} lines total)")
    else:
        print("  (empty — no memories retrieved)")

    # Print injected messages
    section("Injected messages")
    messages = result.get("messages", [])
    kv("total_messages", len(messages))
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = content[:300] if len(content) <= 300 else content[:300] + f"... ({len(content)} chars total)"
        print(f"  msg[{i}] role={role}: {preview}")

    # ================================================================
    # Step 3: afterTurn for the QA response (Stage ④)
    # ================================================================
    banner("Step 3: afterTurn for QA (Stage ④)")
    simulated_response = (
        "Based on Caroline's strong interest in counseling and supporting the transgender community, "
        "she would likely pursue fields such as psychology and counseling certification."
    )
    at_payload = {
        "sessionId": f"trace-qa-at-{int(time.time())}",
        "messages": [
            {"role": "user", "content": input_msg},
            {"role": "assistant", "content": simulated_response},
        ],
        "prePromptMessageCount": 0,
        "userId": USER_ID,
        "agentId": AGENT_ID,
    }
    at_result, at_elapsed = api_call("POST", "/api/v1/after_turn", at_payload, "after_turn(QA)")
    kv("status", at_result.get("status"))
    kv("candidates_extracted", at_result.get("candidates_extracted", "n/a"))
    kv("writes_completed", at_result.get("writes_completed", "n/a"))

    # ================================================================
    # Step 4: Token stats
    # ================================================================
    banner("Step 4: Cumulative Token Stats")
    stats, _ = api_call("GET", "/api/v1/token_stats", label="token_stats")
    for k, v in stats.items():
        kv(k, v)

    # ================================================================
    # SUMMARY
    # ================================================================
    banner("TRACE SUMMARY")
    print(f"  Query: \"{TARGET_QUESTION}\"")
    print(f"  Expected answer: \"{EXPECTED_ANSWER}\"")
    print(f"  Total ingest time: {total_ingest_ms:.0f}ms ({len(sessions)} sessions)")
    print(f"  Total candidates extracted: {total_candidates}")
    print(f"  Total writes completed: {total_writes}")
    print(f"  Compose time: {elapsed:.0f}ms")
    print(f"  Retrieved evidence chars: {len(retrieved)}")
    print(f"  Identity context chars: {len(identity)}")
    ws_hits = retrieved.count("- [") if retrieved else 0
    print(f"  Working set hits: {ws_hits}")
    print(f"\nDone at {ts()}")


if __name__ == "__main__":
    main()
