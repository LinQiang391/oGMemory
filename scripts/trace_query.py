#!/usr/bin/env python3
"""Trace the full ContextEngine pipeline for a single query.

Usage:
    source .venv/bin/activate
    python scripts/trace_query.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["OGMEM_AFTER_TURN_THRESHOLD"] = "1"  # Always extract on after_turn
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from server.memory_service import MemoryService, extract_query, extract_content_text
from providers.unified_config import get_config
from core.models import RequestContext, Role
from uuid import uuid4

# ---- Config ----
LOCOMO_PATH = "tests/e2e/data/locomo10.json"
SAMPLE_INDEX = 0
TARGET_QUESTION = "What fields would Caroline be likely to pursue in her educaton?"
EXPECTED_ANSWER = "Psychology, counseling certification"
USER_ID = "trace-user-1"
AGENT_ID = "main"

# ---- Helpers ----

def timestamp():
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
    if len(val_str) > 200:
        val_str = val_str[:200] + f"... ({len(val_str)} chars)"
    print(f"{prefix}{key}: {val_str}")


def timed(label):
    """Context manager for timing a block."""
    class Timer:
        def __init__(self):
            self.elapsed_ms = 0
        def __enter__(self):
            self._t0 = time.monotonic()
            return self
        def __exit__(self, *_):
            self.elapsed_ms = (time.monotonic() - self._t0) * 1000
            print(f"  ⏱  {label}: {self.elapsed_ms:.1f}ms")
    return Timer()


# ---- Load LoCoMo data ----

def load_sample():
    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)
    return data[SAMPLE_INDEX]


def build_sessions(sample, session_range=None):
    """Build session messages from LoCoMo sample."""
    conv = sample["conversation"]
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    sessions = []
    for sk in session_keys:
        sess_num = int(sk.split("_")[1])
        if session_range and (sess_num < session_range[0] or sess_num > session_range[1]):
            continue
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
        })
    return sessions


# ---- Main ----

def main():
    banner("ContextEngine Pipeline Trace")
    print(f"Query: \"{TARGET_QUESTION}\"")
    print(f"Expected: \"{EXPECTED_ANSWER}\"")
    print(f"Time: {timestamp()}")

    # ================================================================
    # STEP 0: Initialize MemoryService
    # ================================================================
    banner("Step 0: Initialize MemoryService")
    with timed("MemoryService init") as t:
        cfg = get_config()
        svc = MemoryService(config=cfg)
    kv("provider", cfg.provider)
    kv("llm_model", cfg.openai_llm_model)
    kv("agfs_base_url", cfg.agfs_base_url)
    kv("mount_prefix", cfg.agfs_mount_prefix)
    kv("vector_db_type", cfg.vector_db_type)
    kv("account_id", cfg.account_id)
    kv("user_id", USER_ID)

    # Build RequestContext
    ctx = RequestContext(
        account_id=cfg.account_id,
        user_id=USER_ID,
        agent_id=AGENT_ID,
        session_id="trace-session-ingest",
        trace_id=str(uuid4()),
        role=Role.ROOT,
        visible_owner_spaces=(),
    )

    # ================================================================
    # STEP 1: INGEST - Load LoCoMo conversations into memory
    # ================================================================
    banner("Step 1: INGEST — Load conversations into memory (Stage ④ afterTurn)")
    sample = load_sample()
    sessions = build_sessions(sample)
    kv("sample_id", sample["sample_id"])
    kv("total_sessions", len(sessions))

    total_ingest_time = 0
    for i, sess in enumerate(sessions):
        section(f"Ingesting {sess['session_key']} ({sess['date_time']})")
        session_id = f"trace-ingest-{sess['session_key']}"

        # Build after_turn params (simulate what OpenClaw sends)
        messages = [{"role": "user", "content": sess["message"]}]
        params = {
            "sessionId": session_id,
            "messages": messages,
            "prePromptMessageCount": 0,
            "userId": USER_ID,
            "agentId": AGENT_ID,
        }

        with timed(f"after_turn({sess['session_key']})") as t:
            result = svc.after_turn(params)
        total_ingest_time += t.elapsed_ms

        kv("status", result.get("status"))
        kv("candidates_extracted", result.get("candidates_extracted", "n/a"))
        kv("writes_completed", result.get("writes_completed", "n/a"))
        kv("writes_skipped", result.get("writes_skipped", "n/a"))
        token_usage = result.get("token_usage", {})
        if token_usage:
            llm_u = token_usage.get("llm", {})
            kv("llm_tokens", f"in={llm_u.get('input_tokens',0)} out={llm_u.get('output_tokens',0)}")

        # Drain outbox after each session to ensure indexing
        with timed("drain_outbox") as t2:
            drain = svc.drain_outbox_sync()
        kv("outbox_drain", drain)

    print(f"\n  Total ingest time: {total_ingest_time:.0f}ms")

    # ================================================================
    # STEP 2: COMPOSE — Run the target query (Stages ①②)
    # ================================================================
    banner("Step 2: COMPOSE — Assemble context for query (Stages ①②)")

    qa_session_id = f"trace-qa-{uuid4().hex[:8]}"
    qa_ctx = RequestContext(
        account_id=cfg.account_id,
        user_id=USER_ID,
        agent_id=AGENT_ID,
        session_id=qa_session_id,
        trace_id=str(uuid4()),
        role=Role.ROOT,
        visible_owner_spaces=(),
    )

    question_time = "2023-08-23"
    input_msg = f"Current date: {question_time}. Answer the question directly: {TARGET_QUESTION}"

    compose_params = {
        "messages": [{"role": "user", "content": input_msg}],
        "sessionId": qa_session_id,
        "userId": USER_ID,
        "agentId": AGENT_ID,
        "tokenBudget": 128000,
        "_ctx": qa_ctx,
    }

    # ---- Stage ①: message_received ----
    section("Stage ①: message_received")
    kv("input_message", input_msg)
    kv("session_id", qa_session_id)
    kv("user_id", USER_ID)

    # ---- Stage ②: compose (bootstrap + assemble + before_prompt_build) ----
    section("Stage ②: compose — full context assembly")

    # 2a: Query extraction
    with timed("query_extraction") as t:
        from retrieval.query_planner import sanitize_query
        raw_query = extract_query(compose_params["messages"])
        sanitized = sanitize_query(raw_query)
    kv("raw_query (last 3 user msgs)", raw_query[:150])
    kv("sanitized_query", sanitized[:150])

    # 2b: Budget allocation
    from core.models import TokenBudget
    budget = TokenBudget(total=128000)
    allocation = budget.allocate()
    kv("budget_allocation", allocation)

    # 2c: Profile read
    section("Stage ②c: Profile read")
    with timed("profile_read") as t:
        try:
            profile = svc._read_profile(qa_ctx)
        except Exception as e:
            profile = f"[ERROR: {e}]"
    kv("profile_content", profile if profile else "(empty)")

    # 2d: Archive collection
    section("Stage ②d: Archive collection")
    from core.models import TokenBudget as TB
    with timed("archive_collection") as t:
        try:
            latest_archives, pre_archives = svc._collect_archives(qa_ctx, TB(total=128000))
        except Exception as e:
            latest_archives, pre_archives = [], []
            print(f"  [WARN] archive collection failed: {e}")
    kv("latest_archives", len(latest_archives))
    kv("pre_archives", len(pre_archives))
    for ar in latest_archives[:2]:
        kv("  latest.archive_id", ar.archive_id)
        kv("  latest.abstract", (ar.abstract or "")[:100])

    # 2e: Working set search (CORE RETRIEVAL)
    section("Stage ②e: Working set search (RetrievalPipeline)")

    # 2e-1: QueryPlanner
    section("Stage ②e-1: QueryPlanner")
    from retrieval.query_planner import QueryPlanner
    from core.models import RetrievalConfig
    planner = QueryPlanner(RetrievalConfig())
    with timed("QueryPlanner.plan()") as t:
        typed_queries = planner.plan(sanitized, qa_ctx)
    for tq in typed_queries:
        kv("TypedQuery.text", tq.text[:100])
        kv("TypedQuery.context_type", tq.context_type)
        kv("TypedQuery.top_k", tq.top_k)
        kv("TypedQuery.intent", tq.intent)
        kv("TypedQuery.owner_space", tq.owner_space)
        kv("TypedQuery.account_id", tq.account_id)

    # 2e-2 through 2e-4: Full retrieval pipeline
    section("Stage ②e-2..4: Full RetrievalPipeline.run()")
    read_api = svc.get_read_api()
    pipeline = read_api._pipeline

    with timed("RetrievalPipeline.run()") as t:
        search_result = pipeline.run(
            sanitized, qa_ctx,
            top_k=50, mode="QUICK",
            fill_content_for_top_k=6,
        )

    # Print trace
    if search_result.trace:
        trace = search_result.trace
        kv("trace.request_id", trace.request_id)
        kv("trace.total_ms", f"{trace.total_ms:.1f}ms")
        for stage in trace.stages:
            kv(f"  stage={stage.stage}", f"in={stage.input_count} out={stage.output_count} time={stage.elapsed_ms:.1f}ms")

    # Print hits
    kv("total_hits", len(search_result.hits))
    for i, hit in enumerate(search_result.hits[:15]):
        score_str = f"{hit.score:.4f}" if hit.score else "n/a"
        content_preview = (hit.content_excerpt or "")[:80] if hit.content_excerpt else ""
        abstract_preview = (hit.abstract or "")[:80]
        print(f"    [{i}] score={score_str} cat={hit.category} uri={hit.uri}")
        if abstract_preview:
            print(f"         abstract: {abstract_preview}")
        if content_preview:
            print(f"         content:  {content_preview}")

    # 2f: Working set filtering + formatting
    section("Stage ②f: Working set filter + format")
    working_set = [
        {
            "uri": h.uri,
            "abstract": h.abstract or "",
            "content": h.content_excerpt or "",
            "score": h.score,
            "category": h.category,
        }
        for h in search_result.hits
        if h.score >= 0.3
    ][:15]
    kv("working_set_count (score>=0.3)", len(working_set))

    with timed("format_working_set") as t:
        ws_text = svc._format_working_set(working_set, allocation)
    kv("working_set_text_len", len(ws_text))
    # Print first 500 chars
    print(f"  Working set text (first 500 chars):")
    for line in ws_text[:500].split("\n"):
        print(f"    {line}")

    # 2g: Full compose call
    section("Stage ②g: Full compose() call")
    svc.collect_token_usage(reset=True)
    with timed("MemoryService.compose()") as t:
        compose_result = svc.compose(compose_params)

    kv("estimatedTokens", compose_result.get("estimatedTokens"))
    kv("archiveCount", compose_result.get("archiveCount"))
    kv("archiveIncluded", compose_result.get("archiveIncluded"))
    kv("identityContext_len", len(compose_result.get("identityContext", "")))
    kv("retrievedEvidence_len", len(compose_result.get("retrievedEvidence", "")))
    kv("messages_count", len(compose_result.get("messages", [])))
    kv("budgetUsedBySlot", compose_result.get("budgetUsedBySlot"))

    # Print injected messages
    section("Injected messages detail")
    for i, msg in enumerate(compose_result.get("messages", [])):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = content[:200] if len(content) <= 200 else content[:200] + f"... ({len(content)} chars total)"
        print(f"  msg[{i}] role={role}: {preview}")

    # ================================================================
    # STEP 3: afterTurn for the QA response (Stage ④)
    # ================================================================
    banner("Step 3: afterTurn for QA (Stage ④)")
    # Simulate a simple assistant response
    simulated_response = "Based on Caroline's strong interest in counseling and supporting the transgender community, she would likely pursue fields such as psychology and counseling certification."
    after_turn_params = {
        "sessionId": qa_session_id,
        "messages": [
            {"role": "user", "content": input_msg},
            {"role": "assistant", "content": simulated_response},
        ],
        "prePromptMessageCount": 0,
        "userId": USER_ID,
        "agentId": AGENT_ID,
        "_ctx": qa_ctx,
    }
    with timed("after_turn (QA)") as t:
        at_result = svc.after_turn(after_turn_params)
    kv("status", at_result.get("status"))
    kv("candidates_extracted", at_result.get("candidates_extracted", "n/a"))
    kv("writes_completed", at_result.get("writes_completed", "n/a"))

    # ================================================================
    # SUMMARY
    # ================================================================
    banner("TRACE SUMMARY")
    print(f"  Query: \"{TARGET_QUESTION}\"")
    print(f"  Expected answer: \"{EXPECTED_ANSWER}\"")
    print(f"  Working set hits (score>=0.3): {len(working_set)}")
    print(f"  Top hit: {working_set[0]['uri'] if working_set else 'none'}")
    print(f"    score: {working_set[0]['score']:.4f}" if working_set else "")
    print(f"    content: {(working_set[0].get('content') or working_set[0].get('abstract',''))[:150]}" if working_set else "")
    print(f"\nDone at {timestamp()}")


if __name__ == "__main__":
    main()
