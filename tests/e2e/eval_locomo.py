#!/usr/bin/env python3
"""LoCoMo benchmark evaluation — replicates OpenViking's exact evaluation pipeline.

Two modes:
  ingest  - Load LoCoMo conversations through OpenClaw (builds memory via plugin)
  qa      - Run QA questions against accumulated memory, output response vs expected answer

Usage:
    # Step 1: Ingest all conversations (builds memory)
    PYTHONPATH=. python3 tests/e2e/eval_locomo.py ingest tests/e2e/data/locomo10.json

    # Step 2: Run QA evaluation
    PYTHONPATH=. python3 tests/e2e/eval_locomo.py qa tests/e2e/data/locomo10.json --output tests/e2e/results

    # Step 3: Judge scoring (uses LLM)
    PYTHONPATH=. python3 tests/e2e/eval_locomo.py judge tests/e2e/results.csv

    # Step 4: Show statistics
    PYTHONPATH=. python3 tests/e2e/eval_locomo.py stat tests/e2e/results.csv

Prerequisites:
  - AGFS server on port 1833
  - ContextEngine server on port 8090
  - OpenClaw gateway running on port 18789
  - og-memory-context-engine plugin configured in OpenClaw
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENCLAW_BIN = os.path.expanduser("~/.npm-global/bin/openclaw")
DEFAULT_AGENT = "main"
DEFAULT_TIMEOUT = 120
DEFAULT_BASE_URL = "http://127.0.0.1:18789"
DEFAULT_SESSION_KEY = "eval-test-3"
INGEST_RECORD_PATH = "tests/e2e/.ingest_record.json"


# ---------------------------------------------------------------------------
# LoCoMo JSON parsing (same as OpenViking eval.py)
# ---------------------------------------------------------------------------

def format_locomo_message(msg: dict) -> str:
    """Format a single LoCoMo message into a natural chat-style string."""
    speaker = msg.get("speaker", "unknown")
    text = msg.get("text", "")
    line = f"{speaker}: {text}"

    img_urls = msg.get("img_url", [])
    if isinstance(img_urls, str):
        img_urls = [img_urls]
    blip = msg.get("blip_caption", "")

    if img_urls:
        for url in img_urls:
            caption = f": {blip}" if blip else ""
            line += f"\n{url}{caption}"
    elif blip:
        line += f"\n({blip})"

    return line


def load_locomo_data(path: str, sample_index: int | None = None) -> list[dict]:
    """Load LoCoMo JSON and optionally filter to one sample."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(data):
            print(f"Error: sample index {sample_index} out of range (0-{len(data)-1})", file=sys.stderr)
            sys.exit(1)
        return [data[sample_index]]
    return data


def build_session_messages(
    item: dict,
    session_range: tuple[int, int] | None = None,
    tail: str = "[]",
) -> list[dict]:
    """Build bundled session messages for one LoCoMo sample.

    Each session is bundled into a single message with header.
    Returns list of dicts with keys: message, meta.
    """
    conv = item["conversation"]
    speakers = f"{conv['speaker_a']} & {conv['speaker_b']}"

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        sess_num = int(sk.split("_")[1])
        if session_range:
            lo, hi = session_range
            if sess_num < lo or sess_num > hi:
                continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        parts = [f"[group chat conversation: {date_time}]"]
        for msg in conv[sk]:
            parts.append(format_locomo_message(msg))
        if tail:
            parts.append(tail)
        combined = "\n\n".join(parts)

        sessions.append({
            "message": combined,
            "meta": {
                "sample_id": item["sample_id"],
                "session_key": sk,
                "date_time": date_time,
                "speakers": speakers,
            },
        })

    return sessions


def parse_session_range(s: str) -> tuple[int, int]:
    """Parse '1-4' or '3' into (lo, hi) inclusive tuple."""
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


# ---------------------------------------------------------------------------
# OpenClaw CLI interface
# ---------------------------------------------------------------------------

def send_message(
    message: str,
    session_id: str,
    agent: str = DEFAULT_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict]:
    """Send a message to OpenClaw via CLI.

    Returns (response_text, usage_dict).
    """
    cmd = [
        OPENCLAW_BIN, "agent",
        "-m", message,
        "--agent", agent,
        "--session-id", session_id,
        "--json",
        "--timeout", str(timeout),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # subprocess timeout > API timeout
        )
    except subprocess.TimeoutExpired:
        return "[ERROR] Timeout", {}
    except FileNotFoundError:
        return f"[ERROR] openclaw not found at {OPENCLAW_BIN}", {}

    if result.returncode != 0:
        stderr = result.stderr.strip()[-200:] if result.stderr else ""
        return f"[ERROR] exit={result.returncode}: {stderr}", {}

    # Parse JSON output
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return f"[ERROR] JSON parse: {result.stdout[-200:]}", {}

    # Extract response text
    text = ""
    payloads = data.get("result", {}).get("payloads", [])
    if payloads:
        text = payloads[0].get("text", "")

    # Extract usage
    agent_meta = data.get("result", {}).get("meta", {}).get("agentMeta", {})
    usage = agent_meta.get("usage", {})
    if not usage:
        usage = agent_meta.get("lastCallUsage", {})

    return text, usage


def send_message_http(
    message: str,
    base_url: str,
    token: str,
    user: str = "",
    agent_id: str = DEFAULT_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict]:
    """Send a message via OpenClaw HTTP API (like OpenViking eval.py).

    Returns (response_text, usage_dict).
    """
    url = f"{base_url}/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-OpenClaw-Agent-ID": agent_id,
        "X-OpenClaw-Session-Key": DEFAULT_SESSION_KEY,
    }
    payload = {
        "model": "openclaw",
        "input": message,
        "stream": False,
    }
    if user:
        payload["user"] = user

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
    except requests.exceptions.ConnectionError as e:
        return f"[ERROR] Connection: {e}", {}
    except requests.exceptions.Timeout as e:
        return f"[ERROR] Timeout", {}
    except requests.exceptions.HTTPError as e:
        return f"[ERROR] HTTP {e.response.status_code}: {e}", {}

    # Extract response text
    text = ""
    try:
        for item in body.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        break
            if text:
                break
        if not text:
            for item in body.get("output", []):
                if "text" in item:
                    text = item["text"]
                    break
                for content in item.get("content", []):
                    if "text" in content:
                        text = content["text"]
                        break
    except (KeyError, TypeError, IndexError):
        text = f"[ERROR] parse response: {str(body)[:200]}"

    usage = body.get("usage", {"input_tokens": 0, "output_tokens": 0, "cacheRead": 0, "total_tokens": 0})
    # Normalize usage keys to match CLI format
    normalized = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cacheRead": usage.get("cacheRead", 0),
        "total": usage.get("total_tokens", 0),
    }
    return text, normalized


def _is_http_mode(args: argparse.Namespace) -> bool:
    """Check if we should use HTTP API mode (when --token is provided)."""
    return bool(getattr(args, "token", None))


def get_session_file(session_id: str, agent: str = DEFAULT_AGENT) -> str:
    """Get the path to the session jsonl file."""
    return os.path.expanduser(f"~/.openclaw/agents/{agent}/sessions/{session_id}.jsonl")


def archive_session(session_id: str, agent: str = DEFAULT_AGENT) -> str | None:
    """Archive a session jsonl file by renaming with timestamp suffix."""
    src = get_session_file(session_id, agent)
    if not os.path.exists(src):
        return None
    dst = f"{src}.{int(time.time())}"
    try:
        os.rename(src, dst)
        return os.path.basename(dst)
    except OSError as e:
        print(f"    [archive] Error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Ingest record (avoid duplicate ingestion)
# ---------------------------------------------------------------------------

def load_ingest_record(path: str = INGEST_RECORD_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_ingest_record(record: dict, path: str = INGEST_RECORD_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def is_already_ingested(sample_id, session_key, record: dict) -> bool:
    key = f"{sample_id}:{session_key}"
    return key in record and record[key].get("success", False)


def mark_ingested(sample_id, session_key, record: dict, meta: dict | None = None) -> None:
    key = f"{sample_id}:{session_key}"
    record[key] = {"success": True, "timestamp": int(time.time()), "meta": meta or {}}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "sample_id", "sample_idx", "qi", "question", "expected",
    "response", "category", "evidence",
    "input_tokens", "output_tokens", "cacheRead", "total_tokens",
    "timestamp", "is_correct", "reasoning",
]


def load_executed_records(csv_path: str) -> set:
    """Load already executed records from CSV, returns set of (sample_id, qi)."""
    executed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                executed.add((row["sample_id"], int(row["qi"])))
    return executed


def save_record_to_csv(csv_path: str, record: dict) -> None:
    """Save a single QA record to CSV file."""
    file_exists = os.path.exists(csv_path)
    flat = record.copy()
    usage = flat.pop("usage", {})
    flat["input_tokens"] = usage.get("input", 0)
    flat["output_tokens"] = usage.get("output", 0)
    flat["cacheRead"] = usage.get("cacheRead", 0)
    flat["total_tokens"] = usage.get("total", 0)
    flat["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flat.setdefault("is_correct", "")
    flat.setdefault("reasoning", "")

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)
        f.flush()


# ---------------------------------------------------------------------------
# Mode: ingest
# ---------------------------------------------------------------------------

def run_ingest(args: argparse.Namespace) -> None:
    """Load LoCoMo conversations through OpenClaw to build memory."""
    session_range = parse_session_range(args.sessions) if args.sessions else None

    if args.clear_ingest_record:
        record = {}
        save_ingest_record(record)
        print("[INFO] Ingest record cleared", file=sys.stderr)
    else:
        record = load_ingest_record()

    samples = load_locomo_data(args.input, args.sample)

    total_sessions = 0
    completed = 0
    skipped = 0
    errors = 0

    use_http = _is_http_mode(args)

    for idx, item in enumerate(samples):
        sample_id = item["sample_id"]
        sessions = build_session_messages(item, session_range, tail=args.tail)

        # Per-sample user identity (matches OpenViking behavior)
        user_key = args.user or f"eval-{idx}"

        print(f"\n=== Sample {sample_id} [{idx}] ({len(sessions)} sessions, user={user_key}) ===", file=sys.stderr)

        for sess in sessions:
            total_sessions += 1
            meta = sess["meta"]
            msg = sess["message"]
            label = f"{meta['session_key']} ({meta['date_time']})"

            # Skip already ingested
            if not args.force_ingest and is_already_ingested(sample_id, meta["session_key"], record):
                print(f"  [{label}] SKIP (already ingested)", file=sys.stderr)
                skipped += 1
                continue

            # Prepend per-sample user marker for plugin to extract
            tagged_msg = f"[ogmem-user: {user_key}]\n{msg}"

            preview = msg.replace("\n", " | ")[:80]
            print(f"  [{label}] {preview}...", file=sys.stderr)

            # Use unique session ID per conversation session
            session_id = f"locomo-{sample_id}-{meta['session_key']}"

            t0 = time.time()
            if use_http:
                response, usage = send_message_http(
                    tagged_msg, args.base_url, args.token,
                    user=user_key, agent_id=args.agent, timeout=args.timeout,
                )
            else:
                response, usage = send_message(tagged_msg, session_id, agent=args.agent, timeout=args.timeout)
            elapsed = time.time() - t0

            if response.startswith("[ERROR]"):
                print(f"    -> ERROR: {response}", file=sys.stderr)
                errors += 1
            else:
                tok_in = usage.get("input", 0)
                tok_out = usage.get("output", 0)
                tok_cache = usage.get("cacheRead", 0)
                tok_total = usage.get("total", 0)
                print(f"    -> OK ({elapsed:.1f}s, {len(response)} chars) tokens: in={tok_in:,} out={tok_out:,} cache={tok_cache:,} total={tok_total:,}", file=sys.stderr)
                completed += 1
                mark_ingested(sample_id, meta["session_key"], record, {
                    "date_time": meta["date_time"],
                    "elapsed": round(elapsed, 1),
                    "usage": usage,
                    "user": user_key,
                })

            # Archive session file
            archive_session(session_id, args.agent)

            # Save record incrementally after each session
            save_ingest_record(record)

    # Aggregate token usage from ingest record
    total_usage = {"input": 0, "output": 0, "cacheRead": 0, "total": 0}
    for _key, entry in record.items():
        meta = entry.get("meta", {})
        u = meta.get("usage", {}) if isinstance(meta, dict) else {}
        for k in total_usage:
            total_usage[k] += u.get(k, 0)

    print(f"\n=== Ingest Summary ===", file=sys.stderr)
    print(f"  Total sessions:    {total_sessions}", file=sys.stderr)
    print(f"  Completed:         {completed}", file=sys.stderr)
    print(f"  Skipped:           {skipped}", file=sys.stderr)
    print(f"  Errors:            {errors}", file=sys.stderr)
    print(f"  Token usage:", file=sys.stderr)
    print(f"    Input:           {total_usage['input']:,}", file=sys.stderr)
    print(f"    Output:          {total_usage['output']:,}", file=sys.stderr)
    print(f"    Cache read:      {total_usage['cacheRead']:,}", file=sys.stderr)
    print(f"    Total:           {total_usage['total']:,}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mode: QA
# ---------------------------------------------------------------------------

def run_qa(args: argparse.Namespace) -> None:
    """Run QA questions against accumulated memory."""
    samples = load_locomo_data(args.input, args.sample)

    csv_path = f"{args.output}.csv" if args.output else "tests/e2e/qa_results.csv"
    executed = load_executed_records(csv_path)
    print(f"  Loaded {len(executed)} already executed records", file=sys.stderr)

    use_http = _is_http_mode(args)
    total_usage = {"input": 0, "output": 0, "cacheRead": 0, "total": 0}

    for idx, item in enumerate(samples):
        sample_id = item["sample_id"]
        # Per-sample user identity (must match ingest user)
        user_key = args.user or f"eval-{idx}"
        qas = [q for q in item.get("qa", []) if str(q.get("category", "")) != "5"]
        if args.count is not None:
            qas = qas[:args.count]

        print(f"\n=== Sample {sample_id} [{idx}] ({len(qas)} QA, user={user_key}) ===", file=sys.stderr)

        for qi, qa in enumerate(qas, start=1):
            # Skip already executed
            if (str(sample_id), qi) in executed:
                print(f"  [{idx}] Q{qi}: SKIP (already executed)", file=sys.stderr)
                continue

            question = qa["question"]
            expected = str(qa["answer"])
            category = qa.get("category", "")
            evidence = qa.get("evidence", [])

            print(f"  [{idx}] Q{qi}: {question[:60]}{'...' if len(question) > 60 else ''}", file=sys.stderr)

            # Prepend per-sample user marker for plugin to extract
            tagged_question = f"[ogmem-user: {user_key}]\n{question}"

            # Use unique session ID per question (no chat history leakage)
            session_id = f"locomo-qa-{sample_id}-{qi}"

            t0 = time.time()
            if use_http:
                response, usage = send_message_http(
                    tagged_question, args.base_url, args.token,
                    user=user_key, agent_id=args.agent, timeout=args.timeout,
                )
            else:
                response, usage = send_message(tagged_question, session_id, agent=args.agent, timeout=args.timeout)
            elapsed = time.time() - t0

            # Aggregate usage
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            if response.startswith("[ERROR]"):
                print(f"    -> ERROR: {response[:100]}", file=sys.stderr)
            else:
                print(f"    -> {response[:80]}{'...' if len(response) > 80 else ''} ({elapsed:.1f}s)", file=sys.stderr)
                print(f"       tokens: in={usage.get('input',0)} out={usage.get('output',0)} "
                      f"cache={usage.get('cacheRead',0)} total={usage.get('total',0)}", file=sys.stderr)

            # Save record
            record = {
                "sample_id": str(sample_id),
                "sample_idx": idx,
                "qi": qi,
                "question": question,
                "expected": expected,
                "response": response,
                "category": category,
                "evidence": json.dumps(evidence),
                "usage": usage,
            }
            save_record_to_csv(csv_path, record)

            # Archive session
            archive_session(session_id, args.agent)

            # Rate limit delay
            if args.delay:
                time.sleep(args.delay)

    print(f"\n=== QA Summary ===", file=sys.stderr)
    print(f"  Total tokens: in={total_usage['input']} out={total_usage['output']} "
          f"total={total_usage['total']}", file=sys.stderr)
    print(f"  Results saved to: {csv_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mode: judge
# ---------------------------------------------------------------------------

def run_judge(args: argparse.Namespace) -> None:
    """LLM-based scoring of QA responses."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    csv_path = args.input
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    # Configure LLM client for judging
    api_key = args.api_key or os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.api_base
    model = args.judge_model

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load all records
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    judged = 0
    correct = 0
    wrong = 0
    errors_count = 0

    print(f"Judging {len(rows)} responses with {model}...", file=sys.stderr)

    for i, row in enumerate(rows):
        if row.get("is_correct") and row["is_correct"] in ("CORRECT", "WRONG"):
            # Already judged
            if row["is_correct"] == "CORRECT":
                correct += 1
            else:
                wrong += 1
            judged += 1
            continue

        question = row["question"]
        expected = row["expected"]
        response = row["response"]

        prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer

The point of the question is to ask about something one user should know about the other based on prior conversations.
Be generous with grading - as long as it touches on the same topic as the gold answer, count as CORRECT.
For time questions, accept same date/time even if format differs.

Question: {question}
Gold answer: {expected}
Generated answer: {response}

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert grader."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=60,
            )
            content = resp.choices[0].message.content.strip()

            # Extract JSON
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                result = json.loads(content[start:end+1])
                is_correct = result.get("is_correct", "WRONG").strip().upper()
                reasoning = result.get("reasoning", "")
            else:
                is_correct = "WRONG"
                reasoning = f"[PARSE ERROR] {content[:100]}"

        except Exception as e:
            is_correct = "WRONG"
            reasoning = f"[ERROR] {e}"
            errors_count += 1

        # Update row
        row["is_correct"] = is_correct
        row["reasoning"] = reasoning

        if is_correct == "CORRECT":
            correct += 1
        else:
            wrong += 1
        judged += 1

        if judged % 10 == 0 or judged == len(rows):
            print(f"  [{judged}/{len(rows)}] accuracy={correct/judged:.1%} "
                  f"(correct={correct}, wrong={wrong}, errors={errors_count})", file=sys.stderr)

    # Write updated CSV
    output_path = args.output or csv_path.replace(".csv", "_judged.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Judge Results ===", file=sys.stderr)
    print(f"  Total:     {judged}", file=sys.stderr)
    print(f"  Correct:   {correct}", file=sys.stderr)
    print(f"  Wrong:     {wrong}", file=sys.stderr)
    print(f"  Errors:    {errors_count}", file=sys.stderr)
    print(f"  Accuracy:  {correct/judged:.1%}" if judged > 0 else "  Accuracy: N/A", file=sys.stderr)
    print(f"  Output:    {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mode: stat
# ---------------------------------------------------------------------------

def run_stat(args: argparse.Namespace) -> None:
    """Compute statistics from judged CSV."""
    csv_path = args.input
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct") == "CORRECT")
    wrong = sum(1 for r in rows if r.get("is_correct") == "WRONG")
    unjudged = total - correct - wrong

    # Per-category accuracy
    categories = {}
    for r in rows:
        cat = r.get("category", "?")
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("is_correct") == "CORRECT":
            categories[cat]["correct"] += 1

    # Per-sample accuracy
    samples = {}
    for r in rows:
        sid = r.get("sample_id", "?")
        if sid not in samples:
            samples[sid] = {"correct": 0, "total": 0}
        samples[sid]["total"] += 1
        if r.get("is_correct") == "CORRECT":
            samples[sid]["correct"] += 1

    # Token usage
    total_input = sum(int(r.get("input_tokens", 0)) for r in rows)
    total_output = sum(int(r.get("output_tokens", 0)) for r in rows)
    total_cache = sum(int(r.get("cacheRead", 0)) for r in rows)
    total_tokens = sum(int(r.get("total_tokens", 0)) for r in rows)

    print("=" * 60)
    print("LoCoMo Evaluation Statistics")
    print("=" * 60)
    print(f"  Total questions:    {total}")
    print(f"  Judged:             {correct + wrong}")
    print(f"  Unjudged:           {unjudged}")
    print()
    if total > 0:
        print(f"  Overall accuracy:   {correct/total:.1%} ({correct}/{total})")
    print()
    print("  Per category:")
    for cat, stats in sorted(categories.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    Category {cat}:   {acc:.1%} ({stats['correct']}/{stats['total']})")
    print()
    print("  Per sample:")
    for sid, stats in sorted(samples.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {sid}:   {acc:.1%} ({stats['correct']}/{stats['total']})")
    print()
    print(f"  Token usage:")
    print(f"    Input:            {total_input:,}")
    print(f"    Output:           {total_output:,}")
    print(f"    Cache read:       {total_cache:,}")
    print(f"    Total:            {total_tokens:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark evaluation (OpenViking-compatible)")
    parser.add_argument("mode", choices=["ingest", "qa", "judge", "stat"],
                        help="Mode: ingest (load conversations), qa (run QA), judge (LLM scoring), stat (summary)")
    parser.add_argument("input", help="Path to LoCoMo JSON file (ingest/qa) or CSV file (judge/stat)")

    # Common options
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help=f"OpenClaw agent ID (default: {DEFAULT_AGENT})")
    parser.add_argument("--sample", type=int, default=None, help="Sample index (0-based), default: all")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per request (seconds)")

    # HTTP API mode (per-sample user isolation, matches OpenViking eval.py behavior)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenClaw gateway base URL for HTTP mode (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--token", default=os.environ.get("OPENCLAW_GATEWAY_TOKEN"),
                        help="Auth token for HTTP API mode. When set, uses HTTP API with per-sample user isolation. "
                             "Env: OPENCLAW_GATEWAY_TOKEN")
    parser.add_argument("--user", default=None,
                        help="User identity per sample. Default: eval-{sample_idx} (each sample gets unique user). "
                             "Set to a fixed value to share memory across samples.")

    # Ingest options
    parser.add_argument("--sessions", default=None, help="Session range e.g. '1-4'")
    parser.add_argument("--tail", default="[]", help="Tail message per session (default: '[]')")
    parser.add_argument("--force-ingest", action="store_true", help="Force re-ingest")
    parser.add_argument("--clear-ingest-record", action="store_true", help="Clear ingest record")

    # QA options
    parser.add_argument("--count", type=int, default=None, help="Number of QA questions per sample")
    parser.add_argument("--delay", type=float, default=0, help="Delay between QA questions (seconds, to avoid rate limits)")

    # Judge options
    parser.add_argument("--api-key", default=None, help="LLM API key for judging")
    parser.add_argument("--api-base", default="https://open.bigmodel.cn/api/coding/paas/v4",
                        help="LLM API base URL for judging")
    parser.add_argument("--judge-model", default="glm-4.7-flash",
                        help="Model for judging (default: glm-4.7-flash)")

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest(args)
    elif args.mode == "qa":
        run_qa(args)
    elif args.mode == "judge":
        run_judge(args)
    elif args.mode == "stat":
        run_stat(args)


if __name__ == "__main__":
    main()
