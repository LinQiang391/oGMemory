# Performance Recording Module — Quick Start

The `perf/` module captures **wall time, token usage, model attribution, and $-cost** across the six ContextEngine lifecycle stages. It is completely opt-in — zero overhead when disabled.

## Overview

| File | Purpose |
|------|---------|
| `recorder.py` | Singleton `Recorder` — coordinates all perf events, owns the sink |
| `decorators.py` | `@record_stage(name)` decorator and `span(name)` context manager |
| `sinks.py` | `MemorySink` (in-process), `JsonlSink` (file), `HttpSink` (remote) |
| `rate_cards.py` | USD/token rate calculation |
| `rate_cards.yaml` | Default rate card (GPT-4o-mini, GPT-4o, etc.) |
| `token_attribution.py` | Token snapshot diffing |
| `span_stack.py` | Async-safe ContextVar-based span stack |
| `report.py` | Markdown report renderer from JSONL trace |

## Enabling Performance Recording

### Option 1: Environment Variables

```bash
OGMEM_PERF_ENABLED=1 python your_script.py
```

Optional overrides:
```bash
OGMEM_PERF_ENABLED=1 \
OGMEM_PERF_RUN_ID=my-run \
OGMEM_PERF_OUT=./perf_logs/my-run.jsonl \
OGMEM_PERF_HTTP_URL=http://localhost:8090/api/v1/perf/events \
OGMEM_PERF_RATE_CARD=./perf/rate_cards.yaml \
python your_script.py
```

### Option 2: Programmatic

```python
from perf import get_recorder
from perf.sinks import JsonlSink, HttpSink, CompositeSink

recorder = get_recorder()
recorder.enable(
    sink=CompositeSink([
        JsonlSink("./perf_logs/run.jsonl"),
        HttpSink("http://ogmemory-container:8090/api/v1/perf/events"),
    ]),
    run_id="my-run",
    rate_card_path="./perf/rate_cards.yaml",
)
```

## Instrumenting Code

### `@record_stage` — Decorate Lifecycle Methods

```python
from perf import record_stage

class MemoryService:
    @record_stage("after_turn")
    def after_turn(self, params: dict) -> dict:
        # span events captured automatically
        ...

    @record_stage("compose")
    def compose(self, params: dict) -> dict:
        ...
```

### `span()` — Sub-Spans Inside a Stage

```python
from perf import span

@record_stage("after_turn")
def after_turn(self, params):
    with span("extract_llm", num_messages=len(msgs)) as s:
        candidates = extractor.extract(...)
        s.meta["num_candidates"] = len(candidates)

    with span("write_to_vector_db"):
        write_api.upsert(...)
```

### Attaching the Service (Required for Token Tracking)

```python
from perf import get_recorder

service = MemoryService(config)
get_recorder().attach(service)   # enables token snapshot diffing
```

## Running with the e2e Test Script

```bash
# Local mode (loads bridge module directly)
OGMEM_PERF_ENABLED=1 PYTHONPATH=. \
  python scripts/run_e2e_long_context_real.py \
    --record-perf \
    --fixture tests/e2e/data/locomo10.json \
    --sample 0 --max-sessions 2

# Docker mode (calls ogmem container HTTP API)
OGMEM_PERF_ENABLED=1 PYTHONPATH=. \
  python scripts/run_e2e_long_context_real.py \
    --record-perf --docker-mode \
    --ogmem-url http://127.0.0.1:8090 \
    --fixture tests/e2e/data/locomo10.json \
    --sample 0 --max-sessions 2
```

Output:
- JSONL trace: `perf_logs/{run_id}.jsonl`
- Markdown report: `perf_logs/{run_id}.md`

## Token Attribution in Per-Stage Context (Critical)

**The `@record_stage` decorator is the authoritative source for token tracking when it is present in the image.** It uses `TokenTracker.snapshot()` (non-destructive read) at entry and exit of each decorated method to compute per-stage token deltas automatically.

When the image **does NOT** have `@record_stage` (current deployed containers), token tracking must be done client-side via the `token_stats` API using the **POST reset-before/reset-after pattern**:

```python
def measure_stage_tokens(api_call_fn):
    # Reset counter before
    http_post("/token_stats", {"reset": True})
    result = api_call_fn()
    # Reset counter after → delta since last reset
    after = http_post("/token_stats", {"reset": True})
    return {
        "llm_input": after["llm"]["input_tokens"],
        "llm_output": after["llm"]["output_tokens"],
        "embed_tokens": after["embedding"]["total_tokens"],
        "status": result.get("status"),
    }
```

**Why POST reset, not GET?**
- `GET /token_stats` → calls `TokenTracker.snapshot()` (non-destructive) — accumulates globally
- During multi-LLM-call stages (e.g., `compose` with profile + archives + session state), internal operations can cause the provider's token counter to update in ways that cause the *same counter value* to be subtracted multiple times when measured with GET before/after
- `POST /token_stats {"reset": true}` → calls `snapshot_and_reset()` (destructive), giving a clean delta since the last reset

**Example usage in test scripts:**

```python
def run_session_lifecycle(session_id):
    stages = {}

    # Bootstrap (no LLM, just AGFS init)
    stages["bootstrap"] = measure_stage_tokens(
        lambda: http_post("/bootstrap", {"account_id": "acct-demo", ...})
    )

    # After turn
    stages["after_turn"] = measure_stage_tokens(
        lambda: http_post("/after_turn", {"messages": [...], ...})
    )

    # Compose (may call LLM multiple times internally)
    stages["compose"] = measure_stage_tokens(
        lambda: http_post("/compose", {"query": "summary"}, ...)
    )

    # Compact
    stages["compact"] = measure_stage_tokens(
        lambda: http_post("/compact", {...})
    )

    # Dispose
    stages["dispose"] = measure_stage_tokens(
        lambda: http_post("/dispose", {...})
    )

    return stages
```

## Generating a Report from an Existing JSONL

```bash
python -m perf.report perf_logs/my-run.jsonl -o perf_logs/my-run.md
```

Or programmatically:

```python
from perf.report import load_events, render

events = load_events("perf_logs/my-run.jsonl")
md = render(events)
print(md)
```

## Report Sections

1. **Stage Totals** — wall time, tokens, $-cost per lifecycle stage
2. **Sub-span Breakdown** — detailed timing inside each stage
3. **Model Breakdown** — total tokens + cost per LLM/embedding model
4. **Bottleneck Ranking** — top 10 slowest sub-spans
5. **Counters Appendix** — numeric metadata from `span.meta`

## Supported Sinks

| Sink | Use Case |
|------|----------|
| `JsonlSink` | Local file persistence (default) |
| `HttpSink` | Stream events to a remote endpoint (e.g. container's `/api/v1/perf/events`) |
| `MemorySink` | In-process collection for tests |
| `CompositeSink` | Fan-out to multiple sinks simultaneously |

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OGMEM_PERF_ENABLED` | Enable recording (`1`/`true`/`yes`) | — |
| `OGMEM_PERF_RUN_ID` | Identifier for this run | auto-generated UUID |
| `OGMEM_PERF_OUT` | JSONL output file path | `perf_logs/{run_id}.jsonl` |
| `OGMEM_PERF_HTTP_URL` | Remote HTTP endpoint for `HttpSink` | — |
| `OGMEM_PERF_RATE_CARD` | Path to rate card YAML | `perf/rate_cards.yaml` |