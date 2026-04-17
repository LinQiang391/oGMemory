<div align="center">

### ContextEngine: Context Lifecycle Engine <br> for CLI Agents

Across every phase of the Agent loop, manage what goes **in**, what **stays**, what gets **evicted**, and what can be **recalled**.
Seven context types. Three-level cache hierarchy. One virtual filesystem.
Cut token cost. Preserve signal density. Survive session boundaries.

English | [中文](readme_zh.md)
</div>

---

## The Problem: Token Budget is the Bottleneck

Every CLI Agent runs on a **fixed token budget**. The context window is both the most expensive and the most scarce resource.

| Symptom | Root Cause |
| -------- | --------- |
| Agent forgets earlier conversation | Context window fills up, old content is evicted with no persistence mechanism |
| Same mistakes repeated across sessions | No mechanism to carry lessons from one session to the next |
| A query that should cost $0.05 costs $0.50 | Flat retrieval loads entire documents when an abstract would suffice |
| Multi-agent coordination failures | Agents cannot see each other's working context |
| Context bloat in long sessions | No systematic compression — the Agent drowns in its own history |

This isn't a model problem. It's an **infrastructure problem**.

ContextEngine provides the missing infrastructure: a full-stack context lifecycle management system that intercepts the Agent loop at every stage, treating the context window as a **managed resource** — not an unbounded buffer.

---

## How It Works: The Agent Context Lifecycle

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                         Agent Loop (infinite)                         │
  │                                                                       │
  │   ┌─────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐         │
  │   │ ①   │     │    ②     │     │    ③     │     │   ④     │         │
  │   │ MSG │────▶│ REASON   │────▶│ TOOL     │────▶│ TURN    │         │
  │   │ IN  │     │ PREP     │     │ CALL     │     │ END     │         │
  │   └──┬──┘     └──────────┘     └──────────┘     └────┬────┘         │
  │      ▲                                              │               │
  │      │         ┌──────────┐         ┌─────────┐     │               │
  │      │         │    ⑤     │         │   ⑥     │     │               │
  │      └─────────│ COMPRESS │◀────────│ SHUTDOWN │◀────┘               │
  │                └──────────┘         └─────────┘                      │
  │                                                                       │
  └──────────────────────────────────────────────────────────────────────┘
```

ContextEngine intercepts the Agent loop at six stages. Each stage activates a set of **context operators** — functional units that read, write, transform, or evict context.

### Stage ①: message_received — Prefetch Context Before Inference

The message has arrived, but inference has not yet started. The system first determines what context the Agent will need.

| Operator | Responsibility |
| -------- | -------------- |
| **Intent Classifier** | Parse the input message, identify the query intent (fact recall / skill lookup / resource reference), and determine the retrieval strategy and budget priority |
| **Async Prefetch** | Embed the message and run an async similarity search against the L0 abstract layer, staging candidate results in the SessionTopicBuffer's pending slots. Target P99 < 20ms. On timeout, fail silently — never block model inference |

### Stage ②: bootstrap + ingest + assemble + before_prompt_build — Assemble the Context Window

The costliest stage. Assemble the optimal context window within the token budget.

**bootstrap** (once per session)
| Operator | Responsibility |
| -------- | -------------- |
| **Cold-start Injection** | On the first message of a session: load Profile, Handoff (if an active task exists), and relevant Preferences. Recall via L0 and inject into the system prompt by priority. Uses a dynamic token budget occupying 10–15% of the context window. Skipped for group / scheduled / btw sessions |
| **Session Restore** | Detect interrupted sessions via `/_session_buffers/`. Before entering the normal initialization flow, perform compensatory incremental extraction from the WriteLog to ensure continuity after restart |

**ingest / ingestBatch**
| Operator | Responsibility |
| -------- | -------------- |
| **Topic Tracking** | Maintain the SessionTopicBuffer per message: update the topic vector, record the set of injected URIs (for dedup), and compute relevance decay for each injected item. If the topic drifts for 5 consecutive turns, evict stale items from the injection set. In collect mode, `ingestBatch` jointly processes all queued messages |
| **Intent Tracking** | Update the intent classification in real time. When the intent changes (e.g., fact recall → multi-step task), partially reset the SessionTopicBuffer to prevent stale context from polluting the new intent |

**assemble**
| Operator | Responsibility |
| -------- | -------------- |
| **Budget Planner** | Read OpenClaw's `tokenBudget` parameter and compute the injectable budget: `min(fixed_cap, tokenBudget × (1 - (used/window)²))`. Also reserve `reserveTokensFloor` to avoid triggering compaction. Then allocate the budget across different context regions |
| **Relevance Ranker** | Score candidate context items: vector similarity × recency × `reference_count` × relation strength. Pass the ranked list to the assembler |
| **Context Assembly** | Load context tier by tier within the budget: L0 abstracts → L1 overviews → L2 full content (only when the budget allows). Exclude URIs already injected this session. Insert a synthetic assistant memory message before the last user message (temporary, not persisted). Before querying the persistent index, merge the session-local overlay to guarantee read-your-write consistency |
| **Dedup** | Compute pairwise cosine similarity between candidates. If cosine > 0.9, merge them and keep the one with the more complete abstract. Block re-injection of URIs already seen this session |

**before_prompt_build**
| Operator | Responsibility |
| -------- | -------------- |
| **Skill Injection** | Perform a narrow point-lookup on Skill nodes by `routing_key` prefix, bypassing vector recall. Inject best practices and known failure modes into the user message space via `prependContext`. This is separate from memory injection: the former targets capability, the latter targets personalization |
| **Relation Expansion** | Starting from L0 seed hits, fetch one-hop neighbors via a single JOIN SQL on `relation_edges`, populating `RetrievedBlock.relations` without per-file reads |

### Stage ③: before_tool_call + tool_result_persist — Context Processing Around Tool Execution

Context processing is triggered before and after each tool call.

**before_tool_call**
| Operator | Responsibility |
| -------- | -------------- |
| **Tool Context Injection** | Perform a point-lookup on the Skill node for the current tool (via NodeRegistry, < 30ms). Load known parameter combos, historical success/failure patterns, and usage tips, and append to `systemPromptAddition`. Content written by `afterTurn` in the same session is immediately visible via the session-local overlay |
| **Parameter Inference** | Derive reasonable parameter defaults from existing context — e.g., code style from Preference, known API constraints from Entity/Case — reducing parameter hallucination |
| **Failure Pattern Matching** | Match the current tool call against known failure patterns stored in Case context. On a match, inject a warning into `systemPromptAddition`. High-severity patterns can trigger interception |

**tool_result_persist**
| Operator | Responsibility |
| -------- | -------------- |
| **Result Compression** | When a tool result exceeds 1000 tokens, generate a lightweight summary, replace the original result in the transcript with the summary, and store the full content as a `tool_snapshot` node (TTL 24h, URI: `ctx://{account}/agents/{agent}/snapshots/{run_id}_{tool_name}`). `afterTurn` reads the snapshot for extraction rather than relying on a potentially trimmed transcript |
| **Fact Extraction** | Immediately extract reusable facts from tool results (e.g., "this endpoint rate-limits at 100 req/min") and persist them as Pattern/Entity nodes without waiting for `afterTurn`. Register Outbox events for async indexing |

### Stage ④: afterTurn — Extract Context from What Just Happened

The turn is complete. Extract new context from everything that just occurred — the primary write window.

| Operator | Responsibility |
| -------- | -------------- |
| **Incremental Extraction** | LLM-based extraction using a per-session watermark, processing only the delta since the last checkpoint. A single tool use call can output Profile / Preference / Entity / Event / Procedure candidates simultaneously. Read `tool_snapshot` nodes for complete tool results. Filter by a high confidence threshold before writing |
| **Relation Building** | Create edges between new and existing nodes. New Events point to their Entities (`related_to`); sequential Events within the same session get `SEQUENCE` edges (query predecessor via NodeRegistry, no extra LLM call). AggregateTopicPolicy merges also link to existing entities |
| **Conflict Resolution** | Detect contradictions between extracted and existing context, and resolve by type: Profile — new value wins; Preference — merge; Event — always append. Conflicts are written to the audit log |
| **Schema Validation** | Enforce structural invariants: URI format, required fields, and the presence of `account_id + owner_space` in every IndexRecord filter. Block writes on violation |
| **Async Index Build** | Schedule async indexing via the Outbox pattern (embed → upsert). Execute `INSERT outbox_events` with `FOR UPDATE SKIP LOCKED` to support multi-worker concurrent consumption; `pg_notify` wakes workers immediately on insert. This decouples write latency from Agent response time |
| **Tenant Isolation** | Verify that all output writes carry the correct `account_id + owner_space`. Block any write that could leak across account boundaries. All list operations pass through `_is_accessible` filtering |

### Stage ⑤: before_compaction + compact — Optimize Context

The context window is approaching capacity. Reduce token usage while preserving information density.

**before_compaction**
| Operator | Responsibility |
| -------- | -------------- |
| **Signal Scoring** | Score each context item: recency × `reference_count` × relation node degree × explicit user reference, and write the scores to `.meta.json`. This is the last window where tool results are still intact in the transcript |
| **Eviction Protection** | Mark the following as non-evictable: Profile, active Handoff, currently-referenced Entities. Pass the SessionWriteLog to the compression LLM with the instruction "the following content has already been persisted to long-term memory — do not summarize it again". Goal: maximize post-compression information density |

**compact**
| Operator | Responsibility |
| -------- | -------------- |
| **Redundancy Merger** | Identify near-duplicates (cosine > 0.9 or edit distance < 2) and merge by type policy: Preference/Entity use AggregateTopicPolicy; Profile uses merge |
| **Summary Chain** | Compress L2 full content → L1 overview → L0 abstract. Replace detailed entries with generated summaries. Protected nodes are excluded from compression. On `/compact {instructions}`, first perform targeted memory consolidation (e.g., organize "decisions" into Case, "open questions" into Handoff), then compress |
| **Cluster Compressor** | *(Phase 3)* Graph-traversal clustering based on `relation_edges`. Replace a cluster of related items with a single aggregated summary — e.g., compress 5 related Events into 1 Pattern. Depends on GaussVector's AI_aggregation |

### Stage ⑥: session_end + dispose — Persist, Archive, Audit

The session is ending. Persist, archive, audit.

**session_end**
| Operator | Responsibility |
| -------- | -------------- |
| **Task Archival** | Archive completed tasks as immutable Case entries with full execution traces. Future sessions can reference them as operational playbooks |
| **Task Snapshot** | Write the current task state snapshot (goals / completed / incomplete / next steps) to `ctx://{account}/agents/{agent}/memories/handoff/current`. This is a fixed URI, overwritten at the end of every session. When a task is marked complete, it is archived to a Case. Sub-agent Pattern / Case / Skill nodes are promoted to the parent Agent namespace; Event / Preference nodes are discarded |
| **State Versioning** | Checkpoint the working state — active tasks, in-progress relations, uncommitted writes — to support seamless session resumption |
| **Integrity Audit** | Scan all context nodes written during this session, check `.meta.json` status and file completeness, and enqueue PENDING / BROKEN nodes into the Repair Job queue. Audit results are written to the log |

**dispose**
| Operator | Responsibility |
| -------- | -------------- |
| **Buffer Flush** | On gateway shutdown or plugin reload, persist the SessionWriteLog to `/_session_buffers/{session_id}/write_log.json`. The SessionTopicBuffer is discarded since it is transient session-only state. On next bootstrap, if the file is detected, compensatory extraction is performed before normal initialization. Orphan buffer files are cleaned up by the Repair Job |

---

## The Abstraction: ContextFS

All context operators work through a single interface — **ContextFS**, a virtual filesystem that maps every context operation to a file operation on `ctx://` URIs.

```
    Context Operators (see lifecycle table above)
         │
         │  open · read · write · seek · link · unlink · gc
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │                    ContextFS (ctx://)                       │
  │                                                            │
  │  "Everything is a file" — Unix philosophy applied to       │
  │   Agent context management.                                │
  │                                                            │
  │  ctx://{account}/users/{user}/memories/profile              │
  │  ctx://{account}/agents/{agent}/skills/sql_gen              │
  │  ctx://{account}/users/{user}/memories/events/20250315_xxx  │
  │                                                            │
  ├──────────┬───────────┬─────────────┬───────────────────────┤
  │ Semantic │ Relation  │  Content    │  Metadata              │
  │ Layer    │ Layer     │  Layer      │  Layer                  │
  │          │           │             │                         │
  │ L0/L1/L2 │ .relations│  files      │  .meta.json             │
  │ vectors  │  .json    │  blobs      │  IndexRecord            │
  │ indexes  │  edges    │  versions   │  filters                │
  │          │  hops     │  snapshots  │  schema                 │
  ├──────────┼───────────┼─────────────┼───────────────────────┤
  │ InMemory │  .json    │  AGFS       │  AGFS                   │
  │ pgvector │  Neo4j    │  localfs    │  .meta.json             │
  │ ChromaDB │  Apache   │  s3fs       │  Validator              │
  │ Qdrant   │  Age      │  OSS        │                         │
  │ Weaviate │  (Phase 2)│  (Phase 2)  │                         │
  └──────────┴───────────┴─────────────┴───────────────────────┘
```

Why a filesystem?

| Filesystem Concept | Context Management Equivalent | Benefit |
| ------------------ | ---------------------------- | ------- |
| `open(path)` | Load context by URI | Uniform access to any context type |
| `read(fd, offset, len)` | Read L0 / L1 / L2 at chosen depth | Pay only for the tokens you need |
| `write(fd, data)` | Persist new context | Atomic writes with ordering guarantees |
| `link(src, dst)` | Create relation edge between contexts | Graph traversal without a separate API |
| `unlink(path)` | Delete / archive context | GC for stale context |
| `gc()` | Repair Job — reclaim broken/incomplete nodes | Self-healing persistence layer |
| Permissions (chmod) | Tenant isolation (account_id + owner_space) | Enforced at the filesystem level, not by callers |

The L0 → L1 → L2 hierarchy mirrors a CPU cache:

```
  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
  │  L0 Abstract │    │  L1 Overview │    │  L2 Content   │
  │  ~100 tokens │    │  ~500 tokens │    │  ~5000 tokens │
  │  Cheap scan  │    │  Informed    │    │  Full detail  │
  │  Always in   │    │  decision    │    │  On demand    │
  │  budget      │    │  layer       │    │  only         │
  └──────┬───────┘    └──────┬───────┘    └──────┬────────┘
         │                   │                    │
         ▼                   ▼                    ▼
    .abstract.md        .overview.md          content.md
```

**Token savings**: Find 3 relevant items from a corpus of 1000. Flat RAG loads 3 × 5000 = 15K tokens. ContextEngine loads 1000 × 100 (L0 scan) + 3 × 500 (L1 decision) + 1 × 5000 (L2 deep read) = 7.5K tokens. **50% savings on a conservative estimate; 60–80% in practice.**

---

## Context Types

Seven categories of context, each with a distinct lifecycle and write policy.

| Type | Lifecycle | URI Pattern | Write Policy | Example |
| ---- | --------- | ----------- | ------------ | ------- |
| **Profile** | Eternal — one per user/agent, continuously refined | `.../memories/profile` | Merge (new overwrites on conflict) | "Alice, backend engineer, London" |
| **Preference** | Long-lived — accumulates per topic | `.../memories/preferences/{slug}` | Aggregate by slug | "Prefers functional coding style" |
| **Entity** | Long-lived — accumulates per entity | `.../memories/entities/{slug}` | Aggregate by slug | "Project Alpha, launched 2024" |
| **Event** | Immutable — append-only timeline | `.../memories/events/{event_id}` | Append, never overwrite | "2025-03-15 completed API migration" |
| **Case** | Immutable — complete problem-solving trace | `.../memories/cases/{case_id}` | Append, never overwrite | "Redis timeout debug method" |
| **Pattern** | Evolving — emerges from repeated observations | `.../memories/patterns/{slug}` | Aggregate by slug | "User tends to ask in the afternoon" |
| **Skill** | Growing — cumulative tool expertise | `.../skills/{skill_name}` | Append to existing | "Best parameter combo for SQL gen" |

All seven share the `ctx://` URI namespace, atomic write ordering (content → relations → abstract → overview → meta), and multi-tenant isolation at the filesystem level.

---

## Architecture

```
         CLI Agent (Claude Code / OpenClaw / SDK)
              │
              │  HTTP REST (port 8090) or Python SDK
              ▼
    ┌─ HTTP Layer (server/) ─────────────────────┐
    │  Flask REST API  ·  Auth / RBAC             │
    │  Session Management  ·  Admin API           │
    └───────────────┬────────────────────────────┘
                    │
    ┌─ Service Layer (service/) ─────────────────┐
    │  MemoryWriteAPI   conversation → extraction │
    │  MemoryReadAPI    query → retrieval → context│
    └───────────────┬────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │  Write Path           │    Read Path (retrieval/)
        │  (extraction/ +       │    1. IntentClassifier
        │   commit/)            │    2. SeedRetriever (L0/L1/L2)
        │  1. Two-Phase Extract │    3. ResultRanker
        │  2. PolicyRouter      │    4. ContextReader
        │  3. MergePolicy       │
        │  4. ContextWriter     │
        └───────────┬───────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   ContextFS (fs/) — Virtual Filesystem      │
    │   Semantic · Relation · Content · Metadata  │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   Session (session/) · Lifecycle            │
    │   TopicBuffer · RollingCompressor           │
    │   AgingJob · AccessTracker                  │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   Async Index Layer (index/)                │
    │   OutboxWorker  ·  RepairJob                │
    └───────────────┬────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  Storage Backends   │
          │  AGFS · Vector ·   │
          │  Graph · Object    │
          └────────────────────┘
```

**Dev mode**: InMemory vectors + AGFS localfs — zero external dependencies, `pip install -e .` and go.
**Production**: pgvector/Qdrant + AGFS s3fs — horizontally scalable, close-to-open consistency.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Go 1.26+** (to compile AGFS)
- **Docker** (optional, for containerized deployment)

### Step 1: Install and Start AGFS

<details>
<summary><strong>Option 1: Official script (recommended)</strong></summary>

```bash
curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh
```

</details>

<details>
<summary><strong>Option 2: Build from source</strong></summary>

```bash
# Set Go proxy if dependency downloads time out
go env -w GOPROXY=https://goproxy.cn,direct

git clone https://github.com/c4pt0r/agfs.git /tmp/agfs-build
cd /tmp/agfs-build/agfs-server
go build -o ~/.local/bin/agfs-server .
```

</details>

Create config and start:

```bash
mkdir -p /tmp/agfs-data

cat > ~/.agfs-config.yaml << EOF
server:
  address: ":1833"
  log_level: "info"
plugins:
  localfs:
    enabled: true
    path: "/local"
    config:
      local_dir: "/tmp/agfs-data"
EOF

agfs-server -c ~/.agfs-config.yaml &
```

Verify AGFS is running:

```bash
curl http://localhost:1833/api/v1/health
# {"status":"healthy",...}
```

### Step 2: Install ContextEngine

```bash
git clone https://gitcode.com/opengauss/oG-Memory.git
cd oG-Memory

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
# or with uv: uv pip install -e .
```

### Step 3: Use It

<details>
<summary><strong>Option A: HTTP Server (recommended)</strong></summary>

```bash
# Configure runtime (edit ogmem.yaml for LLM provider, etc.)
cp ogmem.yaml.example ogmem.yaml

# Start the HTTP server (default port 8090)
python server/app.py

# Ingest a conversation turn
curl -X POST http://localhost:8090/api/v1/after_turn \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user-1",
    "sessionId": "session-1",
    "messages": [
      {"role": "user", "content": "I am Alice, a backend engineer based in London"},
      {"role": "assistant", "content": "Nice to meet you, Alice!"}
    ]
  }'

# Search memory for the current turn
curl -X POST http://localhost:8090/api/v1/compose \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user-1",
    "sessionId": "session-2",
    "query": "what is alice job"
  }'

# Check health
curl http://localhost:8090/api/v1/health
```

</details>

<details>
<summary><strong>Option B: Docker</strong></summary>

```bash
docker compose up
# Server available on port 8090, AGFS on port 1833
```

</details>

<details>
<summary><strong>Option C: Python SDK</strong></summary>

```python
from service.api import MemoryWriteAPI
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.config import ProviderConfig

# 1. Wire up storage
config = ProviderConfig.from_env()
agfs = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=agfs, mount_prefix="/local")

# 2. Create write API
write_api = MemoryWriteAPI(fs=fs, llm=config.create_llm())

# 3. Define context
ctx = RequestContext(
    account_id="my-account",
    user_id="user-123",
    agent_id="agent-001",
    session_id="session-001",
    trace_id="trace-001",
)

# 4. Write context from a conversation turn
messages = [
    {"role": "user", "content": "I'm Alice, backend engineer, based in London"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
]
result = write_api.commit_session(messages, ctx)
print(f"Persisted {result['writes_completed']} context entries")

# 5. Read context back (see docs/QUICKSTART.md for full pipeline)
```

</details>

**HTTP API Reference** (key endpoints):

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/v1/compose` | POST | Search memory, return context for current turn |
| `/api/v1/after_turn` | POST | Extract + persist memories from conversation |
| `/api/v1/ingest` | POST | Single message ingest |
| `/api/v1/ingest_batch` | POST | Batch message ingest |
| `/api/v1/compact` | POST | Trigger context compression |
| `/api/v1/bootstrap` | POST | Cold-start session with profile + preferences |
| `/api/v1/sessions/{id}/messages` | POST | Add message to session buffer |
| `/api/v1/sessions/{id}/commit` | POST | Commit session: archive + extract |
| `/api/v1/sessions/{id}` | GET | Get session state |
| `/api/v1/sessions/{id}/context` | GET | Get assembled session context |
| `/api/v1/health` | GET | Health check (AGFS + LLM + vector DB) |
| `/api/v1/admin/*` | Various | Tenant admin (accounts, users, roles, agents, audit) |

---

## OpenClaw Integration

Drop-in context engine for OpenClaw:

```bash
cd openclaw_context_engine_plugin
openclaw plugins install -l .
```

```
OpenClaw Agent
    ↓ (hook: on_turn_end)
og_memory_write (TypeScript)
    ↓ (JSON stdin/out via child_process)
write_memory.py (Python Bridge)
    ↓ (MemoryWriteAPI → ContextFS)
$AGFS_DATA_DIR/accounts/...
```

**Agent tools** (mapped to lifecycle stages):

| Tool | Lifecycle Stage | Context Operation |
| ---- | --------------- | ----------------- |
| `og_memory_write` | ④ Turn End | Extract + persist new context |
| `og_memory_search` | ① Message | Prefetch relevant context |
| `og_memory_read` | ② Reasoning Prep | Load full context by URI |

**Automatic behaviors** (no Agent code changes):

| Lifecycle Event | Hook | ContextEngine Action |
| --------------- | ---- | -------------------- |
| New message arrives | `on_message` | Analyze intent → prefetch → inject context |
| Agent completes turn | `on_turn_end` | Extract delta → resolve conflicts → persist + index |
| Context window fills | `on_context_compress` | Score salience → protect critical → compress rest |
| Session ends | `on_session_close` | Archive tasks → checkpoint state → audit integrity |

---

## Roadmap

### Phase 0 — Core Freeze (done)

- Schema, core models, Protocol interfaces
- ContextFS / AGFS adapter basics
- Atomic write ordering + contract tests

### Phase 1 — Extraction + Vector Retrieval (substantially complete)

**Lifecycle stages:** ① message_received → ② bootstrap + ingest + assemble → ④ afterTurn

- **Two-phase extraction**: Phase 1 identifies extractable message spans (cheap JSON-mode LLM); Phase 2 runs focused tool-call extraction per span with surrounding context, parallelized across spans
- **YAML-driven SchemaRegistry**: Dynamic tool generation from YAML schemas replacing hardcoded definitions, with `PolicyRouter` for schema-driven policy routing and `URIResolver` for centralized URI generation
- **Dual extraction modes**: *Eager* mode (prefetch existing memories via vector search, then single-shot extraction); *Lazy* mode (ReAct loop with read/list tools for multi-turn extraction, fallback to eager on failure)
- **Write path**: MergePolicy, atomic writes (content → relations → abstract → overview → meta), Outbox async indexing
- **Hierarchical retrieval**: IntentClassifier → SeedRetriever (L0/L1/L2 vector recall) → ResultRanker (relevance scoring) → ContextReader (full content loading)
- **Service layer**: MemoryWriteAPI, MemoryReadAPI with multi-tenant isolation
- **HTTP REST API**: Flask server with full session management, multi-tenant RBAC (role-based access control), and admin endpoints for tenant/user/agent management
- **Session management**: TopicBuffer, RollingCompressor, SessionManager — stateful session lifecycle with message buffering and commit
- **Lifecycle management**: AgingJob for memory lifecycle, AccessTracker for usage metrics
- **Benchmark**: LoCoMo long-context conversational memory benchmark — 78.1% accuracy on 1343 questions across 9 conversations (4 categories: single-turn facts, cross-session temporal, multi-hop inference, preferences)

Operators unlocked: Intent Classifier, Async Prefetch, Cold-start Loader, Budget Planner, Relevance Ranker, Context Assembly, Dedup, Two-phase Extractor, Relation Builder, Conflict Detector, Index Dispatcher, Tenant Enforcer, Integrity Auditor, Schema Router, URI Resolver, Session Manager, Rolling Compressor

### Phase 2 — Hybrid Retrieval + Compression

**Lifecycle stages:** ③ before_tool_call + tool_result_persist → ⑤ compression → ⑥ session_end

- **Retrieval enhancement**: QueryPlanner dynamic hierarchical retrieval, explicit L0/L1/L2 budget trimming
- **Graph retrieval**: RelationStore 1-hop/multi-hop expansion, working set + graph traversal fusion
- **Tool context**: before_tool_call knowledge injection + tool_result_persist result distillation
- **Compression pipeline**: before_compaction decision layer (protected/evictable sets) + compact execution (summary replacement, archive generation)
- **Session restore**: session_end archival + dispose buffer flush + bootstrap checkpoint recovery
- **AI_Functions integration**: ai_chunking / ai_filter (GaussVector)

Operators unlocked: Session Restorer, Skill Injector, Relation Expander, Tool Context Injector, Parameter Inference, Failure Pattern Matcher, Result Compression, Fact Extractor, Signal Scorer, Eviction Protector, Redundancy Merger, Summary Chain, Task Archiver, Task Snapshot, State Versioning, Buffer Flush

### Phase 3 — Aggregation + Adaptive Planning

**Lifecycle stages:** Full adaptive loop

- **Graph clustering**: Cluster compression via relation_edges (e.g., 5 related Events → 1 Pattern)
- **AI_aggregation**: GaussVector cluster-level aggregated summaries
- **Dynamic retrieval planning**: Adaptive strategy based on task complexity
- **Sub-agent coordination**: prepareSubagentSpawn / onSubagentEnded semantics
- **Observability**: Context lifecycle tracing / metrics / audit — quantifying cost, recall gain, compression gain

Operators unlocked: Cluster Compressor, Retrieval Planner

## Testing

```bash
pytest tests/contract/ -v          # Must pass — core invariants
pytest tests/ -v                   # All tests
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

## Benchmark: LoCoMo

ContextEngine is evaluated on the [LoCoMo](https://arxiv.org/abs/2402.10790) long-context conversational memory benchmark — 10 conversations with ~1500 questions across 4 categories:

| Category | Description | Example |
| -------- | ----------- | ------- |
| cat1 | Single-turn fact recall | "What is Alice's job?" |
| cat2 | Cross-session temporal reasoning | "When did Alice move to London?" |
| cat3 | Multi-hop inference across topics | "Which project started first?" |
| cat4 | User preferences and habits | "What coding style does Alice prefer?" |

**Result: 78.1% accuracy on 1343 questions** (judged by gpt-4o-mini, validated against claude-sonnet-4 with 100% agreement on a 100-question sample).

Evaluation pipeline:

```bash
cd tests/e2e

# 1. Ingest conversations (build memory)
uv run python eval.py ingest data/locomo10.json --sample 0

# 2. Run QA against stored memories
uv run python eval.py qa data/locomo10.json --sample 0

# 3. Judge answers with LLM
uv run python eval.py judge data/locomo10.json

# 4. Generate statistics
uv run python eval.py stat data/locomo10.json
```

## Documentation

| Document | Description |
| -------- | ----------- |
| [CLAUDE.md](CLAUDE.md) | Full technical spec (architecture, interfaces, data models, team protocols) |
| [ENV.md](ENV.md) | Environment setup (AGFS + Python + env vars) |
| [ogmem.yaml](ogmem.yaml) | Runtime configuration (LLM provider, ports, storage) |
| [Quickstart](docs/QUICKSTART.md) | End-to-end usage guide |
| [Deployment](docs/deployment/deployment_operations_guide.md) | Production deployment |
| [OpenClaw Plugin](openclaw_context_engine_plugin/README.md) | Integration setup |
| [Benchmark](tests/e2e/) | LoCoMo evaluation framework (ingest / qa / judge / stat) |

## References

### Benchmark & Evaluation
- [LoCoMo: Evaluating Long-Context Conversational Memory](https://arxiv.org/abs/2402.10790) — Maharana et al., 2024 — long-context conversational memory benchmark used in our evaluation

### Memory Architectures for LLM Agents
- [MemoryBank: Enhancing LLMs with Long-Term Memory](https://arxiv.org/abs/2305.10250) — Zhong et al., 2023 — sentiment-centric memory processing for persistent conversational memory
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) — 2025 — dynamic, agentic memory organization with dynamic indexing
- [Mem0: Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — 2025 — user/session/agent scoped memory with vector store + knowledge graph
- [SeCom: Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/abs/2502.05589) — 2025 — segment-level memory bank with conversation segmentation (related to our two-phase extraction)

### Hierarchical Retrieval & Multi-Granularity
- [ReadAgent: Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) — Lee et al., 2024 — human-inspired reading agent with gist memory, 20x effective context length (related to our L0/L1/L2 hierarchy)
- [MemoRAG: Memory-Augmented Retrieval](https://arxiv.org/abs/2409.05591) — Qian et al., 2024 — dual-system architecture with global memory + retrieval-augmented generation
- [LATTICE: LLM-guided Hierarchical Retrieval](https://arxiv.org/abs/2510.13217) — Gupta et al., 2025 — hierarchical retrieval framework for large document collections

### Context Compression
- [LLMLingua-2: Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) — Pan et al., 2024 — data distillation for efficient prompt compression up to 20x
- [LongLLMLingua: Question-Aware Compression for Long Context](https://aclanthology.org/2024.acl-long.91.pdf) — Jiang et al., 2024, ACL — coarse-to-fine compression with 17.1% improvement at 4x compression (related to our compression pipeline design)
- [Prompt Compression for Large Language Models: A Survey](https://arxiv.org/abs/2410.12388) — 2024 — comprehensive survey of prompt and context compression techniques

### Reasoning & Acting
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., 2023, ICLR — interleaved reasoning traces and task-specific actions (our lazy extraction mode is based on this paradigm)

### Multi-Agent & Governance
- [Governed Memory](https://arxiv.org/abs/2603.17787) — Taheri, 2026 — memory governance and access control
- [Collaborative Memory](https://arxiv.org/abs/2505.18279) — multi-user memory sharing + dynamic ACL (related to our multi-tenant isolation)
- [Multi-Agent Memory Systems for Production](https://mem0.ai/blog/multi-agent-memory-systems) — Mem0, 2026
- [Cemri et al.](https://arxiv.org/abs/2503.13657) — multi-agent coordination failure analysis

### Infrastructure
- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS file storage layer
- [AI Agent Memory Architectures](https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical) — Zylos Research, 2026

## Acknowledgements

- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS file storage layer and core design inspiration

## License

[Apache License 2.0](LICENSE)
