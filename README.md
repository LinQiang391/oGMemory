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

All 27 context operators work through a single interface — **ContextFS**, a virtual filesystem that maps every context operation to a file operation on `ctx://` URIs.

```
    27 Context Operators (see lifecycle table above)
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
              │  6 lifecycle hooks
              ▼
    ┌─ Service Layer (service/) ──────────────────┐
    │  MemoryWriteAPI   conversation → extraction  │
    │  MemoryReadAPI    query → retrieval → context │
    │  MemoryFS         ctx:// URI browsing         │
    └───────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │  Write Path (commit/) │    Read Path (retrieval/)
        │  1. Extract (LLM)     │    1. Query Analyzer
        │  2. MergePolicy       │    2. Prefetch + Expand
        │  3. Relation Build    │    3. Hierarchical Load
        │  4. Atomic Write      │    4. Context Assembly
        └───────────┬───────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   ContextFS (fs/) — Virtual Filesystem      │
    │   Semantic · Relation · Content · Metadata  │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   Async Index Layer (index/)                │
    │   OutboxWorker  ·  RepairJob                │
    │   (index sync)     (node recovery)          │
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
```

### Step 3: Use It

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

### Phase 1 — Extraction + Vector Retrieval (in progress)

**Lifecycle stages:** ① message_received → ② bootstrap + ingest + assemble → ④ afterTurn

- **Extraction pipeline**: CandidateExtractor with when/who/where context, session_time absolute date resolution
- **Write path**: MergePolicy, atomic writes (content → relations → abstract → overview → meta), Outbox async indexing
- **Hierarchical retrieval**: L0/L1/L2 vector recall + budget planning + relevance ranking
- **Service layer**: MemoryWriteAPI, MemoryReadAPI
- **Interface convergence**: Unify compose/assemble naming, clarify bootstrap / ingest / assemble / afterTurn boundaries

Operators unlocked: Intent Classifier, Async Prefetch, Cold-start Loader, Budget Planner, Relevance Ranker, Context Assembly, Dedup, Delta Extractor, Relation Builder, Conflict Detector, Index Dispatcher, Tenant Enforcer, Integrity Auditor

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

## Documentation

| Document | Description |
| -------- | ----------- |
| [CLAUDE.md](CLAUDE.md) | Full technical spec (architecture, interfaces, data models, team protocols) |
| [ENV.md](ENV.md) | Environment setup (AGFS + Python + env vars) |
| [Quickstart](docs/QUICKSTART.md) | End-to-end usage guide |
| [Deployment](docs/deployment/deployment_operations_guide.md) | Production deployment |
| [OpenClaw Plugin](openclaw_context_engine_plugin/README.md) | Integration setup |

## References

- [AI Agent Memory Architectures](https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical) — Zylos Research, 2026
- [Multi-Agent Memory Systems for Production](https://mem0.ai/blog/multi-agent-memory-systems) — Mem0, 2026
- [Governed Memory](https://arxiv.org/abs/2603.17787) — Taheri, 2026
- [Collaborative Memory](https://arxiv.org/abs/2505.18279) — multi-user memory sharing + dynamic ACL
- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS file storage layer
- [Cemri et al.](https://arxiv.org/abs/2503.13657) — multi-agent coordination failure analysis

## Acknowledgements

- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS file storage layer and core design inspiration

## License

[Apache License 2.0](LICENSE)
