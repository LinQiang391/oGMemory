<div align="center">

### ContextEngine: Context Lifecycle Engine <br> for CLI Agents

Manage what goes **in**, what **stays**, what goes **out**, and what **comes back** —
across every phase of the Agent loop.
Seven context types, three-level cache hierarchy, one virtual filesystem.

Slash token cost. Preserve signal. Survive sessions.

English | [中文](README_zh.md)
</div>

---

## The Problem: Token Budget is the Bottleneck

Every CLI Agent runs on a **fixed token budget**. The context window is both the most expensive and the most scarce resource in an Agent system.

| Symptom | Root Cause |
|---------|-----------|
| Agent forgets earlier conversation | Context window fills → old context evicted with no persistence |
| Same mistakes across sessions | No mechanism to carry lessons from one session to the next |
| $0.50 queries that should cost $0.05 | Flat retrieval loads entire documents when an abstract would suffice |
| Multi-agent coordination failures | Agents can't see each other's working context |
| Context bloat after long sessions | No systematic compression — Agent drowns in its own history |

This isn't a model problem. It's an **infrastructure problem**.

ContextEngine provides the missing infrastructure: a full-stack context lifecycle management system that hooks into the Agent loop at every stage, treating the context window as a **managed resource** — not an unbounded buffer.

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

ContextEngine intercepts the Agent loop at six stages. Each stage has a hook that activates a set of **context operators** — functional units that read, write, transform, or evict context.

### ① `on_message` — Context Prefetch

The message arrives. Before any inference, determine what context the Agent will need.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Query Analyzer** | Parse incoming message → classify intent (recall / skill / reference) → determine retrieval strategy | 1 |
| **Prefetch Engine** | Embed the message, run L0 similarity search against context abstracts, pre-load top candidates into a staging buffer | 1 |
| **Cold-start Loader** | First message of a new session → load user Profile + recent Preferences + last-session summary from persistent store | 1 |
| **Session Restorer** | Resume interrupted session → reconstruct working context from persisted state checkpoint | 2 |

### ② `on_before_inference` — Context Assembly

Compose the optimal context window within the token budget. This is the most critical stage for cost and quality.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Budget Planner** | Allocate token budget across context categories based on task type: 40% retrieved knowledge, 30% conversation history, 20% tool context, 10% system | 2 |
| **Relevance Ranker** | Multi-signal scoring — vector similarity × recency × access frequency × relation strength → ranked list | 1 |
| **Context Composer** | Assemble context window in priority order: L0 abstracts → L1 overviews → L2 full content (load deepest only when budget allows) | 1 |
| **Skill Locator** | Anticipate tool usage from intent → locate and load relevant Skill context (best practices, known failure modes) | 1 |
| **Graph Traverser** | Starting from seed hits, follow relation edges (`related_to`, `derived_from`, `SEQUENCE`) to pull in connected context beyond initial vector recall | 2 |
| **De-duplicator** | Remove overlapping context entries; merge near-duplicates using structural similarity (edit distance < 2 or embedding cosine > 0.9) | 1 |

### ③ `on_tool_call` / `on_tool_result` — Context Enrichment

Before and after each tool invocation within a turn. Inject tool-specific knowledge; capture what the tool reveals.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Tool Knowledge Loader** | Load Skill context for the specific tool being called — known parameter combos, historical success/failure rates, usage tips | 1 |
| **Parameter Contextualizer** | Inject context-derived defaults into tool parameters (e.g., preferred coding style from Preference, known API constraints from Case) | 1 |
| **Safety Checker** | Match current tool call against known failure patterns stored in Case context → warn or block if pattern matches | 1 |
| **Result Condenser** | After tool execution → distill output into an L0 abstract + optional L1 overview for future retrieval, keeping full result in L2 | 1 |
| **Fact Extractor** | Capture reusable knowledge from tool results (e.g., "this endpoint rate-limits at 100 req/min") for persistence as Pattern/Entity context | 2 |

### ④ `on_turn_end` — Context Extraction

The turn is done. Extract new context from what just happened — the primary write window.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Delta Extractor** | LLM-based extraction: new facts, entities, preferences, events from the just-completed turn — only what changed, not the full conversation | 1 |
| **Relation Builder** | Create edges between new context nodes and existing ones (link new Event to its Entity, new Preference to its Profile) | 1 |
| **Conflict Detector** | Detect contradictions between extracted and existing context → resolve by type-specific policy (Profile: new overwrites; Preference: merge; Event: always append) | 1 |
| **Schema Validator** | Enforce structural invariants: URI format, required fields, `account_id` + `owner_space` in every filter — blocks write on violation | 1 |
| **Index Dispatcher** | Queue async index updates (embed → upsert) via Outbox pattern — decouples write latency from Agent response time | 1 |
| **Tenant Enforcer** | Verify all outbound context carries correct tenant isolation tags; block any write that would leak across account boundaries | 1 |

### ⑤ `on_context_compress` — Context Optimization

The context window is approaching capacity. Reduce token usage while preserving signal.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Signal Estimator** | Score every context entry by information density: recency × frequency × relation density × explicit user reference → salience score | 2 |
| **Priority Protector** | Mark critical context as non-evictable: Profile, currently-referenced entities, active task Case — survives any compression | 2 |
| **Redundancy Merger** | Identify duplicate or near-identical entries → merge using type-appropriate policy (AggregateTopic for preferences, merge for profile) | 2 |
| **Summarization Engine** | Compress L2 full content → L1 overview → L0 abstract chain; replace detailed entries with generated summaries | 2 |
| **Cluster Compressor** | Graph-traverse related context → cluster → replace individual entries with aggregated group summaries (e.g., 5 related events → 1 pattern) | 3 |
| **Retrieval Planner** | For complex tasks, dynamically plan the retrieval strategy for the next cycle: which levels to load, which categories to search, how many graph hops to traverse | 3 |

### ⑥ `on_session_close` — Context Persistence

The session is ending. Persist, archive, audit.

| Context Operator | Responsibility | Phase |
|-----------------|----------------|-------|
| **Task Finalizer** | Archive completed task context as Case entries with full problem-solving traces — reusable playbooks for future sessions | 1 |
| **State Persister** | Checkpoint working state: active tasks, in-progress relations, uncommitted writes → enables seamless session resumption | 1 |
| **Session Digest** | Generate session-level summary (topics, decisions, skills used) → index as Event for cross-session recall | 2 |
| **Integrity Auditor** | Scan all context nodes touched this session: `.meta.json` status, file completeness → trigger Repair Job for any PENDING/BROKEN nodes | 1 |
| **Schema Auditor** | Final pass — validate that all context written during the session conforms to the frozen core schema; log warnings for any drift | 2 |

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
|-------------------|------------------------------|---------|
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

**Token savings**: A query that needs 3 relevant items from a corpus of 1000. Flat RAG loads 3 × 5000 = 15K tokens. ContextEngine loads 1000 × 100 (L0 scan) + 3 × 500 (L1 decision) + 1 × 5000 (L2 deep read) = 7.5K tokens. **50% savings on a conservative estimate; 60–80% in practice.**

---

## Context Types

Seven categories of context, each with a distinct lifecycle and write policy.

| Type | Lifecycle | URI Pattern | Write Policy | Example |
|------|-----------|-------------|--------------|---------|
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

Drop-in context engine for [OpenClaw](https://docs.openclaw.ai):

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
|------|----------------|-------------------|
| `og_memory_write` | ④ Turn End | Extract + persist new context |
| `og_memory_search` | ① Message | Prefetch relevant context |
| `og_memory_read` | ② Reasoning Prep | Load full context by URI |

**Automatic behaviors** (no Agent code changes):

| Lifecycle Event | Hook | ContextEngine Action |
|----------------|------|---------------------|
| New message arrives | `on_message` | Analyze intent → prefetch → inject context |
| Agent completes turn | `on_turn_end` | Extract delta → resolve conflicts → persist + index |
| Context window fills | `on_context_compress` | Score salience → protect critical → compress rest |
| Session ends | `on_session_close` | Archive tasks → checkpoint state → audit integrity |

---

## Roadmap

| Phase | Focus | Key Deliverables | Context Operators Unlocked |
|-------|-------|-----------------|---------------------------|
| **Phase 0** (done) | Core freeze | Schema + Protocol + AGFS spec + contract tests | Schema Validator |
| **Phase 1** | Extraction + vector retrieval | Write path + L0/L1/L2 retrieval + service layer | Query Analyzer, Prefetch Engine, Cold-start Loader, Context Composer, Relevance Ranker, Delta Extractor, Relation Builder, Conflict Detector, Index Dispatcher, Integrity Auditor |
| **Phase 2** | Hybrid retrieval + compression | RelationStore graph traversal + AI_Functions + context compression | Session Restorer, Budget Planner, Graph Traverser, Tool Knowledge Loader, Signal Estimator, Priority Protector, Redundancy Merger, Summarization Engine, Session Digest, Schema Auditor |
| **Phase 3** | Aggregation + adaptive planning | Graph clustering + AI_aggregation + dynamic query planning | Cluster Compressor, Retrieval Planner, Fact Extractor |

---

## Testing

```bash
pytest tests/contract/ -v          # Must pass — core invariants
pytest tests/ -v                   # All tests
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

## Documentation

| Document | Description |
|----------|-------------|
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

## License

[Apache License 2.0](LICENSE)
