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

| Symptom | Root Cause | Missing Lifecycle Stage |
| -------- | --------- | ----------------------- |
| Agent forgets earlier conversation | Context window fills up, old content is evicted with no persistence | ④ **afterTurn** — no extraction + persist mechanism |
| Same mistakes repeated across sessions | No mechanism to carry lessons from one session to the next | ⑥ **session_end** — no archival + ② **bootstrap** — no cold-start injection |
| A query that should cost $0.05 costs $0.50 | Flat retrieval loads entire documents when an abstract would suffice | ② **assemble** — no L0/L1/L2 hierarchical retrieval |
| Multi-agent coordination failures | Agents cannot see each other's working context | ④ **afterTurn** — no cross-session sharing + multi-tenant isolation |
| Context bloat in long sessions | No systematic compression — the Agent drowns in its own history | ⑤ **compact** — no signal scoring + summary chain |

This isn't a model problem. It's an **infrastructure problem**.

ContextEngine provides the missing infrastructure: a full-stack context lifecycle management system that intercepts the Agent loop at six defined stages, treating the context window as a **managed resource** — not an unbounded buffer.

---

## Design Philosophy

### Core Insight: Context Has a Lifecycle

Current RAG systems treat retrieval as a single operation — embed a query, search a vector store, return results. This misses the fundamental reality: **context in an Agent system has a complete lifecycle**, just like data in a database.

```
  Born          Structured       Stored           Indexed          Recalled        Compressed       Archived
  (extracted    (categorized     (written         (embedded +      (vector         (summarized,     (session end,
   from          by type,        atomically       upserted         search,         deduplicated,    archived,
   conversation) routed by       with ordering    to L0/L1/L2      hierarchical    compressed)      checkpointed)
                  policy)         guarantees)      IndexRecords)    expansion)
```

Each phase has different constraints. Extraction must be incremental (don't reprocess old messages). Storage must be atomic (incomplete writes are detectable). Indexing must be asynchronous (don't block the Agent's response). Retrieval must be budget-aware (load only what fits). Compression must preserve signal (protect important context).

A system that handles only one or two of these phases — say, just retrieval — leaves the rest to chance. ContextEngine covers the full lifecycle.

### The Six Interception Points

The Agent loop isn't a black box. It has well-defined execution phases — message arrival, reasoning preparation, tool execution, turn completion, context compaction, session shutdown. Each phase presents a different opportunity to read, write, or transform context.

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

The key design choice: **intercept at the loop boundary, not inside model inference.** Stages ①–⑥ are all outside the LLM call. This means zero latency impact on model inference itself — all context operations happen before or after, never during.

### Not All Context Is Equal

Seven types of context, each with fundamentally different lifecycle behavior. This isn't arbitrary categorization — it's driven by the semantics of the information itself.

| Type | Why this lifecycle? | Write Policy |
| ---- | ------------------- | ------------ |
| **Profile** | User state changes — "I live in London" may become "I moved to Tokyo" | **Merge** — new overwrites old on conflict |
| **Preference** | Preferences accumulate per topic but each topic has one current view | **Aggregate by slug** — merge within topic |
| **Entity** | Entities accumulate facts, but "Project Alpha" is still "Project Alpha" | **Aggregate by slug** — merge within entity |
| **Event** | History is immutable — "completed migration on March 15" never changes | **Append only** — never overwrite |
| **Case** | Problem-solving traces are historical records | **Append only** — never overwrite |
| **Pattern** | Patterns emerge from repeated observations and evolve over time | **Aggregate by slug** — refine over time |
| **Skill** | Tool expertise grows cumulatively — more experience = better knowledge | **Cumulative append** — knowledge accumulates |

Four distinct write policies, each justified by the information's lifecycle semantics. A single "store everything the same way" approach would either lose mutable state (if append-only) or corrupt immutable history (if overwriting).

### Key Architectural Decisions

**Why YAML-driven schemas?** Hardcoding extraction schemas in Python means every new context type requires code changes. YAML schemas allow adding new types (e.g., "Decision" or "Handoff") without touching the extraction pipeline — just add a YAML file and the SchemaRegistry, PolicyRouter, and URIResolver all adapt automatically.

**Why the Outbox pattern for async indexing?** Embedding + upserting to a vector index takes 100–500ms. If this were synchronous, every `afterTurn` call would block for that duration, adding latency to the Agent's response. The Outbox pattern decouples writing (fast, ~50ms) from indexing (slow, async), with at-least-once delivery guarantees and a DLQ for permanent failures.

**Why L0/L1/L2 three-level retrieval?** In a corpus of 10,000 context nodes, flat retrieval must embed-and-compare against all 10,000. L0 abstracts (~100 tokens each) are cheap to scan — you can evaluate all 10,000 for ~1M tokens of embedding compute. This narrows to ~50 candidates for L1 (~500 tokens each, ~25K tokens), then to ~3 for L2 full reads. The L1 layer is the key innovation: it's the "informed decision" layer — enough detail to decide if full content is worth the token cost, but cheap enough to evaluate many candidates.

**Why a filesystem abstraction?** Three reasons: (1) The file metaphor is universally understood — `open`, `read`, `write`, `link`, `unlink`, `gc` map directly to context operations. (2) Filesystems have decades of proven solutions for atomicity, concurrency, and permissions — we reuse these rather than reinventing them. (3) Multi-tenant isolation enforced at the filesystem level (via `account_id + owner_space` in every path) is impossible to bypass, even by buggy callers.

**Why optimistic locking for concurrent writes?** Profile nodes are written by every session a user has. Distributed locks would require a coordination service (etcd/ZooKeeper). Optimistic locking — read the current `.meta.json` version, write only if unchanged — handles the common case (no contention) with zero infrastructure, and gracefully fails on the rare case (two sessions writing simultaneously).

**Why the ReAct loop for extraction?** Extraction quality depends on knowing what already exists. If the system already knows "Alice is a backend engineer," extracting that again is wasteful. The ReAct (Reason + Act) loop lets the extractor first *read* existing memories, then decide what's genuinely new. This produces higher-quality extractions at lower cost than blind single-shot extraction.

---

## The Agent Context Lifecycle

ContextEngine activates a set of **context operators** at each of the six stages. Each stage has a distinct purpose, and the stages form a pipeline where each one's output becomes the next one's input.

### Stage ①: message_received — Prefetch Context Before Inference

**Why this stage exists:** The Agent hasn't started thinking yet. This is the cheapest window to prepare context — no model inference is happening, so latency budget is generous (~20ms P99). The goal is to have candidate context ready *before* the model needs it, so assembly in Stage ② is just selection, not search.

| Operator | Responsibility |
| -------- | -------------- |
| **Intent Classifier** | Parse the input message, identify the query intent (fact recall / skill lookup / resource reference), and determine the retrieval strategy and budget priority |
| **Async Prefetch** | Embed the message and run an async similarity search against the L0 abstract layer, staging candidate results in the SessionTopicBuffer's pending slots. Target P99 < 20ms. On timeout, fail silently — never block model inference |

**Connects to:** Stage ② — prefetch results feed directly into assembly. If prefetch times out, Stage ② falls back to synchronous search.

### Stage ②: bootstrap + ingest + assemble + before_prompt_build — Assemble the Context Window

**Why this stage exists:** This is the costliest decision point. The system must select the most relevant context within the token budget — too much wastes tokens, too little loses signal. The hierarchical L0→L1→L2 loading strategy ensures the system always loads the highest-value tokens first.

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

**Connects to:** Stage ① — consumes prefetch results. Stage ④ — the assembled context becomes the extraction input for what was *not* injected (context the Agent generates that should be stored for future turns).

### Stage ③: before_tool_call + tool_result_persist — Context Processing Around Tool Execution

**Why this stage exists:** Tool calls are the Agent's highest-leverage actions — and the most error-prone. Wrong parameters, forgotten failure modes, and oversized results are all addressable if the system can intervene before and after tool execution.

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

**Connects to:** Stage ② — tool context comes from Skills loaded during assembly. Stage ④ — tool snapshots and extracted facts feed into the incremental extraction window.

### Stage ④: afterTurn — Extract Context from What Just Happened

**Why this stage exists:** This is the **primary write window**. Everything the Agent said, every tool it used, every fact it encountered — this is where it gets captured. The key design constraint is incremental processing: only extract the delta since the last checkpoint, not the entire conversation history.

| Operator | Responsibility |
| -------- | -------------- |
| **Incremental Extraction** | LLM-based extraction using a per-session watermark, processing only the delta since the last checkpoint. A single tool use call can output Profile / Preference / Entity / Event / Procedure candidates simultaneously. Read `tool_snapshot` nodes for complete tool results. Filter by a high confidence threshold before writing |
| **Relation Building** | Create edges between new and existing nodes. New Events point to their Entities (`related_to`); sequential Events within the same session get `SEQUENCE` edges (query predecessor via NodeRegistry, no extra LLM call). AggregateTopicPolicy merges also link to existing entities |
| **Conflict Resolution** | Detect contradictions between extracted and existing context, and resolve by type: Profile — new value wins; Preference — merge; Event — always append. Conflicts are written to the audit log |
| **Schema Validation** | Enforce structural invariants: URI format, required fields, and the presence of `account_id + owner_space` in every IndexRecord filter. Block writes on violation |
| **Async Index Build** | Schedule async indexing via the Outbox pattern (embed → upsert). Execute `INSERT outbox_events` with `FOR UPDATE SKIP LOCKED` to support multi-worker concurrent consumption; `pg_notify` wakes workers immediately on insert. This decouples write latency from Agent response time |
| **Tenant Isolation** | Verify that all output writes carry the correct `account_id + owner_space`. Block any write that could leak across account boundaries. All list operations pass through `_is_accessible` filtering |

**Connects to:** Stage ② — everything written here becomes available for retrieval in future turns (via session-local overlay) and future sessions (via vector index). Stage ⑤ — the extraction output determines what's available for compression.

### Stage ⑤: before_compaction + compact — Optimize Context

**Why this stage exists:** Long sessions accumulate context. Without systematic compression, the context window fills with low-value content (repeated information, outdated facts, verbose tool outputs). The challenge isn't just removing things — it's deciding what to keep. Signal scoring must happen *before* compression, not during.

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

**Connects to:** Stage ④ — compressed content must preserve what afterTurn extracted. Stage ⑥ — compaction results are audited at session end.

### Stage ⑥: session_end + dispose — Persist, Archive, Audit

**Why this stage exists:** Sessions don't last forever. When a session ends, the system must (1) archive valuable task traces for future reference, (2) snapshot the current state so the next session can pick up where this one left off, and (3) audit all writes for integrity — catching any atomic write failures that weren't detected during the session.

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

**Connects to:** Stage ② — session_end's task snapshot becomes bootstrap's Handoff injection in the next session. The lifecycle is circular.

---

## The Data Flow: An End-to-End Walkthrough

How does a piece of information travel from conversation to retrieval? Here's the complete path:

```
  Session 1: "I'm Alice, a backend engineer based in London"
  ──────────────────────────────────────────────────────────────
                                                                       
    User Message ──────▶ Stage ④ afterTurn                            
                              │                                       
                              ▼                                       
                        Incremental Extraction                        
                        (LLM identifies: name="Alice",               
                         role="backend engineer",                     
                         location="London")                           
                              │                                       
                              ▼                                       
                    CandidateMemory(category="profile")               
                              │                                       
                              ▼                                       
                    PolicyRouter → ProfilePolicy                      
                    (merge into existing profile)                      
                              │                                       
                              ▼                                       
                    ContextWriter (atomic 4-step write)                
                    ① content.md                                      
                    ② .relations.json                                 
                    ③ .abstract.md + .overview.md                     
                    ④ .meta.json (status=ACTIVE) ← commit point       
                              │                                       
                              ▼                                       
                    OutboxEvent → async index                         
                              │                                       
                              ▼                                       
                    IndexRecordBuilder                                
                    L0: abstract → embed → upsert                    
                    L1: overview  → embed → upsert                    
                    L2: content   → embed → upsert                    
                                                                       
                                                                       
  Session 2: "What does Alice do for a living?"                       
  ──────────────────────────────────────────────────────────────
                                                                       
    User Message ──────▶ Stage ① message_received                    
                              │                                       
                              ▼                                       
                        Intent Classifier → "fact recall"              
                        Async Prefetch → embed query                   
                        → L0 vector search → seed hit                 
                        (profile abstract matches)                    
                              │                                       
                              ▼                                       
                        Stage ② assemble                              
                        Budget Planner → allocate tokens               
                        Relevance Ranker → score candidates            
                        Context Assembly → L0 hit → L2 load            
                        → inject "Alice is a backend engineer"         
                              │                                       
                              ▼                                       
                        Agent response: "Alice is a backend engineer"  
```

The complete lifecycle from extraction to retrieval takes ~100ms for indexing (async) and ~50ms for retrieval (synchronous). The key insight: the cost is paid *once* at write time, amortized over *many* reads.

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
  │ Weaviate │(Phase 2)  │  (Phase 2)  │                         │
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

Seven categories of context, each with a distinct lifecycle and write policy. The write policy is dictated by the information's semantics — not an arbitrary choice.

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

## Implementation Status

### What's Built (Phase 0 + 1 — production quality)

| Component | Status | Key Capability |
|-----------|--------|----------------|
| Core models & interfaces | Done | All dataclasses, Protocol interfaces, enums, URI resolver |
| ContextFS / AGFS adapter | Done | Atomic 4-step write, access control, optimistic locking |
| Extraction pipeline | Done | Unified Extractor, YAML SchemaRegistry, ReAct loop, prefetch |
| Write chain (commit) | Done | ContextWriter, PolicyRouter, 4 merge policies, OutboxStore |
| Hierarchical retrieval | Done | IntentClassifier → SeedRetriever → HierarchicalSearcher → ResultRanker |
| Async indexing | Done | OutboxWorker, RepairJob, L0/L1/L2 IndexRecord builder |
| Providers | Done | OpenAI/Volcengine LLM, 4 embedders, InMemory/ChromaDB/pgvector stores |
| Service layer | Done | MemoryWriteAPI, MemoryReadAPI, multi-tenant isolation |
| HTTP REST API | Done | Flask server, auth/RBAC, session management, admin endpoints |
| Session management | Done | SessionManager, TopicBuffer, RollingCompressor, ArchiveStore |
| Lifecycle management | Done | AgingJob memory lifecycle, AccessTracker usage metrics |
| Testing | Done | Contract + unit + integration + e2e + benchmark (100+ test files) |
| Benchmark | Done | LoCoMo 78.1% accuracy on 1343 questions |

**Lifecycle stages covered:** ① message_received, ② bootstrap + ingest + assemble, ④ afterTurn

### What's Planned (Phase 2 + 3)

| Component | Target Phase | Key Capability |
|-----------|-------------|----------------|
| Tool context injection | Phase 2 | before_tool_call Skill lookup + parameter inference |
| Result compression | Phase 2 | tool_result_persist summary + snapshot + fact extraction |
| Compression pipeline | Phase 2 | before_compaction scoring + compact summary chain |
| Session restore | Phase 2 | dispose buffer flush + bootstrap checkpoint recovery |
| Graph retrieval | Phase 2 | RelationStore multi-hop expansion |
| AI_Functions | Phase 2 | GaussVector ai_chunking / ai_filter |
| Graph clustering | Phase 3 | Cluster compressor (5 Events → 1 Pattern) |
| Adaptive planning | Phase 3 | Dynamic retrieval strategy based on task complexity |
| Observability | Phase 3 | Context lifecycle tracing / metrics / audit |

**Lifecycle stages to cover:** ③ before_tool_call + tool_result_persist, ⑤ compression, ⑥ session_end + dispose

---

## Testing

```bash
pytest tests/contract/ -v          # Must pass — core invariants
pytest tests/ -v                   # All tests
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

---

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

---

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

---

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
