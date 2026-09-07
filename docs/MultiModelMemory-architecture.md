# Multimodal Memory — Architecture Overview

> Navigation doc for the multimodal memory design. Three sections you must
> not confuse:
>
> 1. **Multi-model memory** — the abstraction (graph model + operator
>    framework + LLM multimodal protocol).
> 2. **Shared infrastructure** — concrete providers the abstraction sits
>    on (LLM backends, CV backends, graph stores, embedders, the standalone
>    `cv_service`).
> 3. **m3_bench specific** — drivers, pipeline runner, manifests, and
>    YAML that turn (1)+(2) into the M3 long-video QA benchmark.
>
> This file is intentionally short at the top and modular below so we
> can grow each section without rewriting the others.

---

## 0. One-paragraph summary

oG-Memory's heterogeneous graph has three node layers — **Raw → Extracted
→ Memory** — connected by a fixed set of typed edges. *Multimodal memory*
is the abstraction for getting multimodal data into that graph
(`RawNode` of type `IMAGE` / `VIDEO` / `AUDIO` / `DOCUMENT` / `WEBPAGE`)
and structured output out of it (`ExtractedNode` per extraction, optionally
fan-out into child `RawNode`s per item). The abstraction is implemented by
the `MultimodalExtractor` Protocol + `MaterializationPolicy` + `OperatorService`
in `extraction/operators/`. Providers for LLM, CV, embedding, and graph
storage sit underneath as pluggable backends. The `m3/` directory is one
project that drives this stack end-to-end for the M3-Bench long-video QA
benchmark.

---

## 1. Layer map

```
┌─────────────────────────────────────────────────────────────────┐
│  m3_bench SPECIFIC — m3/  (drivers, pipeline runner, YAML)     │
│   preprocess → load_clips → extract_frames → detect_faces →     │
│   episodic_memory → index_episodic → search/qa                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ uses
┌──────────────────────┴──────────────────────────────────────────┐
│  MULTI-MODEL MEMORY ABSTRACTION                                  │
│   • Graph model      core/graph_models.py, core/graph_enums.py  │
│   • Operator         extraction/operators/{base,registry,        │
│                       pipeline,task_store, materialize}.py      │
│   • LLM protocol     core/interfaces.py::LLM.complete_multimodal│
│   • CV protocol      extraction/operators/backends/base.py      │
│   • OperatorService  service/operator_service.py                │
└──────────────────────┬──────────────────────────────────────────┘
                       │ uses
┌──────────────────────┴──────────────────────────────────────────┐
│  SHARED INFRASTRUCTURE — providers/                              │
│   • LLM:      OpenAI / MiniMax / Mock                           │
│   • CV:       MockBackend / OpenCVBackend / HTTPBackend         │
│   • cv_service (standalone FastAPI app in scripts/cv_service.py)│
│   • Graph store:  MemoryGraphStore / SQLGraphStore              │
│   • Embedder: Mock / OpenAI / SentenceTransformer / Volcengine  │
└─────────────────────────────────────────────────────────────────┘
```

Rule of thumb: **abstraction** = `core/`, `extraction/operators/` (no I/O);
**infra** = `providers/`, `scripts/cv_service.py`, `service/graph_service.py`
backed by graph stores; **m3_bench** = `m3/` + `config/m3/` +
`m3.*` blocks in `deploy/ogmemory.yaml`.

---

## 2. Multi-model memory — the abstraction

The abstraction has three pieces. **No I/O happens here.** Drivers are
responsible for graph access, file writes, env reads, and LLM backend
selection.

### 2.1 Graph model (`core/`)

- **Three node kinds** (`GraphNodeType`): `RAW`, `EXTRACTED`, `MEMORY`.
- **Raw data types** (`RawDataType`): `PROMPT`, `IMAGE`, `VIDEO`,
  `DOCUMENT`, `WEBPAGE`, `AUDIO`. Operator outputs (face bbox,
  episodic description, OCR text, …) are `EXTRACTED`, never `RAW`.
- **Four edge kinds** (`GraphEdgeType`): `EXTRACTED_FROM`,
  `REFERENCES`, `SUPPORTS`, `EVOLVES`. Edge semantics are typed; never
  invent new ones ad-hoc.
- **IDs are deterministic** (`sha1(typed_seed)[:32]`). Re-running any
  operator over the same source must produce the same node id, which
  is the basis for idempotency. See §5.
- File: `core/graph_models.py`, `core/graph_enums.py`.

### 2.2 Operator framework (`extraction/operators/`)

- **`MultimodalExtractor` Protocol** (`base.py`) — pure function:
  `extract(raw_node, ctx) -> ExtractionResult`. Operators MUST NOT
  touch the graph, the filesystem, or env vars. Anything they need
  (children, bytes writer, configured model) is passed in explicitly.
- **`OperatorInfo`** — declarative metadata: `operator_type`,
  `supported_raw_types`, `schema_name`, optional `config_schema`,
  optional `materialization`.
- **`MaterializationPolicy`** (`materialize.py`) — the contract between
  an operator's output shape and the graph layout. Two flavors:
  - `child_node_kind="raw"` — each output item becomes a child
    `RawNode` (e.g. a sampled video frame is raw bytes on disk).
    Default edge `REFERENCES`.
  - `child_node_kind="extracted"` — each output item becomes a child
    `ExtractedNode` (e.g. a face detection is structured bbox +
    embedding). Default edge `EXTRACTED_FROM`.
  - Operators that return one item (e.g. `image_caption`, `episodic_memory_generator`)
    leave `info.materialization = None`; the parent `ExtractedNode`
    carries the result inline.
- **`OperatorRegistry`** — code- or YAML-based registration. The
  pipeline runner in §4 looks up by `operator_type`.
- **`OperatorService`** (`service/operator_service.py`) — the only
  place that ties operators to the graph: it runs the operator, fans
  out children via the `MaterializationPolicy`, writes the lightweight
  parent `ExtractedNode` (`child_node_ids` + `child_count`), and tracks
  async tasks via `TaskStore`. This is the facade drivers call.
- **`OperatorPipeline`** — DAG-style orchestration when several
  operators must run against the same raw node in sequence.
- Built-ins live in `extraction/operators/builtin/`. Anything in
  there is reusable across projects.

### 2.3 Multimodal LLM protocol (`core/interfaces.py::LLM`)

- `complete_json(prompt, schema)` for text-only structured output.
- `complete_multimodal(messages, schema)` for joint vision + audio +
  text input. Content parts follow OpenAI's Chat Completions shape:
  `{type: "text"}`, `{type: "image_url"}`, `{type: "input_audio"}`.
- Implementations MUST raise `NotImplementedError` if a part type is
  not supported (no silent dropping).
- A schema is supplied to every call so JSON-mode + text-fallback
  behaviour is consistent across providers.

---

## 3. Shared infrastructure (providers)

These are concrete backends. They conform to a Protocol and are
selected by config / CLI flag; new backends must NOT require changes
to operators.

### 3.1 LLM providers (`providers/llm/`)

- `OpenAILLM` — text + multimodal + tool use + JSON-mode + text
  fallback. The base class for anything OpenAI-compatible.
- `MiniMaxLLM` — subclass routing to `https://api.minimaxi.com/v1`,
  default model `MiniMax-M3`. Normalises full endpoint URLs down to
  API root.
- `MockLLM` — deterministic stub for tests and offline smoke runs.
- Selection: CLI `--llm {minimax,openai,mock}` or `m3.episodic_llm.provider`
  in `deploy/ogmemory.yaml`.

### 3.2 CV providers (`extraction/operators/backends/`)

- `CVBackend` Protocol — `detect_faces` / `detect_objects` /
  `detect_pose`. Return shape: `Detection(label, confidence, bbox,
  metadata)`.
- `MockBackend` — no deps, deterministic fake results.
- `OpenCVBackend` — local inference (Haar cascades, etc.).
- `HTTPBackend` — forwards to a remote CV service (any deployment
  that speaks the contract).
- `scripts/cv_service.py` — the standalone FastAPI app recommended
  in production. Wraps a backend choice (Haar by default,
  InsightFace if installed) and serves `/detect/faces` etc. This is
  how `HTTPBackend` is wired in practice.

### 3.3 Graph stores (`providers/graph_store/`)

- `MemoryGraphStore` — in-process; default for tests and small runs.
- `SQLGraphStore` — Postgres-backed; selected via `--graph-store sql`
  and `OGMEM_GRAPH_CONNECTION_STRING`.
- Both expose the same `GraphService` API (`service/graph_service.py`),
  so no operator or driver cares which is in use.
- M3 long-video pipelines prefer to inject a `VectorIndex`
  (§3.4) into the `GraphService` constructor; when present,
  `search_by_embedding` routes through HNSW + scope columns
  instead of falling back to the store's JSONB scan. See
  §4.5 for the M3-specific layout.

### 3.4 Embedder providers (`providers/embedder/`)

- `MockEmbedder`, `OpenAIEmbedder`, `SentenceTransformerEmbedder`,
  `VolcengineEmbedder`.
- Used by `m3.index_episodic` and `m3.search` for episodic-memory
  retrieval. Not yet wired into operators; keep here because they're
  a shared provider surface.

---

## 4. m3_bench specific

Everything that turns (1)+(2) into the M3-Bench long-video QA
benchmark lives here. **m3_bench is one consumer of the abstraction;
adding a new benchmark should not require changes to §2 or §3.**

### 4.1 Per-stage drivers (`m3/`)

Each driver wraps one operator (or a small piece of glue) for the M3
data shape — 30s clips, JSONL manifests, deterministic IDs — and is
the place where idempotency and JSONL handoff live.

| Driver                       | Operator / function             | Output manifest             | Graph effect                                                                                  |
|------------------------------|---------------------------------|-----------------------------|-----------------------------------------------------------------------------------------------|
| `m3/preprocess.py`           | ffmpeg slice                    | (clips on disk)             | —                                                                                             |
| `m3/load_clips.py`           | —                               | `data.jsonl`                | VIDEO `RawNode` per clip + chain `REFERENCES`                                                 |
| `m3/extract_frames.py`       | `video_frame_extractor`         | `frames.jsonl`              | IMAGE `RawNode` per frame + parent `ExtractedNode` + `REFERENCES`                             |
| `m3/detect_faces.py`         | `face_detection`                | `faces.jsonl`               | FACE `ExtractedNode` per face + `EXTRACTED_FROM` (via CV backend)                             |
| `m3/episodic_memory.py`      | `episodic_memory_generator`     | `episodic_memory.jsonl`     | parent VIDEO `ExtractedNode` with description / characters / events / entities                |
| `m3/index_episodic.py`       | (embedder)                      | `episodic_index.jsonl`      | — (text-only retrieval index)                                                                 |
| `m3/search.py` / `m3/qa.py`  | —                               | `m3_qa_results.jsonl`       | —                                                                                             |

Drivers follow the same shape — see §5.2.

### 4.2 Declarative pipeline runner (`m3/pipeline/`)

The runner lets a YAML file describe an end-to-end M3 pipeline and
the CLI overrides individual stage params without editing YAML.

- `spec.py` — `PipelineSpec` / `StageSpec` (YAML-loadable, structurally
  validated).
- `context.py` — shared `PipelineContext` + per-stage `StageResult`.
- `stages.py` — `DEFAULT_STAGE_REGISTRY` mapping `stage.type` →
  `StageFn`. Each `StageFn` is a thin wrapper that builds the CLI
  argv for a driver in §4.1 and calls its `run()`.
- `runner.py` — topological execution, `--enable-stage` /
  `--disable-stage` / `--set stage.path=value` overrides, summary JSON.
- Entry point: `m3/run_video_pipeline.py` loads
  `config/m3/pipelines/*.yaml` (default `video_episodic.yaml`,
  `video_episodic_faces.yaml`).
- Stage ids are the same as the manifest filenames they emit
  (`extract_frames` → `frames_jsonl`, `episodic_memory` →
  `episodic_jsonl`, …). Adding a stage = adding one entry to
  `DEFAULT_STAGE_REGISTRY` + one `stages:` block in YAML.

### 4.3 M3-specific config + helpers (`m3/config.py`, `m3/resume_status.py`)

- `M3BENCH_ACCOUNT_ID` / `M3BENCH_AGENT_ID` / `M3BENCH_USER_ID` —
  fixed `RequestContext` identifiers used by every driver.
- `load_m3_runtime_config()` — reads the `m3:` block of
  `deploy/ogmemory.yaml` and falls back to top-level `llm` /
  `embedding` for unset keys.
- `resolve_video_path()` — three-step path resolver for the M3
  annotation `video_path` field (project-relative, videos_root subset,
  flat fallback).
- `is_completed_extraction()` — shared idempotency gate that
  recognises both `ExtractionStatus` and its string form after
  SQL round-trip.

### 4.4 M3-specific outputs (the JSONL handoff contract)

The JSONL manifests below are the **handoff contract between
drivers**. New drivers MUST read the upstream shape and write the
downstream shape so existing stages do not need to change.

| File                     | Producer               | Consumer                                  |
|--------------------------|------------------------|-------------------------------------------|
| `data.jsonl`             | `load_clips`           | `extract_frames`, `episodic_memory`       |
| `frames.jsonl`           | `extract_frames`       | `detect_faces`, `episodic_memory`         |
| `faces.jsonl`            | `detect_faces`         | (future `face_clustering`, QA)            |
| `episodic_memory.jsonl`  | `episodic_memory`      | `index_episodic`, `search`, `qa`          |
| `episodic_index.jsonl`   | `index_episodic`       | `search`, `qa`                            |
| `m3_qa_results.jsonl`    | `qa`                   | (eval harness, e.g. `scripts/m3_eval.py`) |

Each line is a flat dict keyed by `video_id` / `clip_id` plus
operator-specific fields (e.g. `child_node_ids`, `face_count`,
`description`). Provenance IDs (`video_raw_node_id`,
`extracted_node_id`, child node ids) flow downstream unchanged.

---

## 5. Cross-cutting conventions

### 5.1 Deterministic IDs

Every child node id is `sha1(typed_seed)[:32]`. Seed templates that
are already pinned by manifest format:

- VIDEO RawNode id: `sha1(f"{video_id}/{clip_id}")`
- IMAGE RawNode id (operator path): `sha1(f"frame/{video_raw_id}/{frame_idx}")`
- IMAGE RawNode id (m3 legacy): `sha1(f"frame/{video_id}/{clip_id}/{frame_idx}")`
- FACE ExtractedNode id: `sha1(f"face/{source_frame_id}/{face_id}")`
- Episodic parent ExtractedNode id: `sha1(f"extracted/{extractor_type}/{source_raw_id}")`

Do not change these seeds without an explicit migration — downstream
tooling and existing manifests rely on the exact formula.

### 5.2 Driver pattern

Every m3 driver follows the same shape (see `m3/extract_frames.py`
as the template):

1. Parse CLI args, build `RequestContext` with `M3BENCH_*` ids.
2. Build graph service (`build_graph_service`) and `OperatorService`.
3. Idempotency pre-check: `graph.get_extracted_node(ext_id, ctx)` —
   if it exists and `is_completed_extraction(status)`, emit a
   `resumed=True` record and skip.
4. Construct operator instance directly (NOT via registry lookup,
   because registry-cached instances ignore per-call kwargs).
5. Call `extractor.extract(raw_node, ctx)` →
   `op_service.persist_extraction(raw_node, operator, extraction,
   ctx, bytes_writer=..., desired_node_id=ext_id)`.
6. Write a JSONL manifest line per processed item with
   `video_id` / `clip_id` / provenance ids / `resumed` flag.

### 5.3 Operator purity contract

Operators MUST NOT touch the graph, the filesystem, or env vars.
- **Bytes on disk** — operators return base64 in `extracted_content`;
  the driver passes a `bytes_writer` closure that decides the file
  layout. Operators stay pure.
- **Children of a source** — operators declare what they need
  (`frame_nodes=`, `audio_node=`); the driver fetches them via the
  graph. Operators never call `graph.get_*` themselves.
- **Provider selection** — operators receive `LLM` / `CVBackend`
  instances in their constructor; they never read env.

### 5.4 Provider selection by config

- Top-level `llm` / `embedding` blocks in `deploy/ogmemory.yaml`
  set service defaults.
- `m3.episodic_llm` / `m3.qa_llm` / `m3.embedding` blocks override
  per use case.
- CLI flags (`--llm`, `--llm-model`, `--embedder`, …) override the
  YAML for one run.

### 5.5 Status & idempotency

- `ExtractionStatus` ∈ {`PENDING`, `IN_PROGRESS`, `COMPLETED`,
  `FAILED`}.
- Drivers short-circuit on `COMPLETED` only — `PENDING` / `FAILED`
  nodes are re-processed so a previously failed run recovers.
- Use `m3.resume_status.is_completed_extraction()` to compare; never
  compare `status == "completed"` directly because the SQL store may
  round-trip the enum through a string form.

---

## 6. Where to start when extending

- **New operator** → add to `extraction/operators/builtin/`, import
  it from `builtin/__init__.py`, register in
  `register_all_defaults()`. Declare a `MaterializationPolicy` if it
  returns multiple items; otherwise leave `info.materialization =
  None`.
- **New LLM backend** → subclass `OpenAILLM` and override the
  defaults (`base_url`, `model`, env precedence). Add a lazy loader
  in `providers/llm/__init__.py`.
- **New CV backend** → implement the `CVBackend` Protocol in
  `extraction/operators/backends/`. Operators do not change.
- **New m3 driver** → copy `m3/extract_frames.py`; the structure in
  §5.2 is the contract.
- **New m3 pipeline stage** → add a `StageFn` to
  `m3/pipeline/stages.py::DEFAULT_STAGE_REGISTRY` and reference it
  from a YAML under `config/m3/pipelines/`.
- **New graph store** → implement the `GraphMemoryStore` protocol
  in `providers/graph_store/`; nothing else changes.

---

## 7. Open questions / roadmap (placeholder)

- `m3/extract_audio.py` not yet written; `episodic_memory_generator`
  has the `--no-audio` path for vision-only runs in the meantime.
- Cross-clip identity (`face_clustering`) and voice segmentation /
  voice clustering are unimplemented — they need the new
  `same_identity` / `co_occurs_in` edge types first.
- `scripts/m3_eval.py` (1276 QA × 5 rounds + GPT-4o judge) and the
  full ReAct `OGMemoryControlAdapter` over `[Answer]` / `[Search]`
  are not yet built.
- `deploy/ogmemory.yaml` `m3.*` block is the single configuration
  point — adding a new M3 model = one new sub-block, no code edits.

For phase-by-phase status and dependencies, see the project's
`m3-bench-phase1-plan` memory; for the multi-model LLM decision and
its caveats, see `m3-bench-multimodal-llm`.

---

## 8. M3 vector storage (post-`VectorIndex` migration)

> New section added when M3 moved vector storage from
> `ExtractedNode.extracted_content["embedding"]` (SQL JSONB column
> scanned in Python) to a dedicated `VectorIndex` (HNSW + SQL
> scope columns).

### 8.1 Where embeddings live now

Each `episodic_memory_generator` clip produces one `IndexRecord`
upserted to the `OpenGaussVectorIndex`. The graph store's JSONB
`extracted_content` is **not** written any more — legacy
embeddings remain as dead data and a future cleanup pass can
purge them.

```
m3.index_episodic ── IndexRecord ──► OpenGaussVectorIndex.m3_episodic_vectors
                                       │  HNSW index on embedding
                                       │  partial indexes on (video_id, extractor_type, before_clip)
m3.qa (run_react_qa) ── graph.search_by_embedding ──► GraphService ──► VectorIndex.search_by_vector
                                                                              │
                                                                              ▼
                                                          ExtractedNode.extracted_content
                                                          (provenance lift, NOT the vector)
```

### 8.2 `IndexRecord` shape for M3

- `id`: `IndexRecord.generate_id(uri, level=2)` where
  `uri = "m3://<video_id>/clips/<clip_id>/episodic"`.
  sha256(uri + ":2")[:16], 16-char hex. Idempotent re-indexing.
- `uri`: as above; matches the JSONL audit manifest's `uri`
  field so external tooling can join the two stores.
- `level`: 2 (content-level; matches `seed_retriever`).
- `filters` (tenant + M3 scope): `{account_id, video_id,
  extractor_type: "episodic_memory_generator"}`. Each maps to a
  SQL column + partial index in
  `OpenGaussVectorIndex._ensure_table` (`video_id`,
  `extractor_type`) or to the existing `account_id` column.
- `metadata` (forward channel + scope columns):
  `{_embedding, extracted_node_id, video_id, clip_id,
  before_clip, source}`. The `extracted_node_id` is what
  `GraphService.search_by_embedding` reads to reverse-resolve to
  the originating `ExtractedNode` via `self._store.get_node`.

### 8.3 Scope semantics

- `video_id` is exact-match: M3-Bench evaluates one video at a
  time, so cross-video bleed is a correctness bug, not a perf
  issue.
- `extractor_type` accepts a scalar or list; uses `= ANY(...)`
  for the list form.
- `before_clip` is the QA evaluation boundary:
  `before_clip <= N`. The `_build_where` clause is
  `(before_clip IS NULL OR before_clip <= N)` so legacy rows
  (the column didn't exist when they were written) remain
  findable until they're re-indexed.

### 8.4 Why we stopped writing `extracted_content["embedding"]`

The slot was originally used so `GraphService.search_by_embedding`
could read the vector back. After the migration:

- Vectors live in `OpenGaussVectorIndex` (HNSW + scope columns).
- `GraphService.search_by_embedding` (when a `VectorIndex` is
  injected) routes through `search_by_vector` and never reads
  the slot.
- Non-M3 callers of `MemoryGraphStore.search_by_embedding`
  (none in the current codebase, but the path is preserved for
  back-compat) still work because the slot's read path is
  unchanged.

No database migration is needed; old JSONB rows simply become
unread by M3.

### 8.5 Configuration surface

`m3.vector_index.*` in `deploy/ogmemory.yaml`:

```yaml
m3:
  vector_index:
    backend: opengauss                 # recommended production target
    connection_string: "${OGMEM_GRAPH_CONNECTION_STRING}"
    table_name: m3_episodic_vectors
    pool_size: 5
```

The ``backend`` field accepts two values:

- ``opengauss`` — Huawei's openGauss database. **The recommended
  production target** for the M3 vector index; the M3 benchmarks
  were validated against openGauss. Deployments that already
  operate an openGauss cluster should keep this setting.
- ``pgvector`` — vanilla PostgreSQL 15+ with the pgvector
  extension. Convenience alias for deployments that prefer to
  share one Postgres instance between the graph store and the
  vector index instead of running a separate openGauss cluster.
  The docker-compose ``postgres`` service now ships the
  ``pgvector/pgvector:pg16`` image so this works out of the box
  in dev / single-DB deployments.

Both backends construct the same :class:`OpenGaussVectorIndex`
instance because the underlying SQL — ``CREATE EXTENSION vector``,
``vector(N)``, ``hnsw(embedding vector_cosine_ops)``, ``MERGE
INTO`` — is portable between the two servers via the standard PG
wire protocol. ``psycopg2`` talks to either without code changes.
The ``_ensure_table`` migration issues ``CREATE EXTENSION IF NOT
EXISTS vector``: openGauss rejects this as a syntax error (caught
and ignored — openGauss ships ``vector(N)`` natively), while PG
+pgvector accepts it and activates the preinstalled extension.

CLI overrides (``--vector-index-connection-string`` etc.) win
over YAML for one-off experiments. This is the canonical
"seamless switch via connection string" path: point
``connection_string`` at the ``postgres`` container in dev (no
openGauss dependency) or at an openGauss cluster in production
without changing a single line of pipeline code.

### 8.6 Pipeline integration

`m3.pipeline.runner.PipelineRunner.run()` builds the
`VectorIndex` once from YAML and injects it via
`m3.load_clips.build_graph_service(..., vector_index=...)` so
the `GraphService` constructed for the QA stage sees the same
instance the `index_episodic` stage wrote to. Tests that
construct `PipelineContext` directly can leave
`vector_index=None` and have the runner build it on first
`run()`.