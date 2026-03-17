# 写入链路模块设计

> 本文基于 `ce_architecture.md` 第 4-12 节的写入链路大纲、模块边界、API 契约和实现建议，结合 OpenViking `session/` 模块实现，为 oG-Memory 的 `commit` 包提供详细的模块设计。
>
> 存储模型：**FS 主存 + 异步索引副本** — 本地文件系统是唯一真相源，openGauss 作为检索用的索引副本，由 IndexWorker 异步同步。

---

## 1. 模块总览

### 1.1 总体架构

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          ContextEngineService                          │
│                        commit_session(ctx, ...)                        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CommitCoordinator                              │
│                 串联全部写入阶段的核心编排器                               │
├─────────┬───────────┬───────────┬───────────┬──────────────────────────┤
│ Phase 1 │ Phase 2   │ Phase 3   │ Phase 4   │ Phase 5                  │
│ 归档    │ 候选抽取   │ 策略规划   │ FS 写入   │ Outbox 登记 + 构造结果    │
└────┬────┴─────┬─────┴─────┬─────┴─────┬─────┴──────────────────────────┘
     │          │           │           │
     ▼          ▼           ▼           ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐   ┌──────────────────────┐
│ Archive │ │Candid. │ │ Policy  │ │ Context │   │  Outbox 登记          │
│ Builder │ │Pipeline│ │ Router  │ │ Writer  │   │  (最佳努力，           │
│         │ │        │ │         │ │         │   │   不阻塞 commit)      │
│  LLM ◄──┤ │ 4x LLM│ │ 4 种    │ │ 纯 FS   │   └──────────┬───────────┘
│  摘要    │ │ 并行   │ │ 策略    │ │ 写入    │              │ 异步
└─────────┘ └────────┘ └────┬────┘ └────┬────┘              ▼
                             │          │        ┌──────────────────────┐
                             ▼          ▼        │    IndexWorker       │
                        ┌─────────┐              │  读 FS → 分片 →     │
                        │Relation │              │  embed → 写 DB 副本  │
                        │ Writer  │              └──────────────────────┘
                        │ (纯 FS) │
                        └────┬────┘
                             │
                             ▼
                        ┌─────────┐
                        │Relation │
                        │ Store   │
                        │ (纯 FS) │
                        └─────────┘
```

### 1.2 写入链路全景流程

```text
ContextEngineService.commit_session(ctx, messages, ...)
│
├─ 1. ArchiveBuilder.build_archive()          → SessionArchive
│
├─ 2. CandidatePipeline.extract_candidates()  → CandidateMemory[]
│     ├─ ProfileExtractor
│     ├─ PreferenceExtractor
│     ├─ EntityEventExtractor
│     └─ ProcedureExtractor
│
├─ 3. CandidatePipeline.normalize_candidates() → 去重 + 规范化
│
├─ 4. for each candidate:
│     ├─ PolicyRouter.select_policy()          → MergePolicy
│     └─ PolicyRouter.build_plan()             → WritePlan
│           ├─ ProfilePolicy.plan()
│           ├─ AggregateTopicPolicy.plan()
│           ├─ AppendOnlyPolicy.plan()
│           └─ SkillToolPolicy.plan()
│
├─ 5. ContextWriter.apply_plan()              → WriteResult
│     └─ FS: 写 .abstract.md / .overview.md / content.md / .meta.json
│
├─ 6. RelationWriter.apply_edges()
│     └─ FS: 写 .relations.json
│
├─ 7. OutboxStore.append_batch()              → OutboxEvent[]
│     └─ 最佳努力，失败不阻塞 commit
│
└─ 8. return CommitResult(status="success")
       └─ FS 写入成功 = commit 成功
```

### 1.3 存储模型

```text
┌──────────────────────────────────┐
│         主存层 (FS)               │  ← 唯一真相源，写入成功即 commit 成功
│  .abstract.md   (L0)             │
│  .overview.md   (L1)             │
│  content.md     (L2)             │
│  .meta.json     (元数据)          │
│  .relations.json (关系)           │
└──────────────┬───────────────────┘
               │ Outbox 异步（最终一致）
               ▼
┌──────────────────────────────────┐
│    索引副本层 (openGauss)         │  ← 可丢失、可从 FS 完整重建
│  context_nodes   (元数据副本)     │
│  context_chunks  (文本分片+向量)  │
│  relation_edges  (关系副本)       │
│  outbox_events   (同步队列)       │
└──────────────────────────────────┘
```

| 存储层 | 写入内容 | 定位 |
|--------|----------|------|
| 本地文件系统（主存） | `.abstract.md`, `.overview.md`, `content.md`, `.meta.json`, `.relations.json` | 唯一真相源，人类可读，可离线恢复 |
| openGauss（索引副本） | `context_nodes`, `context_chunks`, `relation_edges` | 支撑检索链路，由 IndexWorker 异步维护，可从 FS 完整重建 |
| openGauss（Outbox） | `outbox_events` | 驱动 IndexWorker 的异步同步队列 |

### 1.4 提交成功判定

```text
┌─ commit 写入范围 ────────────────────────────────────┐
│                                                       │
│  Phase 4: FS 写入（主存）                              │
│    写入目录节点下的语义文件 + .meta.json + .relations   │
│    FS 写入成功 → commit 成功 ✓                         │
│                                                       │
│  Phase 5: Outbox 登记（最佳努力）                      │
│    向 outbox_events 表登记异步同步任务                   │
│    失败不影响 commit 结果                               │
│    repair job 可扫描 FS 补发缺失事件                    │
│                                                       │
└───────────────────────────────────────────────────────┘
                          │ 异步
                          ▼
┌─ IndexWorker 异步同步 ───────────────────────────────┐
│  claim 事件 → 读 FS → 分片 → embed → 写 DB 副本       │
│  失败可重试，不影响主存完整性                            │
└───────────────────────────────────────────────────────┘
```

---

## 2. 与架构文档偏离说明

本节集中列出设计相对 `ce_architecture.md` 的有意偏离，每条说明偏离内容、理由和影响。

| ID | 架构文档规定 | 本设计实际做法 | 偏离理由 |
|----|-------------|--------------|----------|
| D-1 | `WritePlan` 包含 `outbox_events` 字段（4.1） | `WritePlan` 不含 `outbox_events`，由 `CommitCoordinator` 在 Phase 5 统一生成 | Outbox 事件依赖 `WriteResult`（需要写入后的版本号等信息），在策略层生成时信息不完整。Coordinator 层统一生成更准确 |
| D-2 | `RelationWriter.apply_edges` 返回 `None`（10.4） | 返回 `List[RelationEdge]` | Coordinator 需要返回的关系边列表来构建 `UPSERT_RELATION` outbox 事件。改为返回值不影响其他调用方 |
| D-3 | `WritePlan` 无 `metadata_patch` 字段（4.1） | 新增 `metadata_patch: Dict[str, Any]` | SkillToolPolicy 需要传递统计数据增量到 ContextWriter，`node_patch` 仅覆盖 abstract/overview/content 不够用 |
| D-4 | 所有接口先用同步语义建模（12） | 核心接口使用 async，同时提供同步 wrapper | Python 生态中 psycopg 和 httpx 的异步版本更成熟；同步 wrapper 通过 `asyncio.run()` 包装，保持接口可用性 |

---

## 3. 核心数据模型

以下数据模型以 Python dataclass 定义，与 `ce_architecture.md` 第 4 节对齐，并补充写入链路所需的字段。

### 3.1 RequestContext

```python
@dataclass
class RequestContext:
    """请求级上下文，贯穿整个写入链路"""
    account_id: str
    user_id: str
    agent_id: str
    session_id: str
    trace_id: str

    @property
    def user_space(self) -> str:
        return f"users/{self.user_id}"

    @property
    def agent_space(self) -> str:
        return f"agents/{self.agent_id}"
```

### 3.2 SessionArchive

```python
@dataclass
class SessionArchive:
    """会话归档快照 — ArchiveBuilder 的输出"""
    session_id: str
    archive_uri: str               # ctx://{account}/sessions/{session_id}/archives/{index}
    summary: str                   # LLM 生成的结构化摘要
    recent_messages: List[Dict]    # 最近 N 条消息原文
    used_contexts: List[str]       # 本轮使用过的记忆 URI
    used_tools: List[Dict]         # 本轮工具调用记录
    stats: SessionStats

@dataclass
class SessionStats:
    total_turns: int
    total_tokens: int
    compression_count: int
    memories_extracted: int
```

### 3.3 CandidateMemory

```python
class MemoryCategory(str, Enum):
    PROFILE = "profile"
    PREFERENCES = "preferences"
    ENTITIES = "entities"
    EVENTS = "events"
    CASES = "cases"
    PATTERNS = "patterns"
    SKILLS = "skills"

@dataclass
class CandidateMemory:
    """候选记忆 — CandidatePipeline 的输出，不携带落盘细节"""
    category: MemoryCategory
    owner_scope: str          # "user" | "agent"
    routing_key: str          # 聚合键（主题 slug / 实体名 / 工具名）
    abstract: str             # L0：1-2 句话精炼摘要
    overview: str             # L1：结构化概览
    content: str              # L2：完整叙述
    confidence: float         # 抽取置信度 0.0-1.0
    source_refs: List[str]    # 来源消息 ID
    language: str = "zh"      # 输出语言

@dataclass
class ToolSkillCandidate(CandidateMemory):
    """工具/技能类候选，附带统计数据"""
    tool_name: str = ""
    skill_name: str = ""
    call_count: int = 0
    success_count: int = 0
    total_duration_ms: int = 0
    total_tokens: int = 0
    best_for: str = ""
    common_failures: str = ""
```

### 3.4 WritePlan

```python
class WriteAction(str, Enum):
    CREATE = "create"
    MERGE = "merge"
    APPEND = "append"
    SKIP = "skip"

@dataclass
class RelationOp:
    """关系操作指令"""
    op: str                 # "upsert" | "append" | "remove"
    edges: List['RelationEdge']

@dataclass
class WritePlan:
    """策略层输出 — 描述一次具体的写入动作"""
    action: WriteAction
    target_uri: str              # 目标节点 URI
    node_patch: Dict[str, str]   # {"abstract": ..., "overview": ..., "content": ...}
    relation_ops: List[RelationOp]
    reason: str                  # 策略决策理由（用于审计）
    metadata_patch: Dict[str, Any] = field(default_factory=dict)  # 偏离 D-3
```

### 3.5 WriteResult 与 CommitResult

```python
@dataclass
class WriteResult:
    """单个节点的写入结果"""
    uri: str
    action: WriteAction
    node_version: int
    created: bool
    updated: bool

@dataclass
class CommitResult:
    """一次 commit_session 的完整结果"""
    archive: SessionArchive
    write_results: List[WriteResult]
    outbox_events: List['OutboxEvent']
    stats: CommitStats
    status: str               # "success" | "partial" | "failed"

@dataclass
class CommitStats:
    candidates_extracted: int
    candidates_skipped: int
    nodes_created: int
    nodes_merged: int
    nodes_appended: int
    relations_written: int
    outbox_events_queued: int
    duration_ms: int
```

### 3.6 OutboxEvent

```python
class OutboxEventType(str, Enum):
    UPSERT_CONTEXT = "UPSERT_CONTEXT"
    DELETE_CONTEXT = "DELETE_CONTEXT"
    MOVE_CONTEXT = "MOVE_CONTEXT"
    UPSERT_RELATION = "UPSERT_RELATION"

class OutboxStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"

@dataclass
class OutboxEvent:
    """异步索引同步事件"""
    event_id: str
    event_type: OutboxEventType
    uri: str
    payload: Dict[str, Any]
    status: OutboxStatus = OutboxStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    worker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
```

### 3.7 ContextNode 与 RelationEdge

```python
@dataclass
class ContextNode:
    """主存节点 — 对应文件系统上的一个目录"""
    uri: str
    parent_uri: str
    context_type: str          # "memory" | "session" | "skill"
    category: MemoryCategory
    owner_space: str           # "users/{user_id}" | "agents/{agent_id}"
    abstract: str
    overview: str
    content: str
    metadata: Dict[str, Any]
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class RelationEdge:
    """节点间关系边"""
    from_uri: str
    to_uri: str
    relation_type: str         # "related_to" | "derived_from" | "part_of" | "references"
    reason: str
    weight: float = 1.0
    created_at: Optional[datetime] = None
```

### 3.8 ChunkRecord

由 IndexWorker 在异步索引阶段使用，不在 commit 写入链路中产生。

```python
@dataclass
class ChunkRecord:
    """文本分片记录 — DocumentChunker 的输出，由 IndexWorker 写入 DB 索引副本"""
    chunk_id: str              # stable_hash(uri, level, chunk_index)
    uri: str                   # 所属节点 URI
    parent_uri: str = ""
    level: int = 0             # 0=L0, 1=L1, 2=L2
    chunk_index: int = 0       # 同层级内的分片序号
    text_content: str = ""
    context_type: str = "memory"
    category: str = ""
    owner_space: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.9 NodeStat 与 NodeSummary

与 `ce_architecture.md` 4.1 对齐，`ContextFS` API 返回类型。

```python
@dataclass
class NodeStat:
    """节点状态 — ContextFS.stat_node() 返回"""
    uri: str
    exists: bool
    version: int
    size: int                  # 节点目录总字节数
    updated_at: Optional[datetime] = None

@dataclass
class NodeSummary:
    """列表/树视图条目 — ContextFS.list_children() 返回"""
    uri: str
    name: str
    is_dir: bool
    has_children: bool
    category: str
    updated_at: Optional[datetime] = None
```

### 3.10 CommitOptions 与 DedupConfig

```python
@dataclass
class CommitOptions:
    """commit_session 的可选参数"""
    dry_run: bool = False                  # 只生成 WritePlan 不实际写入
    skip_archive: bool = False             # 跳过归档步骤
    skip_dedup: bool = False               # 跳过去重检查
    skip_relations: bool = False           # 跳过关系写入
    categories_filter: Optional[List[MemoryCategory]] = None  # 只抽取指定类别
    max_candidates: Optional[int] = None   # 覆盖 PipelineConfig.max_candidates_per_commit

@dataclass
class DedupConfig:
    """候选去重器配置"""
    dedup_top_k: int = 5                   # 向量预筛选返回的最相似记录数
    similarity_threshold: float = 0.85     # 高于此阈值才进入 LLM 精确判断
    enable_llm_decision: bool = True       # 是否用 LLM 做精确去重判断
```

---

## 4. openGauss Schema 设计（索引副本，由 IndexWorker 异步维护）

以下表由 IndexWorker 消费 Outbox 事件后异步写入，不在 commit 写入链路中直接操作。commit 链路唯一写入的 DB 表是 `outbox_events`（最佳努力）。

### 4.1 Schema 初始化

```sql
CREATE SCHEMA IF NOT EXISTS ctx;
```

### 4.2 context_nodes 表

```sql
CREATE TABLE ctx.context_nodes (
    uri              VARCHAR(512) PRIMARY KEY,
    parent_uri       VARCHAR(512),
    context_type     VARCHAR(32)  NOT NULL DEFAULT 'memory',
    category         VARCHAR(32)  NOT NULL,
    owner_space      VARCHAR(256) NOT NULL,
    abstract_text    TEXT,
    version          INTEGER      NOT NULL DEFAULT 1,
    metadata         JSONB        NOT NULL DEFAULT '{}',
    fs_path          VARCHAR(1024),
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_context_nodes_parent ON ctx.context_nodes(parent_uri);
CREATE INDEX idx_context_nodes_category ON ctx.context_nodes(category);
CREATE INDEX idx_context_nodes_owner ON ctx.context_nodes(owner_space);
CREATE INDEX idx_context_nodes_updated ON ctx.context_nodes(updated_at);
CREATE INDEX idx_context_nodes_metadata ON ctx.context_nodes USING gin(metadata);
```

### 4.3 context_chunks 表

```sql
CREATE TABLE ctx.context_chunks (
    chunk_id         VARCHAR(128) PRIMARY KEY,
    uri              VARCHAR(512) NOT NULL REFERENCES ctx.context_nodes(uri) ON DELETE CASCADE,
    parent_uri       VARCHAR(512),
    level            SMALLINT     NOT NULL,  -- 0=L0(abstract), 1=L1(overview), 2=L2(content)
    chunk_index      SMALLINT     NOT NULL DEFAULT 0,
    text_content     TEXT         NOT NULL,
    embedding        vector(1536),
    context_type     VARCHAR(32),
    category         VARCHAR(32),
    owner_space      VARCHAR(256),
    metadata         JSONB        NOT NULL DEFAULT '{}',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_embedding_l0 ON ctx.context_chunks
    USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE level = 0;

CREATE INDEX idx_chunks_embedding_all ON ctx.context_chunks
    USING ivfflat(embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_chunks_bm25 ON ctx.context_chunks
    USING bm25(text_content);

CREATE INDEX idx_chunks_uri ON ctx.context_chunks(uri);
CREATE INDEX idx_chunks_level ON ctx.context_chunks(level);
CREATE INDEX idx_chunks_category ON ctx.context_chunks(category);
CREATE INDEX idx_chunks_owner ON ctx.context_chunks(owner_space);
```

### 4.4 relation_edges 表

```sql
CREATE TABLE ctx.relation_edges (
    id               VARCHAR(128) PRIMARY KEY,
    from_uri         VARCHAR(512) NOT NULL,
    to_uri           VARCHAR(512) NOT NULL,
    relation_type    VARCHAR(64)  NOT NULL DEFAULT 'related_to',
    reason           TEXT,
    weight           REAL         NOT NULL DEFAULT 1.0,
    metadata         JSONB        NOT NULL DEFAULT '{}',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(from_uri, to_uri, relation_type)
);

CREATE INDEX idx_relation_from ON ctx.relation_edges(from_uri);
CREATE INDEX idx_relation_to ON ctx.relation_edges(to_uri);
CREATE INDEX idx_relation_type ON ctx.relation_edges(relation_type);
```

### 4.5 outbox_events 表

此表是 commit 链路唯一涉及的 DB 表（最佳努力写入）。

```sql
CREATE TABLE ctx.outbox_events (
    event_id         VARCHAR(64)  PRIMARY KEY,
    event_type       VARCHAR(32)  NOT NULL,
    uri              VARCHAR(512) NOT NULL,
    payload          JSONB        NOT NULL DEFAULT '{}',
    status           VARCHAR(16)  NOT NULL DEFAULT 'pending',
    worker_id        VARCHAR(64),
    retry_count      SMALLINT     NOT NULL DEFAULT 0,
    max_retries      SMALLINT     NOT NULL DEFAULT 3,
    error_message    TEXT,
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_outbox_status ON ctx.outbox_events(status, created_at)
    WHERE status IN ('pending', 'claimed');
CREATE INDEX idx_outbox_uri ON ctx.outbox_events(uri);
```

### 4.6 session_archives 表

由 IndexWorker 或独立同步进程从 FS 归档写入，不在 commit 链路中操作。

```sql
CREATE TABLE ctx.session_archives (
    archive_id       VARCHAR(128) PRIMARY KEY,
    session_id       VARCHAR(128) NOT NULL,
    archive_uri      VARCHAR(512) NOT NULL,
    summary          TEXT,
    message_count    INTEGER      NOT NULL DEFAULT 0,
    stats            JSONB        NOT NULL DEFAULT '{}',
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_archives_session ON ctx.session_archives(session_id);
CREATE INDEX idx_archives_created ON ctx.session_archives(created_at);
```

---

## 5. Provider 协议接口

对齐架构文档第 9 章模块边界和第 10.8 节 Provider API，定义写入链路依赖的 provider 协议。

### 5.1 LLM 协议

```python
class LLM(Protocol):
    """LLM 能力接口 — 为抽取器和策略层提供结构化输出"""

    async def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        ctx: Optional[RequestContext] = None,
    ) -> Dict:
        ...

    def complete_json_sync(
        self,
        prompt: str,
        schema: Dict[str, Any],
        ctx: Optional[RequestContext] = None,
    ) -> Dict:
        """同步 wrapper"""
        ...
```

### 5.2 Embedder 协议

```python
@dataclass
class EmbeddingVector:
    """embedding 输出"""
    dense_vector: List[float]
    sparse_vector: Optional[Dict[int, float]] = None
    model: str = ""

class Embedder(Protocol):
    """文本向量化接口"""

    async def embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[EmbeddingVector]:
        ...
```

### 5.3 VectorIndex 协议

去重器通过 VectorIndex 在索引副本上做相似度检索，不直接依赖 `storage` 包。

```python
class VectorIndex(Protocol):
    """向量索引接口 — 对齐 ce_architecture 10.8"""

    async def search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int = 10,
        level: int = 0,
    ) -> List[Dict]:
        ...
```

### 5.4 与现有 embeddings/ 包的关系

现有 `src/embeddings/base.py` 中的 `EmbeddingBase` 和 `src/embeddings/openai.py` 中的 `OpenAIEmbedding` 作为 `Embedder` 协议的具体实现。适配方式：

```python
class OpenAIEmbedderAdapter:
    """将现有 OpenAIEmbedding 适配为 Embedder 协议"""

    def __init__(self, openai_embedding: OpenAIEmbedding):
        self._impl = openai_embedding

    async def embed_texts(self, texts, model=None) -> List[EmbeddingVector]:
        raw_vectors = await self._impl.aencode(texts)
        return [EmbeddingVector(dense_vector=v) for v in raw_vectors]
```

---

## 6. Service 层入口

对齐架构文档第 10.1 节 Service 层 API。

```python
class ContextEngineService:
    """对外 API 入口 — 写入链路部分"""

    def __init__(self, coordinator: CommitCoordinator):
        self._coordinator = coordinator

    async def commit_session(
        self,
        ctx: RequestContext,
        messages: List[Dict],
        used_contexts: Optional[List[str]] = None,
        used_tools: Optional[List[Dict]] = None,
        options: Optional[CommitOptions] = None,
    ) -> CommitResult:
        """写入主链路入口，委托给 CommitCoordinator"""
        return await self._coordinator.commit(
            ctx, messages, used_contexts, used_tools, options
        )

    def commit_session_sync(
        self,
        ctx: RequestContext,
        messages: List[Dict],
        **kwargs,
    ) -> CommitResult:
        """同步 wrapper"""
        import asyncio
        return asyncio.run(self.commit_session(ctx, messages, **kwargs))
```

---

## 7. ContextFS API 摘要（写入链路引用）

写入链路中 `PolicyRouter` 和 `ContextWriter` 直接依赖 `ContextFS`。以下是写入链路用到的 FS 接口，与架构文档 10.2 对齐。

```python
class ContextFS(Protocol):
    """文件系统主存接口 — 写入链路使用的子集"""

    async def resolve_path(self, ctx: RequestContext, uri: str) -> str: ...
    async def exists(self, ctx: RequestContext, uri: str) -> bool: ...
    async def stat_node(self, ctx: RequestContext, uri: str) -> NodeStat: ...
    async def write_node(self, ctx: RequestContext, node: ContextNode) -> None: ...
    async def read_node(self, ctx: RequestContext, uri: str, include_content: bool = True) -> ContextNode: ...
    async def write_level(self, ctx: RequestContext, uri: str, level: str, text: str) -> None: ...
    async def write_metadata(self, ctx: RequestContext, uri: str, metadata: Dict) -> None: ...
    async def list_children(self, ctx: RequestContext, uri: str, recursive: bool = False, depth: int = 1) -> List[NodeSummary]: ...
```

---

## 8. RelationStore 接口

对齐架构文档第 10.3 节，将关系存储层与 `RelationWriter` 编排层分离。

### 8.1 接口定义

```python
class RelationStore(Protocol):
    """关系边存储接口 — 对齐 ce_architecture 10.3，基于 FS"""

    async def get_edges(self, ctx: RequestContext, uri: str) -> List[RelationEdge]: ...

    async def upsert_edges(self, ctx: RequestContext, uri: str, edges: List[RelationEdge]) -> None: ...

    async def append_edges(self, ctx: RequestContext, uri: str, edges: List[RelationEdge]) -> None: ...

    async def remove_edges(
        self,
        ctx: RequestContext,
        uri: str,
        to_uris: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> int: ...
```

### 8.2 FS 实现

`RelationStore` 基于 ContextFS 操作 `.relations.json` 文件。

```python
class FSRelationStore:
    """基于 FS 的 RelationStore 实现"""

    def __init__(self, context_fs: ContextFS):
        self._fs = context_fs

    async def get_edges(self, ctx, uri) -> List[RelationEdge]:
        fs_path = await self._fs.resolve_path(ctx, uri)
        return self._read_relations_json(fs_path)

    async def upsert_edges(self, ctx, uri, edges) -> None:
        existing = await self.get_edges(ctx, uri)
        merged = self._merge_edges(existing, edges)
        await self._write_relations_json(ctx, uri, merged)

    async def append_edges(self, ctx, uri, edges) -> None:
        existing = await self.get_edges(ctx, uri)
        combined = existing + edges
        await self._write_relations_json(ctx, uri, combined)

    async def remove_edges(self, ctx, uri, to_uris=None, relation_types=None) -> int:
        existing = await self.get_edges(ctx, uri)
        filtered = [e for e in existing
                    if not self._should_remove(e, to_uris, relation_types)]
        removed = len(existing) - len(filtered)
        await self._write_relations_json(ctx, uri, filtered)
        return removed
```

---

## 9. ArchiveBuilder

### 9.1 职责

将当前会话的消息序列压缩为 `SessionArchive`，作为后续候选抽取的输入上下文。归档持久化到 FS。

### 9.2 类设计

```python
class ArchiveBuilder:
    def __init__(self, llm: LLM, context_fs: ContextFS, config: ArchiveConfig):
        self._llm = llm
        self._fs = context_fs
        self._config = config

    async def build_archive(
        self,
        ctx: RequestContext,
        messages: List[Dict],
        used_contexts: Optional[List[str]] = None,
        used_tools: Optional[List[Dict]] = None,
    ) -> SessionArchive:
        ...
```

### 9.3 处理流程

```text
                          ┌──────────────────────┐
                          │     messages 输入      │
                          └──────────┬───────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 1. 裁剪消息                     │
                     │    取最近 N 条                   │
                     │    (config.max_recent_messages)  │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 2. LLM 生成摘要                 │
                     │    prompt: archive_summary      │
                     │    输入: messages + tools        │
                     │    输出: summary JSON            │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 3. 构造 SessionArchive          │
                     │    填充 summary, stats 等字段    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 4. 持久化归档到 FS（可选）       │
                     │    写 archive_uri 对应目录      │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   return Archive      │
                          └──────────────────────┘
```

### 9.4 Prompt 模板

```yaml
id: compression.archive_summary
template: |
  你是一个对话分析专家。请对以下会话内容生成结构化摘要。

  ## 会话消息
  {{ messages }}

  {% if tools %}
  ## 使用的工具
  {{ tools }}
  {% endif %}

  ## 输出要求
  以 JSON 格式输出：
  {
    "summary": "2-3 句话概括本次对话的核心内容和结论",
    "key_topics": ["主题1", "主题2"],
    "key_decisions": ["决策1", "决策2"],
    "unresolved": ["待解决问题1"]
  }
```

### 9.5 配置

```python
@dataclass
class ArchiveConfig:
    max_recent_messages: int = 50
    persist_archive: bool = True
    summary_max_tokens: int = 500
```

---

## 10. CandidatePipeline 与 Extractor

### 10.1 类设计

```python
class CandidateExtractor(ABC):
    """候选抽取器标准接口 — 对齐 ce_architecture 10.5"""

    @abstractmethod
    async def extract(
        self,
        messages: List[Dict],
        archive: SessionArchive,
        ctx: RequestContext,
    ) -> List[CandidateMemory]:
        ...


class CandidatePipeline:
    def __init__(self, extractors: List[CandidateExtractor], config: PipelineConfig):
        self._extractors = extractors
        self._config = config

    async def extract_candidates(
        self, ctx: RequestContext, archive: SessionArchive, messages: List[Dict],
    ) -> List[CandidateMemory]:
        ...

    async def normalize_candidates(
        self, ctx: RequestContext, candidates: List[CandidateMemory],
    ) -> List[CandidateMemory]:
        ...
```

### 10.2 处理流程

```text
                    ┌──────────────────────────────┐
                    │  archive + messages 输入       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │         1. 并行调用所有 Extractor（asyncio.gather）│
        │                                                  │
        │  ┌────────────┐ ┌────────────┐ ┌─────────────┐ ┌─────────────┐
        │  │  Profile   │ │ Preference │ │ EntityEvent │ │  Procedure  │
        │  │ Extractor  │ │ Extractor  │ │  Extractor  │ │  Extractor  │
        │  │            │ │            │ │             │ │             │
        │  │ 身份/职业  │ │ 编码风格   │ │ 项目/人/事件│ │ 案例/模式   │
        │  │ 技术栈     │ │ UI偏好     │ │ 组织/里程碑 │ │ 技能/工具   │
        │  │ 工作习惯   │ │ 禁忌       │ │             │ │             │
        │  └──────┬─────┘ └──────┬─────┘ └──────┬──────┘ └──────┬──────┘
        │         │              │              │              │
        │         └──────────────┴──────┬───────┴──────────────┘
        └───────────────────────────────┼──────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────┐
                     │ 2. 扁平化 + 过滤低置信度候选       │
                     │    confidence >= min_confidence    │
                     └───────────────┬──────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │ 3. normalize_candidates           │
                     │    ├─ 按 (category, routing_key)  │
                     │    │   分组                        │
                     │    ├─ 同组内合并重复候选            │
                     │    │   (保留 confidence 最高的)     │
                     │    └─ 截断 content 超长部分         │
                     └───────────────┬──────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │  return CandidateMemory[]         │
                     └──────────────────────────────────┘
```

### 10.3 四类 Extractor

| 抽取器 | 关注内容 | URI 映射 | 典型策略 |
|--------|----------|----------|----------|
| `ProfileExtractor` | 身份、职业、技术栈、工作习惯 | `memories/profile`（固定） | ProfilePolicy (merge) |
| `PreferenceExtractor` | 编码风格、UI 偏好、禁忌 | `memories/preferences/{slug}` | AggregateTopicPolicy |
| `EntityEventExtractor` | 项目、人、组织、里程碑 | `memories/entities/{slug}` / `events/{id}` | Aggregate / AppendOnly |
| `ProcedureExtractor` | 案例、模式、技能工具 | `cases/{id}` / `patterns/{slug}` / `skills/{name}` | AppendOnly / Aggregate / SkillTool |

### 10.4 Prompt 模板

```yaml
id: extraction.memory_extraction
template: |
  你是一个记忆抽取专家。从以下会话中提取值得长期记忆的信息。

  ## 会话摘要
  {{ summary }}

  ## 最近消息
  {{ messages }}

  {% if tools %}
  ## 工具使用
  {{ tools }}
  {% endif %}

  ## 抽取规则
  1. 分为以下类别：profile, preferences, entities, events, cases, patterns, skills
  2. 每条记忆必须包含三级结构：
     - abstract: 1-2 句话精炼摘要（用于首轮检索）
     - overview: 结构化要点概览（用于快速浏览）
     - content: 完整详细叙述（用于深度阅读）
  3. 高召回优于高精度 — 宁可多提取，下游有去重机制
  4. 避免相对时间表达（"昨天"），使用绝对日期
  5. 保留具体数值、名称、版本号等细节

  ## 输出格式
  以 JSON 输出，每类一个数组。
```

### 10.5 配置

```python
@dataclass
class PipelineConfig:
    min_confidence: float = 0.5
    max_content_length: int = 5000
    max_candidates_per_commit: int = 20
    extraction_concurrency: int = 4
```

---

## 11. PolicyRouter 与策略层

### 11.1 类设计

```python
class MergePolicy(ABC):
    """写入策略标准接口 — 对齐 ce_architecture 10.4"""

    @abstractmethod
    async def plan(
        self,
        candidate: CandidateMemory,
        existing_node: Optional[ContextNode],
        ctx: RequestContext,
    ) -> WritePlan:
        ...


class PolicyRouter:
    def __init__(
        self,
        policies: Dict[str, MergePolicy],
        context_fs: ContextFS,
    ):
        self._policies = policies
        self._fs = context_fs

    def select_policy(self, candidate: CandidateMemory, ctx: RequestContext) -> MergePolicy:
        ...

    async def build_plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        ...
```

### 11.2 路由规则

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         PolicyRouter 路由映射                        │
│                                                                      │
│  CandidateMemory.category ─────────────────────► MergePolicy         │
│                                                                      │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ profile  │───▶│ ProfilePolicy   │───▶│ 固定 URI + merge         │ │
│  └──────────┘    └─────────────────┘    └──────────────────────────┘ │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │preferences│──▶│ AggregateTopic  │───▶│ 按 routing_key 聚合      │ │
│  │entities  │──▶│ Policy          │    │ 去重后 merge 或 create    │ │
│  │patterns  │──▶│                 │    │                          │ │
│  └──────────┘    └─────────────────┘    └──────────────────────────┘ │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ events   │───▶│ AppendOnly      │───▶│ 每次 create 新节点       │ │
│  │ cases    │───▶│ Policy          │    │                          │ │
│  └──────────┘    └─────────────────┘    └──────────────────────────┘ │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ skills   │───▶│ SkillToolPolicy │───▶│ 固定 URI + 统计累加      │ │
│  └──────────┘    └─────────────────┘    │ + LLM 合并指南           │ │
│                                          └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

对应的代码映射：

```python
CATEGORY_POLICY_MAP = {
    MemoryCategory.PROFILE:     "profile",
    MemoryCategory.PREFERENCES: "aggregate_topic",
    MemoryCategory.ENTITIES:    "aggregate_topic",
    MemoryCategory.EVENTS:      "append_only",
    MemoryCategory.CASES:       "append_only",
    MemoryCategory.PATTERNS:    "aggregate_topic",
    MemoryCategory.SKILLS:      "skill_tool",
}
```

### 11.3 build_plan 流程

```text
                    ┌──────────────────────────┐
                    │  candidate 输入            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                  ┌────────────────────────────────┐
                  │ 1. 计算目标 URI                  │
                  │    resolve_target_uri(candidate) │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ 2. 查询已有节点（从 FS 读取）     │
                  │    fs.exists(uri) ?              │
                  │    ├─ 是 → fs.read_node(uri)    │
                  │    └─ 否 → existing = None      │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ 3. 选择策略                      │
                  │    policy = select_policy(       │
                  │      candidate.category)         │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ 4. 执行策略                      │
                  │    plan = policy.plan(           │
                  │      candidate, existing, ctx)   │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ 5. return WritePlan              │
                  └────────────────────────────────┘
```

### 11.4 四类策略

| 策略 | 适用类别 | 默认动作 |
|------|----------|----------|
| `ProfilePolicy` | `profile` | 固定 URI + merge |
| `AggregateTopicPolicy` | `preferences` / `entities` / `patterns` | 按 `routing_key` 聚合，去重后 merge 或 create |
| `AppendOnlyPolicy` | `events` / `cases` | 每次 create 新节点 |
| `SkillToolPolicy` | `skills` | 固定 URI + 统计累加 + LLM 合并指南 |

### 11.5 CandidateDeduplicator

去重器通过 `VectorIndex` provider 和 `ContextFS` 查询索引副本和主存，不直接依赖 `storage` 包。索引副本不可用时优雅降级为 CREATE。

```text
                    ┌──────────────────────────┐
                    │  candidate 输入            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                  ┌────────────────────────────────┐
                  │ 1. 向量预筛选（读索引副本）        │
                  │    embed(candidate.abstract)     │
                  │    vector_index.search(top_k=5)  │
                  │    ※ 副本不可用时降级为 CREATE     │
                  └────────────────┬───────────────┘
                                   │
                          ┌────────┴────────┐
                          │                 │
                    无相似结果          有相似结果
                    或 score <          score >=
                    threshold          threshold
                          │                 │
                          ▼                 ▼
                   ┌─────────────┐  ┌───────────────────┐
                   │ CREATE      │  │ 2. LLM 精确判断    │
                   │ (无需去重)   │  │    (可配置关闭)     │
                   └─────────────┘  └─────────┬─────────┘
                                              │
                                    ┌─────────┼──────────┐
                                    │         │          │
                                    ▼         ▼          ▼
                              ┌────────┐ ┌────────┐ ┌──────────┐
                              │ CREATE │ │ SKIP   │ │  MERGE   │
                              │        │ │ (重复) │ │ EXISTING │
                              └────────┘ └────────┘ └──────────┘
```

```python
class CandidateDeduplicator:
    def __init__(self, embedder: Embedder, vector_index: VectorIndex,
                 context_fs: ContextFS, llm: LLM, config: DedupConfig):
        self._embedder = embedder
        self._vector_index = vector_index
        self._fs = context_fs
        self._llm = llm
        self._config = config

    async def check(self, candidate: CandidateMemory, ctx: RequestContext) -> DedupResult:
        # 1. 向量预筛选（读索引副本，不可用时降级）
        try:
            embedding = await self._embedder.embed_texts([candidate.abstract])
            similar = await self._vector_index.search(
                query_vector=embedding[0].dense_vector,
                filters={"category": candidate.category.value, "level": 0},
                top_k=self._config.dedup_top_k,
            )
        except Exception:
            return DedupResult(decision=DedupDecision.CREATE,
                               merge_target=None, reason="索引副本不可用，降级为 CREATE")

        if not similar or similar[0]["score"] < self._config.similarity_threshold:
            return DedupResult(decision=DedupDecision.CREATE,
                               merge_target=None, reason="无相似记忆")

        # 2. 从 FS 主存读取匹配节点
        top_match_uri = similar[0]["uri"]
        existing_node = await self._fs.read_node(ctx, top_match_uri)

        # 3. LLM 精确判断（可配置关闭）
        if not self._config.enable_llm_decision:
            return DedupResult(decision=DedupDecision.MERGE_EXISTING,
                               merge_target=existing_node, reason="向量相似度超过阈值")

        decision = await self._llm.complete_json(
            prompt=render("policy.dedup_decision", ...),
            schema=DedupDecisionSchema,
        )
        ...
```

### 11.6 Prompt 模板（策略层）

```yaml
# prompts/templates/policy/merge_bundle.yaml
id: policy.merge_bundle
template: |
  你是一个记忆合并专家。将新信息合并到已有记忆中，产出更新后的三级内容。

  ## 已有记忆
  ### Abstract (L0)
  {{ existing_abstract }}
  ### Overview (L1)
  {{ existing_overview }}
  ### Content (L2)
  {{ existing_content }}

  ## 新信息
  ### Abstract (L0)
  {{ new_abstract }}
  ### Overview (L1)
  {{ new_overview }}
  ### Content (L2)
  {{ new_content }}

  ## 合并规则
  1. 保留已有记忆中仍然正确的信息
  2. 用新信息更新过时的内容
  3. 补充新信息中新增的内容
  4. 如果存在冲突，以新信息为准并注明
  5. 类别：{{ category }}

  ## 输出格式
  {
    "abstract": "合并后的 1-2 句话摘要",
    "overview": "合并后的结构化概览",
    "content": "合并后的完整叙述",
    "reason": "合并理由说明"
  }
```

```yaml
# prompts/templates/policy/dedup_decision.yaml
id: policy.dedup_decision
template: |
  判断以下新候选记忆是否与已有记忆重复。

  ## 新候选
  {{ candidate_abstract }}
  {{ candidate_content }}

  ## 已有记忆
  {{ existing_abstract }}

  ## 输出
  {
    "decision": "create | skip | merge_existing",
    "reason": "判断理由"
  }
```

---

## 12. ContextWriter

### 12.1 类设计

```python
class ContextWriter:
    """主存写入器 — 将 WritePlan 写入 ContextFS，纯 FS 操作"""

    def __init__(self, context_fs: ContextFS):
        self._fs = context_fs

    async def apply_plan(self, ctx: RequestContext, plan: WritePlan) -> WriteResult: ...
    async def merge_node(self, ctx: RequestContext, uri: str, patch: Dict[str, str]) -> WriteResult: ...
    async def append_node(self, ctx: RequestContext, uri: str, patch: Dict[str, str]) -> WriteResult: ...
```

### 12.2 apply_plan 流程

```text
                    ┌──────────────────────────┐
                    │   WritePlan 输入           │
                    └────────────┬─────────────┘
                                 │
                      ┌──────────┴──────────────────────────┐
                      │         plan.action 判断              │
                      ├──────────┬────────────┬──────────────┤
                      │          │            │              │
                      ▼          ▼            ▼              ▼
                 ┌────────┐ ┌────────┐  ┌────────┐    ┌────────┐
                 │  SKIP  │ │ CREATE │  │ MERGE  │    │ APPEND │
                 │        │ │        │  │        │    │        │
                 │直接返回 │ │创建新  │  │合并到  │    │追加到  │
                 │        │ │节点    │  │已有节点│    │已有节点│
                 └────────┘ └───┬────┘  └───┬────┘    └───┬────┘
                                │           │             │
                                └─────┬─────┴─────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │  FS 写入（幂等可重试）                   │
                   │  context_fs.write_node(ctx, node)      │
                   │  写入 .abstract.md / .overview.md /    │
                   │       content.md / .meta.json          │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │  return WriteResult                    │
                   └───────────────────────────────────────┘
```

---

## 13. RelationWriter

### 13.1 类设计

```python
class RelationWriter:
    """关系边写入编排器 — 调用 RelationStore 写 FS（偏离 D-2: 返回 List[RelationEdge]）"""

    def __init__(self, relation_store: RelationStore):
        self._store = relation_store

    async def apply_edges(
        self,
        ctx: RequestContext,
        uri: str,
        relation_ops: List[RelationOp],
    ) -> List[RelationEdge]:
        all_written = []
        for op in relation_ops:
            if op.op == "upsert":
                await self._store.upsert_edges(ctx, uri, op.edges)
                all_written.extend(op.edges)
            elif op.op == "append":
                await self._store.append_edges(ctx, uri, op.edges)
                all_written.extend(op.edges)
            elif op.op == "remove":
                to_uris = [e.to_uri for e in op.edges]
                await self._store.remove_edges(ctx, uri, to_uris=to_uris)
        return all_written
```

### 13.2 .relations.json 格式

```json
{
  "edges": [
    {
      "id": "a1b2c3d4",
      "to_uri": "ctx://acme/users/u1/memories/entities/project-alpha",
      "relation_type": "related_to",
      "reason": "同一项目上下文",
      "weight": 1.0,
      "created_at": "2026-03-14T10:30:00Z"
    }
  ],
  "updated_at": "2026-03-14T10:30:00Z"
}
```

---

## 14. OutboxStore

### 14.1 类设计

对齐架构文档 10.4（写入链路接口）和 10.6（IndexWorker 接口）。commit 链路中 Outbox 登记为最佳努力操作，失败不阻塞 commit。

```python
class OutboxStore:
    """Outbox 事件存储 — 基于 openGauss ctx.outbox_events 表"""

    def __init__(self, db: 'ContextDB'):
        self._db = db

    # ---- 写入链路接口（10.4）----

    async def append(self, event: OutboxEvent) -> OutboxEvent: ...
    async def append_batch(self, events: List[OutboxEvent]) -> List[OutboxEvent]: ...

    # ---- IndexWorker 接口（10.6）----

    async def claim_batch(self, worker_id: str, limit: int = 100) -> List[OutboxEvent]: ...
    async def mark_done(self, event_id: str) -> None: ...
    async def mark_failed(self, event_id: str, error: str, retryable: bool = True) -> None: ...
```

### 14.2 事件生成规则

```python
def _build_outbox_events(
    write_results: List[WriteResult],
    relation_edges: List[RelationEdge],
) -> List[OutboxEvent]:
    events = []
    for wr in write_results:
        if wr.action == WriteAction.SKIP:
            continue
        events.append(OutboxEvent(
            event_id=str(uuid4()),
            event_type=OutboxEventType.UPSERT_CONTEXT,
            uri=wr.uri,
            payload={"action": wr.action.value, "version": wr.node_version},
        ))
    for edge in relation_edges:
        events.append(OutboxEvent(
            event_id=str(uuid4()),
            event_type=OutboxEventType.UPSERT_RELATION,
            uri=edge.from_uri,
            payload={"to_uri": edge.to_uri, "relation_type": edge.relation_type},
        ))
    return events
```

### 14.3 claim_batch SQL

```sql
UPDATE ctx.outbox_events
SET status = 'claimed', worker_id = %(worker_id)s, updated_at = CURRENT_TIMESTAMP
WHERE event_id IN (
    SELECT event_id FROM ctx.outbox_events
    WHERE status = 'pending'
    ORDER BY created_at ASC
    LIMIT %(limit)s
    FOR UPDATE SKIP LOCKED
)
RETURNING event_id, event_type, uri, payload, retry_count;
```

---

## 15. CommitCoordinator

### 15.1 类设计

```python
class CommitCoordinator:
    def __init__(
        self,
        archive_builder: ArchiveBuilder,
        candidate_pipeline: CandidatePipeline,
        policy_router: PolicyRouter,
        context_writer: ContextWriter,
        relation_writer: RelationWriter,
        outbox_store: OutboxStore,
        config: CommitConfig,
    ):
        self._archive_builder = archive_builder
        self._pipeline = candidate_pipeline
        self._router = policy_router
        self._writer = context_writer
        self._relation_writer = relation_writer
        self._outbox = outbox_store
        self._config = config

    async def commit(
        self,
        ctx: RequestContext,
        messages: List[Dict],
        used_contexts: Optional[List[str]] = None,
        used_tools: Optional[List[Dict]] = None,
        options: Optional[CommitOptions] = None,
    ) -> CommitResult:
        ...
```

### 15.2 完整编排流程

```text
commit(ctx, messages, used_contexts, used_tools, options)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ Phase 1: 归档                                                   │
├──┤ if options.skip_archive: skip                                   │
│  │ archive = archive_builder.build_archive(ctx, messages, ...)     │
│  │ output: SessionArchive（持久化到 FS）                            │
│  └─────────────────────────────────┬───────────────────────────────┘
│                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐
│  │ Phase 2: 候选抽取                                                │
├──┤ candidates = pipeline.extract_candidates(ctx, archive, messages) │
│  │ candidates = pipeline.normalize_candidates(ctx, candidates)      │
│  │ if options.max_candidates: candidates = candidates[:N]           │
│  │ output: CandidateMemory[]                                        │
│  └─────────────────────────────────┬───────────────────────────────┘
│                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐
│  │ Phase 3: 策略规划                                                │
├──┤ for each candidate:                                              │
│  │   plan = policy_router.build_plan(candidate, ctx)                │
│  │   plans.append((candidate, plan))                                │
│  │ active_plans = filter(action != SKIP)                            │
│  │ output: List[(CandidateMemory, WritePlan)]                       │
│  └─────────────────────────────────┬───────────────────────────────┘
│                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐
│  │ Phase 4: FS 写入（主存，幂等可重试）                               │
├──┤ for each (candidate, plan) in active_plans:                      │
│  │   wr = context_writer.apply_plan(ctx, plan)                      │
│  │   write_results.append(wr)                                       │
│  │   if plan.relation_ops and not options.skip_relations:           │
│  │     edges = relation_writer.apply_edges(ctx, uri, ops)           │
│  │     all_relation_edges.extend(edges)                             │
│  │ output: WriteResult[], RelationEdge[]                            │
│  │                                                                  │
│  │ ★ FS 写入成功 → commit 视为成功                                   │
│  └─────────────────────────────────┬───────────────────────────────┘
│                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐
│  │ Phase 5: Outbox 登记 + 构造结果（最佳努力，不阻塞 commit）        │
├──┤ outbox_events = _build_outbox_events(write_results, edges)       │
│  │ try:                                                             │
│  │   await outbox_store.append_batch(outbox_events)                 │
│  │ except Exception:                                                │
│  │   log.warning("Outbox 登记失败，repair job 将补偿")               │
│  │   outbox_events = []                                             │
│  └─────────────────────────────────┬───────────────────────────────┘
│                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐
│  │ Phase 6: 构造结果                                                │
└──┤ return CommitResult(archive, write_results, outbox_events,       │
   │                     stats=CommitStats(...), status="success")    │
   └─────────────────────────────────────────────────────────────────┘
```

### 15.3 错误处理策略

```text
┌─────────────────┬──────────────────────┬────────────────────────────────────────┐
│   失败阶段       │       影响            │             处理方式                    │
├─────────────────┼──────────────────────┼────────────────────────────────────────┤
│                 │                      │                                        │
│  Phase 1-3      │  无副作用             │  直接抛出 CommitError                   │
│  (归档/抽取/    │  (尚未写入任何数据)    │  上层可重试整个 commit                  │
│   策略规划)      │                      │                                        │
│                 │                      │                                        │
├─────────────────┼──────────────────────┼────────────────────────────────────────┤
│                 │                      │                                        │
│  Phase 4        │  部分 FS 文件已写入    │  跳过失败的节点                         │
│  (FS 写入)      │  对应节点标记为失败    │  已写入的 FS 文件保留（幂等可重试）       │
│                 │                      │  status = "partial"                    │
│                 │                      │                                        │
├─────────────────┼──────────────────────┼────────────────────────────────────────┤
│                 │                      │                                        │
│  Phase 5        │  索引副本不同步        │  不阻塞 commit（status 仍为 success）   │
│  (Outbox 登记)  │  Outbox 事件缺失      │  repair job 扫描 FS 补发缺失事件        │
│                 │                      │                                        │
└─────────────────┴──────────────────────┴────────────────────────────────────────┘
```

```python
class CommitError(Exception):
    def __init__(self, message: str, phase: str, partial_results: Optional[List[WriteResult]] = None):
        super().__init__(message)
        self.phase = phase
        self.partial_results = partial_results
```

### 15.4 配置

```python
@dataclass
class CommitConfig:
    max_candidates_per_commit: int = 20
    skip_archive_on_short_session: bool = True
    short_session_threshold: int = 3
    enable_dedup: bool = True
    enable_relations: bool = True
    dry_run: bool = False
```

---

## 16. 索引副本异步同步（IndexWorker）

commit 链路不直接写入 DB 索引副本。以下描述 IndexWorker 消费 Outbox 事件后如何从 FS 主存构建索引副本，对齐架构文档 §11.2。

### 16.1 IndexWorker 流程

```text
IndexWorker.run_once
│
├─ 1. OutboxStore.claim_batch(worker_id, limit)
│     → events[]
│
├─ 2. for each event:
│     │
│     ├─ UPSERT_CONTEXT:
│     │   ├─ ContextFS.read_node(uri)          ← 从 FS 主存读取
│     │   ├─ DocumentChunker.chunk_for_node()   ← 分片（L0/L1/L2）
│     │   ├─ Embedder.embed_texts()            ← 生成向量
│     │   ├─ VectorIndex.upsert(records)       ← 写入 DB 索引副本
│     │   └─ OutboxStore.mark_done()
│     │
│     ├─ UPSERT_RELATION:
│     │   ├─ RelationStore.get_edges(uri)      ← 从 FS 读关系
│     │   ├─ GraphStore.upsert_edges(edges)    ← 写入 DB 关系副本
│     │   └─ OutboxStore.mark_done()
│     │
│     ├─ DELETE_CONTEXT:
│     │   ├─ VectorIndex.delete_by_uri(uri)
│     │   └─ OutboxStore.mark_done()
│     │
│     └─ MOVE_CONTEXT:
│         ├─ VectorIndex.move_uri(from, to)
│         └─ OutboxStore.mark_done()
│
└─ 3. return IndexBatchResult(claimed, succeeded, failed)
```

### 16.2 DocumentChunker（由 IndexWorker 调用）

```python
class DocumentChunker:
    """文本分片器 — 将 ContextNode 的三级内容分片为 ChunkRecord"""

    def __init__(self, config: ChunkingConfig):
        self._config = config

    def chunk_for_node(self, node: ContextNode) -> List[ChunkRecord]:
        chunks = []

        # L0: abstract → 1 chunk
        chunks.append(ChunkRecord(
            chunk_id=stable_hash(node.uri, 0, 0),
            uri=node.uri, level=0, chunk_index=0,
            text_content=node.abstract,
            category=node.category.value,
            owner_space=node.owner_space,
        ))

        # L1: overview → 1 chunk
        if node.overview:
            chunks.append(ChunkRecord(
                chunk_id=stable_hash(node.uri, 1, 0),
                uri=node.uri, level=1, chunk_index=0,
                text_content=node.overview,
                category=node.category.value,
                owner_space=node.owner_space,
            ))

        # L2: content → 按配置分片
        if node.content:
            for i, text in enumerate(self._split_content(node.content)):
                chunks.append(ChunkRecord(
                    chunk_id=stable_hash(node.uri, 2, i),
                    uri=node.uri, level=2, chunk_index=i,
                    text_content=text,
                    category=node.category.value,
                    owner_space=node.owner_space,
                ))
        return chunks
```

### 16.3 Repair Job

当 Outbox 事件丢失（Phase 5 登记失败）时，repair job 通过扫描 FS 主存来补偿：

```text
repair_job.run()
│
├─ 1. 扫描 FS 中所有 ContextNode
│
├─ 2. 对比 DB 索引副本中的 context_nodes
│     找出 FS 有但 DB 无记录 或 版本不一致的节点
│
├─ 3. 为缺失/过期的节点生成 UPSERT_CONTEXT 事件
│
└─ 4. 写入 outbox_events 表，由 IndexWorker 正常消费
```

---

## 17. 包结构与文件布局

### 17.1 完整包目录

```text
src/
├── service/
│   ├── __init__.py
│   └── context_engine_service.py        # ContextEngineService (对外 API 入口)
│
├── commit/
│   ├── __init__.py                      # 导出 CommitCoordinator, CommitResult
│   ├── archive_builder.py              # ArchiveBuilder
│   ├── candidate_pipeline.py           # CandidatePipeline
│   ├── policy_router.py               # PolicyRouter
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── base.py                    # MergePolicy ABC
│   │   ├── profile_policy.py
│   │   ├── aggregate_topic_policy.py
│   │   ├── append_only_policy.py
│   │   └── skill_tool_policy.py
│   ├── context_writer.py              # ContextWriter（纯 FS 写入）
│   ├── relation_writer.py             # RelationWriter
│   ├── deduplicator.py                # CandidateDeduplicator
│   └── commit_coordinator.py          # CommitCoordinator
│
├── extraction/
│   ├── __init__.py
│   ├── base.py                        # CandidateExtractor ABC
│   ├── profile_extractor.py
│   ├── preference_extractor.py
│   ├── entity_event_extractor.py
│   └── procedure_extractor.py
│
├── index/
│   ├── __init__.py
│   ├── outbox_store.py                # OutboxStore
│   ├── document_chunker.py            # DocumentChunker（由 IndexWorker 调用）
│   ├── index_worker.py                # IndexWorker
│   └── index_projector.py             # IndexProjector
│
├── core/
│   ├── __init__.py
│   ├── models.py                      # 数据模型
│   ├── enums.py                       # MemoryCategory, WriteAction, OutboxEventType 等
│   ├── errors.py                      # CommitError 等
│   └── config.py                      # CommitConfig, ArchiveConfig, PipelineConfig 等
│
├── fs/
│   ├── __init__.py
│   ├── context_fs.py                  # ContextFS 实现
│   └── relation_store.py              # FSRelationStore (RelationStore 实现)
│
├── storage/
│   ├── __init__.py
│   ├── vector_db.py                   # 现有 OpenGaussVectorDB
│   └── context_db.py                  # ContextDB（仅 IndexWorker 和 OutboxStore 使用）
│
├── providers/
│   ├── __init__.py
│   ├── llm.py                         # LLM Protocol
│   ├── embedder.py                    # Embedder Protocol + EmbeddingVector
│   ├── vector_index.py                # VectorIndex Protocol
│   └── adapters/
│       ├── __init__.py
│       ├── openai_adapter.py          # OpenAIEmbedderAdapter
│       └── opengauss_vector.py        # OpenGaussVectorIndex（VectorIndex 实现）
│
└── prompts/
    └── templates/
        ├── compression/
        │   └── archive_summary.yaml
        ├── extraction/
        │   ├── profile.yaml
        │   ├── preferences.yaml
        │   ├── entity_event.yaml
        │   └── procedure.yaml
        └── policy/
            ├── merge_bundle.yaml
            └── dedup_decision.yaml
```

### 17.2 依赖关系

```text
                        ┌──────────┐
                        │   core   │  ← 无外部业务依赖（纯数据模型和枚举）
                        └────┬─────┘
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │ providers │   │    fs     │   │  storage  │
      │ (协议接口) │   │(文件主存) │   │ (DB 操作) │
      └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
            │               │               │
            ├───────┐       │               │
            │       │       │               │
            ▼       │       │               │
     ┌────────────┐ │       │               │
     │ extraction │ │       │               │
     │ (候选抽取) │ │       │               │
     └──────┬─────┘ │       │               │
            │       │       │               │
            └───────┤       │               │
                    │       │               │
                    ▼       ▼               │
            ┌────────────────────┐          │
            │      commit        │          │
            │  依赖: core, fs,   │          │
            │  extraction,       │          │
            │  index, providers  │          │
            │  ※ 不依赖 storage  │          │
            └────────┬───────────┘          │
                     │                      │
                     ▼                      │
            ┌──────────────────┐            │
            │     service      │            │
            └──────────────────┘            │
                                            │
            ┌───────────────────────────────┤
            │                               │
            ▼                               ▼
     ┌─────────────────────────────────────────┐
     │     index (IndexWorker)                  │
     │  依赖: core, fs, storage, providers      │
     │  ※ 从 FS 读主存 → 写 DB 索引副本          │
     └─────────────────────────────────────────┘
```

**关键设计**：`commit` 包不依赖 `storage`，完全通过 FS 主存完成写入。`storage` 仅被 `index` 包和 `OutboxStore` 使用。

### 17.3 与现有代码的关系

| 现有模块 | 处理方式 |
|----------|----------|
| `src/storage/vector_db.py` | 保留不动，`ContextDB` 独立新增 |
| `src/config.py` | 在 `core/config.py` 中扩展 |
| `src/core/memory_engine.py` | 保留作为 v0.1 兼容层 |
| `src/core/document_processor.py` | `DocumentChunker` 可复用其 Markdown 分片逻辑 |
| `src/embeddings/` | 通过 `providers/adapters/openai_adapter.py` 适配为 `Embedder` 协议 |

---

## 18. 同步/异步建模策略

对齐架构文档第 12 章建议："所有接口先用同步语义建模，再用 async 实现"。

### 18.1 策略

核心接口以 `async def` 定义为主版本（Python 的 psycopg 异步池和 httpx 异步客户端更成熟），每个对外暴露的入口同时提供同步 wrapper：

```python
class ContextEngineService:
    async def commit_session(self, ctx, messages, **kwargs) -> CommitResult: ...

    def commit_session_sync(self, ctx, messages, **kwargs) -> CommitResult:
        import asyncio
        return asyncio.run(self.commit_session(ctx, messages, **kwargs))
```

### 18.2 适用范围

| 层 | 提供同步 wrapper | 理由 |
|----|-----------------|------|
| `service/` | 是 | 对外 API 入口，需支持非 async 调用方 |
| `commit/` | 否 | 内部编排层，只由 service 层调用 |
| `extraction/` | 否 | LLM 调用本身是 IO 密集型，async 更合适 |
| `providers/` | 是（LLM.complete_json_sync） | 可能被测试或脚本直接调用 |
| `index/outbox_store` | 否 | 仅 CommitCoordinator 和 IndexWorker 调用 |

---

## 19. 典型时序图

### 19.1 完整 Session Commit 时序

```text
Client              Service         Coordinator      Archive    Pipeline     Router      Writer     RelStore   Outbox
  │                    │                │               │           │           │           │          │          │
  │  commit_session    │                │               │           │           │           │          │          │
  ├───────────────────>│                │               │           │           │           │          │          │
  │                    │   commit()     │               │           │           │           │          │          │
  │                    ├───────────────>│               │           │           │           │          │          │
  │                    │                │ build_archive  │           │           │           │          │          │
  │                    │                ├──────────────>│           │           │           │          │          │
  │                    │                │               │ LLM call  │           │           │          │          │
  │                    │                │               ├──────┐    │           │           │          │          │
  │                    │                │               │<─────┘    │           │           │          │          │
  │                    │                │  archive      │           │           │           │          │          │
  │                    │                │<──────────────┤           │           │           │          │          │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ extract_candidates        │           │           │          │          │
  │                    │                ├──────────────────────────>│           │           │          │          │
  │                    │                │               │  (4x parallel)        │           │          │          │
  │                    │                │               │           ├──────┐    │           │          │          │
  │                    │                │               │           │<─────┘    │           │          │          │
  │                    │                │  candidates[] │           │           │           │          │          │
  │                    │                │<─────────────────────────┤           │           │          │          │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ build_plan    │           │           │           │          │          │
  │                    │                ├──────────────────────────────────────>│           │          │          │
  │                    │                │               │           │  plan     │           │          │          │
  │                    │                │<─────────────────────────────────────┤           │          │          │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ apply_plan (FS only)      │           │           │          │          │
  │                    │                ├──────────────────────────────────────────────────>│          │          │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ apply_edges (FS only)     │           │           │          │          │
  │                    │                ├──────────────────────────────────────────────────────────────>│         │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ ★ FS 写入成功 = commit 成功                       │          │          │
  │                    │                │               │           │           │           │          │          │
  │                    │                │ append_batch (best-effort) │           │           │          │          │
  │                    │                ├─────────────────────────────────────────────────────────────────────────>│
  │                    │                │               │           │           │           │          │          │
  │                    │  CommitResult  │               │           │           │           │          │          │
  │                    │<───────────────┤               │           │           │           │          │          │
  │  CommitResult      │               │               │           │           │           │          │          │
  │<───────────────────┤               │               │           │           │           │          │          │
```

### 19.2 异步索引副本同步时序（IndexWorker）

对齐架构文档 §11.2。IndexWorker 从 FS 主存读取 → 分片 → 嵌入 → 写入 DB 索引副本。

```text
IndexWorker       OutboxStore       ContextFS       Chunker        Embedder       VectorIndex
     │                 │                │               │               │               │
     │  claim_batch    │                │               │               │               │
     ├────────────────>│                │               │               │               │
     │  events[]       │                │               │               │               │
     │<────────────────┤                │               │               │               │
     │                 │                │               │               │               │
     │  read_node(uri) │                │               │               │               │
     ├─────────────────────────────────>│               │               │               │
     │  ContextNode    │                │               │               │               │
     │<─────────────────────────────────┤               │               │               │
     │                 │                │               │               │               │
     │  chunk_for_node(node)            │               │               │               │
     ├──────────────────────────────────────────────────>│               │               │
     │  ChunkRecord[]  │                │               │               │               │
     │<─────────────────────────────────────────────────┤               │               │
     │                 │                │               │               │               │
     │  embed_texts(chunk_texts)        │               │               │               │
     ├──────────────────────────────────────────────────────────────────>│               │
     │  vectors[]      │                │               │               │               │
     │<─────────────────────────────────────────────────────────────────┤               │
     │                 │                │               │               │               │
     │  upsert(records_with_embeddings) │               │               │               │
     ├──────────────────────────────────────────────────────────────────────────────────>│
     │                 │                │               │               │               │
     │  mark_done      │                │               │               │               │
     ├────────────────>│                │               │               │               │
```

---

## 20. 实现优先级建议

| 阶段 | 目标 | 涉及模块 |
|------|------|----------|
| P0 | 核心数据模型 + ContextFS + Provider 协议 | `core/models.py`, `core/enums.py`, `fs/context_fs.py`, `providers/llm.py`, `providers/embedder.py` |
| P1 | CommitCoordinator 最小可用 + 单 Extractor + ContextWriter | `commit/commit_coordinator.py`, `commit/context_writer.py`, `extraction/profile_extractor.py` |
| P2 | 完整 4 类 Extractor + PolicyRouter + Service 入口 | `extraction/*`, `commit/policy_router.py`, `commit/policies/*`, `service/context_engine_service.py` |
| P3 | RelationStore + RelationWriter + OutboxStore + 去重 | `fs/relation_store.py`, `commit/relation_writer.py`, `index/outbox_store.py`, `commit/deduplicator.py` |
| P4 | openGauss Schema + IndexWorker + DocumentChunker | `storage/context_db.py`, `index/index_worker.py`, `index/document_chunker.py` |
| P5 | Prompt 模板体系 + ArchiveBuilder + Repair Job | `prompts/templates/*`, `commit/archive_builder.py`, repair job |
