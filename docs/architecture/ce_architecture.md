# ContextEngine 架构细化文档

> 相关专题设计：
> - `openclaw_compaction_takeover_design.md`：OpenClaw compaction takeover 设计，定义 compaction ownership、pre-compaction extraction 与 `CompactResult` 契约

> 说明：本文基于 `architecture.md`、`ContextEngine.docx` 和 `ov_architecture.md` 细化 ContextEngine 的高层架构，重点补齐模块边界、跨组件交互流程和函数级 API 契约。本文不修改 `ContextEngine.docx`，只为后续开发提供稳定的实现蓝图。

## 1. 设计目标

ContextEngine 的目标是建立一套以文件系统为主存、以异步索引为副本、以分层检索为默认访问方式的长期记忆数据库。

| 目标维度 | 设计要求 |
|----------|----------|
| 主存清晰 | 所有真实内容、摘要、概览、元数据、关系都落到本地文件系统 |
| 写入可控 | 不同类型记忆通过不同策略写入，不再依赖单一 add/update/delete |
| 检索可解释 | 先 `Query Planning`，再 `L0 Seed Retrieval`，最后显式读取 L1/L2 |
| 一致性可恢复 | 主存成功即提交成功；索引和关系副本通过 Outbox 最终一致 |
| 架构可扩展 | 组件边界与函数契约稳定，后续可替换 provider 或实现分布式 worker |

## 2. 非目标

| 非目标 | 说明 |
|--------|------|
| 复杂图推理 | v1 不做图推理，只保留关系副本和可选 Graph provider |
| 批量管理 API | v1 不对外暴露大批量 delete/move/repair |
| 高自治 planner | v1 优先规则化 planner，而不是高度智能 agent planner |
| 强事务跨库一致性 | v1 不追求 FS 与向量库强原子提交，只保证最终一致 |

## 3. 总体架构

### 3.1 核心主线

| 主线 | 核心模块 | 职责 |
|------|----------|------|
| 主存层 | `fs` | 目录节点落盘、URI 映射、层级文件读写、关系文件维护 |
| 写入编排层 | `commit` | 从会话到候选、策略、写主存、Outbox 事件登记 |
| 抽取层 | `extraction` | 从消息/归档中抽取结构化候选记忆 |
| 索引层 | `index` | 消费 Outbox，生成 L0/L1/L2 索引记录并同步向量库 |
| 检索层 | `retrieval` | Planner、首轮召回、层级搜索、按需读取、结果组装 |
| provider 层 | `providers` | LLM、Embedding、Vector、Rerank、Graph 的能力适配 |
| service 层 | `service` | 对外暴露提交、检索、读取、列表等 API |

### 3.2 和 OpenViking 的对齐关系

| OpenViking 组件 | ContextEngine 对齐组件 | 关键差异 |
|-----------------|------------------------|----------|
| `VikingFS` | `ContextFS` | CE 明确以“目录节点”作为唯一主存模型 |
| `SessionCompressor` | `CommitCoordinator` | CE 把 commit 拆成显式策略规划和写入阶段 |
| `MemoryExtractor` | `CandidatePipeline` + `CandidateExtractor` | CE 明确候选抽取接口和多提取器编排 |
| `SemanticQueue` + `EmbeddingQueue` | `OutboxStore` + `IndexWorker` | CE 用 Outbox 统一管理索引副本同步 |
| `IntentAnalyzer` | `QueryPlanner` | CE 默认规则化 planner |
| `HierarchicalRetriever` | `SeedRetriever` + `HierarchicalSearcher` | CE 明确混合召回与递归展开的边界 |

## 4. 核心对象模型

### 4.1 主对象

| 对象 | 用途 | 关键字段 |
|------|------|----------|
| `RequestContext` | 请求级上下文 | `account_id`, `user_id`, `agent_id`, `session_id`, `trace_id` |
| `ContextNode` | 主存节点 | `uri`, `parent_uri`, `context_type`, `category`, `owner_space`, `abstract`, `overview`, `content`, `metadata` |
| `RelationEdge` | 关系边 | `from_uri`, `to_uri`, `relation_type`, `reason`, `weight` |
| `SessionArchive` | 会话归档快照 | `session_id`, `archive_uri`, `summary`, `recent_messages`, `stats` |
| `CandidateMemory` | 候选记忆 | `category`, `owner_scope`, `routing_key`, `abstract`, `overview`, `content`, `confidence`, `source_refs` |
| `WritePlan` | 策略层输出 | `action`, `target_uri`, `node_patch`, `relation_ops`, `outbox_events`, `reason` |
| `WriteResult` | 主存写入结果 | `uri`, `action`, `node_version`, `created`, `updated` |
| `CommitResult` | 一次提交的总结果 | `archive`, `write_results`, `outbox_events`, `stats`, `status` |
| `IndexRecord` | 向量索引副本 | `id`, `uri`, `parent_uri`, `context_type`, `category`, `level`, `text`, `filters`, `metadata` |
| `IndexBatchResult` | 一轮 worker 处理结果 | `claimed`, `succeeded`, `failed`, `retried` |
| `OutboxEvent` | 异步同步事件 | `event_id`, `event_type`, `uri`, `payload`, `status`, `retry_count`, `created_at` |
| `TypedQuery` | 结构化检索请求 | `text`, `context_type`, `categories`, `target_uri`, `top_k`, `filters`, `hints` |
| `SeedHit` | 向量检索命中 | `uri`, `score`, `level`, `abstract`, `category`, `owner_space`, `parent_uri` |
| `RetrievedBlock` | 检索输出块 | `uri`, `level_hit`, `abstract`, `content_excerpt`, `score`, `match_reason` |
| `SearchMemoryResult` | 检索返回集合 | `query`, `results: list[RetrievedBlock]`, `trace` |
| `NodeSummary` | 列表或树视图条目 | `uri`, `name`, `is_dir`, `has_children`, `category`, `updated_at` |
| `NodeStat` | 节点状态 | `uri`, `exists`, `version`, `size`, `updated_at` |
| `MoveProjection` | URI 迁移后的索引映射更新 | `old_uri`, `new_uri`, `old_parent_uri`, `new_parent_uri` |
| `SeedHit` | 向量检索命中 | `id`, `uri`, `score`, `level`, `filters`, `metadata` |
| `EmbeddingVector` | embedding 输出 | `dense_vector`, `sparse_vector`, `model` |
| `RerankScore` | 重排结果 | `uri`, `score`, `reason` |
| `RelationIndexRecord` | 可选关系副本记录 | `id`, `from_uri`, `to_uri`, `relation_type`, `metadata` |

### 4.2 设计约束

| 约束 | 说明 |
|------|------|
| 内容对象和索引对象分离 | `ContextNode` 是主存，`IndexRecord` 是副本 |
| 候选对象和落地对象分离 | `CandidateMemory` 不携带落盘细节，落盘由 `WritePlan` 决定 |
| URI 稳定 | URI 是逻辑主键，Index ID 由 `uri + level` 稳定生成 |
| 目录节点唯一 | v1 不再混用“目录节点”和“文件节点”两套语义 |

## 5. 存储模型与 URI 规范

### 5.1 URI 规范

| 类别 | URI 模式 | 说明 |
|------|----------|------|
| Profile | `ctx://{account}/users/{user}/memories/profile` | 固定 URI，持续 merge |
| Preference | `ctx://{account}/users/{user}/memories/preferences/{slug}` | 按主题聚合 |
| Entity | `ctx://{account}/users/{user}/memories/entities/{slug}` | 按实体聚合 |
| Event | `ctx://{account}/users/{user}/memories/events/{event_id}` | 追加写 |
| Case | `ctx://{account}/agents/{agent}/memories/cases/{case_id}` | 追加写 |
| Pattern | `ctx://{account}/agents/{agent}/memories/patterns/{slug}` | 聚合写 |
| Skill | `ctx://{account}/agents/{agent}/skills/{skill_name}` | 技能聚合 |

### 5.2 目录节点规范

每个 `ContextNode` 统一表现为一个目录：

```text
{node_uri}/
├── .abstract.md
├── .overview.md
├── content.md
├── .meta.json
└── .relations.json
```

### 5.3 主存与物理路径

逻辑 URI 不直接暴露物理路径。`ContextFS` 负责：

1. `ctx://...` 到本地目录的稳定映射。
2. 主目录与语义文件的创建、读取、移动、删除。
3. 元数据和关系文件的统一编码方式。

## 6. 写入链路

### 6.1 目标链路

```text
commit_session
-> ArchiveBuilder
-> CandidatePipeline
-> PolicyRouter
-> MergePolicy.plan
-> ContextWriter
-> RelationWriter
-> OutboxStore
```

### 6.2 详细步骤

| 步骤 | 组件 | 输出 | 说明 |
|------|------|------|------|
| 1 | `ArchiveBuilder` | `SessionArchive` | 提炼本轮会话摘要和检索上下文 |
| 2 | `CandidatePipeline` | `CandidateMemory[]` | 并行调用多个抽取器生成候选记忆 |
| 3 | `PolicyRouter` | `candidate + selected policy` | 按 `category` 选择写入策略 |
| 4 | `MergePolicy` | `WritePlan` | 决定 `create / merge / append / skip` |
| 5 | `ContextWriter` | `WriteResult` | 把节点写入 `ContextFS` |
| 6 | `RelationWriter` | `RelationEdge[]` | 写 `.relations.json` |
| 7 | `OutboxStore` | `OutboxEvent[]` | 记录索引副本同步任务 |

### 6.3 候选抽取器

| 抽取器 | 关注内容 | 典型输出 |
|--------|----------|----------|
| `ProfileExtractor` | 身份、长期背景、稳定画像 | `profile` |
| `PreferenceExtractor` | 偏好、约束、禁忌、风格倾向 | `preferences` |
| `EntityEventExtractor` | 实体、项目、人、离散事件 | `entities`、`events` |
| `ProcedureExtractor` | 案例、模式、技能、工具经验 | `cases`、`patterns`、`skills` |

### 6.4 策略层

| 策略 | 适用类别 | 默认动作 |
|------|----------|----------|
| `ProfilePolicy` | `profile` | 固定 URI + merge |
| `AggregateTopicPolicy` | `preferences` / `entities` / `patterns` | 按 `routing_key` 聚合 |
| `AppendOnlyPolicy` | `events` / `cases` | create / append |
| `SkillToolPolicy` | `skills` / `tools` | 固定 URI + 聚合统计 |

## 7. 索引与一致性

### 7.1 一致性模型

| 主题 | 规则 |
|------|------|
| 提交成功判定 | `ContextFS` 主存写成功即 commit 成功 |
| 索引同步方式 | 通过 `OutboxEvent` 异步同步 |
| 幂等主键 | `IndexRecord.id = stable_hash(uri, level)` |
| 恢复方式 | worker 重试、journal/outbox 留痕、后续 repair job |

### 7.2 Outbox 事件类型

| 事件类型 | 作用 |
|----------|------|
| `UPSERT_CONTEXT` | 节点新增或更新后重建 L0/L1/L2 索引 |
| `DELETE_CONTEXT` | 删除节点对应索引 |
| `MOVE_CONTEXT` | 更新 `uri/parent_uri` 映射 |
| `UPSERT_RELATION` | 关系边副本同步 |

### 7.3 索引记录展开

| 层级 | 来源文件 | 检索用途 |
|------|----------|----------|
| L0 | `.abstract.md` | 默认首轮召回 |
| L1 | `.overview.md` | 目录级概览补充 |
| L2 | `content.md` | 精细事实和完整内容 |

## 8. 检索链路

### 8.1 检索原则

| 原则 | 说明 |
|------|------|
| L0 first | 首轮只查 L0，控制召回成本 |
| L1 explicit | 是否读取 overview 由上层明确触发 |
| L2 on demand | 只有需要精确细节才加载 content |
| planner first | 先结构化 query，再做检索 |

### 8.2 检索流程

```text
search_memory
-> QueryPlanner.plan           → list[TypedQuery]
-> SeedRetriever.search        → SeedResult (L0/L1/L2 混合召回)
-> HierarchicalSearcher.expand → list[LeafHit] (L0/L1 递归展开至 L2)
-> AssemblyService.assemble    → list[RetrievedBlock] (仅 L2)

read_memory
-> MemoryReadService.read      → RetrievedBlock (仅 L2 md 文件内容)
```

### 8.3 组件职责

| 组件 | 作用 |
|------|------|
| `QueryPlanner` | 将自然语言 query 转成 `TypedQuery[]` |
| `SeedRetriever` | L0/L1/L2 混合向量召回，按 level 分流 |
| `HierarchicalSearcher` | 从 L0/L1 节点递归展开至 L2 叶节点 |
| `AssemblyService` | 仅保留 L2 结果，排序组装 `RetrievedBlock` |
| `MemoryReadService` | 按 URI 读取 L2 md 文件内容 |

## 9. 模块边界与包结构

| 包 | 核心职责 | 允许依赖 |
|----|----------|----------|
| `core` | 枚举、对象、错误、配置、协议 | 不依赖其他业务包 |
| `fs` | URI 解析、目录节点读写、关系文件、元数据 | `core`、本地文件 provider |
| `commit` | Session commit、策略路由、写入编排 | `core`、`fs`、`extraction`、`index`、`providers` |
| `extraction` | 候选记忆抽取 | `core`、`providers` |
| `index` | Outbox、索引记录构建、worker | `core`、`fs`、`providers` |
| `retrieval` | Planner、L0 检索、层级搜索、读取和组装 | `core`、`fs`、`providers` |
| `providers` | LLM、Embedding、Vector、Rerank、Graph 适配 | 只向上暴露能力接口 |
| `service` | 对外 API、权限、请求编排 | 可依赖全部业务包 |

### 9.1 推荐子模块

| 包 | 子模块 |
|----|--------|
| `commit` | `archive_builder`, `candidate_pipeline`, `policy_router`, `policies`, `context_writer`, `relation_writer`, `commit_coordinator` |
| `index` | `outbox_store`, `index_projector`, `index_worker`, `record_builder` |
| `retrieval` | `query_planner`, `l0_retriever`, `hierarchical_searcher`, `memory_read_service`, `assembler` |

## 10. 核心接口

### 10.1 Service 层 API

| 接口 | 函数 | 功能 |
|------|------|------|
| `ContextEngineService` | `commit_session(ctx, messages, used_contexts=None, used_tools=None, options=None) -> CommitResult` | 写入主链路入口 |
| `ContextEngineService` | `search_memory(ctx, query, target_uri=None, categories=None, top_k=10, session_archive=None) -> SearchResult` | 默认只做 L0 检索 |
| `ContextEngineService` | `read_memory(ctx, uri, level="L1", expand_relations=False, max_relations=0) -> RetrievedBlock` | 显式读取 L1/L2 |
| `ContextEngineService` | `get_node(ctx, uri) -> ContextNode` | 读取完整节点对象 |
| `ContextEngineService` | `list_children(ctx, uri, recursive=False, depth=1) -> NodeSummary[]` | 列子节点 |

### 10.2 FS 层 API

| 接口 | 函数 | 功能 |
|------|------|------|
| `ContextFS` | `resolve_path(ctx, uri) -> str` | 逻辑 URI 到物理路径映射 |
| `ContextFS` | `exists(ctx, uri) -> bool` | 判断节点是否存在 |
| `ContextFS` | `stat_node(ctx, uri) -> NodeStat` | 读取节点元信息 |
| `ContextFS` | `write_node(ctx, node) -> None` | 写入完整目录节点 |
| `ContextFS` | `read_node(ctx, uri, include_content=True) -> ContextNode` | 读取完整目录节点 |
| `ContextFS` | `write_level(ctx, uri, level, text) -> None` | 写某一层语义文件 |
| `ContextFS` | `read_level(ctx, uri, level) -> str` | 读某一层语义文件 |
| `ContextFS` | `write_metadata(ctx, uri, metadata) -> None` | 写 `.meta.json` |
| `ContextFS` | `read_metadata(ctx, uri) -> Dict` | 读 `.meta.json` |
| `ContextFS` | `move_node(ctx, from_uri, to_uri) -> None` | 移动节点目录 |
| `ContextFS` | `delete_node(ctx, uri, recursive=False) -> None` | 删除节点目录 |
| `ContextFS` | `list_children(ctx, uri, recursive=False, depth=1) -> NodeSummary[]` | 列子节点 |

### 10.3 关系层 API

| 接口 | 函数 | 功能 |
|------|------|------|
| `RelationStore` | `get_edges(ctx, uri) -> RelationEdge[]` | 读某节点的关系边 |
| `RelationStore` | `upsert_edges(ctx, uri, edges) -> None` | 幂等写入关系边 |
| `RelationStore` | `append_edges(ctx, uri, edges) -> None` | 追加关系边 |
| `RelationStore` | `remove_edges(ctx, uri, to_uris=None, relation_types=None) -> int` | 删除关系边 |

### 10.4 Commit 编排 API

| 组件 | 函数 | 功能 |
|------|------|------|
| `CommitCoordinator` | `commit(ctx, messages, used_contexts=None, used_tools=None, options=None) -> CommitResult` | 串起完整写入流水线 |
| `ArchiveBuilder` | `build_archive(ctx, messages, used_contexts=None, used_tools=None) -> SessionArchive` | 构造归档和摘要 |
| `CandidatePipeline` | `extract_candidates(ctx, archive, messages) -> CandidateMemory[]` | 并行调用所有抽取器 |
| `CandidatePipeline` | `normalize_candidates(ctx, candidates) -> CandidateMemory[]` | 规范化和去重同批候选 |
| `PolicyRouter` | `select_policy(candidate, ctx) -> MergePolicy` | 为候选选择策略 |
| `PolicyRouter` | `build_plan(candidate, ctx) -> WritePlan` | 调用策略并产出写入计划 |
| `MergePolicy` | `plan(candidate, existing_node, ctx) -> WritePlan` | 把候选映射成具体写动作 |
| `ContextWriter` | `apply_plan(ctx, plan) -> WriteResult` | 执行 `WritePlan` 写主存 |
| `ContextWriter` | `merge_node(ctx, uri, patch) -> WriteResult` | 聚合型写入 |
| `ContextWriter` | `append_node(ctx, uri, patch) -> WriteResult` | 追加型写入 |
| `RelationWriter` | `apply_edges(ctx, uri, relation_ops) -> None` | 执行关系更新 |
| `OutboxStore` | `append(event) -> OutboxEvent` | 记录单个事件 |
| `OutboxStore` | `append_batch(events) -> List[OutboxEvent]` | 批量登记事件 |

### 10.5 Extraction API

| 接口/组件 | 函数 | 功能 |
|-----------|------|------|
| `CandidateExtractor` | `extract(messages, archive, ctx) -> CandidateMemory[]` | 候选抽取标准接口 |
| `ProfileExtractor` | `extract(messages, archive, ctx) -> CandidateMemory[]` | 抽稳定用户画像 |
| `PreferenceExtractor` | `extract(messages, archive, ctx) -> CandidateMemory[]` | 抽偏好和约束 |
| `EntityEventExtractor` | `extract(messages, archive, ctx) -> CandidateMemory[]` | 抽实体与事件 |
| `ProcedureExtractor` | `extract(messages, archive, ctx) -> CandidateMemory[]` | 抽案例、模式、技能经验 |

### 10.6 Index API

| 组件/接口 | 函数 | 功能 |
|-----------|------|------|
| `OutboxStore` | `claim_batch(worker_id, limit) -> OutboxEvent[]` | worker 拉取待处理事件 |
| `OutboxStore` | `mark_done(event_id) -> None` | 标记事件完成 |
| `OutboxStore` | `mark_failed(event_id, error, retryable=True) -> None` | 标记失败并控制重试 |
| `IndexWorker` | `run_once(limit=100) -> IndexBatchResult` | 执行一轮 outbox 消费 |
| `IndexWorker` | `handle_event(event, ctx) -> None` | 分发单个 outbox 事件 |
| `IndexProjector` | `build_context_records(node) -> IndexRecord[]` | 为节点生成 L0/L1/L2 索引记录 |
| `IndexProjector` | `build_relation_records(uri, edges) -> RelationIndexRecord[]` | 为关系副本生成索引记录 |
| `IndexProjector` | `build_move_records(from_uri, to_uri) -> MoveProjection` | 生成 URI 映射更新 |
| `VectorIndex` | `upsert(records) -> None` | Upsert 索引记录 |
| `VectorIndex` | `delete_by_uri(uri) -> None` | 删除节点对应索引 |
| `VectorIndex` | `move_uri(from_uri, to_uri) -> None` | 更新 `uri/parent_uri` 映射 |
| `VectorIndex` | `search(query_vector, filters, top_k, level=0) -> SeedHit[]` | 向量召回 |
| `Embedder` | `embed_texts(texts, model=None) -> List[EmbeddingVector]` | 批量 embedding |

### 10.7 Retrieval API

| 组件/接口 | 函数 | 功能 |
|-----------|------|------|
| `QueryPlanner` | `plan(query, ctx) -> list[TypedQuery]` | 规则化规划查询 |
| `SeedRetriever` | `retrieve(typed_queries, ctx) -> SeedResult` | L0/L1/L2 混合向量召回 |
| `HierarchicalSearcher` | `expand(typed_query, seed_result, ctx) -> list[LeafHit]` | 从 seed 递归展开至 L2 |
| `AssemblyService` | `assemble(typed_query, leaf_hits, seed_result) -> list[RetrievedBlock]` | 组装 L2 结果块 |
| `MemoryReadService` | `read(uri, ctx) -> RetrievedBlock` | 按 URI 读取 L2 md 文件内容 |

### 10.8 Provider API

| 接口 | 函数 | 功能 |
|------|------|------|
| `LLM` | `complete_json(prompt, schema, ctx=None) -> Dict` | 为抽取器和 planner 提供结构化输出 |
| `Embedder` | `embed_texts(texts, model=None) -> List[EmbeddingVector]` | 文本向量化 |
| `VectorIndex` | `upsert(records) -> None` | Upsert 索引副本 |
| `VectorIndex` | `delete_by_uri(uri) -> None` | 删除索引副本 |
| `VectorIndex` | `search(query_vector, filters, top_k, level=0) -> SeedHit[]` | 检索指定层级索引 |
| `Reranker` | `rerank(query, documents, top_k=None) -> List[RerankScore]` | 对候选做精排 |
| `GraphStore` | `upsert_edges(edges) -> None` | 可选关系图副本 |
| `GraphStore` | `neighbors(uri, relation_types=None, depth=1) -> RelationEdge[]` | 可选图查询 |

## 11. 典型时序

### 11.1 Session Commit

```text
ContextEngineService.commit_session
-> CommitCoordinator.commit
-> ArchiveBuilder.build_archive
-> CandidatePipeline.extract_candidates
-> PolicyRouter.build_plan
-> ContextWriter.apply_plan
-> RelationWriter.apply_edges
-> OutboxStore.append_batch
```

### 11.2 异步索引

```text
IndexWorker.run_once
-> OutboxStore.claim_batch
-> IndexWorker.handle_event
-> ContextFS.read_node / RelationStore.get_edges
-> IndexProjector.build_context_records
-> Embedder.embed_texts
-> VectorIndex.upsert / delete_by_uri / move_uri
-> OutboxStore.mark_done
```

### 11.3 检索与显式下钻

```text
ContextEngineService.search_memory
-> QueryPlanner.plan
-> SeedRetriever.search
-> HierarchicalSearcher.expand
-> AssemblyService.assemble

ContextEngineService.read_memory
-> MemoryReadService.read
-> ContextFS.read_node
-> RelationStore.get_edges
```

## 12. 实现建议

| 建议 | 目的 |
|------|------|
| 先实现 `core/fs/commit/index/retrieval/providers/service` 的最小协议 | 先把边界稳定下来 |
| 先落本地文件主存，再接入向量库 | 保证主链路先可用 |
| `WritePlan` 和 `OutboxEvent` 作为调试/审计对象持久化 | 便于排查策略和一致性问题 |
| `search_memory` 和 `read_memory` 分离 | 避免检索链路自动展开带来的不可控成本 |
| 所有接口先用同步语义建模，再用 async 实现 | 接口稳定性优先于执行细节 |
