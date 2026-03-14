# 检索链路模块设计（API 级）

## 1. 文档目标与范围

本设计文档细化 ContextEngine 的“检索链路”实现，严格落实三段式策略：
- L0 首轮召回（只查 abstract 对应索引）
- L1 主动读取（按 URI 读取 overview）
- L2 按需精读（按 URI 读取 content）

本模块设计对齐 `ce_architecture.md` 第 8 章与第 10.7 节，新增可直接落地开发的 API、协议、约束、错误码和验收标准。

非目标：
- 不在首轮检索阶段自动读取 L1/L2 正文
- 不在 `search_memory` 阶段隐式触发 `read_memory`
- 不在本阶段设计写入链路和索引写入逻辑

## 2. 总体边界

### 2.1 分层边界

- Service 层对外 API 保持 `search_memory` 与 `read_memory` 分离
- Tool 层默认入口是 `search_memory`；当需要下钻时显式调用 `read_memory`
- 检索模块内部包含：
  - `QueryPlanner`
  - `L0Retriever`
  - `HierarchicalSearcher`
  - `RetrievalAssembler`
  - `MemoryReadService`（内部服务，不对外）
- `MemoryReadService` 由 `ContextEngineService.read_memory` 调用，并可被检索编排按需复用

### 2.2 关键业务约束

1. memory 类查询默认强制 `level=0` 向量召回。
2. 首轮召回返回的是“候选入口”，不是完整答案。
3. L1/L2 读取只能基于明确 URI 显式触发。
4. L2 不参与默认首轮召回。
5. 对上层返回必须可解释：包含命中原因、层级、是否建议下钻。
6. 检索与读取都必须绑定 `RequestContext(account_id, role, user_space, agent_space)` 做权限校验。
7. 调用方传入的过滤条件只能“收敛范围”，不能突破租户边界。

### 2.3 租户与访问控制模型

- 强隔离主键：`account_id`
- 业务空间：`owner_space`（如 `user_space` / `agent_space` / shared resource）
- URI 作用域：`target_uri` 仅允许在调用方可见目录内生效

访问规则：
- `search_memory`：
  - 先构造系统过滤（`account_id` + allowed `owner_space`）
  - 再与调用方过滤做 `AND` 合并
- `read_memory`：
  - 对 `uri` 做二次 `ACL` 校验
  - 无权限时返回 `403`，不泄露目标是否存在

## 3. 数据契约

## 3.1 TypedQuery

```json
{
  "text": "string, required",
  "context_type": "memory|skill|resource, required",
  "categories": ["string"],
  "target_uri": "string|null",
  "top_k": "int, required, 1..50",
  "filters": {
    "account_id": "string|null",
    "owner_space": "string|null",
    "time_range": "object|null"
  },
  "hints": {
    "intent": "string|null",
    "must_have_terms": ["string"],
    "exclude_terms": ["string"]
  }
}
```

规则：
- `context_type=memory` 时，检索器必须附加过滤 `level=0`
- `categories` 为空时由 planner 补齐默认类别
- `top_k` 默认 10，超过 50 直接拒绝
- `filters.account_id` 由服务端从 `RequestContext` 注入，调用方不可覆盖

## 3.2 SeedHit（L0 命中）

```json
{
  "uri": "string",
  "score": "float",
  "category": "string",
  "owner_space": "string",
  "abstract": "string",
  "has_overview": "bool",
  "has_content": "bool",
  "match_reason": "string"
}
```

## 3.3 RetrievedBlock（统一返回块）

```json
{
  "uri": "string",
  "level_hit": "L0|L1|L2",
  "score": "float",
  "category": "string",
  "owner_space": "string",
  "abstract": "string|null",
  "overview": "string|null",
  "content_excerpt": "string|null",
  "relations": [
    {"to_uri": "string", "relation_type": "string", "weight": "float"}
  ],
  "has_overview": "bool",
  "has_content": "bool",
  "read_recommendation": "none|read_l1|read_l2",
  "match_reason": "string"
}
```

## 4. API 设计

## 4.1 对外 API 协议（search_memory + read_memory）

### 4.1.1 `search_memory`

请求：

```json
{
  "query": "string, required",
  "top_k": "int, optional, default=10",
  "score_threshold": "float, optional, default=null",
  "score_gte": "bool, optional, default=false",
  "categories": ["string"],
  "target_uri": "string|null",
  "owner_space": "string|null, optional narrowing filter",
  "session_archive": "object|null",
  "planner_hints": "object|null",
  "include_debug": "bool, default=false",
  "trace_level": "basic|detailed, default=basic"
}
```

响应：

```json
{
  "request_id": "string",
  "typed_queries": ["TypedQuery"],
  "seed_hits": ["SeedHit"],
  "hits": ["RetrievedBlock"],
  "next_actions": {
    "recommended_read_l1_uris": ["string"],
    "recommended_read_l2_uris": ["string"]
  },
  "trace": {
    "planner_ms": "int",
    "seed_ms": "int",
    "recheck_ms": "int",
    "hierarchical_ms": "int",
    "assemble_ms": "int",
    "warnings": ["string"],
    "event_count": "int"
  },
  "thinking_trace": "object|null"
}
```

协议约束：
- `search_memory` 不自动读取 L2 content。
- `next_actions` 只提供建议 URI，不隐式执行读取。
- `owner_space/target_uri/filter` 与服务端租户过滤做 `AND` 合并，不允许扩大权限边界。
- `thinking_trace` 仅在 `include_debug=true` 且 `trace_level=detailed` 时返回。

### 4.1.2 `read_memory`

请求：

```json
{
  "uri": "string, required",
  "level": "L1|L2, optional, default=L1",
  "expand_relations": "bool, optional, default=false",
  "max_relations": "int, optional, default=0",
  "include_debug": "bool, optional, default=false"
}
```

响应：

```json
{
  "request_id": "string",
  "block": "RetrievedBlock",
  "trace": {
    "read_ms": "int",
    "relations_ms": "int"
  }
}
```

## 4.2 检索子模块内部 API

### QueryPlanner

```python
plan(query: str,
     session_archive: dict | None = None,
     hints: dict | None = None,
     ctx: RequestContext | None = None) -> list[TypedQuery]
```

规划规则（V1 规则化）：
- query 含“偏好/背景/身份/记忆回顾” => `context_type=memory`
- query 含“怎么做/步骤/命令/流程” => `context_type=skill`
- query 含“文档/资料/规范/链接” => `context_type=resource`
- 若命中多个意图，拆成多个 `TypedQuery`，按优先级 `memory > skill > resource`
- 若分类置信不足或规则冲突，降级为并发三查询：`memory + resource + skill`

### L0Retriever

```python
search(typed_query: TypedQuery,
       ctx: RequestContext) -> list[SeedHit]
```

行为约束：
- 仅查询 `level=0` 索引记录
- 不访问 L1/L2 文本文件
- 返回最小字段集：`uri/score/category/owner_space/abstract/has_overview/has_content`

### HierarchicalSearcher

```python
expand(typed_query: TypedQuery,
       seeds: list[SeedHit],
       ctx: RequestContext) -> list[NodeHit]

score_children(parent_uri: str,
               typed_query: TypedQuery,
               ctx: RequestContext) -> list[NodeHit]
```

行为约束：
- 从“根节点 + seed 节点”并行展开
- 仅用索引元数据与抽象信息评分，不读取 L2 content
- 最大递归深度默认 2，可配置

### CandidateRechecker

```python
recheck(typed_query: TypedQuery,
        hits: list[NodeHit],
        ctx: RequestContext) -> list[NodeHit]
```

行为约束：
- 用轻量规则/精确过滤对 ANN 候选做二次校验（例如 must-have terms、URI scope、metadata 精确匹配）
- `recheck` 只做过滤与原因标注，不改写原始召回分
- 失败时允许降级跳过，但必须写入 `trace.warnings`

### MemoryReadService

```python
read(uri: str,
     level: Literal["L1", "L2"] = "L1",
     expand_relations: bool = False,
     max_relations: int = 0,
     ctx: RequestContext | None = None) -> RetrievedBlock

read_batch(uris: list[str],
           level: Literal["L1", "L2"] = "L1",
           ctx: RequestContext | None = None) -> list[RetrievedBlock]
```

分层返回契约：
- `L1`：返回 `overview`（决策层信息：主题、边界、覆盖范围）
- `L2`：返回 `content`（可被裁剪成 excerpt），可扩一跳 relations

### RetrievalAssembler

```python
assemble(typed_query: TypedQuery,
         hits: list[NodeHit],
         ctx: RequestContext) -> list[RetrievedBlock]
```

组装规则：
- 默认产物以 L0 abstract 为主
- 如果层级评分命中中间节点，可补 `overview=null + read_recommendation=read_l1`
- 仅当编排层显式调用 `MemoryReadService` 后，才填充 `overview/content_excerpt`

## 5. 检索阶段与时序

```text
search_memory
-> QueryPlanner.plan
-> L0Retriever.search            # level=0 only
-> HierarchicalSearcher.expand   # no L2 content read
-> CandidateRechecker.recheck    # optional precise filtering
-> RetrievalAssembler.assemble
=> 返回 L0 候选与建议下钻 URI

read_memory
-> MemoryReadService.read
-> ContextFS.read_level
-> RelationStore.get_edges
```

阶段输出对照：
- Query Planning: `TypedQuery[]`
- Seed Retrieval: `SeedHit[]`
- Hierarchical Search: `NodeHit[]`
- Candidate Recheck: `NodeHit[]`（可选）
- Assembly: `RetrievedBlock[]`

## 6. 评分与排序策略（V1）

总分（归一化到 0~1）：

`final_score = 0.65 * vector_score + 0.20 * hierarchy_score + 0.10 * category_boost + 0.05 * target_uri_boost`

说明：
- `vector_score`：L0 向量相似度
- `hierarchy_score`：根/seed 展开路径一致性
- `category_boost`：类别命中加权
- `target_uri_boost`：目标 URI 子树命中加权

去重：
- 以 `uri` 去重，保留最高分
- 同分时优先 `has_overview=true`

阈值过滤：
- 若 `score_threshold` 为空，则使用服务端默认阈值
- 支持 `>`（默认）与 `>=`（`score_gte=true`）两种判定

Recheck 约束：
- `CandidateRechecker` 仅能淘汰候选，不可抬高分数
- 被淘汰候选需记录 `exclude_reason`

## 7. 错误码与降级

| 错误码 | 场景 | 处理策略 |
|---|---|---|
| `RET-4001` | query 为空/非法 | 返回 400，提示修正参数 |
| `RET-4002` | `top_k` 超限 | 返回 400，拒绝执行 |
| `RET-4003` | `score_threshold` 非法 | 返回 400，提示阈值范围 |
| `RET-4031` | 无权访问 target_uri | 返回 403，不暴露存在性 |
| `RET-4032` | 无权读取 uri | 返回 403，不暴露存在性 |
| `RET-4041` | target_uri 不存在 | 忽略 target 限制并打 warning |
| `RET-4221` | planner 无法分类 | 降级为 `memory+resource+skill` 三查询 |
| `RET-5001` | 向量检索失败 | 返回空命中 + 可重试标记 |
| `RET-5002` | recheck 失败 | 跳过 recheck，保留原候选并打 warning |
| `RET-5003` | 组装失败 | 回退返回原始 `SeedHit[]` |

降级原则：
- 优先“可返回候选”而非直接失败
- 任何降级都必须写入 `trace.warnings`

## 8. 性能与可观测性

性能预算（P95）：
- planner <= 30ms
- seed retrieval <= 120ms
- recheck <= 50ms
- hierarchical <= 100ms
- assemble <= 40ms
- `search_memory` 总计 <= 300ms

埋点指标：
- `retrieval.search_memory.qps`
- `retrieval.search_memory.p95_ms`
- `retrieval.search_memory.threshold_filtered.count`
- `retrieval.seed.hit_count`
- `retrieval.recheck.input_count`
- `retrieval.recheck.pass_count`
- `retrieval.recheck.drop_count`
- `retrieval.read.l1.count`
- `retrieval.read.l2.count`
- `retrieval.fallback.count`

日志字段（结构化）：
- `request_id`, `query_hash`, `typed_query_count`, `top_k`, `seed_count`, `final_count`, `warnings`
- `account_id`, `role`, `effective_owner_spaces`, `recheck_drop_reasons`

## 9. 安全与权限

- 检索必须绑定 `account_id`，禁止跨租户召回
- `owner_space` 作为强过滤条件参与检索（避免跨空间串读）
- `target_uri` 进入检索前必须先做可见性校验
- `MemoryReadService.read` 必须二次校验 URI 访问权限
- `L2` 返回前执行敏感字段裁剪（如 token、secret、凭据）

## 10. 测试与验收标准

单元测试：
1. planner 分类：memory/skill/resource 的规则命中
2. planner 失败降级：自动拆分为三查询
3. L0Retriever 强制 `level=0` 过滤
4. score_threshold 过滤逻辑（`>` 与 `>=`）
5. CandidateRechecker 仅淘汰不提分
6. MemoryReadService 分层返回契约（L1/L2）
7. ACL：无权限 URI 返回 403 且无存在性泄露
8. assembler 不读取 L2 的约束

集成测试：
1. `search_memory` 返回仅 L0 命中，且包含 `next_actions`
2. 对同一 URI 执行 L1 -> L2 显式读取链路可用
3. 向量服务不可用时触发降级返回
4. 无会话上下文时 planner 触发三类型并发检索
5. `include_debug=true` 时返回完整 `thinking_trace`

验收标准：
- 首轮检索全链路无 L2 文件读取 I/O
- 返回字段满足 `SeedHit` 最小字段要求
- Service 层公开 `search_memory` 与 `read_memory`，且语义分离
- 在有 `has_overview=true` 的命中中，`next_actions.recommended_read_l1_uris` 正确生成
- 跨租户与越权读取用例全部失败（符合预期 403）

## 11. 开发落地清单

1. 在 `retrieval/query_planner.py` 实现规则化 planner（含失败降级三查询）。
2. 在 `retrieval/l0_retriever.py` 强制 `level=0` 查询与字段裁剪。
3. 在 `retrieval/hierarchical_searcher.py` 实现 root+seed 并行展开评分。
4. 在 `retrieval/candidate_rechecker.py` 实现候选二次校验与淘汰原因标注。
5. 在 `retrieval/assembler.py` 实现 `RetrievedBlock` 组装与 `next_actions` 生成。
6. 在 `retrieval/memory_read_service.py` 实现 L1/L2 分层读取 API（供 `read_memory` 调用）。
7. 在 `service/context_engine_service.py` 对外同时提供 `search_memory` 与 `read_memory`，并注入租户过滤。
8. 在 `service` 层补齐 `score_threshold/score_gte/include_debug/trace_level` 参数透传。
9. 增加细粒度 trace 事件、错误码、回归测试。

---

与主架构的一致性声明：
- 本文严格遵循 `ce_architecture.md` 第 8 章“L0 first / L1 explicit / L2 on demand”原则。
- 本文将“search_memory 与 read_memory 分离”的建议具体化为“Service 层同时提供两者，`search_memory` 不隐式展开，`read_memory` 负责显式下钻读取”。
