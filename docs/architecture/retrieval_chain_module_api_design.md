# 检索链路模块设计（API 级）

## 1. 文档目标与范围

本设计文档细化 ContextEngine 的检索链路实现，采用四阶段管线策略：
- Stage 1 — QueryPlanner：查询规划（规则化分类）
- Stage 2 — SeedRetriever：L0/L1/L2 混合向量召回
- Stage 3 — HierarchicalSearcher：L0/L1 自动递归展开至 L2
- Stage 4 — AssemblyService：仅保留 L2 结果，排序输出

对外暴露两个工具接口：
- `search_memory`：语义检索，返回结构化 `SearchMemoryResult`（仅含 L2 结果）
- `read_memory`：按 URI 读取 L2 md 文件内容

本模块设计对齐 `ce_architecture.md` 第 8 章与第 10.7 节。

非目标：
- 不在本阶段设计写入链路和索引写入逻辑
- 不在 `search_memory` 中渲染 prompt（由上层自行组装）

## 2. 总体边界

### 2.1 分层边界

- Service 层对外 API 保持 `search_memory` 与 `read_memory` 分离
- Tool 层默认入口是 `search_memory`；当需要完整 md 内容时显式调用 `read_memory`
- 检索模块内部包含：
  - `QueryPlanner` — 规则化查询分类
  - `SeedRetriever` — L0/L1/L2 混合向量召回
  - `HierarchicalSearcher` — L0/L1 递归展开至 L2 叶节点
  - `AssemblyService` — L2 结果组装与排序
  - `MemoryReadService` — 按 URI 读取 L2 md 文件内容

### 2.2 关键业务约束

1. `search_memory` 执行 L0/L1/L2 混合向量召回，但最终只返回 L2 级结果。
2. L0/L1 命中会自动递归展开到 L2 叶节点（Stage 3 强制执行）。
3. `read_memory` 仅读取 L2 md 文件的实际内容，不返回 abstract/overview。
4. 检索与读取都必须绑定 `RequestContext(account_id, user_id, agent_id, session_id, trace_id)` 做权限校验。
5. 调用方传入的过滤条件只能"收敛范围"，不能突破租户边界。

### 2.3 租户与访问控制模型

- 强隔离主键：`account_id`
- 业务空间：`owner_space`（如 `user_space` / `agent_space`）
- URI 作用域：`target_uri` 仅允许在调用方可见目录内生效

访问规则：
- `search_memory`：
  - 先构造系统过滤（`account_id` + allowed `owner_space`）
  - 再与调用方过滤做 `AND` 合并
- `read_memory`：
  - 对 `uri` 做二次 ACL 校验
  - 无权限时返回 403，不泄露目标是否存在

## 3. 数据契约

### 3.1 TypedQuery

```json
{
  "text": "string, required — 原始查询文本，用于向量化",
  "context_type": "MEMORY|SKILL|RESOURCE, required",
  "categories": ["string — planner 推断的候选 category"],
  "target_uri": "string|null — 若已知目标节点 URI，直接读取",
  "top_k": "int, required, 1..50, default=10",
  "account_id": "string — 服务端强制注入",
  "owner_space": "string|null — 进一步限定 user 或 agent 范围"
}
```

规则：
- `categories` 为空时由 planner 补齐默认类别
- `top_k` 默认 10，超过 50 直接拒绝
- `account_id` 由服务端从 `RequestContext` 注入，调用方不可覆盖

### 3.2 SeedHit（向量检索命中）

```json
{
  "uri": "string",
  "score": "float — 语义相似度",
  "level": "int — 0/1/2",
  "parent_uri": "string|null — L2 记录的父目录 URI",
  "context_type": "string",
  "category": "string",
  "owner_space": "string",
  "abstract": "string — text[:200] 摘要",
  "has_overview": "bool",
  "has_content": "bool",
  "active_count": "int — 访问频次",
  "updated_at": "string|null",
  "metadata": "object"
}
```

### 3.3 SeedResult（种子召回输出）

```json
{
  "starting_points": ["SeedHit — L0/L1 命中，需要递归展开"],
  "initial_candidates": ["SeedHit — L2 直接命中"],
  "query_vector": ["float — 查询向量，用于后续展开"],
  "root_uris": ["string — 账户根目录 URI"]
}
```

### 3.4 LeafHit（展开后的 L2 叶节点）

```json
{
  "uri": "string",
  "score": "float — final_score (语义+层级传播)",
  "level": 2,
  "depth": "int — 展开深度",
  "parent_uri": "string|null",
  "category": "string",
  "owner_space": "string",
  "abstract": "string",
  "has_overview": "bool",
  "has_content": "bool",
  "hierarchy_path": ["string — 展开路径 URI 列表"],
  "active_count": "int",
  "updated_at": "string|null"
}
```

### 3.5 RetrievedBlock（最终返回块）

```json
{
  "uri": "string",
  "level_hit": "L2",
  "score": "float",
  "category": "string",
  "owner_space": "string",
  "abstract": "string|null — search_memory 返回",
  "content_excerpt": "string|null — read_memory 填充",
  "match_reason": "string"
}
```

### 3.6 SearchMemoryResult（search_memory 返回）

```json
{
  "request_id": "string",
  "query": "string",
  "typed_queries": ["TypedQuery"],
  "hits": ["RetrievedBlock — 仅 L2"],
  "trace": "RetrievalTrace|null"
}
```

## 4. API 设计

### 4.1 对外 API 协议

#### 4.1.1 `search_memory`

请求参数：

```python
search_memory(
    query: str,                           # 必填，检索文本
    ctx: RequestContext,                   # 必填，租户上下文
    *,
    top_k: int = 10,                      # 可选，结果数量
    categories: list[str] | None = None,  # 可选，类别过滤
    target_uri: str | None = None,        # 可选，指定目录范围
    session_archive: dict | None = None,  # 可选，会话归档
    score_threshold: float | None = None, # 可选，分数阈值
    include_debug: bool = False,          # 可选，返回 trace
    mode: str = "quick",                  # 可选，quick/thinking
) -> SearchMemoryResult
```

协议约束：
- 返回结果仅包含 L2 级别的 `RetrievedBlock`
- `abstract` 字段在 `search_memory` 结果中已填充
- 不返回 prompt，仅返回结构化数据
- `trace` 仅在 `include_debug=True` 时返回

#### 4.1.2 `read_memory`

请求参数：

```python
read_memory(
    uri: str,              # 必填，目标 URI（通常来自 search_memory 结果）
    ctx: RequestContext,   # 必填，租户上下文
) -> RetrievedBlock
```

协议约束：
- 仅读取 L2 md 文件的实际内容，填充 `content_excerpt` 字段
- 不返回 abstract 和 overview（search_memory 已提供 abstract）
- URI 的 account 前缀必须与 `ctx.account_id` 匹配，否则拒绝

### 4.2 检索子模块内部 API

#### QueryPlanner

```python
plan(query: str,
     ctx: RequestContext,
     *,
     session_archive: dict | None = None,
     hints: dict | None = None,
     categories: list[str] | None = None,
     top_k: int | None = None) -> list[TypedQuery]
```

规划规则（V1 规则化）：
- query 含"偏好/背景/身份/记忆回顾" => `context_type=MEMORY`
- query 含"怎么做/步骤/命令/流程" => `context_type=SKILL`
- query 含"文档/资料/规范/链接" => `context_type=RESOURCE`
- 若命中多个意图，拆成多个 `TypedQuery`，按优先级 `MEMORY > SKILL > RESOURCE`
- 若分类置信不足或规则冲突，降级为并发三查询：`MEMORY + RESOURCE + SKILL`

#### SeedRetriever

```python
search(typed_query: TypedQuery,
       ctx: RequestContext,
       mode: str = "quick") -> SeedResult
```

行为约束：
- 执行 L0/L1/L2 混合向量检索（不限制 level）
- 按 level 分流：L0/L1 → `starting_points`，L2 → `initial_candidates`
- 应用热度评分 blended：`(1 - hotness_alpha) * semantic + hotness_alpha * hotness`
- 返回查询向量 `query_vector` 供后续展开使用

#### HierarchicalSearcher

```python
expand(typed_query: TypedQuery,
       seed_result: SeedResult,
       ctx: RequestContext,
       *,
       limit: int = 10,
       mode: str = "quick",
       score_threshold: float | None = None) -> list[LeafHit]
```

行为约束：
- 从 `starting_points` 中的 L0/L1 节点出发，递归调用 `search_children` 展开至 L2
- 评分传播公式：`final_score = alpha * child_score + (1 - alpha) * parent_score`
- 收敛检测：连续 `MAX_CONVERGENCE_ROUNDS` (3) 轮 top-k 无变化则停止
- 与 `initial_candidates` 中已有的 L2 结果合并去重

#### AssemblyService

```python
assemble(typed_query: TypedQuery,
         leaf_hits: list[LeafHit],
         seed_result: SeedResult | None) -> list[RetrievedBlock]
```

组装规则：
- 仅保留 L2 结果
- 如果 `leaf_hits` 为空，兜底从 `seed_result.initial_candidates` 取 L2
- 按 `score` 降序排列，取 `top_k` 个
- 转换为 `RetrievedBlock`，填充 `abstract`、`category`、`match_reason`

#### MemoryReadService

```python
read(uri: str,
     ctx: RequestContext | None = None) -> RetrievedBlock
```

行为约束：
- 通过 ContextFS 读取 URI 对应的 ContextNode
- 仅填充 `content_excerpt`（从 `node.content` 获取）
- 不填充 `abstract` 和 `overview`（search_memory 已提供）

## 5. 检索阶段与时序

```text
search_memory(query, ctx)
  |
  +-> Stage 1: QueryPlanner.plan(query, ctx)
  |     输出: list[TypedQuery]
  |
  +-> Stage 2: SeedRetriever.search(tq, ctx)  [per TypedQuery]
  |     输出: SeedResult (starting_points + initial_candidates + query_vector)
  |
  +-> Stage 3: HierarchicalSearcher.expand(tq, seed_result, ctx)
  |     输出: list[LeafHit]  (L2 only)
  |     触发条件: starting_points 中有 L0/L1 命中
  |     兜底: 直接取 seed_result 中的 L2 候选
  |
  +-> Stage 4: AssemblyService.assemble(tq, leaf_hits, merged_seed)
  |     输出: list[RetrievedBlock]  (L2 only, sorted by score)
  |
  +=> SearchMemoryResult

read_memory(uri, ctx)
  |
  +-> MemoryReadService.read(uri, ctx)
  |     -> ContextFS.read_node(uri, ctx)
  |     -> 填充 content_excerpt
  |
  +=> RetrievedBlock (仅 content_excerpt)
```

阶段输出对照：
- Query Planning: `list[TypedQuery]`
- Seed Retrieval: `SeedResult` (L0/L1 → starting_points, L2 → initial_candidates)
- Hierarchical Expansion: `list[LeafHit]` (L2 only)
- Assembly: `list[RetrievedBlock]` (L2 only)

## 6. 评分与排序策略

### 6.1 语义评分（Seed 阶段）

向量余弦相似度 `semantic_score`，由 VectorIndex 直接返回。

### 6.2 热度评分（Seed 阶段）

```text
age_days = (now - updated_at).days
decay = 0.5 ^ (age_days / half_life_days)
h_score = min(log2(active_count + 1) * decay, 1.0)

blended = (1 - hotness_alpha) * semantic_score + hotness_alpha * h_score
```

### 6.3 层级传播评分（Hierarchical 阶段）

```text
final_score = alpha * child_score + (1 - alpha) * parent_score
```

其中 `alpha = RetrievalConfig.score_propagation_alpha` (默认 0.5)。

### 6.4 去重与排序

- 以 `uri` 去重，保留最高分
- 按 `score` 降序排列，取 `top_k` 个

## 7. 错误码与降级

| 错误码 | 场景 | 处理策略 |
|---|---|---|
| `RET-4001` | query 为空/非法 | 返回 400，提示修正参数 |
| `RET-4002` | `top_k` 超限 | 返回 400，拒绝执行 |
| `RET-4031` | 无权访问 target_uri | 返回 403，不暴露存在性 |
| `RET-4032` | 无权读取 uri | 返回 403，不暴露存在性 |
| `RET-4221` | planner 无法分类 | 降级为 `MEMORY+RESOURCE+SKILL` 三查询 |
| `RET-5001` | 向量检索失败 | 返回空 SeedResult + warning |
| `RET-5002` | 层级展开失败 | 兜底取 SeedResult 中的 L2 候选 |
| `RET-5003` | 组装失败 | 返回空结果 + warning |

降级原则：
- 优先"可返回候选"而非直接失败
- 任何降级都必须写入 `trace.warnings`

## 8. 性能与可观测性

性能预算（P95）：
- planner <= 30ms
- seed retrieval <= 120ms
- hierarchical <= 100ms
- assemble <= 40ms
- `search_memory` 总计 <= 300ms

Trace 事件字段：
- `stage_name`, `input_count`, `output_count`, `elapsed_ms`, `warnings`

日志字段（结构化）：
- `request_id`, `query_hash`, `typed_query_count`, `top_k`, `seed_count`, `final_count`, `warnings`
- `account_id`, `effective_owner_spaces`

## 9. 安全与权限

- 检索必须绑定 `account_id`，禁止跨租户召回
- `owner_space` 作为强过滤条件参与检索（避免跨空间串读）
- `target_uri` 进入检索前必须先做可见性校验
- `MemoryReadService.read` 必须校验 URI 访问权限
- L2 返回前执行敏感字段裁剪（如 token、secret、凭据）

## 10. 测试与验收标准

单元测试：
1. planner 分类：memory/skill/resource 的规则命中
2. planner 失败降级：自动拆分为三查询
3. SeedRetriever 返回 L0/L1/L2 混合结果并正确分流
4. HierarchicalSearcher 递归展开 L0/L1 到 L2
5. AssemblyService 仅保留 L2 结果
6. MemoryReadService 只返回 L2 md 文件内容
7. ACL：无权限 URI 返回 403 且无存在性泄露

集成测试：
1. `search_memory` 返回仅 L2 结果
2. `read_memory` 返回 L2 md 文件内容
3. 向量服务不可用时触发降级返回
4. 无会话上下文时 planner 触发三类型并发检索
5. `include_debug=true` 时返回完整 trace

验收标准：
- `search_memory` 返回的 `hits` 全部为 L2 级 `RetrievedBlock`
- 返回字段满足 `SeedHit` / `RetrievedBlock` 最小字段要求
- Service 层公开 `search_memory` 与 `read_memory`，且语义分离
- 跨租户与越权读取用例全部失败（符合预期 403）

## 11. 开发落地清单

1. 在 `retrieval/query_planner.py` 实现规则化 planner（含失败降级三查询）。
2. 在 `retrieval/seed_retriever.py` 实现 L0/L1/L2 混合向量召回与热度评分。
3. 在 `retrieval/hierarchical_searcher.py` 实现递归展开与层级评分传播。
4. 在 `retrieval/assembly_service.py` 实现 L2-only 结果组装。
5. 在 `retrieval/memory_read_service.py` 实现 L2 md 文件读取。
6. 在 `service/api.py` 对外提供 `search_memory` 与 `read_memory`，注入租户过滤。
7. 在 `providers/vector_index/` 实现 openGauss 向量检索后端。
8. 增加细粒度 trace 事件、错误码、回归测试。

---

与主架构的一致性声明：
- 本文严格遵循 `ce_architecture.md` 四阶段管线设计。
- `search_memory` 返回 L2-only 结构化数据（`SearchMemoryResult`）；`read_memory` 仅读取 L2 md 文件内容。
