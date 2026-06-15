# Memory Graph 架构设计

本文档描述 oG-Memory 中异构图记忆系统（heterogeneous graph memory）的当前实现，用于存储和检索多模态数据（prompt、image、video、document、webpage、audio）及其抽取结果。

---

## 1. 设计目标

- **统一表示**：将异构多模态数据、抽取结果、记忆统一为图中的节点，通过边表达关系。
- **可溯源**：每条记忆由若干抽取节点支撑，可沿边回溯到原始数据节点。
- **可信度评估**：基于 PageRank 评估节点的可信度，记忆的 confidence 由支撑节点的 PageRank 派生。
- **多租户隔离**：所有读写都按 `account_id` 隔离，调用方通过 `RequestContext` 传递。

---

## 2. 节点与边

### 2.1 节点类型（`GraphNodeType`）

| 类型 | 数据类 | 字段 | 含义 |
|------|--------|------|------|
| `raw` | `RawNode` | `raw_type`, `content`, `metadata`, `uri` | 原始多模态数据。`raw_type` 取值 `prompt`/`image`/`video`/`document`/`webpage`/`audio`。`metadata.embedding` 存放向量。 |
| `extracted` | `ExtractedNode` | `source_raw_id`, `extractor_type`, `extraction_schema`, `extracted_content`, `extraction_status`, `confidence` | 抽取算子对 `RawNode` 的结构化输出。`extraction_status` 标识 pending/in_progress/completed/failed。 |
| `memory` | `MemoryNode` | `summary`, `content`, `supporting_node_ids`, `topic`, `tags`, `parent_memory_id` | 一段被多抽取节点支撑的完整记忆。`supporting_node_ids` 是支撑该记忆的 `ExtractedNode` ID 列表。`parent_memory_id` 用于记忆演进。 |

所有节点共享基类 `GraphNode`：`node_id`、`status`、`pagerank`、`created_at`、`updated_at`、`account_id`。

### 2.2 边类型（`GraphEdgeType`）

| 类型 | 方向 | 含义 |
|------|------|------|
| `extracted_from` | `ExtractedNode → RawNode` | 抽取节点来自原始节点。`GraphService.create_extracted_node` 自动建立。 |
| `supports` | `ExtractedNode → MemoryNode` | 抽取节点支撑记忆节点。`GraphService.create_memory_node` 对每个 `supporting_node_id` 自动建立。 |
| `evolves` | `MemoryNode → MemoryNode`（旧 → 新） | 记忆演进链。`parent_memory_id` 非空时自动建立。 |
| `references` | 任意 → 任意 | 通用引用关系，按需手动建立。 |

边属性：`edge_id`、`source_id`、`target_id`、`edge_type`、`weight`、`properties`、`account_id`。

---

## 3. 架构分层

```
┌────────────────────────────────────────────────────────────────┐
│  service/graph_service.py  ←  业务编排层（GraphService）       │
│      · 节点创建时自动维护边                                    │
│      · 抽取状态机（pending → in_progress → completed/failed）  │
│      · 支撑链查询、BFS 遍历、PageRank                          │
├────────────────────────────────────────────────────────────────┤
│  providers/graph_store/                                        │
│      · SQLGraphStore       PostgreSQL 后端（生产）             │
│      · MemoryGraphStore    内存实现（单测/本地）               │
│      两者均实现 core/interfaces.py 中的 GraphStore Protocol    │
├────────────────────────────────────────────────────────────────┤
│  fs/sql_adapter/graph_schema.sql                               │
│      · graph_nodes / graph_edges 表 + 索引                    │
│      · calculate_pagerank_incremental() 等 PL/pgSQL 函数       │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 数据模型层（`core/`）

- `core/graph_enums.py`：`GraphNodeType`、`RawDataType`、`GraphEdgeType`、`NodeStatus`、`ExtractionStatus`。
- `core/graph_models.py`：所有 dataclass 定义（`GraphNode`、`RawNode`、`ExtractedNode`、`MemoryNode`、`GraphEdge`、`GraphStats`、`SupportingChain`、`SubgraphResult`），以及 `create_*` 工厂函数。
- `core/interfaces.py`：`GraphStore` Protocol，定义存储层必须实现的契约（节点/边 CRUD、邻居、遍历、支撑链、PageRank、检索、统计）。

### 3.2 存储层（`providers/graph_store/`）

- `MemoryGraphStore`：基于 dict 的内存实现，供单测与本地快速验证。
- `SQLGraphStore`：PostgreSQL 实现，构造函数自动执行 `graph_schema.sql`；在新建边时调用 `_update_pagerank_incremental` 增量更新相关节点的 PageRank（更新受影响节点及其一跳邻居）。
- 两者都强制 `_check_access(account_id)`，禁止跨租户访问。

### 3.3 服务层（`service/graph_service.py`）

`GraphService` 在存储层之上做业务编排：

- **建链自动化**：创建 `ExtractedNode` 时自动建 `extracted_from` 边；创建 `MemoryNode` 时对每个 `supporting_node_ids` 自动建 `supports` 边，并在 `parent_memory_id` 非空时建 `evolves` 边。
- **状态机**：`mark_extraction_completed` / `mark_extraction_failed` 推进 `ExtractedNode` 的状态并刷新 `confidence`。
- **图遍历**：`traverse`（BFS，深度可配）、`get_neighbors`（支持按 `edge_types` 与 `direction` 过滤）、`get_supporting_chain`（返回支撑整条记忆的原始+抽取节点及其边）。
- **PageRank**：`compute_pagerank`（damping 默认 0.85）与 `get_node_pagerank`，也供上层做置信度筛选。
- **检索**：`search_memories` 按文本 + `min_confidence` + `topic` + `tags`（OR 逻辑）筛选，结果按置信度排序；`search_raw_nodes` 按内容检索原始节点。
- **统计**：`get_graph_stats` 返回 `GraphStats`（节点/边计数、平均 PageRank、Top 记忆）。

### 3.4 抽取算子层（`extraction/operators/`）

- `base.py`：`MultimodalExtractor` Protocol + `ExtractionResult` + `OperatorInfo` dataclass。
- `prompt_extractor.py`：基于规则 + 可选 LLM 的 Prompt 抽取实现，识别 intent / subject / style / format 等维度。
- 其它模态（image/video/document/webpage）按同样的 Protocol 扩展，每个算子声明 `supported_types`、`extractor_type`、`schema_name`、`info`，对 `RawNode.extract()` 返回 `ExtractionResult`，再交由 `GraphService.create_extracted_node` 落地为节点与边。

算子之上还有完整的 **注册中心 + DAG 编排 + CV 后端抽象 + HTTP API + OperatorService 高层门面**，是异构图之上的独立子系统。详见 [`operator_service.md`](operator_service.md)。

---

## 4. 关键机制

### 4.1 支撑链（SupportingChain）

`MemoryNode` → `ExtractedNode`（via `supports`）→ `RawNode`（via `extracted_from`）形成天然的证据链。`GraphService.get_supporting_chain(memory_id, ctx)` 返回 `SupportingChain{memory, raw_nodes, extracted_nodes, edges}`，是可信度评估与可视化的基础。

### 4.2 置信度

`MemoryNode.confidence` 是派生属性：

```
confidence = min(1.0, len(supporting_node_ids) * 0.1 + pagerank * 0.5)
```

支撑越多、PageRank 越高 → 置信度越高。`search_memories` 按此值过滤与排序。

### 4.3 PageRank

- 增量更新：`SQLGraphStore.create_edge` 在事务内调用 `_update_pagerank_incremental`，只重算受影响节点及其一跳邻居，避免全图重算。
- 全量计算：`compute_pagerank(damping=0.85)`，可选 `node_ids` 子集。

### 4.4 演进

`MemoryNode.parent_memory_id` + `evolves` 边把记忆的历次演化串成链，便于追溯同一概念在不同时点的状态。

### 4.5 多租户

每个节点和边在写入时绑定 `ctx.account_id`；所有读操作在 `_check_access` 中校验。`SQLGraphStore` 额外在每条 SQL 会话上 `bind_tenant` 注入 GUC 作为纵深防御。

---

## 5. 典型调用流程

```python
from core.models import RequestContext
from core.graph_enums import RawDataType
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService

ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                     session_id="s1", trace_id="t1")
service = GraphService(MemoryGraphStore())

# 1) 原始 prompt 入图
raw = service.create_raw_node(
    raw_type=RawDataType.PROMPT,
    content="Write a haiku about the moon",
    ctx=ctx,
)

# 2) 抽取算子产出结构化结果
extracted = service.create_extracted_node(
    source_raw_id=raw.node_id,
    extractor_type="prompt_analyzer",
    extraction_schema="prompt_structure_v1",
    extracted_content={"intent": "creative_writing", "subject": "moon"},
    ctx=ctx,
)  # 自动建立 extracted_from 边

# 3) 记忆聚合
memory = service.create_memory_node(
    summary="User wants a haiku about the moon",
    supporting_node_ids=[extracted.node_id],
    topic="poetry",
    ctx=ctx,
)  # 自动建立 supports 边

# 4) 检索与可信度
service.compute_pagerank(ctx)
hits = service.search_memories(query="moon", ctx=ctx, min_confidence=0.3)

# 5) 溯源
chain = service.get_supporting_chain(memory.node_id, ctx)
```

---

## 6. 持久化

- 表：`graph_nodes`、`graph_edges`（见 `fs/sql_adapter/graph_schema.sql`）。
- 关键索引：`(node_type)`、`(raw_type)`、`(pagerank DESC)`、`(topic)`、`(updated_at DESC)`；边的 `(source_id)`、`(target_id)`、`(edge_type)`、`(source_id, target_id)`。
- 触发器：`trigger_graph_node_updated_at` 自动维护 `updated_at`。
- 数据库函数：
  - `calculate_pagerank_incremental(node_id)`：基于入/出边权重的简化 PageRank。
  - `get_node_neighbors(node_id, edge_types[])`：单跳邻居查询。
  - `get_supporting_chain(memory_id)`：递归 CTE，深度上限 5，输出完整证据链。

---

## 7. 测试覆盖

`tests/unit/test_graph.py`（27 个用例，4 个 TestClass）：

- `TestGraphModels`：节点/边 dataclass 的构造、URI、状态机、支撑管理。
- `TestMemoryGraphStore`：CRUD、邻居、BFS、增量 PageRank、检索、统计。
- `TestGraphService`：服务层组合操作（建链自动化、支撑链查询、检索、记忆更新）。
- `TestPromptExtractor`：Prompt 抽取算子的四个维度（intent / subject / style / format）。

契约测试：`tests/contract/test_graph_relations.py` 覆盖跨组件关系约束。

---

## 8. 已知边界与扩展点

- 当前实现的图操作以单跳邻居 + 有限深度 BFS 为主；超大规模图（如 >10^6 节点）的索引与查询优化、跨分区 PageRank 尚未纳入。
- 抽取算子目前只有 `PromptExtractor` 落地，其它模态按 `MultimodalExtractor` Protocol 扩展即可。
- 服务层未直接暴露 HTTP 路由；接入点需要通过 `GraphService` 与上层 API 层对接。
- 演进的撤销/分支合并（多条 `evolves` 链合并到当前记忆）当前仅靠 `parent_memory_id` 串成链，未做冲突合并策略。
