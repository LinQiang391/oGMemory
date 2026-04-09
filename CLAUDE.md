# ContextEngine — 完整项目上下文 (v3.1)

## 项目定位
面向 AI Agent 的长期记忆基础设施。将 Agent 与用户交互产生的知识，以可维护、可审计、可分层检索的方式持久化，并在后续推理中按需召回。物理存储层直接复用 OpenViking AGFS（agfs/ 目录）。

---

## 仓库结构与包依赖方向

```
ContextEngine/
├── agfs/                   # OpenViking AGFS Go 文件服务器（直接克隆）
├── core/                   # 【最底层】领域对象 + 接口 Protocol + 枚举 + 错误类型
│                           # ⚠️ core 不依赖任何其他本项目包
├── fs/agfs_adapter/        # ContextFS 接口的 AGFS 实现。依赖：core
├── extraction/             # 四类 CandidateExtractor。依赖：core
├── commit/                 # 写入编排主链路。依赖：core, fs
├── index/                  # 异步索引同步链路。依赖：core, fs
├── retrieval/              # 检索链路。依赖：core, fs
├── providers/              # 外部能力适配（LLM/Embedder/VectorIndex/RelationStore）。依赖：core（仅接口）
├── service/                # 对外 API 层。依赖：所有包。唯一强制注入 RequestContext 的位置
├── tests/contract/         # 跨团队 contract tests（不得删除或降级）
├── tests/integration/      # 端到端集成测试
├── tests/unit/             # 各包单元测试
├── tests/fixtures/         # mock 数据（fixture IndexRecords / localfs 节点目录）
└── team/                   # Agent 团队协调文件
```

**包依赖单向规则（任何人不得违反）：**
- `core` → 无依赖
- `fs / extraction / index / retrieval / providers` → 只依赖 `core`
- `commit` → 依赖 `core`, `fs`
- `service` → 依赖所有包，是唯一对外入口

---

## Roadmap — 三条能力线分阶段交付

| Phase | 能力线 | 交付物 | 工程依赖 |
|-------|--------|--------|----------|
| **Phase 0** | core 包冻结 | schema + 接口 + AGFS spec + contract tests | — |
| **Phase 1** | 抽取 Pipeline + 向量检索 | 写入链路 + L0/L1/L2 检索 + providers + service | Phase 0 |
| **Phase 2** | 向量＋图混合检索 + 大规模抽取优化 | RelationStore 图跳跃 + AI_Functions 接入（ai_chunking/ai_filter） | Phase 1 |
| **Phase 3** | 记忆压缩与聚合 + 自适应 Planner | 图聚类 + AI_aggregation + 动态 Query Planning | Phase 2 |

---

## 三个不可绕过的接触点

> **修改前必须在 team/messages/ 广播通知，等待相关 teammate 确认后再修改。**

1. **core 包 schema**：ContextNode / OutboxEvent / IndexRecord / WritePlan / TypedQuery / SeedHit / CandidateMemory / RelationEdge / WritePlan
2. **AGFS 目录约定**（见下文 §AGFS 目录 Spec）：写入顺序 + 文件名 + NodeStatus 枚举
3. **IndexRecord.filters**：必须含 `account_id` + `owner_space`，漏写导致跨租户静默泄漏（无运行时报错）

---

## §1  URI 规范（七类，Phase 0 冻结）

| 类别 | URI 模式 | 写入策略 | slug/id 规则 |
|------|---------|---------|-------------|
| Profile | `ctx://{account}/users/{user}/memories/profile` | ProfilePolicy（固定 URI，持续 merge） | 无 slug，全局唯一 |
| Preference | `ctx://{account}/users/{user}/memories/preferences/{slug}` | AggregateTopicPolicy | slug = 主题关键词，小写下划线 |
| Entity | `ctx://{account}/users/{user}/memories/entities/{slug}` | AggregateTopicPolicy | slug = 实体名，规范化 |
| Event | `ctx://{account}/users/{user}/memories/events/{event_id}` | AppendOnlyPolicy | event_id = timestamp_uuid，不复用 |
| Case | `ctx://{account}/agents/{agent}/memories/cases/{case_id}` | AppendOnlyPolicy | case_id = timestamp_uuid，不复用 |
| Pattern | `ctx://{account}/agents/{agent}/memories/patterns/{slug}` | AggregateTopicPolicy | slug = 经验主题 |
| Skill | `ctx://{account}/agents/{agent}/skills/{skill_name}` | SkillToolPolicy（固定 URI，累积） | skill_name = 技能标识符 |

---

## §2  AGFS 目录 Spec（Phase 0 冻结，禁止单方面修改）

### URI → AGFS 物理路径映射
```
ctx://{account}/users/{user}/memories/profile
→ /accounts/{account}/users/{user}/memories/profile/
    ├── content.md          # 写入步骤 ①（最大文件，先写）
    ├── .relations.json     # 写入步骤 ②
    ├── .abstract.md        # 写入步骤 ③a
    ├── .overview.md        # 写入步骤 ③b
    ├── .meta.json          # 写入步骤 ④ ★ 提交点（status: ACTIVE）
    └── .outbox/            # OutboxEvent 存放目录
        └── {event_id}.json
```

### NodeStatus 枚举（.meta.json 中的 status 字段）
- `PENDING`  — 节点目录已创建，写入未完成（步骤①-③完成，④未完成）
- `ACTIVE`   — 步骤④完成，节点对外可见，可被检索
- `BROKEN`   — Repair Job 检测到内容残缺，需人工处理

### 写入原子性约定（Repair Job 分支逻辑依赖此顺序）
- 步骤①失败 → 无 .meta.json，节点不存在，无需处理
- 步骤②③失败 → .meta.json 缺失或 status=PENDING，Repair Job 可检测并补写
- 步骤④完成 → status=ACTIVE = 提交成功，Outbox 在此之后登记
- **write-path-dev 修改写入顺序前必须通知 dfx-dev**（Repair Job 依赖此顺序推断中断位置）

### ContextFS.exists() 实现规则
检查 `.meta.json` 是否存在且 `status == ACTIVE`。PENDING 节点对上层不可见。

---

## §3  核心对象模型（core 包，Phase 0 冻结）

```python
# core/models.py

@dataclass
class RequestContext:
    account_id: str
    user_id: str
    agent_id: str
    session_id: str
    trace_id: str

@dataclass
class ContextNode:
    uri: str
    context_type: str       # MEMORY | SKILL | RESOURCE
    category: str           # profile|preference|entity|event|case|pattern|skill
    level: int              # 节点在目录树中的深度
    owner_space: str        # user_space_name() 或 agent_space_name()
    abstract: str           # ≤100字一句话摘要（→ .abstract.md）
    overview: str           # 结构化概述（→ .overview.md）
    content: str            # 完整原始内容（→ content.md）
    metadata: dict          # tags / timestamps / status / category

@dataclass
class RelationEdge:
    from_uri: str
    to_uri: str
    relation_type: str      # e.g. "related_to" | "derived_from" | "contradicts"
    weight: float           # 0.0-1.0
    reason: str

@dataclass
class CandidateMemory:
    category: str           # 同 ContextNode.category
    owner_scope: str        # "user" | "agent"
    routing_key: str        # slug 或 event_id 候选值
    abstract: str
    overview: str
    content: str
    confidence: float       # 0.0-1.0，低于阈值触发 skip

@dataclass
class WritePlan:
    action: str             # create | merge | append | skip
    target_uri: str
    merged_fields: dict     # merge 时的字段合并结果
    relation_edges: list[RelationEdge]

@dataclass
class IndexRecord:
    id: str                 # stable = sha256(uri + ":" + str(level))[:16]
    uri: str
    level: int              # 0=abstract / 1=overview / 2=content
    text: str               # 对应层级的文本内容
    filters: dict           # ⚠️ 必须含 account_id + owner_space，缺一不可
    metadata: dict          # category / context_type / has_overview / has_content

@dataclass
class OutboxEvent:
    event_id: str           # uuid4
    event_type: str         # UPSERT_CONTEXT|DELETE_CONTEXT|MOVE_CONTEXT|UPSERT_RELATION
    uri: str
    payload: dict
    status: str             # PENDING | PROCESSING | DONE | FAILED
    retry_count: int        # 超过 MAX_RETRY 移入 DLQ
    created_at: str         # ISO8601

@dataclass
class TypedQuery:
    text: str               # 原始查询文本，用于向量化
    context_type: str       # MEMORY | SKILL | RESOURCE
    categories: list[str]   # planner 推断的 category 列表，用于预过滤
    target_uri: str | None  # 若已知目标 URI，直接读取，跳过向量召回
    top_k: int              # L0 召回数量上限
    account_id: str         # ⚠️ service 层强制注入，不可由调用方覆盖
    owner_space: str | None # 进一步限定 user 或 agent 范围

@dataclass
class SeedHit:
    uri: str
    score: float
    level: int = 2
    parent_uri: str | None = None
    context_type: str = ""
    category: str = ""
    owner_space: str = ""
    abstract: str = ""
    has_overview: bool = False
    has_content: bool = False
    active_count: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## §4  核心接口规范（core/interfaces.py，Phase 0 冻结）

```python
from typing import Protocol

class ContextFS(Protocol):
    # ⚠️ 所有方法强制传入 RequestContext，在 agfs_adapter 内执行 _is_accessible
    def write_node(self, node: ContextNode, ctx: RequestContext) -> None: ...
    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode: ...
    def delete_node(self, uri: str, ctx: RequestContext) -> None: ...
    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None: ...
    def list_children(self, uri: str, ctx: RequestContext) -> list[str]: ...
    def exists(self, uri: str, ctx: RequestContext) -> bool: ...

class RelationStore(Protocol):
    # v1 实现 = 读写 AGFS 上的 .relations.json
    # Phase 2+ 可替换为图数据库实现，接口不变
    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]: ...
    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None: ...

class CandidateExtractor(Protocol):
    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]: ...

class MergePolicy(Protocol):
    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan: ...

class VectorIndex(Protocol):
    # 实现层负责注入 account_id 过滤，上层无感知
    def upsert(self, records: list[IndexRecord]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search_by_vector(self, query_vector: list[float], filters: dict, top_k: int) -> list[SeedHit]: ...
    def search_children(self, parent_uri: str, query_vector: list[float], filters: dict, top_k: int) -> list[SeedHit]: ...

class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

class LLM(Protocol):
    # 返回符合 schema 的 JSON dict，供 Extractor 和 Planner 使用
    def complete_json(self, prompt: str, schema: dict) -> dict: ...
```

---

## §5  写入策略行为规则（commit 包，Phase 1 实现）

### ProfilePolicy
- 目标：固定 URI `ctx://{account}/users/{user}/memories/profile`
- 行为：始终 merge。用更清晰、更稳定的表述**替换**旧描述，不追加。
- 冲突解决：新信息与旧信息矛盾时，以新信息为准（用户状态会改变）。

### AggregateTopicPolicy（适用 preference / entity / pattern）
- 按 `routing_key`（= slug）查找已有节点：
  - 存在 → merge，将新内容整合入 overview 和 content
  - 不存在 → create，新建节点
- 相似 slug 归并：编辑距离 < 2 或语义相似度 > 0.9 时视为同一主题，归并而非新建。

### AppendOnlyPolicy（适用 event / case）
- `event_id` / `case_id` 全局唯一（timestamp_uuid），始终 create，**不覆盖历史**。
- 通过 `relation_edges` 与相关节点（同类事件、关联实体）建立上下文连接。
- 同一 session 内的多条事件互相建立 `SEQUENCE` 关系边。

### SkillToolPolicy（适用 skill / tool）
- 固定 URI，始终存在。
- 累积写入：最佳实践、失败模式、推荐参数**追加**到 content，不覆盖。
- 使用频次、成功率等统计字段更新到 metadata。

---

## §6  IndexRecord 三层展开规则（index 包）

每个 ContextNode 异步展开为三条 IndexRecord：

| level | 来源文件 | filters 字段 | 语义 |
|-------|---------|-------------|------|
| 0 | .abstract.md | account_id, owner_space, category, context_type | L0 首轮召回入口 |
| 1 | .overview.md | account_id, owner_space, category, context_type | L1 Agent 决策层 |
| 2 | content.md（可分块） | account_id, owner_space, category, context_type | L2 精读层 |

**IndexRecord.id 生成规则**：`sha256(uri + ":" + str(level))[:16]`（稳定，支持幂等 upsert）

---

## §7  检索链路规则（retrieval 包）

### Planner 规则化决策（Phase 1 不做机器学习）
- 含 "偏好 / 习惯 / 背景 / 记忆" 关键词 → `context_type=MEMORY`
- 含 "如何 / 步骤 / 方法 / 怎么做" → `context_type=SKILL`
- 含 "文档 / 资料 / 参考" → `context_type=RESOURCE`
- `account_id` 由 service 层注入，Planner 透传，不设置

### MemoryReadService 触发条件与返回内容

| 工具调用 | 返回内容 | 触发条件 |
|---------|---------|---------|
| `search_memory(query)` | `SearchMemoryResult`（仅含 L2 的 `RetrievedBlock[]`，含 abstract） | 任意语义检索 |
| `read_memory(uri)` | `RetrievedBlock`（仅 `content_excerpt`，L2 md 文件内容） | Agent 显式调用 |

### 工具接口规范（service 层暴露给 Agent）

```python
# 工具 1：语义检索 — L0/L1/L2 混合召回，自动展开至 L2，仅返回 L2 结果
search_memory(
    query: str,
    ctx: RequestContext,
    *,
    top_k: int = 10,
    categories: list[str] | None = None,
    target_uri: str | None = None,
    score_threshold: float | None = None,
    mode: str = "quick",
) -> SearchMemoryResult
# service 层强制注入 account_id，调用方无法覆盖
# 返回：仅 L2 级 RetrievedBlock[]，含 uri / score / abstract / category

# 工具 2：读取 L2 md 文件内容
read_memory(
    uri: str,
    ctx: RequestContext,
) -> RetrievedBlock
# service 层校验：uri 的 account 前缀必须与 RequestContext.account_id 匹配，否则 403
# 仅返回 content_excerpt，不含 abstract/overview（search_memory 已提供 abstract）
```

---

## §8  多租户隔离规则

### 三级命名空间与 AGFS 物理路径
```
Account  ctx://{account}/          → /accounts/{account}/
User     ctx://{account}/users/    → /accounts/{account}/users/
Agent    ctx://{account}/agents/   → /accounts/{account}/agents/
```

### 同 account 内记忆共享场景
| 场景 | 默认行为 | 授权方式 |
|------|---------|---------|
| 同 agent 多会话 | ✓ 自动共享（同 agent_id，URI 相同） | 无需处理 |
| agent 读取 user 记忆 | ✗ 默认拒绝 | ACL 记录 + _is_accessible 扩展校验 |
| 同 account 多 agent 共享 pattern | ✗ 默认拒绝 | owner_space 读权限授权记录 |
| 跨 account | ✗ 永远拒绝（v1） | service 层直接返回 403 |

### 多节点部署一致性
- `localfs` 后端：单节点强一致，本地开发使用
- `s3fs` 后端：close-to-open 一致性，写关闭后全节点可见，生产多节点使用
- 多节点并发 merge 同一节点（如 Profile）的写安全性：**P1-03 待决问题，测试期验证**

### 向量检索隔离
- `TypedQuery.account_id` 由 service 层注入，不可被覆盖
- `VectorIndex.search()` 在引擎层面过滤，retrieval 层无感知
- `IndexRecord.filters` 漏写 account_id 时无运行时报错，只有 contract test 能拦截

---

## §9  一致性与 DFX 规则（index 包）

### Outbox Worker 行为
- 正常路径：read .outbox/ → embed → upsert VectorIndex → delete .outbox/{event_id}.json
- 失败路径：指数退避重试，retry_count >= MAX_RETRY（默认5）时移入 `.outbox/dlq/`，触发告警
- 幂等保证：IndexRecord.id = sha256(uri+level)[:16]，重复 upsert 安全

### Repair Job 触发与行为
- 触发条件：定时（每日低峰）+ 写入失败主动触发
- 扫描路径：`/accounts/**/`，检查每个节点的 `.meta.json`
- 分支逻辑与 §2 写入顺序严格对应：
  - `.meta.json` 不存在 + `content.md` 存在 → 补写 .meta.json status=ACTIVE，登记 OutboxEvent
  - `.meta.json` 不存在 + `content.md` 不存在 → 记录日志，跳过（步骤①未完成的正常中断）
  - `status=PENDING` + 文件完整 → 更新 status=ACTIVE，登记 OutboxEvent
  - `status=PENDING` + 文件缺失 → 更新 status=BROKEN，人工介入

### OutboxEvent 持久化方案（P1-02 待决）
当前倾向：写入 AGFS 特殊目录 `.outbox/{event_id}.json`，与 VFS 同后端，多节点可见。

---

## §10  待决事项（开发过程中遇到需查此表）

| 编号 | 问题 | 当前倾向 | 决策时机 |
|------|------|---------|---------|
| P1-01 | AGFS 孤立节点清理策略：触发频率与修复范围 | 定时（每日低峰）+ 写入失败主动触发 | Phase 1 写入链路稳定后 |
| **P1-02** | OutboxEvent 持久化：写在哪里 | 写入 AGFS `.outbox/` 目录 | Phase 1 启动前确认 |
| **P1-03** | **多节点并发 merge 同一节点的写安全性（Profile 的竞争写）** | 写前读+merge+写非原子；测试期验证是否需要 AGFS 路径级锁 | Phase 1 多节点测试期 |
| P2-01 | RelationStore 迁移时机：.relations.json → 图数据库 | 关系查询超过 2-hop 或图聚类需求确认时 | Phase 2 规划期 |
| P2-02 | GaussVector AI_Functions 接入协议（ai_chunking/ai_filter） | 预留 Provider 接口，SDK 确认后实现 | Phase 2 启动前 |
| P3-01 | 记忆健康度评分机制 | 访问频率 + 信息密度；尚无定论 | Phase 3 规划期 |

---

## §11  Git Commit 规范（所有 Teammate 强制遵守）

### 原子化提交原则
每完成一个**独立可测试的最小单元**立即 commit，不攒大 commit：
- 一个 dataclass / Protocol 定义 → 一次 commit
- 一个方法实现 + 对应 unit test → 一次 commit
- 一个 contract test 文件 → 一次 commit
- Bug fix → 一次 commit（message 包含 "fix:"）

### Commit Message 格式
```
{type}({scope}): {简短描述}

类型: feat | fix | test | refactor | docs | chore
范围: core | fs | extraction | commit | index | retrieval | providers | service | contract

示例:
feat(core): add ContextNode and IndexRecord dataclasses
feat(core): add ContextFS and VectorIndex Protocol interfaces
test(contract): add test_index_record_filters contract test
feat(fs): implement AGFSContextFS._is_accessible() access control
fix(index): ensure account_id always present in IndexRecord.filters
feat(commit): implement ContextWriter with atomic write order ①-④
```

### 阶段性里程碑提交（Leader 在 Phase 完成时执行）
```
milestone(phase-N): Phase N complete — all tests GREEN

详细说明各 Phase 包含的能力和关键决策。
```

### 提交前检查
```bash
pytest tests/   # 必须全部通过，禁止带失败的 test commit
git diff --stat # 确认 diff 范围符合原子化原则
```

---

## §12  团队协作协议

- 任务状态在 `team/tasks.md` 中维护，用 `{agent}` 标记声明权
- Peer 消息写入 `team/messages/{sender}_to_{receiver}_{timestamp}.md`
- 修改接触点前广播通知，等待相关 teammate 确认
- **每个 Phase 开始前：Leader 解散不需要的 Teammate，spawn 当前 Phase 需要的 Teammate**
- **每个 Phase 结束时：所有测试 GREEN → 阶段性 commit → Leader 宣布完成 → 解散当前团队**
