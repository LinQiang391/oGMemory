# oG-Memory 检索链路设计文档

> 基于 `search319` 分支代码，最后更新：2026-03-17

---

## 1. 设计总览

oG-Memory 检索链路采用 **4 阶段管线** 设计，实现从自然语言查询到 L2 级结构化记忆结果的端到端检索。

**核心设计原则：**

- **L0/L1/L2 混合召回，L2 唯一输出** — 全局搜索所有层级，但最终只返回 L2 级叶子记忆
- **自动递归展开** — L0/L1 命中自动展开至 L2，无需调用方显式请求
- **渐进披露** — `search_memory` 返回摘要，`read_memory` 按需读取完整内容
- **纯结构化输出** — 不生成 prompt，由外部 agent 定制化拼接

### 1.1 管线总览图

```
┌─────────────┐   TypedQuery[]   ┌─────────────┐   SeedResult     ┌──────────────────┐   LeafHit[]    ┌─────────────────┐   RetrievedBlock[]
│ Stage 1     │ ──────────────→  │ Stage 2     │ ──────────────→  │ Stage 3          │ ────────────→ │ Stage 4         │
│             │                  │             │  starting_points  │                  │  递归展开      │                 │
│ QueryPlanner│                  │SeedRetriever│  + initial_cands  │HierarchicalSearch│  + score 传播  │ AssemblyService │
└─────────────┘                  └─────────────┘                   └──────────────────┘               └────────┬────────┘
                                                                                                              │
                                                                                                              ▼
                                                                                                  ┌─────────────────────┐
                                                                                                  │ SearchMemoryResult  │
                                                                                                  │ (仅 L2, top-k)     │
                                                                                                  └─────────────────────┘
```

### 1.2 时序图

```
Agent            MemoryReadAPI    Pipeline         QueryPlanner     SeedRetriever    Embedder     openGauss        HierarchicalSearcher  AssemblyService
  │                  │               │                │                │               │              │                   │                   │
  │ search_memory()  │               │                │                │               │              │                   │                   │
  │─────────────────→│ run()         │                │                │               │              │                   │                   │
  │                  │──────────────→│                │                │               │              │                   │                   │
  │                  │               │                │                │               │              │                   │                   │
  │                  │               │  plan()        │                │               │              │                   │                   │
  │                  │               │───────────────→│                │               │              │                   │                   │
  │                  │               │  TypedQuery[]  │                │               │              │                   │                   │
  │                  │               │←───────────────│                │               │              │                   │                   │
  │                  │               │                                 │               │              │                   │                   │
  │                  │               │  ┌──── 循环: 每个 TypedQuery ──────────────────────────────────────────────────────┐                   │
  │                  │               │  │ search()                    │               │              │                   │                   │
  │                  │               │──│────────────────────────────→│  embed_texts() │              │                   │                   │
  │                  │               │  │                             │──────────────→│              │                   │                   │
  │                  │               │  │                             │  query_vector │              │                   │                   │
  │                  │               │  │                             │←──────────────│              │                   │                   │
  │                  │               │  │                             │  search_by_vector(L0/L1/L2)  │                   │                   │
  │                  │               │  │                             │─────────────────────────────→│                   │                   │
  │                  │               │  │                             │  [SeedHit(L0), ..., (L2)]  │                   │                   │
  │                  │               │  │                             │←─────────────────────────────│                   │                   │
  │                  │               │  │  SeedResult                 │               │              │                   │                   │
  │                  │               │←─│─────────────────────────────│               │              │                   │                   │
  │                  │               │  │                             │               │              │                   │                   │
  │                  │               │  │ expand()                                                   │                   │                   │
  │                  │               │──│──────────────────────────────────────────────────────────→ │                   │                   │
  │                  │               │  │                                                            │                   │                   │
  │                  │               │  │                             │               │   ┌──── 循环: BFS 优先队列 ────┐  │                   │
  │                  │               │  │                             │               │   │ search_children()         │  │                   │
  │                  │               │  │                             │               │   │──────────────────────────→│  │                   │
  │                  │               │  │                             │               │   │ [SeedHit...]            │  │                   │
  │                  │               │  │                             │               │   │←──────────────────────────│  │                   │
  │                  │               │  │                             │               │   │ score 传播 + 收集L2      │  │                   │
  │                  │               │  │                             │               │   └──────────────────────────┘  │                   │
  │                  │               │  │                                              hotness blending                 │                   │
  │                  │               │  │  [LeafHit, ...]                                                               │                   │
  │                  │               │←─│──────────────────────────────────────────────────────────────────────────────  │                   │
  │                  │               │  └─────────────────────────────────────────────────────────────────────────────────┘                   │
  │                  │               │                                                                                                      │
  │                  │               │  assemble()                                                                                          │
  │                  │               │─────────────────────────────────────────────────────────────────────────────────────────────────────→ │
  │                  │               │  [RetrievedBlock(L2), ...]                                                                           │
  │                  │               │←─────────────────────────────────────────────────────────────────────────────────────────────────────  │
  │                  │               │                │                │               │              │                   │                   │
  │                  │  Result       │                │                │               │              │                   │                   │
  │                  │←──────────────│                │                │               │              │                   │                   │
  │  SearchMemoryResult (JSON)       │                │                │               │              │                   │                   │
  │←─────────────────│               │                │                │               │              │                   │                   │
  │                  │               │                │                │               │              │                   │                   │
  │  [可选] Agent 需要详情           │                │                │               │              │                   │                   │
  │ read_memory(uri) │               │                │                │               │              │                   │                   │
  │─────────────────→│               │                │                │               │              │                   │                   │
  │  RetrievedBlock  │               │                │                │               │              │                   │                   │
  │  (content_       │               │                │                │               │              │                   │                   │
  │   excerpt:       │               │                │                │               │              │                   │                   │
  │   md文件内容)     │               │                │                │               │              │                   │                   │
  │←─────────────────│               │                │                │               │              │                   │                   │
```

---

## 2. 数据模型

### 2.1 目录结构与层级映射

```
ctx://acct-1/users/u1/memories/
├── abstract.md       → L0, uri="ctx://…/memories"
├── overview.md       → L1, uri="ctx://…/memories"
├── profile/
│   ├── abstract.md   → L0, uri="ctx://…/memories/profile"
│   ├── overview.md   → L1, uri="ctx://…/memories/profile"
│   ├── coding.md     → L2, uri="ctx://…/memories/profile/coding.md"
│   └── editor.md     → L2, uri="ctx://…/memories/profile/editor.md"
└── events/
    ├── abstract.md   → L0, uri="ctx://…/memories/events"
    ├── overview.md   → L1, uri="ctx://…/memories/events"
    └── meeting.md    → L2, uri="ctx://…/memories/events/meeting.md"
```

**URI 规则：**

| 层级 | 来源文件 | uri | parent_uri |
|------|----------|-----|------------|
| L0 | abstract.md | 目录路径（不含文件名） | 上级目录路径 |
| L1 | overview.md | 目录路径（与 L0 相同） | 上级目录路径 |
| L2 | 具体 .md 文件 | 完整文件路径 | 所属目录路径 |

### 2.2 数据库表结构（openGauss）

```sql
CREATE TABLE IF NOT EXISTS vector_index (
    id          VARCHAR(16) PRIMARY KEY,     -- sha256(uri:level)[:16]
    uri         VARCHAR(512) NOT NULL,
    level       INTEGER NOT NULL,            -- 0=abstract / 1=overview / 2=content
    text        TEXT NOT NULL,               -- 用于 embedding 的文本内容
    embedding   vector(1536) NOT NULL,
    filters     JSONB NOT NULL,              -- {account_id, owner_space}
    metadata    JSONB NOT NULL DEFAULT '{}', -- {parent_uri, context_type, category, ...}
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 索引策略
CREATE INDEX idx_vector_embedding    ON vector_index USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_vector_account      ON vector_index ((filters->>'account_id'));
CREATE INDEX idx_vector_owner_space  ON vector_index ((filters->>'owner_space'));
CREATE INDEX idx_vector_uri          ON vector_index (uri);
CREATE INDEX idx_vector_level        ON vector_index (level);
CREATE INDEX idx_vector_metadata     ON vector_index USING GIN (metadata);
```

**示例数据行：**

| id | uri | level | text | filters | metadata |
|----|-----|-------|------|---------|----------|
| `a1b2c3d4e5f6a1b2` | `ctx://acct-1/users/u1/memories/profile` | 0 | "用户是高级后端工程师，偏好 Python..." | `{"account_id":"acct-1","owner_space":"user_u1"}` | `{"parent_uri":"ctx://…/memories","context_type":"MEMORY","category":"profile"}` |
| `b2c3d4e5f6a1b2c3` | `ctx://acct-1/users/u1/memories/profile` | 1 | "## 用户背景\n- 5年 Python...\n## 技术栈..." | `{"account_id":"acct-1","owner_space":"user_u1"}` | `{"parent_uri":"ctx://…/memories","context_type":"MEMORY","category":"profile"}` |
| `c3d4e5f6a1b2c3d4` | `ctx://…/profile/coding.md` | 2 | "## 编程偏好\n偏好 Vim 键位绑定..." | `{"account_id":"acct-1","owner_space":"user_u1"}` | `{"parent_uri":"ctx://…/memories/profile","context_type":"MEMORY","category":"profile"}` |

### 2.3 核心数据类型

```python
# --- Stage 2 输出 ---
@dataclass
class SeedHit:
    uri: str                          # 记录 URI
    score: float                      # 向量相似度 (cosine)
    level: int = 2                    # 0/1/2
    parent_uri: str | None = None     # 父目录 URI
    context_type: str = ""            # MEMORY / SKILL / RESOURCE
    category: str = ""                # profile / preference / entity / event ...
    owner_space: str = ""
    abstract: str = ""                # text[:200] 摘要
    has_overview: bool = False
    has_content: bool = False
    active_count: int = 0             # 访问计数（用于 hotness）
    updated_at: str | None = None     # 最后更新时间（用于 hotness）
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SeedResult:
    starting_points: list[SeedHit]     # L0/L1 命中 → 待递归展开
    initial_candidates: list[SeedHit]  # L2 直接命中 → 直接收集
    query_vector: list[float]            # 查询向量（复用于 search_children）
    root_uris: list[str]                 # 根 URI 兜底

# --- Stage 3 输出 ---
@dataclass
class LeafHit:
    uri: str
    score: float              # blended score（语义 + hotness）
    level: int = 2
    parent_uri: str | None = None
    category: str = ""
    abstract: str = ""
    # ... 其他字段

# --- Stage 4 输出 ---
@dataclass
class RetrievedBlock:
    uri: str
    level_hit: str = "L2"
    score: float = 0.0
    category: str = ""
    abstract: str | None = None
    overview: str | None = None          # read_memory 时填充
    content_excerpt: str | None = None   # read_memory 时填充
    match_reason: str = ""

@dataclass
class SearchMemoryResult:
    request_id: str
    query: str
    typed_queries: list[TypedQuery]
    hits: list[RetrievedBlock]
    trace: Any = None
```

---

## 3. Stage 1 — QueryPlanner（查询规划）

> 文件：`retrieval/query_planner.py`

### 3.1 职责

将自然语言查询分类为一个或多个 `TypedQuery`，决定搜索范围和过滤条件。

### 3.2 分类规则

```python
# 关键词正则匹配
_MEMORY_PATTERNS   = [r"偏好|背景|身份|记忆|profile|preference|remember"]
_SKILL_PATTERNS    = [r"怎么做|步骤|命令|how\s*to|workflow|procedure"]
_RESOURCE_PATTERNS = [r"文档|资料|规范|document|spec|reference"]
```

| 匹配结果 | 行为 |
|----------|------|
| 命中 1 种类型 | 生成 1 个 TypedQuery |
| 命中 0 或 ≥2 种 | 退化为三路查询（MEMORY + SKILL + RESOURCE） |

### 3.3 示例

```python
# 输入
query = "我的编程偏好是什么"
ctx = RequestContext(account_id="acct-1", user_id="u1", ...)

# plan() 过程
_classify("我的编程偏好是什么")
  → 匹配 "偏好" → hits = ["MEMORY"]
  → 单一命中，生成 1 个 TypedQuery

# 输出
[TypedQuery(
    text="我的编程偏好是什么",
    context_type="MEMORY",
    categories=[],
    top_k=10,
    account_id="acct-1",        # 强制注入
    owner_space="user_u1",
)]
```

### 3.4 降级机制

Planner 抛异常时，pipeline 自动生成三路 fallback 查询：

```python
# pipeline.py → _run_planner()
except Exception as exc:
    queries = [
        TypedQuery(text=query, context_type=ct.value, ...)
        for ct in ContextType  # MEMORY, SKILL, RESOURCE
    ]
```

---

## 4. Stage 2 — SeedRetriever（种子召回）

> 文件：`retrieval/seed_retriever.py`

### 4.1 职责

执行 **L0/L1/L2 全层级混合向量搜索**，然后按 level 分流为两个流：

- `starting_points`：L0/L1 命中（需递归展开）
- `initial_candidates`：L2 直接命中（直接进入候选池）

### 4.2 执行流程

```
                    ┌─────────────┐
                    │ TypedQuery  │
                    └──────┬──────┘
                           │
                           ▼
                ┌────────────────────┐
                │ embed_texts()      │
                │ → query_vector     │
                └──────────┬─────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ search_by_vector             │
            │ (level=[0, 1, 2], 混合召回)  │
            └──────────────┬───────────────┘
                           │
                    ┌──────┴──────┐
                    │ 按 level 分流│
                    └──┬───────┬──┘
                       │       │
            level < 2  │       │  level == 2
                       ▼       ▼
          ┌──────────────┐  ┌───────────────────┐
          │starting_points│  │initial_candidates │
          └───────┬──────┘  └─────────┬─────────┘
                  │                   │
   ┌──────────┐  │                   │
   │root_uris │──┘                   │
   │(兜底)    │                      │
   └──────────┘                      │
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    ┌──────────────┐
                    │  SeedResult  │
                    └──────────────┘
```

### 4.3 核心代码

**全局搜索：**

```python
# seed_retriever.py → _global_vector_search()
filters = {
    "level": [0, 1, 2],               # 混合召回关键
    "account_id": ctx.account_id,      # 租户隔离
}
# 可选过滤
if typed_query.context_type:
    filters["context_type"] = typed_query.context_type
if typed_query.owner_space:
    filters["owner_space"] = typed_query.owner_space

return self.vector_index.search_by_vector(
    query_vector=query_vector,
    filters=filters,
    top_k=self.config.global_search_topk,  # 默认 5
)
```

对应的 SQL（openGauss）：

```sql
SELECT id, uri, level, text, filters, metadata,
       1 - (embedding <=> query_vector::vector) AS score
FROM   vector_index
WHERE  filters->>'account_id' = 'acct-1'
  AND  level = ANY(ARRAY[0, 1, 2])
  AND  metadata @> '{"context_type":"MEMORY"}'::jsonb
ORDER  BY embedding <=> query_vector::vector
LIMIT  5
```

**分流逻辑：**

```python
# seed_retriever.py → _merge_starting_points()
for hit in global_results:
    if hit.level == 2:
        initial_candidates.append(hit)   # L2 直接收集
    else:
        if hit.uri not in seen:          # L0/L1 同 URI 去重
            points.append(hit)
            seen.add(hit.uri)

# 补充 root_uris 兜底
for uri in root_uris:
    if uri not in seen:
        points.append(SeedHit(uri=uri, score=0.0, level=0))
```

> **去重说明**：L0 和 L1 共享同一个 URI（目录路径），`seen` 集合确保每个目录只有一个代表进入 `starting_points`，避免重复展开。先到者（通常是语义得分更高者）胜出。

### 4.4 示例

查询 "我的编程偏好是什么"，全局搜索返回：

| uri | level | score |
|-----|-------|-------|
| `ctx://…/memories/profile` | 0 | 0.85 |
| `ctx://…/memories/profile` | 1 | 0.78 |
| `ctx://…/memories/profile/coding.md` | 2 | 0.91 |
| `ctx://…/memories/events` | 0 | 0.42 |

分流结果：

```
starting_points = [
    SeedHit(uri="…/profile", score=0.85, level=0),   # L1(0.78) 被去重
    SeedHit(uri="…/events",  score=0.42, level=0),
]
initial_candidates = [
    SeedHit(uri="…/profile/coding.md", score=0.91, level=2),
]
```

---

## 5. Stage 3 — HierarchicalSearcher（层级递归展开）

> 文件：`retrieval/hierarchical_searcher.py`

### 5.1 职责

从 L0/L1 起始点递归展开子节点，直到找到所有可达的 L2 叶子节点，同时进行 **score 传播** 和 **hotness blending**。

### 5.2 算法概览

```
┌─────────────────────────────────────────────┐
│ 初始化                                       │
│  collected_by_uri = { L2 直接命中 }           │
│  dir_queue = [ starting_points (L0/L1) ]     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
                ┌──────────────┐     否
                │ 队列非空?     │─────────────────────────────────┐
                └──────┬───────┘                                  │
                       │ 是                                       │
                       ▼                                          │
          ┌────────────────────────┐                              │
          │ Pop 最高分目录节点      │                              │
          │ (current_uri, score)   │                              │
          └───────────┬────────────┘                              │
                      │                                           │
                      ▼                                           │
          ┌────────────────────────┐                              │
          │ search_children        │                              │
          │ parent_uri=current_uri │                              │
          └───────────┬────────────┘                              │
                      │                                           │
                      ▼                                           │
          ┌────────────────────────┐                              │
          │ 遍历每个 child:         │                              │
          │                        │                              │
          │ final_score =          │                              │
          │  α×child + (1-α)×parent│                              │
          └───┬───────────────┬────┘                              │
              │               │                                   │
     level==2 │               │ level<2                           │
     (叶子)   │               │ (目录)                            │
              ▼               ▼                                   │
   ┌──────────────┐  ┌───────────────┐                            │
   │ 收集到        │  │ 入队继续展开   │                            │
   │collected_by_ │  │ dir_queue     │                            │
   │uri (保留高分) │  └───────────────┘                            │
   └──────┬───────┘                                               │
          │                                                       │
          ▼                                                       │
   ┌──────────────────────────┐                                   │
   │ 收敛检测                   │                                   │
   │ top-k 连续 3 轮不变?       │                                   │
   └──┬───────────────────┬───┘                                   │
      │ 未收敛             │ 已收敛                                 │
      │                   ▼                                       │
      │          ┌────────────────┐                                │
      │          │ 提前终止        │────────┐                       │
      │          └────────────────┘        │                       │
      │                                    │                       │
      └──────→ (回到队列非空检查)            │                       │
                                           │                       │
                       ┌───────────────────┘                       │
                       │                   ┌───────────────────────┘
                       ▼                   ▼
               ┌─────────────────────────────┐
               │ 取 top-k 候选                │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │ hotness blending             │
               │ blended = 0.8×sem + 0.2×hot │
               │ → LeafHit 列表              │
               └─────────────────────────────┘
```

### 5.3 核心代码详解

#### 5.3.1 初始化

```python
# hierarchical_searcher.py → _recursive_search()
collected_by_uri: dict[str, dict] = {}     # URI → L2 候选 (保留最高分)
dir_queue: list[tuple[float, str]] = []    # 优先队列 (负分, URI)
visited: set[str] = set()                  # 已展开的 URI

# 预收集 L2 直接命中
for hit in initial_candidates:
    if hit.level == 2 and hit.uri:
        collected_by_uri[hit.uri] = self._hit_to_dict(hit, hit.score)

# L0/L1 入队
for hit in starting_points:
    heapq.heappush(dir_queue, (-hit.score, hit.uri))
```

#### 5.3.2 递归展开 + Score 传播

```python
while dir_queue:
    neg_score, current_uri = heapq.heappop(dir_queue)
    current_score = -neg_score

    if current_uri in visited:     # 去重（L0/L1 同 URI 只展开一次）
        continue
    visited.add(current_uri)

    # 查找子节点
    children = self.vector_index.search_children(
        parent_uri=current_uri,    # SQL: metadata->>'parent_uri' = current_uri
        query_vector=query_vector,
        filters=base_filters,
        top_k=pre_filter_limit,
    )

    for child in children:
        # ★ Score 传播公式
        final_score = alpha * child.score + (1 - alpha) * current_score

        if final_score <= effective_threshold:
            continue

        # 更新候选（保留更高分）
        uri = child.uri
        prev = collected_by_uri.get(uri)
        if prev is None or final_score > prev.get("_final_score", 0):
            collected_by_uri[uri] = self._hit_to_dict(child, final_score)

        # L2 = 叶子节点；L0/L1 = 入队继续展开
        if uri not in visited and child.level != 2:
            heapq.heappush(dir_queue, (-final_score, uri))
```

对应的 `search_children` SQL：

```sql
SELECT id, uri, level, text, filters, metadata,
       1 - (embedding <=> query_vector::vector) AS score
FROM   vector_index
WHERE  filters->>'account_id' = 'acct-1'
  AND  metadata->>'parent_uri' = 'ctx://…/memories/profile'
ORDER  BY embedding <=> query_vector::vector
LIMIT  20
```

#### 5.3.3 收敛检测

```python
current_topk_uris = {取 top-k URI 集合}

if current_topk_uris == prev_topk_uris and len >= limit:
    convergence_rounds += 1
    if convergence_rounds >= 3:   # max_convergence_rounds
        break                      # 提前终止
else:
    convergence_rounds = 0
    prev_topk_uris = current_topk_uris
```

#### 5.3.4 Hotness Blending

递归展开完成后，对收集到的 L2 候选应用时间衰减加权：

```python
# hierarchical_searcher.py → _convert_results()
def _convert_results(self, candidates, limit):
    for c in candidates:
        semantic_score = c["_final_score"]

        # hotness = sigmoid(log1p(active_count)) × exp_decay(age)
        h_score = hotness_score(
            active_count=c.get("active_count", 0),
            updated_at=updated_at_val,
            half_life_days=self.cfg.hotness_half_life_days,  # 默认 7 天
        )

        # ★ Blending 公式
        blended = (1 - hotness_alpha) * semantic_score + hotness_alpha * h_score
        # 默认: blended = 0.8 × semantic + 0.2 × hotness

        results.append(LeafHit(uri=..., score=blended, ...))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]
```

**Hotness 公式（`retrieval/hotness.py`）：**

```
hotness = freq × recency

freq    = sigmoid(log1p(active_count))    # 访问频率
recency = exp(-ln2 / half_life × age)     # 时间衰减，半衰期 7 天
```

### 5.4 Score 体系总结

| 分数 | 公式 | 含义 |
|------|------|------|
| `child.score` | `1 - cosine_distance(embedding, query_vector)` | 子节点与查询的向量相似度 |
| `current_score` | 父节点的 `final_score` | 父目录的传播分 |
| `final_score` | `α × child.score + (1-α) × current_score` | 语义传播分（α 默认 0.5） |
| `h_score` | `sigmoid(log1p(count)) × exp_decay(age)` | 热度分 |
| `blended` | `(1-β) × final_score + β × h_score` | 最终排序分（β 默认 0.2） |

### 5.5 完整实例跟踪

**场景**：查询 "我的编程偏好是什么"，`alpha=0.5`, `hotness_alpha=0.2`

**初始状态：**

```
collected_by_uri = {
    "…/profile/coding.md": {_final_score: 0.91}   ← Stage 2 直接命中
}
dir_queue = [(-0.85, "…/profile"), (-0.42, "…/events")]
```

**第 1 轮：Pop "…/profile" (current_score=0.85)**

`search_children(parent_uri="…/profile")` 返回：

| child uri | level | child.score |
|-----------|-------|-------------|
| `…/profile/coding.md` | 2 | 0.88 |
| `…/profile/editor.md` | 2 | 0.72 |

Score 传播：

```
coding.md: final = 0.5 × 0.88 + 0.5 × 0.85 = 0.865
  → 已有 0.91 > 0.865 → 不替换

editor.md: final = 0.5 × 0.72 + 0.5 × 0.85 = 0.785
  → 新增到 collected
```

```
collected_by_uri = {
    "…/profile/coding.md": {_final_score: 0.91},    ← 保持
    "…/profile/editor.md": {_final_score: 0.785},   ← 新增
}
```

**第 2 轮：Pop "…/events" (current_score=0.42)**

`search_children(parent_uri="…/events")` 返回：

| child uri | level | child.score |
|-----------|-------|-------------|
| `…/events/meeting.md` | 2 | 0.38 |

Score 传播：

```
meeting.md: final = 0.5 × 0.38 + 0.5 × 0.42 = 0.400
```

```
collected_by_uri = {
    "…/profile/coding.md":  {_final_score: 0.91},
    "…/profile/editor.md":  {_final_score: 0.785},
    "…/events/meeting.md":  {_final_score: 0.400},
}
```

**队列为空，取 top-3 进入 hotness blending：**

| uri | final_score | hotness | blended (0.8×sem + 0.2×hot) |
|-----|-------------|---------|-----|
| `…/profile/coding.md` | 0.91 | 0.65（3天前更新，count=5） | 0.858 |
| `…/profile/editor.md` | 0.785 | 0.30（15天前，count=1） | 0.688 |
| `…/events/meeting.md` | 0.400 | 0.82（今天更新，count=8） | 0.484 |

**最终 LeafHit 列表（按 blended 降序）：**

```
1. [0.858] ctx://…/profile/coding.md   abstract: "偏好 Vim 键位绑定，常用 Python..."
2. [0.688] ctx://…/profile/editor.md   abstract: "使用 VSCode + Vim 插件..."
3. [0.484] ctx://…/events/meeting.md   abstract: "团队代码规范评审会决定..."
```

### 5.6 多层目录递归示例

```
ctx://…/memories/ (L0, score=0.70)
├── profile/ (L0, score=0.80)
│   ├── coding.md  (L2, score=0.88)
│   └── editor.md  (L2, score=0.72)
└── events/ (L0, score=0.60)
    └── meeting.md (L2, score=0.55)
```

展开过程（`alpha=0.5`）：

```
第 1 轮: Pop memories(0.70) → search_children
  → profile L0(similarity=0.80): final = 0.5×0.80 + 0.5×0.70 = 0.75 → 入队
  → events  L0(similarity=0.60): final = 0.5×0.60 + 0.5×0.70 = 0.65 → 入队

第 2 轮: Pop profile(0.75) → search_children
  → coding.md  L2(sim=0.88): final = 0.5×0.88 + 0.5×0.75 = 0.815 → 收集
  → editor.md  L2(sim=0.72): final = 0.5×0.72 + 0.5×0.75 = 0.735 → 收集

第 3 轮: Pop events(0.65) → search_children
  → meeting.md L2(sim=0.55): final = 0.5×0.55 + 0.5×0.65 = 0.600 → 收集

队列空 → 结束
```

Score 传播链（alpha=0.5）：

```
                                                        ┌─ coding.md  L2  final=0.815 ✓
                       ┌─ profile  L0  final=0.75  ────┤
                       │                                └─ editor.md  L2  final=0.735 ✓
  memories  L0  0.70 ──┤
                       │                                ┌─ meeting.md L2  final=0.600 ✓
                       └─ events   L0  final=0.65  ────┘

  传播计算:
    profile : 0.5 × 0.80(sim) + 0.5 × 0.70(parent) = 0.75
    coding  : 0.5 × 0.88(sim) + 0.5 × 0.75(parent) = 0.815
    editor  : 0.5 × 0.72(sim) + 0.5 × 0.75(parent) = 0.735
    events  : 0.5 × 0.60(sim) + 0.5 × 0.70(parent) = 0.65
    meeting : 0.5 × 0.55(sim) + 0.5 × 0.65(parent) = 0.600
```

---

## 6. Stage 4 — AssemblyService（结果组装）

> 文件：`retrieval/assembly_service.py`

### 6.1 职责

过滤仅保留 L2 结果，排序截断，转换为 `RetrievedBlock` 输出。

### 6.2 核心代码

```python
# assembly_service.py → assemble()
def assemble(self, typed_query, leaf_hits, seed_result):
    candidates = list(leaf_hits) if leaf_hits else []

    # 兜底：如果 Stage 3 无输出，从 SeedResult 直取 L2
    if not candidates and seed_result:
        candidates = self._l2_from_seed(seed_result)

    l2_only = [h for h in candidates if h.level == 2]
    l2_only.sort(key=lambda h: h.score, reverse=True)
    top = l2_only[:typed_query.top_k]

    return [
        RetrievedBlock(
            uri=hit.uri,
            level_hit="L2",
            score=hit.score,
            category=hit.category,
            abstract=hit.abstract or None,
            match_reason=f"vector score {hit.score:.4f}",
        )
        for hit in top
    ]
```

### 6.3 示例输出

```python
SearchMemoryResult(
    request_id="a1b2c3d4e5f6",
    query="我的编程偏好是什么",
    typed_queries=[TypedQuery(text="我的编程偏好是什么", context_type="MEMORY", ...)],
    hits=[
        RetrievedBlock(
            uri="ctx://acct-1/users/u1/memories/profile/coding.md",
            level_hit="L2",
            score=0.858,
            category="profile",
            abstract="偏好 Vim 键位绑定，常用 Python 和 Go，习惯使用类型注解...",
            match_reason="vector score 0.8580",
        ),
        RetrievedBlock(
            uri="ctx://acct-1/users/u1/memories/profile/editor.md",
            level_hit="L2",
            score=0.688,
            category="profile",
            abstract="使用 VSCode + Vim 插件，暗色主题，字号 14...",
            match_reason="vector score 0.6880",
        ),
    ],
    trace=None,
)
```

---

## 7. read_memory — 按需读取 L2 md 文件内容

> 文件：`retrieval/memory_read_service.py`、`service/api.py`

### 7.1 定位

`search_memory` 已返回 L2 结果的 `abstract`（摘要），agent 需要详细内容时调用 `read_memory` 读取对应 `.md` 文件的完整内容。

> **注意**：`read_memory` 只读 L2 的 md 文件内容，不读 abstract.md 和 overview.md。abstract 已在 `search_memory` 结果中返回。

### 7.2 核心代码

```python
# memory_read_service.py → read()
def read(self, uri, ctx):
    self._check_acl(uri, ctx)               # account_id 前缀校验

    block = RetrievedBlock(uri=uri, level_hit="L2")
    node = self._fs.read_node(uri, ctx)      # 读 AGFS 中的 md 文件

    block.category = node.category
    block.owner_space = node.owner_space
    block.content_excerpt = node.content      # ← 仅读 md 文件内容

    return block
```

### 7.3 示例

```python
# Agent 调用
block = api.read_memory("ctx://acct-1/users/u1/memories/profile/coding.md", ctx)

# 返回
RetrievedBlock(
    uri="ctx://acct-1/users/u1/memories/profile/coding.md",
    level_hit="L2",
    category="profile",
    content_excerpt="## 编程偏好\n\n### 主要语言\n用户主要使用 Python（5年经验）...",
)
```

---

## 8. Pipeline 容错与降级

> 文件：`retrieval/pipeline.py`

### 8.1 降级策略

| 阶段 | 异常场景 | 降级行为 |
|------|----------|----------|
| Stage 1 | Planner 分类失败 | 生成三路 fallback 查询（MEMORY + SKILL + RESOURCE） |
| Stage 2 | 向量搜索失败 | 返回空 SeedResult |
| Stage 3 | 递归展开失败 | 回退到 `_fallback_l2_from_seed`（直取 SeedResult 中的 L2） |
| Stage 4 | 组装失败 | 返回空列表 |

### 8.2 Stage 3 触发条件

```python
# pipeline.py → _run_expand()
has_dirs = any(v.level < 2 for v in sr.starting_points)

if has_dirs and self.hierarchical_searcher:
    # 有目录级命中 + 有 searcher → 递归展开
    leaves = self.hierarchical_searcher.expand(...)
else:
    # 无目录命中或无 searcher → 直接取 L2
    return self._fallback_l2_from_seed(sr)
```

### 8.3 Trace 记录

每个阶段通过 `TraceTimer` 上下文管理器记录：

```python
with TraceTimer("hierarchical") as st:
    st.input_count = len(sr.starting_points) + len(sr.initial_candidates)
    leaves = self.hierarchical_searcher.expand(...)
    st.output_count = len(leaves)
trace.add_stage(st)
```

输出示例：

```json
{
    "request_id": "a1b2c3d4e5f6",
    "stages": [
        {"stage": "planner",        "elapsed_ms": 1.2,  "input_count": 1, "output_count": 1},
        {"stage": "seed_retrieval", "elapsed_ms": 45.8, "input_count": 1, "output_count": 4},
        {"stage": "hierarchical",   "elapsed_ms": 82.3, "input_count": 3, "output_count": 3},
        {"stage": "assembly",       "elapsed_ms": 0.5,  "input_count": 3, "output_count": 3}
    ],
    "total_ms": 130.1,
    "warnings": []
}
```

---

## 9. 对外 API

> 文件：`service/api.py`

### 9.1 MemoryReadAPI

```python
class MemoryReadAPI:
    def search_memory(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int = 10,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        score_threshold: float | None = None,
        mode: str = RetrieverMode.QUICK,
    ) -> SearchMemoryResult: ...

    def read_memory(
        self,
        uri: str,
        ctx: RequestContext,
    ) -> RetrievedBlock: ...
```

### 9.2 典型调用流程

```python
from core.models import RequestContext

ctx = RequestContext(
    account_id="acct-1",
    user_id="u1",
    agent_id="agent-main",
    session_id="sess-001",
    trace_id="trace-001",
)

# Step 1: 语义搜索
result = api.search_memory("我的编程偏好是什么", ctx, top_k=3)

for block in result.hits:
    print(f"[{block.score:.3f}] {block.uri}")
    print(f"  abstract: {block.abstract}")

# Step 2: Agent 判断需要详细信息，调用 read_memory 读 md 文件内容
detail = api.read_memory(result.hits[0].uri, ctx)
print(detail.content_excerpt)
```

---

## 10. 配置参数

```python
@dataclass
class RetrievalConfig:
    max_top_k: int = 50                    # top_k 上限
    default_top_k: int = 10                # 默认 top_k
    global_search_topk: int = 5            # Stage 2 全局召回数量
    max_convergence_rounds: int = 3        # 收敛检测轮数阈值
    score_propagation_alpha: float = 0.5   # α：child 权重（0.5 表示父子各半）
    hotness_alpha: float = 0.2             # β：hotness 权重（0.2 语义为主）
    hotness_half_life_days: float = 7.0    # hotness 时间衰减半衰期
    default_score_threshold: float = 0.0   # 最低 score 阈值
```

---

## 11. 完整调用链路一览

```
Agent
  │
  ▼
MemoryReadAPI.search_memory(query, ctx, top_k=3)
  │
  ▼
RetrievalPipeline.run()
  │
  ├─ Stage 1: QueryPlanner.plan()
  │    输入: "我的编程偏好是什么"
  │    输出: [TypedQuery(context_type=MEMORY)]
  │
  ├─ Stage 2: SeedRetriever.search()
  │    ├─ Embedder.embed_texts() → query_vector
  │    ├─ VectorIndex.search_by_vector(level=[0,1,2]) → 混合命中
  │    └─ _merge_starting_points()
  │         输出: SeedResult(
  │           starting_points = [profile L0(0.85), events L0(0.42)]
  │           initial_candidates = [coding.md L2(0.91)]
  │         )
  │
  ├─ Stage 3: HierarchicalSearcher.expand()
  │    ├─ 预收集 L2: {coding.md: 0.91}
  │    ├─ Pop profile(0.85) → search_children
  │    │    → coding.md L2(0.88) → final=0.865 < 0.91 → 不替换
  │    │    → editor.md L2(0.72) → final=0.785 → 新增
  │    ├─ Pop events(0.42) → search_children
  │    │    → meeting.md L2(0.38) → final=0.400 → 新增
  │    ├─ 队列空 → top-3
  │    └─ hotness blending → [0.858, 0.688, 0.484]
  │
  └─ Stage 4: AssemblyService.assemble()
       过滤 L2 only → 排序 → 截断 top-3
       输出: [RetrievedBlock, RetrievedBlock, RetrievedBlock]

  ▼
SearchMemoryResult → JSON → Agent
```
