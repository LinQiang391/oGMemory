<div align="center">

### ContextEngine: CLI Agent 的 <br> 上下文生命周期引擎

贯穿 Agent 循环的每一个阶段，管理什么**进入**上下文窗口、什么**被保留**、什么**被淘汰**、以及什么能够被**重新召回**。
七类上下文。三级缓存层次。一个虚拟文件系统。
降低 token 消耗。保持信号密度。跨越会话边界。

[English](readme.md) | 中文
</div>

---

## 问题：Token 预算是瓶颈

每个 CLI Agent 运行在**固定的 token 预算**上。上下文窗口既是最昂贵的，也是最稀缺的资源。

| 症状 | 根因 | 缺失的生命周期阶段 |
| ---- | ---- | ------------------- |
| Agent 会遗忘对话早期的内容 | 窗口占满后旧内容被淘汰，没有持久化 | ④ **afterTurn** — 缺少抽取 + 持久化 |
| 跨会话重复犯同样的错误 | 没有跨会话传递经验的机制 | ⑥ **session_end** 归档 + ② **bootstrap** 冷启动 |
| 一次本该花 $0.05 的查询花了 $0.50 | 扁平检索加载整篇文档，摘要就够了 | ② **assemble** — 缺少分层检索 |
| 多 Agent 协作失效 | Agent 之间看不到彼此的工作上下文 | ④ **afterTurn** 跨会话共享 + 多租户隔离 |
| 长会话中上下文不断膨胀 | 没有系统化压缩 | ⑤ **compact** — 缺少信号评分 + 摘要链 |

这不是模型的问题。这是**基础设施问题**。ContextEngine 提供的正是缺失的那层基础设施。

---

## 设计哲学

### 核心洞察：上下文有生命周期

当前 RAG 系统把检索当作单一操作 — 向量化查询、搜索向量库、返回结果。这忽略了一个基本事实：**Agent 系统中的上下文有完整的生命周期**，就像数据库中的数据一样。

```
  诞生           结构化            存储              索引             召回            压缩             归档
  (从对话中      (按类型分类,      (原子写入,        (向量化 +        (向量搜索,      (摘要化,         (会话结束,
   抽取)          按策略路由)      有顺序保证)       写入 L0/L1/L2    分层展开,       去重,            归档,
                                                  IndexRecords)    按预算加载)     压缩)            状态快照)
```

每个阶段有不同的约束。抽取必须增量。存储必须原子。索引必须异步。检索必须感知预算。压缩必须保留信号。只处理其中一两个阶段的系统把其余阶段留给了偶然。ContextEngine 覆盖完整生命周期。

### 六个拦截点

Agent 循环不是黑盒。它有明确的执行阶段。关键设计选择：**在循环边界拦截，而不是在模型推理内部拦截。** 所有上下文操作都发生在推理前后，绝不在推理过程中 — 零延迟影响。

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Agent Loop（无限循环）                          │
  │                                                                       │
  │   ┌─────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐         │
  │   │ ①   │     │    ②     │     │    ③     │     │   ④     │         │
  │   │ 消息 │────▶│ 推理准备 │────▶│ 工具调用 │────▶│ 轮次结束 │         │
  │   │ 到达 │     │          │     │          │     │         │         │
  │   └──┬──┘     └──────────┘     └──────────┘     └────┬────┘         │
  │      ▲                                              │               │
  │      │         ┌──────────┐         ┌─────────┐     │               │
  │      │         │    ⑤     │         │   ⑥     │     │               │
  │      └─────────│ 压缩管理 │◀────────│ 会话关闭 │◀────┘               │
  │                └──────────┘         └─────────┘                      │
  │                                                                       │
  └──────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 时机 | 做什么 |
| ---- | ---- | ------ |
| ① 消息到达 | Agent 推理前 | 解析意图，预取候选上下文 |
| ② 推理准备 | 组装上下文窗口 | 冷启动注入 / 主题跟踪 / 预算规划 / 分层加载 / 去重 |
| ③ 工具调用 | 工具执行前后 | 注入工具技能 / 参数推导 / 结果压缩 / 事实抽取 |
| ④ 轮次结束 | Agent 完成一轮后 | 增量抽取 / 关系构建 / 冲突解决 / 异步索引 |
| ⑤ 压缩管理 | 上下文窗口接近满时 | 信号打分 / 淘汰保护 / 冗余合并 / 摘要链 |
| ⑥ 会话关闭 | 会话结束 | 任务归档 / 状态快照 / 完整性审计 |

### 上下文类型不平等

七类上下文，各有根本不同的生命周期行为。这不是随意分类 — 由信息本身的语义决定。单一的"用同一种方式存储一切"方案，要么丢失可变状态（如果仅追加），要么破坏不可变历史（如果覆盖写入）。

| 类型 | 为什么是这个生命周期？ | 写入策略 | URI 模式 |
| ---- | --------------------- | -------- | -------- |
| **Profile** | 用户状态会变化 — "我住在北京"可能变成"我搬到了东京" | **Merge** — 冲突时新覆盖旧 | `.../memories/profile` |
| **Preference** | 偏好按主题累积，每个主题只有一个当前视图 | **按 slug 归并** | `.../memories/preferences/{slug}` |
| **Entity** | 实体累积事实，但"项目 Alpha"仍然是"项目 Alpha" | **按 slug 归并** | `.../memories/entities/{slug}` |
| **Event** | 历史不可变 — "3月15日完成迁移"永远不会改变 | **仅追加** | `.../memories/events/{event_id}` |
| **Case** | 问题解决轨迹是历史记录 | **仅追加** | `.../memories/cases/{case_id}` |
| **Pattern** | 模式从反复观察中涌现并随时间演化 | **按 slug 归并** | `.../memories/patterns/{slug}` |
| **Skill** | 工具专长累积增长 — 经验越多知识越好 | **累积追加** | `.../skills/{skill_name}` |

---

## 关键架构决策

### 为什么用 YAML 驱动 Schema？

**问题**：在 Python 中硬编码抽取 Schema，每新增一个上下文类型都要改代码。每次改动都要动抽取管线、策略路由、URI 解析三处。

**方案**：YAML Schema 声明式定义上下文类型。SchemaRegistry 在启动时加载所有 YAML 文件，PolicyRouter 和 URIResolver 根据注册信息自动适配。

**不用这个方案会怎样**：每次新增类型（比如"Decision"或"Handoff"）都是一次跨三个模块的代码变更，引入回归风险。YAML 驱动让新类型变成"加一个配置文件"而不是"改三处代码"。

### 为什么用 Outbox 模式做异步索引？

**问题**：向量化 + 写入向量索引需要 100-500ms。如果同步执行，每次 `afterTurn` 都会阻塞，增加 Agent 响应延迟。

**方案**：写入完成后投递一个 OutboxEvent，后台 Worker 消费事件、执行 embed + upsert。提供至少一次投递保证，超过最大重试次数移入死信队列。Worker 支持 `FOR UPDATE SKIP LOCKED` 多进程并发消费。

**不用这个方案会怎样**：同步索引意味着 Agent 每轮等待额外 100-500ms。在交互式场景中这是不可接受的。或者干脆不做索引，那就没有后续的语义检索能力。

### 为什么 L0/L1/L2 三级检索？

**问题**：内容越长，向量相似度反而可能越低 — 而非越高。一个 5000 token 的文档 embedding 必须表示其中**每一个**概念，导致任何单一主题的信号被稀释。查询"Alice 是做什么的？"时，一篇塞满 Go、Kubernetes、PostgreSQL、迁移策略细节的 L2 内容产生的向量是弥散的，"后端工程师"只是几十个信号中的一个。而一句聚焦的 L0 摘要（"Alice 是后端工程师"）产生的向量集中精准，直接匹配查询。

**方案**：每个上下文节点按三种粒度索引。L0 摘要（~100 token）产生聚焦向量 — 精准的主题路标。L1 概述（~500 token）信号居中。L2 完整内容（~5000 token）信息最全但向量弥散。检索引擎对三个层级做一次统一的向量搜索，然后把 L0/L1 命中作为**目录入口点**，递归展开树结构：当 L0 摘要匹配时，搜索器展开其子节点，发现扁平搜索会遗漏的 L2 内容。分数传播（`final = α·child + (1-α)·parent`）让强匹配父节点下的边缘 L2 结果获得加成。

```
  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
  │  L0 摘要     │    │  L1 概述     │    │  L2 完整内容  │
  │  ~100 token  │    │  ~500 token  │    │  ~5000 token  │
  │  聚焦向量    │    │  平衡信号    │    │  信息全面     │
  │  → 路标定位  │    │  → 决策参考  │    │  但向量弥散   │
  └──────┬───────┘    └──────┬───────┘    └──────┬────────┘
         │                   │                    │
         ▼                   ▼                    ▼
    .abstract.md        .overview.md          content.md
```

**召回优势**：仅对 L2 做扁平向量搜索，会遗漏那些因细节稀释导致向量分数低于阈值的相关内容。L0/L1 路标引导搜索器定位到正确的目录，然后树展开发现其下的完整内容 — 包括原始向量分数不高但父主题强匹配的 chunk。

### 为什么用文件系统抽象？

**问题**：上下文操作需要原子性、并发控制、权限隔离、垃圾回收。每一样都是独立的工程挑战。

**方案**：复用文件系统数十年的成熟方案。`open/read/write/link/unlink/gc` 直接映射到上下文操作。多租户隔离在文件系统层面通过路径中的 `account_id + owner_space` 强制执行 — 即使调用方有 bug 也无法绕过。

**不用这个方案会怎样**：要么为每项能力自研解决方案（原子写入、并发锁、权限检查），要么放弃这些能力。文件系统隐喻让我们站在巨人的肩膀上。

### 为什么用乐观锁做并发写入？

**问题**：Profile 节点会被用户的所有会话同时写入。需要并发控制。

**方案**：乐观锁 — 读取当前 `.meta.json` 版本，仅在版本未变时写入。无竞争时零开销；竞争时优雅失败并重试。无需额外基础设施。

**不用这个方案会怎样**：分布式锁需要协调服务（etcd/ZooKeeper），增加运维复杂度。不加锁则可能出现"后写覆盖"导致数据丢失。乐观锁在常见场景（无竞争）下性能最优，在罕见场景下安全失败。

### 为什么用 ReAct 循环做抽取？

**问题**：如果系统已经知道"Alice 是后端工程师"，再次抽取就是浪费。但不知道已有什么就无法判断。

**方案**：LLM 被赋予一组工具 — `read(uri)`、`list(uri)`、`get_relations(uri)`，以及 `extract_*` 抽取动作 — 然后在循环中执行。每一轮：先*读取*已有记忆节点（Reason），判断哪些信息是新的，再调用对应的 `extract_*` 工具（Act）。如果不确定，就读更多节点继续循环。这是真正的 ReAct：推理和工具调用交替进行，不是单次盲目抽取。

```
  迭代 1:  read(profile_uri) → "Alice, 后端工程师, 伦敦"
            → 无新信息，跳过

  迭代 2:  read(entities/go) → "Go 专家，偏好错误处理模式 X"
            → 新增: "Alice 现在也在用 Rust 做副项目"
            → extract_entity(slug="rust", ...)
```

**不用这个方案会怎样**：盲目的单次抽取要么重复存储已知信息（浪费 token 和存储），要么遗漏隐含的新信息（丢失信号）。ReAct 循环用少量额外成本换取显著更高的抽取质量。

---

## 架构

### 总体架构

```
  CLI Agent ──── HTTP REST / Python SDK ────┐
                                             ▼
  ┌─ HTTP 层 ───── Flask REST API · 认证/RBAC · 会话管理 ─────┐
  └─ Service 层 ── MemoryWriteAPI（写） · MemoryReadAPI（读）───┘
          │                              │
   ┌──────┴──────┐               ┌───────┴──────┐
   │ 写入链路     │               │ 读取链路      │
   │ 抽取→策略路由│               │ 意图→L0/L1/L2│
   │ →原子写入    │               │ →评分→组装    │
   └──────┬──────┘               └───────┬──────┘
          └──────────┬───────────────────┘
                     ▼
  ┌─ ContextFS (fs/) — 虚拟文件系统：语义 · 关系 · 内容 · 元数据 ─┐
  └─ 会话层 — TopicBuffer · RollingCompressor · AgingJob ─────────┘
                     │
  ┌─ 异步索引层 — OutboxWorker · RepairJob ─┐
  └─ 存储后端 — AGFS（文件）│ PostgreSQL · 向量库 · 图数据库 ─┘
```

**AGFS 模式（默认）**：InMemory/ChromaDB 向量 + AGFS localfs，零外部依赖。
**SQL 模式**：PostgreSQL + pgvector，RLS 行级租户隔离，`LISTEN/NOTIFY` 异步索引。

### 写入路径

一条信息从对话到可检索的完整路径：

```
  Agent 轮次结束
     │
     ▼
  ┌──────────────────────────────────────────────────┐
  │  ReAct 抽取循环                                  │
  │                                                  │
  │  LLM 拥有工具: read(uri), list(uri),             │
  │  get_relations(uri), extract_*()                 │
  │                                                  │
  │  迭代 1: read(profile) → "Alice, 工程师"          │
  │  迭代 2: "Alice 现在也在用 Rust"                  │
  │          → extract_entity(slug="rust")            │
  └────────────────────┬─────────────────────────────┘
                       │  CandidateMemory[]
                       ▼
  ┌──────────────────┐     ┌──────────────────┐
  │  PolicyRouter     │     │  ContextWriter    │
  │  根据 category    │────▶│  原子 4 步写入    │
  │  路由到对应策略    │     │  ① content.md    │
  │  Profile→Merge    │     │  ② .relations    │
  │  Entity→Aggregate │     │  ③ abstract+over │
  │  Event→Append     │     │  ④ .meta.json ✱  │
  └──────────────────┘     └────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  OutboxStore       │
                          │  投递 OutboxEvent  │
                          │  异步消费          │
                          └─────────┬─────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Index Worker (后台)           │
                    │  embed(abstract) → L0 upsert  │
                    │  embed(overview)  → L1 upsert  │
                    │  embed(content)   → L2 upsert  │
                    └───────────────────────────────┘
```

### 读取路径

从查询到上下文注入的完整路径：

```
  用户查询
     │
     ▼
  ┌──────────────────┐     ┌──────────────────────────────────────────┐
  │  IntentClassifier │     │  全局向量搜索（所有层级）                   │
  │                   │     │                                          │
  │  关键词规则化分类： │────▶│  向量化查询，一次搜索 L0+L1+L2            │
  │  "偏好/习惯"→MEMORY│    │  → L0/L1 命中 = 目录路标                   │
  │  "如何/步骤"→SKILL │     │  → L2 命中 = 直接内容匹配                  │
  │  "文档/资料"→RES.  │     └────────────────────┬─────────────────────┘
  └──────────────────┘                          │
                                          ┌──────▼──────┐
                                          │  有 L0/L1   │
                                          │  命中？      │
                                          └──────┬──────┘
                                           是   │   否
                                          ┌──────▼──────▼──────┐
                                          │  递归展开           │  直接使用
                                          │                    │  L2 结果
                                          │  search_children() │
                                          │  按目录节点展开     │
                                          │  分数传播           │
                                          │  α·child+(1-α)·   │
                                          │  parent            │
                                          └────────┬───────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  组装              │
                                          │  合并 + 去重        │
                                          │  (cos>0.9)         │
                                          │  按混合分数排序      │
                                          │  （语义 + 热度）     │
                                          │  注入上下文窗口      │
                                          └──────────────────┘
```

一次向量搜索，不是三次逐层扫描。L0/L1 命中作为目录入口点进行树展开 — 发现因长文档 embedding 稀释而扁平搜索会遗漏的 L2 内容。

**命名空间隔离：** 查询按意图类型限定搜索范围。MEMORY 查询同时搜索 `users/{user}/memories/` 和 `agents/{agent}/memories/`。SKILL 查询只搜索 `agents/{agent}/skills/`。`owner_space` 过滤由 QueryPlanner 根据 `context_type` 和 `visible_owner_spaces` 设置，在向量索引层面强制执行 — 调用方无法覆盖。

### ContextFS 虚拟文件系统

所有上下文算子通过单一接口工作。文件操作直接映射到上下文管理：`open` = 按 URI 加载，`read` = 按深度读 L0/L1/L2，`write` = 原子持久化，`link` = 创建关系边，`gc` = Repair Job 自修复。租户隔离在文件系统层面通过路径强制执行。

L0 → L1 → L2：每个层级产生信号浓度不同的 embedding。短文本 → 聚焦向量 → 主题匹配更精准。长文本 → 弥散向量 → 信息更全但语义精度更弱。

---

## 数据流：端到端追踪

一条信息是如何从对话走向检索的？以下是完整路径：

```
  会话 1："我叫 Alice，是一名后端工程师，住在伦敦"
  ──────────────────────────────────────────────────────
                                                                       
    用户消息 ──────────▶ 阶段 ④ afterTurn                            
                              │                                       
                              ▼                                       
                        增量抽取                                     
                        (LLM 识别: name="Alice",                     
                         role="后端工程师",                            
                         location="伦敦")                              
                              │                                       
                              ▼                                       
                    CandidateMemory(category="profile")               
                              │                                       
                              ▼                                       
                    PolicyRouter → ProfilePolicy                      
                    (合并到已有 profile)                                
                              │                                       
                              ▼                                       
                    ContextWriter（原子 4 步写入）                      
                    ① content.md                                      
                    ② .relations.json                                 
                    ③ .abstract.md + .overview.md                     
                    ④ .meta.json (status=ACTIVE) ← 提交点              
                              │                                       
                              ▼                                       
                    OutboxEvent → 异步索引                             
                              │                                       
                              ▼                                       
                    IndexRecordBuilder                                
                    L0: 摘要 → embed → upsert                        
                    L1: 概述 → embed → upsert                         
                    L2: 内容 → embed → upsert                         
                                                                       
                                                                       
  会话 2："Alice 是做什么工作的？"                                    
  ──────────────────────────────────────────────────────
                                                                       
    用户消息 ──────────▶ 阶段 ① message_received                    
                              │                                       
                              ▼                                       
                        意图分类 → "事实回忆"                           
                        异步预取 → 向量化查询                           
                        → L0 向量搜索 → seed hit                      
                        (profile 摘要匹配)                             
                              │                                       
                              ▼                                       
                        阶段 ② assemble                              
                        预算规划 → 分配 token                          
                        相关性排序 → 评分候选                          
                        上下文组装 → L0 hit → L2 加载                  
                        → 注入 "Alice 是一名后端工程师"                 
                              │                                       
                              ▼                                       
                        Agent 响应: "Alice 是一名后端工程师"            
```

从抽取到检索的完整生命周期：索引耗时约 100ms（异步），检索耗时约 50ms（同步）。成本在写入时支付*一次*，在读取时*多次*分摊。

---

## 与典型方案的区别

| 维度 | ContextEngine | 标准向量 RAG | Mem0 |
| ---- | ------------- | ------------ | ---- |
| **生命周期覆盖** | 六阶段全覆盖：抽取 → 结构化 → 存储 → 索引 → 召回 → 压缩 → 归档 | 仅召回阶段 | 抽取 + 召回，无压缩/归档 |
| **写入策略** | 四种策略（Merge / Aggregate / Append / Cumulative），按信息语义选择 | 无写入策略，所有数据同等对待 | 统一 upsert，无策略区分 |
| **检索粒度** | L0/L1 作为目录路标 → 递归树展开 → L2 内容发现，分数传播 | 单层扁平 top-k 相似搜索 | 单层向量检索 + 知识图谱 |
| **上下文类型** | 七类，每类独立生命周期 | 无类型区分 | user/session/agent 三级作用域 |
| **多租户隔离** | 文件系统层面强制（路径中 account_id + owner_space），调用方无法绕过 | 应用层过滤，依赖调用方正确传参 | 应用层 scope 隔离 |
| **并发写入安全** | 乐观锁，无额外基础设施 | 通常无并发控制 | 依赖向量库自身的 upsert 语义 |
| **异步索引** | Outbox 模式，至少一次投递，死信队列 | 通常同步写入 | 同步写入 |

核心区别：ContextEngine 把上下文当作有生命周期的数据来管理，而不是当作一堆需要向量化的文本。

---

## 快速开始

### 前置条件

- **Python 3.11+**
- **Go 1.26+**（编译 AGFS）
- **Docker**（可选，用于容器化部署）

### 安装

```bash
git clone https://gitcode.com/opengauss/oG-Memory.git && cd oG-Memory
python3 -m venv .venv && source .venv/bin/activate && pip install -e .

# 交互式配置向导（引导 AGFS 安装 + LLM + Embedding + 向量库配置）
ogmem onboard

# 一键启动（AGFS + ContextEngine）
ogmem start local
```

<details>
<summary><strong>非交互模式（CI / 自动化）</strong></summary>

```bash
ogmem onboard --non-interactive --mode local \
  --provider openai --api-key sk-xxx \
  --embedding-model text-embedding-ada-002 --vector-db chroma
```

</details>

### 使用

<details>
<summary><strong>HTTP 服务（推荐）</strong></summary>

```bash
ogmem start local    # 启动 AGFS + ContextEngine，默认端口 1833 + 8090

# 写入对话
curl -X POST http://localhost:8090/api/v1/after_turn \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user-1", "sessionId": "session-1",
    "messages": [
      {"role": "user", "content": "我叫 Alice，是一名后端工程师，住在伦敦"},
      {"role": "assistant", "content": "你好 Alice！"}
    ]
  }'

# 搜索记忆
curl -X POST http://localhost:8090/api/v1/compose \
  -H "Content-Type: application/json" \
  -d '{"userId": "user-1", "sessionId": "session-2", "query": "Alice 是做什么工作的"}'
```

</details>

<details>
<summary><strong>Docker</strong></summary>

```bash
docker compose up  # 服务端口 8090，AGFS 端口 1833
```

</details>

<details>
<summary><strong>Python SDK</strong></summary>

```python
from service.api import MemoryWriteAPI
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.config import ProviderConfig

config = ProviderConfig.from_env()
agfs = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=agfs, mount_prefix="/local")
write_api = MemoryWriteAPI(fs=fs, llm=config.create_llm())

ctx = RequestContext(
    account_id="my-account", user_id="user-123",
    agent_id="agent-001", session_id="session-001", trace_id="trace-001",
)

messages = [
    {"role": "user", "content": "我叫 Alice，后端工程师，住在伦敦"},
    {"role": "assistant", "content": "你好 Alice！"},
]
result = write_api.commit_session(messages, ctx)
```

</details>

<details>
<summary><strong>HTTP API 参考</strong></summary>

| 端点 | 方法 | 说明 |
| ---- | ---- | ---- |
| `/api/v1/compose` | POST | 搜索记忆，返回当前轮次上下文 |
| `/api/v1/after_turn` | POST | 从对话中抽取并持久化记忆 |
| `/api/v1/ingest` | POST | 单条消息写入 |
| `/api/v1/ingest_batch` | POST | 批量消息写入 |
| `/api/v1/compact` | POST | 触发上下文压缩 |
| `/api/v1/bootstrap` | POST | 冷启动会话 |
| `/api/v1/sessions/{id}/messages` | POST | 向会话缓冲区添加消息 |
| `/api/v1/sessions/{id}/commit` | POST | 提交会话 |
| `/api/v1/sessions/{id}` | GET | 获取会话状态 |
| `/api/v1/sessions/{id}/context` | GET | 获取组装后的会话上下文 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/admin/*` | Various | 租户管理 |

</details>

### 配置

主要环境变量（完整参考见 [ENV.md](ENV.md)）：

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `OGMEM_API_KEY` | — | LLM API 密钥（OpenAI 兼容） |
| `OGMEM_BASE_URL` | — | 自定义 LLM API 基础 URL |
| `OGMEM_LLM_MODEL` | `gpt-4o-mini` | 抽取 + 分类使用的 LLM 模型 |
| `OGMEM_EMBEDDING_MODEL` | `text-embedding-ada-002` | 向量索引使用的嵌入模型 |
| `OGMEM_EMBEDDING_API_KEY` | — | 嵌入 API 独立密钥（缺省回退 `OGMEM_API_KEY`） |
| `VECTOR_DB_TYPE` | `chroma` | 向量后端：`memory` / `chroma` / `opengauss` |
| `STORAGE_BACKEND` | `agfs` | 存储后端：`agfs`（AGFS 文件系统）或 `sql`（PostgreSQL 直连） |
| `SQL_CONNECTION_STRING` | — | PostgreSQL DSN（仅 `storage_backend=sql` 时需要） |
| `AGFS_BASE_URL` | `http://127.0.0.1:1833` | AGFS 文件服务器地址 |
| `AGFS_MOUNT_PREFIX` | `/local` | AGFS 挂载点前缀 |
| `OGMEM_HTTP_PORT` | `8090` | HTTP 服务监听端口 |
| `OGMEM_CONFIG` | `ogmem.yaml` | YAML 配置文件路径 |

### 仓库结构

```
ContextEngine/
├── core/                   # 领域模型、Protocol 接口、枚举、错误类型
├── fs/agfs_adapter/        # ContextFS → AGFS 适配器（原子写入、访问控制）
├── fs/sql_adapter/         # ContextFS → PostgreSQL 适配器（原子 upsert、RLS 租户隔离）
├── extraction/             # CandidateExtractor（ReAct 循环 + YAML SchemaRegistry）
│   ├── prompts/            #   LLM 提示模板
│   └── schemas/            #   各上下文类型的 YAML 抽取 Schema
├── commit/                 # 写入链路：PolicyRouter → MergePolicy → ContextWriter
├── index/                  # 异步索引：OutboxWorker → IndexRecordBuilder → VectorDB
├── retrieval/              # 读取链路：IntentClassifier → SeedRetriever → ResultRanker
├── providers/              # 外部适配：LLM、Embedder、VectorIndex、RelationStore
│   ├── llm/                #   OpenAI 兼容 LLM 提供商
│   ├── embedder/           #   4 种嵌入后端（OpenAI、火山引擎等）
│   ├── vector_index/       #   InMemory / ChromaDB / pgvector
│   └── relation_store/     #   AGFS + SQL 关系存储
├── service/                # API 层：MemoryWriteAPI、MemoryReadAPI、多租户
├── server/                 # HTTP REST 服务（Flask）、认证/RBAC、会话管理
├── session/                # SessionManager、TopicBuffer、RollingCompressor
├── lifecycle/              # AgingJob、AccessTracker
├── agfs/                   # OpenViking AGFS Go 文件服务器（内嵌）
├── tests/
│   ├── contract/           # 跨团队契约测试（不变量检查）
│   ├── unit/               # 各包单元测试
│   ├── integration/        # 端到端集成测试
│   ├── e2e/                # LoCoMo 基准评测框架
│   └── fixtures/           # 共享测试数据
├── docs/                   # 架构、部署、快速上手指南
├── examples/               # 使用示例（SDK、Claude Code 集成）
├── cli/                    # 统一管理 CLI（ogmem 命令）
├── openclaw_context_engine_plugin/  # OpenClaw 插件（TypeScript 桥接）
└── scripts/                # 辅助脚本
```

---

## OpenClaw 集成

即插即用的上下文引擎：

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
AGFS / PostgreSQL
```

| 工具 | 生命周期阶段 | 上下文操作 |
| ---- | ------------ | ---------- |
| `og_memory_write` | ④ 轮次结束 | 提取 + 持久化新上下文 |
| `og_memory_search` | ① 消息到达 | 预取相关上下文 |
| `og_memory_read` | ② 推理准备 | 按 URI 加载完整上下文 |

---

## 实现状态

**已覆盖的生命周期阶段：** ① message_received、② bootstrap + ingest + assemble、④ afterTurn

**LoCoMo 基准测试：1343 道题准确率 78.1%**（gpt-4o-mini 评判，与 claude-sonnet-4 在 100 题样本上 100% 一致）。评测框架见 `tests/e2e/`。

### 已完成（Phase 0 + 1）

| 组件 | 核心能力 |
| ---- | -------- |
| 核心模型与接口 | 全部数据类、Protocol 接口、枚举、URI 解析器 |
| ContextFS 存储适配器 | AGFS 适配器（4 步原子写入）+ SQL 适配器（upsert、RLS 租户隔离） |
| 抽取管线 | 统一 Extractor、YAML SchemaRegistry、ReAct 循环 |
| 写入链路 | ContextWriter、PolicyRouter、4 种合并策略、OutboxStore |
| 分层检索 | IntentClassifier → SeedRetriever → HierarchicalSearcher → ResultRanker |
| 异步索引 | OutboxWorker、RepairJob、L0/L1/L2 IndexRecord 构建器 |
| 外部能力适配 | OpenAI/火山引擎 LLM、4 种嵌入器、InMemory/ChromaDB/pgvector |
| Service 层 | MemoryWriteAPI、MemoryReadAPI、多租户隔离 |
| HTTP REST API | Flask 服务、认证/RBAC、会话管理 |
| 会话管理 | SessionManager、TopicBuffer、RollingCompressor |
| 生命周期管理 | AgingJob、AccessTracker |
| 测试 | Contract + 单元 + 集成 + 端到端（100+ 测试文件） |

**待覆盖：** ③ before_tool_call + tool_result_persist、⑤ compression、⑥ session_end + dispose（Phase 2/3）

### 文档

| 文档 | 说明 |
| ---- | ---- |
| [CLAUDE.md](CLAUDE.md) | 完整技术规范 |
| [ENV.md](ENV.md) | 环境配置 |
| [快速开始](docs/QUICKSTART.md) | 端到端使用指南 |
| [部署指南](docs/deployment/deployment_operations_guide.md) | 生产环境部署 |
| [OpenClaw 插件](openclaw_context_engine_plugin/README.md) | 集成搭建 |
| [基准测试](tests/e2e/) | LoCoMo 评测框架 |

---

## 参考文献

### 基准测试与评估
- [LoCoMo: Evaluating Long-Context Conversational Memory](https://arxiv.org/abs/2402.10790) — Maharana et al., 2024 — 长上下文对话记忆基准

### LLM Agent 记忆架构
- [MemoryBank: Enhancing LLMs with Long-Term Memory](https://arxiv.org/abs/2305.10250) — Zhong et al., 2023 — 情感驱动的记忆处理机制
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) — 2025 — 动态记忆组织与索引
- [Mem0: Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — 2025 — user/session/agent 三级作用域记忆
- [SeCom: Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/abs/2502.05589) — 2025 — 段级记忆库与对话分割

### 分层检索与多粒度
- [ReadAgent: Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) — Lee et al., 2024 — gist 记忆，有效上下文长度提升 20 倍
- [MemoRAG: Memory-Augmented Retrieval](https://arxiv.org/abs/2409.05591) — Qian et al., 2024 — 双系统架构
- [LATTICE: LLM-guided Hierarchical Retrieval](https://arxiv.org/abs/2510.13217) — Gupta et al., 2025 — 层次化检索框架

### 上下文压缩
- [LLMLingua-2: Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) — Pan et al., 2024 — 高效 prompt 压缩，最高 20 倍
- [LongLLMLingua: Question-Aware Compression for Long Context](https://aclanthology.org/2024.acl-long.91.pdf) — Jiang et al., 2024, ACL — 粗到细压缩策略
- [Prompt Compression for Large Language Models: A Survey](https://arxiv.org/abs/2410.12388) — 2024 — 综述

### 推理与行动
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., 2023, ICLR

### 多 Agent 与治理
- [Governed Memory](https://arxiv.org/abs/2603.17787) — Taheri, 2026 — 记忆治理与访问控制
- [Collaborative Memory](https://arxiv.org/abs/2505.18279) — 多用户记忆共享 + 动态 ACL
- [Multi-Agent Memory Systems for Production](https://mem0.ai/blog/multi-agent-memory-systems) — Mem0, 2026
- [Cemri et al.](https://arxiv.org/abs/2503.13657) — Multi-Agent 协调失败分析

### 基础设施
- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS 文件存储层与核心设计理念来源
- [AI Agent Memory Architectures](https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical) — Zylos Research, 2026

## 许可证

[Apache License 2.0](LICENSE) | 致谢 [OpenViking](https://github.com/volcengine/OpenViking) AGFS 文件存储层
