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

| 症状                      | 根因                        |
| ----------------------- | ------------------------- |
| Agent 会遗忘对话早期的内容        | 上下文窗口被占满，旧内容被淘汰，且没有持久化机制  |
| 跨会话重复犯同样的错误             | 缺乏将一次会话中的经验传递到下一次会话的机制    |
| 一次本该花 $0.05 的查询花了 $0.50 | 扁平式检索把整篇文档都加载进来，而实际上摘要就够了 |
| 多 Agent 协作失效            | Agent 之间无法看到彼此的工作上下文      |
| 长会话中上下文不断膨胀             | 没有系统化压缩，Agent 最终淹没在自己的历史里 |

这不是模型的问题。这是**基础设施问题**。

ContextEngine 提供的正是缺失的那层基础设施：一个覆盖 Agent 循环每个阶段的全栈式上下文生命周期管理系统。它把上下文窗口视为一种被管理的资源，而不是一个无限膨胀的缓冲区。

---

## 工作原理：Agent 上下文生命周期

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

ContextEngine 在 Agent 循环的六个阶段进行拦截。每个阶段都会激活一组**上下文算子**，即一类执行上下文读取、写入、变换或淘汰的功能单元。

### 阶段①：message_received — 在推理开始前预取上下文

消息到达，但推理尚未开始。系统需要先判断 Agent 接下来需要哪些上下文。

| 算子        | 作用                                                                                              |
| --------- | ----------------------------------------------------------------------------------------------- |
| **意图分类** | 解析输入消息，识别查询意图（事实回忆 / 技能查找 / 资源引用），决定检索策略和预算优先级                                                  |
| **异步预取**  | 对消息做向量化，异步在 L0 摘要层执行相似度搜索，将候选结果存入 SessionTopicBuffer 的 pending 槽位。目标 P99 小于 20ms。超时则静默失败，不阻塞模型推理 |


### 阶段②：bootstrap + ingest + assemble + before_prompt_build — 组装上下文窗口

成本最高的阶段。在 token 预算内组装最优上下文窗口。

bootstrap（每个会话仅一次）
| 算子        | 作用                                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **冷启动注入** | 在一个会话的第一条消息到来时：加载 Profile、Handoff（若存在活跃任务）以及相关 Preference，通过 L0 召回后按优先级注入 system prompt。采用动态 token 预算，占上下文窗口的 10–15%。群组 / 定时任务 / btw 会话跳过 |
| **会话恢复**  | 通过 `/_session_buffers/` 检测中断的会话，在初始化正常流程前，从 WriteLog 中执行补偿式增量抽取，保证重启后的连续性                                                                 |

ingest / ingestBatch
| 算子        | 作用                                                                                                                               |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **主题跟踪** | 按消息维护 SessionTopicBuffer：更新主题向量，记录已注入 URI 集合（用于去重），计算每个注入项的相关性衰减。若主题连续 5 轮漂移，则将陈旧项移出注入集合。在 collect 模式下，`ingestBatch` 会联合处理所有排队消息 |
| **意图跟踪** | 实时更新意图分类。当意图发生变化时（例如从事实回忆转为多步任务），会部分重置 SessionTopicBuffer，防止旧上下文污染新的意图                                                           |


assemble
| 算子         | 作用                                                                                                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **预算规划**  | 读取 OpenClaw 的 `tokenBudget` 参数，计算可注入预算：`min(fixed_cap, tokenBudget × (1 - (used/window)²))`。同时保留 `reserveTokensFloor`，避免触发 compaction。然后把预算分配到不同的上下文区域                 |
| **相关性排序** | 对候选上下文项进行评分：向量相似度 × 时间新鲜度 × `reference_count` × 关系强度。将排序结果传给 assembler                                                                                                 |
| **上下文组装** | 按预算逐层加载上下文：L0 摘要 → L1 概览 → L2 完整内容（仅在预算允许时）。排除本会话中已注入的 URI。在最后一条用户消息前插入一条合成的 assistant memory message（临时存在，不持久化）。在查询持久索引前，先合并 session-local overlay，保证 read-your-write 一致性 |
| **去重**    | 计算候选项之间 embedding 的余弦相似度。若 cosine > 0.9，则合并，并保留摘要更完整的那条。阻止本会话已见 URI 被再次注入                                                                                              |

before_prompt_build
| 算子       | 作用                                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------- |
| **技能注入** | 基于 routing_key 前缀对 Skill 节点做窄范围点查找，不走向量召回。通过 `prependContext` 将最佳实践和已知失败模式注入到用户消息空间。它与 memory injection 分离：前者面向能力，后者面向个性化 |
| **关系扩展** | 从 L0 seed hits 出发，通过对 `relation_edges` 的一次 JOIN SQL 获取一跳邻居，填充 `RetrievedBlock.relations`，无需逐文件读取                          |

### 阶段③：before_tool_call + tool_result_persist — 工具执行前后的上下文处理

每次 tool call 的前后都会触发上下文处理。

before_tool_call
| 算子          | 作用                                                                                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **工具上下文注入** | 针对当前工具对应的 Skill 节点执行点查找（通过 NodeRegistry，小于 30ms），加载已知参数组合、历史成功/失败模式、使用技巧，并追加到 `systemPromptAddition` 中。同一会话内由 `afterTurn` 写入的内容可立即可见（通过 session-local overlay） |
| **参数推导**  | 从现有上下文中推断合理的参数默认值，例如从 Preference 推断代码风格、从 Entity/Case 推断已知 API 约束，从而减少参数幻觉                                                                                     |
| **失败模式匹配**    | 将当前 tool call 与存储在 Case 上下文中的已知失败模式进行匹配。若命中，则向 `systemPromptAddition` 注入警告。对于高严重度模式，还可触发拦截                                                                     |


tool_result_persist
| 算子        | 作用                                                                                                                                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **结果压缩** | 当工具结果超过 1000 tokens 时，生成轻量摘要，用摘要替换原始 transcript 中的结果，同时将完整内容存储为一个 `tool_snapshot` 节点（TTL 24 小时，URI: `ctx://{account}/agents/{agent}/snapshots/{run_id}_{tool_name}`）。`afterTurn` 会读取 snapshot 做抽取，而不是依赖可能已被裁剪的 transcript |
| **事实抽取** | 立即从工具结果中抽取可复用事实（例如"该接口限流为每分钟 100 次请求"），并以 Pattern/Entity 节点持久化，无需等到 `afterTurn`。同时注册 Outbox 事件以进行异步索引                                                                                                                     |


### 阶段④：afterTurn — 从刚刚发生的内容中抽取上下文

轮次完成。从刚才发生的一切中提取新上下文——主要的写入窗口。

| 算子             | 作用                                                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **增量抽取**      | 基于 LLM 的抽取，使用每个会话的 watermark，只处理自上次检查点以来的增量内容。一次 tool use 调用可同时输出 Profile / Preference / Entity / Event / Procedure 候选。读取 `tool_snapshot` 节点获得完整工具结果。写入前会用高置信度阈值过滤 |
| **关系构建**      | 在新旧节点之间建立边。新 Event 指向其 Entity（`related_to`）；同一会话中连续的 Events 之间建立 `SEQUENCE` 边（通过 NodeRegistry 查询前驱，无需额外 LLM 调用）；AggregateTopicPolicy 合并时，也会关联到已有 entity            |
| **冲突解决**      | 检测抽取结果与现有上下文之间的矛盾，并按类型解决：Profile 以新值为准，Preference 合并，Event 永远追加。冲突会写入审计日志                                                                                          |
| **结构校验**      | 强制执行结构不变量：URI 格式、必填字段、以及每个 IndexRecord filter 中都必须存在 `account_id + owner_space`。违反时阻止写入                                                                            |
| **异步索引构建** | 通过 Outbox 模式调度异步索引（embed → upsert）。执行 `INSERT outbox_events` 并结合 `FOR UPDATE SKIP LOCKED` 支持多 worker 并发消费；`pg_notify` 在插入时立即唤醒 worker。这样可将写入延迟与 Agent 响应时间解耦       |
| **租户隔离**       | 校验所有输出写入都携带正确的 `account_id + owner_space`。任何可能跨账户泄露的数据写入都会被阻止。所有列表操作也会经过 `_is_accessible` 过滤                                                                       |

### 阶段⑤：before_compaction + compact — 优化上下文

上下文窗口接近容量上限。在保留信息密度的同时削减 token 用量。

before_compaction
| 算子        | 作用                                                                                                                      |
| --------- | ----------------------------------------------------------------------------------------------------------------------- |
| **信号打分** | 给每个上下文项打分：时间新鲜度 × `reference_count` × 关系节点度数 × 用户显式引用，并把分数写入 `.meta.json`。这是工具结果仍完整存在于 transcript 中的最后一个窗口              |
| **淘汰保护**  | 将以下内容标记为不可淘汰：Profile、活跃 Handoff、当前正在被引用的 Entity。同时把 SessionWriteLog 传给压缩 LLM，并说明"以下内容已经持久化到长期记忆中，不要再次总结"。目标是最大化压缩后的信息密度 |

compact
| 算子        | 作用                                                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **冗余合并器** | 识别近重复项（cosine > 0.9 或编辑距离 < 2），并按类型策略合并：Preference/Entity 使用 AggregateTopicPolicy，Profile 使用 merge                                         |
| **摘要链**   | 将 L2 完整内容压缩成 L1 概览，再压缩成 L0 摘要。用生成的摘要替换详细条目。受保护节点不参与压缩。当调用 `/compact {instructions}` 时，会先做定向的记忆整合（例如把"决策"整理进 Case，把"未决问题"整理进 Handoff），再执行压缩 |
| **簇压缩器**  | *(Phase 3)* 基于 `relation_edges` 的图遍历聚类。把一簇相关项替换为一个聚合摘要，例如把 5 个相关 Event 压缩成 1 个 Pattern。依赖 GaussVector 的 AI_aggregation                        |

### 阶段⑥：session_end + dispose — 持久化、归档、审计

会话结束。持久化、归档、审计。

session_end
| 算子        | 作用                                                                                                                                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **任务归档** | 将已完成任务归档为不可变的 Case 条目，并保留完整执行轨迹。未来会话可将其作为操作手册引用                                                                                                                                                                   |
| **任务快照** | 将当前任务状态快照（目标 / 已完成 / 未完成 / 下一步）写入 `ctx://{account}/agents/{agent}/memories/handoff/current`。这是固定 URI，每次会话结束都会覆盖；当任务被标记完成时，会归档到 Case。子 Agent 的 Pattern / Case / Skill 节点会提升到父 Agent 命名空间；Event / Preference 节点则被丢弃 |
| **状态版本控制** | 对工作状态做 checkpoint，包括活跃任务、进行中的关系、未提交写入，以支持无缝会话恢复                                                                                                                                                                   |
| **完整性审计** | 扫描本会话写入的所有上下文节点，检查 `.meta.json` 状态和文件完整性，并把 PENDING / BROKEN 节点加入 Repair Job 队列，审计结果写入日志                                                                                                                          |

dispose
| 算子        | 作用                                                                                                                                                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **缓冲区刷新** | 在网关关闭或插件重载时，将 SessionWriteLog 持久化到 `/_session_buffers/{session_id}/write_log.json`。SessionTopicBuffer 会被丢弃，因为它只属于会话期的瞬态状态。下次 bootstrap 时若检测到该文件，会先做补偿式抽取，再进入正常初始化。孤儿 buffer 文件由 Repair Job 清理 |

---

## 抽象层：ContextFS

上下文算子通过单一接口工作——**ContextFS**，一个虚拟文件系统，将每个上下文操作映射为对 `ctx://` URI 的文件操作。

```
    上下文算子（见上方生命周期表）
         │
         │  open · read · write · seek · link · unlink · gc
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │                    ContextFS (ctx://)                       │
  │                                                            │
  │  "一切皆文件" —— Unix 哲学应用于                             │
  │   Agent 上下文管理。                                        │
  │                                                            │
  │  ctx://{account}/users/{user}/memories/profile              │
  │  ctx://{account}/agents/{agent}/skills/sql_gen              │
  │  ctx://{account}/users/{user}/memories/events/20250315_xxx  │
  │                                                            │
  ├──────────┬───────────┬─────────────┬───────────────────────┤
  │ 语义层    │  关系层    │   内容层     │   元数据层             │
  │ Semantic │  Relation │  Content    │  Metadata              │
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
  │ Weaviate │（Phase 2）│ （Phase 2）  │                         │
  └──────────┴───────────┴─────────────┴───────────────────────┘
```

为什么是文件系统？

| 文件系统概念 | 上下文管理对应 | 收益 |
|------------|--------------|------|
| `open(path)` | 按 URI 加载上下文 | 统一访问任何上下文类型 |
| `read(fd, offset, len)` | 读取选定深度的 L0 / L1 / L2 | 只为你需要的 token 付费 |
| `write(fd, data)` | 持久化新上下文 | 带顺序保证的原子写入 |
| `link(src, dst)` | 在上下文之间创建关系边 | 图遍历无需额外 API |
| `unlink(path)` | 删除 / 归档上下文 | 过时上下文的 GC |
| `gc()` | Repair Job——回收损坏/不完整的节点 | 自修复持久化层 |
| 权限 (chmod) | 租户隔离（account_id + owner_space） | 在文件系统层面强制，不由调用方负责 |

L0 → L1 → L2 层次映射 CPU 缓存模型：

```
  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
  │  L0 摘要     │    │  L1 概述     │    │  L2 完整内容  │
  │  ~100 token  │    │  ~500 token  │    │  ~5000 token  │
  │  廉价扫描    │    │  决策支撑    │    │  完整细节     │
  │  始终在预算内 │    │  有据可依    │    │  仅按需加载   │
  └──────┬───────┘    └──────┬───────┘    └──────┬────────┘
         │                   │                    │
         ▼                   ▼                    ▼
    .abstract.md        .overview.md          content.md
```

**Token 节省**：从 1000 条语料中找到 3 条相关内容。平坦 RAG 加载 3 × 5000 = 15K token。ContextEngine 加载 1000 × 100（L0 扫描）+ 3 × 500（L1 决策）+ 1 × 5000（L2 精读）= 7.5K token。**保守估计节省 50%；实践中可达 60–80%。**

---

## 上下文类型

七类上下文，各有不同的生命周期和写入策略。

| 类型 | 生命周期 | URI 模式 | 写入策略 | 示例 |
|------|---------|---------|---------|------|
| **Profile** | 永久——每个用户/Agent 一个，持续精炼 | `.../memories/profile` | Merge（冲突时新覆盖旧） | "张三，后端工程师，住在北京" |
| **Preference** | 长期——按主题累积 | `.../memories/preferences/{slug}` | 按 slug 归并 | "编码风格偏好函数式" |
| **Entity** | 长期——按实体累积 | `.../memories/entities/{slug}` | 按 slug 归并 | "项目 Alpha，2024 年启动" |
| **Event** | 不可变——仅追加的时间线 | `.../memories/events/{event_id}` | 追加，永不覆写 | "2025-03-15 完成 API 迁移" |
| **Case** | 不可变——完整的问题解决轨迹 | `.../memories/cases/{case_id}` | 追加，永不覆写 | "排查 Redis 连接超时的方法" |
| **Pattern** | 演化——从反复观察中涌现 | `.../memories/patterns/{slug}` | 按 slug 归并 | "用户倾向于在下午提问" |
| **Skill** | 增长——累积的工具专长 | `.../skills/{skill_name}` | 追加到已有内容 | "SQL 生成器的最佳参数组合" |

七类共享 `ctx://` URI 命名空间、原子写入顺序（content → relations → abstract → overview → meta）和文件系统级的多租户隔离。

---

## 架构

```
         CLI Agent（Claude Code / OpenClaw / SDK）
              │
              │  HTTP REST（端口 8090）或 Python SDK
              ▼
    ┌─ HTTP 层 (server/) ────────────────────────┐
    │  Flask REST API  ·  认证 / RBAC              │
    │  会话管理  ·  管理 API                        │
    └───────────────┬────────────────────────────┘
                    │
    ┌─ Service 层 (service/) ────────────────────┐
    │  MemoryWriteAPI   对话 → 抽取 → 写入         │
    │  MemoryReadAPI    查询 → 检索 → 上下文组装     │
    └───────────────┬────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │  写入链路              │    检索链路 (retrieval/)
        │  (extraction/ +       │    1. IntentClassifier
        │   commit/)            │    2. SeedRetriever (L0/L1/L2)
        │  1. 两阶段抽取         │    3. ResultRanker
        │  2. PolicyRouter      │    4. ContextReader
        │  3. MergePolicy       │
        │  4. ContextWriter     │
        └───────────┬───────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   ContextFS (fs/) — 虚拟文件系统             │
    │   语义 · 关系 · 内容 · 元数据                │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   会话 (session/) · 生命周期 (lifecycle/)     │
    │   TopicBuffer · RollingCompressor            │
    │   AgingJob · AccessTracker                   │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   异步索引层 (index/)                        │
    │   OutboxWorker  ·  RepairJob                │
    └───────────────┬────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  存储后端            │
          │  AGFS · 向量数据库  │
          │  图数据库 · 对象存储 │
          └────────────────────┘
```

**开发模式**：InMemory 向量 + AGFS localfs，零外部依赖，`pip install -e .` 即可。
**生产模式**：pgvector / Qdrant + AGFS s3fs，水平可扩展，close-to-open 一致性。

---

## 快速开始

### 前置条件

- **Python 3.11+**
- **Go 1.26+**（编译 AGFS）
- **Docker**（可选，用于容器化部署）

### 第 1 步：安装并启动 AGFS

<details>
<summary><strong>方式 1：官方脚本（推荐）</strong></summary>

```bash
curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh
```

</details>

<details>
<summary><strong>方式 2：从源码编译</strong></summary>

```bash
# 如果 Go 下载依赖超时，先设置代理
go env -w GOPROXY=https://goproxy.cn,direct

git clone https://github.com/c4pt0r/agfs.git /tmp/agfs-build
cd /tmp/agfs-build/agfs-server
go build -o ~/.local/bin/agfs-server .
```

</details>

创建配置并启动：

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

验证 AGFS 已启动：

```bash
curl http://localhost:1833/api/v1/health
# {"status":"healthy",...}
```

### 第 2 步：安装 ContextEngine

```bash
git clone https://gitcode.com/opengauss/oG-Memory.git
cd oG-Memory

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
# 或使用 uv：uv pip install -e .
```

### 第 3 步：使用

<details>
<summary><strong>方式 A：HTTP 服务（推荐）</strong></summary>

```bash
# 配置运行时（编辑 ogmem.yaml 设置 LLM 提供商等）
cp ogmem.yaml.example ogmem.yaml

# 启动 HTTP 服务（默认端口 8090）
python server/app.py

# 写入一轮对话
curl -X POST http://localhost:8090/api/v1/after_turn \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user-1",
    "sessionId": "session-1",
    "messages": [
      {"role": "user", "content": "我叫张三，住在北京，是一名后端工程师"},
      {"role": "assistant", "content": "你好张三！很高兴认识你。"}
    ]
  }'

# 搜索记忆
curl -X POST http://localhost:8090/api/v1/compose \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user-1",
    "sessionId": "session-2",
    "query": "张三是做什么工作的"
  }'

# 健康检查
curl http://localhost:8090/api/v1/health
```

</details>

<details>
<summary><strong>方式 B：Docker</strong></summary>

```bash
docker compose up
# 服务端口 8090，AGFS 端口 1833
```

</details>

<details>
<summary><strong>方式 C：Python SDK</strong></summary>

```python
from service.api import MemoryWriteAPI
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.config import ProviderConfig

# 1. 接入存储
config = ProviderConfig.from_env()
agfs = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=agfs, mount_prefix="/local")

# 2. 创建写入 API
write_api = MemoryWriteAPI(fs=fs, llm=config.create_llm())

# 3. 定义上下文
ctx = RequestContext(
    account_id="my-account",
    user_id="user-123",
    agent_id="agent-001",
    session_id="session-001",
    trace_id="trace-001",
)

# 4. 从对话轮次写入上下文
messages = [
    {"role": "user", "content": "我叫张三，住在北京，是一名后端工程师"},
    {"role": "assistant", "content": "你好张三！很高兴认识你。"},
]
result = write_api.commit_session(messages, ctx)
print(f"持久化 {result['writes_completed']} 条上下文")

# 5. 回读上下文（完整管线见 docs/QUICKSTART.md）
```

</details>

**HTTP API 参考**（主要端点）：

| 端点 | 方法 | 说明 |
| ---- | ---- | ---- |
| `/api/v1/compose` | POST | 搜索记忆，返回当前轮次的上下文 |
| `/api/v1/after_turn` | POST | 从对话中抽取并持久化记忆 |
| `/api/v1/ingest` | POST | 单条消息写入 |
| `/api/v1/ingest_batch` | POST | 批量消息写入 |
| `/api/v1/compact` | POST | 触发上下文压缩 |
| `/api/v1/bootstrap` | POST | 冷启动会话（加载 profile + preferences） |
| `/api/v1/sessions/{id}/messages` | POST | 向会话缓冲区添加消息 |
| `/api/v1/sessions/{id}/commit` | POST | 提交会话：归档 + 抽取 |
| `/api/v1/sessions/{id}` | GET | 获取会话状态 |
| `/api/v1/sessions/{id}/context` | GET | 获取组装后的会话上下文 |
| `/api/v1/health` | GET | 健康检查（AGFS + LLM + 向量库） |
| `/api/v1/admin/*` | Various | 租户管理（账户、用户、角色、Agent、审计） |

---

## 与 OpenClaw 集成

OpenClaw 的即插即用上下文引擎：

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

**Agent 工具**（映射到生命周期阶段）：

| 工具 | 生命周期阶段 | 上下文操作 |
|------|------------|----------|
| `og_memory_write` | ④ 轮次结束 | 提取 + 持久化新上下文 |
| `og_memory_search` | ① 消息到达 | 预取相关上下文 |
| `og_memory_read` | ② 推理准备 | 按 URI 加载完整上下文 |

**自动行为**（无需修改 Agent 代码）：

| 生命周期事件 | Hook | ContextEngine 行为 |
|-------------|------|-------------------|
| 新消息到达 | `on_message` | 解析意图 → 预取 → 注入上下文 |
| Agent 完成轮次 | `on_turn_end` | 提取增量 → 解决冲突 → 持久化并索引 |
| 上下文窗口满 | `on_context_compress` | 评分显著度 → 保护关键内容 → 压缩其余 |
| 会话结束 | `on_session_close` | 归档任务 → 检查点状态 → 审计完整性 |

---

## 路线图

### Phase 0 — 核心冻结（已完成）

- Schema、核心模型、Protocol 接口
- ContextFS / AGFS 适配基础能力
- 原子写入顺序 + contract tests

### Phase 1 — 抽取 + 向量检索（基本完成）

**覆盖生命周期：** ① message_received → ② bootstrap + ingest + assemble → ④ afterTurn

- **两阶段抽取**：Phase 1 识别可抽取的消息片段（轻量 JSON-mode LLM）；Phase 2 对每个片段运行聚焦的工具调用抽取，附带 +/- 3 轮上下文，跨片段并行化
- **YAML 驱动的 SchemaRegistry**：从 YAML schema 动态生成工具定义，替代硬编码；`PolicyRouter` 实现 schema 驱动的策略路由，`URIResolver` 统一 URI 生成
- **双抽取模式**：*Eager* 模式（通过向量搜索预取已有记忆，然后单次抽取）；*Lazy* 模式（ReAct 循环带 read/list 工具做多轮抽取，失败时回退到 eager）
- **写入链路**：MergePolicy、原子写入（content → relations → abstract → overview → meta）、Outbox 异步索引
- **分层检索**：IntentClassifier → SeedRetriever（L0/L1/L2 向量召回）→ ResultRanker（相关性评分）→ ContextReader（完整内容加载）
- **Service 层**：MemoryWriteAPI、MemoryReadAPI，强制多租户隔离
- **HTTP REST API**：Flask 服务，完整会话管理、多租户 RBAC（基于角色的访问控制）、租户/用户/Agent 管理端点
- **会话管理**：TopicBuffer、RollingCompressor、SessionManager — 有状态会话生命周期，消息缓冲与提交
- **生命周期管理**：AgingJob 记忆生命周期管理，AccessTracker 使用指标追踪
- **基准测试**：LoCoMo 长上下文对话记忆基准 — 9 段对话 1343 道题准确率 78.1%（4 类：单轮事实、跨会话时序、多跳推理、偏好）

解锁算子：意图分类、异步预取、冷启动加载器、预算规划、相关性排序、上下文组装、去重、两阶段抽取器、关系构建、冲突检测、索引调度、租户隔离、完整性审计、Schema 路由、URI 解析、会话管理、滚动压缩器

### Phase 2 — 混合检索 + 压缩

**覆盖生命周期：** ③ before_tool_call + tool_result_persist → ⑤ compression → ⑥ session_end

- **检索增强**：QueryPlanner 动态分层检索，L0/L1/L2 预算裁剪策略显式化
- **图检索**：RelationStore 一跳/多跳展开，working set 与 graph traversal 融合
- **工具上下文**：before_tool_call 知识注入 + tool_result_persist 结果提炼
- **压缩管线**：before_compaction 决策（保护集/可压缩集）+ compact 执行（摘要替换、archive 生成）
- **会话恢复**：session_end 归档 + dispose 缓冲区刷新 + bootstrap 断点重建
- **AI_Functions 接入**：ai_chunking / ai_filter（GaussVector）

解锁算子：会话恢复器、技能注入、关系扩展、工具上下文注入、参数推导、失败模式匹配、结果压缩、事实抽取、信号打分、淘汰保护、冗余合并器、摘要链、任务归档、任务快照、状态版本控制、缓冲区刷新

### Phase 3 — 聚合 + 自适应规划

**覆盖生命周期：** 全阶段自适应循环

- **图聚类**：基于 relation_edges 的聚类压缩（如 5 个相关 Event → 1 个 Pattern）
- **AI_aggregation**：GaussVector 簇级聚合摘要
- **动态检索规划**：根据任务复杂度自适应调整检索策略
- **子 Agent 协调**：prepareSubagentSpawn / onSubagentEnded 语义落地
- **可观测性**：上下文生命周期 tracing / metrics / audit，量化上下文成本、召回收益、压缩收益

解锁算子：簇压缩器、检索规划器

## 测试

```bash
pytest tests/contract/ -v          # 必须通过——核心不变量
pytest tests/ -v                   # 全部测试
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

## 基准测试：LoCoMo

ContextEngine 使用 [LoCoMo](https://arxiv.org/abs/2402.10790) 长上下文对话记忆基准进行评测 — 10 段对话，约 1500 道题，4 个类别：

| 类别 | 说明 | 示例 |
| ---- | ---- | ---- |
| cat1 | 单轮事实回忆 | "张三是做什么工作的？" |
| cat2 | 跨会话时序推理 | "张三什么时候搬到的北京？" |
| cat3 | 跨主题多跳推理 | "哪个项目先启动的？" |
| cat4 | 用户偏好和习惯 | "张三喜欢什么编码风格？" |

**结果：1343 道题准确率 78.1%**（gpt-4o-mini 评判，与 claude-sonnet-4 在 100 题样本上 100% 一致）。

评测管线：

```bash
cd tests/e2e

# 1. 写入对话（构建记忆）
uv run python eval.py ingest data/locomo10.json --sample 0

# 2. 对存储的记忆运行 QA
uv run python eval.py qa data/locomo10.json --sample 0

# 3. 使用 LLM 评判答案
uv run python eval.py judge data/locomo10.json

# 4. 生成统计报告
uv run python eval.py stat data/locomo10.json
```

## 文档

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 完整技术规范（架构、接口、数据模型、团队协作协议） |
| [ENV.md](ENV.md) | 环境配置（AGFS + Python + 环境变量） |
| [ogmem.yaml](ogmem.yaml) | 运行时配置（LLM 提供商、端口、存储） |
| [快速开始](docs/QUICKSTART.md) | 端到端使用指南 |
| [部署指南](docs/deployment/deployment_operations_guide.md) | 生产环境部署 |
| [OpenClaw 插件](openclaw_context_engine_plugin/README.md) | 集成搭建指南 |
| [基准测试](tests/e2e/) | LoCoMo 评测框架（ingest / qa / judge / stat） |

## 参考文献

### 基准测试与评估
- [LoCoMo: Evaluating Long-Context Conversational Memory](https://arxiv.org/abs/2402.10790) — Maharana et al., 2024 — 长上下文对话记忆基准，用于我们的评测

### LLM Agent 记忆架构
- [MemoryBank: Enhancing LLMs with Long-Term Memory](https://arxiv.org/abs/2305.10250) — Zhong et al., 2023 — 情感驱动的记忆处理机制，用于持久化对话记忆
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) — 2025 — 动态记忆组织与索引
- [Mem0: Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — 2025 — user/session/agent 三级作用域记忆，向量存储 + 知识图谱
- [SeCom: Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/abs/2502.05589) — 2025 — 段级记忆库与对话分割（与我们的两阶段抽取相关）

### 分层检索与多粒度
- [ReadAgent: Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) — Lee et al., 2024 — 受人类阅读启发的 gist 记忆，有效上下文长度提升 20 倍（与我们的 L0/L1/L2 层次相关）
- [MemoRAG: Memory-Augmented Retrieval](https://arxiv.org/abs/2409.05591) — Qian et al., 2024 — 双系统架构：全局记忆 + 检索增强生成
- [LATTICE: LLM-guided Hierarchical Retrieval](https://arxiv.org/abs/2510.13217) — Gupta et al., 2025 — 大规模文档集合的层次化检索框架

### 上下文压缩
- [LLMLingua-2: Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) — Pan et al., 2024 — 基于数据蒸馏的高效 prompt 压缩，最高 20 倍
- [LongLLMLingua: Question-Aware Compression for Long Context](https://aclanthology.org/2024.acl-long.91.pdf) — Jiang et al., 2024, ACL — 粗到细的压缩策略，4 倍压缩下 17.1% 性能提升（与我们的压缩管线设计相关）
- [Prompt Compression for Large Language Models: A Survey](https://arxiv.org/abs/2410.12388) — 2024 — prompt 与上下文压缩技术综述

### 推理与行动
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., 2023, ICLR — 交错推理与任务行动（我们的 lazy 抽取模式基于此范式）

### 多 Agent 与治理
- [Governed Memory](https://arxiv.org/abs/2603.17787) — Taheri, 2026 — 记忆治理与访问控制
- [Collaborative Memory](https://arxiv.org/abs/2505.18279) — 多用户记忆共享 + 动态 ACL（与我们的多租户隔离相关）
- [Multi-Agent Memory Systems for Production](https://mem0.ai/blog/multi-agent-memory-systems) — Mem0, 2026
- [Cemri et al.](https://arxiv.org/abs/2503.13657) — Multi-Agent 协调失败分析

### 基础设施
- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS 文件存储层
- [AI Agent Memory Architectures](https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical) — Zylos Research, 2026

## 致谢

- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS 文件存储层与核心设计理念来源

## 许可证

[Apache License 2.0](LICENSE)
