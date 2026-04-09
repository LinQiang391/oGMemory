<div align="center">

### ContextEngine: CLI Agent 的 <br> 上下文生命周期引擎

管理什么**进入**上下文窗口、什么**留在**其中、什么被**移出**、什么又能**回来**——
贯穿 Agent 循环的每一个阶段。
七类上下文，三级缓存层次，一个虚拟文件系统。

削减 token 成本。保留信号密度。跨越会话边界。

[English](readme.md) | 中文
</div>

---

## 问题：Token 预算是瓶颈

每个 CLI Agent 运行在**固定的 token 预算**上。上下文窗口既是最昂贵的、也是最稀缺的资源。

| 症状 | 根因 |
|------|------|
| Agent 忘记之前的对话 | 上下文窗口填满 → 旧内容被驱逐，且没有持久化 |
| 跨会话重复犯同样的错 | 没有机制将一个会话的教训带到下一个会话 |
| 本应 $0.05 的查询花了 $0.50 | 平坦检索加载整个文档，而一个摘要就够了 |
| 多 Agent 协调失败 | Agent 彼此看不到对方的工作上下文 |
| 长会话后上下文膨胀 | 没有系统性压缩——Agent 被自己的历史淹没 |

这不是模型的问题。这是**基础设施问题**。

ContextEngine 提供了缺失的基础设施：一个全栈上下文生命周期管理系统，在 Agent 循环的每个阶段介入，将上下文窗口视为**受管资源**——而非无界缓冲区。

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

ContextEngine 在六个阶段拦截 Agent 循环。每个阶段的 hook 激活一组**上下文算子**——负责读取、写入、变换或驱逐上下文的功能单元。

### ① `on_message` — 上下文预取

消息到达。推理尚未开始，先判断 Agent 需要什么上下文。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **查询解析器** | 解析 incoming 消息 → 分类意图（事实召回 / 技能查找 / 资料引用）→ 决定检索策略 | 1 |
| **预取引擎** | 将消息向量化，对 L0 摘要层做相似度搜索，将候选上下文预加载到暂存缓冲区 | 1 |
| **冷启动加载器** | 新会话首条消息 → 从持久化存储加载用户 Profile + 近期 Preference + 上次会话摘要 | 1 |
| **会话恢复器** | 恢复中断的会话 → 从持久化的状态检查点重建工作上下文 | 2 |

### ② `on_before_inference` — 上下文组装

在 token 预算内组装最优上下文窗口。这是影响成本和质量最关键的阶段。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **预算规划器** | 按任务类型分配 token 预算：40% 检索知识、30% 对话历史、20% 工具上下文、10% 系统指令 | 2 |
| **相关性排序器** | 多信号综合打分——向量相似度 × 时效性 × 访问频率 × 关系强度 → 排序列表 | 1 |
| **上下文编排器** | 按优先级组装上下文窗口：L0 摘要 → L1 概述 → L2 完整内容（仅在预算允许时加载最深层） | 1 |
| **技能定位器** | 从意图推断可能使用的工具 → 定位并加载相关 Skill 上下文（最佳实践、已知失败模式） | 1 |
| **图遍历器** | 从种子命中出发，沿关系边（`related_to`、`derived_from`、`SEQUENCE`）拉入初始向量召回之外的关联上下文 | 2 |
| **去重器** | 移除重叠的上下文条目；用结构相似度（编辑距离 < 2 或嵌入余弦 > 0.9）归并近似重复 | 1 |

### ③ `on_tool_call` / `on_tool_result` — 上下文增强

单轮对话中每次工具执行前后。注入工具知识，捕获工具揭示的信息。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **工具知识加载器** | 加载当前调用工具的 Skill 上下文——已知参数组合、历史成功/失败率、使用技巧 | 1 |
| **参数上下文化器** | 将上下文推导出的默认值注入工具参数（例如从 Preference 获取偏好的编码风格，从 Case 获取已知的 API 约束） | 1 |
| **安全检查器** | 将当前工具调用与 Case 上下文中存储的已知失败模式匹配 → 命中时警告或拦截 | 1 |
| **结果凝练器** | 工具执行后 → 将输出提炼为 L0 摘要 + 可选 L1 概述供后续检索，完整结果保留在 L2 | 1 |
| **事实提取器** | 从工具结果中捕获可复用的知识（例如"此端点限速 100 req/min"）持久化为 Pattern/Entity 上下文 | 2 |

### ④ `on_turn_end` — 上下文抽取

轮次完成。从刚才发生的一切中提取新上下文——主要的写入窗口。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **增量提取器** | 基于 LLM 的抽取：从刚完成的轮次中提取新的事实、实体、偏好、事件——只取增量，不是全量对话 | 1 |
| **关系构建器** | 在新上下文节点与已有节点之间创建边（将新 Event 链接到其 Entity，新 Preference 链接到 Profile） | 1 |
| **冲突检测器** | 检测提取内容与已有上下文的矛盾 → 按类型策略解决（Profile：新覆盖旧；Preference：归并；Event：始终追加） | 1 |
| **Schema 校验器** | 强制结构不变量：URI 格式、必填字段、每条 filter 中的 `account_id` + `owner_space`——违规则阻止写入 | 1 |
| **索引调度器** | 通过 Outbox 模式将异步索引更新（embed → upsert）排入队列——将写入延迟与 Agent 响应时间解耦 | 1 |
| **租户隔离器** | 校验所有出站上下文携带正确的租户隔离标签；阻止任何跨账户边界泄漏的写入 | 1 |

### ⑤ `on_context_compress` — 上下文优化

上下文窗口接近容量上限。在保留信号的同时削减 token 用量。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **信号评估器** | 对每条上下文条目按信息密度打分：时效性 × 访问频率 × 关系密度 × 用户显式引用 → 显著度分数 | 2 |
| **优先保护器** | 标记关键上下文为不可驱逐：Profile、当前引用的实体、活跃任务的 Case——在任何压缩中存活 | 2 |
| **冗余归并器** | 识别重复或近似相同的条目 → 使用类型对应策略归并（Preference/Entity 用 AggregateTopic，Profile 用 merge） | 2 |
| **摘要引擎** | 压缩 L2 完整内容 → L1 概述 → L0 摘要链；用生成的摘要替换详细条目 | 2 |
| **聚类压缩器** | 通过图遍历关联上下文 → 聚类 → 用聚合后的群组摘要替代个体条目（例如 5 条相关事件 → 1 条模式） | 3 |
| **检索规划器** | 针对复杂任务，动态规划下一周期的检索策略：加载哪些层级、搜索哪些类别、遍历多少跳 | 3 |

### ⑥ `on_session_close` — 上下文持久化

会话结束。持久化、归档、审计。

| 上下文算子 | 职责 | 阶段 |
|----------|------|------|
| **任务终结器** | 将已完成的任务上下文归档为 Case 条目，包含完整的问题解决轨迹——未来会话可复用的操作手册 | 1 |
| **状态持久化器** | 检查点工作状态：活跃任务、进行中的关系、未提交的写入 → 支持无缝的会话恢复 | 1 |
| **会话摘要器** | 生成会话级摘要（话题、决策、使用的技能）→ 索引为 Event 以支持跨会话召回 | 2 |
| **完整性审计器** | 扫描本会话涉及的所有上下文节点：`.meta.json` 状态、文件完整度 → 对 PENDING/BROKEN 节点触发 Repair Job | 1 |
| **Schema 审计器** | 最终校验轮——确保会话期间写入的所有上下文符合冻结的核心 schema；记录任何偏移的告警 | 2 |

---

## 抽象层：ContextFS

27 个上下文算子通过单一接口工作——**ContextFS**，一个虚拟文件系统，将每个上下文操作映射为对 `ctx://` URI 的文件操作。

```
    27 个上下文算子（见上方生命周期表）
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
              │  6 个生命周期 hooks
              ▼
    ┌─ Service Layer (service/) ──────────────────┐
    │  MemoryWriteAPI   对话 → 抽取 → 写入         │
    │  MemoryReadAPI    查询 → 检索 → 上下文组装     │
    │  MemoryFS         ctx:// URI 文件浏览          │
    └───────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │  写入链路 (commit/)    │    检索链路 (retrieval/)
        │  1. Extract (LLM)     │    1. 查询解析
        │  2. MergePolicy       │    2. 预取 + 展开
        │  3. 关系构建           │    3. 分层加载
        │  4. 原子写入           │    4. 上下文组装
        └───────────┬───────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   ContextFS (fs/) — 虚拟文件系统             │
    │   语义 · 关系 · 内容 · 元数据                │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │   异步索引层 (index/)                        │
    │   OutboxWorker  ·  RepairJob                │
    │   （索引同步）      （节点修复）               │
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
```

### 第 3 步：使用

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

| 阶段 | 重点 | 关键交付物 | 解锁上下文算子 |
|------|------|----------|-------------|
| **Phase 0**（已完成） | Core 冻结 | Schema + Protocol + AGFS spec + contract tests | Schema 校验器 |
| **Phase 1** | 抽取 + 向量检索 | 写入链路 + L0/L1/L2 检索 + service 层 | 查询解析器、预取引擎、冷启动加载器、上下文编排器、相关性排序器、增量提取器、关系构建器、冲突检测器、索引调度器、完整性审计器 |
| **Phase 2** | 混合检索 + 压缩 | RelationStore 图遍历 + AI_Functions + 上下文压缩 | 会话恢复器、预算规划器、图遍历器、工具知识加载器、信号评估器、优先保护器、冗余归并器、摘要引擎、会话摘要器、Schema 审计器 |
| **Phase 3** | 聚合 + 自适应规划 | 图聚类 + AI_aggregation + 动态查询规划 | 聚类压缩器、检索规划器、事实提取器 |

## 测试

```bash
pytest tests/contract/ -v          # 必须通过——核心不变量
pytest tests/ -v                   # 全部测试
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

## 文档

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 完整技术规范（架构、接口、数据模型、团队协作协议） |
| [ENV.md](ENV.md) | 环境配置（AGFS + Python + 环境变量） |
| [快速开始](docs/QUICKSTART.md) | 端到端使用指南 |
| [部署指南](docs/deployment/deployment_operations_guide.md) | 生产环境部署 |
| [OpenClaw 插件](openclaw_context_engine_plugin/README.md) | 集成搭建指南 |

## 参考文献

- [AI Agent Memory Architectures](https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical) — Zylos Research, 2026
- [Multi-Agent Memory Systems for Production](https://mem0.ai/blog/multi-agent-memory-systems) — Mem0, 2026
- [Governed Memory](https://arxiv.org/abs/2603.17787) — Taheri, 2026
- [Collaborative Memory](https://arxiv.org/abs/2505.18279) — 多用户记忆共享 + 动态 ACL
- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS 文件存储层
- [Cemri et al.](https://arxiv.org/abs/2503.13657) — Multi-Agent 协调失败分析

## 致谢

- [OpenViking](https://github.com/volcengine/OpenViking) — AGFS 文件存储层与核心设计理念来源

## 许可证

[Apache License 2.0](LICENSE)
