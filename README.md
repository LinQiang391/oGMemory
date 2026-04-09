# ContextEngine

面向 AI Agent 的长期记忆基础设施。通过结构化存储和分层检索，让 Agent 拥有可维护、可审计、跨会话持久化的记忆能力。

## 核心特性

- **七类记忆抽取**: Profile / Preference / Entity / Event / Case / Pattern / Skill 自动从对话提取
- **三层索引**: L0(摘要) → L1(概述) → L2(完整内容) 分层展开，按需召回
- **多租户隔离**: account_id + owner_space 双层命名空间，租户间完全隔离
- **异步索引**: 写入非阻塞，后台自动同步向量索引，支持重试和 DLQ
- **可靠持久化**: 基于 AGFS 文件存储，支持 localfs/s3fs 后端

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Service Layer (service/)                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ MemoryWriteAPI   │  │ MemoryReadAPI    │  │   MemoryFS       │  │
│  │  commit_session  │  │  search_memory   │  │  文件系统浏览     │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘  │
└───────────┼──────────────────────┼──────────────────────────────────┘
            │                      │
    ┌───────▼──────────┐  ┌────────▼─────────────┐
    │   Write Path     │  │    Read Path         │
    │  (commit/)       │  │   (retrieval/)       │
    ├──────────────────┤  ├─────────────────────┤
    │ 1. Extract       │  │ 1. QueryPlanner      │
    │    (extraction/) │  │ 2. SeedRetriever     │
    │ 2. Plan          │  │ 3. HierarchicalSearch│
    │    (policies)    │  │ 4. AssemblyService   │
    │ 3. Build         │  └─────────────────────┘
    │    (ArchiveBuilder)            │
    │ 4. Write         │              │
    └────┬─────────────┘              │
         │                            │
    ┌────▼────────────────────────────▼─────┐
    │         Storage Layer (fs/)            │
    │  ┌──────────────────────────────────┐  │
    │  │    AGFSContextFS                 │  │
    │  │  (基于 AGFS 的文件存储)            │  │
    │  └──────────────┬───────────────────┘  │
    └─────────────────┼──────────────────────┘
                      │
         ┌────────────▼─────────────┐
         │   AGFS                   │
         │   /accounts/{account}/   │
         │     /users/{user}/       │
         │     /agents/{agent}/     │
         └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│               Async Index Layer (index/)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │OutboxWorker  │  │  Directory   │  │   RepairJob             │   │
│  │  (索引同步)   │  │  Summarizer  │  │   (损坏节点修复)         │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────────┘   │
└─────────┼──────────────────────────────────────────────────────────┘
          │
    ┌─────▼──────────────────────────────┐
    │  VectorIndex (providers/vector_index)│
    │  InMemoryVectorIndex                │
    │  OpenGaussVectorIndex               │
    └──────────────────────────────────────┘
```

## 快速开始

### 1. 安装

```bash
git clone <repository-url>
cd ContextEngine
pip install -e .
```

### 2. 配置 AGFS

AGFS 是 ContextEngine 的文件存储层，需要先启动：

```bash
# 使用官方脚本安装
curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh

# 或从源码编译
git clone https://github.com/c4pt0r/agfs.git /tmp/agfs-build
cd /tmp/agfs-build/agfs-server
go build -o ~/.local/bin/agfs-server .

# 创建配置
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

# 启动
agfs-server -c ~/.agfs-config.yaml &
```

详细配置请参考 [ENV.md](ENV.md)

### 3. 基础使用

```python
from service.api import MemoryWriteAPI, MemoryReadAPI
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.config import ProviderConfig

# 1. 初始化存储
config = ProviderConfig.from_env()
agfs_client = AGFSClient(api_base_url="http://localhost:1833")
fs = AGFSContextFS(client=agfs_client, mount_prefix="/local")

# 2. 初始化 API
llm = config.create_llm()
vector_index = config.create_vector_index()
embedder = config.create_embedder()

from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.assembly_service import AssemblyService
from retrieval.memory_read_service import MemoryReadService

# 初始化写入 API
write_api = MemoryWriteAPI(fs=fs, llm=llm)

# 初始化读取 API
cfg = RetrievalConfig()
pipeline = RetrievalPipeline(
    planner=QueryPlanner(cfg),
    seed_retriever=SeedRetriever(vector_index, embedder, cfg),
    hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
    assembly=AssemblyService(cfg),
    config=cfg,
)
read_api = MemoryReadAPI(
    pipeline=pipeline,
    read_service=MemoryReadService(fs=fs),
    config=cfg,
)

# 3. 创建请求上下文
ctx = RequestContext(
    account_id="my-account",
    user_id="user-123",
    agent_id="agent-001",
    session_id="session-001",
    trace_id="trace-001"
)

# 4. 写入对话记忆
messages = [
    {"role": "user", "content": "我叫张三，住在北京，是一名后端工程师"},
    {"role": "assistant", "content": "你好张三！很高兴认识你。"}
]
result = write_api.commit_session(messages, ctx)
print(f"写入 {result['writes_completed']} 条记忆")

# 5. 检索记忆
search_result = read_api.search_memory(
    query="张三是做什么工作的？",
    ctx=ctx,
    top_k=5
)
for hit in search_result.hits:
    print(f"- [{hit.category}] {hit.abstract}")
```

## 核心概念

### 记忆类型 (Memory Categories)

| 类型 | URI 模式 | 写入策略 | 示例 |
|------|---------|---------|------|
| Profile | `.../memories/profile` | 固定 URI，持续 merge | 用户姓名、职业、所在地 |
| Preference | `.../memories/preferences/{slug}` | 按 slug 归并 | 编码风格、框架偏好 |
| Entity | `.../memories/entities/{slug}` | 按 slug 归并 | 项目名称、同事信息 |
| Event | `.../memories/events/{event_id}` | 追加，不复用 | 会议记录、重要事件 |
| Case | `.../memories/cases/{case_id}` | 追加，不复用 | 问题解决案例 |
| Pattern | `.../memories/patterns/{slug}` | 按 slug 归并 | 常见问题模式 |
| Skill | `.../skills/{skill_name}` | 固定 URI，累积 | 工具使用经验 |

### 三层索引 (3-Level Indexing)

每个记忆节点展开为三条索引记录：

```
ContextNode (ctx://.../profile/)
├── .abstract.md    → IndexRecord(level=0)  L0 首轮召回
├── .overview.md    → IndexRecord(level=1)  L1 决策层
└── content.md      → IndexRecord(level=2)  L2 精读层
```

检索时自动展开：L0 召回相关节点 → L1 过滤 → L2 返回完整内容

### 多租户隔离

```
RequestContext
├── account_id: "my-account"      # 租户 ID (强制注入)
├── owner_space: "user:alice"     # 用户/agent 空间 (colon 格式)
├── user_id: "alice"
├── agent_id: "assistant"
└── session_id: "session-001"

物理路径映射：
ctx://{account}/users/{user}/memories/profile
→ /accounts/{account}/users/{user}/memories/profile/
```

## 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `AGFS_BASE_URL` | ❌ | `http://localhost:1833` | AGFS 服务地址 |
| `OPENAI_API_KEY` | ✅* | - | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | ❌ | `https://api.openai.com/v1` | OpenAI API 地址 |
| `CONTEXTENGINE_PROVIDER` | ❌ | `mock` | LLM 提供商 (mock/openai) |
| `OPENAI_EMBEDDING_MODEL` | ❌ | `text-embedding-ada-002` | 嵌入模型 |
| `OPENAI_LLM_MODEL` | ❌ | `gpt-4o-mini` | LLM 模型 |
| `VECTOR_DB_TYPE` | ❌ | `memory` | 向量数据库 (memory/opengauss) |

\* 使用 mock 提供商时不需要

## 集成

### OpenClaw 集成

```bash
cd openclaw_context_engine_plugin
npm install
```

在 OpenClaw 配置中启用记忆插件：

```json
{
  "memory": {
    "enabled": true,
    "backend": "og-memory"
  }
}
```

### 其他 Agent 框架

参考 `examples/` 目录中的集成示例：
- `basic_usage.py` - 基础 API 使用
- `claude_code_integration.py` - Claude Code 集成

## 测试

```bash
# Contract tests (必须全部通过)
pytest tests/contract/ -v

# 所有测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=core --cov=fs --cov=service --cov-report=html
```

## 文档

- [CLAUDE.md](CLAUDE.md) - 完整技术文档 (架构规范、接口定义)
- [ENV.md](ENV.md) - 环境配置指南
- [docs/deployment.md](docs/deployment.md) - 部署文档

## 项目结构

```
ContextEngine/
├── core/              # 核心模型和接口 (models, interfaces, errors, enums)
├── fs/                # 存储层 (AGFS 适配器)
├── extraction/        # LLM 记忆抽取工具
├── commit/            # 写入链路编排 (policies, writer, outbox)
├── index/             # 异步索引同步 (worker, summarizer, repair)
├── retrieval/         # 检索链路 (planner, retriever, assembly)
├── providers/         # 外部服务适配 (LLM, Embedder, VectorIndex)
├── service/           # 对外 API (MemoryWriteAPI, MemoryReadAPI, MemoryFS)
├── tests/             # 测试 (contract, integration, unit)
├── examples/          # 使用示例
└── docs/              # 文档
```

## License

Apache-2.0
