# ContextEngine 快速启动指南

## 一键启动

```bash
# 运行一键启动脚本
./scripts/start_all.sh
```

该脚本会：
1. 检查环境 (Python, Node.js, OpenClaw)
2. 配置环境变量
3. 安装 Python 依赖
4. 启动 AGFS 服务
5. 运行健康检查
6. 验证记忆写入

## 手动启动

### 1. 启动 AGFS

```bash
# 方式一：使用一键启动脚本
./scripts/start_all.sh

# 方式二：手动启动
cd agfs/agfs-server
go build -o build/agfs-server
./build/agfs-server -c config.yaml
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，设置 OGMEM_API_KEY
export OGMEM_API_KEY="sk-..."
export AGFS_BASE_URL="http://localhost:1833"
```

### 3. 运行测试

```bash
# 端到端测试
pytest tests/integration/test_e2e_full_pipeline.py -v -s

# 所有测试
pytest tests/ -v
```

## OpenClaw 集成

### 安装插件

```bash
cd openclaw_context_engine_plugin
npm install
```

### 配置 OpenClaw

编辑 `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "pythonBin": "python3",
        "logLevel": "debug"
      }
    }
  }
}
```

### 安装并启动

```bash
# 安装插件到 OpenClaw
openclaw plugins install -l ./openclaw_context_engine_plugin

# 重启 OpenClaw
openclaw restart
```

### 验证

```bash
openclaw doctor
```

应该看到 `og-memory-context-engine` 作为活跃的 context engine。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenClaw Gateway                       │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         og-memory-context-engine Plugin              │   │
│  │  ┌────────────┐        ┌────────────┐               │   │
│  │  │ after_turn │        │  assemble  │               │   │
│  │  │  (写入)     │        │  (检索)     │               │   │
│  │  └─────┬──────┘        └─────┬──────┘               │   │
│  └────────┼──────────────────────┼──────────────────────┘   │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            bridge/memory_api.py                      │   │
│  │  (Python subprocess - execFileSync bridge)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ContextEngine                           │
│                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ MemoryWrite │   │ MemoryRead  │   │  AGFS       │      │
│  │    API      │   │    API      │   │  Storage    │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   LLM       │   │  Vector     │   │   LocalFS   │      │
│  │ (Extraction)│   │  Index      │   │   / S3FS    │      │
│  └─────────────┘   └─────────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 数据流

### 写入路径 (after_turn)

```
OpenClaw messages
    │
    ▼
after_turn() → bridge/memory_api.py
    │
    ▼
MemoryWriteAPI.commit_session()
    │
    ├─→ Extraction (LLM) → CandidateMemory[]
    │       │
    │       ├─ extract_profile
    │       ├─ extract_preference
    │       ├─ extract_entity
    │       ├─ extract_event
    │       ├─ extract_case
    │       ├─ extract_pattern
    │       └─ extract_skill
    │
    ├─→ PolicyRouter → WritePlan
    │
    ├─→ ArchiveBuilder → ContextNode
    │
    ├─→ AGFSContextFS.write_node() → AGFS
    │       │
    │       ▼
    │   /accounts/{account}/users/{user}/memories/{category}/{slug}
    │       │
    │       ├── content.md
    │       ├── .abstract.md
    │       ├── .overview.md
    │       ├── .relations.json
    │       └── .meta.json
    │
    └─→ OutboxEvent (异步索引)
```

### 检索路径 (assemble)

```
OpenClaw assemble call
    │
    ▼
bridge/memory_api.py assemble()
    │
    ├─→ _extract_query(messages) → query_text
    │
    ▼
MemoryReadAPI.search_memory(query)
    │
    ├─→ QueryPlanner → TypedQuery
    │
    ├─→ RetrievalPipeline
    │       │
    │       ├─→ SeedRetriever (向量检索)
    │       ├─→ HierarchicalSearcher (L0→L1→L2)
    │       └─→ AssemblyService (排序/截断)
    │
    └─→ RetrievedBlock[] → system prompt addition
```

## 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `AGFS_BASE_URL` | ❌ | `http://localhost:1833` | AGFS 服务地址 |
| `OGMEM_API_KEY` | ✅* | - | OpenAI API 密钥 |
| `OGMEM_BASE_URL` | ❌ | `https://api.openai.com/v1` | API 地址 |
| `CONTEXTENGINE_PROVIDER` | ❌ | `openai` | LLM 提供商 |
| `VECTOR_DB_TYPE` | ❌ | `memory` | 向量数据库类型 |

\* 如不设置，将使用 mock 模式 (无 LLM 调用)

## 停止服务

```bash
# 停止 AGFS
pkill -f agfs-server

# 或使用保存的 PID
kill $(cat .agfs_pid)
```

## 故障排查

### AGFS 连接失败

```bash
curl http://localhost:1833/
```

### OpenAI API 调用失败

检查 `.env` 文件中的 `OGMEM_API_KEY` 和 `OGMEM_BASE_URL`。

### 记忆未写入

查看 OpenClaw 日志：
```bash
openclaw logs --follow
```

### 测试验证

```bash
# 端到端测试
pytest tests/integration/test_e2e_full_pipeline.py -v -s

# Contract tests
pytest tests/contract/ -v
```
