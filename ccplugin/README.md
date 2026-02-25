# oG-Memory — Claude Code 插件

基于 openGauss 的语义记忆，为 Claude Code 提供持久化记忆能力。自动索引会话摘要，支持语义检索与上下文注入。

## 快速开始

### 1. 使用 uv 初始化环境

```bash
# 在 oG-Memory 项目根目录
cd /path/to/oG-Memory
uv sync

# 设置嵌入 API 密钥（必填）
export OPENAI_API_KEY="sk-..."
```

### 2. 配置 openGauss

```sql
CREATE DATABASE memory_db;
```

### 3. 安装并启用插件

```bash
# 使用本地插件目录
claude --plugin-dir ./ccplugin
```

### 4. 实际使用示例

启动 Claude Code 后，插件会自动：

- **SessionStart**：启动 watch 进程、注入近期记忆
- **Stop**：解析会话、写入当日 `.ogmemory/memory/YYYY-MM-DD.md`、索引到 openGauss
- **SessionEnd**：停止 watch 进程

#### 手动 CLI 操作（在项目根目录执行）

```bash
# 搜索记忆
uv run python -m ogmemory search "your query" --top-k 5

# 索引目录
uv run python -m ogmemory index .ogmemory/memory/

# 查看统计
uv run python -m ogmemory stats

# 重置数据
uv run python -m ogmemory reset --yes

# 展开块获取完整上下文
uv run python -m ogmemory expand <chunk_id>

# 手动添加记忆
uv run python -m ogmemory add "Your memory here" --metadata key=value
```

## 架构

```
┌─────────────────────────────────────────┐
│     Claude Code Session                  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼─────────┐
│ SessionStart │  │   Stop Hook    │
│ - Start watch│  │  - Summarize   │
│ - Inject ctx │  │  - Index      │
└──────┬──────┘  └──────┬─────────┘
       │                │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ memory-recall   │
       │   Skill (fork)  │
       └──────┬──────────┘
              │
       ┌──────▼────────┐
       │  openGauss DB  │
       │  + Vectors     │
       └────────────────┘
```

## 钩子说明

| 钩子 | 作用 |
|------|------|
| **SessionStart** | 启动 watch、写入会话标题、注入近期记忆 |
| **UserPromptSubmit** | 显示 "[ogmemory] Memory available" 提示 |
| **Stop** | 解析 transcript、摘要写入当日 md、索引 |
| **SessionEnd** | 停止 watch 进程 |

## 记忆存储

记忆以 Markdown 形式存储在 `.ogmemory/memory/`：

```
your-project/
├── .ogmemory/
│   ├── .watch.pid
│   └── memory/
│       ├── 2026-02-10.md
│       ├── 2026-02-11.md
│       └── 2026-02-12.md
└── ...
```

## 渐进式检索（memory-recall skill）

1. **L1: Search** — 向量语义搜索
2. **L2: Expand** — 获取完整 markdown 段落
3. **L3: Summary** — 汇总后返回主会话

## 配置

```bash
# openGauss
export OG_DB_HOST=localhost
export OG_DB_PORT=5432
export OG_DB_NAME=memory_db
export OG_DB_USER=postgres
export OG_DB_PASSWORD=your_password

# 嵌入
export OG_EMBEDDING_PROVIDER=openai
export OG_EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_API_KEY=sk-...
```

## 故障排查

**OPENAI_API_KEY 未设置**

```bash
export OPENAI_API_KEY="sk-..."
```

**搜索无结果**

```bash
uv run python -m ogmemory stats
uv run python -m ogmemory index .ogmemory/memory/ --force
```

**Watch 未运行**

```bash
cat .ogmemory/.watch.pid
kill $(cat .ogmemory/.watch.pid) 2>/dev/null
rm -f .ogmemory/.watch.pid
```

## 许可证

MIT License
