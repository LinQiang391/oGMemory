# oG-Memory for OpenClaw

基于 openGauss 的语义记忆插件，为 OpenClaw 提供持久化记忆能力。

支持能力：

- 语义检索（`ogmemory_search`）
- 手动写入记忆（`ogmemory_add`）
- 记忆统计（`ogmemory_stats`）
- 聊天命令检索（`/memory <query>`）
- 会话前自动注入近期文件记忆（来自 `memoryDir`）

---

## 1. 安装

将插件目录放到 OpenClaw 扩展目录，并命名为 `ogmemory`：

```bash
# 复制
cp -r openclaw-plugin ~/.openclaw/extensions/ogmemory

# 或开发模式软链接
ln -s /path/to/oG-Memory/openclaw-plugin ~/.openclaw/extensions/ogmemory
```

> Windows 下可手动复制到 `%USERPROFILE%/.openclaw/extensions/ogmemory`。

---

## 2. 最小配置

在 OpenClaw 配置中启用插件并绑定到 memory slot：

```json
{
  "plugins": {
    "allow": ["ogmemory"],
    "slots": {
      "memory": "ogmemory"
    },
    "entries": {
      "ogmemory": {
        "enabled": true,
        "config": {
          "dbHost": "localhost",
          "dbPort": 5432,
          "dbName": "memory_db",
          "dbUser": "postgres",
          "dbPassword": "your_password",
          "embeddingProvider": "openai",
          "embeddingModel": "text-embedding-3-small",
          "openaiApiKey": "sk-xxx",
          "openaiBaseUrl": "https://api.openai.com/v1",
          "memoryDir": ".ogmemory/memory"
        }
      }
    }
  }
}
```

> 插件当前同时兼容两种配置输入：直接传 `config`，或 `{ "config": {...} }` 包裹形式。

---

## 3. Docker 场景（推荐）

如果你在容器中运行 OpenClaw：

1. `dbHost` 应填数据库服务名（如 `opengauss`），不要填容器内的 `127.0.0.1`。
2. 需要让插件使用包含 Python 依赖的解释器：
   - 通过环境变量指定 `PYTHON_PATH`（例如 `/tmp/ogmemory-build/.venv/bin/python3`）。
3. 将项目挂载到容器内并保证 `ogmemory_cli.py` 可访问（默认会自动尝试 `/workspace/oG-Memory`）。

---

## 4. 使用方式

### 4.1 Agent Tools

- `ogmemory_search`
- `ogmemory_add`
- `ogmemory_stats`
- `ogmemory_clear_cache`

### 4.2 CLI（如果 OpenClaw CLI 已注册扩展命令）

```bash
# 搜索
openclaw ogmemory search "你的问题" --top-k 5 --hybrid

# 添加
openclaw ogmemory add "一条记忆"

# 统计
openclaw ogmemory stats

# 重置（危险）
openclaw ogmemory reset --yes
```

### 4.3 聊天 Slash 命令

```text
/memory 你要检索的内容
```

---

## 5. 配置项说明

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `dbHost` | string | `localhost` | openGauss 主机 |
| `dbPort` | number | `5432` | openGauss 端口 |
| `dbName` | string | `memory_db` | 数据库名 |
| `dbUser` | string | `postgres` | 用户名 |
| `dbPassword` | string | `""` | 数据库密码 |
| `embeddingProvider` | string | `openai` | 向量服务提供方 |
| `embeddingModel` | string | `text-embedding-3-small` | 向量模型 |
| `openaiApiKey` | string | 读取环境变量 `OPENAI_API_KEY` | API Key |
| `openaiBaseUrl` | string | 读取环境变量 `OPENAI_BASE_URL` | 兼容 OpenAI 的 Base URL |
| `chunkSize` | number | `1000` | 文本分块大小 |
| `chunkOverlap` | number | `200` | 分块重叠 |
| `memoryDir` | string | `.ogmemory/memory` | 本地记忆目录（用于自动注入） |

---

## 6. 常见问题

### 6.1 `ogmemory command failed (code=1)`

插件会透传 stderr 的尾部信息，请优先看报错末尾关键字。

常见原因：

- `OPENAI_API_KEY` 未生效
- `embeddingModel` 在当前供应商不可用（例如渠道不支持 `text-embedding-3-small`）
- `dbHost` 填错（容器内应使用服务名）

### 6.2 `ModuleNotFoundError: No module named 'ogmemory'`

说明插件调用到了错误的 Python 解释器。请设置：

- `PYTHON_PATH=/tmp/ogmemory-build/.venv/bin/python3`（或你的实际 venv 路径）

### 6.3 数据库连接失败（`connection refused`）

- 检查 openGauss 是否启动并健康
- 检查网络连通（容器网络 / 端口映射）
- 检查用户名、密码、数据库名是否一致

---

## 7. 验证建议

完成配置后，按顺序验证：

1. `ogmemory_stats` 可返回统计
2. `ogmemory_add` 可成功写入
3. `ogmemory_search` 或 `/memory` 可检索到新记忆

---

## License

MIT