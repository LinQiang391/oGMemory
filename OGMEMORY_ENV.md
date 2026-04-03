# oG-Memory 配置参考

> 从 v0.9 起支持 **YAML 配置文件**，同时保持环境变量向后兼容。

---

## 配置方式

### 方式一：YAML 配置文件（推荐）

将配置写入一个 YAML 文件，Docker 启动时挂载到容器的 `/etc/ogmem/config.yaml`：

```bash
docker run -d --name ogmemory --network host \
  -v ./ogmemory.yaml:/etc/ogmem/config.yaml \
  ogmemory:v0.9
```

也可通过 `OGMEM_CONFIG` 环境变量指定其他路径：

```bash
docker run -d --name ogmemory --network host \
  -e OGMEM_CONFIG=/data/my-config.yaml \
  -v ./my-config.yaml:/data/my-config.yaml \
  ogmemory:v0.9
```

完整 YAML 示例见 `docker/ogmemory.example.yaml`。

### 方式二：环境变量（向后兼容）

不挂载 YAML 文件时，行为与旧版完全一致，通过 `-e` 传递环境变量。

### 优先级

```
YAML 文件值  >  环境变量  >  代码默认值
```

YAML 中未设置的字段会自动回退到对应的环境变量。

---

## YAML 配置结构

```yaml
llm:
  provider: openai          # mock / openai / openai-cached
  api_key: "sk-xxx"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  model: "doubao-seed-2-0-code-preview-260215"
  temperature: 0.7
  max_tokens: 4096

embedding:
  model: "doubao-embedding-vision-250615"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  api_key: null              # 缺省回退到 llm.api_key
  multimodal: true

vector_db:
  type: opengauss
  connection_string: "host=127.0.0.1 port=8799 dbname=postgres user=xxx password=xxx"
  dimension: 1024
  table_name: vector_index
  pool_size: 5

service:
  http_port: 8090
  workers: 2

agfs:
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"

index:
  interval: 15
  workers: 1

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

cache:
  enabled: true
  max_size: 1000
```

---

## 环境变量对照表

下表列出每个 YAML 字段对应的环境变量和默认值。

### LLM 配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `llm.provider` | `CONTEXTENGINE_PROVIDER` | `openai` (Dockerfile) / `mock` (代码) | 否 | Provider 类型：`mock` / `openai` / `openai-cached` |
| `llm.api_key` | `OPENAI_API_KEY` | 无 | **是** | LLM 和 Embedding 共用的 API Key |
| `llm.base_url` | `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 否 | LLM API Base URL。不以 `/v1` `/v2` `/v3` 结尾时自动追加 `/v1` |
| `llm.model` | `OPENAI_LLM_MODEL` | `gpt-4o-mini` | 否 | LLM 模型名称 |
| `llm.temperature` | `LLM_TEMPERATURE` | `0.7` | 否 | LLM 采样温度 |
| `llm.max_tokens` | `LLM_MAX_TOKENS` | `4096` | 否 | LLM 最大输出 token 数 |

### Embedding 配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `embedding.model` | `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | 否 | Embedding 模型名称 |
| `embedding.base_url` | `OPENAI_EMBEDDING_BASE_URL` | 回退到 `llm.base_url` | 否 | Embedding API Base URL |
| `embedding.api_key` | `OPENAI_EMBEDDING_API_KEY` | 回退到 `llm.api_key` | 否 | Embedding API Key |
| `embedding.multimodal` | `OPENAI_EMBEDDING_MULTIMODAL` | `false` | 否 | 使用 `/embeddings/multimodal` 端点。设为 `true` / `1` / `yes` 启用 |

### 向量数据库

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `vector_db.type` | `VECTOR_DB_TYPE` | `opengauss` (Dockerfile) / `memory` (代码) | 否 | `memory` 内存 / `opengauss` 持久化 |
| `vector_db.connection_string` | `OPENGAUSS_CONNECTION_STRING` | 无 | type=opengauss 时**必填** | 连接串格式：`host=... port=... dbname=... user=... password=...` |
| `vector_db.dimension` | `OPENGAUSS_DIMENSION` | `1536` | 否 | 向量维度，须与 Embedding 模型一致 |
| `vector_db.table_name` | `OPENGAUSS_TABLE_NAME` | `vector_index` | 否 | 向量存储表名 |
| `vector_db.pool_size` | `OPENGAUSS_POOL_SIZE` | `5` | 否 | 数据库连接池大小 |

### 服务配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `service.http_port` | `OGMEM_HTTP_PORT` | `8090` | 否 | HTTP API 监听端口 |
| `service.workers` | `OGMEM_WORKERS` | `2` | 否 | Gunicorn worker 数量 |

### AGFS

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `agfs.base_url` | `AGFS_BASE_URL` | `http://127.0.0.1:1833` | 否 | 容器内自启动，通常无需修改 |
| `agfs.mount_prefix` | `AGFS_MOUNT_PREFIX` | `/local/plugin` | 否 | AGFS 挂载路径前缀 |

### Index Service

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `index.interval` | `INDEX_INTERVAL` | `30` | 否 | 轮询间隔（秒） |
| `index.workers` | `INDEX_WORKERS` | `1` | 否 | 索引 worker 数量 |

### 身份/多租户

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `identity.account_id` | `OG_ACCOUNT_ID` | `acct-demo` | 否 | 默认 account ID |
| `identity.user_id` | `OG_USER_ID` | `u-alice` | 否 | 默认 user ID |
| `identity.agent_id` | `OG_AGENT_ID` | `main` | 否 | 默认 agent ID |

### 缓存

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `cache.enabled` | `OGMEM_CACHE_ENABLED` | `true` | 否 | 启用 Embedder/LLM 缓存 |
| `cache.max_size` | `OGMEM_CACHE_MAX_SIZE` | `1000` | 否 | 缓存条目上限 |

---

## 启动示例

### YAML 文件方式（推荐）

```bash
docker run -d --name ogmemory --network host \
  -v ./ogmemory.yaml:/etc/ogmem/config.yaml \
  ogmemory:v0.9
```

### 环境变量方式（向后兼容）

```bash
docker run -d --name ogmemory --network host \
  -e OPENAI_API_KEY=your-api-key \
  -e OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3 \
  -e OPENAI_LLM_MODEL=doubao-seed-2-0-code-preview-260215 \
  -e OPENAI_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3 \
  -e OPENAI_EMBEDDING_MODEL=doubao-embedding-vision-250615 \
  -e OPENAI_EMBEDDING_MULTIMODAL=true \
  -e OPENGAUSS_DIMENSION=1024 \
  -e OPENGAUSS_CONNECTION_STRING="host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=xxx" \
  ogmemory:v0.9
```

---

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/assemble` | POST | 组装上下文（由 OpenClaw context engine 调用） |
| `/api/v1/after_turn` | POST | 对话轮结束后回调（记忆提取 + 存储） |
| `/api/v1/ingest` | POST | 消息摄入 |
| `/api/v1/ingest_batch` | POST | 批量消息摄入 |
| `/api/v1/compact` | POST | 上下文压缩 |
| `/api/v1/bootstrap` | POST | 引擎初始化 |
| `/api/v1/dispose` | POST | 释放资源 |
