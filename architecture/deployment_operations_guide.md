# oG-Memory + OpenClaw 部署运维指南

## 1. 架构总览

```
┌─────────────────────────────────┐
│         用户浏览器               │
│   http://<server-ip>:18789      │
└──────────┬──────────────────────┘
           │ WebSocket / HTTP
           ▼
┌─────────────────────────────────┐
│     OpenClaw 容器                │
│     alpine/openclaw:latest      │
│     端口: 18789                  │
│                                 │
│  ┌─────────────────────────┐    │
│  │ og-memory-context-engine│    │
│  │ (Node.js 插件)          │    │
│  │  HTTP fetch ──────────────────┐
│  └─────────────────────────┘    ││
└─────────────────────────────────┘│
                                   │
┌──────────────────────────────────▼┐
│     oG-Memory 容器                │
│     ogmem-server:v3               │
│     端口: 8090                    │
│                                   │
│  ┌──────────┐ ┌──────────────┐   │
│  │ Flask/   │ │ IndexService │   │
│  │ Gunicorn │ │ (后台轮询)    │   │
│  └────┬─────┘ └──────┬───────┘   │
│       │              │           │
│  ┌────▼──────────────▼────┐      │
│  │    AGFS Server (:1833) │      │
│  │    文件存储引擎          │      │
│  └────────────────────────┘      │
└───────────┬──────────────────────┘
            │ volume mount
            ▼
┌───────────────────────┐
│   宿主机磁盘           │
│   /data1/.../testfs    │
└───────────────────────┘

┌───────────────────────┐
│   openGauss 数据库     │
│   端口: 8799           │
│   (外部依赖)           │
└───────────────────────┘
```

**组件说明：**

| 组件 | 镜像 | 职责 |
|------|------|------|
| OpenClaw | `alpine/openclaw:latest` | AI Agent 网关，提供前端 UI 和对话能力 |
| oG-Memory | `ogmem-server:<tag>` | 长期记忆服务，提供语义检索和记忆写入 |
| openGauss | 外部数据库 | 向量索引存储（需提前部署） |

---

## 2. 前置依赖

### 2.1 openGauss 数据库

oG-Memory 需要 openGauss 作为向量数据库（已内置向量扩展，无需手动安装）。

**确认数据库可用：**

```bash
gsql -U <user> -W <password> -d <dbname> -p <port> -r
```

> 表 `vector_index` 会在 oG-Memory 首次启动时自动创建，无需手动建表。

### 2.2 LLM API Key

需要一个兼容 OpenAI 接口的 LLM 服务（如 DashScope）。当前使用配置：

- **API Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **LLM 模型**: `glm-5`
- **Embedding 模型**: `text-embedding-v2`

### 2.3 Docker 环境

```bash
docker --version   # 确认 Docker 已安装
```

### 2.4 镜像准备

| 镜像 | 获取方式 |
|------|---------|
| `alpine/openclaw:latest` | `docker load < opencl.tar` 或从 registry 拉取 |
| `ogmem-server:<tag>` | 本地构建（见第 3 节） |

---

## 3. 构建 oG-Memory 镜像

### 3.1 源码目录结构

```
oG-Memory/
├── core/           # 数据模型、接口定义
├── retrieval/      # 检索链路（QueryPlanner → SeedRetriever → HierarchicalSearcher → AssemblyService）
├── extraction/     # 记忆提取（LLM 驱动的对话分析）
├── commit/         # 写入链路（OutboxStore 异步事件）
├── index/          # 索引服务（OutboxWorker、IndexService）
├── providers/      # 外部服务适配（LLM、Embedder、VectorIndex）
├── service/        # IndexService 全局管理
├── fs/             # 文件系统抽象（AGFSContextFS）
├── server/         # HTTP API 服务（Flask app + MemoryService）
├── scripts/        # 运维脚本（run_index_service.py）
├── agfs/           # AGFS 二进制（agfs-server）
├── docker/         # Docker 相关配置
│   ├── agfs-config.yaml
│   └── entrypoint-standalone.sh
├── openclaw_context_engine_plugin/   # OpenClaw 插件
│   ├── index.js                      # 插件入口（支持 HTTP / subprocess 双模式）
│   └── bridge/memory_api.py          # CLI 桥接（all-in-one 模式用）
├── Dockerfile.standalone             # 独立镜像 Dockerfile
└── .dockerignore
```

### 3.2 构建命令

```bash
cd /path/to/oG-Memory

# 构建镜像（指定版本 tag，避免覆盖旧版本）
docker build --network host -f Dockerfile.standalone -t ogmem-server:v3 .
```

**构建参数说明：**
- `--network host`：构建期间使用宿主机网络（加速 pip 下载）
- `-f Dockerfile.standalone`：指定独立模式 Dockerfile
- `-t ogmem-server:v3`：镜像名和版本 tag

> 构建约需 30 秒（依赖已缓存时）至 2-3 分钟（首次构建）。

### 3.3 验证镜像

```bash
docker images | grep ogmem-server
```

---

## 4. 启动 oG-Memory 服务

### 4.1 启动命令

```bash
docker run -d \
  --name ogmem-server \
  --network host \
  -v /path/to/agfs-data:/data/agfs \
  -e OPENAI_API_KEY="<your-api-key>" \
  -e OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OPENAI_EMBEDDING_MODEL="text-embedding-v2" \
  -e OPENAI_LLM_MODEL="glm-5" \
  -e "OPENGUASS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=<user> password=<password>" \
  ogmem-server:v3
```

### 4.2 环境变量参考

#### 必填参数

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENAI_API_KEY` | LLM/Embedding 服务的 API Key | `sk-xxxxxxxxxxxx` |
| `OPENGUASS_CONNECTION_STRING` | openGauss 连接串（libpq 格式） | `host=127.0.0.1 port=8799 dbname=postgres user=xxx password=xxx` |

#### 可选参数（有默认值）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API 地址 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-v2` | Embedding 模型名 |
| `OPENAI_LLM_MODEL` | （无默认） | LLM 模型名，用于记忆提取 |
| `OGMEM_HTTP_PORT` | `8090` | HTTP API 监听端口 |
| `OGMEM_WORKERS` | `2` | Gunicorn worker 进程数 |
| `OG_ACCOUNT_ID` | `acct-demo` | 默认租户 ID |
| `AGFS_BASE_URL` | `http://127.0.0.1:1833` | AGFS 内部地址（容器内使用，一般不改） |
| `AGFS_MOUNT_PREFIX` | `/local/plugin` | AGFS 文件挂载前缀 |
| `VECTOR_DB_TYPE` | `opengauss` | 向量数据库类型 |
| `INDEX_INTERVAL` | `15` | IndexService 轮询间隔（秒） |
| `CONTEXTENGINE_PROVIDER` | `openai` | LLM/Embedding 提供商 |

#### 数据卷

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `/path/to/agfs-data` | `/data/agfs` | AGFS 文件存储（记忆原始 .md 文件） |

> 此目录需要确保容器内用户有写权限。首次使用时执行 `chmod 777 /path/to/agfs-data`。

### 4.3 验证启动

```bash
# 检查容器运行状态
docker ps | grep ogmem-server

# 查看启动日志（应看到 AGFS ready、IndexService started、Gunicorn listening）
docker logs ogmem-server 2>&1 | head -20

# 健康检查
curl http://127.0.0.1:8090/api/v1/health
```

预期健康检查返回：

```json
{
  "status": "ok",
  "backend": "og-memory",
  "agfs": true,
  "agfs_url": "http://127.0.0.1:1833",
  "llm": "glm-5"
}
```

### 4.4 容器内部进程

oG-Memory 容器启动后会运行 3 个进程：

| 进程 | 角色 | 说明 |
|------|------|------|
| AGFS Server | 文件存储 | 监听 `:1833`，管理记忆 .md 文件的读写 |
| IndexService | 异步索引 | 每 15 秒轮询 `.outbox/` 目录，将新记忆向量化写入 openGauss |
| Gunicorn | HTTP API | 监听 `:8090`，处理 OpenClaw 的检索和写入请求 |

---

## 5. 启动 OpenClaw

### 5.1 准备 OpenClaw 配置

OpenClaw 需要一个 `openclaw.json` 配置文件来连接 oG-Memory。

创建配置目录和文件：

```bash
mkdir -p /path/to/openclaw-home
```

写入 `openclaw.json`：

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "dashscope/glm-5"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "dashscope": {
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "<your-api-key>",
        "api": "openai-completions",
        "models": [
          {
            "id": "glm-5",
            "name": "glm-5",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 131072,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "gateway": {
    "http": {
      "endpoints": {
        "responses": {
          "enabled": true
        }
      }
    }
  },
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "memoryApiBaseUrl": "http://127.0.0.1:8090"
        }
      },
      "memory-core": {
        "enabled": false
      }
    }
  }
}
```

### 5.2 安装 oG-Memory 插件到 OpenClaw

OpenClaw 通过插件系统加载 oG-Memory。需要将插件目录挂载到 OpenClaw 容器中：

```bash
# oG-Memory 插件源码路径
PLUGIN_SRC=/path/to/oG-Memory/openclaw_context_engine_plugin

# OpenClaw 容器内插件路径
PLUGIN_DST=/app/extensions/og-memory-context-engine
```

### 5.3 启动命令

```bash
docker run -d \
  --name openclaw \
  --network host \
  -v /path/to/openclaw-home:/home/node/.openclaw \
  -v /path/to/oG-Memory/openclaw_context_engine_plugin:/app/extensions/og-memory-context-engine \
  -e OPENCLAW_GATEWAY_TOKEN="your-gateway-token" \
  alpine/openclaw:latest
```

### 5.4 启动参数说明

#### 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENCLAW_GATEWAY_TOKEN` | 前端 WebSocket 连接凭证（首次访问时输入） | `my-token-123` |

#### 数据卷

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `/path/to/openclaw-home` | `/home/node/.openclaw` | OpenClaw 配置和状态持久化（含 `openclaw.json`） |
| `/path/to/oG-Memory/openclaw_context_engine_plugin` | `/app/extensions/og-memory-context-engine` | oG-Memory 插件代码 |

### 5.5 验证启动

```bash
# 检查容器状态
docker ps | grep openclaw

# 查看日志，确认插件加载成功
docker logs openclaw 2>&1 | grep "og-memory"
```

预期看到：

```
[plugins] [og-memory] mode=http url=http://127.0.0.1:8090
```

这表示 OpenClaw 已通过 HTTP 模式连接到 oG-Memory 服务。

---

## 6. OpenClaw 与 oG-Memory 对接原理

### 6.1 通信链路

```
用户发送消息
      │
      ▼
OpenClaw Gateway (Node.js)
      │
      ▼
og-memory-context-engine 插件 (index.js)
      │
      ├─── assemble (对话前) ──► POST http://ogmem:8090/api/v1/assemble
      │     ↳ 返回 systemPromptAddition（从记忆中召回的相关上下文）
      │
      ├─── LLM 生成回复 ◄──── 注入记忆上下文
      │
      └─── afterTurn (对话后) ──► POST http://ogmem:8090/api/v1/after_turn
            ↳ 从对话中提取新记忆，写入 AGFS + openGauss
```

### 6.2 关键配置项

对接的核心在于 `openclaw.json` 中的以下配置：

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "memoryApiBaseUrl": "http://127.0.0.1:8090"
        }
      },
      "memory-core": {
        "enabled": false
      }
    }
  }
}
```

| 配置 | 说明 |
|------|------|
| `slots.contextEngine` | 将 `og-memory-context-engine` 注册为 context engine 插槽 |
| `memoryApiBaseUrl` | oG-Memory HTTP API 地址（`--network host` 时用 `127.0.0.1`） |
| `memory-core.enabled: false` | **必须禁用** OpenClaw 原生记忆，避免两套记忆系统冲突 |

### 6.3 插件工作模式

`index.js` 插件支持两种通信模式，根据配置自动选择：

| 模式 | 触发条件 | 使用场景 |
|------|---------|---------|
| **HTTP 模式** | 配置了 `memoryApiBaseUrl` 或环境变量 `OGMEM_API_URL` | 独立部署（推荐） |
| **Subprocess 模式** | 未配置 URL | all-in-one 部署（仅内部使用） |

---

## 7. oG-Memory HTTP API 接口

oG-Memory 对外暴露以下 RESTful 端点（均为 `POST`，JSON body）：

| 端点 | 说明 | 调用时机 |
|------|------|---------|
| `GET /api/v1/health` | 健康检查 | 运维监控 |
| `POST /api/v1/assemble` | 检索相关记忆，返回 systemPromptAddition | 每轮对话前 |
| `POST /api/v1/after_turn` | 从对话中提取并写入新记忆 | 每轮对话后 |
| `POST /api/v1/ingest` | 单条消息注入（当前透传） | OpenClaw 消息处理 |
| `POST /api/v1/compact` | 记忆压缩（当前透传） | OpenClaw 触发 |
| `POST /api/v1/dispose` | 清理资源 | 会话结束 |

**assemble 请求示例：**

```bash
curl -X POST http://127.0.0.1:8090/api/v1/assemble \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "周末想跟男朋友出去玩"}],
    "prompt": "周末想跟男朋友出去玩",
    "sessionId": "test-session",
    "accountId": "acct-demo",
    "userId": "u-alice",
    "agentId": "agent-1"
  }'
```

---

## 8. 完整启动流程（Quick Start）

按照以下顺序操作，即可实现完整功能：

### Step 1：确认 openGauss 可用

```bash
gsql -U sundechao -W 'Huawei@123' -d postgres -p 8799 -r
# 执行 \q 退出
```

### Step 2：准备数据目录

```bash
mkdir -p /data1/sundechao/ogmem/testfs
chmod 777 /data1/sundechao/ogmem/testfs
```

### Step 3：构建 oG-Memory 镜像

```bash
cd /path/to/oG-Memory
docker build --network host -f Dockerfile.standalone -t ogmem-server:v3 .
```

### Step 4：启动 oG-Memory

```bash
docker run -d \
  --name ogmem-server \
  --network host \
  -v /data1/sundechao/ogmem/testfs:/data/agfs \
  -e OPENAI_API_KEY="sk-xxxxxxxxxxxx" \
  -e OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OPENAI_EMBEDDING_MODEL="text-embedding-v2" \
  -e OPENAI_LLM_MODEL="glm-5" \
  -e "OPENGUASS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=Huawei@123" \
  ogmem-server:v3
```

等待 10 秒后验证：

```bash
curl http://127.0.0.1:8090/api/v1/health
# 预期：{"status":"ok", ...}
```

### Step 5：准备 OpenClaw 配置

```bash
mkdir -p /path/to/openclaw-home
# 将上面第 5.1 节的 openclaw.json 写入此目录
# 注意替换 apiKey 和 memoryApiBaseUrl
```

### Step 6：启动 OpenClaw

```bash
docker run -d \
  --name openclaw \
  --network host \
  -v /path/to/openclaw-home:/home/node/.openclaw \
  -v /path/to/oG-Memory/openclaw_context_engine_plugin:/app/extensions/og-memory-context-engine \
  -e OPENCLAW_GATEWAY_TOKEN="test-token-123" \
  alpine/openclaw:latest
```

### Step 7：访问前端

浏览器打开：`http://<server-ip>:18789`

首次访问时输入 Gateway Token（即 `OPENCLAW_GATEWAY_TOKEN` 的值）。

### Step 8：验证 oG-Memory 工作

发送几条消息后，检查日志：

```bash
# 检查 oG-Memory 是否命中记忆
docker logs ogmem-server 2>&1 | grep "assemble"

# 检查 OpenClaw 是否调用了 oG-Memory
docker logs openclaw 2>&1 | grep "og-memory"
```

---

## 9. 常用运维命令

```bash
# 查看所有相关容器
docker ps | grep -E "ogmem|openclaw"

# 查看 oG-Memory 日志（实时跟踪）
docker logs -f ogmem-server

# 查看 OpenClaw 日志
docker logs -f openclaw

# 重启 oG-Memory（不丢失数据）
docker restart ogmem-server

# 停止并删除容器（镜像和数据不受影响）
docker stop ogmem-server openclaw
docker rm ogmem-server openclaw

# 查看 AGFS 存储的记忆文件
ls /data1/sundechao/ogmem/testfs/plugin/accounts/

# 查看 openGauss 中的向量索引数据
gsql -U sundechao -W 'Huawei@123' -d postgres -p 8799 -c "SELECT count(*) FROM vector_index;"
```

---

## 10. 故障排查

| 问题 | 检查方法 | 解决方案 |
|------|---------|---------|
| oG-Memory health 返回 503 | `docker logs ogmem-server` | 检查 openGauss 连接串和 AGFS 启动日志 |
| OpenClaw 日志无 `[og-memory]` | `docker logs openclaw \| grep plugin` | 确认插件目录挂载正确、`openclaw.json` 配置了 `contextEngine` 插槽 |
| assemble 返回 0 hits | `docker logs ogmem-server \| grep assemble` | 正常现象（首次使用无历史记忆），多轮对话后会积累 |
| `permission denied` 写入失败 | `docker logs ogmem-server \| grep permission` | `chmod 777 /path/to/agfs-data` |
| 前端输入 Token 无效 | 清除浏览器 localStorage | 使用 `OPENCLAW_GATEWAY_TOKEN` 环境变量中设置的值 |
| `OPENGUASS_CONNECTION_STRING is required` | 检查 `docker run` 参数 | 确保 `-e` 参数中包含此连接串 |
| `Network is unreachable` (LLM) | `curl https://dashscope.aliyuncs.com` | 确认 `OPENAI_BASE_URL` 正确且网络可达 |
