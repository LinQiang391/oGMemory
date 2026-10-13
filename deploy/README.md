# OpenClaw + oGMemory 一键安装部署指南

## 概述

本部署方案采用 Docker 容器化部署，包含以下组件：

| 组件 | 容器名称 | 端口 | 说明 |
|------|---------|------|------|
| openGauss | opengauss | 15432 | 向量数据库（可选） |
| oGMemory | ogmem | 8090 | 记忆服务 HTTP API |
| OpenClaw Gateway | openclaw_ogmem | 18789 | Web 界面入口 |

> **⚠️ 架构说明：** 本项目提供的 Docker 镜像仅支持 **ARM64 (鲲鹏)** 架构，不支持 x86_64。请确保部署环境为 ARM 服务器（如华为云鲲鹏实例）。

## 环境要求

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+, CentOS 7+, openEuler)
- **CPU 架构**: **ARM64 (鲲鹏)** — 镜像仅支持 ARM，不支持 x86_64
- **内存**: 至少 8GB (推荐 16GB+)
- **磁盘**: 至少 20GB 可用空间

### 软件要求

- **Docker**: 20.10+
- **curl**: 用于下载脚本

检查 Docker 是否安装：

```bash
docker --version
```

确认当前系统为 ARM64 架构：

```bash
uname -m
# 输出应为 aarch64
```

如未安装 Docker，请参考 [Docker 官方文档](https://docs.docker.com/engine/install/) 安装。

### 网络要求

- 确保能访问以下地址：
  - 华为云 SWR 镜像仓库: `swr.cn-north-4.myhuaweicloud.com`
  - LLM API 服务地址（如华为云、火山引擎、阿里云等）

---

## 快速开始

多用户服务器中，每位用户使用独立实例、模型和端口时，参见[用户一键部署与体验指南](user-quickstart/README_zh.md)。

### 远程一键安装

```bash
curl -fsSL https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/install.sh | bash
```

执行后会下载以下文件到 `./ogmem-deploy/` 目录：
- `deploy.env`: 配置文件（OpenClaw + 基础设施配置）
- `ogmemory.example.yaml`: oGMemory 服务配置模板
- `deploy.sh`: 部署脚本

---

## 配置说明

### 配置文件结构

本部署方案使用 **两个配置文件**，各司其职：

| 文件 | 职责 | 内容 |
|------|------|------|
| `deploy.env` | OpenClaw + 基础设施 | LLM 共享配置、openGauss、容器名/镜像、部署参数 |
| `ogmemory.yaml` | oGMemory 业务配置 | LLM、Embedding、向量数据库、身份认证、缓存等 |

**关键机制**：`ogmemory.yaml` 支持 `${ENV_VAR}` 语法引用 `deploy.env` 中的变量，避免重复配置。

**职责边界更新：**

- `deploy.sh` 不再从 `ogmemory.yaml` 反向推导 OpenClaw 默认认证参数。
- 单实例模式下，OpenClaw 认证参数只按 `OG_AUTH_API_KEY -> OG_ROOT_API_KEY`、`OG_AUTH_ACCOUNT_ID -> OG_ACCOUNT_ID` 回退。
- `openclaw.json` 由 OpenClaw 容器入口脚本托管；升级后如果检测到缺少关键字段，会先备份旧文件，再按当前模板重建。

**推荐做法：**

- 把共享配置统一放在 `deploy.env`，例如 `OG_ROOT_API_KEY`、`OG_ACCOUNT_ID`、`OG_USER_ID`、`OG_AGENT_ID`。
- `ogmemory.yaml` 默认通过 `${ENV_VAR}` 引用这些值，不再手工重复填写相同内容。
- 只有在 oGMemory 需要和 OpenClaw 使用不同身份或不同认证策略时，才在 `ogmemory.yaml` 里单独覆盖。

```
deploy.env                    ogmemory.yaml
┌──────────────────┐          ┌────────────────────────┐
│ LLM_API_KEY=sk-xxx│──────→  │ api_key: "${LLM_API_KEY}" │
│ LLM_BASE_URL=... │──────→  │ base_url: "${LLM_BASE_URL}"│
│ LLM_MODEL=qwen3  │──────→  │ model: "${LLM_MODEL}"      │
│ OG_ROOT_API_KEY= │──────→  │ root_api_key: "${OG_ROOT...│
│ OG_ACCOUNT_ID=   │──────→  │ account_id: "${OG_ACCOUNT..│
│ ...              │          │ ...                      │
└──────────────────┘          └────────────────────────┘
```

- **共享配置**：`ogmemory.yaml` 中写 `${LLM_API_KEY}` 等占位符，自动使用 `deploy.env` 的值
- **oGMemory 专用**：将 `${...}` 替换为硬编码值即可覆盖

### 第一步：编辑 deploy.env

```bash
cd ogmem-deploy  # 或 cd oGMemory/deploy
vim deploy.env
```

### 第二步：创建 ogmemory.yaml

```bash
cp ogmemory.example.yaml ogmemory.yaml
vim ogmemory.yaml
```

> **💡 提示：** 默认模板中 LLM/Embedding 配置已使用 `${...}` 引用，大多数情况下只需修改 Embedding 模型名称和向量维度即可。

### deploy.env 环境变量列表

> **💡 重要提示：**
> 1. `LLM_PROVIDER`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 这四个变量同时供 OpenClaw 和 oGMemory 使用（通过 `ogmemory.yaml` 中的 `${...}` 引用）。如果 oGMemory 需要不同的配置，请在 `ogmemory.yaml` 中将对应字段改为硬编码值。
> 2. `OPENCLAW_HOST_IP` 建议填写宿主机实际 IP 地址（而非 `127.0.0.1`），以便其他机器或容器能够访问服务。

#### 必填环境变量

| 变量名 | 说明 | 示例值 | 适用服务 |
|--------|------|--------|---------|
| `LLM_PROVIDER` | LLM 提供商标识 | `dashscope` | OpenClaw + oGMemory 共用 |
| `LLM_API_KEY` | LLM API 密钥 | `sk-xxxxx` | OpenClaw + oGMemory 共用 |
| `LLM_BASE_URL` | LLM API 基础地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenClaw + oGMemory 共用 |
| `LLM_MODEL` | LLM 模型名称 | `qwen3.6-plus` | OpenClaw + oGMemory 共用 |
| `ENABLE_OPENGAUSS` | 是否启用 openGauss | `true` / `false` | openGauss |
| `OG_HOST_PORT` | openGauss 宿主机端口（启用 openGauss 时必填） | `15432` | openGauss |
| `OPENGAUSS_HOST_IP` | openGauss 数据库宿主机 IP 地址 | `127.0.0.1` | openGauss |
| `OPENCLAW_HOST_IP` | OpenClaw 宿主机 IP 地址（建议填写宿主机实际 IP） | `192.168.1.100` | OpenClaw + oGMemory |

#### 可选环境变量 — OpenClaw 连接配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `GATEWAY_PORT` | OpenClaw Gateway 端口 | `18789` | OpenClaw |
| `OPENCLAW_GATEWAY_TOKEN` | 前端配对密码 | `ogmem-default-token` | OpenClaw |
| `OPENCLAW_BIND_MODE` | 绑定模式（`lan` / `localhost`） | `lan` | OpenClaw |
| `OGMEM_URL` | oGMemory API 地址（供 OpenClaw 连接） | `http://127.0.0.1:8090` | OpenClaw |
| `OG_AUTH_API_KEY` | OpenClaw 连接 oGMemory 的认证 API Key | 回退至 `OG_ROOT_API_KEY` | OpenClaw |
| `OG_AUTH_ACCOUNT_ID` | OpenClaw 连接 oGMemory 的账户 ID | 回退至 `OG_ACCOUNT_ID` | OpenClaw |

> **说明：** 单实例模式默认不需要额外配置 `OG_AUTH_API_KEY` / `OG_AUTH_ACCOUNT_ID`，脚本会分别回退到 `OG_ROOT_API_KEY` / `OG_ACCOUNT_ID`。只有当 OpenClaw 需要使用不同于默认值的认证身份时，才建议单独覆盖。

#### 可选环境变量 — 多租认证配置

> 启用多租认证后，所有数据路由和管理路由都需要 API Key 认证。启用后需要同时配置 `OG_ROOT_API_KEY` 或 `OG_ADMIN_API_KEYS`。

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OG_ROOT_API_KEY` | Root API Key（超级管理员密钥，拥有全局最高权限） | 无 | oGMemory |
| `OG_ADMIN_API_KEYS` | Admin API Keys（账户管理员密钥，逗号分隔） | 无 | oGMemory |
| `OG_ACCOUNT_ID` | 默认租户标识，同时也是单实例模式下 OpenClaw 默认访问账户 | `acct-demo` | oGMemory / OpenClaw |
| `OG_USER_ID` | 默认用户标识 | `u-alice` | oGMemory |
| `OG_AGENT_ID` | 默认 Agent 标识 | `main` | oGMemory |

> **推荐：** 单实例或共享默认身份场景下，只需要在 `deploy.env` 里配置一次 `OG_ROOT_API_KEY`、`OG_ACCOUNT_ID`、`OG_USER_ID`、`OG_AGENT_ID`；默认模板中的 `ogmemory.yaml` 会直接复用这些值。

**`OG_ADMIN_API_KEYS` 格式说明：**

- 绑定账户格式：`account_id:key`（只能管理指定账户）
- 全局格式：`key`（绑定到 `OG_ACCOUNT_ID` 对应的默认账户）
- 多个 key 用逗号分隔

```bash
# 示例：两个账户各一个 Admin Key
OG_ADMIN_API_KEYS="acct-company-a:admin-key-for-a,acct-company-b:admin-key-for-b"

# 示例：全局 Admin Key（绑定到 OG_ACCOUNT_ID）
OG_ADMIN_API_KEYS="global-admin-key"
```

**`OG_AUTH_API_KEY` 说明：**

- 默认使用 `OG_ROOT_API_KEY`（ROOT 权限，可访问所有账户数据）
- 也可设置为 Admin Key（限定账户权限），此时需填写 `account_id:key` 中的 **key 部分**
- 当 `OG_AUTH_API_KEY` 是 Admin Key 时，`OG_AUTH_ACCOUNT_ID` 必须与 Key 绑定的账户一致

#### 可选环境变量 — 数据持久化配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `AGFS_DATA_DIR` | AGFS 数据存储目录（持久化记忆文件） | 无（不挂载） | oGMemory |
| `OPENCLAW_HOME_DIR` | OpenClaw 状态目录（避免重复配对） | 无（不挂载） | OpenClaw |

> **💡 推荐：** 设置 `AGFS_DATA_DIR` 以持久化记忆数据，设置 `OPENCLAW_HOME_DIR` 以避免重复配对。

#### 可选环境变量 — 容器配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OGMEM_CONTAINER_NAME` | ogmemory 容器名称 | `ogmem` | 容器 |
| `OPENCLAW_CONTAINER_NAME` | openclaw-ogmemory 容器名称 | `openclaw_ogmem` | 容器 |
| `OGMEM_IMAGE` | ogmemory 镜像地址 | 见 deploy.env | 容器 |
| `OPENCLAW_IMAGE` | openclaw-ogmemory 镜像地址 | 见 deploy.env | 容器 |

#### 可选环境变量 — openGauss 容器配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OG_CONTAINER_NAME` | openGauss 容器名称 | `opengauss` | openGauss |
| `OG_PORT` | openGauss 容器内端口 | `5432` | openGauss |
| `OG_NODE_NAME` | openGauss 节点名称 | `gaussdb` | openGauss |
| `OG_USERNAME` | openGauss 用户名 | `gaussdb` | openGauss |
| `OG_DB_NAME` | openGauss 数据库名 | `postgres` | openGauss |
| `OG_IMAGE_REPO` | openGauss 镜像仓库 | 见 deploy.env | openGauss |
| `OG_IMAGE_TAG` | openGauss 镜像标签 | `0328` | openGauss |
| `OG_CPUSET_CPUS` | CPU 亲和性 | 无 | openGauss |
| `OG_WAIT_TIMEOUT` | 启动超时时间（秒） | `120` | openGauss |

#### 可选环境变量 — 部署控制

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `AUTO_HEALTH_CHECK` | 是否自动健康检查 | `true` | 部署控制 |
| `SKIP_PULL` | 是否跳过镜像拉取 | `false` | 部署控制 |

### ogmemory.yaml 配置字段

> 完整字段说明请参考 `ogmemory.example.yaml` 中的注释。

#### YAML 字段与环境变量的对应关系

下面这张表结合了 [OGMEMORY_ENV.md](../docs/OGMEMORY_ENV.md) 和当前代码默认值，方便直接查询字段映射、默认值和可选值。

| YAML 字段 | 对应环境变量 | 默认值 | 常见可选值 / 格式 | 说明 |
|-----------|-------------|--------|------------------|------|
| `llm.provider` | `CONTEXTENGINE_PROVIDER` | 代码默认 `mock`；部署时通常直接复用 `LLM_PROVIDER` | `mock` / `openai` / `openai-cached`，以及 `dashscope`、`volcengine`、`zhipu` 等兼容 OpenAI 协议的提供商标识 | LLM 提供商 |
| `llm.api_key` | `LLM_API_KEY` | 空 | API Key 字符串 | 部署模板通常复用 OpenClaw 的 `LLM_API_KEY` |
| `llm.base_url` | `LLM_BASE_URL` | 空 | 兼容接口地址 | 部署模板通常复用 OpenClaw 的 `LLM_BASE_URL` |
| `llm.model` | `OGMEM_LLM_MODEL` 或模板中的 `LLM_MODEL` | `gpt-4o-mini` | 任意可用模型名，如 `qwen3-max`、`gpt-4o-mini`、`glm-4.5` | 部署模板默认直接引用 `LLM_MODEL` |
| `llm.temperature` | `LLM_TEMPERATURE` | `0.7` | 通常 `0.0` 到 `2.0` | 数值越低越稳定 |
| `llm.max_tokens` | `LLM_MAX_TOKENS` | `4096` | 正整数 | 单次生成最大 token 数 |
| `embedding.provider` | `EMBEDDING_PROVIDER` | 空，回退到 `llm.provider` | `openai` / `st` / `volcengine` / `mock` | Embedding 提供商 |
| `embedding.model` | `OGMEM_EMBEDDING_MODEL` | `text-embedding-ada-002` | 你的 Embedding 模型名，如 `text-embedding-v4` | 需要和向量维度匹配 |
| `embedding.base_url` | `OGMEM_EMBEDDING_BASE_URL` | 空，回退到 `llm.base_url` | 兼容接口地址 | Embedding API 地址 |
| `embedding.api_key` | `OGMEM_EMBEDDING_API_KEY` | 空，回退到 `llm.api_key` | API Key 字符串 | Embedding API Key |
| `embedding.multimodal` | `OGMEM_EMBEDDING_MULTIMODAL` | `false` | `true` / `false` | 仅多模态 Embedding 端点时设为 `true` |
| `embedding.st_model` | `ST_MODEL` | `BAAI/bge-large-en-v1.5` | 任意 Sentence Transformer 模型名 | 仅 `embedding.provider=st` 时通常有意义 |
| `vector_db.type` | `VECTOR_DB_TYPE` | 代码默认 `chroma`；部署模板默认写的是 `opengauss` | `memory` / `opengauss` / `chroma` | 向量数据库类型 |
| `vector_db.connection_string` | — | 空 | `host=... port=... dbname=... user=... password=...` | 启用 openGauss 时由 `deploy.sh` 自动替换 |
| `vector_db.dimension` | `OPENGAUSS_DIMENSION` | `1024` | 正整数，如 `1024` / `1536` / `3072` | 必须与 Embedding 模型维度一致 |
| `vector_db.table_name` | `OPENGAUSS_TABLE_NAME` | `vector_index` | 任意合法表名 | openGauss 向量表名 |
| `vector_db.pool_size` | `OPENGAUSS_POOL_SIZE` | `5` | 正整数 | openGauss 连接池大小 |
| `vector_db.chroma_persist_dir` | `CHROMA_PERSIST_DIR` | `.chroma_data` | 目录路径 | 使用 Chroma 持久化时生效 |
| `vector_db.chroma_collection` | `CHROMA_COLLECTION` | `contextengine` | 任意集合名 | 使用 Chroma 时生效 |
| `service.http_port` | `OGMEM_HTTP_PORT` | `8090` | 合法端口号 | oGMemory HTTP 服务端口 |
| `service.workers` | `OGMEM_WORKERS` | `2` | 正整数 | Gunicorn worker 数量 |
| `service.http_ip_allowlist` | `OG_HTTP_IP_ALLOWLIST` | 空 | 逗号分隔 IP / CIDR，例如 `127.0.0.1,10.0.0.0/8` | HTTP 白名单 |
| `service.http_ip_allowlist_trust_proxy` | `OG_HTTP_IP_ALLOWLIST_TRUST_PROXY` | `false` | `true` / `false` | 只有在反向代理前置时才建议开启 |
| `service.http_trusted_proxies` | `OG_HTTP_TRUSTED_PROXIES` | 空 | 逗号分隔 IP / CIDR | 受信任代理列表 |
| `agfs.base_url` | `AGFS_BASE_URL` | `http://127.0.0.1:1833` | URL | AGFS 服务地址 |
| `agfs.mount_prefix` | `AGFS_MOUNT_PREFIX` | `/local` | 路径前缀 | AGFS 挂载前缀 |
| `index.interval` | `INDEX_INTERVAL` | `30` | 正整数，单位秒 | 索引轮询间隔 |
| `index.workers` | `INDEX_WORKERS` | `1` | 正整数 | 索引 worker 数量 |
| `identity.account_id` | `OG_ACCOUNT_ID` | `acct-demo` | 任意账户标识，如 `acct-company-a` | 租户标识 |
| `identity.user_id` | `OG_USER_ID` | `u-alice` | 任意用户标识，如 `u-bob` | 用户标识 |
| `identity.agent_id` | `OG_AGENT_ID` | `main` | 任意 Agent 标识，如 `shared-agent` | Agent 标识 |
| `auth.role_control_enabled` | `OG_ROLE_CONTROL_ENABLED` | `false` | `true` / `false` | 是否启用多租认证 |
| `auth.root_api_key` | `OG_ROOT_API_KEY` | 空 | API Key 字符串 | Root API Key |
| `auth.admin_api_keys` | `OG_ADMIN_API_KEYS` | 空 | `account_id:key` 或 `key`，多个值逗号分隔 | Admin API Keys 列表 |
| `sharing.agent_shared_mode` | `OG_AGENT_SHARED_MODE` | `off` | `off` / `user` | Agent 共享模式 |
| `sharing.agent_shared_list` | `OG_AGENT_SHARED_LIST` | 空 | 逗号分隔 Agent ID，例如 `shared-agent,agent-handbook` | 共享 Agent ID 列表 |
| `memory.after_turn_threshold` | `OGMEM_AFTER_TURN_THRESHOLD` | `200` | 正整数 | 达到阈值后触发记忆提取 |
| `memory.rolling_compress_enabled` | `OGMEM_ROLLING_COMPRESS_ENABLED` | `true` | `true` / `false` | 是否启用滚动压缩 |
| `memory.compact_prepare_token_ttl` | `OGMEM_COMPACT_PREPARE_TOKEN_TTL` | `300` | 正整数，单位秒 | prepare_compaction token 有效期 |
| `memory.directory_summary_enabled` | `OGMEM_DIRECTORY_SUMMARY_ENABLED` | `false` | `true` / `false` | 是否启用目录摘要 |
| `memory.prefetch_enabled` | `OGMEM_PREFETCH_ENABLED` | `false` | `true` / `false` | 是否启用 compose 前预取并填充 SessionTopicBuffer |
| `memory.prefetch_top_k` | `OGMEM_PREFETCH_TOP_K` | `5` | 正整数 | 预取写入 pending_injection 的最大命中数 |
| `memory.session_state_bridge_enabled` | `OGMEM_SESSION_STATE_BRIDGE_ENABLED` | `true` | `true` / `false` | 是否将 SessionState 同步到 SessionWindowState |
| `memory.session_state_sync_interval_turns` | `OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS` | `1` | 正整数 | SessionState 版本变化后至少间隔多少用户轮次同步一次 |
| `memory.topic_detection_enabled` | `OGMEM_TOPIC_DETECTION_ENABLED` | `false` | `true` / `false` | 是否启用 P2 话题检测 |
| `memory.compression_quality_enabled` | `OGMEM_COMPRESSION_QUALITY_ENABLED` | `false` | `true` / `false` | 是否计算 P2 archive 压缩质量指标 |
| `memory.compression_quality_persist_metadata` | `OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA` | `false` | `true` / `false` | 是否将压缩质量指标持久化到 archive metadata |
| `cache.enabled` | `OGMEM_CACHE_ENABLED` | `true` | `true` / `false` | 是否启用缓存 |
| `cache.max_size` | `OGMEM_CACHE_MAX_SIZE` | `1000` | 正整数 | 缓存最大条目数 |

> **说明：**
> 1. 布尔值建议统一写成 `true` / `false`。
> 2. 列表型变量通常使用逗号分隔。
> 3. 如果某个字段已经在 `ogmemory.yaml` 中写死为硬编码值，则该字段优先级高于环境变量。

#### `${ENV_VAR}` 引用规则

- `${VAR_NAME}` → 如果环境变量存在，替换为实际值
- `${VAR_NAME}` → 如果环境变量不存在，保持原样（方便排查缺失配置）
- 非 `${...}` 的硬编码值 → 不受影响，优先级最高

#### 推荐的去冗余写法

默认模板已经按下面的方式处理：

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"
  user_id: "${OG_USER_ID}"
  agent_id: "${OG_AGENT_ID}"

auth:
  root_api_key: "${OG_ROOT_API_KEY}"
```

对应地，你只需要在 `deploy.env` 中维护一份共享默认值：

```bash
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-demo"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"
```

这样：

- OpenClaw 默认通过 `OG_ROOT_API_KEY` / `OG_ACCOUNT_ID` 连接 oGMemory
- oGMemory 默认也通过同一组 `OG_*` 变量确定身份与认证
- 不需要在 `deploy.env` 和 `ogmemory.yaml` 中手工写两遍相同的账户和 Root Key

只有在以下场景下，才建议在 `ogmemory.yaml` 中单独覆盖：

- oGMemory 需要与 OpenClaw 使用不同的默认 `account_id`
- oGMemory 需要不同的 `user_id` 或 `agent_id`
- oGMemory 不想使用 Root Key，而是使用单独的认证配置

---

## 部署方式

### 方式一：完整部署（openGauss + ogmemory + openclaw-ogmemory）

适用于首次部署，需要向量数据库的场景。

```bash
bash deploy.sh -password <密码>
```

**密码要求：**
- 长度至少 8 个字符
- 包含大写字母、小写字母、数字、特殊符号(#?!@$%^&*)中的至少三种

**示例：**

```bash
bash deploy.sh -password "OpenGauss@2024"
```

### 方式二：仅部署服务（不使用 openGauss）

适用于已有外部数据库，或使用内存模式的场景。

1. 修改配置文件：

```bash
ENABLE_OPENGAUSS="false"
```

2. 执行部署：

```bash
bash deploy.sh
```

### 方式三：多租户部署

适用于多个租户共享同一套 oGMemory 服务，但各自拥有独立 OpenClaw Gateway 的场景。

1. 在 `ogmemory.yaml` 中启用多租认证：

```yaml
auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"
    - "acct-company-b:admin-key-for-b"
```

2. 在 `deploy.env` 中配置认证 Key 和多租户 OpenClaw 实例：

```bash
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ADMIN_API_KEYS="acct-company-a:admin-key-for-a,acct-company-b:admin-key-for-b"

OPENCLAW_INSTANCES="openclaw_tenant_a|18789|tenant-a-web-token|admin-key-for-a|acct-company-a|/data/openclaw-a;openclaw_tenant_b|18790|tenant-b-web-token|admin-key-for-b|acct-company-b|/data/openclaw-b"
```

**`OPENCLAW_INSTANCES` 格式说明：**

```
container_name|gateway_port|gateway_token|auth_api_key|auth_account_id|openclaw_home_dir
```

各字段含义：

| 字段 | 说明 |
|------|------|
| `container_name` | OpenClaw 容器名称 |
| `gateway_port` | Gateway 端口 |
| `gateway_token` | Web 界面配对密码 |
| `auth_api_key` | 连接 oGMemory 的 API Key（Admin Key 的 key 部分，或 Root Key） |
| `auth_account_id` | 绑定的账户 ID |
| `openclaw_home_dir` | OpenClaw 状态持久化目录（可选，留空则不挂载） |

多个实例用分号 `;` 分隔。

> **说明：** 配置了 `OPENCLAW_INSTANCES` 后，每个实例中的 `auth_api_key` / `auth_account_id` 优先级最高；单实例模式才会使用 `OG_AUTH_API_KEY` / `OG_AUTH_ACCOUNT_ID` 的默认回退逻辑。

3. 执行部署：

```bash
bash deploy.sh -password <密码>
```

---

## 多租认证与 Agent 共享场景详解

### 核心概念

| 概念 | 说明 |
|------|------|
| **Account（账户）** | 数据隔离的最小单位，不同 Account 的记忆数据完全隔离 |
| **User（用户）** | 同一 Account 下的用户，数据共享 |
| **Agent（智能体）** | 同一 Account 下的 Agent，数据共享 |
| **Agent 共享** | 将指定 Agent 的记忆对同 Account 下所有 User 可见 |
| **Root API Key** | 超级管理员密钥，可访问所有 Account 的数据 |
| **Admin API Key** | 账户管理员密钥，只能访问绑定的 Account 数据 |

### 场景一：同 Account + 不同 User + Agent 共享

**需求**：一个公司（`acct-company-a`）有多个员工（`u-alice`、`u-bob`），他们共享同一个 Agent（`shared-agent`）的记忆，同时各自也有私有 Agent。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"    # acct-company-a
  user_id: "${OG_USER_ID}"          # 默认用户，OpenClaw 可覆盖
  agent_id: "${OG_AGENT_ID}"        # 默认 Agent

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"

sharing:
  agent_shared_mode: "user"          # 启用用户级共享
  agent_shared_list:
    - "shared-agent"                 # shared-agent 对所有 MEMBER 用户可见
```

**deploy.env 配置：**

```bash
# 共享 LLM 配置
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"

# 多租认证
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-company-a"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"

# 两个员工各一个 OpenClaw Gateway，共享同一个 Account
OPENCLAW_INSTANCES="openclaw_alice|18789|alice-web-token|admin-key-for-a|acct-company-a|/data/openclaw-alice;openclaw_bob|18790|bob-web-token|admin-key-for-a|acct-company-a|/data/openclaw-bob"
```

**数据隔离效果：**

| 用户 | 可见的 Agent | 可见的记忆 |
|------|-------------|-----------|
| u-alice（通过 openclaw_alice） | `main`（私有）+ `shared-agent`（共享） | 自己的 + 共享的 |
| u-bob（通过 openclaw_bob） | `main`（私有）+ `shared-agent`（共享） | 自己的 + 共享的 |

> **说明：** 两个 OpenClaw Gateway 使用同一个 Admin Key（`admin-key-for-a`）和同一个 Account（`acct-company-a`），但各自以不同 User 身份访问 oGMemory。`shared-agent` 的记忆对两个用户都可见，而各自的 `main` Agent 记忆互相隔离。

### 场景二：不同 Account + Agent 共享

**需求**：两个公司（`acct-company-a`、`acct-company-b`）各自独立，数据完全隔离，每个公司有自己的 Admin Key。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"
  user_id: "${OG_USER_ID}"
  agent_id: "${OG_AGENT_ID}"

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"
    - "acct-company-b:admin-key-for-b"

sharing:
  agent_shared_mode: "user"
  agent_shared_list:
    - "shared-agent"
```

**deploy.env 配置：**

```bash
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"

OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-company-a"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"

# 两个公司各一个 OpenClaw Gateway，各自使用自己的 Admin Key
OPENCLAW_INSTANCES="openclaw_company_a|18789|company-a-web-token|admin-key-for-a|acct-company-a|/data/openclaw-a;openclaw_company_b|18790|company-b-web-token|admin-key-for-b|acct-company-b|/data/openclaw-b"
```

**数据隔离效果：**

| 公司 | OpenClaw | 认证 | 可见数据 |
|------|---------|------|---------|
| acct-company-a | openclaw_company_a:18789 | admin-key-for-a | 仅 company-a 的记忆 |
| acct-company-b | openclaw_company_b:18790 | admin-key-for-b | 仅 company-b 的记忆 |

> **说明：** 两个公司使用不同的 Admin Key 和不同的 Account，数据完全隔离。每个公司内部的 `shared-agent` 仅对本公司的用户可见。

### 场景三：单租户（无认证，最简配置）

**需求**：个人使用或内网环境，不需要多租隔离。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

auth:
  role_control_enabled: false

sharing:
  agent_shared_mode: "off"
```

**deploy.env 配置：**

```bash
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"
```

> **说明：** 无认证模式下，所有请求默认使用 `identity` 中配置的账户，无需 API Key。

### 关于 `openclaw.json` 的自动迁移

如果设置了 `OPENCLAW_HOME_DIR`，`openclaw.json` 会持久化保存。当前版本的容器入口脚本会这样处理：

1. 如果 `openclaw.json` 不存在，按当前模板直接生成。
2. 如果文件中仍残留模板占位符，也会自动重建。
3. 如果文件缺少关键字段，例如 `authApiKey`、`authAccountId`，会先备份旧文件，再按当前模板重建。
4. 自动备份最多保留最近 `3` 份，超过后会清理更旧的备份。

备份文件名示例：

```text
openclaw.json.bak.20260420-101530
```

如果你曾手工修改过 `openclaw.json`，升级后请优先对比备份文件，再决定是否手工迁移你的自定义内容。

---

## 配置修改与生效

### 修改 oGMemory 配置

1. 编辑 `deploy/ogmemory.yaml`
2. 重启容器：

```bash
docker restart ogmem
```

> **💡 提示：** `ogmemory.yaml` 直接挂载到容器中（只读），容器启动时 entrypoint 会自动完成 `${...}` 替换和 openGauss 连接信息替换。修改后只需 `docker restart` 即可生效，无需重新运行 `deploy.sh`。

### 修改 OpenClaw 或基础设施配置

1. 编辑 `deploy/deploy.env`
2. 重新部署：

```bash
bash deploy.sh
```

> **说明：** 如果只修改了 `OG_AUTH_API_KEY`、`OG_AUTH_ACCOUNT_ID`、`OG_ROOT_API_KEY`、`OG_ACCOUNT_ID` 这类 OpenClaw 连接参数，也建议重新执行 `bash deploy.sh`，让容器以新的环境变量启动，并在需要时触发 `openclaw.json` 自动迁移。

### 配置生效流程

```
ogmemory.yaml (用户编辑)
    │
    ▼ docker run -v ogmemory.yaml:/etc/ogmem/config.yaml:ro
    │
    ▼ 容器 entrypoint 自动执行：
    │   1. 读取 /etc/ogmem/config.yaml
    │   2. 替换 ${LLM_API_KEY} 等环境变量引用
    │   3. 替换 vector_db.connection_string（如有 OPENGAUSS_* 环境变量）
    │   4. 写入 /tmp/ogmem.resolved.yaml
    │   5. 设置 OGMEM_CONFIG=/tmp/ogmem.resolved.yaml
    │   6. 启动应用
    │
    ▼ 应用读取 /tmp/ogmem.resolved.yaml
```

---

## 部署过程说明

部署脚本会依次执行以下步骤：

1. **前置检查**: 检查 Docker 环境
2. **拉取镜像**: 从华为云 SWR 拉取所需镜像
3. **部署 openGauss**: 启动数据库容器（如启用）
4. **等待数据库就绪**: 最长等待 120 秒
5. **部署 oGMemory**: 挂载 `ogmemory.yaml`，传入环境变量，启动容器
6. **部署 OpenClaw Gateway**: 启动 Web 服务（单实例或多实例）
7. **健康检查**: 验证服务是否正常

---

## 常用命令

### 查看部署状态

```bash
bash deploy.sh --status
```

> `--status` 只查看当前目录下这份 `deploy.env` 解析出来的容器名，不会再按全局标签扫描同机上的其它部署实例。

### 清理部署

```bash
bash deploy.sh --cleanup
```

> `--cleanup` 只会删除当前目录下这份 `deploy.env` 中声明的容器名，例如 `OG_CONTAINER_NAME`、`OGMEM_CONTAINER_NAME` 和 `OPENCLAW_INSTANCES` / `OPENCLAW_CONTAINER_NAME` 对应的容器。
> 这样同一台机器上可以并存多套部署；在某一套目录里执行 cleanup，不会误删另一套环境的容器。数据仍会保留在你配置的持久化目录中。
> 这里说的“所有容器”是指当前这份 `deploy.env` 对应的全部容器，不是宿主机上所有带 `ogmem.managed=true` 标签的容器。

> 注意：此命令会停止并删除所有容器，数据不会丢失（如果配置了持久化目录）

### 查看日志

```bash
# 查看 oGMemory 日志
docker logs ogmem -f

# 查看 OpenClaw 日志
docker logs openclaw_ogmem -f

# 查看 openGauss 日志
docker logs opengauss -f
```

### 重启服务

```bash
# 重启所有服务
docker restart ogmem openclaw_ogmem

# 重启单个服务
docker restart ogmem
```

### 进入容器

```bash
# 进入 oGMemory 容器
docker exec -it ogmem bash

# 进入 openGauss 容器
docker exec -it opengauss bash
```

### 健康检查

```bash
# 检查 oGMemory 服务
curl http://<OPENCLAW_HOST_IP>:8090/api/v1/health

# 检查 OpenClaw 服务
curl http://<OPENCLAW_HOST_IP>:18789/health
```

### 查看 OpenClaw 当前配置及备份

```bash
ls -lt <OPENCLAW_HOME_DIR>/openclaw.json*
```

---

## 访问服务

部署成功后，通过以下地址访问：

### 单租户模式

| 服务 | 地址 | 说明 |
|------|------|------|
| OpenClaw Web | http://\<OPENCLAW_HOST_IP\>:18789 | Web 界面入口 |
| oGMemory API | http://\<OPENCLAW_HOST_IP\>:8090 | HTTP API |
| openGauss | \<OPENGAUSS_HOST_IP\>:15432 | 数据库连接 |

### 多租户模式

每个租户拥有独立的 OpenClaw Gateway 端口：

| 租户 | OpenClaw Web | 绑定账户 |
|------|-------------|---------|
| 租户 A | http://\<OPENCLAW_HOST_IP\>:18789 | acct-company-a |
| 租户 B | http://\<OPENCLAW_HOST_IP\>:18790 | acct-company-b |

### 首次访问

1. 进入 openclaw 容器内
2. 输入 `openclaw tui` 进行交互对话

---

## 文件说明

```
deploy/
├── README.md                # 本文档
├── install.sh               # 一键安装脚本（下载部署文件）
├── deploy.sh                # 部署执行脚本
├── deploy.env               # OpenClaw + 基础设施配置
├── ogmemory.example.yaml    # oGMemory 配置模板（复制为 ogmemory.yaml 后编辑）
└── ogmemory.yaml            # oGMemory 实际配置（用户创建，不纳入版本控制）
```

## Agent 工具与经验记忆

部署模板默认关闭 Agent 记忆抽取。需要启用时，按 [Agent 记忆部署说明](agent-memory.md) 修改挂载 YAML、重启服务并验证配置和 Trace 抽取结果。
