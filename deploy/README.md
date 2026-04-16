# OpenClaw + oG-Memory 一键安装部署指南

## 概述

本部署方案采用 Docker 容器化部署，包含以下组件：

| 组件 | 容器名称 | 端口 | 说明 |
|------|---------|------|------|
| openGauss | opengauss | 15432 | 向量数据库（可选） |
| oG-Memory | ogmem | 8090 | 记忆服务 HTTP API |
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

### 远程一键安装

```bash
curl -fsSL https://raw.gitcode.com/opengauss/oG-Memory/raw/dev/deploy/install.sh | bash
```

执行后会下载以下文件到 `./ogmem-deploy/` 目录：
- `deploy.env`: 配置文件
- `deploy.sh`: 部署脚本

## 配置说明

### 第一步：编辑配置文件

```bash
cd ogmem-deploy  # 或 cd oG-Memory/deploy
vim deploy.env   # 或使用其他编辑器
```

### 环境变量列表

> **💡 重要提示：**
> 1. OpenClaw 和 oG-Memory 的大模型相关环境变量可以共用。`LLM_PROVIDER`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 这四个变量同时供两个服务使用。如果 oG-Memory 需要使用不同的大模型配置，可通过 `OGMEM_API_KEY`、`OGMEM_BASE_URL`、`OGMEM_LLM_MODEL`、`CONTEXTENGINE_PROVIDER` 单独覆盖，不设置则自动复用 OpenClaw 的配置。
> 2. oG-Memory 需要额外配置 `OGMEM_EMBEDDING_MODEL`（Embedding 模型名称），该变量为 oG-Memory 必填项，OpenClaw 不使用此变量。
> 3. `OPENCLAW_HOST_IP` 建议填写宿主机实际 IP 地址（而非 `127.0.0.1`），以便其他机器或容器能够访问 OpenClaw 和 oG-Memory 服务。`OPENGAUSS_HOST_IP` 通常保持 `127.0.0.1` 即可（数据库与 oG-Memory 在同一主机时）。

#### 必填环境变量

| 变量名 | 说明 | 示例值 | 适用服务 |
|--------|------|--------|---------|
| `LLM_PROVIDER` | LLM 提供商标识 | `volcengine` | OpenClaw + oG-Memory 共用 |
| `LLM_API_KEY` | LLM API 密钥 | `xxxxx-xxxx-xxxx` | OpenClaw + oG-Memory 共用 |
| `LLM_BASE_URL` | LLM API 基础地址 | `https://ark.cn-beijing.volces.com/api/v3` | OpenClaw + oG-Memory 共用 |
| `LLM_MODEL` | LLM 模型名称 | `doubao-seed-2-0-code-preview-260215` | OpenClaw + oG-Memory 共用 |
| `OGMEM_EMBEDDING_MODEL` | Embedding 模型名称 | `doubao-embedding-vision-250615` | oG-Memory |
| `ENABLE_OPENGAUSS` | 是否启用 openGauss | `true` / `false` | openGauss |
| `OG_HOST_PORT` | openGauss 宿主机端口（启用 openGauss 时必填） | `15432` | openGauss |
| `OPENGAUSS_HOST_IP` | openGauss 数据库宿主机 IP 地址 | `127.0.0.1` | openGauss |
| `OPENCLAW_HOST_IP` | OpenClaw 宿主机 IP 地址（建议填写宿主机实际 IP） | `127.0.0.1` | OpenClaw + oG-Memory |
| `AGFS_DATA_DIR` | AGFS 数据存储目录，用于持久化记忆文件 | `/home/test/ogmemory/agfs-auto` | oG-Memory |
| `OPENCLAW_HOME_DIR` | OpenClaw 状态目录，避免重复配对 | `/home/test/ogmemory/openclaw-auto` | OpenClaw |
| `OPENGAUSS_DIMENSION` | 向量维度（需与 Embedding 模型匹配） | `1536` | oG-Memory |

#### 可选环境变量 — oG-Memory 独立大模型配置

> 以下变量不设置时，oG-Memory 自动复用上方 `LLM_*` 共用变量。仅在需要 oG-Memory 使用不同大模型时配置。

| 变量名 | 说明 | 默认值（回退） | 适用服务 |
|--------|------|---------------|---------|
| `OGMEM_API_KEY` | oG-Memory 专用 API 密钥 | 回退至 `LLM_API_KEY` | oG-Memory |
| `OGMEM_BASE_URL` | oG-Memory 专用 API 地址 | 回退至 `LLM_BASE_URL` | oG-Memory |
| `OGMEM_LLM_MODEL` | oG-Memory 专用模型名称 | 回退至 `LLM_MODEL` | oG-Memory |
| `CONTEXTENGINE_PROVIDER` | oG-Memory LLM 提供商 | 回退至 `LLM_PROVIDER` | oG-Memory |

#### 可选环境变量 — Embedding 配置

| 变量名 | 说明 | 默认值（回退） | 适用服务 |
|--------|------|---------------|---------|
| `EMBEDDING_PROVIDER` | Embedding 提供商 | 回退至 `LLM_PROVIDER` | oG-Memory |
| `OGMEM_EMBEDDING_BASE_URL` | Embedding API 地址 | 回退至 `LLM_BASE_URL` | oG-Memory |
| `OGMEM_EMBEDDING_API_KEY` | Embedding API 密钥 | 回退至 `LLM_API_KEY` | oG-Memory |
| `OGMEM_EMBEDDING_MULTIMODAL` | 是否使用多模态 Embedding | `false` | oG-Memory |
| `ST_MODEL` | 本地 Sentence Transformer 模型 | `BAAI/bge-large-en-v1.5` | oG-Memory |

#### 可选环境变量 — 向量数据库配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `VECTOR_DB_TYPE` | 向量数据库类型 | `opengauss` | oG-Memory |
| `OPENGAUSS_TABLE_NAME` | 向量表名 | `vector_index` | oG-Memory |
| `OPENGAUSS_POOL_SIZE` | 连接池大小 | `5` | oG-Memory |

#### 可选环境变量 — 服务与容器配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OGMEM_HTTP_PORT` | oG-Memory HTTP 端口 | `8090` | oG-Memory |
| `OGMEM_WORKERS` | oG-Memory 工作进程数 | `2` | oG-Memory |
| `GATEWAY_PORT` | OpenClaw Gateway 端口 | `18789` | OpenClaw |
| `OPENCLAW_GATEWAY_TOKEN` | 前端配对密码 | 无（不需要 Token） | OpenClaw |
| `OPENCLAW_BIND_MODE` | 绑定模式（`lan` / `localhost`） | `lan` | OpenClaw |
| `OGMEM_URL` | oG-Memory API 地址（供 OpenClaw 连接） | `http://127.0.0.1:8090` | OpenClaw |
| `OGMEM_CONTAINER_NAME` | ogmemory 容器名称 | `ogmem` | 容器 |
| `OPENCLAW_CONTAINER_NAME` | openclaw-ogmemory 容器名称 | `openclaw_ogmem` | 容器 |
| `OGMEM_IMAGE` | ogmemory 镜像地址 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:v0.8` | 容器 |
| `OPENCLAW_IMAGE` | openclaw-ogmemory 镜像地址 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmemory:v0.2` | 容器 |

#### 可选环境变量 — openGauss 容器配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OG_CONTAINER_NAME` | openGauss 容器名称 | `opengauss` | openGauss |
| `OG_PORT` | openGauss 容器内端口 | `5432` | openGauss |
| `OG_NODE_NAME` | openGauss 节点名称 | `gaussdb` | openGauss |
| `OG_USERNAME` | openGauss 用户名 | `gaussdb` | openGauss |
| `OG_DB_NAME` | openGauss 数据库名 | `postgres` | openGauss |
| `OG_IMAGE_REPO` | openGauss 镜像仓库 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed` | openGauss |
| `OG_IMAGE_TAG` | openGauss 镜像标签 | `0328` | openGauss |
| `OG_CPUSET_CPUS` | CPU 亲和性 | 无 | openGauss |
| `OG_WAIT_TIMEOUT` | 启动超时时间（秒） | `120` | openGauss |

#### 可选环境变量 — 其他

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `LLM_TEMPERATURE` | LLM 温度参数（0.0-2.0） | `0.7` | oG-Memory |
| `LLM_MAX_TOKENS` | LLM 最大 Token 数 | `4096` | oG-Memory |
| `INDEX_INTERVAL` | 索引轮询间隔（秒） | `15` | oG-Memory |
| `INDEX_WORKERS` | 索引服务工作进程数 | `1` | oG-Memory |
| `OG_ACCOUNT_ID` | 租户标识 | `acct-demo` | oG-Memory |
| `OG_USER_ID` | 用户标识 | `u-alice` | oG-Memory |
| `OG_AGENT_ID` | Agent 标识 | `main` | oG-Memory |
| `OGMEM_CACHE_ENABLED` | 是否启用缓存 | `true` | oG-Memory |
| `OGMEM_CACHE_MAX_SIZE` | 缓存最大大小 | `1000` | oG-Memory |
| `AUTO_HEALTH_CHECK` | 是否自动健康检查 | `true` | 部署控制 |
| `SKIP_PULL` | 是否跳过镜像拉取 | `false` | 部署控制 |

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

### 部署过程说明

部署脚本会依次执行以下步骤：

1. **前置检查**: 检查 Docker 环境
2. **拉取镜像**: 从华为云 SWR 拉取所需镜像
3. **部署 openGauss**: 启动数据库容器（如启用）
4. **等待数据库就绪**: 最长等待 120 秒
5. **部署 oG-Memory**: 启动记忆服务
6. **部署 OpenClaw Gateway**: 启动 Web 服务
7. **健康检查**: 验证服务是否正常

---

## 常用命令

### 查看部署状态

```bash
bash deploy.sh --status
```

### 清理部署

```bash
bash deploy.sh --cleanup
```

> 注意：此命令会停止并删除所有容器，数据不会丢失（如果配置了持久化目录）

### 查看日志

```bash
# 查看 oG-Memory 日志
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
# 进入 oG-Memory 容器
docker exec -it ogmem bash

# 进入 openGauss 容器
docker exec -it opengauss bash
```

### 健康检查

```bash
# 检查 oG-Memory 服务
curl http://<OPENCLAW_HOST_IP>:8090/api/v1/health

# 检查 OpenClaw 服务
curl http://<OPENCLAW_HOST_IP>:18789/health
```

---

## 访问服务

部署成功后，通过以下地址访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| OpenClaw Web | http://\<OPENCLAW_HOST_IP\>:18789 | Web 界面入口 |
| oG-Memory API | http://\<OPENCLAW_HOST_IP\>:8090 | HTTP API |
| openGauss | \<OPENGAUSS_HOST_IP\>:15432 | 数据库连接 |

### 首次访问

1. 进入openclaw容器内
2. 输入`openclaw tui`进行交互对话

---

## 文件说明

```
deploy/
├── README.md       # 本文档
├── install.sh      # 一键安装脚本（下载部署文件）
├── deploy.sh       # 部署执行脚本
└── deploy.env      # 配置文件
```

---
