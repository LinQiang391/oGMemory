# oG-Memory 本地部署与 perf 监控完整指南

> 本文档覆盖从零开始的完整操作流程：Docker 一键部署 → 构建含 perf 的镜像 → 启用 perf 监控 → locomo10 基准测试 → 日志分析。  
> 单 worker / 多 worker、Docker 部署 / 本地部署、macOS / Linux 差异均在对应章节内说明。

---

## 目录

1. [架构概览](#1-架构概览)
2. [Docker 一键部署（基线）](#2-docker-一键部署基线)
3. [重新构建含 perf 的 Docker 镜像](#3-重新构建含-perf-的-docker-镜像)
4. [Docker 部署 + perf 启用](#4-docker-部署--perf-启用)
5. [本地部署（macOS / Linux）](#5-本地部署macos--linux)
6. [locomo10 基准测试](#6-locomo10-基准测试)
7. [perf 日志分析与报告](#7-perf-日志分析与报告)
8. [环境变量速查表](#8-环境变量速查表)

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  测试客户端 (eval.py)                                    │
│  POST http://localhost:18789/v1/responses                │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   OpenClaw Gateway  │  :18789
              │   (openclaw-ogmem)  │
              └──────────┬──────────┘
                         │ HTTP
              ┌──────────▼──────────┐
              │   oG-Memory HTTP    │  :8090
              │   (ogmemory)        │
              │   ┌──────────────┐  │
              │   │ perf 模块    │  │  ← OGMEM_PERF_ENABLED=1
              │   │ TokenTracker │  │  ← 记录每阶段 token/延迟
              │   └──────────────┘  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   openGauss / AGFS  │
              └─────────────────────┘
```

**perf 数据来源**：每次请求（compose / after_turn / compact 等）前后，分别对 `service._llm.token_tracker` 和 `service._embedder.token_tracker` 进行快照，差值即为本次请求消耗的 token。结果以 JSONL 格式写入日志文件。

---

## 2. Docker 一键部署（基线）

> 默认镜像（poc1_428）**不含** perf 模块，适合验证基础功能。  
> 如需启用 perf，请直接跳至 [第 3 节](#3-重新构建含-perf-的-docker-镜像)。

### 2.1 前提条件

| 工具 | 最低版本 | 安装 |
|------|----------|------|
| Docker | 20.10+ | `docker --version` |
| bash | 任意 | 内置 |
| curl | 任意 | `curl --version` |

确认 Docker 已启动：

```bash
docker info
```

### 2.2 准备配置

```bash
cd deploy/

# 1. 复制并编辑 deploy.env
cp deploy.env.example deploy.env   # 如有模板；否则直接编辑 deploy.env
vi deploy.env
```

**必填项：**

```bash
# deploy.env 关键字段
OPENGAUSS_HOST_IP="<宿主机 IP>"        # openGauss 绑定的宿主机 IP
OPENCLAW_HOST_IP="<宿主机 IP>"         # 服务对外暴露的 IP

LLM_PROVIDER="openai"                  # 或 dashscope / volcengine / zhipu
LLM_API_KEY="sk-..."
LLM_BASE_URL="https://api.openai.com/v1"
LLM_MODEL="gpt-4o-mini"
```

复制默认 ogmemory 配置：

```bash
cp ogmemory.example.yaml ogmemory.yaml
# 如需定制 LLM / embedding / 向量库，编辑 ogmemory.yaml
```

### 2.3 执行部署

```bash
# 启用 openGauss（默认）
bash deploy.sh -password <openGauss密码>

# 或仅部署 ogmem + openclaw，不启用 openGauss：
# ENABLE_OPENGAUSS=false 写入 deploy.env 后
bash deploy.sh
```

部署成功后，三个容器均运行：

```bash
bash deploy.sh --status
```

健康检查：

```bash
curl http://127.0.0.1:8090/api/v1/health
# 期望: {"status":"ok",...}
```

---

## 3. 重新构建含 perf 的 Docker 镜像

默认镜像不含 `perf/` 目录，`server/app.py` 在 import 时会硬失败。需从本仓库重新构建：

### 3.1 确认 Dockerfile 已包含 perf

```bash
grep "COPY perf" docker/Dockerfile.standalone
# 期望输出: COPY perf/       /opt/ogmem/perf/
```

### 3.2 构建镜像

在仓库根目录执行（需要 Docker 可访问阿里云镜像源，或修改 Dockerfile 中的 pip 镜像地址）：

```bash
docker build \
  -f docker/Dockerfile.standalone \
  -t ogmemory:local \
  .
```

> **Linux 离线/内网环境**：如无法访问 `mirrors.aliyun.com`，修改 `Dockerfile.standalone` 第 13 行的 pip 参数为内网镜像或去掉 `-i` 参数。

构建时间约 3–5 分钟（首次下载依赖）。完成后验证：

```bash
docker run --rm ogmemory:local python3 -c "from perf import get_recorder; print('perf OK')"
# 期望: perf OK
```

### 3.3 更新 deploy.env 指向本地镜像

```bash
# deploy.env
OGMEM_IMAGE="ogmemory:local"
SKIP_PULL="true"               # 跳过从远程 registry 拉取
```

---

## 4. Docker 部署 + perf 启用

### 4.1 单 worker（推荐用于 perf 分析）

单 worker 时所有请求写入同一个 JSONL 文件，`perf.report` 可直接生成报告。

在 `deploy.env` 中追加：

```bash
# perf 配置
OGMEM_WORKERS="1"
OGMEM_PERF_ENABLED="1"
OGMEM_PERF_OUT="/opt/ogmem/perf_logs/perf.jsonl"

# 将容器内日志目录挂载到宿主机，避免容器销毁后日志丢失
PERF_LOGS_HOST_DIR="/home/user/ogmem/perf_logs"   # 改为实际宿主机路径
```

然后重新部署：

```bash
bash deploy.sh --cleanup
bash deploy.sh -password <密码>
```

验证 perf 已启用：

```bash
# 打一个请求，触发 service 懒加载和 recorder 初始化
curl -s http://127.0.0.1:8090/api/v1/health

# 查看 perf 录制状态
curl -s http://127.0.0.1:8090/api/v1/perf/events | python3 -m json.tool | head -20
# 期望: "enabled": true

# 确认日志文件已创建（宿主机路径）
ls -lh /home/user/ogmem/perf_logs/
```

### 4.2 多 worker（worker=4）

多 worker 时每个 worker 进程有独立 `run_id`，日志分散到多个文件。**per-request 的 token 归因仍然准确**（每条 SpanEvent 是同一进程内的前后快照差），但需要聚合才能得到全局统计。

```bash
# deploy.env
OGMEM_WORKERS="4"
OGMEM_PERF_ENABLED="1"
OGMEM_PERF_OUT="/opt/ogmem/perf_logs/perf.jsonl"   # 每个 worker 会加 run_id 后缀
PERF_LOGS_HOST_DIR="/home/user/ogmem/perf_logs"
```

> 注意：`OGMEM_PERF_OUT` 只在 `OGMEM_PERF_RUN_ID` 被显式设置时才固定文件名。未设置时，每个 worker 生成独立 UUID 作为 `run_id`，日志文件名为 `perf_logs/<uuid>.jsonl`。  
> 若希望所有 worker 写入同一文件，显式设置：
> ```bash
> OGMEM_PERF_RUN_ID="locomo-run-20260519"
> OGMEM_PERF_OUT="/opt/ogmem/perf_logs/locomo-run-20260519.jsonl"
> ```
> 此时多个 worker 追加写同一文件，单行 JSON 写入在 Linux 上 ≤4096B 时是原子的，不会交叉。

重新部署后，确认 4 个 worker 日志文件（或 1 个共享文件）均有输出：

```bash
ls -lh /home/user/ogmem/perf_logs/
```

---

## 5. 本地部署（macOS / Linux）

> 本节适用于直接在宿主机运行 oG-Memory，不使用 Docker。  
> locomo10 测试仍需 OpenClaw（可用 Docker 容器运行，或独立安装）。

### 5.1 macOS 与 Linux 差异对照

| 项目 | macOS | Linux（x86_64）|
|------|-------|---------------|
| Python 安装 | `brew install python@3.11` 或系统自带 | `apt install python3.11` / `yum install python3.11` |
| uv 安装 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | 同左，无差异 |
| AGFS 二进制 | `agfs/agfs-server`（arm64 或 x86_64，取决于构建） | 同目录，x86_64 版本 |
| 进程管理 | 手动或 `launchd` | 手动或 `systemd` |
| 路径分隔符 | `/` | `/` |
| 本质差异 | 几乎无 | 几乎无 |

**结论**：本节所有命令在 macOS 和 Linux 上完全相同，无需分支。唯一需要确认的是 Python 版本和 AGFS 二进制架构。

### 5.2 安装依赖

**安装 uv（如未安装）：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 或重新打开 terminal
uv --version
```

**安装项目依赖：**

```bash
cd /path/to/oG-Memory   # 仓库根目录
uv sync
```

**确认 AGFS 二进制可执行：**

```bash
# macOS
file agfs/agfs-server
chmod +x agfs/agfs-server

# Linux x86_64 — 确认架构匹配
file agfs/agfs-server
# 应输出: ELF 64-bit LSB executable, x86-64
```

### 5.3 准备配置

```bash
# 复制示例配置
cp deploy/ogmemory.example.yaml config/ogmem.yaml

# 编辑配置（至少填写 llm / embedding / vector_db）
vi config/ogmem.yaml
```

最小化配置示例：

```yaml
# config/ogmem.yaml
llm:
  provider: openai
  api_key: "${LLM_API_KEY}"          # 从环境变量读取
  base_url: "${LLM_BASE_URL}"
  model: gpt-4o-mini

embedding:
  provider: openai
  model: text-embedding-3-small
  base_url: "${LLM_BASE_URL}"
  api_key: "${LLM_API_KEY}"

vector_db:
  type: opengauss                     # 或 chroma（无需 PG）
  connection_string: "host=127.0.0.1 port=5432 dbname=postgres user=gaussdb password=CHANGE_ME"

service:
  http_port: 8090
  workers: 1                          # 启用 perf 时先用 1
```

设置环境变量：

```bash
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.openai.com/v1"
```

### 5.4 启用 perf 并启动服务

#### 方式一：通过环境变量（推荐，最简单）

```bash
# 设置 perf 环境变量
export OGMEM_PERF_ENABLED="1"
export OGMEM_PERF_OUT="$(pwd)/perf_logs/perf.jsonl"

# 启动 oG-Memory（单 worker）
uv run ogmem start
```

#### 方式二：通过 ogmem.yaml 配置

在 `config/ogmem.yaml` 末尾添加：

```yaml
perf:
  enabled: true
  file_path: "/absolute/path/to/perf_logs/perf.jsonl"
```

然后正常启动：

```bash
uv run ogmem start
```

#### 方式三：直接用 gunicorn（更灵活，适合多 worker）

```bash
# 单 worker（perf 分析首选）
OGMEM_PERF_ENABLED=1 \
OGMEM_PERF_OUT="$(pwd)/perf_logs/perf.jsonl" \
OGMEM_CONFIG=config/ogmem.yaml \
PYTHONPATH=. \
uv run gunicorn \
  --bind 0.0.0.0:8090 \
  --workers 1 \
  --timeout 900 \
  server.app:app

# 多 worker（4 workers），每个 worker 独立日志文件
OGMEM_PERF_ENABLED=1 \
OGMEM_PERF_RUN_ID="locomo-run-$(date +%Y%m%d)" \
OGMEM_PERF_OUT="$(pwd)/perf_logs/locomo-run-$(date +%Y%m%d).jsonl" \
OGMEM_CONFIG=config/ogmem.yaml \
PYTHONPATH=. \
uv run gunicorn \
  --bind 0.0.0.0:8090 \
  --workers 4 \
  --timeout 900 \
  server.app:app
```

### 5.5 验证本地服务

```bash
# 健康检查
curl http://127.0.0.1:8090/api/v1/health

# 确认 perf 已启用（发送一条请求触发 recorder 初始化后检查）
curl http://127.0.0.1:8090/api/v1/perf/events | python3 -m json.tool | grep enabled
# 期望: "enabled": true

# 检查日志文件（首次请求后）
ls -lh perf_logs/
```

### 5.6 本地 OpenClaw 配置（locomo10 测试所需）

locomo10 的 `eval.py` 通过 OpenClaw Gateway（`:18789`）而非直接访问 ogmemory（`:8090`）。本地测试需要一个 OpenClaw 实例指向本地 ogmemory。

**最简方式：用 Docker 运行 OpenClaw 容器**

```bash
docker run -d --name openclaw_local \
  --network host \
  -e LLM_PROVIDER=openai \
  -e LLM_API_KEY="${LLM_API_KEY}" \
  -e LLM_BASE_URL="${LLM_BASE_URL}" \
  -e LLM_MODEL=gpt-4o-mini \
  -e GATEWAY_PORT=18789 \
  -e OGMEM_URL=http://127.0.0.1:8090 \
  -e OPENCLAW_GATEWAY_TOKEN=local-test-token \
  -e OPENCLAW_BIND_MODE=localhost \
  -e no_proxy='*' \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmemory:poc1_428

# 等待就绪
sleep 5
curl http://127.0.0.1:18789/health
```

---

## 6. locomo10 基准测试

以下步骤适用于 **所有部署方式**（Docker / 本地，单 / 多 worker）。唯一差异在测试前的服务启动方式（已在前面各节说明）。

### 6.1 前提检查

```bash
# oG-Memory 健康
curl -s http://127.0.0.1:8090/api/v1/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('status')=='ok' else 'FAIL:', d)"

# OpenClaw 健康
curl -s http://127.0.0.1:18789/health

# perf 已启用（Docker 部署需发送一个请求触发初始化）
curl -s http://127.0.0.1:8090/api/v1/perf/events | python3 -c "import sys,json; d=json.load(sys.stdin); print('perf enabled:', d['enabled'])"
```

### 6.2 完整测试流程（run_benchmark.sh）

```bash
cd tests/e2e

# 设置 OpenClaw Gateway Token（与部署时一致）
export OPENCLAW_GATEWAY_TOKEN="local-test-token"   # 或 ogmem-default-token

# 完整流程：清理 → ingest → qa → judge → report
PARALLEL=2 bash run_benchmark.sh

# 跳过清理，在现有数据上继续（恢复中断的测试）
PARALLEL=2 bash run_benchmark.sh --no-clean

# 仅 QA + judge（数据已 ingest）
PARALLEL=2 bash run_benchmark.sh --qa-only
```

**`PARALLEL` 含义**：同时并发的 QA 问题数，不影响 ogmemory 的 worker 数量。建议值：

| ogmemory workers | PARALLEL 推荐值 | 说明 |
|-----------------|-----------------|------|
| 1 | 2–4 | 单 worker 串行处理每条请求，PARALLEL 影响不大 |
| 4 | 4–8 | 充分利用 4 个 worker |

### 6.3 手动分步执行

如需精细控制，分步执行：

```bash
cd tests/e2e

# 第一步：Ingest（将 10 个对话写入 ogmemory）
uv run python eval.py ingest data/locomo10.json \
  --base-url http://127.0.0.1:18789 \
  --token "${OPENCLAW_GATEWAY_TOKEN}" \
  --user eval-1

# 第二步：QA（向 OpenClaw 提问，收集回答）
mkdir -p result/run1
uv run python eval.py qa data/locomo10.json \
  --base-url http://127.0.0.1:18789 \
  --token "${OPENCLAW_GATEWAY_TOKEN}" \
  --user eval-1 \
  --output result/run1/qa \
  --parallel 2

# 第三步：Judge（用 LLM 给回答打分）
uv run python eval.py judge result/run1/qa.csv \
  --output result/run1/qa_judged

# 第四步：统计
uv run python eval.py stat result/run1/qa_judged.csv
```

### 6.4 测试运行时间参考

| 阶段 | locomo10（10 样本）| 耗时估算 |
|------|-------------------|----|
| ingest | 10 对话 × ~5 session | 10–20 分钟 |
| qa | ~130 题 × 2 并发 | 20–40 分钟 |
| judge | ~130 题 | 5–10 分钟 |

---

## 7. perf 日志分析与报告

### 7.1 日志位置

| 部署方式 | 日志路径 |
|----------|---------|
| Docker 单 worker（设了 `PERF_LOGS_HOST_DIR`）| `$PERF_LOGS_HOST_DIR/perf.jsonl` |
| Docker 多 worker（设了 `OGMEM_PERF_RUN_ID`）| `$PERF_LOGS_HOST_DIR/<run_id>.jsonl` |
| Docker（未挂载卷）| `docker exec ogmem ls /opt/ogmem/perf_logs/` |
| 本地 gunicorn | `./perf_logs/perf.jsonl`（或 `./perf_logs/<uuid>.jsonl`）|

从 Docker 容器中取出日志（未挂载卷时）：

```bash
docker cp ogmem:/opt/ogmem/perf_logs ./perf_logs_from_docker
```

### 7.2 多 worker 日志聚合

多 worker 时，每个 worker 有独立日志文件（除非显式设置了 `OGMEM_PERF_RUN_ID`）：

```bash
ls perf_logs/
# perf_logs/a1b2c3d4-xxxx.jsonl
# perf_logs/e5f6g7h8-yyyy.jsonl
# perf_logs/i9j0k1l2-zzzz.jsonl
# perf_logs/m3n4o5p6-wwww.jsonl
```

合并所有文件后生成报告：

```bash
# 合并
cat perf_logs/*.jsonl > perf_logs/merged.jsonl

# 生成 Markdown 报告
uv run python -m perf.report perf_logs/merged.jsonl -o perf_report.md

# 直接打印到终端
uv run python -m perf.report perf_logs/merged.jsonl
```

### 7.3 报告内容解读

`perf.report` 生成的报告包含 5 个部分：

| 部分 | 内容 |
|------|------|
| Stage Totals | 各生命周期阶段（compose / after_turn / compact 等）的总耗时、token 消耗、美元成本 |
| Sub-span Breakdown | 阶段内子 span 细分（如 extract_llm / rank 等）|
| Model Breakdown | 按模型分组的 token 用量（LLM + embedding）|
| Bottleneck Ranking | 按 wall_ms 降序排列，快速定位最慢阶段 |
| Counters Appendix | 原始计数器（input / output / cache_read / cache_write）|

### 7.4 实时查看（测试进行中）

```bash
# 实时追踪日志行数
watch -n5 'wc -l perf_logs/*.jsonl'

# 实时生成临时报告
watch -n30 'cat perf_logs/*.jsonl > /tmp/perf_tmp.jsonl && uv run python -m perf.report /tmp/perf_tmp.jsonl'
```

### 7.5 token_stats 端点（轻量备选）

如不需要阶段级细分，`/api/v1/token_stats` 提供累计 token 统计（无需 perf 模块启用）：

```bash
# 查看累计统计（非破坏性）
curl http://127.0.0.1:8090/api/v1/token_stats

# 重置计数器并返回快照（测试前后各调一次，差值即为本次测试消耗）
curl -X POST -H "Content-Type: application/json" \
  -d '{"reset": true}' \
  http://127.0.0.1:8090/api/v1/token_stats
```

> 注意：多 worker 时此端点只返回处理该请求的 worker 的累计值，不是全局总和。需要全局精确值时，使用 perf JSONL 聚合。

---

## 8. 环境变量速查表

### perf 相关

| 变量 | 说明 | 示例值 |
|------|------|-------|
| `OGMEM_PERF_ENABLED` | 启用 perf 录制 | `1` |
| `OGMEM_PERF_OUT` | JSONL 输出路径 | `/opt/ogmem/perf_logs/perf.jsonl` |
| `OGMEM_PERF_HTTP_URL` | HTTP 推送端点（备选） | `http://collector:9090/events` |
| `OGMEM_PERF_RUN_ID` | 固定 run_id（多 worker 共享文件时设置）| `locomo-run-20260519` |
| `OGMEM_PERF_RATE_CARD` | 自定义费率卡路径 | `/etc/ogmem/rate_card.yaml` |

### 服务相关

| 变量 | 说明 | 示例值 |
|------|------|-------|
| `OGMEM_CONFIG` | ogmem.yaml 路径 | `/etc/ogmem/config.yaml` |
| `OGMEM_HTTP_PORT` | HTTP 监听端口 | `8090` |
| `OGMEM_WORKERS` | gunicorn worker 数 | `1`（perf 分析）/ `4`（高并发）|
| `PYTHONPATH` | Python 模块搜索路径 | `/opt/ogmem` |

### deploy.sh 专用

| 变量 | 说明 | 示例值 |
|------|------|-------|
| `OGMEM_IMAGE` | ogmemory 镜像名 | `ogmemory:local` |
| `SKIP_PULL` | 跳过镜像拉取 | `true` |
| `PERF_LOGS_HOST_DIR` | perf 日志宿主机挂载目录 | `/home/user/ogmem/perf_logs` |
| `AGFS_DATA_DIR` | AGFS 数据宿主机挂载目录 | `/home/user/ogmem/agfs` |

---

## 附录 A：单 worker 与多 worker 差异总结

| 维度 | 单 worker | 多 worker (4) |
|------|-----------|--------------|
| perf JSONL 文件数 | 1 | 最多 4（除非设 `OGMEM_PERF_RUN_ID`）|
| per-request token 归因 | 准确 | 准确（每条 span 是同 worker 内差值）|
| 跨 worker 全局累计 token | 即为该文件总和 | 需合并后 `cat *.jsonl \| wc -l` |
| `GET /token_stats` 准确性 | 全局准确 | 仅反映随机一个 worker |
| 适用场景 | perf 分析、调试 | 高并发负载测试 |
| 报告生成 | 直接 `perf.report perf.jsonl` | 先合并再生成 |

## 附录 B：常见问题

**Q: perf 启用后服务启动失败，报 `ImportError: No module named 'perf'`**

A: 使用的 Docker 镜像未包含 `perf/` 目录（默认 poc1_428 镜像）。需按[第 3 节](#3-重新构建含-perf-的-docker-镜像)重新构建。

**Q: `perf_logs/` 目录不存在或没有文件**

A: Docker 部署时需设置 `PERF_LOGS_HOST_DIR`，否则日志在容器内。本地部署时 `perf_logs/` 在 CWD 下自动创建（`cd /opt/ogmem` 后为 `/opt/ogmem/perf_logs/`）。

**Q: `GET /api/v1/perf/events` 返回 `"events": []`**

A: 该端点只读取 `MemorySink`（用于测试），默认使用 `JsonlSink` 写文件。`events` 始终为空是正常的——日志在 JSONL 文件里。

**Q: 多 worker 时 `token_stats` 数据偏低**

A: 正常现象。`token_stats` 端点只返回处理本次请求的那个 worker 的累计值。用 perf JSONL 聚合获取全局准确值。

**Q: locomo10 benchmark 连不上 OpenClaw**

A: 检查 `eval.py` 的 `--base-url`（默认 `http://127.0.0.1:18789`）和 `--token`（默认硬编码值），确保与 OpenClaw 部署时的 `OPENCLAW_GATEWAY_TOKEN` 一致。
