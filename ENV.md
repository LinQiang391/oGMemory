# oG-Memory 环境配置指南

> 本文档说明 oG-Memory 的通用环境配置，不涉及特定 Agent 框架集成。
> 如需 OpenClaw 插件配置，请参阅 [`openclaw-plugin/ENV.md`](openclaw-plugin/ENV.md)。

---

## 📋 依赖模块概览

oG-Memory 依赖以下模块，按安装顺序：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Go 1.26+  │────▶│    AGFS     │────▶│ oG-Memory   │
│  (编译 AGFS) │     │  (文件存储)  │     │  (Python)   │
└─────────────┘     └─────────────┘     └─────────────┘
                         │
                         ▼
                    ┌─────────────┐
                    │ LocalFS     │
                    │ (/tmp/agfs) │
                    └─────────────┘
```

---

## 🔧 前置要求

| 组件 | 版本要求 | 用途 | 检查命令 |
|------|---------|------|---------|
| **Go** | 1.26+ | 编译 AGFS | `go version` |
| **Python** | 3.11 | 运行 oG-Memory | `python --version` |
| **conda** | 最新 | 管理 Python 环境（可选） | `conda --version` |
| **pip** | 最新 | 安装 Python 包 | `pip --version` |

---

## 📦 AGFS 安装配置

### 什么是 AGFS？

AGFS (Another Go File System) 是 oG-Memory 的文件存储层，提供 HTTP API 接口。

### 安装方式

**方式 1：官方脚本（推荐）**
```bash
curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh
```

**方式 2：从源码编译**
```bash
# 如果 Go 下载依赖超时，先设置代理
go env -w GOPROXY=https://goproxy.cn,direct

git clone https://github.com/c4pt0r/agfs.git /tmp/agfs-build
cd /tmp/agfs-build/agfs-server
go build -o ~/.local/bin/agfs-server .
```

### 配置 AGFS

```bash
# 1. 设置 AGFS 数据目录（可选环境变量，方便后续引用）
export AGFS_DATA_DIR="${AGFS_DATA_DIR:-/tmp/agfs-data}"

# 2. 创建数据目录
mkdir -p "$AGFS_DATA_DIR"

# 3. 确保 ~/.local/bin 在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 4. 创建配置文件（必需）
cat > ~/.agfs-config.yaml << EOF
server:
  address: ":1833"
  log_level: "info"

plugins:
  localfs:
    enabled: true
    path: "/local"
    config:
      local_dir: "$AGFS_DATA_DIR"
EOF

# 4. 后台启动
agfs-server -c ~/.agfs-config.yaml &

# 5. 验证
curl http://localhost:1833/api/v1/health
# 预期输出: {"status":"healthy","version":"..."}
```

## 🐍 oG-Memory 安装

```bash
# 1. 创建 conda 环境（如果不存在，env 名称可自定义）
conda env list | grep og-memory || conda create -n og-memory python=3.11 -y

# 2. 激活环境（使用你的环境名称）
conda activate og-memory  # 或你的环境名称

# 3. 进入项目目录
cd /path/to/oG-Memory

# 4. 安装项目（可编辑模式）
pip install -e .

# 5. 验证安装
python -c "from core.models import RequestContext; print('✅ oG-Memory 安装成功')"
```

---

## 📊 环境变量配置表

| 变量名 | 必需 | 默认值 | 说明 | 配置位置 |
|--------|------|--------|------|---------|
| `AGFS_DATA_DIR` | ❌ | `/tmp/agfs-data` | AGFS LocalFS 数据目录 | 环境变量 / `~/.agfs-config.yaml` |
| `AGFS_BASE_URL` | ❌ | `http://localhost:1833` | AGFS 服务地址 | 环境变量 / 代码配置 |
| `CONTEXTENGINE_PROVIDER` | ❌ | `mock` | LLM 提供商 (`mock`\|`openai`) | 环境变量 |
| `OPENAI_API_KEY` | ✅* | - | OpenAI API 密钥（非 mock 模式必需） | 环境变量 |
| `OPENAI_BASE_URL` | ❌ | `https://api.openai.com/v1` | OpenAI API 地址 | 环境变量 |
| `OG_MEMORY_ACCOUNT_ID` | ❌ | `default` | 默认账户 ID | 环境变量 |
| `OG_MEMORY_USER_ID` | ❌ | `default` | 默认用户 ID | 环境变量 |
| `OG_MEMORY_AGENT_ID` | ❌ | `default` | 默认 Agent ID | 环境变量 |

\* 使用 `mock` 提供商时不需要

### 存储路径

| 环境变量 | 默认值 | 用途 | 说明 |
|---------|--------|------|------|
| `AGFS_DATA_DIR` | `/tmp/agfs-data` | AGFS LocalFS 数据目录 | 可选变量，方便脚本引用 |
| - | `$AGFS_DATA_DIR/accounts/{account_id}/users/{user_id}/memories/` | 用户记忆存储 | 物理路径，由 URI 映射 |

---

## ✅ 一键环境检查脚本

```bash
# 从项目根目录运行
bash check-env.sh
```

该脚本会检查：
- **基础依赖**：Go、Python、Node.js、OpenClaw
- **AGFS 服务**：运行状态、配置文件、数据目录
- **oG-Memory**：Python 包安装情况
- **OpenClaw 插件**：加载状态、配置项
- **环境变量**：API key、代理设置等

---

## 🔍 常见问题

### AGFS 启动失败

```bash
# 检查配置文件
cat ~/.agfs-config.yaml

# 检查端口占用
lsof -i :1833

# 查看日志
agfs-server -c ~/.agfs-config.yaml
```

### 502 Bad Gateway

如果系统配置了 HTTP 代理，localhost 连接会返回 502：

```bash
# 检查是否受影响
curl http://localhost:1833/api/v1/health
# 如果返回 502 Bad Gateway，说明代理干扰

# 解决方案 1：永久设置（推荐）
echo 'export no_proxy="localhost,127.0.0.1"' >> ~/.bashrc
source ~/.bashrc

# 解决方案 2：临时禁用
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

---

## 📚 下一步

- **集成到 Agent 框架**: 参阅各框架的插件文档
  - OpenClaw: [`openclaw-plugin/ENV.md`](openclaw-plugin/ENV.md)
- **API 使用**: 参阅 `service/api.py` 文档字符串
- **运行测试**: `pytest tests/`
