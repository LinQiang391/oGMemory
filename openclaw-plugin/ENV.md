# OpenClaw 插件环境配置

> 本文档说明 OpenClaw 插件的专用环境配置。
> 通用 oG-Memory 环境配置请参阅项目根目录 [`../ENV.md`](../ENV.md)。

---

## 📋 前置要求

### 依赖检查

| 组件 | 版本要求 | 检查命令 |
|------|---------|---------|
| **OpenClaw** | 2026.3.13+ | `openclaw --version` |
| **Node.js** | v22+ | `node --version` |
| **oG-Memory** | 已安装 | 见 [通用配置](../ENV.md) |
| **AGFS** | 运行中 | `curl http://localhost:1833/api/v1/health` |
| **Python** | 3.11 (conda) | `conda activate <env>; python --version` |

---

## 🔌 插件安装

```bash
# 1. 进入 oG-Memory 项目目录
cd /path/to/oG-Memory

# 2. 安装插件（本地链接）
openclaw plugins install -l openclaw-plugin

# 3. 验证插件加载
openclaw plugins info og-memory
# 预期输出: Status: loaded
#           Tools: og_memory_write
```

---

## ⚙️ 插件配置

### 配置项说明

| 配置项 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `pythonPath` | ✅ | - | Python 可执行文件路径 |
| `projectRoot` | ✅ | - | oG-Memory 项目根目录 |
| `apiKey` | ✅* | - | OpenAI API 密钥 |
| `openaiBaseUrl` | ❌ | `https://api.openai.com/v1` | OpenAI API 地址 |

\* 使用 `CONTEXTENGINE_PROVIDER=mock` 时不需要，但插件当前默认使用 OpenAI

### 配置步骤

```bash
# 1. 激活 conda 环境（使用你的环境名称）
conda activate og-memory  # 或 py11，或其他你创建的环境名称

# 2. 获取 Python 路径
PYTHON_PATH=$(which python)
echo "Python path: $PYTHON_PATH"

# 3. 设置项目根目录（按实际修改）
PROJECT_ROOT="/data/Workspace2/oG-Memory"

# 4. 配置插件
openclaw config set plugins.entries.og-memory.config.pythonPath "$PYTHON_PATH"
openclaw config set plugins.entries.og-memory.config.projectRoot "$PROJECT_ROOT"
openclaw config set plugins.entries.og-memory.config.apiKey "YOUR_API_KEY"
openclaw config set plugins.entries.og-memory.config.openaiBaseUrl "https://api.openai.com/v1"

# 5. 设置工具可见性（重要！）
openclaw config set tools.profile full
```

---

## 🔄 重启 OpenClaw Gateway

插件安装或配置更改后需要重启服务：

```bash
# 检查服务状态
systemctl --user status openclaw-gateway

# 重启服务
systemctl --user restart openclaw-gateway

# 查看日志
openclaw logs --follow
# 或
journalctl --user -u openclaw-gateway -f
```

**何时需要重启**:
- ✅ 需要：插件安装/移除、插件配置更改、`tools.profile` 更改
- ❌ 不需要：Agent 配置更改、模型配置更改

---

## 📊 插件环境变量

插件运行时自动设置以下环境变量，**无需手动配置**：

```bash
# AGFS 服务地址
AGFS_BASE_URL="http://localhost:1833"

# LLM 提供商（插件当前硬编码为 OpenAI）
CONTEXTENGINE_PROVIDER="openai"

# 禁用代理（避免本地请求被代理拦截）
http_proxy=""
https_proxy=""
HTTP_PROXY=""
HTTPS_PROXY=""
```

从 OpenClaw 配置传递到 Python 的变量：
- `OG_MEMORY_ROOT` ← `projectRoot`
- `OGMEM_API_KEY` ← `apiKey`
- `OGMEM_BASE_URL` ← `openaiBaseUrl` (可选)

**存储路径变量** (可选，方便脚本引用)：
- `AGFS_DATA_DIR` ← 默认 `/tmp/agfs-data`，可在 `~/.agfs-config.yaml` 中修改
  - 设置方式: `export AGFS_DATA_DIR=/your/custom/path`

---

## ✅ 一键环境检查脚本

```bash
# 从项目根目录运行（包含通用环境 + 插件检查）
bash ../check-env.sh
```

该脚本会检查：
- **基础依赖**：Go、Python、Node.js、OpenClaw
- **AGFS 服务**：运行状态、配置文件、数据目录
- **oG-Memory**：Python 包安装情况
- **OpenClaw 插件**：加载状态、配置项、工具可见性
- **环境变量**：API key、代理设置等

---

## 🔍 常见问题

### 插件未加载

```bash
# 检查插件安装
openclaw plugins list

# 重新安装
openclaw plugins install -l openclaw-plugin

# 查看错误日志
openclaw logs --limit 100 | grep og-memory
```

### 工具调用失败

```bash
# 1. 检查 Python 路径
openclaw config get plugins.entries.og-memory.config.pythonPath

# 2. 验证 Python 可执行
$PYTHON_PATH --version

# 3. 检查 oG-Memory 安装
$PYTHON_PATH -c "from core.models import RequestContext; print('✅')"

# 4. 手动测试 Python 桥接
echo '{"content":"test"}' | \
  OG_MEMORY_ROOT=$(pwd) \
  AGFS_BASE_URL=http://localhost:1833 \
  CONTEXTENGINE_PROVIDER=mock \
  python bridge/write_memory.py
```

### AGFS 连接失败 (502)

见 [通用配置文档 - 代理问题解决](../ENV.md#-代理问题解决)

---

## 📚 下一步

- **功能介绍与使用**: [`README.md`](README.md)
- **通用环境配置**: [`../ENV.md`](../ENV.md)
- **故障排查**: [`README.md` - 故障排查章节](README.md#故障排查)
