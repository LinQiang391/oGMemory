# oG-Memory OpenClaw Plugin

> 将 oG-Memory 图记忆系统集成到 OpenClaw Agent 框架的插件

## 项目概述

**目标**：为 OpenClaw AI Agent 提供持久化记忆能力，使用 oG-Memory 作为底层存储。

**成果**：实现了完整的 Agent → TypeScript → Python → LocalFS 数据写入链路，支持通过对话形式长期记忆用户偏好和重要信息。

---

## 功能特性

- ✅ **记忆写入工具**：`og_memory_write` - 将对话中的重要信息持久化到图存储
- ✅ **多语言支持**：支持中英文内容
- ✅ **本地存储**：使用 LocalFS 适配器，无需外部依赖
- ✅ **分类记忆**：支持 profile, preference, entity, event 等多种记忆类型
- ✅ **标签系统**：可选标签用于组织和检索

---

### 文件清单

```
openclaw-plugin/
├── README.md                    # 本文档
├── package.json                 # NPM 包配置
├── openclaw.plugin.json         # 插件元数据和配置架构
├── index.ts                     # TypeScript 插件入口
├── tsconfig.json                # TypeScript 配置
└── bridge/
    └── write_memory.py          # Python 桥接脚本
```

### 关键组件

| 组件 | 路径 | 职责 |
|------|------|------|
| TypeScript 插件 | `index.ts` | 工具注册、参数验证、进程管理 |
| Python 桥接 | `bridge/write_memory.py` | 序列化、API 调用、错误处理 |
| oG-Memory API | `service/api.py` | 写入逻辑、策略路由、元数据管理 |
| AGFS 适配器 | `fs/agfs_adapter/` | ContextFS 协议实现、AGFS 文件存储 |

### 调用链路

```
OpenClaw Agent
    ↓ (工具调用)
og_memory_write (TypeScript)
    ↓ (JSON stdin/out via child_process)
write_memory.py (Python Bridge)
    ↓ (MemoryWriteAPI)
oG-Memory Service Layer
    ↓ (AGFS ContextFS)
/tmp/agfs-data/accounts/...
```

---

## 安装配置

### 前置要求

- OpenClaw 2026.3.13+
- Node.js v22+
- Python 3.11 (conda py11 环境)
- Go 1.26+ (编译 AGFS)
- oG-Memory 项目依赖包 (pyagfs, openai 等)

### 安装步骤

#### **克隆项目**
```bash
git clone <oG-Memory-repo>
cd oG-Memory
```

#### **安装 Python 依赖**
```bash
# 1. 创建 conda 环境（如果不存在）
conda env list | grep py11 || conda create -n py11 python=3.11 -y

# 2. 激活环境
conda activate py11

# 3. 获取 Python 路径（配置时需要）
PYTHON_PATH=$(which python)
echo "Python path: $PYTHON_PATH"

# 4. 安装 oG-Memory
cd /data/Workspace2/oG-Memory
pip install -e .
```

#### **AGFS 安装与启动**

本项目依赖 [AGFS](https://github.com/c4pt0r/agfs) 服务运行在 `localhost:8080`

**检查是否已安装：**
```bash
agfs-server --help 2>/dev/null && echo "已安装" || echo "未安装"
curl -s http://localhost:8080/api/v1/health
# 预期输出: {"status":"healthy","version":"1.4.0"}
```

**安装（任选一种）：**
```bash
# 方式 1：官方脚本（推荐）
curl -fsSL https://raw.githubusercontent.com/c4pt0r/agfs/master/install.sh | sh

# 方式 2：从源码编译（需要 Go 1.21+）
git clone https://github.com/c4pt0r/agfs.git /tmp/agfs-build
cd /tmp/agfs-build/agfs-server
go build -o ~/.local/bin/agfs-server .
```

> 如果方式 2 下载依赖超时，先设置 Go 代理：`go env -w GOPROXY=https://goproxy.cn,direct`

**配置与启动：**
```bash
# 1. 创建数据目录
mkdir -p /tmp/agfs-data

# 2. 确保 ~/.local/bin 在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 3. 创建配置文件（必需）
cat > ~/.agfs-config.yaml << 'EOF'
server:
  address: ":8080"
  log_level: "info"

plugins:
  memfs:
    enabled: true
    path: "/memfs"

  localfs:
    enabled: true
    path: "/local"
    config:
      local_dir: "/tmp/agfs-data"
EOF

# 4. ⚠️ 禁用代理环境变量（重要：否则 localhost 连接会返回 502）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 5. 后台启动
agfs-server -c ~/.agfs-config.yaml &

# 6. 验证
curl http://localhost:8080/api/v1/health
# 预期输出: {"status":"healthy","version":"1.4.0"}
```

> **注意**：本插件使用 AGFS 的 `/local` 后端，配置文件中必须启用并指定 `local_dir` 参数。

#### **解决代理干扰问题**

如果您的系统配置了 HTTP 代理，AGFS 的本地连接会失败（返回 502 Bad Gateway）。

**永久解决（推荐）**:
在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
export no_proxy="localhost,127.0.0.1"
```

然后重新加载配置：
```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

**临时解决（每次启动 AGFS 前）**:
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

**验证修复**:
```bash
curl http://localhost:8080/api/v1/health
# 预期输出: {"status":"healthy","version":"1.4.0"}
```

> **说明**: 插件会在子进程中自动禁用代理，所以即使环境中有代理，插件功能仍然正常。只有手动运行健康检查时会受影响。

#### **安装插件**
```bash
cd /path/to/oG-Memory
openclaw plugins install -l openclaw-plugin

# 验证插件加载
openclaw plugins info og-memory
# 预期输出: Status: loaded
```

#### **配置插件**
```bash
# 确保 conda 环境已激活
conda activate py11

# 设置 Python 路径（使用上一步获取的路径）
openclaw config set plugins.entries.og-memory.config.pythonPath "$PYTHON_PATH"

# 设置项目根目录（按实际修改）
PROJECT_ROOT="/data/Workspace2/oG-Memory"
openclaw config set plugins.entries.og-memory.config.projectRoot "$PROJECT_ROOT"

# 设置 API Key（从 https://platform.openai.com/api-keys 获取）
openclaw config set plugins.entries.og-memory.config.apiKey "YOUR_API_KEY"

# 设置 OpenAI 兼容 API 地址（可选）
openclaw config set plugins.entries.og-memory.config.openaiBaseUrl "https://api.openai.com/v1"

# 设置工具配置文件（确保插件工具可见）
openclaw config set tools.profile full
```

#### **重启 OpenClaw Gateway**

OpenClaw Gateway 作为 systemd 服务运行，插件或配置更改后需要重启：

```bash
# 检查服务状态
systemctl --user status openclaw-gateway

# 重启服务（插件安装、配置更改后必需）
systemctl --user restart openclaw-gateway

# 查看日志（排查问题）
openclaw logs --follow
# 或
journalctl --user -u openclaw-gateway -f
```

**何时需要重启**:
- ✅ 需要：插件安装/移除、插件配置更改、`tools.profile` 更改
- ❌ 不需要：Agent 配置更改、模型配置更改

**注意**: `openclaw agent --agent main --local` 用于运行单次交互测试，不是启动 gateway 服务。

---

### 环境检查 (可选) 

以下环境变量由插件自动设置，无需手动配置：

```bash
# AGFS 服务地址
AGFS_BASE_URL="http://localhost:8080"

# LLM 提供商类型
CONTEXTENGINE_PROVIDER="openai"

# 禁用代理 (硬编码空字符串，避免本地请求被代理拦截)
http_proxy=""
https_proxy=""
HTTP_PROXY=""
HTTPS_PROXY=""
```

其他配置项（`OG_MEMORY_ROOT`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`）通过 `openclaw config set` 配置后自动传递给 Python 桥接进程。

复制以下脚本到 terminal 执行，检查所有相关配置：

> **注意**: 执行时可能会看到 `[plugins] [og-memory] Plugin registered successfully` 等消息，这是 OpenClaw 的正常日志输出，不影响功能。脚本已自动过滤这些消息。

```bash
echo -e "\n\n=== OpenClaw 版本 ==="
openclaw --version

echo -e "\n=== Node.js 版本 ==="
node --version

echo -e "\n=== Python 版本 ==="
python --version

echo -e "\n=== 插件状态 ==="
openclaw plugins info og-memory 2>/dev/null || echo "❌ 插件未加载"

echo -e "\n=== AGFS 服务状态 ==="
# 检查 HTTP 状态码和响应内容
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v1/health 2>/dev/null)
AGFS_HEALTH=$(curl -s http://localhost:8080/api/v1/health 2>/dev/null)

if [ "$HTTP_CODE" = "200" ]; then
  echo "✅ AGFS 运行中 (HTTP $HTTP_CODE)"
  if [ -n "$AGFS_HEALTH" ]; then
    echo "   $AGFS_HEALTH"
  fi
else
  echo "❌ AGFS 未运行 (HTTP $HTTP_CODE)"
  echo "   启动命令: agfs-server &"
fi

echo -e "\n=== 插件配置 ==="
echo "pythonPath: $(openclaw config get plugins.entries.og-memory.config.pythonPath 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[" || echo '❌ 未配置')"
echo "projectRoot: $(openclaw config get plugins.entries.og-memory.config.projectRoot 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[" || echo '❌ 未配置')"
echo "apiKey: $(if [ -n "$(openclaw config get plugins.entries.og-memory.config.apiKey 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[")" ]; then echo '✅ 已配置'; else echo '❌ 未配置'; fi)"
echo "openaiBaseUrl: $(openclaw config get plugins.entries.og-memory.config.openaiBaseUrl 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[" || echo '(使用默认)')"
echo "tools.profile: $(openclaw config get tools.profile 2>&1 | grep -v "^\[plugins\]" | grep -v "^  \[" || echo '❌ 未配置')"

echo -e "\n=== Shell 环境变量 (仅作参考，插件优先使用 config) ==="
echo "OPENAI_API_KEY: $(if [ -n "$OPENAI_API_KEY" ]; then echo '✅ 已设置'; else echo '❌ 未设置'; fi)"
echo "OPENAI_BASE_URL: ${OPENAI_BASE_URL:-(未设置)}"
echo "http_proxy: ${http_proxy:-(未设置)}"
echo "https_proxy: ${https_proxy:-(未设置)}"

echo -e "\n=== 存储路径 ==="
echo "/tmp/agfs-data: $(if [ -d "/tmp/agfs-data" ]; then echo '✅ 存在'; else echo '❌ 不存在'; fi)"
```

---

## 数据存储

### 物理路径

```
/tmp/agfs-data/
└── accounts/
    └── {account_id}/
        └── users/
            └── {user_id}/
                └── memories/
                    └── {category}/
                        └── {slug}/
                            ├── content.md         # 完整内容
                            ├── .abstract.md       # 一句话摘要
                            ├── .overview.md       # 结构化概述
                            ├── .relations.json    # 关系边
                            └── .meta.json         # 元数据 (status=ACTIVE)
```

### 记忆类型

| 类型 | URI 模式 | 写入策略 | 示例 |
|------|----------|----------|------|
| profile | `.../memories/profile` | merge (固定 URI) | 用户背景信息 |
| preference | `.../memories/preferences/{slug}` | merge (按主题) | 喜好、习惯 |
| entity | `.../memories/entities/{slug}` | merge (按实体) | 人物、组织 |
| event | `.../memories/events/{event_id}` | append (唯一 ID) | 事件记录 |
| case | `.../agents/{agent}/memories/cases/{case_id}` | append | 案例记录 |

---

## 测试验证

### 端到端测试

```bash
# 1. 验证插件加载
openclaw plugins info og-memory
# ✅ Status: loaded
# ✅ Tools: og_memory_write

# 2. 验证工具调用
openclaw agent --agent main --local --message "Save: I drink coffee"
# ✅ Agent 调用工具成功
# ✅ 日志显示: "Python stdout: {"ok": true, ...}"

# 3. 验证数据写入
ls /tmp/agfs-data/accounts/default/users/default/memories/preference/
# ✅ 新目录创建
# ✅ 包含 5 个文件 (content.md, .abstract.md, .overview.md, .relations.json, .meta.json)

# 4. 验证内容正确
cat /tmp/agfs-data/.../preference_xxx/content.md
# ✅ "I drink coffee"
```
---

## 故障排查

### 常见问题

**1. 工具调用失败：Command failed: ...**
- 检查 Python 路径是否正确
- 检查 OPENAI_API_KEY 是否配置
- 查看日志：`openclaw logs --limit 100 | grep og-memory`

**2. AGFS 连接失败 (502 Bad Gateway)**
- **原因**：代理环境变量干扰 localhost 连接
- **检查**：`echo $http_proxy $https_proxy`
- **解决**：
  ```bash
  # 临时禁用代理
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

  # 重启 AGFS
  pkill agfs-server
  agfs-server -c ~/.agfs-config.yaml &
  ```
- **验证**：`curl http://localhost:8080/api/v1/health`

**3. 文件未创建**
- 验证 AGFS 运行：`curl http://localhost:8080/api/v1/health`
- 验证存储路径：`ls -la /tmp/agfs-data/accounts/`（AGFS LocalFS 路径）
- 检查写入权限

**4. LLM 调用失败**
- 验证 API Key 是否有效
- 检查网络连接
- 查看错误日志

### 调试技巧

```bash
# 查看 Python 桥接日志
openclaw agent --agent main --local --message "test" 2>&1 | grep "og-memory"

# 手动测试 Python 脚本
echo '{"content":"test"}' | python openclaw-plugin/bridge/write_memory.py

# 检查存储文件
find /tmp/agfs-data -name "*.meta.json" 2>/dev/null | head -5
```
