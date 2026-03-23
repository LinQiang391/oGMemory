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
$AGFS_DATA_DIR/accounts/...
```

---

## 快速开始

### 环境配置

⚠️ **首次使用需要配置环境**，请参阅：

- **通用环境**（AGFS、oG-Memory）: [`../ENV.md`](../ENV.md)
- **插件配置**（OpenClaw 集成）: [`ENV.md`](ENV.md)

---

## 数据存储

### 物理路径

> 默认路径：`$AGFS_DATA_DIR` (通常为 `/tmp/agfs-data`)，可通过 `~/.agfs-config.yaml` 修改。

```
$AGFS_DATA_DIR/
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
ls ${AGFS_DATA_DIR:-/tmp/agfs-data}/accounts/default/users/default/memories/preference/
# ✅ 新目录创建
# ✅ 包含 5 个文件 (content.md, .abstract.md, .overview.md, .relations.json, .meta.json)

# 4. 验证内容正确
cat ${AGFS_DATA_DIR:-/tmp/agfs-data}/.../preference_xxx/content.md
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
- **验证**：`curl http://localhost:1833/api/v1/health`

**3. 文件未创建**
- 验证 AGFS 运行：`curl http://localhost:1833/api/v1/health`
- 验证存储路径：`ls -la ${AGFS_DATA_DIR:-/tmp/agfs-data}/accounts/`（AGFS LocalFS 路径）
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
find ${AGFS_DATA_DIR:-/tmp/agfs-data} -name "*.meta.json" 2>/dev/null | head -5
```
