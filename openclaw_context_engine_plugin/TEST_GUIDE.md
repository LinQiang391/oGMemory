# TEST_GUIDE.md — oG-Memory Context Engine 测试指南

> 使用 OpenClaw Dashboard + 日志验证 oG-Memory 集成是否正常工作。
> 假设插件已安装并配置完毕（参考 ENV.md）。

---

## 当前实现状态

| 功能 | 状态 | 说明 |
|------|------|------|
| **写入 (after_turn)** | ✅ 已实现 | 每轮对话后调用 oG-Memory `commit_session()` 写入记忆 |
| **检索 (assemble)** | ⏸️ 暂未实现 | 当前仅透传消息，Phase 3 将集成向量检索 |
| **压缩 (compact)** | ⏸️ 暂未实现 | 返回 noop，Phase 3 将实现记忆压缩 |
| **其他 lifecycle** | ⏸️ 暂未实现 | bootstrap/ingest/prepare_subagent_spawn 等仅记录日志 |

---

## 1. 准备工作：打开日志监控

在开始任何测试之前，先打开一个终端窗口用于实时查看日志：

```bash
# 推荐：systemd 日志过滤（只显示 oG-Memory 插件相关）
journalctl --user -u openclaw-gateway -f | grep -E "memory-api"

# 或查看完整 gateway 日志
journalctl --user -u openclaw-gateway -f
```

保持这个终端窗口打开，后续所有操作都在 **浏览器 Dashboard** 中进行。

---

## 2. 确认插件已加载

```bash
openclaw doctor
```

**期望看到：**
- `mock-context-engine` 出现在已加载插件列表中
- Context engine slot 显示为 `mock-context-engine`（而非 `legacy`）
- 无 plugin error 或 manifest validation 错误

---

## 3. 打开 Dashboard

```bash
openclaw dashboard
```

浏览器会自动打开 Dashboard 界面。

---

## 4. 测试写入功能（after_turn）

这是当前唯一实现的核心功能。每轮对话后，插件会调用 oG-Memory 写入对话内容。

### 4.1 发送测试消息

在 Dashboard 对话框中发送包含个人信息的消息，例如：

```
我叫张三，是一名后端工程师，喜欢用 Python 写代码。
```

### 4.2 检查日志输出

**期望看到的关键日志（按顺序）：**

```
[memory-api ...] [INFO] bootstrap  sessionId=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
[memory-api ...] [INFO] assemble  sessionId=...  input_msgs=1  budget=128000
[memory-api ...] [INFO] assemble  memory search not yet implemented, passing through messages
[memory-api ...] [INFO] LLM initialized: openai model=gpt-4o-mini
[memory-api ...] [INFO] after_turn  sessionId=...
[memory-api ...] [INFO] after_turn  committing 1 messages to oG-Memory
[memory-api ...] [INFO] commit_messages_to_memory session=... extracted=1 completed=1 failed=0
[memory-api ...] [INFO] after_turn  successfully committed 1 writes to oG-Memory
```

**关键指标说明：**
- `extracted=N`: LLM 从消息中提取了 N 个候选记忆（如 profile、entity、preference 等）
- `completed=N`: 成功写入 AGFS 的记忆数量
- `failed=N`: 写入失败的数量（应为 0）

### 4.3 验证 AGFS 数据写入

检查 oG-Memory 是否成功写入 AGFS：

```bash
# 查看 AGFS 数据目录
ls -la /tmp/agfs-data/local/plugin/accounts/default/users/default-user/memories/

# 期望看到新增的记忆节点目录，例如：
# profile/  - 用户简介
# entities/ - 提取的实体（如人名、公司名等）
# preferences/ - 用户偏好
```

**验证单个记忆节点结构：**

```bash
# 找一个具体的记忆节点
find /tmp/agfs-data/local/plugin/accounts/default/users/default-user/memories/ -name ".meta.json" | head -1 | xargs cat

# 期望看到：
{
  "uri": "ctx://...",
  "status": "ACTIVE",
  "created_at": "2026-03-24T...",
  ...
}
```

---

## 5. 测试多轮对话

继续在 Dashboard 中发送多条消息，测试累计写入：

```
你呢？叫什么名字？
我最喜欢的编程语言是 Python，因为它简洁易读。
我不喜欢 Java，因为语法太繁琐。
```

**检查日志：**
- 每轮对话后都应有 `commit_messages_to_memory` 日志
- `completed` 数量应累计增长

**检查 AGFS：**
```bash
# 统计写入的记忆节点总数
find /tmp/agfs-data/local/plugin/accounts/default/users/default-user/memories/ -name ".meta.json" | wc -l
```

---

## 6. 测试暂未实现的功能

以下功能当前仅记录日志，暂未接到 oG-Memory：

### 6.1 assemble（记忆检索）

**当前行为：**
```
[memory-api ...] [INFO] assemble  memory search not yet implemented, passing through messages
```

**预期：** 直接透传所有消息给 Agent，不做记忆检索。

**未来计划：** Phase 3 将集成向量检索，根据当前对话上下文召回相关记忆。

### 6.2 compact（记忆压缩）

**当前行为：**
```
[memory-api ...] [INFO] compact  memory compaction not yet implemented, returning noop
```

**预期：** 返回 `{"ok": true, "compacted": true, "method": "og-memory-noop"}`

**未来计划：** Phase 3 将实现记忆压缩，将多条相关记忆合并为更高层次的抽象。

---

## 7. 故障排查

### 7.1 看不到 `commit_messages_to_memory` 日志

**可能原因：**
1. AGFS 服务未运行
2. Python 环境缺少依赖（pyagfs、openai）

**排查步骤：**
```bash
# 检查 AGFS 服务
curl http://localhost:1833/api/v1/health

# 检查 Python 依赖
/home/aaa/miniconda3/envs/py11/bin/python -c "import pyagfs; print('pyagfs OK')"

# 检查插件健康状态
/home/aaa/miniconda3/envs/py11/bin/python bridge/memory_api.py memory_health '{}'
```

### 7.2 `extracted=0 completed=0`

**可能原因：**
1. LLM 未正确初始化（MockLLM 不会提取有效记忆）
2. OPENAI_API_KEY 未设置或无效

**排查步骤：**
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 查看 LLM 初始化日志
grep "LLM initialized" <(journalctl --user -u openclaw-gateway -n 100)
```

### 7.3 AGFS 写入失败

**检查 AGFS 日志：**
```bash
journalctl --user -u agfs-server -f
```

**常见问题：**
- 代理干扰：确保设置了 `export no_proxy="localhost,127.0.0.1"`
- 端口冲突：确认 AGFS 运行在 1833 端口

---

## 8. 回滚到默认引擎

测试完毕后如需恢复：

编辑 `~/.openclaw/openclaw.json`，将 `plugins.slots.contextEngine` 改为 `"legacy"`（或删除该字段），然后：

```bash
openclaw restart
openclaw doctor  # 确认已恢复到 legacy
```
