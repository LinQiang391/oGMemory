# TEST_GUIDE.md — Mock Context Engine 测试指南

> 使用 OpenClaw Dashboard + 日志查看方式验证插件是否正常工作。
> 假设插件已安装并配置完毕（参考 README.md）。

---

## 1. 准备工作：打开日志监控

在开始任何测试之前，先打开一个终端窗口用于实时查看日志。二选一：

```bash
# 方式 A：openclaw 内置日志命令
openclaw logs --follow

# 方式 B：systemd 日志（推荐，输出更完整，可配合 grep 过滤）
journalctl --user -u openclaw-gateway -f | grep -E "mock-engine|memory-api"
```

> **提示：** 方式 B 只显示插件相关日志，噪音更少。如需看完整 gateway 日志去掉 `| grep ...` 即可。

保持这个终端窗口打开，后续所有操作都在 **另一个终端** 或 **浏览器 Dashboard** 中进行。

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

浏览器会自动打开 Dashboard 界面。如果没有自动打开，手动访问终端中提示的 URL。

---

## 4. 逐步触发生命周期 Hook

### 4.1 触发 `bootstrap` + `assemble` + `after_turn` + `dispose`

**操作：** 在 Dashboard 的对话框中发送任意一条消息，例如：

```
hello
```

**日志中应看到（按顺序）：**

```
[mock-engine ...] [DEBUG] ──── bootstrap() called ────
[mock-engine ...] [INFO]  bootstrap  sessionId=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

[mock-engine ...] [DEBUG] ──── assemble() called ────
[mock-engine ...] [INFO]  assemble  sessionId=...  input_msgs=N  budget=16000
[memory-api ...]  memory_search() params={...}

[mock-engine ...] [DEBUG] ──── after_turn() called ────
[mock-engine ...] [INFO]  after_turn  sessionId=...

[mock-engine ...] [DEBUG] ──── dispose() called ────
[mock-engine ...] [INFO]  dispose  releasing resources
```

这一步验证了 **4 个核心 hook** 全部打通。

### 4.2 触发 `compact`

`compact` 在以下两种情况会被调用：
- 用户发送 `/compact` 斜杠命令
- context 超出 token budget 时自动触发溢出恢复

**⚠️ 已知问题：Dashboard 中 `/compact` 可能不生效**（见下方「已知问题」章节）。
推荐通过 **撑满 context 触发自动 compact**：

**操作：** 在 Dashboard 中连续发送多条较长的消息，或发送一条很长的消息

**日志中应看到：**

```
[mock-engine ...] [DEBUG] ──── compact() called ────
[mock-engine ...] [INFO]  compact  sessionId=...  force=false
```

---

## 5. 已知问题

### 5.1 `ingest` 未被调用

**现象：** 日志中始终没有 `ingest()` 的输出，即使已注释掉 `ingestBatch`。

**原因分析：** OpenClaw 的 session manager 自行处理消息持久化（文档中 legacy 引擎的 ingest 描述为 "no-op"）。runtime 可能在某些版本中不会主动调用插件的 `ingest`，而是依赖 session manager 完成存储后，仅在 `assemble` 时将消息列表传给引擎。

**影响：** 不影响核心流程。引擎在 `assemble` 阶段已能拿到完整的 `messages` 数组，可以在此处做索引、检索等操作。

**后续：** 如果需要逐条消息的实时索引，可在 `assemble` 或 `afterTurn` 中对比前后 messages 差异来实现。

### 5.2 `/compact` 在 Dashboard 中被当作 bash 命令执行

**现象：** 在 Dashboard 对话框中输入 `/compact` 发送后，agent 将其交给 exec tool 当作 shell 命令执行，报错：

```json
{
  "status": "error",
  "tool": "exec",
  "error": "/bin/bash: line 1: /compact: No such file or directory"
}
```

**原因：** Dashboard 没有将 `/compact` 识别为内置斜杠命令，而是作为普通消息传给了 agent。这可能与 agent 配置、channel 类型、或 OpenClaw 版本有关。

**规避方法：**
- 通过连续对话撑满 context，让 runtime 自动触发 compact（见 4.2）
- 通过 CLI 指定 session-id 尝试发送

### 5.3 `ingestBatch` 与 `ingest` 互斥

**现象：** 同时定义 `ingestBatch` 和 `ingest` 时，两者都不被调用。

**原因推测：** OpenClaw runtime 检测到 `ingestBatch` 存在时可能跳过逐条 `ingest`，但 batch 路径的调用时机与预期不同。

**建议：** 开发阶段先注释掉 `ingestBatch`，确保 `ingest`（如果被调用）能正常工作。后续根据实际需求决定启用哪个。

### 5.4 `prepareSubagentSpawn` 无法触发

**原因：** OpenClaw 文档明确说明：

> The `prepareSubagentSpawn` hook is part of the interface for future use, but the runtime does not invoke it yet.

这是预留接口，当前版本不会调用，无需测试。

### 5.5 `onSubagentEnded` 需要 subagent 场景

**触发条件：** 需要 agent 配置了 subagent 并且 subagent session 结束。属于进阶场景，基础测试中可跳过。

---

## 6. 回滚到默认引擎

测试完毕后如需恢复：

编辑 `~/.openclaw/openclaw.json`，将 `plugins.slots.contextEngine` 改为 `"legacy"`（或删除该字段），然后：

```bash
openclaw restart
openclaw doctor  # 确认已恢复到 legacy
```