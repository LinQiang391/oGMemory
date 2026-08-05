# Claude Code 对接

## 1. 前置条件

- oGMemory HTTP 服务可访问。
- Python 3 可用；插件 Hook 只使用标准库。
- Claude Code 支持插件和 Hooks。

```bash
curl http://127.0.0.1:8090/api/v1/health
```

## 2. 安装

Marketplace：

```text
/plugin marketplace add https://gitcode.com/opengauss/oGMemory.git#dev
/plugin install og-memory@og-memory
/reload-plugins
```

源码开发：

```bash
claude --plugin-dir ./claude-plugin --debug
```

## 3. 配置

在启动 `claude` 的进程环境配置：

```bash
export OG_MEMORY_URL='http://127.0.0.1:8090'
export OG_MEMORY_ACCOUNT_ID='acct-1'
export OG_MEMORY_USER_ID='user-1'
export OG_MEMORY_AGENT_ID='claude-code'
export OG_AUTH_API_KEY='<tenant-key>'
export OGMEM_TRACE_CAPTURE_CONTENT='1'
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OG_MEMORY_URL` | `http://localhost:8090` | HTTP API 基址，不带 `/api/v1` |
| `OG_MEMORY_ACCOUNT_ID` | `acct-demo` | accountId；也兼容 `OG_ACCOUNT_ID` |
| `OG_MEMORY_USER_ID` | `u-claude` | userId；也兼容 `OG_USER_ID` |
| `OG_MEMORY_AGENT_ID` | `claude-code` | agentId；也兼容 `OG_AGENT_ID` |
| `OG_AUTH_API_KEY` | 空 | oGMemory 租户 Key，不是模型 Key |
| `OGMEM_TRACE_CAPTURE_CONTENT` | `1` | `0/false/no/off` 时 Trace 只保留结构和元数据 |

如果 Claude Code 运行目录能加载 `OgMemConfig`，插件优先使用配置中的 `account_id`、`user_id`、`agent_id`、`http_port`；`OG_MEMORY_URL` 始终可覆盖地址。

设置 `OG_AUTH_API_KEY` 后，插件发送：

```text
X-API-Key: <tenant-key>
X-Account-ID: <account-id>
X-User-ID: <user-id>
Content-Type: application/json
```

## 4. Hook 映射

| Claude Code Hook | 脚本 | oGMemory 调用 | 超时 |
| --- | --- | --- | --- |
| `SessionStart` | `call_add_session_message.py` | Trace `session_start` | 5s |
| `UserPromptSubmit` | `call_add_session_message.py` | Trace `user_message` | 5s |
| `UserPromptSubmit` | `call_compose.py` | `POST /api/v1/compose` | 15s |
| `PreToolUse` | `call_add_session_message.py` | Trace `tool_start` | 5s |
| `PostToolUse` | `call_add_session_message.py` | Trace `tool_success`；副作用工具追加 session message | 10s |
| `PostToolUseFailure` | `call_add_session_message.py` | Trace `tool_failure` | 5s |
| `Stop` | `call_add_session_message.py` | Trace `turn_end` | 5s |
| `Stop` | `stop_detach.sh` -> `call_after_turn.py` | 后台 `POST /after_turn` | 10s 派发 |
| `SubagentStop` | `call_add_session_message.py` | Trace `assistant_message` | 5s |
| `PreCompact` | 两个脚本 | Trace `pre_compact` + 同步 `/after_turn` | 5s + 600s |
| `SessionEnd` | `call_add_session_message.py` | Trace `session_end` | 5s |

Hook 定义：[`../../claude-plugin/hooks/hooks.json`](../../claude-plugin/hooks/hooks.json)。

## 5. Prompt 前自动召回

Claude Code 传给 `UserPromptSubmit` Hook 的 stdin 包含 `prompt` 和 `session_id`。插件发送：

```http
POST /api/v1/compose
Content-Type: application/json

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "claude-code",
  "sessionId": "claude-session-id",
  "prompt": "当前用户提示词"
}
```

插件依次读取：

1. `messages` 中 `_ogmem: true` 的消息。
2. 兼容字段 `systemPromptAddition`、`systemPromptSuffix`、`memoryUserMessage`。
3. 语义字段 `identityContext`、`episodicContext`、`sessionContext`、`retrievedEvidence`。

非空内容以 Claude Code Hook JSON 返回：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "...retrieved memory..."
  }
}
```

Claude Code 将 `additionalContext` 注入本轮上下文。compose 失败时脚本返回成功但不注入，主对话继续。

## 6. 一轮结束后自动写回

`call_after_turn.py` 读取 Claude Code transcript JSONL，只读取上次成功写回后的新增字节；偏移保存在 transcript 同目录的 `.ingest-offset` 侧车文件。

过滤规则：

- 只保留 user/assistant 文本。
- 跳过 system、progress、file-history-snapshot。
- assistant 内容数组只提取 `type: text` 块。
- 没有文本的消息跳过。

请求：

```http
POST /api/v1/after_turn
Content-Type: application/json

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "claude-code",
  "sessionId": "claude-session-id",
  "hook_event_name": "Stop",
  "messages": [
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

只有 HTTP 成功后才推进 offset。`Stop` 使用分离后台进程，避免阻塞 Claude Code；`PreCompact` 同步等待，防止压缩前内容丢失。

## 7. 工具消息

以下副作用工具成功执行后，其输入/输出摘要追加到会话缓冲：

```text
Write
Edit
MultiEdit
Bash
NotebookEdit
```

请求：

```http
POST /api/v1/sessions/<session-id>/messages
Content-Type: application/json

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "claude-code",
  "sessionId": "claude-session-id",
  "role": "tool",
  "content": "<bounded tool summary>"
}
```

Read、Glob、Grep 等只读工具不追加 session message；所有支持的工具 Hook 仍可记录 Trace。超长工具报文会截断。

## 8. Trace 报文

```http
POST /api/v1/trace/events
Content-Type: application/json

{
  "request_id": "event-id",
  "timestamp": "2026-08-03T17:00:00+08:00",
  "account_id": "acct-1",
  "user_id": "user-1",
  "agent_id": "claude-code",
  "session_id": "claude-session-id",
  "event": {
    "event_type": "tool_success",
    "object_id": "tool-call-id",
    "terminal_state": "success",
    "metadata": {"tool": "Write"}
  }
}
```

上传前不可逆脱敏：邮箱、手机号、证件号、银行卡号、IP、Bearer/API Key、Cookie、带凭据连接串替换为占位符。单个 Trace 请求上限 64 KiB。

Trace 失败会写入私有 spool 并尝试后续恢复；不会阻塞 Claude Code 主流程。

## 9. 手动命令

```text
/og-compose <关键词>
```

调用 `/api/v1/compose`，返回完整检索上下文。

```text
/og-add-history
```

读取并导入当前仓库历史对话，调用 `/api/v1/after_turn`；会提示成本并进行去重。

## 10. 验证

```bash
pytest tests/unit/claude_plugin/test_plugin_scripts.py -q
```

运行验证：

1. `/plugins` 能看到 `og-memory`。
2. `claude --debug` 日志出现 `[call_compose]`、`[call_after_turn]`、`[call_add_session_message]`。
3. UserPromptSubmit 时服务收到 `/api/v1/compose`。
4. Stop 时服务收到 `/api/v1/after_turn`。
5. 副作用工具执行后，同一 session 出现 `role: tool`。
6. transcript 目录出现 `.ingest-offset`。
7. `/og-compose <已写入关键词>` 能检索到此前内容。

## 11. 卸载

```text
/plugin uninstall og-memory@og-memory
/reload-plugins
```
