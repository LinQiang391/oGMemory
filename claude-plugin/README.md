# oGMemory Claude Code 插件

将 Claude Code 对话同步到 oGMemory，在用户每次提问时检索相关记忆；成功工具调用的 I/O 会经 `PostToolUse` 以 `role: tool` 补录到同一会话，与 `after_turn` 基于 transcript 的 user/assistant 行互补。`call_compose.py` / `call_after_turn.py` 与 `call_add_session_message.py` 复用 `scripts/ogm_plugin_request.py`（`base_api_url`、请求头/身份、`base_ctx`），并在各脚本内将 `claude-plugin/scripts` 置前 `sys.path` 以稳定 `import ogm_plugin_request`。

## 工作原理

| Hook | 脚本 | 调用接口 | 方向 | 说明 |
|------|------|---------|------|------|
| `PostToolUse` | `call_add_session_message.py` | `/api/v1/sessions/{id}/messages` | 写入 | 仅 **有副作用** 的工具（Write / Edit / MultiEdit / Bash / NotebookEdit），matcher + 脚本白名单，避免 Read/Glob/Grep 刷屏；摘要过长截断 |
| `UserPromptSubmit` | `call_compose.py` | `/api/v1/compose` | 读取 | 用户提交 prompt 时检索相关记忆，注入 `additionalContext` |
| `Stop` | `call_after_turn.py` (async) 或 `stop_detach.sh` | `/api/v1/after_turn` | 写入 | 对话结束增量同步 transcript 到 oGMemory（JSONL 字节偏移），支持 headless -p 模式 |
| `PreCompact` | `call_after_turn.py` | `/api/v1/after_turn` | 写入 | 压缩前同步，防未落盘内容丢失 |

### Trace 捕获

`call_add_session_message.py` 还会把 Claude Code hook 转成统一的 trace event，并发送到 `POST /api/v1/trace/events`。进入服务端后只保留 canonical event 语义，不再依赖 Claude Code 的原生 hook 类型。兼容旧请求时，服务端仍会把 legacy `Stop` 当作 `turn_end` 边界处理。

| Claude Code hook | Canonical event |
| --- | --- |
| `SessionStart` | `session_start` |
| `UserPromptSubmit` | `user_message` |
| `PreToolUse` | `tool_start` |
| `PostToolUse` | `tool_success` |
| `PostToolUseFailure` | `tool_failure` |
| `Stop` | `turn_end` |
| `SubagentStop` | `assistant_message` |
| `PreCompact` | `pre_compact` |
| `SessionEnd` | `session_end` |

Trace 默认保存可诊断内容，但在 hook2Event 阶段会先做不可逆脱敏：邮箱、手机号、证件号、银行卡号、IP、Bearer/API key、cookie、带凭据的连接串等会替换成 `{{email_1}}`、`{{credential_1}}` 这类占位符。同一个 event 内相同敏感值复用同一个占位符；跨 event 不共享映射。若当前环境不希望捕获内容，可设置 `OGMEM_TRACE_CAPTURE_CONTENT=0`（也支持 `false` / `no` / `off`）只保留结构化边界和元信息。

## 配置与鉴权（可选）

- **基址：** 默认 `http://localhost:8090`，可用 `OG_MEMORY_URL` 覆盖。
- **与 OpenClaw 对齐的密钥：** 在 `claude` 进程环境中设置 `OG_AUTH_API_KEY` 时，请求会附带 `X-API-Key` 与 `X-Account-ID` / `X-User-ID`（身份解析见 `ogm_plugin_request.py`：可选 `OgMemConfig` 或环境变量回退）。服务端无密钥鉴权时可不设置，**禁止将密钥提交到仓库**。

## Skills

| 命令 | 说明 |
|------|------|
| `/og-compose <关键词>` | 内联 bash 调用 `/api/v1/compose`，返回完整记忆上下文 |
| `/og-add-history` | 内联 bash 调用 `/api/v1/after_turn`，导入本仓库历史对话（有成本提示与去重） |

**英文说明：** 见同目录 [README-EN.md](./README-EN.md)。

## 安装

### 通过 Marketplace 安装（推荐）

```bash
/plugin marketplace add https://gitcode.com/opengauss/oGMemory.git#dev
/plugin install og-memory@og-memory
/reload-plugins
```

更新时：

```bash
/plugin marketplace update og-memory
/plugin install og-memory@og-memory
/reload-plugins
```

### 本地开发

```bash
claude --plugin-dir ./claude-plugin
# 可选: --debug
```

## 前置条件

- oGMemory 服务可访问，例如 `curl -s http://localhost:8090/api/v1/health` 返回 `status: ok`（地址随 `OG_MEMORY_URL` 调整）
- Python 3；hook 仅标准库
- 单测（在仓库根）：`pytest tests/unit/claude_plugin/test_plugin_scripts.py -q`

## 快速验证

1. `/plugins` 中可看到已安装的 `og-memory`。
2. 对话多轮后执行 `/og-compose marketplace plugin`，能搜到此前内容即读侧正常；工具调用后服务侧同会话应出现 `role: tool` 补录；`~/.claude/projects/.../*.jsonl.ingest-offset` 可反映 `after_turn` 写入。

## 调试

- 执行 `/reload-plugins`，应看到 1 个 plugin 与若干 hooks。
- 使用 `--debug`，在输出中搜：`[call_compose]`、`[call_after_turn]`、`[call_add_session_message]`。
- `PreCompact` 与 `Stop` 的写入可对照上述 `ingest-offset` 与 transcript 同目录的侧车文件。
