# oG-Memory Claude Code Plugin

将 Claude Code 对话自动同步到 oG-Memory，并在每次用户提问时检索相关记忆注入上下文。

## 工作原理

| Hook | 脚本 | 方向 | 说明 |
|------|------|------|------|
| `UserPromptSubmit` | `context_retrieve.py` | 读取 | 用户提交 prompt 时检索相关记忆，注入 additionalContext |
| `Stop` | `ingest_sync.py` (async) | 写入 | 对话结束时增量同步 transcript 到 oG-Memory |
| `PreCompact` | `ingest_sync.py` | 写入 | 压缩前同步，防止未同步的内容丢失 |

## 安装

在项目根目录启动 Claude Code 时通过 `--plugin-dir` 加载插件：

```bash
claude --plugin-dir ./claude-plugin
# 可开启 debug 模式
claude --plugin-dir ./claude-plugin --debug
```

## 前置条件

- oG-Memory 服务运行在 `localhost:8090`（可通过 `OG_MEMORY_URL` 环境变量覆盖）
- Python 3（仅使用标准库，无第三方依赖）

## 测试

### 1. 测试 context_retrieve.py（手动）

先启动 oG-Memory 服务，然后模拟 hook 输入：

```bash
echo '{"session_id":"test123","prompt":"重构 authentication 模块","hook_event_name":"UserPromptSubmit","transcript_path":"/tmp/test.jsonl","cwd":"/tmp","permission_mode":"default"}' \
  | python3 claude-plugin/scripts/context_retrieve.py
```

有记忆时输出：
```
{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[oG-Memory]\n..."}}
```

无记忆或服务不可达时：无输出，exit 0。

### 2. 测试 ingest_sync.py（手动）

```bash
# 需要一个真实的 transcript JSONL 文件
echo '{"session_id":"test123","prompt":"","hook_event_name":"Stop","transcript_path":"/path/to/real/transcript.jsonl","cwd":"/tmp"}' \
  | python3 claude-plugin/scripts/ingest_sync.py
```

### 3. 端到端测试（安装后）

1. 启动 oG-Memory
2. 在 Claude Code 中正常对话几轮（触发 ingest_sync 写入）
3. 开一个新对话，输入之前讨论过的话题
4. 检查 Claude 的回复是否体现了之前对话的记忆

### 4. 检查日志

**方式 A：使用 `--debug` 模式**

启动 Claude Code 时加 `--debug`，所有 hook 的 stderr 输出会出现在 debug 日志中：

然后在 debug 日志中搜索：
- `[context_retrieve]` — 读取侧（每次用户提交 prompt）
- `[ingest_sync]` — 写入侧（Stop / PreCompact 时）

**方式 B：查看 ingest_sync 的 sidecar 文件**

sidecar 文件与 transcript 同目录，后缀 `.ingest-offset`：

```bash
# 找到当前项目的 transcript 目录
ls ~/.claude/projects/*project-name/*.jsonl.ingest-offset

# 查看内容（数字 = 已同步的字节偏移）
cat ~/.claude/projects/*project-name/*.jsonl.ingest-offset
```

文件存在且偏移量 > 0 说明 ingest_sync 已成功写入。