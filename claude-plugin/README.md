# oG-Memory Claude Code Plugin

将 Claude Code 对话自动同步到 oG-Memory，并在每次用户提问时检索相关记忆注入上下文。

## 工作原理

| Hook | 脚本 | 调用接口 | 方向 | 说明 |
|------|------|---------|------|------|
| `UserPromptSubmit` | `call_compose.py` | `/api/v1/compose` | 读取 | 用户提交 prompt 时检索相关记忆，注入 additionalContext |
| `Stop` | `call_after_turn.py` (async) | `/api/v1/after_turn` | 写入 | 对话结束时增量同步 transcript 到 oG-Memory |
| `PreCompact` | `call_after_turn.py` | `/api/v1/after_turn` | 写入 | 压缩前同步，防止未同步的内容丢失 |

## Skills

| 命令 | 说明 |
|------|------|
| `/og-compose <关键词>` | 通过内联 bash 调用 `/api/v1/compose`，返回完整记忆上下文（profile + archives + session + working set） |
| `/og-add-history` | 通过内联 bash 调用 `/api/v1/after_turn`，添加当前项目的历史 Claude 对话到 oG-Memory（执行前估算成本，已处理的自动跳过） |

## 安装

### 通过 Marketplace 安装（推荐）

```bash
/plugin marketplace add https://gitcode.com/akushonkamen/oG-Memory.git#dev
/plugin install og-memory@og-memory
/reload-plugins
```

更新插件版本时：

```bash
/plugin marketplace update og-memory
/plugin install og-memory@og-memory
/reload-plugins
```

### 本地开发安装

```bash
claude --plugin-dir ./claude-plugin
# 可开启 debug 模式
claude --plugin-dir ./claude-plugin --debug
```

## 前置条件

- oG-Memory 服务运行在 `localhost:8090`（可通过 `OG_MEMORY_URL` 环境变量覆盖），可通过 `curl -s http://localhost:8090/api/v1/health` 确认（应返回 `status: ok`）
- Python 3（仅使用标准库，无第三方依赖）

## 快速验证

### 1. 确认插件已安装

在 Claude Code 中输入 `/plugins`，搜索 `og-memory` 应能看到已安装。

### 2. 验证读写正常

正常对话几轮，然后用 skill 搜索：

```bash
/og-compose marketplace plugin
```

能搜到之前对话的内容即表示写入和读取都正常工作。

## 调试

### 确认插件已加载

在 Claude Code 中执行：

```bash
/reload-plugins
```

输出应包含 `1 plugin` 和相关 hooks 数量。

### 检查数据是否正常写入

正常对话后，检查 ingest offset 文件（存在且偏移量 > 0 即表示已写入）：

```bash
ls ~/.claude/projects/*oG-Memory/*.jsonl.ingest-offset
cat ~/.claude/projects/*oG-Memory/*.jsonl.ingest-offset
```

### 查看 debug 日志

启动时加 `--debug`，在日志中搜索：
- `[call_compose]` — 读取侧（每次用户提交 prompt）
- `[call_after_turn]` — 写入侧（Stop / PreCompact 时）
