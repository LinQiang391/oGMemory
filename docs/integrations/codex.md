# Codex 接入 oGMemory

Codex 插件支持连接远端或本机 oGMemory 服务。

## 1. 准备环境

需要以下软件：

- 支持插件的 Codex Desktop 或 Codex CLI
- Python 3.11 或更高版本，用于运行插件 Hook

## 2. 配置远端服务

在用户主目录创建 `.ogmemory/codex-plugin.json`：

```json
{
  "url": "https://memory.example.com",
  "api_key": "<tenant-key>",
  "account_id": "<account-id>",
  "user_id": "<user-id>",
  "agent_id": "codex"
}
```

配置要求：

- `url` 使用 HTTPS，末尾不包含 `/api/v1`
- 未启用鉴权时，`api_key` 使用空字符串
- `account_id`、`user_id` 和 `agent_id` 使用服务端配置的身份

`~/.ogmemory/codex-plugin.json` 中的 `~` 表示当前用户主目录。不要提交包含密钥的配置文件。

## 3. 部署并配置本机服务

安装 Git，克隆并配置 oGMemory：

```text
git clone -b dev https://gitcode.com/opengauss/oGMemory.git
cd oGMemory
python -m pip install -e .
ogmem onboard --mode headless
```

若引导程序中未立即启动服务，执行：

```text
ogmem start headless -d
```

检查服务：

```text
ogmem status
```

预期结果包含：

```text
ContextEngine   http://127.0.0.1:8090  UP
```

命令中的 `python` 指向本机 Python 3.11+ 解释器；若系统命令名为 `python3`，使用 `python3`。

本机默认服务可省略配置文件，插件使用以下默认值：

```json
{
  "url": "http://127.0.0.1:8090",
  "api_key": "",
  "account_id": "acct-demo",
  "user_id": "u-codex",
  "agent_id": "codex"
}
```

也可使用环境变量覆盖配置文件：

| 环境变量 | 配置字段 |
| --- | --- |
| `OG_MEMORY_URL` | `url` |
| `OG_AUTH_API_KEY` | `api_key` |
| `OG_MEMORY_ACCOUNT_ID` | `account_id` |
| `OG_MEMORY_USER_ID` | `user_id` |
| `OG_MEMORY_AGENT_ID` | `agent_id` |

## 4. 安装插件

添加仓库插件市场并安装 `ogmemory`：

```text
codex plugin marketplace add https://gitcode.com/opengauss/oGMemory.git --ref dev --sparse .agents/plugins --sparse plugins/ogmemory
codex plugin add ogmemory@ogmemory
codex plugin list
```

预期插件列表包含：

```text
ogmemory@ogmemory
```

重启 Codex。打开 `/hooks`，检查并信任 `ogmemory` 的 11 个生命周期 Hook。

## 5. 验证跨会话记忆

在第一个 Codex 任务中输入：

```text
请记住：Codex 插件验收码是 OG-CODEX-2026。只回复“已记录”。
```

预期回复：

```text
已记录
```

该轮完成后，新建 Codex 任务并输入：

```text
Codex 插件验收码是什么？只回复验收码。
```

预期回复：

```text
OG-CODEX-2026
```

若结果不符，检查 `/hooks` 信任状态、服务地址、API Key 和身份字段。本机部署还可检查：

```text
ogmem status
ogmem logs --service ogmem
```

## 生命周期映射

| Codex 阶段 | oGMemory 接口 | 用途 |
| --- | --- | --- |
| `SessionStart` | `/bootstrap`、`/compose` | 初始化会话并注入相关记忆 |
| `UserPromptSubmit` | `/compose` | 记录用户输入并召回当前任务记忆 |
| `PreToolUse` | `/compose` | 记录工具意图并召回相关约束 |
| `PermissionRequest` | `/trace/events` | 记录权限申请上下文 |
| `PostToolUse` | `/trace/events` | 记录有界的工具输入与输出 |
| `PreCompact` | `/prepare_compaction` | 准备提取和压缩状态 |
| `PostCompact` | `/compact` | 提交压缩状态 |
| `SubagentStart` | `/prepare_subagent_spawn`、`/compose` | 初始化子代理并注入相关记忆 |
| `SubagentStop` | `/on_subagent_ended` | 记录子代理结果并结束生命周期 |
| `Stop` | `/after_turn` | 持久化完整用户与助手轮次 |
| `SessionEnd` | `/dispose` | 刷新待处理记忆并释放会话 |

所有阶段同时写入 `/api/v1/trace/events`。轨迹内容有长度上限，并对常见凭据字段脱敏。Hook 失败时不会阻塞 Codex；注入的记忆作为不可信上下文处理。
