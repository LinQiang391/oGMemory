# HTTP REST API 对接

## 1. 启动服务

AGFS 模式：

```bash
ogmem start local
```

PostgreSQL/openGauss 模式：

```bash
python server/app.py
```

生产进程：

```bash
gunicorn -w 1 -b 0.0.0.0:8090 server.app:app
```

会话缓冲和两阶段压缩的 `prepareToken` 当前保存在进程内。不要直接增加
Gunicorn worker 数；多 worker 部署必须确保同一会话使用粘性路由，或先将
会话状态迁移到共享存储。

默认基址：`http://127.0.0.1:8090`。

健康检查：

```http
GET /api/v1/health HTTP/1.1
Host: 127.0.0.1:8090
```

## 2. 鉴权

角色控制关闭时，请求体中的 `accountId`、`userId`、`agentId` 用于构造 `RequestContext`。

角色控制开启时，请求头决定租户和用户身份：

```http
X-API-Key: <tenant-key>
X-Account-ID: acct-1
X-User-ID: user-1
Content-Type: application/json
```

请求体仍传 `agentId` 和 `sessionId`。`agentId` 必须在该用户可见的 Agent 列表中。

通用错误：

| HTTP 状态 | 含义 |
| --- | --- |
| `400` | 参数或 JSON 不合法 |
| `401` | API Key 缺失或无效 |
| `403` | 租户、用户或 Agent 无权限；来源 IP 不允许 |
| `404` | 资源不存在 |
| `413` | Trace 报文超过 64 KiB或嵌套过深 |
| `500` | 服务内部错误 |
| `503` | 控制面或依赖不可用 |

错误报文：

```json
{"error": "error message"}
```

### 2.1 Skill 手动导入权限

手动导入 Skill 使用以下 HTTP REST API：

```http
POST /api/v1/skills?agent_id=agent-1 HTTP/1.1
X-API-Key: <tenant-admin-key>
X-Account-ID: acct-1
X-User-ID: admin-1
Content-Type: application/json

{
  "data": {
    "name": "review-code",
    "description": "Review code changes",
    "content": "# Review code\n\nFollow the review checklist."
  }
}
```

也可以直接上传 ZIP 格式的完整 Skill 包。ZIP 文件大小和解压后的包大小均不得超过
20 MiB，包内必须在根目录或单一外层目录中包含 `SKILL.md`：

```bash
curl -X POST "http://127.0.0.1:8090/api/v1/skills?agent_id=agent-1" \
  -H "X-API-Key: <tenant-admin-key>" \
  -H "X-Account-ID: acct-1" \
  -H "X-User-ID: admin-1" \
  -F "file=@review-code.zip;type=application/zip"
```

也可将 ZIP 作为 `application/zip` 请求体直接发送。上传后，ZIP 中的 Skill 目录
作为一个完整 Skill 安装；`SKILL.md`、`scripts/`、`references/`、`assets/` 等文件
均可通过 Skill 文件 API 读取。

`/api/v1/skills/validate` 支持相同的两种 ZIP 上传方式，且只校验、不持久化。
校验返回 `valid`、`errors`、`warnings`，并支持
`strict`、`skill_dir_name`、`source_path`。ZIP 上传时自动使用唯一外层目录名执行
名称匹配校验。超过大小限制时返回 `413`。

`agent_id` 必须通过查询参数或 `X-Agent-ID` 请求头提供。导入成功返回
`201 Created`，Skill 存储在当前 account 的 Agent 命名空间：
`ctx://{account_id}/agents/{agent_id}/skills/imported/{name}`。

开启角色控制时，权限如下：

| 角色 | 是否可导入 | 作用范围 |
| --- | --- | --- |
| `ROOT` | 是 | 可通过 `X-Account-ID` 选择 account，并导入到指定 Agent |
| `ADMIN` | 是 | 仅限 API Key 所属 account，且 Agent 必须对该管理员可见 |
| `MEMBER` | 否 | 即使是导入到自己可见的 Agent，也返回 `403` |

Skill 采用 Agent 级命名空间。普通用户没有可自行管理的“个人 Skill”
命名空间。

可在导入前校验完整 Skill 包，该接口不持久化数据：

```http
POST /api/v1/skills/validate HTTP/1.1
X-API-Key: <tenant-key>
X-Account-ID: acct-1
X-User-ID: user-1
Content-Type: application/json

{"data": {"name": "review-code", "description": "Review code changes", "content": "# Review code"}}
```

全量替换已导入 Skill package（不修改同名 generated Skill）：

```http
PUT /api/v1/skills/review-code?agent_id=agent-1 HTTP/1.1
X-API-Key: <tenant-admin-key>
X-Account-ID: acct-1
X-User-ID: admin-1
Content-Type: application/json

{
  "data": {
    "files": [
      {
        "path": "SKILL.md",
        "content": "---\nname: review-code\ndescription: Updated review flow\n---\n\n# Review code\n\nUse scripts/check.py."
      },
      {
        "path": "scripts/check.py",
        "content": "print('updated review check')\n",
        "media_type": "text/x-python"
      },
      {
        "path": "assets/rules.bin",
        "content_base64": "AAEC",
        "media_type": "application/octet-stream"
      }
    ]
  }
}
```

也可直接使用 ZIP 更新：

```bash
curl -X PUT "http://127.0.0.1:8090/api/v1/skills/review-code?agent_id=agent-1" \
  -H "X-API-Key: <tenant-admin-key>" \
  -H "X-Account-ID: acct-1" \
  -H "X-User-ID: admin-1" \
  -F "file=@review-code.zip;type=application/zip"
```

PUT 同时支持 `application/zip` 原始请求体。更新是完整 package 替换，不是单文件
PATCH：请求必须重新提交所有需要保留的文件；旧 package 中存在、但新 package
未包含的文件会被移除。package 必须包含 `SKILL.md`，请求路径必须与其中的
`name` 一致。成功响应的 `action` 为 `update`，`version` 加一。目标 imported
Skill 不存在时返回 `404`；并发更新冲突返回 `409`。

删除已发布 Skill 时必须明确来源，避免同名的 generated/imported Skill 被误删：

```http
DELETE /api/v1/skills/review-code?source=imported&agent_id=agent-1 HTTP/1.1
X-API-Key: <tenant-admin-key>
X-Account-ID: acct-1
X-User-ID: admin-1
```

`source` 必须是 `generated` 或 `imported`。删除权限与手动导入相同。

角色控制关闭时不执行上述角色检查；该模式只适用于已通过防火墙或安全组
隔离的本地验证环境。

## 3. 标准 Agent 生命周期

```text
用户消息
  -> POST /prefetch（可选）
  -> POST /compose
  -> 将返回的 _ogmem 消息注入模型上下文
  -> 模型生成回复
  -> POST /after_turn
  -> 压缩前 POST /prepare_compaction -> POST /compact
  -> 会话结束 POST /dispose
```

## 4. `POST /api/v1/compose`

检索记忆并返回已注入记忆的消息列表。

请求字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `accountId` | string | 否 | 配置值 | 租户 ID |
| `userId` | string | 否 | 配置值 | 用户 ID |
| `agentId` | string | 否 | 配置值 | Agent ID |
| `sessionId` | string | 建议 | `unknown` | 当前会话 |
| `prompt` | string | 二选一 | `""` | 当前用户问题 |
| `query` | string | 二选一 | `""` | `prompt` 的别名 |
| `messages` | array | 二选一 | `[]` | 当前消息；未传 prompt/query 时从消息提取查询 |
| `tokenBudget` | integer | 否 | `128000` | 本轮上下文预算 |

请求：

```http
POST /api/v1/compose HTTP/1.1
Host: 127.0.0.1:8090
Content-Type: application/json

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "prompt": "用户偏好的开发语言是什么？",
  "messages": [
    {"role": "user", "content": "用户偏好的开发语言是什么？"}
  ],
  "tokenBudget": 32000
}
```

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `messages` | array | 原消息与记忆消息；记忆消息带 `_ogmem: true` |
| `estimatedTokens` | integer | 组装后估算 token |
| `archiveCount` | integer | 命中的会话归档数量 |
| `archiveIncluded` | boolean | 是否注入归档 |
| `identityContext` | string | Profile 层 |
| `episodicContext` | string | 历史会话层 |
| `sessionContext` | string | 当前会话状态层 |
| `taskContext` | string | 任务上下文 |
| `retrievedEvidence` | string | 动态检索结果 |
| `openLoops` | array | 未闭环事项 |
| `uncertainties` | array | 不确定信息 |
| `budgetUsedBySlot` | object | 各层预算使用量 |
| `systemPromptAddition` | string | 当前实现为空 |
| `systemPromptSuffix` | string | 当前实现为空 |
| `memoryUserMessage` | string | 当前实现为空 |

响应示例：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "## Profile\n用户主要使用 Python。",
      "_ogmem": true
    },
    {"role": "user", "content": "用户偏好的开发语言是什么？"},
    {
      "role": "user",
      "content": "## Retrieved Memories\n- [preference] 用户偏好 Python。",
      "_ogmem": true
    }
  ],
  "systemPromptAddition": "",
  "systemPromptSuffix": "",
  "memoryUserMessage": "",
  "estimatedTokens": 84,
  "archiveCount": 0,
  "archiveIncluded": false,
  "identityContext": "## Profile\n用户主要使用 Python。",
  "episodicContext": "",
  "sessionContext": "",
  "taskContext": "",
  "retrievedEvidence": "## Retrieved Memories\n- [preference] 用户偏好 Python。",
  "openLoops": [],
  "uncertainties": [],
  "budgetUsedBySlot": {},
  "stats": {}
}
```

客户端处理：保留原消息顺序，直接使用响应的 `messages` 调模型；或只抽取 `_ogmem: true` 的内容放入单独的“不可信记忆上下文”区。

## 5. `POST /api/v1/prefetch`

在 compose 前快速检索，并把候选暂存到当前会话 TopicBuffer。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 身份字段 | string | 同 compose | 配置值 | account/user/agent/session |
| `prompt` / `query` | string | 二选一 | `""` | 查询文本 |
| `messages` | array | 否 | `[]` | 查询文本为空时用于提取查询 |
| `prefetchTopK` | integer | 否 | 配置值，通常 `5` | 候选数 |

```json
{
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "query": "Python 偏好",
  "prefetchTopK": 5
}
```

```json
{"ok": true, "prefetched": 3}
```

未启用 prefetch：

```json
{"ok": true, "prefetched": 0, "reason": "disabled"}
```

## 6. `POST /api/v1/after_turn`

把一轮完整对话写入会话缓冲；达到阈值或强制抽取时生成长期记忆。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 身份字段 | string | 建议 | 配置值 | account/user/agent/session |
| `messages` | array | 是 | `[]` | 本轮消息 |
| `messages[].role` | string | 是 | `user` | `system`、`user`、`assistant`、`tool` |
| `messages[].content` | string/object | 是 | `""` | 消息内容 |
| `messages[].created_at` | string | 否 | 当前时间 | ISO-8601 时间 |
| `prePromptMessageCount` | integer | 否 | `0` | 跳过前置系统/记忆消息数量 |
| `forceExtract` | boolean | 否 | `false` | 忽略抽取阈值，立即抽取 |

```http
POST /api/v1/after_turn HTTP/1.1
Content-Type: application/json

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "messages": [
    {
      "role": "user",
      "content": "以后 Python 项目统一使用 uv。",
      "created_at": "2026-08-03T17:00:00+08:00"
    },
    {"role": "assistant", "content": "收到。"}
  ],
  "forceExtract": true
}
```

常见响应：

```json
{"ok": true, "status": "accumulating", "pending_tokens": 37}
```

```json
{
  "ok": true,
  "status": "processing",
  "extraction_run_id": "a1b2c3d4e5f6"
}
```

```json
{
  "ok": true,
  "status": "completed",
  "extraction_run_id": "a1b2c3d4e5f6",
  "candidates_extracted": 1,
  "candidates_filtered": 0,
  "writes_completed": 1,
  "writes_skipped": 0,
  "writes_failed": 0
}
```

`accumulating` 表示消息已接收，但尚未达到 `after_turn_threshold`，此时不保证可立即检索。

## 7. `POST /api/v1/ingest` 与 `/ingest_batch`

仅把消息加入会话缓冲，不做长期记忆抽取。`ingest_batch` 与 `ingest` 使用同一报文。

```json
{
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "messages": [
    {"role": "user", "content": "第一条消息"},
    {"role": "assistant", "content": "第一条回复"}
  ]
}
```

```json
{"ingested": true, "count": 2}
```

之后仍需调用 `after_turn` 或 `compact`。`after_turn` 会对已 ingest 的相同 role/content 去重。

## 8. 两阶段压缩

### 8.1 `POST /api/v1/prepare_compaction`

先抽取压缩边界前尚未处理的消息，签发一次性 token。

```json
{
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "summaryMaxChars": 4000,
  "shortTermIndexMode": "sync"
}
```

```json
{
  "ok": true,
  "prepared": true,
  "prepareToken": "<one-time-token>",
  "messagesPrepared": 2
}
```

### 8.2 `POST /api/v1/compact`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `prepareToken` | string | 推荐 | 无 | prepare 返回的一次性 token |
| `prepared` | boolean | 否 | `false` | 为 true 时必须同时传 token |
| `tokenBudget` | integer | 否 | `128000` | 压缩后上下文预算 |
| `summaryMaxChars` | integer | 否 | 配置值 | summary 字符上限 |
| `shortTermIndexMode` | string | 否 | 配置值 | `sync`、`async`、`off` |

```json
{
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001",
  "prepareToken": "<one-time-token>",
  "tokenBudget": 32000,
  "summaryMaxChars": 4000,
  "shortTermIndexMode": "sync"
}
```

```json
{
  "ok": true,
  "compacted": true,
  "result": {
    "summary": "压缩后的会话摘要",
    "tokensBefore": 8200,
    "tokensAfter": 640,
    "firstKeptEntryId": "optional-entry-id"
  }
}
```

token 无效或已消费：`400 {"error":"invalid prepare token"}`。

## 9. `POST /api/v1/dispose`

会话结束时释放内存态；未处理内容可能在后台刷新。

```json
{
  "userId": "user-1",
  "agentId": "agent-1",
  "sessionId": "session-20260803-001"
}
```

```json
{
  "disposed": true,
  "flushed": false,
  "status": "background_flush_dispatched",
  "reason": "background_flush_dispatched",
  "pending_tokens": 860
}
```

`dispose` 是非阻塞接口。成功返回不表示后台抽取已经完成。

## 10. Session API

### 添加单条消息

```http
POST /api/v1/sessions/session-1/messages
Content-Type: application/json

{"userId":"user-1","agentId":"agent-1","role":"tool","content":"写入文件 a.py"}
```

`content` 必填；`role` 默认 `user`；可传 `created_at`。

### 获取会话

```http
GET /api/v1/sessions/session-1?userId=user-1&agentId=agent-1
```

### 提交会话

```http
POST /api/v1/sessions/session-1/commit
Content-Type: application/json

{"userId":"user-1","agentId":"agent-1","wait":true}
```

### 获取组装上下文

```http
GET /api/v1/sessions/session-1/context?userId=user-1&agentId=agent-1&token_budget=32000
```

## 11. `POST /api/v1/trace/events`

请求体最大 64 KiB。`request_id` 和 `session_id` 必填。

```json
{
  "request_id": "event-0001",
  "timestamp": "2026-08-03T17:00:00+08:00",
  "account_id": "acct-1",
  "user_id": "user-1",
  "agent_id": "agent-1",
  "session_id": "session-20260803-001",
  "event": {
    "event_type": "tool_success",
    "object_id": "tool-call-1",
    "terminal_state": "success",
    "metadata": {"tool": "write_file"}
  }
}
```

常用 `event_type`：`session_start`、`user_message`、`assistant_message`、`tool_start`、`tool_success`、`tool_failure`、`pre_compact`、`turn_end`、`session_end`。

`turn_end` 会触发当前 turn 的 Trace finalize；`session_end` 会唤醒 Skill 沉淀流程。

## 12. 客户端最小实现

```python
import requests

BASE = "http://127.0.0.1:8090"
IDENTITY = {
    "accountId": "acct-1",
    "userId": "user-1",
    "agentId": "agent-1",
    "sessionId": "session-1",
}

def post(path, payload):
    response = requests.post(
        BASE + path,
        json={**IDENTITY, **payload},
        headers={
            "X-API-Key": "<tenant-key>",
            "X-Account-ID": IDENTITY["accountId"],
            "X-User-ID": IDENTITY["userId"],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

def before_model(user_text, messages):
    return post("/api/v1/compose", {
        "prompt": user_text,
        "messages": messages,
        "tokenBudget": 32000,
    })["messages"]

def after_model(user_text, assistant_text):
    return post("/api/v1/after_turn", {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    })
```

生产客户端应对 `408`、`429`、`5xx` 使用有上限的指数退避；对 `400`、`401`、`403` 不自动重试；compose/write 失败时 Agent 主对话 fail-open。
