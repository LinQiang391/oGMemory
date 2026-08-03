# oGMemory MCP SSE Python 接入示例

本目录提供两份功能相同的 SSE 客户端：

- `stdlib/ogmemory_mcp_sse_demo.py`：只用 Python 标准库，完整展示 SSE 长连接、消息端点和 JSON-RPC 交互。
- `sdk/ogmemory_mcp_sse_demo.py`：使用官方 MCP Python SDK，适合直接集成到自研 Agent。

两者均支持工具发现、查询、读取、写入/修改和软删除。修改记忆不需要单独的 update 工具；继续调用 `ogm_write_memory` 提交用户最新的纠正信息，由 oGMemory 完成抽取、去重和演化。

## 0. 服务端口约定

本示例与验证服务器统一使用 SSE 端口 `8191`。该端口不是 oGMemory 的默认 MCP 端口。本机隔离调试可显式启用 HTTP 鉴权和完整工具集：

```bash
export OGMEM_MCP_PORT=8191
export OG_ROLE_CONTROL_ENABLED=true
export OG_ROOT_API_KEY='local-debug-root-key'
export OGMEM_MCP_DEFAULT_TOOLSET=all
python -m cli.main mcp \
  --transport sse \
  --host 0.0.0.0 \
  --port "$OGMEM_MCP_PORT" \
  --path /mcp
```

上面的 root key 仅限本机隔离调试。生产环境必须由服务管理员启用鉴权并向 Agent 交付租户级 admin/member key；Agent 不得持有 root key，root key 仅用于管理和初始化。如果服务提供方使用其他端口，以其交付的 MCP URL 为准，并通过 `OGMEM_MCP_URL` 覆盖示例值。

## 1. 客户端需要配置什么

复制 `.env.example` 中的变量到进程环境。示例程序只读取环境变量，不自动读取 `.env`，也不会打印认证请求头。

```bash
export OGMEM_MCP_URL='http://127.0.0.1:8191/mcp'
export OGMEM_MCP_API_KEY='<tenant-member-key>'
export OGMEM_MCP_ACCOUNT_ID='acct-demo'
export OGMEM_MCP_USER_ID='shared-agent-user'
export OGMEM_MCP_AGENT_ID='shared-agent'
```

连接前述本机调试服务时，将 `OGMEM_MCP_API_KEY` 改为 `local-debug-root-key`。生产环境保持使用服务提供方交付的租户级 key。

字段来源：

| 配置                   | 含义与来源                                                   |
| ---------------------- | ------------------------------------------------------------ |
| `OGMEM_MCP_URL`        | oGMemory SSE 地址。部署在哪里就填写哪里。 |
| `OGMEM_MCP_API_KEY`    | oGMemory 服务提供方交付的访问凭证，对应请求头 `X-API-Key`；它不是模型 API Key。 |
| `OGMEM_MCP_ACCOUNT_ID` | oGMemory 服务提供方预先配置并交付的租户/业务账号 ID，对应 `X-Account-ID`。 |
| `OGMEM_MCP_USER_ID`    | oGMemory 服务提供方预先配置并交付的用户 ID，对应 `X-User-ID`。 |
| `OGMEM_MCP_AGENT_ID`   | oGMemory 服务提供方预先分配、授权并交付的 Agent ID。客户端会同时将其作为 `X-Agent-ID` 请求头和各记忆工具的默认 `agent_id`。 |

生产环境中，MCP URL、访问凭证、Account ID、User ID 和 Agent ID 均由 oGMemory 服务提供方预先配置并安全交付。客户端应原样使用这些参数，无需执行用户或 Agent 注册，也不应持有服务端管理凭证。`OGMEM_MCP_AGENT_ID` 必须是已授权的 Agent 作用域，不能由客户端随意填写。

## 2. 标准库版本

无需安装依赖：

```bash
python3 stdlib/ogmemory_mcp_sse_demo.py list-tools
python3 stdlib/ogmemory_mcp_sse_demo.py query '我叫什么名字？' --top-k 5
python3 stdlib/ogmemory_mcp_sse_demo.py read \
  'ogmemory://memories/preference/REPLACE_ID'
```

写入一条用户信息：

```bash
python3 stdlib/ogmemory_mcp_sse_demo.py write \
  --session-id demo-session-001 \
  --text '用户喜欢喝无糖咖啡。' \
  --force-extract \
  --wait
```

短文本默认会在同一 session 中累积，达到服务端 `after_turn_threshold` 后再抽取。需要立即处理时同时传入 `--force-extract --wait`：前者绕过本次写入的抽取阈值，后者等待抽取任务完成。不传 `--force-extract` 时仍保持批量抽取行为，短文本可能返回 `status: "accumulating"`，此时尚未形成可查询的持久记忆。

用户纠正旧信息时仍调用 write：

```bash
python3 stdlib/ogmemory_mcp_sse_demo.py write \
  --session-id demo-session-002 \
  --text '用户现在喜欢喝拿铁，不再喝无糖美式。' \
  --force-extract \
  --wait
```

删除前应先 query/read 确认准确 URI：

```bash
python3 stdlib/ogmemory_mcp_sse_demo.py delete \
  'ogmemory://memories/preference/REPLACE_ID' \
  --reason '用户明确要求遗忘' \
  --yes
```

## 3. 官方 MCP Python SDK 版本

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r sdk/requirements.txt

python sdk/ogmemory_mcp_sse_demo.py list-tools
python sdk/ogmemory_mcp_sse_demo.py query '我叫什么名字？' --top-k 5
python sdk/ogmemory_mcp_sse_demo.py write \
  --session-id sdk-session-001 \
  --text '用户的称呼是大王。' \
  --force-extract \
  --wait
```

自研 Agent 的核心连接代码就是：

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

headers = {
    "X-API-Key": settings.ogmemory_api_key,
    "X-Account-ID": request_context.account_id,
    "X-User-ID": request_context.user_id,
    "X-Agent-ID": settings.agent_id,
}

async with sse_client(settings.ogmemory_mcp_url, headers=headers) as streams:
    read_stream, write_stream = streams
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool(
            "ogm_query_memory",
            {"query": user_message, "top_k": 5, "agent_id": settings.agent_id},
        )
```

Agent 启动时调用 `tools/list`，把返回的工具描述和 `inputSchema` 注册给模型。模型产生 tool call 后，客户端执行 `tools/call`，再把工具结果作为 tool message 返回模型。两个示例客户端都会把 `OGMEM_MCP_AGENT_ID` 作为 query/read/write/delete 的默认 `agent_id`；需要临时访问另一个已授权 Agent 作用域时，可在对应命令后传入 `--agent-id OTHER_AGENT` 覆盖。

## 4. OpenClaw SSE 配置

OpenClaw 的 `openclaw.json` 可配置为：

```json
{
  "mcp": {
    "servers": {
      "ogmemory": {
        "url": "http://127.0.0.1:8191/mcp",
        "transport": "sse",
        "headers": {
          "X-API-Key": "<tenant-member-key>",
          "X-Account-ID": "acct-demo",
          "X-User-ID": "shared-agent-user",
          "X-Agent-ID": "shared-agent"
        },
        "connectTimeout": 30,
        "timeout": 120
      }
    }
  }
}
```

使用 oGMemory 服务提供方交付的连接参数配置 OpenClaw。`X-Agent-ID` 必须填写服务方分配且已授权的 Agent ID；它只用于选择记忆作用域，不是认证凭证。OpenClaw 客户端无需持有任何服务端管理凭证。

修改后重启或 reload，并验证：

```bash
openclaw mcp status --json
openclaw mcp probe ogmemory --json
```

成功标准不是模型回答中声称“使用了 MCP”，而是：

1. `mcp status` 显示 `transport: sse` 且 `ok: true`。
2. `mcp probe` 列出 `ogmemory__ogm_query_memory` 等工具且无 diagnostics。
3. Agent JSON 结果的 `toolSummary.tools` 包含实际工具名，`failures` 为 0。
4. oGMemory 日志同时出现 `GET /mcp` 和 `POST /messages/?session_id=...`。
