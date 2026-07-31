# MCP Agent Scope

oGMemory keeps caller authentication separate from Agent memory scope:

- `Authorization: Bearer ...` or the legacy `X-API-Key` identifies the account, user, and role.
- The MCP tool argument `agent_id` selects the Agent memory scope. It is not a credential.

The effective Agent ID is resolved in this order:

1. The tool argument `agent_id` (standard and recommended).
2. `X-Agent-ID` (oGMemory HTTP compatibility extension).
3. `identity.agent_id` from the oGMemory configuration.

The selected Agent is still subject to the normal ownership and sharing checks. An API key does not grant access to an arbitrary `agent_id`.

## Standard MCP call

This form works with every MCP client because `agent_id` is part of the tool's input schema:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "ogm_write_memory",
    "arguments": {
      "session_id": "session-001",
      "text": "The user prefers Python.",
      "agent_id": "coding-agent"
    }
  }
}
```

For HTTP transports, send authentication separately:

```http
Authorization: Bearer <user-token>
Content-Type: application/json
Accept: application/json, text/event-stream
```

Python MCP SDK clients pass the same schema arguments:

```python
result = await session.call_tool(
    "ogm_write_memory",
    {
        "session_id": "session-001",
        "text": "The user prefers Python.",
        "agent_id": "python-client",
    },
)
```

## Configured default

A deployment that primarily serves one Agent may configure a default:

```yaml
identity:
  account_id: "acct-demo"
  user_id: "alice"
  agent_id: "main"
```

Calls may then omit `agent_id`; oGMemory uses `main`. If multiple Agents share the same endpoint and credentials, they must pass different `agent_id` arguments because the server cannot infer which client sent otherwise identical requests.

## HTTP compatibility header

Clients that cannot reliably add `agent_id` to every tool call may send:

```http
X-Agent-ID: openclaw
```

This is an oGMemory compatibility extension, not an MCP authentication header. An explicit tool argument always overrides it, and clients should prefer the tool argument whenever possible.
