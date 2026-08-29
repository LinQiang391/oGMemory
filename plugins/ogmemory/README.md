# oGMemory Codex plugin

Codex plugin integration over the oGMemory HTTP API.

## Lifecycle coverage

| Codex hook | oGMemory use |
| --- | --- |
| `SessionStart` | Bootstrap the session and inject durable workspace context |
| `UserPromptSubmit` | Capture the prompt and inject relevant memory |
| `PreToolUse` | Capture tool intent and inject relevant constraints and prior outcomes |
| `PermissionRequest` | Capture permission request context as lifecycle trace data |
| `PostToolUse` | Capture bounded tool input and output |
| `PreCompact` | Prepare extraction and compaction state |
| `PostCompact` | Commit oGMemory compaction state |
| `SubagentStart` | Prepare the subagent and inject relevant context |
| `SubagentStop` | Capture the subagent result and close its lifecycle |
| `Stop` | Persist the completed user/assistant turn |
| `SessionEnd` | Finalize trace data and dispose the session |

Every event is also sent to `/api/v1/trace/events` for turn reconstruction and reusable-skill extraction. Trace content is bounded and common credential fields are redacted.

## Service connection

The plugin can connect to an existing remote oGMemory service. Create `.ogmemory/codex-plugin.json` in the user home directory with the connection details supplied by the service administrator. Use an empty `api_key` when authentication is disabled:

```json
{
  "url": "https://memory.example.com",
  "api_key": "<tenant-key>",
  "account_id": "<account-id>",
  "user_id": "<user-id>",
  "agent_id": "codex"
}
```

For local development, install and start oGMemory with `ogmem onboard --mode headless`. The default local service URL is `http://127.0.0.1:8090`; local default settings require no configuration file.

## Install from this repository

```text
codex plugin marketplace add https://gitcode.com/opengauss/oGMemory.git --ref dev --sparse .agents/plugins --sparse plugins/ogmemory
codex plugin add ogmemory@ogmemory
```

Refresh Codex, install `ogmemory`, then open `/hooks` and trust its lifecycle hooks.

## Verify

- `/hooks` lists all 11 lifecycle events as trusted.
- A new session receives recalled workspace context.
- Tool calls appear in the oGMemory session trace.
- A completed turn becomes recallable in a later Codex session.
- Compaction, subagent, and session-end events appear in the trace.

The bridge is fail-open: an unavailable memory service does not block Codex.
