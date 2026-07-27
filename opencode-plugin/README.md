# oGMemory OpenCode Plugin

An OpenCode plugin that makes oGMemory retrieval, conversation write-back, and canonical Trace capture automatic. It integrates above MCP: MCP remains available for explicit memory operations, but normal context lifecycle management does not depend on the model choosing a tool.

## Requirements

- OpenCode 1.1.13 or later
- Node.js 18 or later
- A running oGMemory HTTP service

The minimum OpenCode version is required because the plugin uses the session-aware `experimental.chat.system.transform` hook. If a detected version is older, the plugin disables itself. If version detection is unavailable, it logs a warning and continues fail-open. OpenCode added the optional `dispose` hook in 1.15.11; versions 1.1.13 through 1.15.10 retain disk durability and normal lifecycle synchronization but do not perform the extra bounded graceful-shutdown flush.

## Install from npm

After the package is published, add it to `~/.config/opencode/opencode.json`:

```json
{
  "plugin": ["@opengauss/ogmemory-opencode-plugin"]
}
```

Publishing the package is intentionally independent of merging this source package.

## Install from source

Copy this directory and its wrapper into OpenCode's global plugin directory:

```text
~/.config/opencode/plugins/
├── ogmemory.js
└── ogmemory/
    ├── index.mjs
    ├── package.json
    └── lib/
```

On Unix-like systems:

```bash
mkdir -p ~/.config/opencode/plugins/ogmemory
cp opencode-plugin/wrappers/ogmemory.js ~/.config/opencode/plugins/ogmemory.js
cp opencode-plugin/index.mjs opencode-plugin/package.json ~/.config/opencode/plugins/ogmemory/
cp -r opencode-plugin/lib ~/.config/opencode/plugins/ogmemory/
```

The source wrapper expects exactly the layout above. See [INSTALL-ZH.md](INSTALL-ZH.md) for a Windows-oriented Chinese guide.

## Configuration

Create `~/.config/opencode/ogmemory-config.json`:

```json
{
  "enabled": true,
  "endpoint": "http://127.0.0.1:8090",
  "apiKey": "",
  "accountId": "",
  "userId": "",
  "agentId": "opencode",
  "composeTimeoutMs": 5000,
  "writeTimeoutMs": 15000,
  "traceTimeoutMs": 3000,
  "traceCaptureContent": true,
  "traceSpoolLimit": 1024,
  "traceSpoolTtlSeconds": 3600,
  "transcriptTimeoutMs": 5000,
  "transcriptLimit": 500,
  "composeCacheTtlMs": 60000,
  "disposeTimeoutMs": 3000,
  "sessionStateLimit": 128,
  "sessionStateTtlSeconds": 604800,
  "sessionStateMaxBytes": 16777216
}
```

Environment variables override the corresponding file values:

- `OG_MEMORY_URL`
- `OG_AUTH_API_KEY`
- `OG_AUTH_ACCOUNT_ID`
- `OG_AUTH_USER_ID`
- `OG_AUTH_AGENT_ID`
- `OGMEM_TRACE_CAPTURE_CONTENT` (`0`, `false`, `off`, or `no` selects metadata-only Trace)

Runtime state is kept under `~/.config/opencode/ogmemory/`. Trace content is redacted before it enters the bounded local spool. Failed final conversation writes are retained locally as bounded pending session state (containing the original conversation text needed for write-back) and replayed at startup; defaults are 128 sessions, 16 MiB total, and 7 days of retention. Protect this directory with normal user-profile permissions. Sub-agent sessions are recorded as independent OpenCode sessions; v1 does not infer or persist parent-child relationships.

Unreadable session-state files are deleted with a content-free warning instead of retained outside these limits. Compose responses, injected context, transcript deltas, lifecycle responses, and Trace events each have explicit size bounds.

Plaintext HTTP is accepted only for loopback endpoints (`localhost`, `127.0.0.0/8`, and `::1`). Remote oGMemory deployments must use HTTPS because requests can contain API credentials, conversation content, and Trace data.

Before starting OpenCode, verify the configured service:

```bash
curl http://127.0.0.1:8090/api/v1/health
```

```powershell
Invoke-RestMethod http://127.0.0.1:8090/api/v1/health
```

Set `enabled` to `false` to disable all plugin hooks, network calls, and runtime-state initialization. Unknown configuration fields are rejected so spelling errors cannot silently select defaults.

## Lifecycle behavior

- `chat.message` calls `/api/v1/compose`; the system transform injects only server-marked memory context.
- `session.idle`, compaction, errors, and deletion synchronize transcript deltas through `/api/v1/after_turn`.
- session, user, assistant, tool, compaction, error, and turn-boundary events are written to `/api/v1/trace/events`.
- failed retryable Trace requests remain in a bounded, TTL-limited disk spool and replay at startup.
- all service failures are fail-open for the OpenCode conversation.

Synthetic or ignored OpenCode parts and oGMemory's own injected marker are excluded from write-back, preventing recall feedback loops.

## Development

```bash
npm run check
npm test
npm pack --dry-run
```
