# oGMemory OpenCode Plugin Integration Design

**Date:** 2026-07-25

**Status:** Approved design

**Baseline:** `opengauss-official/dev` at `ddbf67d5e4cddb6164f50c70b087937f6c009587f`

## 1. Objective

Add first-class OpenCode integration to oGMemory through an OpenCode plugin. The
plugin must provide the same class of lifecycle integration as the existing
Claude Code plugin rather than merely registering oGMemory as an MCP server.

The first release must support all three capabilities below:

1. automatic memory retrieval and hidden context injection;
2. automatic conversation synchronization and memory writes; and
3. canonical lifecycle and tool Trace capture with durable local retry.

The existing oGMemory MCP server remains the explicit tool plane for agents that
need to query, read, write, or delete memory deliberately. The plugin is the
policy and lifecycle plane above MCP. Automatic plugin behavior calls the
existing oGMemory HTTP APIs directly and does not depend on the model choosing to
invoke an MCP tool.

## 2. Compatibility Boundary

The plugin supports **OpenCode 1.1.13 and later**.

Compatibility is capability-tiered: OpenCode 1.1.13 provides the required
session-aware system transform and all normal turn lifecycle behavior. The
optional `dispose` hook was introduced in OpenCode 1.15.11, so 1.1.13 through
1.15.10 rely on per-event disk durability and normal lifecycle synchronization
without the additional graceful-shutdown flush.

This minimum is required because:

- `experimental.chat.system.transform` first appeared in OpenCode 1.0.154;
- `experimental.session.compacting` first appeared in OpenCode 1.0.165; and
- `experimental.chat.system.transform` did not receive `sessionID` until
  OpenCode 1.1.13.

Session-scoped injection is unsafe without `sessionID` when multiple OpenCode
sessions run concurrently. Supporting older OpenCode versions with a global
cache is therefore explicitly out of scope.

At startup, the plugin performs a best-effort version check using the OpenCode
CLI when it is available on `PATH`. A detected version below 1.1.13 disables the
automatic integration and logs one actionable compatibility error. When the
host version cannot be detected, the plugin continues in fail-open mode and
logs the documented minimum version. The package documentation and integration
test matrix remain the authoritative compatibility contract because OpenCode
does not expose hook capability negotiation to plugins.

The plugin must tolerate experimental hook signature changes without breaking
the OpenCode agent loop. Hook-specific failures are logged and disable only the
affected automatic capability.

## 3. Chosen Approach

### 3.1 Native plugin plus direct HTTP lifecycle calls

The chosen design is a native JavaScript OpenCode plugin that calls these
existing endpoints:

- `GET /api/v1/health`
- `POST /api/v1/compose`
- `POST /api/v1/after_turn`
- `POST /api/v1/trace/events`

This matches the existing Claude Code integration boundary and requires no
Python subprocesses in the OpenCode process.

### 3.2 Rejected alternatives

#### Synthetic user-message injection

The OpenViking OpenCode plugin inserts recalled memories as a synthetic text
part in the current user message. That is compatible with the stable
`chat.message` hook, but the injected memory enters OpenCode session history.
When message update events are also used for automatic writes, recalled memory
can be captured and written back as if it were new user input. The oGMemory
plugin instead caches retrieval during `chat.message` and injects it through
`experimental.chat.system.transform`.

#### Reusing Claude Code Python scripts

Launching the Claude hook scripts from an OpenCode plugin would reduce initial
code reuse at the cost of process startup latency, event-shape impedance,
cross-platform failures, and duplicated lifecycle semantics. Shared behavior is
reused at the protocol and test-fixture level, not through subprocesses.

## 4. High-Level Architecture

```text
OpenCode agent loop
  |
  +-- chat.message ----------------------> ComposeCoordinator
  |                                          |
  |                                          +-- POST /api/v1/compose
  |                                          +-- per-turn cache
  |
  +-- experimental.chat.system.transform -> hidden memory injection
  |
  +-- message/session lifecycle events ---> SessionSynchronizer
  |                                          |
  |                                          +-- OpenCode SDK transcript delta
  |                                          +-- POST /api/v1/after_turn
  |                                          +-- durable sync cursor
  |
  +-- tool and lifecycle hooks ----------> TraceAdapter
                                             |
                                             +-- local redaction
                                             +-- bounded durable spool
                                             +-- POST /api/v1/trace/events

Explicit agent memory operations ----------> existing oGMemory MCP server
```

The plugin is fail-open: oGMemory unavailability must not fail an OpenCode turn,
tool call, or compaction.

## 5. Package Layout

The repository gains an independent publishable package:

```text
opencode-plugin/
├── index.mjs
├── package.json
├── README.md
├── INSTALL-ZH.md
├── wrappers/
│   └── ogmemory.js
├── lib/
│   ├── compatibility.mjs
│   ├── compose.mjs
│   ├── config.mjs
│   ├── http-client.mjs
│   ├── message-sync.mjs
│   ├── plugin.mjs
│   ├── redaction.mjs
│   ├── session-state.mjs
│   ├── trace-adapter.mjs
│   └── trace-spool.mjs
└── tests/
```

The publishable package name is
`@opengauss/ogmemory-opencode-plugin`. Source installation remains supported
through a top-level wrapper in OpenCode's local plugin directory. Actual npm
publication requires the openGauss npm scope and release credentials and is not
a prerequisite for merging the source implementation.

The runtime uses Node.js built-ins where practical. `@opencode-ai/plugin` is a
development/type dependency unless a runtime helper from that package is
required by the final implementation.

## 6. Configuration

The default configuration file is:

```text
~/.config/opencode/ogmemory-config.json
```

The supported initial configuration is:

```json
{
  "enabled": true,
  "endpoint": "http://127.0.0.1:8090",
  "accountId": "",
  "userId": "",
  "agentId": "opencode",
  "apiKey": "",
  "composeTimeoutMs": 5000,
  "writeTimeoutMs": 15000,
  "traceTimeoutMs": 3000,
  "traceCaptureContent": true,
  "traceSpoolLimit": 1024,
  "traceSpoolTtlSeconds": 3600
}
```

Environment variables override file configuration. Existing oGMemory names are
reused:

- `OG_MEMORY_URL`
- `OG_AUTH_API_KEY`
- `OG_AUTH_ACCOUNT_ID`
- `OG_AUTH_USER_ID`
- `OGMEM_TRACE_CAPTURE_CONTENT`

Secrets are never emitted in logs or persisted inside Trace payloads. The
configuration loader allows plaintext HTTP only for loopback hosts and requires
HTTPS for remote services. It also rejects non-HTTP(S) endpoints, invalid
numeric bounds, and unexpected object shapes with an actionable log entry while
leaving OpenCode usable.

## 7. Automatic Retrieval and Injection

### 7.1 `chat.message`

For each real user message, the plugin:

1. extracts text only from non-synthetic, non-ignored user parts, preserving
   literal marker text in genuine user content;
3. emits a canonical `user_message` Trace event;
4. calls `/api/v1/compose` with the authenticated oGMemory context;
5. caches the result under `(sessionID, messageID)`; and
6. returns without changing the persisted user parts.

Compose has a configurable five-second default timeout. Timeout, transport,
authentication, and server errors are logged once per bounded interval and do
not fail the user message.

### 7.2 `experimental.chat.system.transform`

Before each model call, the plugin looks up the current session's cached compose
result and appends a marked system-context block. The block states that memory
content is untrusted data and must not be followed as instructions.

The same cached result is reused across all model calls in one tool-use loop.
The hook may run multiple times per user turn, but it never triggers another
retrieval request. A unique marker prevents duplicate insertion into one system
array. The cache is cleared when the turn reaches `session.idle`, a newer user
message supersedes it, or its TTL expires.

No recalled memory is stored as an OpenCode message part. This prevents memory
feedback loops and keeps the visible user transcript unchanged.

## 8. Conversation Synchronization

### 8.1 Session state

The plugin keeps a small durable state record per OpenCode session containing:

- the last synchronized message identifier;
- the active user-turn identifier;
- pending transcript messages and turn boundaries needed for retry;
- deletion and error recovery markers; and
- timestamps used for cleanup.

State is written atomically through a temporary file and rename. Unreadable
state is discarded with a content-free diagnostic and rebuilt from OpenCode
session data, so corrupt copies cannot bypass the retention policy.
The state file is bounded by entry count, serialized byte size, and TTL; the
defaults are 128 sessions, 16 MiB, and seven days.

### 8.2 `session.idle`

`session.idle` is the closest OpenCode equivalent to Claude Code `Stop`, but it
is a fire-and-forget event and cannot block or reactivate the agent. On idle the
plugin:

1. obtains the unsynchronized transcript delta through the OpenCode SDK;
2. includes real user and assistant text, including literal marker text written
   by the user;
3. excludes synthetic parts, ignored parts, and compaction continuation messages;
4. posts the delta to `/api/v1/after_turn`;
5. emits `assistant_message` and one idempotent `turn_end` event; and
6. advances the durable sync cursor only after a successful write.

The existing `/after_turn` content deduplication and the local message cursor
provide retry safety. Trace replay uses the stronger server-side idempotency of
`request_id`.

### 8.3 Compaction

`experimental.session.compacting` is awaited by OpenCode. Before compaction the
plugin:

1. emits `pre_compact`;
2. synchronizes any unsent transcript delta;
3. waits up to `writeTimeoutMs` for `/after_turn`; and
4. allows compaction to proceed even when oGMemory is unavailable.

The first release does not replace OpenCode's compaction prompt or final
summary. The awaited pre-compaction hook is the synchronization boundary.

### 8.4 Other boundaries

- `session.created` initializes state and emits `session_start`.
- `session.error` emits `session_error` and attempts a final delta sync.
- `session.deleted` emits `session_end` and removes state after pending work has
  been made durable.
- On OpenCode 1.15.11+, `dispose` performs a bounded process-level flush and
  closes background work. Correctness does not depend on this optional hook.
  It is a fallback, not a substitute for a session-end event.

Main-agent and sub-agent OpenCode sessions are recorded independently using
their native session identifiers. When OpenCode supplies `parentID`, the child
`session_start` event persists it as `parent_session_id`; the plugin does not
infer a relationship when OpenCode omits one.

## 9. Canonical Trace Design

### 9.1 Event mapping

| OpenCode source | Canonical event |
| --- | --- |
| `session.created` | `session_start` |
| `chat.message` | `user_message` |
| `tool.execute.before` | `tool_start` |
| `tool.execute.after` | `tool_success` |
| tool part enters an error state | `tool_failure` |
| finalized assistant message | `assistant_message` |
| `session.idle` | `turn_end` |
| `experimental.session.compacting` | `pre_compact` |
| `session.error` | `session_error` |
| `session.deleted` | `session_end` |

Tool failures that bypass `tool.execute.after` are recovered from tool-part
state events. Events are deduplicated by call ID and terminal state.

Each event receives a stable identifier formatted as `oc_<48 hex chars>`. The
hex value is the leading 48 characters of SHA-256 over the canonical type,
session ID, message or call ID, and relevant terminal state, joined with NUL
separators. The identifier fits the server's 64-character limit and remains
stable across local spool replay. Account isolation is enforced separately by
the server's `(account_id, event_id)` uniqueness boundary.

### 9.2 Content policy

The default Trace includes locally redacted:

- user prompt text;
- assistant text;
- tool input, output, and errors; and
- event type, tool name, timestamps, and content-size metadata.

Redaction behavior matches the existing Claude Code and OpenClaw adapters.
API keys, bearer tokens, cookies, passwords, credential-bearing connection
strings, email addresses, phone numbers, government identifiers, bank cards,
and IP addresses are replaced before transport.

When `OGMEM_TRACE_CAPTURE_CONTENT=0`, only structured boundaries and size
metadata are emitted.

Every event is bounded by encoded size, nesting depth, node count, and
per-field character limits before it can enter the spool.

## 10. Durable Trace Spool

The default runtime directory is:

```text
~/.config/opencode/ogmemory/
├── sessions.json
├── ogmemory-plugin.log
└── trace-spool/
```

Trace events are made durable locally before asynchronous delivery whenever the
source hook itself is not awaited by OpenCode. The spool provides:

- atomic entry creation;
- a default capacity of 1024 events;
- a default TTL of one hour;
- stable request IDs for idempotent replay;
- bounded exponential backoff with jitter;
- bounded retry scheduling with operational status logs;
- startup replay; and
- deletion only after a successful server response.

HTTP 408, 429, 5xx, and transport failures are retryable. Other 4xx responses
are permanent and are logged without retaining the rejected event forever.
Expired and capacity-evicted entries are counted and logged without including
sensitive content.

Trace delivery never blocks the OpenCode agent loop. Compaction waits only for
conversation synchronization, not for full Trace spool drainage.

## 11. Authentication and Isolation

The plugin reuses the same authenticated request semantics as the Claude Code
adapter:

- `X-API-Key` for the configured API key;
- `X-Account-ID` and `X-User-ID` when API-key authentication is configured;
- `accountId`, `userId`, `agentId`, and `sessionId` in lifecycle request bodies;
  and
- server-side identity resolution remains authoritative.

No client-supplied sub-agent identity is trusted inside canonical event content.
Concurrent OpenCode sessions use isolated compose caches, sync cursors, event
deduplication sets, and spool identities. One session must never receive another
session's cached memory block.

## 12. Distribution

Normal installation uses OpenCode's package plugin mechanism:

```json
{
  "plugin": ["@opengauss/ogmemory-opencode-plugin"]
}
```

Source installation copies the package directory and a top-level JavaScript
wrapper into `~/.config/opencode/plugins/`. The wrapper only re-exports the
package entry point.

A release workflow modeled after OpenViking performs:

1. Node syntax validation;
2. unit and contract tests;
3. `npm pack --dry-run`;
4. version-existence detection; and
5. provenance-enabled public npm publication when authorized credentials or
   trusted publishing are configured.

The source implementation and local installation path are complete without npm
credentials. The workflow must never expose or print publication tokens.

## 13. Testing Strategy

### 13.1 Unit tests

- OpenCode-to-canonical event mapping;
- tool success and failure deduplication;
- compose caching across multi-step tool loops;
- concurrent-session cache isolation;
- exclusion of OpenCode-provenance synthetic/ignored content from writeback,
  while preserving literal marker text in genuine user content;
- redaction and metadata-only behavior;
- payload size, depth, node, and field limits;
- spool atomicity, TTL, capacity, retry scheduling, and replay behavior;
- durable session cursor recovery;
- configuration validation and environment precedence; and
- compatibility version parsing and fail-open behavior.

### 13.2 Contract tests

Mock HTTP tests verify exact paths, body shapes, authentication headers,
timeouts, retry classification, and error handling for health, compose,
after-turn, and Trace requests.

### 13.3 Integration tests

- compose results appear in model-facing system context but not user history;
- after-turn receives exactly the unsynchronized real messages;
- Trace `turn_end` finalizes one server-side turn;
- compaction synchronizes pending messages before proceeding;
- server downtime does not fail an OpenCode turn;
- Trace replays after service recovery; and
- two active sessions do not share memory, state, or Trace events.

### 13.4 End-to-end acceptance scenarios

1. A later OpenCode turn recalls durable information created by an earlier
   turn.
2. Injected memory is absent from the user-visible and synchronized transcript.
3. Successful and failed tools create the correct Trace sequence.
4. Each completed turn creates exactly one `turn_end` boundary.
5. Trace generated during an oGMemory restart is delivered after recovery.
6. Metadata-only mode stores no original prompt, assistant, tool, or error text.
7. Parallel sessions cannot observe each other's compose results.
8. Messages pending at compaction become retrievable from oGMemory.
9. Plugin and service failures leave the OpenCode agent usable.

## 14. Scope Boundaries

The first release does not:

- infer parent-child Agent relationships when OpenCode does not provide one; supplied `parentID` values are persisted as `parent_session_id` on `session_start`;
- take over OpenCode's final compaction summary;
- support OpenCode versions older than 1.1.13;
- inject recalled memory into persisted user message parts;
- replace the existing MCP server or duplicate all MCP tools as native plugin
  tools;
- automatically start the oGMemory server; or
- require npm publication credentials for source delivery.

These boundaries keep the feature focused on lifecycle parity with Claude Code
while leaving explicit memory operations in the existing MCP layer.

## 15. Operational Implementation Baseline

Implementation must start from `opengauss-official/dev` at the baseline commit
listed at the top of this document, or from a newer official `dev` after an
explicit rebase and compatibility review. The existing primary working tree
contains unrelated user changes, so implementation work must remain in an
isolated worktree and a `codex/` branch.
