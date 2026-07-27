# OpenCode Plugin Integration Implementation Plan

**Design:** `docs/superpowers/specs/2026-07-25-opencode-plugin-integration-design.md`

**Base:** `ddbf67d5e4cddb6164f50c70b087937f6c009587f`

**Goal:** ship a publishable OpenCode plugin that automatically retrieves,
synchronizes, and traces oGMemory context without duplicating MCP tools or
breaking the OpenCode agent loop when oGMemory is unavailable.

## Task 1: Package and configuration foundation

Create:

- `opencode-plugin/package.json`
- `opencode-plugin/index.mjs`
- `opencode-plugin/lib/config.mjs`
- `opencode-plugin/lib/http-client.mjs`
- `opencode-plugin/lib/compatibility.mjs`
- `opencode-plugin/tests/config.test.mjs`
- `opencode-plugin/tests/http-client.test.mjs`

Implement a dependency-light ESM package, bounded configuration validation,
environment precedence, authenticated JSON requests, timeout classification,
and best-effort OpenCode version detection.

Verify:

```bash
node --check opencode-plugin/index.mjs
node --test opencode-plugin/tests/config.test.mjs opencode-plugin/tests/http-client.test.mjs
```

## Task 2: Compose coordination and hidden injection

Create:

- `opencode-plugin/lib/compose.mjs`
- `opencode-plugin/tests/compose.test.mjs`

Implement real-user-text extraction, per-session/per-message compose caching,
bounded cache cleanup, `/api/v1/compose`, injection markers, untrusted-data
notice, and duplicate-free system injection. Do not mutate persisted user
parts.

Verify tests for success, timeout, empty prompts, synthetic inputs, repeated
model calls, cache expiry, and concurrent-session isolation.

## Task 3: Durable session state and message synchronization

Create:

- `opencode-plugin/lib/session-state.mjs`
- `opencode-plugin/lib/message-sync.mjs`
- `opencode-plugin/tests/session-state.test.mjs`
- `opencode-plugin/tests/message-sync.test.mjs`

Persist sync state with atomic replace. Fetch OpenCode messages through the SDK,
normalize only real user/assistant text, exclude injected/synthetic/compaction
content, and post transcript deltas to `/api/v1/after_turn`. Advance the cursor
only after success. Serialize concurrent syncs per session.

Verify crash recovery, corrupt-state quarantine, no-op deltas, retry behavior,
message filtering, and two-session isolation.

## Task 4: Trace redaction and canonical mapping

Create:

- `opencode-plugin/lib/redaction.mjs`
- `opencode-plugin/lib/trace-adapter.mjs`
- `opencode-plugin/tests/redaction.test.mjs`
- `opencode-plugin/tests/trace-adapter.test.mjs`

Port the established oGMemory client-side redaction contract without importing
Python. Map OpenCode lifecycle, message, and tool events to canonical events.
Use stable `oc_<48 hex chars>` request IDs. Recover tool failures from tool-part
error updates and deduplicate terminal events by stable IDs.

Verify every mapping, content-off mode, sensitive categories, size/depth/node
bounds, deterministic IDs, and duplicate suppression.

## Task 5: Bounded durable Trace spool

Create:

- `opencode-plugin/lib/trace-spool.mjs`
- `opencode-plugin/tests/trace-spool.test.mjs`

Implement atomic on-disk entries, capacity and TTL eviction, startup replay,
retry classification, bounded exponential backoff, and a circuit breaker. Keep
the queue inspectable and avoid storing unredacted data.

Use dependency injection for clock, random, sender, and paths so tests need no
production-only switches.

Verify successful delivery, transport/408/429/5xx retry, permanent 4xx drop,
restart replay, stable request IDs, TTL, capacity, and corrupt-entry quarantine.

## Task 6: Hook orchestration

Wire all components in `opencode-plugin/index.mjs`:

- `chat.message`
- `experimental.chat.system.transform`
- `tool.execute.before`
- `tool.execute.after`
- `experimental.session.compacting`
- `event`
- `dispose`

Route `session.created`, message updates, tool-part failures, `session.idle`,
`session.error`, and `session.deleted`. Serialize fire-and-forget events per
session while allowing different sessions to proceed independently. Keep
sub-agent sessions independent. Ensure every hook catches its own operational
errors and does not fail OpenCode.

Use `opencode-plugin/tests/hooks.test.mjs` with a factory-level fake SDK client,
fake HTTP transport, and focused hook tests to exercise complete turns,
compaction, deletion, service failure, restart recovery, and parallel sessions.

## Task 7: Distribution and documentation

Create:

- `opencode-plugin/wrappers/ogmemory.js`
- `opencode-plugin/README.md`
- `opencode-plugin/INSTALL-ZH.md`

Update:

- `.gitignore` for plugin runtime/test artifacts only if required
- the repository README integration lists where appropriate

Document OpenCode 1.1.13+ (with graceful `dispose` on 1.15.11+), npm-ready and source installation, configuration,
MCP layering, health verification, Trace privacy, runtime files, and failure
diagnostics. Do not claim that the npm package is published until publication
actually succeeds.

Run `npm pack --dry-run` and inspect the file list.

## Task 8: Verification and production review

Run:

```bash
npm --prefix opencode-plugin test
npm --prefix opencode-plugin run check
npm --prefix opencode-plugin pack --dry-run
pytest tests/unit/claude_plugin/test_plugin_scripts.py -q
pytest tests/unit/server/test_memory_service_trace.py tests/unit/server/test_http_auth_v1.py -q
git diff --check
```

Then run the mandatory six independent production-review roles against the full
diff from `ddbf67d5e` to the implementation head. Validate and deduplicate every
finding. If fixes are needed, rerun all affected tests and repeat a complete
six-role review round until no new actionable finding remains.

## Acceptance Criteria

- Automatic recall reaches model-facing system context and never persisted user
  history.
- `session.idle` synchronizes exactly the unsent real transcript delta.
- Compaction attempts pending synchronization without blocking indefinitely.
- Canonical user, assistant, tool, turn, compaction, error, and session events
  reach `/api/v1/trace/events`.
- Offline Trace survives plugin restart and replays idempotently.
- Metadata-only mode contains no original conversation or tool text.
- Concurrent sessions cannot share cache, cursor, or Trace identity.
- OpenCode remains usable through every tested oGMemory failure mode.
