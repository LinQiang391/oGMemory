# OpenClaw Hook Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Claude Code and OpenClaw hooks into the same irreversibly redacted canonical events and feed both into the existing SQL trace pipeline.

**Architecture:** Each integration owns a small `hook2Event` adapter at its runtime boundary. The adapters remove agent-specific hook semantics, redact content, and emit canonical `event_type` values to the existing `/api/v1/trace/events` endpoint; the server persists and finalizes only canonical events while accepting legacy Claude `Stop` payloads.

**Tech Stack:** Python 3.11, pytest, JavaScript ESM on Node.js 18+, Node test runner, OpenClaw typed plugin hooks (verified locally against OpenClaw 2026.6.9), Flask, PostgreSQL/openGauss JSONB.

## Global Constraints

- Keep `POST /api/v1/trace/events`; do not add a parallel endpoint.
- Do not add an agent discriminator to canonical events or branch internally by agent.
- Preserve the existing `trace_events` and `trace_sessions` SQL schemas.
- Capture trace content by default, after irreversible redaction; allow the existing Claude environment setting to opt out explicitly.
- Detect credentials, email, phone, identity/passport/bank-card-like values, IP addresses, and credential-bearing connection strings.
- Repeated sensitive values within one event reuse a typed placeholder; mappings never cross events and are never persisted.
- Trace delivery failures must not fail or cancel an agent turn.
- Use tests first and observe the expected failure before every production change.

---

## File Structure

- `claude-plugin/scripts/call_add_session_message.py`: expose the Claude `claude_hook_to_event(data)` adapter and typed irreversible redaction while retaining its existing transport/spool behavior.
- `openclaw_context_engine_plugin/trace.js`: own OpenClaw redaction, `openClawHookToEvent(hookName, event, ctx)`, trace HTTP delivery, and typed hook registration.
- `openclaw_context_engine_plugin/index.js`: resolve trace configuration and register the trace hook module with the existing plugin.
- `openclaw_context_engine_plugin/openclaw.plugin.json`: document bounded trace timeout configuration.
- `tests/fixtures/trace/redaction_cases.json`: language-neutral redaction inputs and expected placeholder types used by both adapters.
- `tests/unit/claude_plugin/test_plugin_scripts.py`: Claude adapter, content-default, redaction, and legacy transport coverage.
- `openclaw_context_engine_plugin/trace.test.js`: OpenClaw adapter, redaction, hook registration, transport, and failure-isolation coverage.
- `server/app.py`: normalize the HTTP compatibility boundary to canonical event types and finalize `turn_end`.
- `fs/sql_adapter/sql_trace_store.py`: recognize canonical `turn_end` rows and legacy `Stop` rows as aggregation boundaries.
- `tests/unit/server/test_http_auth_v1.py`: endpoint normalization and finalization tests.
- `tests/unit/fs/test_sql_trace_store.py`: canonical/legacy boundary persistence tests.
- `claude-plugin/README.md` and `openclaw_context_engine_plugin/README.md`: describe hook capture, redaction, configuration, and compatibility.

---

### Task 1: Claude hook2Event and irreversible redaction

**Files:**
- Create: `tests/fixtures/trace/redaction_cases.json`
- Modify: `tests/unit/claude_plugin/test_plugin_scripts.py`
- Modify: `claude-plugin/scripts/call_add_session_message.py`

**Interfaces:**
- Produces: `claude_hook_to_event(data: dict) -> dict`
- Produces: `_redact(value: object) -> object`, returning a recursively redacted value with typed `{{type_n}}` placeholders.
- Canonical turn completion: `{"event_type": "turn_end", "assistant_message": "..."}`.

- [ ] **Step 1: Add the shared redaction fixture and failing Claude tests**

Create fixture cases containing a secret-key field, bearer token, email, Chinese phone number, Chinese identity-card-like value, passport-like value, bank-card-like value, IPv4, IPv6, and credential-bearing PostgreSQL URL. Add tests asserting:

```python
event = mod.claude_hook_to_event({
    "hook_event_name": "UserPromptSubmit",
    "prompt": fixture["text"],
})
serialized = json.dumps(event, ensure_ascii=False)
assert fixture["plaintext"] not in serialized
assert fixture["placeholder"] in serialized
```

Also assert content is captured when `OGMEM_TRACE_CAPTURE_CONTENT` is absent, setting it to `0` omits content, `Stop` maps to `turn_end`, and the returned event does not contain `hook_type`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
D:\code\oGMemory-dev\.venv\Scripts\python.exe -m pytest tests/unit/claude_plugin/test_plugin_scripts.py -q
```

Expected: failures because `claude_hook_to_event` and typed placeholders do not exist, content defaults off, and `Stop` still maps to `assistant_message`.

- [ ] **Step 3: Implement the minimal Claude adapter and redactor**

Implement event-local redaction state so repeated plaintext reuses its placeholder:

```python
def _redact(value: object) -> object:
    state = _RedactionState()
    return state.redact(value)

def claude_hook_to_event(data: dict) -> dict:
    hook_type = _hook_name(data)
    event = {"event_type": _TRACE_EVENT_TYPES.get(hook_type, "hook_event")}
    # Add bounded generic content and size fields, using _redact before serialization.
    return event
```

Set `Stop` to `turn_end`, keep `SubagentStop` as `assistant_message`, and make `_capture_trace_content()` default true unless the environment value is one of `0`, `false`, `no`, or `off`. Keep `_build_trace_event = claude_hook_to_event` only as a temporary in-process compatibility alias if existing callers require it.

- [ ] **Step 4: Run focused and existing Claude tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass with no plaintext fixture values in produced events.

- [ ] **Step 5: Commit the task**

```powershell
git add tests/fixtures/trace/redaction_cases.json tests/unit/claude_plugin/test_plugin_scripts.py claude-plugin/scripts/call_add_session_message.py
git commit -m "feat(trace): normalize and redact Claude hooks"
```

---

### Task 2: OpenClaw hook2Event and hook registration

**Files:**
- Create: `openclaw_context_engine_plugin/trace.js`
- Create: `openclaw_context_engine_plugin/trace.test.js`
- Modify: `openclaw_context_engine_plugin/index.js`
- Modify: `openclaw_context_engine_plugin/openclaw.plugin.json`

**Interfaces:**
- Produces: `redactTraceValue(value) -> redactedValue`.
- Produces: `openClawHookToEvent(hookName, event, ctx) -> canonicalEvent | null`.
- Produces: `postTraceEvent(cfg, envelope, deps?) -> Promise<boolean>`; makes at most two attempts, catches transport failures, and returns false.
- Produces: `registerOpenClawTraceHooks(api, cfg, deps?) -> void`.

- [ ] **Step 1: Write failing adapter and redaction tests**

Load `tests/fixtures/trace/redaction_cases.json` in `trace.test.js`. Assert equivalent hooks map as follows:

```javascript
assert.equal(openClawHookToEvent("session_start", { sessionId: "s" }, {}).event_type, "session_start");
assert.equal(openClawHookToEvent("before_agent_run", { prompt: "hello" }, {}).event_type, "user_message");
assert.equal(openClawHookToEvent("before_tool_call", { toolName: "exec", params: {} }, {}).event_type, "tool_start");
assert.equal(openClawHookToEvent("after_tool_call", { toolName: "exec", result: "ok" }, {}).event_type, "tool_success");
assert.equal(openClawHookToEvent("after_tool_call", { toolName: "exec", error: "bad" }, {}).event_type, "tool_failure");
assert.equal(openClawHookToEvent("agent_end", { messages: [], success: true }, {}).event_type, "turn_end");
assert.equal(openClawHookToEvent("before_compaction", { messageCount: 2 }, {}).event_type, "pre_compact");
assert.equal(openClawHookToEvent("session_end", { sessionId: "s", messageCount: 2 }, {}).event_type, "session_end");
```

Assert each shared fixture plaintext is absent and its typed placeholder is present. Assert the same sensitive value repeated in tool params/result gets the same placeholder.

- [ ] **Step 2: Run the Node test and verify RED**

Run:

```powershell
node --test openclaw_context_engine_plugin/trace.test.js
```

Expected: failure because `trace.js` does not exist.

- [ ] **Step 3: Implement the minimal OpenClaw adapter and redactor**

Create `trace.js` with deterministic bounded extraction from the verified OpenClaw 2026.6.9 hook fields. Extract the last assistant text from `agent_end.messages`, retain `runId`, `toolCallId`, sizes, result/error, duration, compaction counts, and session-end reason when present. Do not emit the OpenClaw hook name or an agent source field.

- [ ] **Step 4: Add failing transport and registration tests**

Use a fake API with `on(name, handler)` to assert registration of:

```javascript
[
  "session_start",
  "before_agent_run",
  "before_tool_call",
  "after_tool_call",
  "agent_end",
  "before_compaction",
  "session_end",
]
```

Invoke handlers and assert `fetchFn` receives `POST /api/v1/trace/events` with the existing identity envelope and canonical event. Make `fetchFn` reject twice and assert every handler resolves without throwing after exactly two attempts. Assert each attempt's timeout uses `cfg.traceTimeoutMs`.

- [ ] **Step 5: Run the Node test and verify the new tests fail for missing transport/registration**

Run the Step 2 command. Expected: adapter tests pass; transport/registration tests fail because exports are missing.

- [ ] **Step 6: Implement transport, registration, and configuration**

Add `traceTimeoutMs` to `resolveConfig`, defaulting to `3000`, and to the manifest schema. Build headers using the same API key/account/user identity rules as the existing plugin transport. Generate one UUID event id, post the unchanged envelope with no more than two total attempts, catch and log final errors, and return false without throwing. Register hooks from `index.js` before context-engine registration. Do not create a plaintext or reversible disk spool; the redacted in-memory event is discarded after the bounded attempts.

For conversation-bearing `before_agent_run` and `agent_end`, document that OpenClaw requires `plugins.entries.og-memory-context-engine.hooks.allowConversationAccess=true` for non-bundled plugins.

- [ ] **Step 7: Run all OpenClaw plugin tests and verify GREEN**

```powershell
node --test openclaw_context_engine_plugin/*.test.js
```

Expected: all tests pass and simulated trace HTTP failures do not reject hook handlers.

- [ ] **Step 8: Commit the task**

```powershell
git add openclaw_context_engine_plugin/trace.js openclaw_context_engine_plugin/trace.test.js openclaw_context_engine_plugin/index.js openclaw_context_engine_plugin/openclaw.plugin.json
git commit -m "feat(trace): capture OpenClaw hooks"
```

---

### Task 3: Canonical trace persistence and turn finalization

**Files:**
- Modify: `tests/unit/server/test_http_auth_v1.py`
- Modify: `tests/unit/fs/test_sql_trace_store.py`
- Modify: `server/app.py`
- Modify: `server/memory_service.py`
- Modify: `fs/sql_adapter/sql_trace_store.py`

**Interfaces:**
- HTTP compatibility boundary derives `canonical_type = "turn_end"` for legacy `hook_type == "Stop"`; otherwise it uses `event.event_type` and falls back to legacy `hook_type`.
- `MemoryService.add_session_trace()` persists the canonical type in the existing `hook_type` SQL column.
- `SQLTraceEventStore.finalize_session()` accepts both `turn_end` and historical `Stop` rows as boundaries.

- [ ] **Step 1: Write failing HTTP normalization tests**

Add one test posting `{"event": {"event_type": "turn_end"}}` and assert finalization occurs. Strengthen the legacy `Stop` test to assert `add_session_trace` receives `hookType == "turn_end"`. Add a non-boundary canonical event test asserting no finalization.

- [ ] **Step 2: Run the HTTP tests and verify RED**

```powershell
D:\code\oGMemory-dev\.venv\Scripts\python.exe -m pytest tests/unit/server/test_http_auth_v1.py -q
```

Expected: canonical `turn_end` does not finalize and legacy `Stop` is not normalized.

- [ ] **Step 3: Implement minimal HTTP-boundary normalization**

In `handle_trace_event()` derive one canonical type before persistence:

```python
legacy_hook_type = str(event.get("hook_type") or "").strip()
event_type = str(event.get("event_type") or "").strip()
canonical_type = "turn_end" if legacy_hook_type == "Stop" else (event_type or legacy_hook_type or "hook_event")
event = {**event, "event_type": canonical_type}
event.pop("hook_type", None)
```

Pass `canonical_type` as `hookType` and finalize only when it equals `turn_end`.

- [ ] **Step 4: Run HTTP tests and verify GREEN**

Run the Step 2 command. Expected: all route tests pass.

- [ ] **Step 5: Write failing SQL boundary compatibility tests**

Update the fake cursor expectations to require the selected boundary row and previous boundary lookup to accept `hook_type IN ('turn_end', 'Stop')`. Add canonical insert/finalize coverage while retaining a historical `Stop` fixture.

- [ ] **Step 6: Run SQL trace-store tests and verify RED**

```powershell
D:\code\oGMemory-dev\.venv\Scripts\python.exe -m pytest tests/unit/fs/test_sql_trace_store.py tests/unit/server/test_memory_service_trace.py -q
```

Expected: SQL assertions fail because queries still require only `Stop`.

- [ ] **Step 7: Implement generic boundary queries and terminology**

Change only boundary predicates to `hook_type IN ('turn_end', 'Stop')`. Preserve method and parameter names where renaming would expand compatibility scope, but update docstrings and error text from “Stop event” to “turn-end event”. Remove Claude-specific wording from `server/memory_service.py` and `fs/sql_adapter/sql_trace_store.py`.

- [ ] **Step 8: Run focused trace tests and verify GREEN**

```powershell
D:\code\oGMemory-dev\.venv\Scripts\python.exe -m pytest tests/unit/server/test_http_auth_v1.py tests/unit/server/test_memory_service_trace.py tests/unit/fs/test_sql_trace_store.py -q
```

Expected: all focused trace tests pass.

- [ ] **Step 9: Commit the task**

```powershell
git add tests/unit/server/test_http_auth_v1.py tests/unit/fs/test_sql_trace_store.py server/app.py server/memory_service.py fs/sql_adapter/sql_trace_store.py
git commit -m "refactor(trace): persist canonical agent events"
```

---

### Task 4: Documentation, regression verification, and review handoff

**Files:**
- Modify: `claude-plugin/README.md`
- Modify: `openclaw_context_engine_plugin/README.md`
- Modify: `docs/superpowers/specs/2026-07-16-openclaw-hook-trace-design.md` only if implementation verification exposes a factual mismatch.

**Interfaces:**
- Documents canonical event mapping, default content capture, irreversible placeholders, OpenClaw conversation-access permission, timeout, endpoint, and non-blocking failure behavior.

- [ ] **Step 1: Add documentation assertions or executable manifest checks**

Extend existing Python/Node tests to assert manifest `traceTimeoutMs` default and README-required OpenClaw permission text where the repository already tests documentation/config contracts.

- [ ] **Step 2: Run those tests and verify RED**

Expected: failure until documentation and schema describe the new behavior.

- [ ] **Step 3: Update both integration READMEs**

Document exact hook mappings, `{{type_n}}` placeholders, `OGMEM_TRACE_CAPTURE_CONTENT=0` opt-out for Claude, OpenClaw `traceTimeoutMs`, and:

```json
{
  "plugins": {
    "entries": {
      "og-memory-context-engine": {
        "hooks": { "allowConversationAccess": true }
      }
    }
  }
}
```

- [ ] **Step 4: Run focused tests, then the relevant regression suite**

```powershell
D:\code\oGMemory-dev\.venv\Scripts\python.exe -m pytest tests/unit/claude_plugin/test_plugin_scripts.py tests/unit/server/test_http_auth_v1.py tests/unit/server/test_memory_service_trace.py tests/unit/fs/test_sql_trace_store.py -q
node --test openclaw_context_engine_plugin/*.test.js
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 5: Commit documentation**

```powershell
git add claude-plugin/README.md openclaw_context_engine_plugin/README.md openclaw_context_engine_plugin/openclaw.plugin.json
git commit -m "docs(trace): describe agent hook capture"
```

- [ ] **Step 6: Invoke production review before remote submission**

Run the `production-code-review` skill with six independent review roles over `7bd401e0..HEAD`, including all commits and changed files. Validate and deduplicate every finding. If any code changes, run a fresh complete six-role review round. Only after a clean round, rerun all commands from Step 4 plus any broader tests identified by reviewers.

- [ ] **Step 7: Push only after clean review and fresh verification**

```powershell
git push -u origin codex/openclaw-hook-trace
```

Expected: push succeeds after the production verdict is `READY`.
