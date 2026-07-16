import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  openClawHookToEvent,
  postTraceEvent,
  redactTraceValue,
  registerOpenClawTraceHooks,
} from "./trace.js";
import plugin, { resolveConfig } from "./index.js";

const fixtureUrl = new URL("../tests/fixtures/trace/redaction_cases.json", import.meta.url);
const redactionCases = JSON.parse(await readFile(fixtureUrl, "utf8"));

test("maps supported OpenClaw hooks to canonical event types", () => {
  const cases = [
    ["session_start", { sessionId: "s" }, "session_start"],
    ["before_agent_run", { prompt: "hello", messages: [] }, "user_message"],
    ["before_tool_call", { toolName: "exec", params: {} }, "tool_start"],
    ["after_tool_call", { toolName: "exec", params: {}, result: "ok" }, "tool_success"],
    ["after_tool_call", { toolName: "exec", params: {}, error: "bad" }, "tool_failure"],
    ["agent_end", { messages: [], success: true }, "turn_end"],
    ["before_compaction", { messageCount: 2 }, "pre_compact"],
    ["session_end", { sessionId: "s", messageCount: 2 }, "session_end"],
  ];

  for (const [hookName, event, expectedType] of cases) {
    const canonical = openClawHookToEvent(hookName, event, {});
    assert.equal(canonical.event_type, expectedType);
    assert.equal("hook_type" in canonical, false);
    assert.equal("source" in canonical, false);
    assert.equal("agent_id" in canonical, false);
  }
  assert.equal(openClawHookToEvent("unknown", {}, {}), null);
});

test("extracts bounded fields from verified OpenClaw typed hooks", () => {
  assert.deepEqual(
    openClawHookToEvent(
      "after_tool_call",
      {
        toolName: "exec",
        params: { command: "echo ok" },
        result: { output: "ok" },
        runId: "run-event",
        toolCallId: "call-1",
        durationMs: 12,
      },
      { runId: "run-context", agentId: "must-not-leak" },
    ),
    {
      event_type: "tool_success",
      run_id: "run-event",
      tool_name: "exec",
      tool_use_id: "call-1",
      tool_input: { command: "echo ok" },
      tool_input_size: 21,
      tool_response: { output: "ok" },
      tool_response_size: 15,
      duration_ms: 12,
    },
  );

  const turnEnd = openClawHookToEvent(
    "agent_end",
    {
      runId: "run-2",
      messages: [
        { role: "assistant", content: "first" },
        { role: "user", content: "question" },
        { role: "assistant", content: [{ type: "text", text: "final" }] },
      ],
      success: false,
      error: "failed",
      durationMs: 34,
    },
    {},
  );
  assert.equal(turnEnd.assistant_message, "final");
  assert.equal(turnEnd.assistant_message_size, 5);
  assert.equal(turnEnd.status, "error");
  assert.equal(turnEnd.error, "failed");
  assert.equal(turnEnd.duration_ms, 34);

  assert.deepEqual(
    openClawHookToEvent(
      "before_compaction",
      { messageCount: 8, compactingCount: 5, tokenCount: 1200 },
      { runId: "run-3" },
    ),
    {
      event_type: "pre_compact",
      run_id: "run-3",
      message_count: 8,
      compacting_count: 5,
      token_count: 1200,
    },
  );
  assert.deepEqual(
    openClawHookToEvent(
      "session_end",
      { sessionId: "s", messageCount: 8, durationMs: 99, reason: "idle" },
      {},
    ),
    {
      event_type: "session_end",
      message_count: 8,
      duration_ms: 99,
      reason: "idle",
    },
  );

  const bounded = openClawHookToEvent(
    "before_agent_run",
    { prompt: "x".repeat(20_000), messages: [] },
    {},
  );
  assert.equal(bounded.prompt_size, 20_000);
  assert.ok(bounded.prompt.length <= 10_000);
});

for (const fixture of redactionCases) {
  test(`redacts shared fixture: ${fixture.name}`, () => {
    const redacted = redactTraceValue(fixture.text);
    const serialized = JSON.stringify(redacted);
    assert.equal(serialized.includes(fixture.plaintext), false);
    assert.equal(serialized.includes(fixture.placeholder), true);
  });
}

test("reuses a typed placeholder for repeated values in one event", () => {
  const event = openClawHookToEvent(
    "after_tool_call",
    {
      toolName: "lookup",
      params: { email: "repeat.fixture@example.com" },
      result: "repeat.fixture@example.com",
    },
    {},
  );
  const serialized = JSON.stringify(event);
  assert.equal(serialized.includes("repeat.fixture@example.com"), false);
  assert.equal(serialized.match(/\{\{email_1\}\}/g)?.length, 2);
});

test("redacts named credentials embedded in strings", () => {
  const redacted = redactTraceValue(
    "password=hunter2 authorization: Basic auth-value cookie=session-value --api-key 'quoted-value'",
  );
  assert.equal(redacted.includes("hunter2"), false);
  assert.equal(redacted.includes("auth-value"), false);
  assert.equal(redacted.includes("session-value"), false);
  assert.equal(redacted.includes("quoted-value"), false);
  assert.match(redacted, /\{\{password_1\}\}/);
  assert.match(redacted, /\{\{authorization_1\}\}/);
  assert.match(redacted, /\{\{cookie_1\}\}/);
  assert.match(redacted, /\{\{api_key_1\}\}/);
});

test("does not classify an invalid dotted number as IPv4", () => {
  assert.equal(redactTraceValue("version 999.999.999.999"), "version 999.999.999.999");
});

test("posts an unchanged trace envelope with identity headers and configured timeout", async () => {
  const calls = [];
  const timeouts = [];
  const envelope = {
    request_id: "event-1",
    timestamp: "2026-07-16T00:00:00.000Z",
    account_id: "account",
    user_id: "user",
    agent_id: "agent",
    session_id: "session",
    event: { event_type: "tool_start" },
  };
  const result = await postTraceEvent(
    {
      memoryApiBaseUrl: "http://memory.test/",
      authApiKey: "key",
      authAccountId: "auth-account",
      traceTimeoutMs: 1234,
    },
    envelope,
    {
      fetchFn: async (...args) => {
        calls.push(args);
        return { ok: true };
      },
      timeoutSignalFn: (timeoutMs) => {
        timeouts.push(timeoutMs);
        return { timeoutMs };
      },
    },
  );

  assert.equal(result, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "http://memory.test/api/v1/trace/events");
  assert.deepEqual(calls[0][1].headers, {
    "Content-Type": "application/json",
    "X-API-Key": "key",
    "X-Account-ID": "auth-account",
    "X-User-ID": "user",
  });
  assert.deepEqual(JSON.parse(calls[0][1].body), envelope);
  assert.deepEqual(timeouts, [1234]);
});

test("registers typed hooks and swallows transport failures after two attempts", async () => {
  const handlers = new Map();
  const fetchCalls = [];
  const timeouts = [];
  let uuidCount = 0;
  const cfg = {
    memoryApiBaseUrl: "http://memory.test",
    authApiKey: "",
    authAccountId: "",
    accountId: "account",
    userId: "user",
    agentId: "configured-agent",
    traceTimeoutMs: 75,
  };
  registerOpenClawTraceHooks(
    {
      on(name, handler) {
        handlers.set(name, handler);
      },
      logger: { error() {} },
    },
    cfg,
    {
      fetchFn: async (...args) => {
        fetchCalls.push(args);
        throw new Error("offline");
      },
      timeoutSignalFn: (timeoutMs) => {
        timeouts.push(timeoutMs);
        return { timeoutMs };
      },
      randomUUID: () => `uuid-${++uuidCount}`,
      now: () => new Date("2026-07-16T01:02:03.000Z"),
    },
  );

  const expectedNames = [
    "session_start",
    "before_agent_run",
    "before_tool_call",
    "after_tool_call",
    "agent_end",
    "before_compaction",
    "session_end",
  ];
  assert.deepEqual([...handlers.keys()], expectedNames);

  const invocations = {
    session_start: [{ sessionId: "s" }, { sessionId: "s", agentId: "runtime-agent" }],
    before_agent_run: [{ prompt: "hello", messages: [] }, { sessionId: "s", agentId: "runtime-agent" }],
    before_tool_call: [{ toolName: "exec", params: {} }, { sessionId: "s", agentId: "runtime-agent" }],
    after_tool_call: [{ toolName: "exec", params: {}, result: "ok" }, { sessionId: "s", agentId: "runtime-agent" }],
    agent_end: [{ messages: [], success: true }, { sessionId: "s", agentId: "runtime-agent" }],
    before_compaction: [{ messageCount: 2 }, { sessionId: "s", agentId: "runtime-agent" }],
    session_end: [{ sessionId: "s", messageCount: 2 }, { sessionId: "s", agentId: "runtime-agent" }],
  };
  for (const name of expectedNames) {
    assert.equal(await handlers.get(name)(...invocations[name]), undefined);
  }

  assert.equal(fetchCalls.length, expectedNames.length * 2);
  assert.deepEqual(timeouts, Array(expectedNames.length * 2).fill(75));
  assert.equal(uuidCount, expectedNames.length);
  for (let index = 0; index < fetchCalls.length; index += 2) {
    const firstBody = JSON.parse(fetchCalls[index][1].body);
    const retryBody = JSON.parse(fetchCalls[index + 1][1].body);
    assert.deepEqual(retryBody, firstBody);
    assert.equal(firstBody.account_id, "account");
    assert.equal(firstBody.user_id, "user");
    assert.equal(firstBody.agent_id, "runtime-agent");
    assert.equal(firstBody.session_id, "s");
    assert.match(firstBody.event.event_type, /^(session_start|user_message|tool_start|tool_success|turn_end|pre_compact|session_end)$/);
  }
});

test("plugin configuration and registration enable trace hooks before the context engine", () => {
  assert.equal(resolveConfig({}).traceTimeoutMs, 3000);
  assert.equal(resolveConfig({ traceTimeoutMs: 456 }).traceTimeoutMs, 456);

  const registrations = [];
  plugin.register({
    pluginConfig: { mode: "remote" },
    logger: { info() {}, error() {} },
    on(name) {
      registrations.push(`hook:${name}`);
    },
    registerService() {},
    registerContextEngine() {
      registrations.push("context-engine");
    },
  });
  assert.equal(registrations.filter((value) => value.startsWith("hook:")).length, 7);
  assert.ok(registrations.indexOf("hook:session_start") < registrations.indexOf("context-engine"));
});
