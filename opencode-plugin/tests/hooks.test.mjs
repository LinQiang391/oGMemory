import test from "node:test";
import assert from "node:assert/strict";
import { createHooks, OgMemoryPlugin, replayStoredSessions } from "../lib/plugin.mjs";

test("chat message trace uses the persisted output message ID when input ID is absent", async () => {
  const emitted = [];
  let recalledMessageId;
  const hooks = createHooks({
    logger: { warn() {} },
    trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: { recall: async (_sessionId, messageId) => { recalledMessageId = messageId; } },
    sync: { flush: async () => {} },
    store: {},
  });
  await hooks["chat.message"](
    { sessionID: "s" },
    { message: { id: "m1" }, parts: [{ type: "text", text: "remember" }] },
  );
  assert.equal(emitted[0][2].objectId, "m1");
  assert.equal(recalledMessageId, "m1");
});

test("chat message trace falls back to the legacy input message ID", async () => {
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} },
    trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: { recall: async () => {} },
    sync: { flush: async () => {} },
    store: {},
  });
  await hooks["chat.message"](
    { sessionID: "s", messageID: "legacy-m1" },
    { parts: [{ type: "text", text: "remember" }] },
  );
  assert.equal(emitted[0][2].objectId, "legacy-m1");
});

test("idle boundaries use the synchronized message cursor and ignore duplicate idle", async () => {
  const results = [
    { success: true, count: 2, cursor: "m2", lastAssistant: "first", boundaries: [{ cursor: "m2", lastAssistant: "first", count: 2 }] },
    { success: true, count: 0, cursor: "m2", lastAssistant: "", boundaries: [] },
    { success: true, count: 2, cursor: "m4", lastAssistant: "second", boundaries: [{ cursor: "m4", lastAssistant: "second", count: 2 }] },
  ];
  const emitted = [];
  let state = {};
  const hooks = createHooks({
    logger: { warn() {} },
    trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: { clear() {}, inject() {}, recall: async () => {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  const idle = { event: { type: "session.idle", properties: { sessionID: "s" } } };
  await hooks.event(idle);
  await hooks.event(idle);
  await hooks.event(idle);
  const turns = emitted.filter(([type]) => type === "turn_end");
  assert.deepEqual(turns.map(([, , options]) => options.objectId), ["m2", "m4"]);
  assert.deepEqual(turns.map(([, , options]) => options.content), ["first", "second"]);
});

test("trace enqueue completes before tool hooks return", async () => {
  let completed = false;
  const hooks = createHooks({
    logger: { warn() {} },
    trace: { emit: async () => { await Promise.resolve(); completed = true; }, flush: async () => {} },
    compose: { clear() {}, inject() {}, recall: async () => {} },
    sync: { sync: async () => ({ success: true, count: 0 }), flush: async () => {} },
    store: { remove: async () => {} },
  });
  await hooks["tool.execute.before"]({ sessionID: "s", callID: "c", tool: "read" }, { args: {} });
  assert.equal(completed, true);
});

test("tool failure resolves session ID from the real part event shape", async () => {
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: { clear() {}, inject() {}, recall: async () => {} }, sync: { sync: async () => ({ success: true, count: 0 }), flush: async () => {} }, store: { remove: async () => {} },
  });
  await hooks.event({ event: { type: "message.part.updated", properties: { part: { type: "tool", sessionID: "s", callID: "c", tool: "shell", state: { status: "error", error: "failed" } } } } });
  assert.equal(emitted[0][0], "tool_failure");
  assert.equal(emitted[0][1], "s");
});

test("failed deletion sync retains durable session state", async () => {
  let removed = false;
  let pendingSessionEnd = false;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => {}, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: false, count: 0, boundaries: [] }), flush: async () => {} }, store: { get: () => ({}), remove: async () => { removed = true; }, set: async (_id, value) => { pendingSessionEnd = value.pendingSessionEnd; } },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.equal(removed, false);
  assert.equal(pendingSessionEnd, true);
});

test("dispose has a hard deadline", async () => {
  const never = new Promise(() => {});
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => {}, flush: () => never }, compose: {},
    sync: { flush: () => never }, store: {}, disposeTimeoutMs: 20,
  });
  const started = Date.now();
  await hooks.dispose();
  assert.ok(Date.now() - started < 200);
});

test("startup replay finalizes and removes a deleted-session tombstone", async () => {
  let removed = false;
  const events = [];
  await replayStoredSessions({
    store: { list: () => [["s", { pendingMessages: [{ role: "user", content: "x" }], pendingSessionEnd: true }]], remove: async () => { removed = true; } },
    sync: { replayPending: async () => ({ success: true, boundaries: [{ cursor: "m2", lastAssistant: "done" }] }) },
    trace: { emit: async (type) => events.push(type) },
  });
  assert.deepEqual(events, ["turn_end", "session_end"]);
  assert.equal(removed, true);
});

test("startup replay replaces same-turn intermediate boundary", async () => {
  const events = [];
  await replayStoredSessions({
    store: { list: () => [["s", { pendingMessages: [{ role: "assistant", content: "final" }], pendingTurnBoundaries: [{ cursor: "cur1", turnKey: "u", lastAssistant: "mid" }] }]], set: async () => {} },
    sync: { replayPending: async () => ({ success: true, boundaries: [{ cursor: "cur2", turnKey: "u", lastAssistant: "final" }] }) },
    trace: { emit: async (type, _session, options) => events.push([type, options.objectId]) },
  });
  assert.deepEqual(events, [["turn_end", "cur2"]]);
});

test("startup replay retains boundaries when Trace cannot be spooled", async () => {
  let state = { pendingTurnBoundaries: [{ cursor: "m2", turnKey: "u1", lastAssistant: "answer" }], pendingSessionEnd: true };
  let removed = false;
  await replayStoredSessions({
    store: {
      list: () => [["s", state]],
      set: async (_id, value) => { state = { ...state, ...value }; },
      remove: async () => { removed = true; },
    },
    sync: { replayPending: async () => ({ success: true, boundaries: [] }) },
    trace: { emit: async () => false },
  });
  assert.equal(removed, false);
  assert.equal(state.pendingSessionEnd, true);
  assert.equal(state.pendingTurnBoundaries[0].cursor, "m2");
});

test("completed assistant messages sync before a later deletion event", async () => {
  let syncs = 0;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => {}, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => { syncs += 1; return { success: true, count: 0, boundaries: [] }; }, flush: async () => {} },
    store: { get: () => ({}), set: async () => {}, remove: async () => {} },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { id: "m", sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  assert.equal(syncs, 1);
});

test("assistant completion stages one final turn boundary until idle", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, count: 2, boundaries: [{ cursor: "m2", turnKey: "u1", lastAssistant: "tooling", count: 2 }] },
    { success: true, count: 1, boundaries: [{ cursor: "m4", turnKey: "u1", lastAssistant: "final", count: 1 }] },
    { success: true, count: 0, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  const completed = (id) => ({ event: { type: "message.updated", properties: { info: { id, sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  await hooks.event(completed("m2"));
  await hooks.event(completed("m4"));
  assert.equal(emitted.filter(([type]) => type === "turn_end").length, 0);
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  const turns = emitted.filter(([type]) => type === "turn_end");
  assert.equal(turns.length, 1);
  assert.equal(turns[0][2].objectId, "m4");
  assert.equal(turns[0][2].content, "final");
});

test("a recovered prior turn and current turn keep separate boundaries", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, count: 4, boundaries: [
      { cursor: "m2", lastAssistant: "recovered", count: 2 },
      { cursor: "m4", lastAssistant: "current", count: 2 },
    ] },
    { success: true, count: 0, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  assert.deepEqual(emitted.filter(([type]) => type === "turn_end").map(([, , options]) => options.objectId), ["m2", "m4"]);
});

test("a recovered turn survives later tool-loop completion updates", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, boundaries: [
      { cursor: "old", turnKey: "u0", lastAssistant: "recovered" },
      { cursor: "mid", turnKey: "u1", lastAssistant: "tooling" },
    ] },
    { success: true, boundaries: [{ cursor: "final", turnKey: "u1", lastAssistant: "done" }] },
    { success: true, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  const completed = { event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } };
  await hooks.event(completed);
  await hooks.event(completed);
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  assert.deepEqual(emitted.filter(([type]) => type === "turn_end").map(([, , options]) => options.objectId), ["old", "final"]);
});

test("pending recovery replaces an intermediate boundary from the same user turn", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, boundaries: [{ cursor: "cur1", turnKey: "user1", lastAssistant: "tooling" }] },
    { success: true, boundaries: [{ cursor: "cur2", turnKey: "user1", lastAssistant: "final" }] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  assert.deepEqual(emitted.filter(([type]) => type === "turn_end").map(([, , options]) => options.objectId), ["cur2"]);
});

test("independent session errors receive distinct stable sequence IDs", async () => {
  let state = {};
  const ids = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (_type, _session, options) => ids.push(options.objectId), flush: async () => {} }, compose: {},
    sync: { sync: async () => ({ success: true, count: 0, boundaries: [] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } },
  });
  const error = { event: { type: "session.error", properties: { sessionID: "s", error: { name: "Error", message: "same" } } } };
  await hooks.event(error);
  await hooks.event(error);
  assert.deepEqual(ids, ["error-1", "error-2"]);
});

test("session error upgrades an already staged active turn boundary", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, boundaries: [{ cursor: "m2", lastAssistant: "partial" }] },
    { success: true, boundaries: [] },
    { success: true, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  await hooks.event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "Error" } } } });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  const turn = emitted.find(([type]) => type === "turn_end");
  assert.equal(turn[2].terminalState, "error");
});

test("fire-and-forget events preserve per-session boundary order", async () => {
  let state = {};
  const emitted = [];
  const results = [
    { success: true, boundaries: [{ cursor: "m2", turnKey: "u1", lastAssistant: "partial" }] },
    { success: true, boundaries: [] },
    { success: true, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => { await new Promise((resolve) => setImmediate(resolve)); return results.shift(); }, flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { await Promise.resolve(); state = { ...state, ...value }; }, remove: async () => {} },
  });
  const operations = [
    hooks.event({ event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } }),
    hooks.event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "Error" } } } }),
    hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } }),
  ];
  await Promise.all(operations);
  const turns = emitted.filter(([type]) => type === "turn_end");
  assert.equal(turns.length, 1);
  assert.equal(turns[0][2].objectId, "m2");
  assert.equal(turns[0][2].terminalState, "error");
});

test("event serialization does not block a different session", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (type, session) => { if (session === "slow") await gate; emitted.push([type, session]); }, flush: async () => {} },
    compose: {}, sync: { flush: async () => {} }, store: {},
  });
  const slow = hooks.event({ event: { type: "session.created", properties: { sessionID: "slow" } } });
  const fast = hooks.event({ event: { type: "session.created", properties: { sessionID: "fast" } } });
  await fast;
  assert.deepEqual(emitted, [["session_start", "fast"]]);
  release();
  await slow;
});

test("failed Trace enqueue retains turn and deletion recovery state", async () => {
  let state = {};
  let removed = false;
  const results = [
    { success: true, boundaries: [{ cursor: "m2", turnKey: "u1", lastAssistant: "answer" }] },
    { success: true, boundaries: [] },
  ];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => false, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => results.shift(), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } },
  });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  assert.equal(state.pendingTurnBoundaries[0].cursor, "m2");
  await hooks.event({ event: { type: "session.deleted", properties: { sessionID: "s" } } });
  assert.equal(removed, false);
  assert.equal(state.pendingSessionEnd, true);
  assert.equal(state.pendingTurnBoundaries[0].cursor, "m2");
});

test("pre-compact waits for sync, emits its cursor, and fails open", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const emitted = [];
  const warnings = [];
  let fail = false;
  const hooks = createHooks({
    logger: { warn: (message) => warnings.push(message) },
    trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: {},
    sync: {
      sync: async () => {
        if (fail) throw new Error("transcript unavailable");
        await gate;
        return { success: true, cursor: "m9", count: 9, boundaries: [] };
      },
      flush: async () => {},
    },
    store: { get: () => ({}), set: async () => {} },
  });

  const pending = hooks["experimental.session.compacting"]({ sessionID: "s" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(emitted, []);
  release();
  await pending;
  assert.deepEqual(emitted, [["pre_compact", "s", { objectId: "m9", metadata: { message_count: 9 } }]]);

  fail = true;
  await assert.doesNotReject(hooks["experimental.session.compacting"]({ sessionID: "s" }));
  assert.match(warnings[0], /failed open/);
});

test("plugin factory wires recall, write-back, and canonical Trace", async () => {
  let state = {};
  const traceEnvelopes = [];
  const writes = [];
  const logs = [];
  const messages = [
    { info: { id: "m1", sessionID: "s", role: "user", time: { created: 1 } }, parts: [{ type: "text", text: "question" }] },
    { info: { id: "m2", sessionID: "s", role: "assistant", time: { created: 2, completed: 3 } }, parts: [{ type: "text", text: "answer" }] },
  ];
  const store = {
    load: async () => {}, get: () => state, list: () => [],
    set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { state = {}; },
  };
  const hooks = await OgMemoryPlugin({ client: {
    app: { log: async ({ body }) => logs.push(body) },
    session: { messages: async () => ({ data: messages }) },
  } }, {
    detectCompatibility: async () => ({ supported: true, hasDispose: true }),
    loadConfig: async () => ({
      enabled: true, endpoint: "http://127.0.0.1:8090", runtimeDir: "unused", accountId: "a", userId: "u", agentId: "opencode",
      traceCaptureContent: true, traceSpoolLimit: 10, traceSpoolTtlSeconds: 60, transcriptTimeoutMs: 1000, transcriptLimit: 100,
      composeCacheTtlMs: 60000, disposeTimeoutMs: 100, sessionStateLimit: 10, sessionStateTtlSeconds: 60, sessionStateMaxBytes: 1048576,
    }),
    createHttpClient: () => ({
      compose: async () => ({ retrievedEvidence: "remembered context" }),
      afterTurn: async (_session, payload) => { writes.push(payload); return { ok: true }; },
      trace: async () => ({ ok: true }),
    }),
    createSessionStore: () => store,
    createTraceSpool: () => ({ enqueue: async (value) => traceEnvelopes.push(value), drain: async () => {}, flush: async () => {} }),
  });
  await hooks["chat.message"]({ sessionID: "s" }, { message: messages[0].info, parts: [{ type: "text", text: "question" }] });
  const transformed = { system: [] };
  await hooks["experimental.chat.system.transform"]({ sessionID: "s" }, transformed);
  assert.match(transformed.system[0], /remembered context/);
  await hooks.event({ event: { type: "message.updated", properties: { info: messages[1].info } } });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  assert.deepEqual(writes[0], [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }]);
  const types = traceEnvelopes.map((item) => item.event.event_type);
  assert.ok(types.includes("user_message"));
  assert.ok(types.includes("assistant_message"));
  assert.ok(types.includes("turn_end"));
  const userTraces = traceEnvelopes.filter((item) => item.event.event_type === "user_message");
  assert.equal(userTraces.length, 2);
  assert.equal(userTraces[0].request_id, userTraces[1].request_id);
  assert.deepEqual(logs, []);
});

test("plugin factory honors configured disablement before runtime initialization", async () => {
  let initialized = false;
  const logs = [];
  const hooks = await OgMemoryPlugin({ client: { app: { log: async ({ body }) => logs.push(body) } } }, {
    detectCompatibility: async () => ({ supported: true, hasDispose: true }),
    loadConfig: async () => ({ enabled: false }),
    createHttpClient: () => { initialized = true; },
  });
  assert.deepEqual(hooks, {});
  assert.equal(initialized, false);
  assert.match(logs[0].message, /disabled/);
});

test("plugin factory fails open when durable runtime initialization fails", async () => {
  const logs = [];
  const hooks = await OgMemoryPlugin({ client: { app: { log: async ({ body }) => logs.push(body) } } }, {
    detectCompatibility: async () => ({ supported: true, hasDispose: true }),
    loadConfig: async () => ({ enabled: true, runtimeDir: "unused" }),
    createHttpClient: () => ({}),
    createSessionStore: () => ({ load: async () => { throw new Error("disk denied /secret/path"); } }),
  });
  assert.deepEqual(hooks, {});
  assert.equal(logs[0].level, "error");
  assert.match(logs[0].message, /initialization failed open/);
  assert.doesNotMatch(logs[0].message, /disk denied|secret/);
});

test("console fallback preserves informational log severity", async () => {
  const infos = [];
  const warnings = [];
  const originalInfo = console.info;
  const originalWarn = console.warn;
  console.info = (message) => infos.push(message);
  console.warn = (message) => warnings.push(message);
  try {
    const hooks = await OgMemoryPlugin({ client: {} }, {
      detectCompatibility: async () => ({ supported: true, hasDispose: true }),
      loadConfig: async () => ({ enabled: false }),
    });
    assert.deepEqual(hooks, {});
    assert.match(infos[0], /disabled/);
    assert.equal(warnings.length, 0);
  } finally {
    console.info = originalInfo;
    console.warn = originalWarn;
  }
});

test("session-scoped events without a session ID do not create empty state", async () => {
  let writes = 0;
  let syncs = 0;
  const warnings = [];
  const hooks = createHooks({
    logger: { warn: (message) => warnings.push(message) }, trace: { emit: async () => true, flush: async () => {} }, compose: {},
    sync: { sync: async () => { syncs += 1; return { success: true, boundaries: [] }; }, flush: async () => {} },
    store: { get: () => ({}), set: async () => { writes += 1; } },
  });
  await hooks.event({ event: { type: "session.error", properties: { error: { name: "GlobalError" } } } });
  assert.equal(writes, 0);
  assert.equal(syncs, 0);
  assert.match(warnings[0], /without a session ID/);
});
