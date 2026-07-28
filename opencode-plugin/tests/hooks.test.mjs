import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHooks, OgMemoryPlugin, replayStoredSessions } from "../lib/plugin.mjs";
import { createMessageSync } from "../lib/message-sync.mjs";
import { createSessionStore } from "../lib/session-state.mjs";

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

test("chat message remains fail-open when deletion fallback persistence fails", async () => {
  let recalled = false;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} },
    compose: { recall: async () => { recalled = true; } }, sync: { flush: async () => {} },
    store: { set: async () => { throw new Error("disk full"); } },
  });
  await hooks["chat.message"]({ sessionID: "s" }, { message: { id: "m1" }, parts: [{ type: "text", text: "remember" }] });
  assert.equal(recalled, true);
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

test("tool hooks emit one start and one successful terminal event", async () => {
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: {}, sync: { flush: async () => {} }, store: {},
  });
  const input = { sessionID: "s", callID: "call-1", tool: "read" };
  await hooks["tool.execute.before"](input, { args: { path: "x" } });
  await hooks["tool.execute.after"](input, { output: "done" });
  assert.deepEqual(emitted.map(([type]) => type), ["tool_start", "tool_success"]);
  assert.equal(emitted[1][2].objectId, "call-1");
  assert.equal(emitted[1][2].terminalState, "success");
});

test("session start Trace records the OpenCode parent session relationship", async () => {
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => emitted.push(args), flush: async () => {} },
    compose: {}, sync: { flush: async () => {} }, store: {},
  });
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "child", parentID: "parent" } } } });
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "root" } } } });
  assert.deepEqual(emitted[0], ["session_start", "child", { objectId: "child", metadata: { parent_session_id: "parent" } }]);
  assert.deepEqual(emitted[1], ["session_start", "root", { objectId: "root", metadata: {} }]);
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

test("deletion tombstone is durable without advancing the transcript cursor", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "last question" }], deletionFallbackCursor: "m1" };
  let stateSeenBySync;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => { stateSeenBySync = structuredClone(state); return { success: false, count: 0, boundaries: [] }; }, flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.equal(stateSeenBySync.pendingSessionEnd, true);
  assert.equal(stateSeenBySync.pendingMessages, undefined);
  assert.equal(stateSeenBySync.cursor, undefined);
  assert.deepEqual(stateSeenBySync.deletionFallbackMessages, [{ role: "user", content: "last question" }]);
});

test("deletion synchronizes older transcript turns before the latest fallback", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "latest question" }], deletionFallbackCursor: "m3" };
  const writes = [];
  let removed = false;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } };
  const messages = [
    { info: { id: "m1", role: "user", time: { created: 1 } }, parts: [{ type: "text", text: "old question" }] },
    { info: { id: "m2", role: "assistant", time: { created: 2 } }, parts: [{ type: "text", text: "old answer" }] },
    { info: { id: "m3", role: "user", time: { created: 3 } }, parts: [{ type: "text", text: "latest question" }] },
  ];
  const sync = createMessageSync({ client: { session: { messages: async () => ({ data: messages }) } }, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store });
  const hooks = createHooks({ logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} }, sync, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[
    { role: "user", content: "old question" }, { role: "assistant", content: "old answer" }, { role: "user", content: "latest question" },
  ]]);
  assert.equal(removed, true);
});

test("a transient tombstone persistence failure does not abandon deletion synchronization", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "last question" }], deletionFallbackCursor: "m1" };
  let firstSet = true;
  const writes = [];
  let removed = false;
  const store = {
    get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; if (firstSet) { firstSet = false; throw new Error("disk busy"); } },
    remove: async () => { removed = true; },
  };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: false, committed: false, committedMessageIds: [], count: 0, boundaries: [], transcriptUnavailable: true, transcriptGone: true }), flush: async () => {} },
    http: { afterTurn: async (_id, payload) => writes.push(payload) }, store,
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[{ role: "user", content: "last question" }]]);
  assert.equal(removed, true);
});

test("deleted session writes the durable user fallback when transcript is already gone", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "last question" }], deletionFallbackCursor: "m1" };
  const writes = [];
  let removed = false;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: false, committed: false, committedMessageIds: [], count: 0, cursor: null, boundaries: [], transcriptUnavailable: true, transcriptGone: true }), flush: async () => {} },
    http: { afterTurn: async (_sessionId, messages) => writes.push(messages) },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[{ role: "user", content: "last question" }]]);
  assert.equal(removed, true);
});

test("concurrent plugin processes commit a deletion fallback only once", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-delete-lock-"));
  const path = join(directory, "sessions.json");
  const seed = createSessionStore(path);
  await seed.load();
  await seed.set("s", {
    deletionFallbackMessages: [{ role: "user", content: "last question" }],
    deletionFallbackCursor: "m1",
  });
  const stores = [createSessionStore(path), createSessionStore(path)];
  await Promise.all(stores.map((candidate) => candidate.load()));

  let releaseFirst;
  let firstStarted;
  const started = new Promise((resolve) => { firstStarted = resolve; });
  const gate = new Promise((resolve) => { releaseFirst = resolve; });
  let writes = 0;
  const http = {
    afterTurn: async () => {
      writes += 1;
      if (writes === 1) { firstStarted(); await gate; }
    },
  };
  const create = (store) => createHooks({
    logger: { warn() {} },
    trace: { emit: async () => true, flush: async () => {} },
    compose: { clear() {} },
    sync: {
      sync: async () => ({ success: false, committed: false, committedMessageIds: [], resolvedMessageIds: [], count: 0, cursor: null, boundaries: [], transcriptUnavailable: true, transcriptGone: true }),
      flush: async () => {},
    },
    http,
    store,
  });
  const event = { event: { type: "session.deleted", properties: { info: { id: "s" } } } };
  const first = create(stores[0]).event(event);
  await started;
  const second = create(stores[1]).event(event);
  releaseFirst();
  await Promise.all([first, second]);

  assert.equal(writes, 1);
});

test("committed deletion fallback retains its turn boundary for startup recovery", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-delete-boundary-"));
  const path = join(directory, "sessions.json");
  const store = createSessionStore(path);
  await store.load();
  await store.set("s", {
    deletionFallbackMessages: [{ role: "user", content: "last question" }],
    deletionFallbackCursor: "m1",
  });
  const gone = { success: false, committed: false, committedMessageIds: [], resolvedMessageIds: [], count: 0, cursor: null, boundaries: [], transcriptUnavailable: true, transcriptGone: true };
  const hooks = createHooks({
    logger: { warn() {} },
    trace: { emit: async (type) => type !== "turn_end", flush: async () => {} },
    compose: { clear() {} },
    sync: { sync: async () => gone, flush: async () => {} },
    http: { afterTurn: async () => ({ ok: true }) },
    store,
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });

  const recovered = createSessionStore(path);
  await recovered.load();
  assert.deepEqual(recovered.get("s").pendingTurnBoundaries.map((boundary) => boundary.cursor), ["m1"]);
  const emitted = [];
  await replayStoredSessions({
    store: recovered,
    sync: { sync: async () => ({ ...gone, success: true, transcriptUnavailable: false, transcriptGone: false }) },
    trace: { emit: async (type, _sessionId, options) => { emitted.push([type, options.objectId]); return true; } },
  });
  assert.deepEqual(emitted, [["turn_end", "m1"], ["session_end", "s"]]);
  assert.deepEqual(recovered.list(), []);
});

test("failed deletion sync promotes the durable fallback for startup replay", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "last question" }], deletionFallbackCursor: "m1" };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: false, count: 0, boundaries: [] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(state.pendingMessages, [{ role: "user", content: "last question" }]);
  assert.equal(state.pendingCursor, "m1");
  assert.equal(state.pendingSessionEnd, true);
});

test("failed deletion sync preserves the complete pending transcript", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "last question" }], deletionFallbackCursor: "m1" };
  const pending = [{ role: "user", content: "last question" }, { role: "assistant", content: "last answer" }];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => { state = { ...state, pendingMessages: pending, pendingCursor: "m2", pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "last answer" }] }; return { success: false, count: 0, boundaries: [] }; }, flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(state.pendingMessages, pending);
  assert.equal(state.pendingCursor, "m2");
  assert.equal(state.pendingSessionEnd, true);
});

test("failed pending replay appends a newer deletion fallback without losing prior messages", async () => {
  const prior = [{ role: "user", content: "old question" }, { role: "assistant", content: "old answer" }];
  let state = { pendingMessages: prior, pendingCursor: "m2", pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "old answer" }], deletionFallbackMessages: [{ role: "user", content: "new question" }], deletionFallbackCursor: "m3" };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: false, count: 0, boundaries: [] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(state.pendingMessages, [...prior, { role: "user", content: "new question" }]);
  assert.equal(state.pendingCursor, "m3");
  assert.deepEqual(state.pendingBoundaries.map((boundary) => boundary.turnKey), ["m1", "m3"]);
});

test("deleted session does not replay fallback after pending messages were already committed", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1" };
  const emitted = [];
  const fallbackWrites = [];
  let removed = false;
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => { emitted.push(args); return true; }, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => { state = { ...state, pendingMessages: [], pendingBoundaries: [] }; return { success: false, committed: true, committedMessageIds: ["m1", "m2"], count: 2, cursor: "m2", lastAssistant: "answer", boundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }] }; }, flush: async () => {} },
    http: { afterTurn: async (...args) => fallbackWrites.push(args) },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(fallbackWrites, []);
  assert.equal(removed, true);
  assert.deepEqual(emitted.map(([type]) => type), ["turn_end", "session_end"]);
  assert.equal(emitted[0][2].content, "answer");
});

test("deleted session writes a newer fallback after an older pending turn committed", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "new question" }], deletionFallbackCursor: "m3" };
  const writes = [];
  let removed = false;
  const emitted = [];
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async (...args) => { emitted.push(args); return true; }, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => {
      return { success: false, committed: true, committedMessageIds: ["m1", "m2"], count: 2, cursor: "m2", lastAssistant: "old answer", boundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "old answer", count: 2 }], transcriptUnavailable: true, transcriptGone: true };
    }, flush: async () => {} },
    http: { afterTurn: async (_sessionId, messages) => writes.push(messages) },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } },
  });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[{ role: "user", content: "new question" }]]);
  assert.equal(removed, true);
  assert.deepEqual(emitted.map(([type]) => type), ["turn_end", "turn_end", "session_end"]);
});

test("transient deletion transcript failure keeps the tombstone and later recovers all messages", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "latest question" }], deletionFallbackCursor: "m3" };
  let transcriptAttempts = 0;
  let removed = false;
  const writes = [];
  const messages = [
    { info: { id: "m1", role: "user", time: { created: 1 } }, parts: [{ type: "text", text: "old question" }] },
    { info: { id: "m2", role: "assistant", time: { created: 2 } }, parts: [{ type: "text", text: "old answer" }] },
    { info: { id: "m3", role: "user", time: { created: 3 } }, parts: [{ type: "text", text: "latest question" }] },
  ];
  const store = {
    get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; },
    remove: async () => { removed = true; },
  };
  const http = { afterTurn: async (_id, payload) => { writes.push(payload); return { ok: true }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => { transcriptAttempts += 1; if (transcriptAttempts <= 2) throw new Error("temporary SDK failure"); return { data: messages }; } } },
    http, store, logger: { warn() {} },
  });
  const hooks = createHooks({ logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} }, sync, http, store });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.equal(removed, false);
  assert.equal(state.pendingSessionEnd, true);
  assert.deepEqual(writes, []);

  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.equal(removed, true);
  assert.deepEqual(writes, [[
    { role: "user", content: "old question" },
    { role: "assistant", content: "old answer" },
    { role: "user", content: "latest question" },
  ]]);
});

test("startup recovery preserves a deletion tombstone across transient transcript failure", async () => {
  let state = { pendingSessionEnd: true, deletionFallbackMessages: [{ role: "user", content: "latest question" }], deletionFallbackCursor: "m3" };
  let removed = false;
  let transcriptAttempts = 0;
  const writes = [];
  const messages = [
    { info: { id: "m1", role: "user", time: { created: 1 } }, parts: [{ type: "text", text: "old question" }] },
    { info: { id: "m2", role: "assistant", time: { created: 2 } }, parts: [{ type: "text", text: "old answer" }] },
    { info: { id: "m3", role: "user", time: { created: 3 } }, parts: [{ type: "text", text: "latest question" }] },
  ];
  const store = {
    get: () => state,
    list: () => removed ? [] : [["s", state]],
    set: async (_id, value) => { state = { ...state, ...value }; },
    remove: async () => { removed = true; },
  };
  const http = { afterTurn: async (_id, payload) => { writes.push(payload); return { ok: true }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => { transcriptAttempts += 1; if (transcriptAttempts <= 2) throw new Error("temporary startup SDK failure"); return { data: messages }; } } },
    http, store, logger: { warn() {} },
  });
  const trace = { emit: async () => true };
  await replayStoredSessions({ store, sync, trace });
  assert.equal(removed, false);
  assert.deepEqual(writes, []);
  assert.equal(state.pendingSessionEnd, true);

  await replayStoredSessions({ store, sync, trace });
  assert.equal(removed, true);
  assert.deepEqual(writes, [[
    { role: "user", content: "old question" },
    { role: "assistant", content: "old answer" },
    { role: "user", content: "latest question" },
  ]]);
});

test("scan cursor alone does not clear an uncommitted deletion fallback", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "must survive" }], deletionFallbackCursor: "m1" };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: true, committed: false, committedMessageIds: [], count: 0, cursor: "m1", boundaries: [] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { id: "m1", sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  assert.deepEqual(state.deletionFallbackMessages, [{ role: "user", content: "must survive" }]);
  assert.equal(state.deletionFallbackCursor, "m1");
});

test("boundary-less legacy pending state commits a fallback only once", async () => {
  let state = {
    pendingMessages: [{ role: "user", content: "question" }], pendingCursor: "m1", pendingBoundaries: [],
    deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1",
  };
  const writes = [];
  let removed = false;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store, logger: { warn() {} } });
  const hooks = createHooks({ logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} }, sync, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[{ role: "user", content: "question" }]]);
  assert.equal(removed, true);
});

test("pending user-assistant recovery covers its user deletion fallback", async () => {
  let state = {
    pendingMessages: [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }], pendingCursor: "m2",
    pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }],
    deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1",
  };
  const writes = [];
  let removed = false;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store, logger: { warn() {} } });
  const hooks = createHooks({ logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} }, sync, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.deepEqual(writes, [[{ role: "user", content: "question" }, { role: "assistant", content: "answer" }]]);
  assert.equal(removed, true);
});

test("permanently rejected pending fallback is not submitted twice during deletion", async () => {
  let state = {
    pendingMessages: [{ role: "user", content: "invalid" }], pendingCursor: "m1",
    pendingBoundaries: [{ cursor: "m1", turnKey: "m1", lastAssistant: "", count: 1 }],
    deletionFallbackMessages: [{ role: "user", content: "invalid" }], deletionFallbackCursor: "m1",
  };
  let attempts = 0;
  let removed = false;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } };
  const rejected = Object.assign(new Error("invalid"), { status: 400, retryable: false });
  const sync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async () => { attempts += 1; throw rejected; } }, store, logger: { warn() {} } });
  const hooks = createHooks({ logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} }, sync, http: { afterTurn: async () => { attempts += 1; throw rejected; } }, store });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s" } } } });
  assert.equal(attempts, 1);
  assert.equal(removed, true);
  assert.deepEqual(state.deletionFallbackMessages, []);
});

test("later completion cannot downgrade a staged error boundary", async () => {
  let state = { pendingTurnBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "failed", count: 2, terminalState: "error" }] };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: { clear() {} },
    sync: { sync: async () => ({ success: true, committed: true, resolvedMessageIds: ["m1", "m3"], count: 2, cursor: "m3", boundaries: [{ cursor: "m3", turnKey: "m1", lastAssistant: "late", count: 2 }] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } },
  });
  await hooks.event({ event: { type: "message.updated", properties: { info: { id: "m3", sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  assert.equal(state.pendingTurnBoundaries[0].terminalState, "error");
  assert.equal(state.pendingTurnBoundaries[0].cursor, "m2");
});

test("dispose deadline keeps an otherwise idle process alive through trace close", () => {
  const pluginUrl = new URL("../lib/plugin.mjs", import.meta.url).href;
  const script = `
    import { createHooks } from ${JSON.stringify(pluginUrl)};
    const never = new Promise(() => {});
    let closed = false;
    const hooks = createHooks({
      logger: { warn() {} }, trace: { emit: async () => {}, flush: () => never, close: () => { closed = true; } }, compose: {},
      sync: { flush: () => never }, store: {}, disposeTimeoutMs: 20,
    });
    await hooks.dispose();
    if (!closed) throw new Error("trace close was not called");
    console.log("dispose-closed");
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "--eval", script], { encoding: "utf8", timeout: 1000 });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /dispose-closed/);
});

test("dispose waits for event hooks before flushing sync and trace", async () => {
  const order = [];
  let releaseEvent;
  const eventGate = new Promise((resolve) => { releaseEvent = resolve; });
  const hooks = createHooks({
    logger: { warn() {} },
    trace: {
      emit: async () => { order.push("event-start"); await eventGate; order.push("event-end"); return true; },
      flush: async () => { order.push("trace-flush"); },
      close: () => { order.push("trace-close"); },
    },
    compose: {},
    sync: { flush: async () => { order.push("sync-flush"); } },
    store: {},
    disposeTimeoutMs: 1000,
  });
  const event = hooks.event({ event: { type: "session.created", properties: { info: { id: "s" } } } });
  await new Promise((resolve) => setImmediate(resolve));
  const disposed = hooks.dispose();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(order, ["event-start"]);
  releaseEvent();
  await Promise.all([event, disposed]);
  assert.deepEqual(order, ["event-start", "event-end", "sync-flush", "trace-flush", "trace-close"]);
});

test("startup replay finalizes and removes a deleted-session tombstone", async () => {
  let state = { pendingMessages: [{ role: "user", content: "x" }], pendingSessionEnd: true };
  let removed = false;
  const events = [];
  await replayStoredSessions({
    store: { list: () => [["s", state]], get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } },
    sync: { sync: async () => { state = { ...state, pendingMessages: [] }; return { success: false, committed: true, boundaries: [{ cursor: "m2", lastAssistant: "done" }] }; }, replayPending: async () => ({ success: true, boundaries: [] }) },
    trace: { emit: async (type) => events.push(type) },
  });
  assert.deepEqual(events, ["turn_end", "session_end"]);
  assert.equal(removed, true);
});

test("startup recovery composes the real pending producer and consumer contracts", async () => {
  let state = {
    pendingMessages: [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }],
    pendingCursor: "m2", pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }],
    pendingSessionEnd: true,
  };
  const writes = [];
  const emitted = [];
  let removed = false;
  const store = {
    list: () => [["s", state]], get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; },
  };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [] }) } },
    http: { afterTurn: async (_sessionId, messages) => writes.push(messages) }, store, logger: { warn() {} },
  });
  await replayStoredSessions({ store, sync, trace: { emit: async (...args) => { emitted.push(args); return true; } } });
  assert.deepEqual(writes, [[{ role: "user", content: "question" }, { role: "assistant", content: "answer" }]]);
  assert.deepEqual(emitted.map(([type]) => type), ["turn_end", "session_end"]);
  assert.equal(removed, true);
});

test("startup trace failure does not resend an acknowledged deletion fallback", async () => {
  let state = { pendingSessionEnd: true, deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1" };
  const writes = [];
  const store = {
    list: () => [["s", state]], get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {},
  };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store, logger: { warn() {} } });
  const trace = { emit: async () => false };
  await replayStoredSessions({ store, sync, trace });
  await replayStoredSessions({ store, sync, trace });
  assert.deepEqual(writes, [[{ role: "user", content: "question" }]]);
  assert.deepEqual(state.deletionFallbackMessages, []);
});

test("startup retryable pending failure is attempted only once", async () => {
  let state = {
    pendingSessionEnd: true, pendingMessages: [{ role: "user", content: "question" }], pendingCursor: "m1",
    pendingBoundaries: [{ cursor: "m1", turnKey: "m1", lastAssistant: "", count: 1 }],
    deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1",
  };
  let attempts = 0;
  const store = { list: () => [["s", state]], get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => {} };
  const sync = createMessageSync({ client: { session: { messages: async () => ({ data: [] }) } }, http: { afterTurn: async () => { attempts += 1; throw new Error("timeout"); } }, store, logger: { warn() {} } });
  await replayStoredSessions({ store, sync, trace: { emit: async () => true } });
  assert.equal(attempts, 1);
  assert.deepEqual(state.pendingMessages, [{ role: "user", content: "question" }]);
});

test("startup stale boundary cannot suppress a fallback payload", async () => {
  let state = {
    pendingSessionEnd: true, pendingMessages: [], pendingBoundaries: [{ cursor: "m1", turnKey: "m1", lastAssistant: "", count: 1 }],
    deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1",
  };
  const writes = [];
  let removed = false;
  const store = { list: () => [["s", state]], get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { removed = true; } };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async (_id, payload) => writes.push(payload) }, store, logger: { warn() {} } });
  await replayStoredSessions({ store, sync, trace: { emit: async () => true } });
  assert.deepEqual(writes, [[{ role: "user", content: "question" }]]);
  assert.equal(removed, true);
});

test("acknowledged pending boundary survives a restart before Trace emission", async () => {
  let state = {
    pendingSessionEnd: true, pendingMessages: [{ role: "user", content: "question" }], pendingCursor: "m1",
    pendingBoundaries: [{ cursor: "m1", turnKey: "m1", lastAssistant: "", count: 1 }],
  };
  const store = { list: () => [["s", state]], get: () => state, set: async (_id, value) => { state = { ...state, ...value }; }, remove: async () => { state = { removed: true }; } };
  const firstSync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async () => ({ ok: true }) }, store, logger: { warn() {} } });
  const firstResult = await firstSync.sync("s");
  assert.equal(firstResult.committed, true);
  assert.deepEqual(state.pendingTurnBoundaries, [{ cursor: "m1", turnKey: "m1", lastAssistant: "", count: 1 }]);
  const secondSync = createMessageSync({ client: { session: { messages: async () => { throw Object.assign(new Error("deleted"), { status: 404 }); } } }, http: { afterTurn: async () => { throw new Error("must not resend"); } }, store, logger: { warn() {} } });
  const emitted = [];
  await replayStoredSessions({ store, sync: secondSync, trace: { emit: async (type) => { emitted.push(type); return true; } } });
  assert.deepEqual(emitted, ["turn_end", "session_end"]);
  assert.equal(state.removed, true);
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
      get: () => state,
      set: async (_id, value) => { state = { ...state, ...value }; },
      remove: async () => { removed = true; },
    },
    sync: { sync: async () => ({ success: true, committed: false, boundaries: [] }), replayPending: async () => ({ success: true, boundaries: [] }) },
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

test("real hooks selectively acknowledge a boundary staged by another process", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-hook-boundary-cas-"));
  const path = join(root, "sessions.json");
  const firstStore = createSessionStore(path);
  const secondStore = createSessionStore(path);
  await Promise.all([firstStore.load(), secondStore.load()]);
  const delivered = { cursor: "m1", turnKey: "u1", lastAssistant: "first", count: 2, terminalState: "idle" };
  const newer = { cursor: "m2", turnKey: "u1", lastAssistant: "revised", count: 3, terminalState: "error" };
  await firstStore.updateTurnBoundaries("s", { retained: [delivered] });
  let releaseTrace;
  let traceStarted;
  const started = new Promise((resolve) => { traceStarted = resolve; });
  const gate = new Promise((resolve) => { releaseTrace = resolve; });
  const hooks = createHooks({
    logger: { warn() {} }, compose: { clear() {} }, store: firstStore,
    sync: { sync: async () => ({ success: true, boundaries: [] }), flush: async () => {} },
    trace: { emit: async (type) => { if (type === "turn_end") { traceStarted(); await gate; } return true; }, flush: async () => {} },
  });
  const idle = hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } });
  await started;
  await secondStore.updateTurnBoundaries("s", { retained: [newer] });
  releaseTrace();
  await idle;
  const restarted = createSessionStore(path);
  await restarted.load();
  assert.deepEqual(restarted.get("s").pendingTurnBoundaries, [newer]);
});

test("real hooks do not replace a newer cross-process boundary with a stale sync result", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-hook-boundary-stage-"));
  const path = join(root, "sessions.json");
  const firstStore = createSessionStore(path);
  const secondStore = createSessionStore(path);
  await Promise.all([firstStore.load(), secondStore.load()]);
  const stale = { cursor: "m1", turnKey: "u1", lastAssistant: "first", count: 2 };
  const newer = { cursor: "m2", turnKey: "u1", lastAssistant: "revised", count: 3 };
  await firstStore.updateTurnBoundaries("s", { retained: [stale] });
  let releaseSync;
  let syncStarted;
  const started = new Promise((resolve) => { syncStarted = resolve; });
  const gate = new Promise((resolve) => { releaseSync = resolve; });
  const hooks = createHooks({
    logger: { warn() {} }, compose: { clear() {} }, store: firstStore,
    sync: { sync: async () => { syncStarted(); await gate; return { success: true, boundaries: [stale] }; }, flush: async () => {} },
    trace: { emit: async () => true, flush: async () => {} },
  });
  const completed = hooks.event({ event: { type: "message.updated", properties: { info: { sessionID: "s", role: "assistant", time: { completed: 1 } } } } });
  await started;
  await secondStore.updateTurnBoundaries("s", { retained: [newer] });
  releaseSync();
  await completed;
  const restarted = createSessionStore(path);
  await restarted.load();
  assert.equal(restarted.get("s").pendingTurnBoundaries[0].cursor, "m2");
  assert.equal(restarted.get("s").pendingTurnBoundaries[0].lastAssistant, "revised");
});

test("parallel plugin processes emit distinct session-error sequence IDs", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-hook-error-sequence-"));
  const path = join(root, "sessions.json");
  const firstStore = createSessionStore(path);
  const secondStore = createSessionStore(path);
  await Promise.all([firstStore.load(), secondStore.load()]);
  const objectIds = [];
  const warnings = [];
  const makeHooks = (store) => createHooks({
    logger: { warn: (message) => warnings.push(message) }, compose: {}, store,
    sync: { sync: async () => ({ success: true, boundaries: [] }), flush: async () => {} },
    trace: { emit: async (type, _session, options) => { if (type === "session_error") objectIds.push(options.objectId); return true; }, flush: async () => {} },
  });
  await Promise.all([
    makeHooks(firstStore).event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "Error" } } } }),
    makeHooks(secondStore).event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "Error" } } } }),
  ]);
  assert.deepEqual(warnings, []);
  assert.deepEqual(objectIds.sort(), ["error-1", "error-2"]);
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

test("successful pre-compact sync clears a covered deletion fallback", async () => {
  let state = { deletionFallbackMessages: [{ role: "user", content: "question" }], deletionFallbackCursor: "m1" };
  const hooks = createHooks({
    logger: { warn() {} }, trace: { emit: async () => true, flush: async () => {} }, compose: {},
    sync: { sync: async () => ({ success: true, committed: true, committedMessageIds: ["m1", "m2"], count: 2, cursor: "m2", boundaries: [{ cursor: "m2", turnKey: "m1", count: 2 }] }), flush: async () => {} },
    store: { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } },
  });
  await hooks["experimental.session.compacting"]({ sessionID: "s" });
  assert.deepEqual(state.deletionFallbackMessages, []);
  assert.equal(state.deletionFallbackCursor, null);
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
      compose: async () => ({
        systemPromptAddition: "remembered profile",
        systemPromptSuffix: "remembered archive and session",
        memoryUserMessage: "remembered evidence",
      }),
      afterTurn: async (_session, payload) => { writes.push(payload); return { ok: true }; },
      trace: async () => ({ ok: true }),
    }),
    createSessionStore: () => store,
    createTraceSpool: () => ({ enqueue: async (value) => traceEnvelopes.push(value), drain: async () => {}, flush: async () => {} }),
  });
  await hooks["chat.message"]({ sessionID: "s" }, { message: messages[0].info, parts: [{ type: "text", text: "question" }] });
  const transformed = { system: [] };
  await hooks["experimental.chat.system.transform"]({ sessionID: "s" }, transformed);
  for (const context of ["remembered profile", "remembered archive and session", "remembered evidence"]) {
    assert.equal(transformed.system[0].split(context).length - 1, 1);
  }
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
