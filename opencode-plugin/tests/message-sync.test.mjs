import test from "node:test";
import assert from "node:assert/strict";
import { createMessageSync, normalizeTranscript } from "../lib/message-sync.mjs";

const row = (id, role, text, extra = {}) => ({ info: { id, role, time: { created: Number(id.slice(1)) } }, parts: [{ type: "text", text, ...extra }] });

test("transcript normalization filters synthetic memory and advances cursor", () => {
  const result = normalizeTranscript([row("m1", "user", "old"), row("m2", "assistant", "hidden", { synthetic: true }), row("m3", "assistant", "answer")], "m1");
  assert.deepEqual(result.messages, [{ role: "assistant", content: "answer" }]);
  assert.equal(result.cursor, "m3");
});

test("OpenCode user diff summary metadata is not treated as generated content", () => {
  const user = row("m1", "user", "remember this");
  user.info.summary = { diffs: [] };
  const result = normalizeTranscript([user, row("m2", "assistant", "noted")]);
  assert.deepEqual(result.messages, [
    { role: "user", content: "remember this" },
    { role: "assistant", content: "noted" },
  ]);
});

test("assistant compaction summary remains generated content", () => {
  const summary = row("m2", "assistant", "generated summary");
  summary.info.summary = true;
  const result = normalizeTranscript([row("m1", "user", "keep"), summary]);
  assert.deepEqual(result.messages, [{ role: "user", content: "keep" }]);
  assert.equal(result.cursor, "m2");
});

test("failed after_turn retains cursor for retry", async () => {
  let cursor = null;
  const store = { get: () => ({ cursor }), set: async (_id, value) => { if (value.cursor !== undefined) cursor = value.cursor; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "hello")] }) } },
    http: { afterTurn: async () => { throw new Error("offline"); } }, store, logger: { warn() {} },
  });
  assert.equal((await sync.sync("s1")).success, false);
  assert.equal(cursor, null);
});

test("successful sync returns a stable turn cursor and assistant content", async () => {
  let cursor = null;
  const store = { get: () => ({ cursor }), set: async (_id, value) => { cursor = value.cursor; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "hello"), row("m2", "assistant", "answer")] }) } },
    http: { afterTurn: async () => ({ ok: true }) }, store,
  });
  const result = await sync.sync("s1");
  assert.deepEqual(result, { success: true, count: 2, cursor: "m2", lastAssistant: "answer", boundaries: [{ cursor: "m2", lastAssistant: "answer", count: 2, turnKey: "m1" }] });
  const duplicate = await sync.sync("s1");
  assert.equal(duplicate.count, 0);
});

test("missing cursor uses timestamp watermark and filters generated messages", () => {
  const generated = row("m3", "user", "compaction prompt");
  generated.info.synthetic = true;
  const result = normalizeTranscript([row("m2", "assistant", "new"), generated], "evicted", 1);
  assert.deepEqual(result.messages, [{ role: "assistant", content: "new" }]);
  assert.equal(result.cursor, "m3");
});

test("transcript SDK timeout fails open and retains cursor", async () => {
  const store = { get: () => ({ cursor: "m1" }), set: async () => {} };
  const sync = createMessageSync({ client: { session: { messages: () => new Promise(() => {}) } }, http: {}, store, timeoutMs: 20, logger: { warn() {} } });
  const started = Date.now();
  const result = await sync.sync("s");
  assert.equal(result.success, false);
  assert.ok(Date.now() - started < 200);
});

test("failed final write is durable and replay advances its cursor", async () => {
  let state = { cursor: null };
  let fail = true;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "keep me")] }) } },
    http: { afterTurn: async () => { if (fail) throw new Error("offline"); return { ok: true }; } }, store, logger: { warn() {} },
  });
  assert.equal((await sync.sync("s")).success, false);
  assert.deepEqual(state.pendingMessages, [{ role: "user", content: "keep me" }]);
  fail = false;
  assert.equal((await sync.replayPending("s")).success, true);
  assert.equal(state.cursor, "m1");
  assert.deepEqual(state.pendingMessages, []);
});

test("startup replay and live sync share the same per-session lock", async () => {
  let state = { cursor: null, pendingMessages: [{ role: "user", content: "pending" }], pendingCursor: "m1", pendingCursorCreatedAt: 1 };
  let release;
  let sends = 0;
  const gate = new Promise((resolve) => { release = resolve; });
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [] }) } },
    http: { afterTurn: async () => { sends += 1; await gate; } }, store,
  });
  const replay = sync.replayPending("s");
  const live = sync.sync("s");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(sends, 1);
  release();
  await Promise.all([replay, live]);
  assert.equal(sends, 1);
});

test("transcript pagination recovers a delta larger than one page", async () => {
  const all = Array.from({ length: 7 }, (_, index) => row(`m${index + 1}`, index % 2 ? "assistant" : "user", `text-${index + 1}`));
  let state = { cursor: "m1", cursorCreatedAt: 1 };
  let written;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const client = { session: { messages: async ({ limit, before }) => {
    const end = before ? all.findIndex((message) => message.info.id === before) : all.length;
    return { data: all.slice(Math.max(0, end - limit), end) };
  } } };
  const sync = createMessageSync({ client, http: { afterTurn: async (_session, messages) => { written = messages; } }, store, limit: 2 });
  const result = await sync.sync("s");
  assert.equal(result.success, true);
  assert.equal(written.length, 6);
  assert.equal(state.cursor, "m7");
});

test("one recovered delta preserves every user turn boundary", async () => {
  let state = { cursor: null };
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const messages = [row("m1", "user", "u1"), row("m2", "assistant", "a1"), row("m3", "user", "u2"), row("m4", "assistant", "a2")];
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: messages }) } },
    http: { afterTurn: async () => ({ ok: true }) }, store,
  });
  const result = await sync.sync("s");
  assert.deepEqual(result.boundaries.map((boundary) => [boundary.turnKey, boundary.cursor, boundary.lastAssistant]), [
    ["m1", "m2", "a1"],
    ["m3", "m4", "a2"],
  ]);
});

test("permanently rejected payload does not poison later writes", async () => {
  let state = { cursor: null };
  let rows = [row("m1", "user", "bad payload")];
  let calls = 0;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: rows }) } },
    http: { afterTurn: async () => { calls += 1; if (calls === 1) { const error = new Error("bad request"); error.retryable = false; error.status = 400; throw error; } } },
    store, logger: { warn() {} },
  });
  assert.equal((await sync.sync("s")).success, true);
  assert.equal(state.rejectedWriteCount, 1);
  rows = [...rows, row("m2", "assistant", "good")];
  assert.equal((await sync.sync("s")).success, true);
  assert.equal(calls, 2);
  assert.equal(state.cursor, "m2");
});

test("pending payload that becomes permanently rejected does not block later writes", async () => {
  let state = { cursor: null };
  let rows = [row("m1", "user", "pending payload")];
  let calls = 0;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: rows }) } },
    http: { afterTurn: async () => {
      calls += 1;
      const error = new Error(calls === 1 ? "offline" : "bad request");
      error.retryable = calls === 1;
      error.status = calls === 1 ? 503 : 400;
      if (calls <= 2) throw error;
    } },
    store, logger: { warn() {} },
  });
  assert.equal((await sync.sync("s")).success, false);
  assert.equal(state.pendingMessages.length, 1);
  rows = [...rows, row("m2", "assistant", "later answer")];
  assert.equal((await sync.sync("s")).success, true);
  assert.equal(calls, 3);
  assert.equal(state.cursor, "m2");
  assert.deepEqual(state.pendingMessages, []);
  assert.equal(state.rejectedWriteCount, 1);
});

test("legacy OpenCode SDK path/query signature remains supported", async () => {
  let state = { cursor: null };
  const calls = [];
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async (options) => {
      calls.push(options);
      if (!options.path?.id) return { error: { message: "missing path id" } };
      return { data: [row("m1", "user", "legacy")] };
    } } },
    http: { afterTurn: async () => ({ ok: true }) }, store,
  });
  assert.equal((await sync.sync("legacy-session")).success, true);
  assert.equal(calls[1].path.id, "legacy-session");
  assert.equal(state.cursor, "m1");
});

test("legacy cursor gap drops only the unavailable prefix and remains live", async () => {
  let state = { cursor: "m1", cursorCreatedAt: 1 };
  let rows = [row("m6", "assistant", "six"), row("m7", "user", "seven")];
  const writes = [];
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async (options) => options.path ? { data: rows } : { error: {} } } },
    http: { afterTurn: async (_session, messages) => writes.push(messages) }, store, limit: 2, logger: { warn() {} },
  });
  assert.equal((await sync.sync("s")).success, true);
  assert.equal(state.cursor, "m7");
  rows = [row("m7", "user", "seven"), row("m8", "assistant", "eight")];
  assert.equal((await sync.sync("s")).success, true);
  assert.deepEqual(writes.at(-1), [{ role: "assistant", content: "eight" }]);
  assert.equal(state.transcriptGapCount, 1);
});

test("transcript normalization bounds parts and total records incrementally", () => {
  const huge = { info: { id: "m1", role: "user", time: { created: 1 } }, parts: Array.from({ length: 100 }, () => ({ type: "text", text: "x".repeat(100000) })) };
  const many = Array.from({ length: 20 }, (_, index) => ({ info: { id: `n${String(index).padStart(2, "0")}`, role: "assistant", time: { created: index + 2 } }, parts: [{ type: "text", text: "y".repeat(100000) }] }));
  const result = normalizeTranscript([huge, ...many]);
  assert.equal(result.records[0].content.length, 100000);
  assert.ok(result.records.reduce((sum, item) => sum + item.content.length, 0) <= 1000000);
  assert.ok(result.records.length < 21);
  assert.notEqual(result.cursor, many.at(-1).info.id);
});
