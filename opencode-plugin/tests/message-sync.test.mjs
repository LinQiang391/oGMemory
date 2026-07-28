import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { link, mkdtemp, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createMessageSync, normalizeTranscript } from "../lib/message-sync.mjs";
import { createSessionStore } from "../lib/session-state.mjs";
import { createTraceAdapter } from "../lib/trace-adapter.mjs";
import { createTraceSpool } from "../lib/trace-spool.mjs";
import { HttpError } from "../lib/http-client.mjs";

const row = (id, role, text, extra = {}) => ({ info: { id, role, time: { created: Number(id.slice(1)) } }, parts: [{ type: "text", text, ...extra }] });

test("transcript normalization filters synthetic memory and advances cursor", () => {
  const result = normalizeTranscript([row("m1", "user", "old"), row("m2", "assistant", "hidden", { synthetic: true }), row("m3", "assistant", "answer")], "m1");
  assert.deepEqual(result.messages, [{ role: "assistant", content: "answer" }]);
  assert.equal(result.cursor, "m3");
});

test("transcript normalization preserves user text containing the memory marker", () => {
  const content = "explain the literal marker [oGMemory retrieved context] in this document";
  const result = normalizeTranscript([row("m1", "user", content)]);
  assert.deepEqual(result.messages, [{ role: "user", content }]);
  assert.equal(result.cursor, "m1");
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
  assert.deepEqual(result, { success: true, committed: true, committedMessageIds: ["m1", "m2"], resolvedMessageIds: ["m1", "m2"], count: 2, cursor: "m2", lastAssistant: "answer", boundaries: [{ cursor: "m2", lastAssistant: "answer", count: 2, turnKey: "m1" }] });
  const duplicate = await sync.sync("s1");
  assert.equal(duplicate.count, 0);
});

test("transcript recovery emits message traces through one batch callback", async () => {
  let cursor = null;
  const batches = [];
  const store = { get: () => ({ cursor }), set: async (_id, value) => { if (value.cursor !== undefined) cursor = value.cursor; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "hello"), row("m2", "assistant", "answer")] }) } },
    http: { afterTurn: async () => ({ ok: true }) }, store,
    onMessage: async () => { throw new Error("per-message callback should not be used when batching is available"); },
    onMessages: async (sessionId, records) => { batches.push({ sessionId, records }); },
  });
  await sync.sync("s1");
  assert.deepEqual(batches, [{ sessionId: "s1", records: [
    { id: "m1", role: "user", content: "hello" },
    { id: "m2", role: "assistant", content: "answer" },
  ] }]);
});

test("failed message-trace batch keeps the transcript cursor and skips write-back", async () => {
  let cursor = null;
  let writes = 0;
  const store = { get: () => ({ cursor }), set: async (_id, value) => { if (value.cursor !== undefined) cursor = value.cursor; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "hello")] }) } },
    http: { afterTurn: async () => { writes += 1; return { ok: true }; } }, store,
    onMessages: async () => false, logger: { warn() {} },
  });
  const result = await sync.sync("s1");
  assert.equal(result.success, false);
  assert.equal(cursor, null);
  assert.equal(writes, 0);
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
  assert.equal(result.transcriptUnavailable, true);
  assert.equal(result.transcriptGone, false);
  assert.ok(Date.now() - started < 200);
});

test("two definitive transcript not-found responses are classified as gone", async () => {
  const missing = new Error("missing");
  missing.status = 404;
  const store = { get: () => ({ cursor: null }), set: async () => {} };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw missing; } } }, http: {}, store, logger: { warn() {} } });
  const result = await sync.sync("gone");
  assert.equal(result.success, false);
  assert.equal(result.transcriptUnavailable, true);
  assert.equal(result.transcriptGone, true);
});

test("mixed not-found and transient SDK probes keep the transcript recoverable", async () => {
  let attempts = 0;
  const store = { get: () => ({ cursor: null }), set: async () => {} };
  const sync = createMessageSync({
    client: { session: { messages: async () => {
      attempts += 1;
      if (attempts === 1) throw Object.assign(new Error("missing flat signature"), { status: 404 });
      throw Object.assign(new Error("legacy temporarily unavailable"), { status: 503 });
    } } },
    http: {}, store, logger: { warn() {} },
  });
  const result = await sync.sync("recoverable");
  assert.equal(result.transcriptUnavailable, true);
  assert.equal(result.transcriptGone, false);
});

test("two resolved SDK not-found results are classified as gone", async () => {
  const store = { get: () => ({ cursor: null }), set: async () => {} };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ error: { message: "missing" }, response: { status: 404 } }) } },
    http: {}, store, logger: { warn() {} },
  });
  const result = await sync.sync("resolved-gone");
  assert.equal(result.transcriptUnavailable, true);
  assert.equal(result.transcriptGone, true);
});

test("mixed resolved not-found and transient results remain recoverable", async () => {
  let attempts = 0;
  const store = { get: () => ({ cursor: null }), set: async () => {} };
  const sync = createMessageSync({
    client: { session: { messages: async () => {
      attempts += 1;
      return attempts === 1
        ? { error: { message: "missing" }, response: { status: 404 } }
        : { error: { message: "unavailable" }, response: { status: 503 } };
    } } },
    http: {}, store, logger: { warn() {} },
  });
  const result = await sync.sync("resolved-recoverable");
  assert.equal(result.transcriptUnavailable, true);
  assert.equal(result.transcriptGone, false);
});

test("real trace batch partial failure blocks write-back until every message is durable", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmem-sync-trace-batch-"));
  let linkCalls = 0;
  let interrupt = true;
  const spool = createTraceSpool({
    directory, limit: 10, ttlSeconds: 60,
    linkFile: async (...args) => {
      linkCalls += 1;
      if (interrupt && linkCalls === 4) { interrupt = false; throw new Error("partial trace batch"); }
      return link(...args);
    },
    send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} },
  });
  const trace = createTraceAdapter({ config: { traceCaptureContent: false, accountId: "a", userId: "u", agentId: "opencode" }, spool, logger: { warn() {} } });
  let cursor = null;
  let writes = 0;
  const store = { get: () => ({ cursor }), set: async (_id, value) => { if (value.cursor !== undefined) cursor = value.cursor; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "hello"), row("m2", "assistant", "answer")] }) } },
    http: { afterTurn: async () => { writes += 1; return { ok: true }; } }, store, logger: { warn() {} },
    onMessages: (sessionId, records) => trace.emitMany(records.map((record) => ({
      type: `${record.role}_message`, sessionId, options: { objectId: record.id, content: record.content },
    }))),
  });
  assert.equal((await sync.sync("s")).success, false);
  assert.equal(cursor, null);
  assert.equal(writes, 0);
  assert.equal((await sync.sync("s")).success, true);
  assert.equal(cursor, "m2");
  assert.equal(writes, 1);
  await spool.flush();
  spool.close();
  assert.equal((await readdir(directory)).filter((name) => name.endsWith(".json")).length, 2);
});

test("transcript timeout keeps an otherwise idle process alive through fail-open", () => {
  const moduleUrl = new URL("../lib/message-sync.mjs", import.meta.url).href;
  const script = `
    import { createMessageSync } from ${JSON.stringify(moduleUrl)};
    const store = { get: () => ({ cursor: "m1" }), set: async () => {} };
    const sync = createMessageSync({
      client: { session: { messages: () => new Promise(() => {}) } },
      http: {}, store, timeoutMs: 20, logger: { warn() {} },
    });
    const result = await sync.sync("s");
    if (result.success || result.cursor !== "m1") process.exit(2);
    console.log("transcript-timeout-complete");
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "--eval", script], { encoding: "utf8", timeout: 1000 });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /transcript-timeout-complete/);
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

test("two processes serialize the same session operation before reading its cursor", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-session-operation-"));
  try {
    const path = join(directory, "sessions.json");
    const firstStore = createSessionStore(path, { logger: { warn() {} } });
    const secondStore = createSessionStore(path, { logger: { warn() {} } });
    await Promise.all([firstStore.load(), secondStore.load()]);
    let release;
    let started;
    const firstSendStarted = new Promise((resolve) => { started = resolve; });
    const gate = new Promise((resolve) => { release = resolve; });
    let sends = 0;
    const http = { afterTurn: async () => { sends += 1; if (sends === 1) { started(); await gate; } } };
    const client = { session: { messages: async () => ({ data: [row("m1", "user", "question"), row("m2", "assistant", "answer")] }) } };
    const firstSync = createMessageSync({ client, http, store: firstStore, logger: { warn() {} } });
    const secondSync = createMessageSync({ client, http, store: secondStore, logger: { warn() {} } });
    const first = firstSync.sync("s");
    await firstSendStarted;
    const second = secondSync.sync("s");
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(sends, 1);
    release();
    await Promise.all([first, second]);
    assert.equal(sends, 1);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("successful pending replay remains visible when the following transcript fetch fails", async () => {
  let state = {
    cursor: null,
    pendingMessages: [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }],
    pendingCursor: "m2",
    pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }],
  };
  const writes = [];
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => { throw new Error("transcript unavailable"); } } },
    http: { afterTurn: async (_session, messages) => writes.push(messages) }, store, logger: { warn() {} },
  });
  const result = await sync.sync("s");
  assert.equal(result.success, false);
  assert.equal(result.committed, true);
  assert.deepEqual(result.committedMessageIds, ["m1", "m2"]);
  assert.equal(result.cursor, "m2");
  assert.deepEqual(result.boundaries, [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }]);
  assert.deepEqual(writes, [[{ role: "user", content: "question" }, { role: "assistant", content: "answer" }]]);
  assert.deepEqual(state.pendingMessages, []);
});

test("pending commit remains explicit when a later transcript delta write fails", async () => {
  let state = {
    cursor: null,
    pendingMessages: [{ role: "user", content: "old question" }, { role: "assistant", content: "old answer" }],
    pendingCursor: "m2",
    pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "old answer", count: 2 }],
  };
  let writes = 0;
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "old question"), row("m2", "assistant", "old answer"), row("m3", "user", "new question")] }) } },
    http: { afterTurn: async () => { writes += 1; if (writes === 2) throw new Error("offline"); } }, store, logger: { warn() {} },
  });
  const result = await sync.sync("s");
  assert.equal(result.success, false);
  assert.equal(result.committed, true);
  assert.deepEqual(result.committedMessageIds, ["m1", "m2"]);
  assert.deepEqual(result.boundaries, [{ cursor: "m2", turnKey: "m1", lastAssistant: "old answer", count: 2 }]);
  assert.deepEqual(state.pendingMessages, [{ role: "user", content: "new question" }]);
  assert.equal(state.pendingCursor, "m3");
});

test("pending HTTP acknowledgement remains explicit when state persistence fails", async () => {
  let state = {
    cursor: null, pendingMessages: [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }], pendingCursor: "m2",
    pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }],
  };
  const store = {
    get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; throw new Error("disk flush failed"); },
  };
  const sync = createMessageSync({ client: { session: { messages: async () => ({ data: [] }) } }, http: { afterTurn: async () => ({ ok: true }) }, store, logger: { warn() {} } });
  const result = await sync.sync("s");
  assert.equal(result.success, false);
  assert.equal(result.committed, true);
  assert.deepEqual(result.committedMessageIds, ["m1", "m2"]);
  assert.deepEqual(result.boundaries, [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }]);
});

test("pending acknowledgement cannot downgrade a persisted error boundary", async () => {
  let state = {
    cursor: null, pendingMessages: [{ role: "user", content: "question" }, { role: "assistant", content: "answer" }], pendingCursor: "m2",
    pendingBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }],
    pendingTurnBoundaries: [{ cursor: "m2", turnKey: "m1", lastAssistant: "failed", count: 2, terminalState: "error" }],
  };
  const store = { get: () => state, set: async (_id, value) => { state = { ...state, ...value }; } };
  const sync = createMessageSync({ client: { session: { messages: async () => { throw new Error("deleted"); } } }, http: { afterTurn: async () => ({ ok: true }) }, store, logger: { warn() {} } });
  await sync.sync("s");
  assert.equal(state.pendingTurnBoundaries[0].terminalState, "error");
  assert.equal(state.pendingTurnBoundaries[0].lastAssistant, "failed");
});

test("live HTTP acknowledgement retains its boundary when cursor persistence fails", async () => {
  let state = { cursor: null };
  const store = {
    get: () => state,
    set: async (_id, value) => { state = { ...state, ...value }; throw new Error("disk flush failed"); },
  };
  const sync = createMessageSync({
    client: { session: { messages: async () => ({ data: [row("m1", "user", "question"), row("m2", "assistant", "answer")] }) } },
    http: { afterTurn: async () => ({ ok: true }) }, store, logger: { warn() {} },
  });
  const result = await sync.sync("s");
  assert.equal(result.success, false);
  assert.equal(result.committed, true);
  assert.deepEqual(result.committedMessageIds, ["m1", "m2"]);
  assert.deepEqual(result.boundaries, [{ cursor: "m2", turnKey: "m1", lastAssistant: "answer", count: 2 }]);
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
