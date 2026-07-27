import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, readdir, rm, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createSessionStore } from "../lib/session-state.mjs";

test("session-state save queue recovers after a transient filesystem failure", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-"));
  const parent = join(root, "runtime");
  const path = join(parent, "sessions.json");
  await writeFile(parent, "temporarily a file");
  const store = createSessionStore(path);
  await assert.rejects(store.set("s", { cursor: "m1" }));
  await rm(parent);
  await mkdir(parent);
  await store.set("s", { cursor: "m2" });
  const saved = JSON.parse(await readFile(path, "utf8"));
  assert.equal(saved.sessions.s.cursor, "m2");
});

test("session-state prunes expired, oldest, and oversized entries", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-bounds-"));
  const path = join(root, "sessions.json");
  let clock = 100000;
  const store = createSessionStore(path, { limit: 2, ttlSeconds: 10, maxBytes: 700, now: () => clock, logger: { warn() {} } });
  await store.set("expired", { pendingMessages: [{ role: "user", content: "old" }] });
  clock += 11000;
  await store.set("one", { pendingMessages: [{ role: "user", content: "1".repeat(100) }] });
  clock += 1;
  await store.set("two", { pendingMessages: [{ role: "user", content: "2".repeat(100) }] });
  clock += 1;
  await store.set("three", { pendingMessages: [{ role: "user", content: "3".repeat(100) }] });
  assert.deepEqual(store.list().map(([id]) => id).sort(), ["three", "two"]);

  await store.set("large", { pendingMessages: [{ role: "user", content: "x".repeat(500) }] });
  const serialized = await readFile(path, "utf8");
  assert.ok(Buffer.byteLength(serialized) <= 700);
  assert.ok(store.list().length <= 2);
});

test("session-state persists TTL pruning during startup load", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-load-"));
  const path = join(root, "sessions.json");
  await writeFile(path, JSON.stringify({ version: 1, sessions: { old: { stateUpdatedAt: 1, pendingMessages: [{ role: "user", content: "sensitive" }] } } }));
  const store = createSessionStore(path, { ttlSeconds: 60, now: () => 120000, logger: { warn() {} } });
  await store.load();
  assert.deepEqual(store.list(), []);
  assert.deepEqual(JSON.parse(await readFile(path, "utf8")).sessions, {});
});

test("session-state discards corruption and survives a disk restart", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-restart-"));
  const corruptPath = join(root, "corrupt.json");
  await writeFile(corruptPath, "not-json");
  const corrupt = createSessionStore(corruptPath, { logger: { warn() {} } });
  await corrupt.load();
  assert.deepEqual(corrupt.list(), []);
  assert.equal((await readdir(root)).some((name) => name.startsWith("corrupt.json")), false);

  const path = join(root, "sessions.json");
  const first = createSessionStore(path);
  await first.set("s", { cursor: "m1", pendingMessages: [{ role: "assistant", content: "final" }], pendingBoundaries: [{ cursor: "m1", turnKey: "u1" }] });
  const second = createSessionStore(path);
  await second.load();
  assert.equal(second.get("s").cursor, "m1");
  assert.equal(second.get("s").pendingMessages[0].content, "final");
  assert.equal(second.get("s").pendingBoundaries[0].turnKey, "u1");
});

test("two session-store instances merge interleaved process writes", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-multiprocess-"));
  const path = join(root, "sessions.json");
  const first = createSessionStore(path);
  const second = createSessionStore(path);
  await Promise.all([first.load(), second.load()]);
  await first.set("session-a", { cursor: "a1" });
  await second.set("session-b", { cursor: "b1" });
  const restarted = createSessionStore(path);
  await restarted.load();
  assert.equal(restarted.get("session-a").cursor, "a1");
  assert.equal(restarted.get("session-b").cursor, "b1");
});

test("two stores merge same-session field patches without cursor rollback", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-same-session-"));
  const path = join(root, "sessions.json");
  const seed = createSessionStore(path);
  await seed.set("s", { cursor: "m0" });
  const first = createSessionStore(path);
  const second = createSessionStore(path);
  await Promise.all([first.load(), second.load()]);
  await first.set("s", { cursor: "m1" });
  await second.set("s", { errorSequence: 1 });
  let restarted = createSessionStore(path);
  await restarted.load();
  assert.equal(restarted.get("s").cursor, "m1");
  assert.equal(restarted.get("s").errorSequence, 1);
  await second.set("s", { cursor: "m2" });
  restarted = createSessionStore(path);
  await restarted.load();
  assert.equal(restarted.get("s").cursor, "m2");
});

test("valid JSON with an invalid nested session is discarded fail-open", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-invalid-entry-"));
  const path = join(root, "sessions.json");
  await writeFile(path, JSON.stringify({ version: 1, sessions: { broken: null } }));
  const warnings = [];
  const store = createSessionStore(path, { logger: { warn: (message) => warnings.push(message) } });
  await store.load();
  assert.deepEqual(store.list(), []);
  assert.equal((await readdir(root)).includes("sessions.json"), false);
  assert.doesNotMatch(warnings.join("\n"), /broken|null/);
});

test("future session-state schema is quarantined instead of deleted", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-future-"));
  const path = join(root, "sessions.json");
  const future = JSON.stringify({ version: 2, sessions: { pending: { futureField: "preserve-me" } } });
  await writeFile(path, future);
  const warnings = [];
  const store = createSessionStore(path, { logger: { warn: (message) => warnings.push(message) } });
  await store.load();
  assert.deepEqual(store.list(), []);
  const names = await readdir(root);
  const quarantine = names.find((name) => name === "sessions.json.unsupported");
  assert.ok(quarantine);
  assert.equal(await readFile(join(root, quarantine), "utf8"), future);
  assert.match(warnings[0], /preserved unsupported/);

  const newer = JSON.stringify({ version: 3, sessions: { pending: { futureField: "newest" } } });
  await writeFile(path, newer);
  const restarted = createSessionStore(path, { logger: { warn() {} } });
  await restarted.load();
  assert.equal((await readdir(root)).filter((name) => name.startsWith("sessions.json.unsupported")).length, 1);
  assert.equal(await readFile(join(root, "sessions.json.unsupported"), "utf8"), newer);
});

test("expired future-schema quarantine follows the session-state TTL", async () => {
  const root = await mkdtemp(join(tmpdir(), "ogmem-state-future-expired-"));
  const path = join(root, "sessions.json");
  const quarantine = `${path}.unsupported`;
  await writeFile(quarantine, JSON.stringify({ version: 2, sessions: { private: { pending: "content" } } }));
  const old = new Date(Date.now() - 120000);
  await utimes(quarantine, old, old);
  const warnings = [];
  const store = createSessionStore(path, { ttlSeconds: 60, logger: { warn: (message) => warnings.push(message) } });
  await store.load();
  assert.equal((await readdir(root)).includes("sessions.json.unsupported"), false);
  assert.match(warnings[0], /removed expired unsupported/);
  assert.doesNotMatch(warnings.join("\n"), /private|content/);
});
