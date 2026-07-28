import { createHash, randomUUID } from "node:crypto";
import { readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { acquireFileLock } from "./file-lock.mjs";

const emptyState = () => ({ version: 1, sessions: {} });
const isPlainObject = (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
class UnsupportedStateSchema extends Error {}
class InvalidStateData extends Error {}
const boundaryKey = (boundary) => boundary?.turnKey || boundary?.cursor;
const sameBoundary = (left, right) => boundaryKey(left) === boundaryKey(right)
  && left?.cursor === right?.cursor
  && (left?.terminalState || "idle") === (right?.terminalState || "idle")
  && (left?.lastAssistant || "") === (right?.lastAssistant || "")
  && Number(left?.count || 0) === Number(right?.count || 0);

export function mergeTurnBoundaries(current = [], { delivered = [], retained = [], expected = [], preferExisting = false, errorTurnKey } = {}) {
  const remaining = current.filter((boundary) => !delivered.some((candidate) => sameBoundary(boundary, candidate)));
  const byTurn = new Map(remaining.filter((boundary) => boundary?.cursor).map((boundary) => [boundaryKey(boundary), boundary]));
  const expectedByTurn = new Map(expected.filter((boundary) => boundary?.cursor).map((boundary) => [boundaryKey(boundary), boundary]));
  for (const boundary of retained) {
    if (!boundary?.cursor) continue;
    const key = boundaryKey(boundary);
    const existing = byTurn.get(key);
    if (existing?.terminalState === "error" && boundary.terminalState !== "error") continue;
    if (preferExisting && existing && !sameBoundary(existing, boundary)
      && !sameBoundary(existing, expectedByTurn.get(key))) continue;
    byTurn.set(key, boundary);
  }
  if (errorTurnKey && byTurn.has(errorTurnKey)) {
    byTurn.set(errorTurnKey, { ...byTurn.get(errorTurnKey), terminalState: "error" });
  }
  return [...byTurn.values()];
}

function validSession(value) {
  if (!isPlainObject(value)) return false;
  for (const key of ["pendingMessages", "pendingBoundaries", "pendingTurnBoundaries"]) if (value[key] !== undefined && !Array.isArray(value[key])) return false;
  if (value.pendingMessages?.some((message) => !isPlainObject(message) || !["user", "assistant"].includes(message.role) || typeof message.content !== "string" || message.content.length > 100000)) return false;
  if (value.pendingMessages?.reduce((sum, message) => sum + message.content.length, 0) > 1000000) return false;
  for (const key of ["pendingBoundaries", "pendingTurnBoundaries"]) {
    if (value[key]?.some((boundary) => !isPlainObject(boundary) || (boundary.cursor !== undefined && typeof boundary.cursor !== "string") || (boundary.turnKey !== undefined && typeof boundary.turnKey !== "string") || (boundary.count !== undefined && (!Number.isSafeInteger(boundary.count) || boundary.count < 0)))) return false;
  }
  return true;
}

export function createSessionStore(path, {
  limit = 128, ttlSeconds = 604800, maxBytes = 16777216, now = () => Date.now(), logger = console,
  readFileImpl = readFile, statImpl = stat,
} = {}) {
  let data = emptyState();
  let saving = Promise.resolve();
  const dirty = new Map();

  const prune = (state) => {
    const entries = Object.entries(state.sessions).sort((a, b) => Number(a[1].stateUpdatedAt || 0) - Number(b[1].stateUpdatedAt || 0) || a[0].localeCompare(b[0]));
    const expiredBefore = now() - ttlSeconds * 1000;
    const removed = [];
    for (const [id, value] of entries) {
      if (Number(value.stateUpdatedAt || 0) < expiredBefore) { delete state.sessions[id]; removed.push([id, Number(value.stateUpdatedAt || 0)]); }
    }
    const retained = Object.entries(state.sessions).sort((a, b) => Number(a[1].stateUpdatedAt || 0) - Number(b[1].stateUpdatedAt || 0) || a[0].localeCompare(b[0]));
    while (retained.length > limit) { const [id, value] = retained.shift(); delete state.sessions[id]; removed.push([id, Number(value.stateUpdatedAt || 0)]); }
    while (retained.length && Buffer.byteLength(JSON.stringify(state)) > maxBytes) { const [id, value] = retained.shift(); delete state.sessions[id]; removed.push([id, Number(value.stateUpdatedAt || 0)]); }
    if (removed.length) logger.warn?.(`oGMemory session state pruned ${removed.length} entries (limit=${limit}, ttl=${ttlSeconds}s, maxBytes=${maxBytes})`);
    return removed;
  };

  async function readState() {
    const quarantine = `${path}.unsupported`;
    try {
      if ((await statImpl(quarantine)).mtimeMs < now() - ttlSeconds * 1000) {
        await rm(quarantine, { force: true });
        logger.warn?.("oGMemory removed expired unsupported session-state quarantine");
      }
    } catch (error) { if (error?.code !== "ENOENT") throw error; }
    try {
      if ((await statImpl(path)).size > maxBytes) throw new InvalidStateData("session-state file exceeds configured size");
      let parsed;
      try { parsed = JSON.parse(await readFileImpl(path, "utf8")); }
      catch (error) { if (error instanceof SyntaxError) throw new InvalidStateData("invalid session-state JSON"); throw error; }
      if (parsed?.version !== 1) throw new UnsupportedStateSchema("unsupported session-state schema");
      if (!parsed.sessions || Array.isArray(parsed.sessions) || typeof parsed.sessions !== "object") throw new InvalidStateData("invalid session-state schema");
      if (Object.values(parsed.sessions).some((value) => !validSession(value))) throw new InvalidStateData("invalid session-state entry");
      return parsed;
    } catch (error) {
      if (error?.code === "ENOENT") return emptyState();
      if (error instanceof UnsupportedStateSchema) {
        await rm(quarantine, { force: true }).catch(() => {});
        await rename(path, quarantine).catch(() => { throw error; });
        logger.warn?.("oGMemory preserved unsupported session-state schema and started with empty state");
        return emptyState();
      }
      if (!(error instanceof InvalidStateData)) throw error;
      await rm(path, { force: true }).catch(() => {});
      logger.warn?.("oGMemory discarded unreadable session state without logging its content");
      return emptyState();
    }
  }

  async function cleanupTemps() {
    const prefix = `${basename(path)}.`;
    const names = await readdir(dirname(path)).catch((error) => error?.code === "ENOENT" ? [] : Promise.reject(error));
    const temporary = names.filter((name) => name.startsWith(prefix) && name.endsWith(".tmp"));
    await Promise.all(temporary.map((name) => rm(join(dirname(path), name), { force: true }).catch(() => {})));
    if (temporary.length) logger.warn?.(`oGMemory session state removed ${temporary.length} orphan temporary file(s)`);
  }

  async function withLock(operation) {
    const lock = await acquireFileLock({ directory: dirname(path), name: `${basename(path)}.write`, waitMs: 5000 });
    if (!lock) throw new Error("Timed out acquiring oGMemory session-state lock");
    try { await cleanupTemps(); return await operation(); }
    finally { await lock.release(); }
  }

  async function writeState(state) {
    const temporary = `${path}.${randomUUID()}.tmp`;
    await writeFile(temporary, JSON.stringify(state), { mode: 0o600 });
    await rename(temporary, path);
  }

  const enqueue = (operation) => {
    const current = saving.catch(() => {}).then(operation);
    saving = current.catch(() => {});
    return current;
  };

  const flushDirty = () => enqueue(() => withLock(async () => {
    const state = await readState();
    const changes = [...dirty.entries()];
    for (const [id, value] of changes) {
      if (value === null) delete state.sessions[id];
      else state.sessions[id] = { ...state.sessions[id], ...value };
    }
    const removed = prune(state);
    await writeState(state);
    for (const [id, value] of changes) if (dirty.get(id) === value) dirty.delete(id);
    for (const [id, removedAt] of removed) if (!dirty.has(id) && Number(data.sessions[id]?.stateUpdatedAt || 0) <= removedAt) delete data.sessions[id];
  }));

  async function runSessionExclusive(id, operation) {
    await flushDirty();
    const sessionKey = createHash("sha256").update(String(id)).digest("hex");
    const lock = await acquireFileLock({ directory: dirname(path), name: `${basename(path)}.session.${sessionKey}`, waitMs: 10000, staleMs: 30000 });
    if (!lock) throw new Error("Timed out acquiring oGMemory session operation lock");
    try {
      await enqueue(() => withLock(async () => { data = await readState(); }));
      return await operation();
    } finally { await lock.release(); }
  }

  return {
    async load() {
      return enqueue(() => withLock(async () => {
        data = await readState();
        if (prune(data).length) await writeState(data);
        return data;
      }));
    },
    get(id) { return data.sessions[id] || { cursor: null }; },
    list() { return Object.entries(data.sessions); },
    async set(id, value) {
      const patch = { ...value, stateUpdatedAt: now() };
      const next = { ...data.sessions[id], ...patch };
      data.sessions[id] = next;
      dirty.set(id, { ...(dirty.get(id) || {}), ...patch });
      await flushDirty();
    },
    async remove(id) { delete data.sessions[id]; dirty.set(id, null); await flushDirty(); },
    runExclusive: runSessionExclusive,
    async nextErrorSequence(id) {
      return runSessionExclusive(id, async () => {
        const errorSequence = Number(data.sessions[id]?.errorSequence || 0) + 1;
        const patch = { errorSequence, stateUpdatedAt: now() };
        data.sessions[id] = { ...data.sessions[id], ...patch };
        dirty.set(id, { ...(dirty.get(id) || {}), ...patch });
        await flushDirty();
        return errorSequence;
      });
    },
    async updateDeletionFallback(id, { expectedCursor, messages, cursor } = {}) {
      return runSessionExclusive(id, async () => {
        if (expectedCursor !== undefined && data.sessions[id]?.deletionFallbackCursor !== expectedCursor) return false;
        const patch = { deletionFallbackMessages: messages, deletionFallbackCursor: cursor, stateUpdatedAt: now() };
        data.sessions[id] = { ...data.sessions[id], ...patch };
        dirty.set(id, { ...(dirty.get(id) || {}), ...patch });
        await flushDirty();
        return true;
      });
    },
    async updateTurnBoundaries(id, { delivered = [], retained = [], expected = [], preferExisting = false, errorTurnKey } = {}) {
      return runSessionExclusive(id, async () => {
        const current = Array.isArray(data.sessions[id]?.pendingTurnBoundaries) ? [...data.sessions[id].pendingTurnBoundaries] : [];
        const pendingTurnBoundaries = mergeTurnBoundaries(current, { delivered, retained, expected, preferExisting, errorTurnKey });
        const patch = { pendingTurnBoundaries, stateUpdatedAt: now() };
        data.sessions[id] = { ...data.sessions[id], ...patch };
        dirty.set(id, { ...(dirty.get(id) || {}), ...patch });
        await flushDirty();
        return pendingTurnBoundaries;
      });
    },
    async clear() {
      data = emptyState();
      dirty.clear();
      await enqueue(() => withLock(() => rm(path, { force: true })));
    },
  };
}
