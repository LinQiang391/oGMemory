import { createHash, randomUUID } from "node:crypto";
import { link, mkdir, open, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { acquireFileLock } from "./file-lock.mjs";
import { safeErrorMessage } from "./redaction.mjs";

const STALE_TEMP_MS = 5 * 60 * 1000;
const MAX_SPOOL_ENTRY_BYTES = 64 * 1024;
const SEQUENCE_WIDTH = 24;
const MAX_SEQUENCE = (10n ** BigInt(SEQUENCE_WIDTH)) - 1n;

export function createTraceSpool({ directory, limit, ttlSeconds, send, logger = console, random = Math.random, linkFile = link, sequenceRename = rename }) {
  let draining = null;
  let drainRequested = false;
  let failures = 0;
  let retryAfter = 0;
  let retryTimer = null;
  let writeQueue = Promise.resolve();
  let knownIds = null;
  let closed = false;

  const keyFromId = (id) => createHash("sha256").update(String(id)).digest("hex");
  const keyFromName = (name) => name.match(/(?:^|_)([0-9a-f]{64})\.json(?:\.sending\.\d+\.[0-9a-f-]{36})?$/i)?.[1] || name.slice(0, -5);
  const idClaimPath = (key) => join(directory, `${key}.claim`);
  const sequenceOf = (name) => name.match(new RegExp(`^(\\d{${SEQUENCE_WIDTH}})_`))?.[1] || null;
  const compareEntries = (a, b) => {
    const aSequence = sequenceOf(a.name);
    const bSequence = sequenceOf(b.name);
    if (aSequence && bSequence) return aSequence.localeCompare(bSequence);
    if (aSequence) return 1;
    if (bSequence) return -1;
    return a.info.mtimeMs - b.info.mtimeMs || a.name.localeCompare(b.name);
  };
  const claimPattern = /^(.*\.json)\.sending\.(\d+)\.[0-9a-f-]{36}$/i;
  const processAlive = (pid) => {
    try { process.kill(pid, 0); return true; }
    catch (error) { return error?.code === "EPERM"; }
  };

  async function recoverAbandonedClaims() {
    const names = await readdir(directory).catch((error) => error?.code === "ENOENT" ? [] : Promise.reject(error));
    for (const name of names) {
      const match = name.match(claimPattern);
      if (!match) continue;
      const claimPath = join(directory, name);
      try {
        const info = await stat(claimPath);
        if (processAlive(Number(match[2])) && info.mtimeMs >= Date.now() - STALE_TEMP_MS) continue;
        const target = join(directory, match[1]);
        try { await rename(claimPath, target); }
        catch (error) {
          if (error?.code === "EEXIST" || error?.code === "EPERM") await rm(claimPath, { force: true });
          else if (error?.code !== "ENOENT") throw error;
        }
        logger.warn?.("oGMemory trace spool recovered an abandoned delivery claim");
      } catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
  }
  async function ensureKnownIds() {
    if (knownIds) return knownIds;
    const names = await readdir(directory).catch((error) => error?.code === "ENOENT" ? [] : Promise.reject(error));
    const temporary = names.filter((name) => /\.json\.[0-9a-f-]{36}\.tmp$/i.test(name));
    const stale = (await Promise.all(temporary.map(async (name) => {
      try { return (await stat(join(directory, name))).mtimeMs < Date.now() - STALE_TEMP_MS ? name : null; }
      catch (error) { if (error?.code === "ENOENT") return null; throw error; }
    }))).filter(Boolean);
    if (stale.length) {
      await Promise.all(stale.map((name) => rm(join(directory, name), { force: true }).catch(() => {})));
      logger.warn?.(`oGMemory trace spool removed ${stale.length} stale temporary entries`);
    }
    const activeKeys = new Set(names.map(keyFromName));
    const publicationClaims = names.filter((name) => /^[0-9a-f]{64}\.claim$/i.test(name));
    const claimedKeys = [];
    for (const name of publicationClaims) {
      const path = join(directory, name);
      try {
        const info = await stat(path);
        const key = name.slice(0, -6);
        if (!activeKeys.has(key) && info.mtimeMs < Date.now() - STALE_TEMP_MS) await rm(path, { force: true });
        else claimedKeys.push(key);
      } catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
    const persistedKeys = names.filter((name) => name.endsWith(".json")).map(keyFromName);
    const deliveringKeys = names.map((name) => name.match(claimPattern)?.[1]).filter(Boolean).map(keyFromName);
    knownIds = new Set([...persistedKeys, ...deliveringKeys, ...claimedKeys]);
    return knownIds;
  }

  async function entries() {
    const names = (await readdir(directory).catch((error) => error?.code === "ENOENT" ? [] : Promise.reject(error)))
      .filter((name) => name.endsWith(".json"));
    const found = await Promise.all(names.map(async (name) => {
      try { return { name, info: await stat(join(directory, name)) }; }
      catch (error) { if (error?.code === "ENOENT") return null; throw error; }
    }));
    return found.filter(Boolean);
  }

  async function nextSequence() {
    const sequencePath = join(directory, ".sequence");
    let current;
    try {
      const info = await stat(sequencePath);
      if (info.size > 64) throw new SyntaxError("oversized sequence");
      const value = (await readFile(sequencePath, "utf8")).trim();
      if (!/^\d+$/.test(value)) throw new SyntaxError("invalid sequence");
      current = BigInt(value);
      if (current >= MAX_SEQUENCE) throw new SyntaxError("sequence out of range");
    } catch (error) {
      if (error?.code !== "ENOENT" && !(error instanceof SyntaxError)) throw error;
      if (error?.code !== "ENOENT") logger.warn?.("oGMemory trace spool recovered invalid sequence metadata");
      const names = await readdir(directory).catch((readError) => readError?.code === "ENOENT" ? [] : Promise.reject(readError));
      current = names.reduce((maximum, name) => {
        const value = sequenceOf(name);
        return value && BigInt(value) > maximum ? BigInt(value) : maximum;
      }, 0n);
    }
    if (current >= MAX_SEQUENCE) throw new RangeError("oGMemory trace spool sequence space exhausted");
    const next = current + 1n;
    const temporary = `${sequencePath}.${randomUUID()}.tmp`;
    await writeFile(temporary, next.toString(), { mode: 0o600 });
    try { await sequenceRename(temporary, sequencePath); }
    finally { await rm(temporary, { force: true }).catch(() => {}); }
    return next.toString().padStart(SEQUENCE_WIDTH, "0");
  }

  async function prune() {
    await ensureKnownIds();
    const found = await entries();
    found.sort(compareEntries);
    const expired = Date.now() - ttlSeconds * 1000;
    const expiredEntries = found.filter((entry) => entry.info.mtimeMs < expired);
    const liveEntries = found.filter((entry) => entry.info.mtimeMs >= expired);
    const capacityEntries = liveEntries.slice(0, Math.max(0, liveEntries.length - limit));
    const remove = [...new Map([...expiredEntries, ...capacityEntries].map((entry) => [entry.name, entry])).values()];
    let failed = 0;
    await Promise.all(remove.map(async (entry) => {
      const key = keyFromName(entry.name);
      try { await rm(join(directory, entry.name), { force: true }); await rm(idClaimPath(key), { force: true }); knownIds?.delete(key); }
      catch { failed += 1; }
    }));
    if (remove.length) logger.warn?.(`oGMemory trace spool dropped ${remove.length - failed} entries (ttl=${expiredEntries.length}, capacity=${capacityEntries.length}, remaining=${Math.max(0, found.length - remove.length + failed)})${failed ? `; ${failed} removals failed` : ""}`);
  }

  async function writeEnvelope(envelope) {
    await mkdir(directory, { recursive: true });
    const ids = await ensureKnownIds();
    const key = keyFromId(envelope.request_id);
    if (ids.has(key)) return;
    const order = await nextSequence();
    const target = join(directory, `${order}_${key}.json`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    const publicationClaim = idClaimPath(key);
    await writeFile(temporary, JSON.stringify(envelope), { mode: 0o600 });
    try {
      let hardLinkPublication = true;
      try {
        await linkFile(temporary, publicationClaim);
      } catch (error) {
        if (error?.code === "EEXIST") return;
        if (!["ENOTSUP", "EPERM", "EACCES", "EXDEV"].includes(error?.code)) throw error;
        hardLinkPublication = false;
        let claim;
        try { claim = await open(publicationClaim, "wx", 0o600); }
        catch (claimError) { if (claimError?.code === "EEXIST") return; throw claimError; }
        try {
          await claim.writeFile(JSON.stringify({ pid: process.pid, createdAt: Date.now() }));
          await claim.close();
          claim = null;
        } catch (claimError) {
          await claim?.close().catch(() => {});
          await rm(publicationClaim, { force: true }).catch(() => {});
          throw claimError;
        }
      }
      try {
        if (hardLinkPublication) await linkFile(temporary, target);
        else await rename(temporary, target);
      } catch (error) {
        await rm(publicationClaim, { force: true }).catch(() => {});
        throw error;
      }
    } finally {
      await rm(temporary, { force: true }).catch(() => {});
    }
    ids.add(key);
    if (ids.size > limit) {
      const found = await entries();
      found.sort(compareEntries);
      for (const entry of found.slice(0, ids.size - limit)) {
        const key = keyFromName(entry.name);
        await rm(join(directory, entry.name), { force: true });
        await rm(idClaimPath(key), { force: true });
        ids.delete(key);
      }
      logger.warn?.(`oGMemory trace spool dropped entries at capacity limit ${limit}`);
    }
  }

  async function enqueue(envelope) {
    if (closed) throw new Error("oGMemory trace spool is closed");
    const operation = writeQueue.then(async () => {
      const lock = await acquireFileLock({ directory, name: ".write", waitMs: 5000 });
      if (!lock) throw new Error("Timed out acquiring oGMemory trace-spool write lock");
      try { knownIds = null; await writeEnvelope(envelope); }
      finally { await lock.release(); }
    });
    writeQueue = operation.catch(() => {});
    await operation;
    void drain();
  }

  function scheduleRetry(delay) {
    if (closed || retryTimer) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (!closed) void drain();
    }, delay);
    retryTimer.unref?.();
  }

  async function drainClaimedPass() {
    await recoverAbandonedClaims();
    await prune();
    const found = await entries();
    found.sort(compareEntries);
    const seen = new Set();
    for (const entry of found) {
      if (closed) break;
      const path = join(directory, entry.name);
      const claimPath = `${path}.sending.${process.pid}.${randomUUID()}`;
      try { await rename(path, claimPath); }
      catch (error) { if (error?.code === "ENOENT" || error?.code === "EACCES" || error?.code === "EPERM") continue; throw error; }
      let envelope;
      try {
        if (entry.info.size > MAX_SPOOL_ENTRY_BYTES) throw new RangeError(`entry exceeds ${MAX_SPOOL_ENTRY_BYTES} bytes`);
        envelope = JSON.parse(await readFile(claimPath, "utf8"));
        if (!envelope?.request_id || !envelope?.session_id || !envelope?.event?.event_type) throw new SyntaxError("invalid trace envelope");
      } catch (error) {
        logger.warn?.(`Dropping corrupt oGMemory trace spool entry ${entry.name}: ${safeErrorMessage(error)}`);
        await rm(claimPath, { force: true }).catch(() => {});
        const key = keyFromName(entry.name);
        await rm(idClaimPath(key), { force: true }).catch(() => {});
        knownIds?.delete(key);
        continue;
      }
      if (seen.has(envelope.request_id)) {
        await rm(claimPath, { force: true }).catch(() => {});
        logger.warn?.("oGMemory trace spool removed a duplicate request ID");
        continue;
      }
      seen.add(envelope.request_id);
      try {
        const priorFailures = failures;
        await send(envelope);
        await rm(claimPath, { force: true });
        const key = keyFromId(envelope.request_id);
        await rm(idClaimPath(key), { force: true }).catch(() => {});
        knownIds?.delete(key);
        failures = 0;
        retryAfter = 0;
        if (priorFailures) logger.info?.(`oGMemory trace delivery recovered after ${priorFailures} failed attempt(s)`);
      } catch (error) {
        if (error?.retryable !== false) {
          failures += 1;
          const delay = traceBackoffDelay(failures, random);
          retryAfter = Date.now() + delay;
          logger.warn?.(`oGMemory trace delivery retry scheduled (status=${error?.status || "unknown"}, failures=${failures}, retryMs=${delay}, queued=${found.length})`);
          try { await rename(claimPath, path); }
          catch (renameError) {
            if (renameError?.code === "EEXIST" || renameError?.code === "EPERM") await rm(claimPath, { force: true });
            else throw renameError;
          }
          scheduleRetry(delay);
          break;
        }
        logger.warn?.(`Dropping permanent oGMemory trace failure ${entry.name}: ${safeErrorMessage(error)}`);
        await rm(claimPath, { force: true }).catch(() => {});
        const key = keyFromId(envelope.request_id);
        await rm(idClaimPath(key), { force: true }).catch(() => {});
        knownIds?.delete(key);
      }
    }
  }

  async function drainPass() {
    await writeQueue;
    if (closed || Date.now() < retryAfter) return;
    await mkdir(directory, { recursive: true });
    const claim = await acquireFileLock({ directory, name: ".drain", waitMs: 0 });
    if (!claim) { scheduleRetry(Math.floor(100 + random() * 400)); return; }
    try { await drainClaimedPass(); }
    finally { await claim.release(); }
  }

  function drain() {
    if (closed) return draining || Promise.resolve();
    drainRequested = true;
    if (!draining) {
      draining = (async () => {
        while (drainRequested) {
          drainRequested = false;
          await drainPass();
        }
      })().catch((error) => {
        const delay = Math.floor(100 + random() * 400);
        logger.warn?.(`oGMemory trace drain failed open; retry scheduled in ${delay}ms: ${safeErrorMessage(error)}`);
        scheduleRetry(delay);
      }).finally(() => {
        draining = null;
        if (drainRequested && !closed) void drain();
      });
    }
    return draining;
  }

  function close() {
    closed = true;
    drainRequested = false;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = null;
  }

  return { enqueue, drain, flush: drain, close };
}

export function traceBackoffDelay(failures, random = Math.random) {
  const cap = Math.min(30000, 500 * (2 ** Math.min(Math.max(0, failures - 1), 6)));
  return Math.floor(cap / 2 + random() * cap / 2);
}
