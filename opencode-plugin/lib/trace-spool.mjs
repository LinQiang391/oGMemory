import { createHash, randomUUID } from "node:crypto";
import { link, mkdir, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { acquireFileLock } from "./file-lock.mjs";
import { safeErrorMessage } from "./redaction.mjs";

const STALE_TEMP_MS = 5 * 60 * 1000;
const MAX_SPOOL_ENTRY_BYTES = 64 * 1024;
const SEQUENCE_WIDTH = 24;
const MAX_SEQUENCE = (10n ** BigInt(SEQUENCE_WIDTH)) - 1n;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const ACTIVE_CLAIMS_KEY = Symbol.for("@opengauss/ogmemory-active-trace-claims");
const activeDeliveryClaims = globalThis[ACTIVE_CLAIMS_KEY] ||= new Set();

export function createTraceSpool({ directory, limit, ttlSeconds, send, logger = console, random = Math.random, linkFile = link, sequenceRename = rename, publishRename = rename, claimRename = rename }) {
  let draining = null;
  let drainRequested = false;
  let failures = 0;
  let retryAfter = 0;
  let retryTimer = null;
  let writeQueue = Promise.resolve();
  let knownIds = null;
  let closed = false;
  const ownedDeliveryClaims = new Map();
  const instanceDeliveryClaims = new Set();

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
    if (!Number.isInteger(pid) || pid <= 0) return false;
    try { process.kill(pid, 0); return true; }
    catch (error) { return error?.code === "EPERM"; }
  };
  function registerDeliveryClaim(claimPath) {
    instanceDeliveryClaims.add(claimPath);
    if (!closed) activeDeliveryClaims.add(claimPath);
  }
  function releaseDeliveryClaim(claimPath) {
    activeDeliveryClaims.delete(claimPath);
    instanceDeliveryClaims.delete(claimPath);
    ownedDeliveryClaims.delete(claimPath);
  }
  async function renameWithTransientRetry(source, target) {
    let lastError;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try { await claimRename(source, target); return; }
      catch (error) {
        if (!["EPERM", "EACCES"].includes(error?.code)) throw error;
        lastError = error;
        if (attempt < 2) await wait(10);
      }
    }
    throw lastError;
  }
  async function restoreDeliveryClaim(claimPath, target) {
    try { await renameWithTransientRetry(claimPath, target); releaseDeliveryClaim(claimPath); }
    catch (error) {
      if (error?.code === "ENOENT") { releaseDeliveryClaim(claimPath); return; }
      if (error?.code === "EEXIST" || error?.code === "EPERM" || error?.code === "EACCES") {
        try { await stat(target); }
        catch (statError) { if (statError?.code === "ENOENT") throw error; throw statError; }
        await rm(claimPath, { force: true });
        releaseDeliveryClaim(claimPath);
        return;
      }
      throw error;
    }
  }
  async function recoverOwnedDeliveryClaims() {
    for (const [claimPath, target] of ownedDeliveryClaims) {
      await restoreDeliveryClaim(claimPath, target);
      ownedDeliveryClaims.delete(claimPath);
    }
  }

  async function recoverAbandonedClaims() {
    const names = await readdir(directory).catch((error) => error?.code === "ENOENT" ? [] : Promise.reject(error));
    for (const name of names) {
      const match = name.match(claimPattern);
      if (!match) continue;
      const claimPath = join(directory, name);
      try {
        const info = await stat(claimPath);
        const ownerPid = Number(match[2]);
        const activelyOwned = ownerPid === process.pid ? activeDeliveryClaims.has(claimPath) : processAlive(ownerPid);
        if (activelyOwned && info.mtimeMs >= Date.now() - STALE_TEMP_MS) continue;
        const target = join(directory, match[1]);
        await restoreDeliveryClaim(claimPath, target);
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
    const promotedKeys = [];
    for (const name of publicationClaims) {
      const path = join(directory, name);
      try {
        const info = await stat(path);
        const key = name.slice(0, -6);
        if (!activeKeys.has(key) && info.size <= MAX_SPOOL_ENTRY_BYTES) {
          const envelope = await readFile(path, "utf8").then(JSON.parse).catch(() => null);
          if (envelope?.request_id && envelope?.session_id && envelope?.event && keyFromId(envelope.request_id) === key) {
            const order = await nextSequence();
            await rename(path, join(directory, `${order}_${key}.json`));
            promotedKeys.push(key);
            logger.warn?.("oGMemory trace spool recovered an interrupted publication claim");
            continue;
          }
        }
        if (!activeKeys.has(key) && info.mtimeMs < Date.now() - STALE_TEMP_MS) await rm(path, { force: true });
        else claimedKeys.push(key);
      } catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
    const persistedKeys = names.filter((name) => name.endsWith(".json")).map(keyFromName);
    const deliveringKeys = names.map((name) => name.match(claimPattern)?.[1]).filter(Boolean).map(keyFromName);
    knownIds = new Set([...persistedKeys, ...deliveringKeys, ...claimedKeys, ...promotedKeys]);
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

  async function enforceCapacity(ids) {
    if (ids.size <= limit) return;
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

  async function writeEnvelope(envelope, enforceLimit = true) {
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
      }
      try {
        if (hardLinkPublication) await linkFile(temporary, target);
        else await publishRename(temporary, target);
      } catch (error) {
        if (hardLinkPublication) await rm(publicationClaim, { force: true }).catch(() => {});
        throw error;
      }
    } finally {
      await rm(temporary, { force: true }).catch(() => {});
    }
    ids.add(key);
    if (enforceLimit) await enforceCapacity(ids);
  }

  async function enqueueMany(envelopes) {
    if (closed) throw new Error("oGMemory trace spool is closed");
    const batch = Array.isArray(envelopes) ? envelopes : [];
    if (!batch.length) return;
    const operation = writeQueue.then(async () => {
      const lock = await acquireFileLock({ directory, name: ".write", waitMs: 5000 });
      if (!lock) throw new Error("Timed out acquiring oGMemory trace-spool write lock");
      try {
        knownIds = null;
        const ids = await ensureKnownIds();
        const unique = new Map();
        for (const envelope of batch) {
          const key = keyFromId(envelope.request_id);
          if (!ids.has(key) && !unique.has(key)) unique.set(key, envelope);
        }
        const publish = [...unique.values()].slice(-limit);
        if (publish.length < unique.size) logger.warn?.(`oGMemory trace spool dropped ${unique.size - publish.length} batch entries at capacity limit ${limit}`);
        for (const envelope of publish) await writeEnvelope(envelope, false);
        await enforceCapacity(knownIds);
      }
      finally { await lock.release(); }
    });
    writeQueue = operation.catch(() => {});
    await operation;
    void drain();
  }

  async function enqueue(envelope) { return enqueueMany([envelope]); }

  function scheduleRetry(delay) {
    if (closed || retryTimer) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (!closed) void drain();
    }, delay);
    retryTimer.unref?.();
  }

  async function drainClaimedPass() {
    const writeLock = await acquireFileLock({ directory, name: ".write", waitMs: 5000 });
    if (!writeLock) throw new Error("Timed out acquiring oGMemory trace-spool write lock for recovery");
    try {
      knownIds = null;
      await recoverOwnedDeliveryClaims();
      await recoverAbandonedClaims();
      await prune();
    } finally { await writeLock.release(); }
    const found = await entries();
    found.sort(compareEntries);
    const seen = new Set();
    for (const entry of found) {
      if (closed) break;
      const path = join(directory, entry.name);
      const claimPath = `${path}.sending.${process.pid}.${randomUUID()}`;
      try { await renameWithTransientRetry(path, claimPath); }
      catch (error) {
        if (error?.code === "ENOENT") continue;
        if (error?.code === "EACCES" || error?.code === "EPERM") {
          scheduleRetry(Math.floor(100 + random() * 400));
          continue;
        }
        throw error;
      }
      registerDeliveryClaim(claimPath);
      let envelope;
      try {
        if (entry.info.size > MAX_SPOOL_ENTRY_BYTES) throw new RangeError(`entry exceeds ${MAX_SPOOL_ENTRY_BYTES} bytes`);
        envelope = JSON.parse(await readFile(claimPath, "utf8"));
        if (!envelope?.request_id || !envelope?.session_id || !envelope?.event?.event_type) throw new SyntaxError("invalid trace envelope");
      } catch (error) {
        logger.warn?.(`Dropping corrupt oGMemory trace spool entry ${entry.name}: ${safeErrorMessage(error)}`);
        await rm(claimPath, { force: true }).catch(() => {});
        releaseDeliveryClaim(claimPath);
        const key = keyFromName(entry.name);
        await rm(idClaimPath(key), { force: true }).catch(() => {});
        knownIds?.delete(key);
        continue;
      }
      if (seen.has(envelope.request_id)) {
        await rm(claimPath, { force: true }).catch(() => {});
        releaseDeliveryClaim(claimPath);
        logger.warn?.("oGMemory trace spool removed a duplicate request ID");
        continue;
      }
      seen.add(envelope.request_id);
      try {
        const priorFailures = failures;
        await send(envelope);
        await rm(claimPath, { force: true });
        releaseDeliveryClaim(claimPath);
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
          try { await restoreDeliveryClaim(claimPath, path); }
          catch { ownedDeliveryClaims.set(claimPath, path); }
          scheduleRetry(delay);
          break;
        }
        logger.warn?.(`Dropping permanent oGMemory trace failure ${entry.name}: ${safeErrorMessage(error)}`);
        await rm(claimPath, { force: true }).catch(() => {});
        releaseDeliveryClaim(claimPath);
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
    for (const claimPath of instanceDeliveryClaims) activeDeliveryClaims.delete(claimPath);
  }

  return { enqueue, enqueueMany, drain, flush: drain, close };
}

export function traceBackoffDelay(failures, random = Math.random) {
  const cap = Math.min(30000, 500 * (2 ** Math.min(Math.max(0, failures - 1), 6)));
  return Math.floor(cap / 2 + random() * cap / 2);
}
