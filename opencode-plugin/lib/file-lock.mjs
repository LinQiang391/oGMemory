import { randomUUID } from "node:crypto";
import { mkdir, open, readdir, readFile, rename, rm, stat, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function processAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return error?.code === "EPERM"; }
}

async function metadataAt(path) {
  return readFile(path, "utf8").then(JSON.parse).catch(() => null);
}

async function statWithRetry(path, statImpl = stat) {
  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try { return await statImpl(path); }
    catch (error) {
      if (!["EPERM", "EACCES"].includes(error?.code)) throw error;
      lastError = error;
      if (attempt < 2) await wait(1);
    }
  }
  throw lastError;
}

async function openExclusive(path, openImpl, statImpl, ...args) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try { return await openImpl(path, "wx", ...args); }
    catch (error) {
      if (error?.code === "EEXIST") return null;
      if (error?.code === "EACCES" || error?.code === "EPERM") {
        try { await statWithRetry(path, statImpl); return null; }
        catch (statError) {
          if (statError?.code !== "ENOENT") throw statError;
          if (attempt < 2) { await wait(1); continue; }
        }
      }
      throw error;
    }
  }
}

async function releaseOwnedPath(path, token, { readFileImpl = readFile, removeFile = rm, writeFileImpl = writeFile } = {}) {
  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const metadata = JSON.parse(await readFileImpl(path, "utf8"));
      if (metadata?.token !== token) return;
      await removeFile(path, { force: true });
      return;
    } catch (error) {
      if (error?.code === "ENOENT") return;
      lastError = error;
      if (attempt < 2) await wait(10);
    }
  }
  try {
    const metadata = JSON.parse(await readFileImpl(path, "utf8"));
    if (metadata?.token === token) await writeFileImpl(path, JSON.stringify({ pid: 0, token }), { mode: 0o600 });
  } catch {}
  throw lastError;
}

async function acquireCanonical({ directory, name, waitMs, staleMs, openFile, statImpl, releaseIo }) {
  const token = randomUUID();
  const owner = JSON.stringify({ pid: process.pid, token });
  const lockPath = join(directory, `${name}.lock`);
  const reclaimPath = join(directory, `${name}.reclaim`);
  const deadline = Date.now() + waitMs;

  const tryAcquire = async () => {
    const handle = await openExclusive(lockPath, openFile, statImpl, 0o600);
    if (!handle) return null;
    try { await handle.writeFile(owner); }
    catch (error) {
      await handle.close().catch(() => {});
      await rm(lockPath, { force: true }).catch(() => {});
      throw error;
    }
    await handle.close();
    const heartbeat = setInterval(() => {
      const date = new Date();
      void utimes(lockPath, date, date).catch(() => {});
    }, Math.max(1000, Math.floor(staleMs / 3)));
    heartbeat.unref?.();
    return {
      async release() {
        clearInterval(heartbeat);
        await releaseOwnedPath(lockPath, token, releaseIo);
      },
    };
  };

  while (true) {
    try {
      const reclaimInfo = await statWithRetry(reclaimPath, statImpl);
      const reclaimMetadata = await metadataAt(reclaimPath);
      const reclaimable = reclaimMetadata
        ? !processAlive(reclaimMetadata.pid)
        : reclaimInfo.mtimeMs < Date.now() - staleMs;
      if (reclaimable) await rm(reclaimPath, { force: true });
      else {
        if (Date.now() >= deadline) return null;
        await wait(10);
        continue;
      }
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }

    const acquired = await tryAcquire();
    if (acquired) return acquired;

    try {
      const [info, metadata] = await Promise.all([statWithRetry(lockPath, statImpl), metadataAt(lockPath)]);
      const reclaimable = metadata ? !processAlive(metadata.pid) : info.mtimeMs < Date.now() - staleMs;
      if (reclaimable) {
        const reclaimToken = randomUUID();
        let reclaimHandle = await openExclusive(reclaimPath, open, statImpl, 0o600);
        if (!reclaimHandle) continue;
        try {
          await reclaimHandle.writeFile(JSON.stringify({ pid: process.pid, token: reclaimToken }));
          await reclaimHandle.close();
          reclaimHandle = null;
          const current = await metadataAt(lockPath);
          if ((!metadata && !current) || (metadata?.token && current?.token === metadata.token)) {
            await rm(lockPath, { force: true });
            const replacement = await tryAcquire();
            if (replacement) {
              if ((await metadataAt(reclaimPath))?.token === reclaimToken) await rm(reclaimPath, { force: true });
              return replacement;
            }
          }
        } finally {
          if (reclaimHandle) await reclaimHandle.close().catch(() => {});
          if ((await metadataAt(reclaimPath))?.token === reclaimToken) await rm(reclaimPath, { force: true });
        }
      }
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (Date.now() >= deadline) return null;
    await wait(10);
  }
}

async function acquireLegacyLease({ directory, name, waitMs, staleMs, statImpl, releaseIo }) {
  const token = randomUUID();
  const owner = JSON.stringify({ pid: process.pid, token });
  const choosingPrefix = `${name}.choosing.`;
  const leasePrefix = `${name}.lease.`;
  const choosingPath = join(directory, `${choosingPrefix}${token}`);

  async function activeEntries() {
    const names = await readdir(directory);
    const candidates = names.filter((entry) => entry.startsWith(choosingPrefix) || entry.startsWith(leasePrefix));
    const active = [];
    for (const entry of candidates) {
      const path = join(directory, entry);
      try {
        const [info, metadata] = await Promise.all([statWithRetry(path, statImpl), metadataAt(path)]);
        const reclaimable = metadata ? !processAlive(metadata.pid) : info.mtimeMs < Date.now() - staleMs;
        if (reclaimable) { await rm(path, { force: true }); continue; }
        active.push({ entry, path });
      } catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
    return active;
  }

  let leasePath;
  let heartbeat;
  try {
    const choosing = await open(choosingPath, "wx", 0o600);
    try { await choosing.writeFile(owner); }
    finally { await choosing.close(); }
    const entries = await activeEntries();
    const tickets = entries.filter(({ entry }) => entry.startsWith(leasePrefix))
      .map(({ entry }) => Number(entry.slice(leasePrefix.length).split(".", 1)[0])).filter(Number.isFinite);
    const ticket = (tickets.length ? Math.max(...tickets) : 0) + 1;
    leasePath = join(directory, `${leasePrefix}${String(ticket).padStart(12, "0")}.${token}`);
    await rename(choosingPath, leasePath);
    heartbeat = setInterval(() => {
      const date = new Date();
      void utimes(leasePath, date, date).catch(() => {});
    }, Math.max(1000, Math.floor(staleMs / 3)));
    heartbeat.unref?.();
    const deadline = Date.now() + waitMs;
    while (true) {
      const current = await activeEntries();
      const choosingActive = current.some(({ entry }) => entry.startsWith(choosingPrefix));
      const leases = current.filter(({ entry }) => entry.startsWith(leasePrefix)).sort((a, b) => a.entry.localeCompare(b.entry));
      if (!choosingActive && leases[0]?.path === leasePath) {
        return { async release() { clearInterval(heartbeat); await releaseOwnedPath(leasePath, token, releaseIo); } };
      }
      if (Date.now() >= deadline) {
        clearInterval(heartbeat);
        await rm(leasePath, { force: true });
        return null;
      }
      await wait(10);
    }
  } catch (error) {
    if (heartbeat) clearInterval(heartbeat);
    await rm(leasePath || choosingPath, { force: true }).catch(() => {});
    throw error;
  }
}

export async function acquireFileLock({
  directory, name, waitMs = 5000, staleMs = 30000, openFile = open,
  readFileImpl = readFile, removeFile = rm, statImpl = stat, writeFileImpl = writeFile,
}) {
  await mkdir(directory, { recursive: true });
  const startedAt = Date.now();
  const releaseIo = { readFileImpl, removeFile, writeFileImpl };
  const canonical = await acquireCanonical({ directory, name, waitMs, staleMs, openFile, statImpl, releaseIo });
  if (!canonical) return null;
  try {
    const legacy = await acquireLegacyLease({ directory, name, waitMs: Math.max(0, waitMs - (Date.now() - startedAt)), staleMs, statImpl, releaseIo });
    if (!legacy) { await canonical.release(); return null; }
    return {
      async release() {
        const errors = [];
        try { await legacy.release(); } catch (error) { errors.push(error); }
        try { await canonical.release(); } catch (error) { errors.push(error); }
        if (errors.length) throw new AggregateError(errors, "Failed to release oGMemory file lock");
      },
    };
  } catch (error) {
    await canonical.release();
    throw error;
  }
}
