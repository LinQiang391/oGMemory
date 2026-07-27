import { randomUUID } from "node:crypto";
import { mkdir, open, readdir, readFile, rename, rm, stat, utimes } from "node:fs/promises";
import { join } from "node:path";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function processAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return error?.code === "EPERM"; }
}

export async function acquireFileLock({ directory, name, waitMs = 5000, staleMs = 30000, openFile = open }) {
  await mkdir(directory, { recursive: true });
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
        const info = await stat(path);
        const metadata = await readFile(path, "utf8").then(JSON.parse).catch(() => null);
        if (info.mtimeMs < Date.now() - staleMs || (metadata && !processAlive(metadata.pid))) { await rm(path, { force: true }); continue; }
        active.push({ entry, path });
      } catch (error) { if (error?.code !== "ENOENT") throw error; }
    }
    return active;
  }

  let leasePath;
  let heartbeat;
  try {
    const choosing = await openFile(choosingPath, "wx", 0o600);
    try { await choosing.writeFile(owner); }
    finally { await choosing.close(); }
    const entries = await activeEntries();
    const tickets = entries.filter(({ entry }) => entry.startsWith(leasePrefix))
      .map(({ entry }) => Number(entry.slice(leasePrefix.length).split(".", 1)[0])).filter(Number.isFinite);
    const ticket = (tickets.length ? Math.max(...tickets) : 0) + 1;
    const leaseName = `${leasePrefix}${String(ticket).padStart(12, "0")}.${token}`;
    leasePath = join(directory, leaseName);
    await rename(choosingPath, leasePath);
    heartbeat = setInterval(() => { const date = new Date(); void utimes(leasePath, date, date).catch(() => {}); }, Math.max(1000, Math.floor(staleMs / 3)));
    heartbeat.unref?.();
    const deadline = Date.now() + waitMs;
    while (true) {
      const current = await activeEntries();
      const choosingActive = current.some(({ entry }) => entry.startsWith(choosingPrefix));
      const leases = current.filter(({ entry }) => entry.startsWith(leasePrefix)).sort((a, b) => a.entry.localeCompare(b.entry));
      if (!choosingActive && leases[0]?.path === leasePath) {
        return { async release() { clearInterval(heartbeat); await rm(leasePath, { force: true }); } };
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
