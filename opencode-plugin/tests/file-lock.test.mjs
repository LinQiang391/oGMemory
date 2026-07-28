import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, open, readdir, rm, stat, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { acquireFileLock } from "../lib/file-lock.mjs";

test("file-lock stale cleanup preserves mutual exclusion for simultaneous reclaimers", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-"));
  const stale = join(dir, "state.lock");
  await writeFile(stale, JSON.stringify({ pid: 2147483647, token: "dead-owner" }));
  const old = new Date(Date.now() - 60000);
  await utimes(stale, old, old);
  let active = 0;
  let maximum = 0;
  const contender = async () => {
    const lock = await acquireFileLock({ directory: dir, name: "state", waitMs: 1000 });
    assert.ok(lock);
    active += 1;
    maximum = Math.max(maximum, active);
    await new Promise((resolve) => setTimeout(resolve, 20));
    active -= 1;
    await lock.release();
  };
  await Promise.all([contender(), contender()]);
  assert.equal(maximum, 1);
});

test("file-lock removes its canonical lock file when owner metadata cannot be written", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-write-failure-"));
  const openFile = async (...args) => {
    const handle = await open(...args);
    return {
      writeFile: async () => { throw new Error("disk write failed"); },
      close: () => handle.close(),
    };
  };

  await assert.rejects(
    acquireFileLock({ directory: dir, name: "state", waitMs: 50, openFile }),
    /disk write failed/,
  );
  assert.deepEqual(await readdir(dir), []);

  const lock = await acquireFileLock({ directory: dir, name: "state", waitMs: 50 });
  assert.ok(lock);
  await lock.release();
});

test("file-lock never reclaims a stale-mtime lock owned by a live process", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-live-owner-"));
  const first = await acquireFileLock({ directory: dir, name: "state", waitMs: 50, staleMs: 10 });
  assert.ok(first);
  const old = new Date(Date.now() - 60000);
  await utimes(join(dir, "state.lock"), old, old);

  const second = await acquireFileLock({ directory: dir, name: "state", waitMs: 30, staleMs: 10 });
  assert.equal(second, null);
  await first.release();

  const third = await acquireFileLock({ directory: dir, name: "state", waitMs: 50, staleMs: 10 });
  assert.ok(third);
  await third.release();
});

test("file-lock remains mutually exclusive with a live legacy lease", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-legacy-owner-"));
  const legacy = join(dir, "state.lease.000000000001.legacy-owner");
  await writeFile(legacy, JSON.stringify({ pid: process.pid, token: "legacy-owner" }));
  const blocked = await acquireFileLock({ directory: dir, name: "state", waitMs: 30 });
  assert.equal(blocked, null);
  await rm(legacy, { force: true });

  const acquired = await acquireFileLock({ directory: dir, name: "state", waitMs: 100 });
  assert.ok(acquired);
  const names = await readdir(dir);
  assert.equal(names.some((name) => name.startsWith("state.lease.")), true);
  await acquired.release();
});

test("file-lock retries a transient canonical release failure", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-release-retry-"));
  let failed = false;
  const lock = await acquireFileLock({
    directory: dir,
    name: "state",
    removeFile: async (path, options) => {
      if (path.endsWith("state.lock") && !failed) {
        failed = true;
        throw Object.assign(new Error("transient remove failure"), { code: "EACCES" });
      }
      return rm(path, options);
    },
  });
  await lock.release();
  const reacquired = await acquireFileLock({ directory: dir, name: "state", waitMs: 50 });
  assert.ok(reacquired);
  await reacquired.release();
});

test("file-lock releases canonical state when legacy release remains unavailable", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-release-finally-"));
  const lock = await acquireFileLock({
    directory: dir,
    name: "state",
    removeFile: async (path, options) => {
      if (path.includes("state.lease.")) throw Object.assign(new Error("legacy remove failure"), { code: "EACCES" });
      return rm(path, options);
    },
  });
  await assert.rejects(lock.release(), AggregateError);
  assert.equal((await readdir(dir)).includes("state.lock"), false);

  const reacquired = await acquireFileLock({ directory: dir, name: "state", waitMs: 100 });
  assert.ok(reacquired);
  await reacquired.release();
});

test("file-lock treats Windows EPERM as contention only when the lock path exists", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-eperm-"));
  let first = true;
  const lock = await acquireFileLock({
    directory: dir,
    name: "state",
    waitMs: 100,
    openFile: async (path, ...args) => {
      if (first) {
        first = false;
        await writeFile(path, JSON.stringify({ pid: 2147483647, token: "dead-owner" }));
        throw Object.assign(new Error("Windows lock contention"), { code: "EPERM" });
      }
      return open(path, ...args);
    },
  });
  assert.ok(lock);
  await lock.release();

  await assert.rejects(
    acquireFileLock({
      directory: dir,
      name: "unwritable",
      openFile: async () => { throw Object.assign(new Error("permission denied"), { code: "EPERM" }); },
    }),
    /permission denied/,
  );
});

test("file-lock retries when an EPERM contention path vanishes before stat", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-eperm-vanished-"));
  let calls = 0;
  const lock = await acquireFileLock({
    directory: dir,
    name: "state",
    openFile: async (path, ...args) => {
      calls += 1;
      if (calls === 1) {
        await writeFile(path, JSON.stringify({ pid: process.pid, token: "departing-owner" }));
        await rm(path, { force: true });
        throw Object.assign(new Error("Windows lock contention"), { code: "EPERM" });
      }
      return open(path, ...args);
    },
  });
  assert.ok(lock);
  assert.equal(calls, 2);
  await lock.release();
});

test("file-lock retries a transient EPERM while inspecting a contended path", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-stat-eperm-"));
  let openCalls = 0;
  let statFailed = false;
  const lock = await acquireFileLock({
    directory: dir,
    name: "state",
    openFile: async (path, ...args) => {
      openCalls += 1;
      if (openCalls === 1) {
        await writeFile(path, JSON.stringify({ pid: process.pid, token: "departing-owner" }));
        await rm(path, { force: true });
        throw Object.assign(new Error("Windows open contention"), { code: "EPERM" });
      }
      return open(path, ...args);
    },
    statImpl: async (path) => {
      if (path.endsWith("state.lock") && !statFailed) {
        statFailed = true;
        throw Object.assign(new Error("Windows stat contention"), { code: "EPERM" });
      }
      return stat(path);
    },
  });
  assert.ok(lock);
  assert.equal(openCalls, 2);
  await lock.release();
});
