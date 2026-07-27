import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, open, readdir, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { acquireFileLock } from "../lib/file-lock.mjs";

test("file-lock stale cleanup preserves mutual exclusion for simultaneous reclaimers", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-file-lock-"));
  const stale = join(dir, "state.lease.000000000001.dead-owner");
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

test("file-lock removes a choosing file when owner metadata cannot be written", async () => {
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
