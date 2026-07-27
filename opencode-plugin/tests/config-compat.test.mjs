import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { loadConfig } from "../lib/config.mjs";
import { detectCompatibility, parseVersion, versionAtLeast } from "../lib/compatibility.mjs";

test("configuration file is overridden by environment", async () => {
  const home = await mkdtemp(join(tmpdir(), "ogmem-config-"));
  const path = join(home, ".config", "opencode", "ogmemory-config.json");
  await mkdir(join(home, ".config", "opencode"), { recursive: true });
  await writeFile(path, JSON.stringify({ endpoint: "https://file.example", traceCaptureContent: true }));
  const config = await loadConfig({ home, env: { OG_MEMORY_URL: "https://env.example/", OGMEM_TRACE_CAPTURE_CONTENT: "0" } });
  assert.equal(config.endpoint, "https://env.example");
  assert.equal(config.traceCaptureContent, false);
});

test("configuration rejects plaintext HTTP for non-loopback endpoints", async () => {
  await assert.rejects(loadConfig({ env: { OG_MEMORY_URL: "http://memory.example:8090" } }), /HTTPS.*non-loopback/i);
  assert.equal((await loadConfig({ env: { OG_MEMORY_URL: "http://127.0.0.1:8090" } })).endpoint, "http://127.0.0.1:8090");
  assert.equal((await loadConfig({ env: { OG_MEMORY_URL: "http://[::1]:8090" } })).endpoint, "http://[::1]:8090");
});

test("configuration validates disablement and rejects unknown fields", async () => {
  const home = await mkdtemp(join(tmpdir(), "ogmem-config-disabled-"));
  const path = join(home, ".config", "opencode", "ogmemory-config.json");
  await mkdir(join(home, ".config", "opencode"), { recursive: true });
  await writeFile(path, JSON.stringify({ enabled: false }));
  assert.equal((await loadConfig({ home, env: {} })).enabled, false);
  await writeFile(path, JSON.stringify({ endpont: "https://typo.example" }));
  await assert.rejects(loadConfig({ home, env: {} }), /unexpected field.*endpont/i);
});

test("compatibility reports graceful-dispose support separately", async () => {
  const old = await detectCompatibility(async () => ({ stdout: "1.1.13", stderr: "" }));
  const current = await detectCompatibility(async () => ({ stdout: "1.15.11", stderr: "" }));
  assert.equal(old.supported, true);
  assert.equal(old.hasDispose, false);
  assert.equal(current.hasDispose, true);
});

test("OpenCode minimum version comparison is exact", () => {
  assert.deepEqual(parseVersion("opencode v1.1.13"), [1, 1, 13]);
  assert.equal(versionAtLeast("1.1.12"), false);
  assert.equal(versionAtLeast("1.1.13"), true);
  assert.equal(versionAtLeast("1.2.0"), true);
  assert.equal(versionAtLeast("unknown"), null);
  assert.equal(versionAtLeast("1.1.13-beta.1"), false);
  assert.equal(versionAtLeast("1.15.11-beta.1", "1.15.11"), false);
});
