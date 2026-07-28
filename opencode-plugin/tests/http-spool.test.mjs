import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { link, mkdtemp, readdir, readFile, rename, rm, stat, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHttpClient, HttpError } from "../lib/http-client.mjs";
import { createTraceSpool, traceBackoffDelay } from "../lib/trace-spool.mjs";

const config = { endpoint: "http://memory", apiKey: "key", accountId: "a", userId: "u", agentId: "opencode", composeTimeoutMs: 1000, writeTimeoutMs: 1000, traceTimeoutMs: 1000 };

test("HTTP client sends identity and authentication using server contracts", async () => {
  let request;
  const http = createHttpClient(config, async (url, options) => {
    request = { url, options, body: JSON.parse(options.body) };
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  });
  await http.afterTurn("s", [{ role: "user", content: "hello" }]);
  assert.equal(request.url, "http://memory/api/v1/after_turn");
  assert.equal(request.options.headers["X-API-Key"], "key");
  assert.deepEqual(request.body, { accountId: "a", userId: "u", agentId: "opencode", sessionId: "s", messages: [{ role: "user", content: "hello" }] });
});

test("HTTP retry classification distinguishes transient and permanent status", async () => {
  const failing = (status) => createHttpClient(config, async () => new Response(JSON.stringify({ error: "failure" }), { status })).trace({});
  await assert.rejects(failing(503), (error) => error instanceof HttpError && error.retryable === true);
  await assert.rejects(failing(401), (error) => error instanceof HttpError && error.retryable === true);
  const html401 = createHttpClient(config, async () => new Response("unauthorized", { status: 401 })).trace({});
  await assert.rejects(html401, (error) => error instanceof HttpError && error.retryable === true);
  const write401 = createHttpClient(config, async () => new Response("unauthorized", { status: 401 })).afterTurn("s", []);
  await assert.rejects(write401, (error) => error instanceof HttpError && error.retryable === false);
  const semanticFailure = createHttpClient(config, async () => new Response(JSON.stringify({ ok: false, error: "commit failed", retryable: true }), { status: 200 })).afterTurn("s", []);
  await assert.rejects(semanticFailure, (error) => error instanceof HttpError && error.retryable === true);
});

test("write and trace require an explicit success acknowledgement", async () => {
  for (const body of ["", "{}"] ) {
    const client = createHttpClient(config, async () => new Response(body, { status: 200 }));
    await assert.rejects(client.afterTurn("s", []), (error) => error instanceof HttpError && error.retryable === true);
    await assert.rejects(client.trace({}), (error) => error instanceof HttpError && error.retryable === true);
  }
});

test("HTTP errors never expose server-provided error text", async () => {
  const client = createHttpClient(config, async () => new Response(JSON.stringify({ error: "password=hunter2 person@example.com" }), { status: 400 }));
  await assert.rejects(client.afterTurn("s", []), (error) => {
    assert.equal(error.status, 400);
    assert.doesNotMatch(error.message, /hunter2|person@example.com/);
    return true;
  });
});

test("HTTP request timeouts abort and remain retryable without exposing content", async () => {
  const aborted = [];
  const hangingFetch = async (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener("abort", () => {
      aborted.push(url);
      const error = new Error("secret prompt and transcript");
      error.name = "AbortError";
      reject(error);
    }, { once: true });
  });
  const client = createHttpClient({ ...config, composeTimeoutMs: 10, writeTimeoutMs: 10 }, hangingFetch);

  await assert.rejects(client.compose("s", "private prompt"), (error) => error instanceof HttpError && error.retryable && /timed out/.test(error.message) && !/private|secret/.test(error.message));
  await assert.rejects(client.afterTurn("s", [{ role: "user", content: "private transcript" }]), (error) => error instanceof HttpError && error.retryable && /timed out/.test(error.message) && !/private|secret/.test(error.message));
  assert.equal(aborted.length, 2);
});

test("HTTP client rejects oversized responses before JSON parsing", async () => {
  const client = createHttpClient(config, async () => new Response("x".repeat(70000), { status: 200 }));
  await assert.rejects(client.afterTurn("s", []), (error) => error instanceof HttpError && error.retryable === true && /exceeded 65536 bytes/.test(error.message));
});

test("trace spool retains an oversized successful response without an acknowledgement", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-oversized-ack-"));
  const client = createHttpClient(config, async () => new Response("x".repeat(70000), { status: 200 }));
  const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: (envelope) => client.trace(envelope), logger: { warn() {} } });
  await spool.enqueue({ request_id: "oversized-ack", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await spool.flush();
  assert.equal((await readdir(dir)).filter((name) => name.endsWith(".json")).length, 1);
  await spool.close();
  await rm(dir, { recursive: true, force: true });
});

test("oversized HTTP errors preserve status retry classification", async () => {
  const declaredTransient = createHttpClient(config, async () => new Response("", { status: 503, headers: { "content-length": "70000" } }));
  await assert.rejects(declaredTransient.afterTurn("s", []), (error) => error instanceof HttpError && error.status === 503 && error.retryable === true);

  const streamedTransient = createHttpClient(config, async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(70000)); controller.close(); },
  }), { status: 429 }));
  await assert.rejects(streamedTransient.afterTurn("s", []), (error) => error instanceof HttpError && error.status === 429 && error.retryable === true);

  const bufferedTransient = createHttpClient(config, async () => ({
    status: 408, ok: false, headers: { get: () => null }, body: null, text: async () => "x".repeat(70000),
  }));
  await assert.rejects(bufferedTransient.afterTurn("s", []), (error) => error instanceof HttpError && error.status === 408 && error.retryable === true);

  const permanent = createHttpClient(config, async () => new Response("", { status: 400, headers: { "content-length": "70000" } }));
  await assert.rejects(permanent.afterTurn("s", []), (error) => error instanceof HttpError && error.status === 400 && error.retryable === false);
});

test("declared oversized HTTP responses cancel their body", async () => {
  let cancelled = false;
  const body = new ReadableStream({ cancel() { cancelled = true; } });
  const client = createHttpClient(config, async () => new Response(body, { status: 200, headers: { "content-length": "70000" } }));
  await assert.rejects(client.afterTurn("s", []), /exceeded 65536 bytes/);
  assert.equal(cancelled, true);
});

test("streamed oversized HTTP responses cancel their reader", async () => {
  let cancelled = false;
  let chunks = 0;
  const body = new ReadableStream({
    pull(controller) { chunks += 1; controller.enqueue(new Uint8Array(40000)); if (chunks > 2) controller.close(); },
    cancel() { cancelled = true; },
  });
  const client = createHttpClient(config, async () => new Response(body, { status: 200 }));
  await assert.rejects(client.afterTurn("s", []), /exceeded 65536 bytes/);
  assert.equal(cancelled, true);
});

test("spool drops permanent failures but retains retryable failures", async () => {
  const permanentDir = await mkdtemp(join(tmpdir(), "ogmem-permanent-"));
  const permanent = createTraceSpool({ directory: permanentDir, limit: 2, ttlSeconds: 3600, send: async () => { throw new HttpError("bad", { status: 400, retryable: false }); }, logger: { warn() {} } });
  await permanent.enqueue({ request_id: "a", session_id: "s", timestamp: "2026-01-01T00:00:00Z", event: { event_type: "session_start" } });
  await permanent.flush();
  assert.deepEqual((await readdir(permanentDir)).filter((name) => name.endsWith(".json") || name.endsWith(".claim")), []);

  const retryDir = await mkdtemp(join(tmpdir(), "ogmem-retry-"));
  const retry = createTraceSpool({ directory: retryDir, limit: 2, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  await retry.enqueue({ request_id: "b", session_id: "s", timestamp: "2026-01-01T00:00:00Z", event: { event_type: "session_start" } });
  await retry.flush();
  assert.equal((await readdir(retryDir)).filter((name) => name.endsWith(".json")).length, 1);
});

test("trace spool retains authentication failures and missing acknowledgements", async () => {
  for (const [index, response] of [
    () => new Response("unauthorized", { status: 401 }),
    () => new Response("", { status: 200 }),
  ].entries()) {
    const dir = await mkdtemp(join(tmpdir(), "ogmem-trace-contract-"));
    const client = createHttpClient(config, async () => response());
    const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: client.trace, logger: { warn() {} } });
    await spool.enqueue({ request_id: `retry-${index}`, session_id: "s", timestamp: "2026-01-01T00:00:00Z", event: { event_type: "session_start" } });
    await spool.flush();
    spool.close();
    assert.equal((await readdir(dir)).filter((name) => name.endsWith(".json")).length, 1);
  }
});

test("spool replays FIFO, skips corruption, and deduplicates concurrent IDs", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-fifo-"));
  const sent = [];
  const spool = createTraceSpool({ directory: dir, limit: 20, ttlSeconds: 3600, send: async (event) => sent.push(event.request_id), logger: { warn() {} } });
  const envelope = (id, timestamp) => ({ request_id: id, session_id: "s", timestamp, event: { event_type: "user_message", prompt: id } });
  await Promise.all(Array.from({ length: 20 }, () => spool.enqueue(envelope("same", "2026-01-01T00:00:00Z"))));
  await spool.flush();
  assert.deepEqual(sent, ["same"]);

  const offlineDir = await mkdtemp(join(tmpdir(), "ogmem-order-"));
  const offline = createTraceSpool({ directory: offlineDir, limit: 20, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  await offline.enqueue(envelope("later", "2026-01-01T00:00:02Z"));
  await offline.enqueue(envelope("earlier", "2026-01-01T00:00:01Z"));
  await offline.flush();
  offline.close();
  await writeFile(join(offlineDir, "000-corrupt.json"), "not-json");
  const replayed = [];
  const replay = createTraceSpool({ directory: offlineDir, limit: 20, ttlSeconds: 3600, send: async (event) => replayed.push(event.request_id), logger: { warn() {} } });
  await replay.drain();
  assert.deepEqual(replayed, ["later", "earlier"]);
});

test("spool removes crash-orphaned temporary files", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-temp-"));
  const { writeFile } = await import("node:fs/promises");
  const stale = join(dir, "event.json.00000000-0000-4000-8000-000000000000.tmp");
  const fresh = join(dir, "event.json.11111111-1111-4111-8111-111111111111.tmp");
  await writeFile(stale, "redacted but orphaned");
  await writeFile(fresh, "active writer");
  const old = new Date(Date.now() - 10 * 60 * 1000);
  await utimes(stale, old, old);
  const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: async () => {}, logger: { warn() {} } });
  await spool.drain();
  assert.deepEqual(await readdir(dir), ["event.json.11111111-1111-4111-8111-111111111111.tmp"]);
});

test("spool promotes and delivers an interrupted hard-link publication claim", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-publication-claim-"));
  const envelope = { request_id: "claim-event", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  const key = createHash("sha256").update(envelope.request_id).digest("hex");
  await writeFile(join(dir, `${key}.claim`), JSON.stringify(envelope));
  const sent = [];
  const spool = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 3600, send: async (value) => sent.push(value), logger: { warn() {} } });
  await spool.flush();
  assert.deepEqual(sent, [envelope]);
  assert.equal((await readdir(dir)).some((name) => name.endsWith(".json") || name.endsWith(".claim")), false);
  await spool.close();
  await rm(dir, { recursive: true, force: true });
});

test("drain does not promote a publication claim while its writer is active", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-active-publication-"));
  try {
    let release;
    let claimLinked;
    const claimReady = new Promise((resolve) => { claimLinked = resolve; });
    const gate = new Promise((resolve) => { release = resolve; });
    let links = 0;
    const gatedLink = async (source, target) => {
      await link(source, target);
      links += 1;
      if (links === 1) { claimLinked(); await gate; }
    };
    const sent = [];
    const writer = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 3600, send: async (value) => sent.push(`writer:${value.request_id}`), linkFile: gatedLink, logger: { warn() {} } });
    const drainer = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 3600, send: async (value) => sent.push(`drainer:${value.request_id}`), logger: { warn() {} } });
    const publishing = writer.enqueue({ request_id: "race-id", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
    await claimReady;
    const draining = drainer.flush();
    await new Promise((resolve) => setTimeout(resolve, 30));
    release();
    await Promise.all([publishing, draining, writer.flush(), drainer.flush()]);
    assert.equal(sent.length, 1);
    assert.match(sent[0], /:race-id$/);
    await writer.close();
    await drainer.close();
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("spool immediately recovers a malformed zero-pid delivery claim", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-zero-pid-"));
  const envelope = { request_id: "zero-pid", session_id: "s", timestamp: "2026-01-01T00:00:00Z", event: { event_type: "session_start" } };
  await writeFile(join(dir, "orphan.json.sending.0.00000000-0000-4000-8000-000000000000"), JSON.stringify(envelope));
  const sent = [];
  const spool = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 3600, send: async (value) => sent.push(value.request_id), logger: { warn() {} } });
  await spool.drain();
  assert.deepEqual(sent, ["zero-pid"]);
});

test("spool processes entries incrementally and drops oversized files before reading", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-oversized-spool-"));
  const { writeFile } = await import("node:fs/promises");
  await writeFile(join(dir, "0000000000000-000001_first.json"), JSON.stringify({
    request_id: "first", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" },
  }));
  const oversizedName = "9999999999999-000002_oversized.json";
  await writeFile(join(dir, oversizedName), "x".repeat(64 * 1024 + 1));
  let sends = 0;
  let oversizedExistedAtFirstSend = false;
  const warnings = [];
  const spool = createTraceSpool({
    directory: dir, limit: 2, ttlSeconds: 3600,
    send: async () => { sends += 1; oversizedExistedAtFirstSend = (await readdir(dir)).includes(oversizedName); },
    logger: { warn: (message) => warnings.push(message) },
  });
  await spool.drain();
  assert.equal(sends, 1);
  assert.equal(oversizedExistedAtFirstSend, true);
  assert.deepEqual(await readdir(dir), []);
  assert.match(warnings[0], /exceeds 65536 bytes/);
});

test("trace backoff uses bounded equal jitter", () => {
  assert.equal(traceBackoffDelay(1, () => 0), 250);
  assert.equal(traceBackoffDelay(1, () => 0.999), 499);
  assert.equal(traceBackoffDelay(20, () => 0), 15000);
  assert.ok(traceBackoffDelay(20, () => 0.999) < 30000);
});

test("spool enforces capacity and TTL before replay", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-bounds-"));
  const offline = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const envelope = (id) => ({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "user_message" } });
  await offline.enqueue(envelope("a"));
  await offline.enqueue(envelope("b"));
  await offline.enqueue(envelope("c"));
  await offline.flush();
  let files = (await readdir(dir)).filter((name) => name.endsWith(".json")).sort();
  assert.equal(files.length, 2);
  const persisted = await Promise.all(files.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  assert.deepEqual(new Set(persisted.map((item) => item.request_id)), new Set(["b", "c"]));

  const expired = files[persisted.findIndex((item) => item.request_id === "b")];
  const old = new Date(Date.now() - 120000);
  await utimes(join(dir, expired), old, old);
  assert.ok((await stat(join(dir, expired))).mtimeMs < Date.now() - 60000);
  const replayed = [];
  const replay = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send: async (event) => replayed.push(event.request_id), logger: { warn() {} } });
  await replay.drain();
  assert.deepEqual(replayed, ["c"]);
});

test("TTL pruning does not count expired middle entries against capacity", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-ttl-middle-"));
  const offline = createTraceSpool({ directory: dir, limit: 3, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const item = (id) => ({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  for (const id of ["a", "b", "c"]) await offline.enqueue(item(id));
  await offline.flush();
  offline.close();
  const files = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  const envelopes = await Promise.all(files.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  const middle = files[envelopes.findIndex((entry) => entry.request_id === "b")];
  const old = new Date(Date.now() - 120000);
  await utimes(join(dir, middle), old, old);
  const sent = [];
  const replay = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send: async (entry) => sent.push(entry.request_id), logger: { warn() {} } });
  await replay.drain();
  assert.deepEqual(sent, ["a", "c"]);
});

test("capacity eviction removes the persistent ID claim", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-claim-capacity-"));
  const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const item = (id) => ({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  for (const id of ["a", "b", "c", "d"]) await spool.enqueue(item(id));
  await spool.flush();
  spool.close();
  const names = await readdir(dir);
  const jsonNames = names.filter((name) => name.endsWith(".json"));
  assert.equal(jsonNames.length, 2);
  assert.equal(names.filter((name) => name.endsWith(".claim")).length, 2);
  const retained = await Promise.all(jsonNames.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  assert.deepEqual(new Set(retained.map((item) => item.request_id)), new Set(["c", "d"]));
});

test("an expired orphan publication claim does not drop the first retry", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-orphan-publication-"));
  const { createHash } = await import("node:crypto");
  const id = "retry-after-orphan";
  const key = createHash("sha256").update(id).digest("hex");
  const claim = join(dir, `${key}.claim`);
  await (await import("node:fs/promises")).writeFile(claim, "orphan");
  const old = new Date(Date.now() - 6 * 60 * 1000);
  await utimes(claim, old, old);
  const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  await spool.enqueue({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await spool.flush();
  spool.close();
  assert.equal((await readdir(dir)).filter((name) => name.endsWith(".json")).length, 1);
});

test("spool FIFO does not depend on filesystem timestamp precision", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-fifo-equal-mtime-"));
  const offline = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const item = (id) => ({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await offline.enqueue(item("a"));
  await offline.enqueue(item("z"));
  await offline.flush();
  offline.close();
  const files = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  const same = new Date();
  await Promise.all(files.map((name) => utimes(join(dir, name), same, same)));
  const sent = [];
  const replay = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 3600, send: async (envelope) => sent.push(envelope.request_id), logger: { warn() {} } });
  await replay.drain();
  assert.deepEqual(sent, ["a", "z"]);
});

test("persistent sequence preserves FIFO and capacity across wall-clock rollback", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-sequence-rollback-"));
  const offline = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const item = (id, timestamp) => ({ request_id: id, session_id: "s", timestamp, event: { event_type: "session_start" } });
  await offline.enqueue(item("old", "2030-01-01T00:00:00Z"));
  await offline.enqueue(item("middle", "2020-01-01T00:00:00Z"));
  await offline.enqueue(item("new", "2010-01-01T00:00:00Z"));
  await offline.flush();
  offline.close();
  const retainedFiles = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  const retained = await Promise.all(retainedFiles.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  assert.deepEqual(new Set(retained.map((entry) => entry.request_id)), new Set(["middle", "new"]));
  const sent = [];
  const replay = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 3600, send: async (entry) => sent.push(entry.request_id), logger: { warn() {} } });
  await replay.drain();
  assert.deepEqual(sent, ["middle", "new"]);
});

test("sequence recovery bounds metadata and includes abandoned delivery claims", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-sequence-recovery-"));
  const { writeFile } = await import("node:fs/promises");
  await writeFile(join(dir, ".sequence"), "9".repeat(1024));
  const abandonedName = `${"5".padStart(24, "0")}_${"a".repeat(64)}.json.sending.2147483647.00000000-0000-4000-8000-000000000000`;
  await writeFile(join(dir, abandonedName), JSON.stringify({ request_id: "abandoned", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } }));
  const warnings = [];
  const spool = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 3600, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn: (message) => warnings.push(message) } });
  await spool.enqueue({ request_id: "next", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await spool.flush();
  spool.close();
  const names = await readdir(dir);
  assert.ok(names.some((name) => name.startsWith("000000000000000000000006_")));
  assert.equal(await readFile(join(dir, ".sequence"), "utf8"), "6");
  assert.match(warnings.join("\n"), /recovered invalid sequence metadata/);
  assert.doesNotMatch(warnings.join("\n"), /999999/);
});

test("sequence rejects wider metadata and guards the fixed-width maximum", async () => {
  const { writeFile } = await import("node:fs/promises");
  const item = { request_id: "bounded-sequence", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  const wideDir = await mkdtemp(join(tmpdir(), "ogmem-sequence-wide-"));
  await writeFile(join(wideDir, ".sequence"), "1".repeat(25));
  const warnings = [];
  const recovered = createTraceSpool({ directory: wideDir, limit: 2, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn: (message) => warnings.push(message) } });
  await recovered.enqueue(item);
  recovered.close();
  assert.equal(await readFile(join(wideDir, ".sequence"), "utf8"), "1");
  assert.match(warnings.join("\n"), /recovered invalid sequence metadata/);

  const maximumDir = await mkdtemp(join(tmpdir(), "ogmem-sequence-maximum-"));
  const maximum = "9".repeat(24);
  await writeFile(join(maximumDir, ".sequence"), maximum);
  await writeFile(join(maximumDir, `${maximum}_${"a".repeat(64)}.json`), JSON.stringify(item));
  const exhausted = createTraceSpool({ directory: maximumDir, limit: 2, ttlSeconds: 60, send: async () => {}, logger: { warn() {} } });
  await assert.rejects(exhausted.enqueue({ ...item, request_id: "next" }), /sequence space exhausted/);
});

test("publication failures clean sequence and envelope temporary files immediately", async () => {
  const item = { request_id: "cleanup", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  const failure = (code) => async () => { const error = new Error(code); error.code = code; throw error; };

  const sequenceDir = await mkdtemp(join(tmpdir(), "ogmem-sequence-temp-cleanup-"));
  const sequenceFailure = createTraceSpool({ directory: sequenceDir, limit: 2, ttlSeconds: 60, sequenceRename: failure("EIO"), send: async () => {}, logger: { warn() {} } });
  await assert.rejects(sequenceFailure.enqueue(item), /EIO/);
  assert.equal((await readdir(sequenceDir)).some((name) => name.startsWith(".sequence.") && name.endsWith(".tmp")), false);

  for (const code of ["EEXIST", "EIO"]) {
    const dir = await mkdtemp(join(tmpdir(), `ogmem-envelope-temp-cleanup-${code}-`));
    const spool = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, linkFile: failure(code), send: async () => {}, logger: { warn() {} } });
    if (code === "EEXIST") await spool.enqueue(item);
    else await assert.rejects(spool.enqueue(item), /EIO/);
    assert.equal((await readdir(dir)).some((name) => name.endsWith(".tmp")), false);
  }
});

test("retryable trace failures and recovery are observable without content", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-observable-"));
  const warnings = [];
  const infos = [];
  let attempts = 0;
  const spool = createTraceSpool({
    directory: dir, limit: 2, ttlSeconds: 60, random: () => 0,
    send: async () => { attempts += 1; if (attempts === 1) throw new HttpError("offline secret", { status: 503, retryable: true }); },
    logger: { warn: (message) => warnings.push(message), info: (message) => infos.push(message) },
  });
  await spool.enqueue({ request_id: "observable", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "user_message", prompt: "private" } });
  await spool.flush();
  assert.match(warnings.join("\n"), /status=503.*failures=1.*retryMs=250.*queued=1/);
  assert.doesNotMatch(warnings.join("\n"), /offline secret|private/);
  await new Promise((resolve) => setTimeout(resolve, 275));
  await spool.drain();
  assert.match(infos.join("\n"), /recovered after 1 failed attempt/);
});

test("closing the spool cancels scheduled retry and rejects new entries", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-close-"));
  let attempts = 0;
  const spool = createTraceSpool({
    directory: dir, limit: 2, ttlSeconds: 60, random: () => 0,
    send: async () => { attempts += 1; throw new HttpError("offline", { status: 503, retryable: true }); }, logger: { warn() {} },
  });
  const envelope = { request_id: "close", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  await spool.enqueue(envelope);
  await spool.flush();
  assert.equal(attempts, 1);
  spool.close();
  await new Promise((resolve) => setTimeout(resolve, 275));
  assert.equal(attempts, 1);
  await assert.rejects(spool.enqueue({ ...envelope, request_id: "after-close" }), /closed/);
});

test("two spool instances claim a shared entry for one delivery", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-claim-"));
  const offline = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  await offline.enqueue({ request_id: "shared", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await offline.flush();
  offline.close();
  let sends = 0;
  const createReplay = () => createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send: async () => { sends += 1; await new Promise((resolve) => setImmediate(resolve)); }, logger: { warn() {} } });
  const first = createReplay();
  const second = createReplay();
  await Promise.all([first.drain(), second.drain()]);
  assert.equal(sends, 1);
});

test("enqueue deduplicates an ID while another instance is delivering it", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-inflight-dedupe-"));
  let release;
  let started;
  const startedPromise = new Promise((resolve) => { started = resolve; });
  const gate = new Promise((resolve) => { release = resolve; });
  let sends = 0;
  const send = async () => { sends += 1; started(); await gate; };
  const first = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send, logger: { warn() {} } });
  const second = createTraceSpool({ directory: dir, limit: 2, ttlSeconds: 60, send, logger: { warn() {} } });
  const item = { request_id: "inflight", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };

  await first.enqueue(item);
  await startedPromise;
  await second.enqueue(item);
  release();
  await Promise.all([first.flush(), second.flush()]);

  assert.equal(sends, 1);
  assert.equal((await readdir(dir)).filter((name) => name.includes(".json")).length, 0);
});

test("busy cross-process drain claim schedules delivery for a later enqueue", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-wakeup-"));
  const seed = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const envelope = (id) => ({ request_id: id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await seed.enqueue(envelope("old"));
  await seed.flush();
  seed.close();

  let release;
  let started;
  const startedPromise = new Promise((resolve) => { started = resolve; });
  const gate = new Promise((resolve) => { release = resolve; });
  const sent = [];
  const first = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async (event) => { sent.push(event.request_id); started(); await gate; }, logger: { warn() {} } });
  const second = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, random: () => 0, send: async (event) => { sent.push(event.request_id); }, logger: { warn() {} } });
  const draining = first.drain();
  await startedPromise;
  await second.enqueue(envelope("new"));
  release();
  await draining;
  await new Promise((resolve) => setTimeout(resolve, 175));
  await second.flush();
  assert.deepEqual(sent, ["old", "new"]);
});

test("transient spool filesystem failure retries without a new enqueue", async () => {
  const parent = await mkdtemp(join(tmpdir(), "ogmem-drain-fs-retry-"));
  const dir = join(parent, "spool");
  const { mkdir, rm, writeFile } = await import("node:fs/promises");
  await writeFile(dir, "temporarily not a directory");
  const sent = [];
  const warnings = [];
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    send: async (event) => sent.push(event.request_id),
    logger: { warn: (message) => warnings.push(message) },
  });

  await spool.drain();
  await rm(dir, { force: true });
  await mkdir(dir);
  await writeFile(join(dir, "0000000000000-000001_recovered.json"), JSON.stringify({
    request_id: "recovered", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" },
  }));
  await new Promise((resolve) => setTimeout(resolve, 150));

  assert.deepEqual(sent, ["recovered"]);
  assert.match(warnings.join("\n"), /retry scheduled in 100ms/);
  spool.close();
});

test("two spool instances deduplicate the same request ID before persistence", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-cross-dedupe-"));
  const createOffline = () => createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} } });
  const first = createOffline();
  const second = createOffline();
  const envelope = { request_id: "same-cross-process", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  await Promise.all([first.enqueue(envelope), second.enqueue(envelope)]);
  await Promise.all([first.flush(), second.flush()]);
  first.close();
  second.close();
  assert.equal((await readdir(dir)).filter((name) => name.endsWith(".json")).length, 1);
  let sends = 0;
  const replay = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async () => { sends += 1; }, logger: { warn() {} } });
  await replay.drain();
  assert.equal(sends, 1);
});

test("batch enqueue persists a recovered transcript with one bounded capacity pass", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-batch-spool-"));
  let linkCalls = 0;
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60,
    linkFile: async (...args) => { linkCalls += 1; return link(...args); },
    send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} },
  });
  const envelopes = Array.from({ length: 20 }, (_, index) => ({
    request_id: `batch-${index}`, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "assistant_message" },
  }));
  await spool.enqueueMany(envelopes);
  await spool.flush();
  spool.close();
  const files = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  assert.equal(files.length, 5);
  assert.equal(linkCalls, 10);
  const retained = await Promise.all(files.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  assert.deepEqual(retained.map((item) => item.request_id).sort(), ["batch-15", "batch-16", "batch-17", "batch-18", "batch-19"]);
});

test("partial batch publication retries every request ID without duplicates", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-batch-partial-"));
  let linkCalls = 0;
  let interrupt = true;
  const spool = createTraceSpool({
    directory: dir, limit: 10, ttlSeconds: 60,
    linkFile: async (...args) => {
      linkCalls += 1;
      if (interrupt && linkCalls === 4) { interrupt = false; throw new Error("simulated partial batch interruption"); }
      return link(...args);
    },
    send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} },
  });
  const envelopes = ["one", "two", "three"].map((request_id) => ({
    request_id, session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "assistant_message" },
  }));
  await assert.rejects(spool.enqueueMany(envelopes), /partial batch interruption/);
  await spool.enqueueMany(envelopes);
  await spool.flush();
  spool.close();
  let names = await readdir(dir);
  assert.equal(names.filter((name) => name.endsWith(".json")).length, 3);
  assert.equal(names.some((name) => name.endsWith(".tmp")), false);

  const sent = [];
  const replay = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 60, send: async (envelope) => sent.push(envelope.request_id), logger: { warn() {} } });
  await replay.drain();
  replay.close();
  assert.deepEqual(sent.sort(), ["one", "three", "two"]);
  names = await readdir(dir);
  assert.equal(names.some((name) => name.endsWith(".json") || name.endsWith(".claim") || name.endsWith(".tmp")), false);
});

test("spool falls back when hard links are unsupported", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-no-hardlink-"));
  const unsupported = async () => { const error = new Error("hard links unsupported"); error.code = "ENOTSUP"; throw error; };
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, linkFile: unsupported,
    send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} },
  });
  await spool.enqueue({ request_id: "fallback", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await spool.flush();
  spool.close();
  const files = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  assert.equal(files.length, 1);
  assert.equal(JSON.parse(await readFile(join(dir, files[0]), "utf8")).request_id, "fallback");
  assert.equal((await readdir(dir)).some((name) => name.endsWith(".claim")), false);
});

test("unsupported-hardlink publication retries after an interrupted atomic rename", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-no-hardlink-crash-"));
  const unsupported = async () => { const error = new Error("hard links unsupported"); error.code = "ENOTSUP"; throw error; };
  let interrupt = true;
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, linkFile: unsupported,
    publishRename: async (...args) => {
      if (interrupt) { interrupt = false; throw new Error("simulated process interruption before publication"); }
      return rename(...args);
    },
    send: async () => { throw new HttpError("offline", { retryable: true }); }, logger: { warn() {} },
  });
  const envelope = { request_id: "fallback-after-crash", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } };
  await assert.rejects(spool.enqueue(envelope), /simulated process interruption/);
  assert.equal((await readdir(dir)).some((name) => name.endsWith(".claim")), false);
  await spool.enqueue(envelope);
  await spool.flush();
  spool.close();
  const files = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  assert.equal(files.length, 1);
  assert.equal(JSON.parse(await readFile(join(dir, files[0]), "utf8")).request_id, envelope.request_id);
});

test("retry rollback preserves the only delivery claim when rename returns EPERM", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-no-hardlink-rollback-eperm-"));
  const unsupported = async () => { const error = new Error("hard links unsupported"); error.code = "ENOTSUP"; throw error; };
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, linkFile: unsupported,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json")) {
        throw Object.assign(new Error("transient rollback contention"), { code: "EPERM" });
      }
      return rename(source, target);
    },
    send: async () => { throw new HttpError("offline", { retryable: true }); },
    logger: { warn() {} }, random: () => 0,
  });
  await spool.enqueue({ request_id: "rollback-eperm", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await spool.flush();
  spool.close();
  const sending = (await readdir(dir)).find((name) => name.includes(".sending."));
  assert.ok(sending);

  const abandoned = sending.replace(/\.sending\.\d+\./, ".sending.2147483647.");
  await rename(join(dir, sending), join(dir, abandoned));
  const sent = [];
  const replay = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async (envelope) => sent.push(envelope.request_id), logger: { warn() {} } });
  await replay.drain();
  replay.close();
  assert.deepEqual(sent, ["rollback-eperm"]);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
});

test("initial delivery-claim EPERM retries without another enqueue or flush", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-claim-eperm-retry-"));
  let failures = 0;
  let delivered;
  let deadline;
  const deliveredPromise = new Promise((resolve, reject) => {
    deadline = setTimeout(() => reject(new Error("claim retry timed out")), 1000);
    delivered = (requestId) => { clearTimeout(deadline); resolve(requestId); };
  });
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (!source.includes(".sending.") && target.includes(".sending.") && failures < 3) {
        failures += 1;
        throw Object.assign(new Error("transient claim contention"), { code: "EPERM" });
      }
      return rename(source, target);
    },
    send: async (envelope) => { delivered(envelope.request_id); }, logger: { warn() {} },
  });
  await spool.enqueue({ request_id: "claim-retry", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  assert.equal(await deliveredPromise, "claim-retry");
  spool.close();
});

test("retry rollback EPERM resumes on the configured backoff in the same spool", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-rollback-eperm-retry-"));
  let rollbackFailures = 0;
  let sends = 0;
  let delivered;
  let deadline;
  const deliveredPromise = new Promise((resolve, reject) => {
    deadline = setTimeout(() => reject(new Error("rollback retry timed out")), 1200);
    delivered = (requestId) => { clearTimeout(deadline); resolve(requestId); };
  });
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json") && rollbackFailures < 3) {
        rollbackFailures += 1;
        throw Object.assign(new Error("transient rollback contention"), { code: "EPERM" });
      }
      return rename(source, target);
    },
    send: async (envelope) => {
      sends += 1;
      if (sends === 1) throw new HttpError("offline", { retryable: true });
      delivered(envelope.request_id);
    },
    logger: { warn() {} },
  });
  await spool.enqueue({ request_id: "rollback-retry", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  assert.equal(await deliveredPromise, "rollback-retry");
  await spool.flush();
  assert.equal(rollbackFailures, 3);
  assert.equal(sends, 2);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
  spool.close();
});

test("retry rollback EIO resumes on the configured backoff in the same spool", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-rollback-eio-retry-"));
  let rollbackFailures = 0;
  let sends = 0;
  let delivered;
  let deadline;
  const deliveredPromise = new Promise((resolve, reject) => {
    deadline = setTimeout(() => reject(new Error("EIO rollback retry timed out")), 1200);
    delivered = (requestId) => { clearTimeout(deadline); resolve(requestId); };
  });
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json") && rollbackFailures === 0) {
        rollbackFailures += 1;
        throw Object.assign(new Error("transient rollback I/O failure"), { code: "EIO" });
      }
      return rename(source, target);
    },
    send: async (envelope) => {
      sends += 1;
      if (sends === 1) throw new HttpError("offline", { retryable: true });
      delivered(envelope.request_id);
    },
    logger: { warn() {} },
  });
  await spool.enqueue({ request_id: "rollback-eio-retry", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  assert.equal(await deliveredPromise, "rollback-eio-retry");
  await spool.flush();
  assert.equal(rollbackFailures, 1);
  assert.equal(sends, 2);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
  spool.close();
});

test("rollback EACCES reconciles an existing target and continues retry", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-rollback-eacces-target-"));
  let rollbackDenials = 0;
  let sends = 0;
  let delivered;
  let deadline;
  const deliveredPromise = new Promise((resolve, reject) => {
    deadline = setTimeout(() => reject(new Error("EACCES rollback retry timed out")), 1200);
    delivered = (requestId) => { clearTimeout(deadline); resolve(requestId); };
  });
  const spool = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json") && rollbackDenials < 3) {
        if (rollbackDenials === 0) await writeFile(target, await readFile(source));
        rollbackDenials += 1;
        throw Object.assign(new Error("target sharing contention"), { code: "EACCES" });
      }
      return rename(source, target);
    },
    send: async (envelope) => {
      sends += 1;
      if (sends === 1) throw new HttpError("offline", { retryable: true });
      delivered(envelope.request_id);
    },
    logger: { warn() {} },
  });
  await spool.enqueue({ request_id: "rollback-eacces", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  assert.equal(await deliveredPromise, "rollback-eacces");
  await spool.flush();
  assert.equal(rollbackDenials, 3);
  assert.equal(sends, 2);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
  spool.close();
});

test("same-process spool replacement recovers claims relinquished by close", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-spool-same-pid-reload-"));
  const first = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json")) {
        throw Object.assign(new Error("persistent rollback contention"), { code: "EPERM" });
      }
      return rename(source, target);
    },
    send: async () => { throw new HttpError("offline", { retryable: true }); },
    logger: { warn() {} },
  });
  await first.enqueue({ request_id: "same-pid-reload", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await first.flush();
  first.close();
  assert.equal((await readdir(dir)).some((name) => name.includes(`.sending.${process.pid}.`)), true);

  const sent = [];
  const replacement = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async (envelope) => sent.push(envelope.request_id), logger: { warn() {} } });
  await replacement.drain();
  replacement.close();
  assert.deepEqual(sent, ["same-pid-reload"]);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
});

test("same-process replacement recovers an in-flight claim after close", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-spool-close-in-flight-"));
  let releaseSend;
  let sendStarted;
  const started = new Promise((resolve) => { sendStarted = resolve; });
  const blocked = new Promise((resolve) => { releaseSend = resolve; });
  const first = createTraceSpool({
    directory: dir, limit: 5, ttlSeconds: 60, random: () => 0,
    claimRename: async (source, target) => {
      if (source.includes(".sending.") && target.endsWith(".json")) {
        throw Object.assign(new Error("persistent rollback contention"), { code: "EPERM" });
      }
      return rename(source, target);
    },
    send: async () => {
      sendStarted();
      await blocked;
      throw new HttpError("offline", { retryable: true });
    },
    logger: { warn() {} },
  });
  await first.enqueue({ request_id: "close-in-flight", session_id: "s", timestamp: new Date().toISOString(), event: { event_type: "session_start" } });
  await started;
  first.close();
  releaseSend();
  await first.flush();

  const sent = [];
  const replacement = createTraceSpool({ directory: dir, limit: 5, ttlSeconds: 60, send: async (envelope) => sent.push(envelope.request_id), logger: { warn() {} } });
  await replacement.drain();
  replacement.close();
  assert.deepEqual(sent, ["close-in-flight"]);
  assert.equal((await readdir(dir)).some((name) => name.includes(".sending.") || name.endsWith(".json")), false);
});
