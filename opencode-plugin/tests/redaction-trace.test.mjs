import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { redact } from "../lib/redaction.mjs";
import { createTraceAdapter, stableEventId } from "../lib/trace-adapter.mjs";
import { createTraceSpool } from "../lib/trace-spool.mjs";

test("redaction removes secrets before persistence", () => {
  const value = redact({ authorization: "Bearer secret", body: "email person@example.com password=hunter2 AKIA1234567890ABCDEF eyJabcdefghij.abcdefghij.abcdefghij redis://user:pass@host/0", apiKey: "nope", count: 1n });
  assert.equal(value.authorization, "[REDACTED]");
  assert.match(value.body, /REDACTED_EMAIL/);
  assert.equal(value.apiKey, "[REDACTED]");
  assert.doesNotMatch(value.body, /hunter2|AKIA|eyJ|user:pass/);
  assert.equal(value.count, "1");
});

test("redaction removes secrets embedded in serialized JSON", () => {
  const value = redact(JSON.stringify({
    apiKey: "custom-secret-value",
    nested: { password: "pw-secret", authorization: ["Basic credential", "Bearer other-credential"] },
    token: { access: "access-credential", refresh: "refresh-credential" },
  }));
  assert.doesNotMatch(value, /custom-secret-value|pw-secret|credential/);
  assert.equal(JSON.parse(value).apiKey, "[REDACTED]");
  assert.equal(JSON.parse(value).nested.password, "[REDACTED]");
  assert.equal(JSON.parse(value).nested.authorization, "[REDACTED]");
  assert.equal(JSON.parse(value).token, "[REDACTED]");
});

test("serialized JSON redaction preserves non-secret key names", () => {
  const value = redact(JSON.stringify({ tokenizer: "qwen", secretary: "Alice", cookiePolicy: "strict", authorizationStatus: "pending", input_tokens: 42 }));
  assert.deepEqual(JSON.parse(value), { tokenizer: "qwen", secretary: "Alice", cookiePolicy: "strict", authorizationStatus: "pending", input_tokens: 42 });
});

test("structured redaction recognizes suffixed credential keys", () => {
  const value = redact({ APIKey: "upper-api-credential", openAIApiKey: "camel-api-credential", githubToken: "github-credential", sessionToken: "session-credential", dbPassword: "db-credential", passwordHash: "hash-credential", password_digest: "digest-credential", api_key_value: "api-credential", token_value: "token-credential", signingSecret: "signing-credential", sessionCookie: "cookie-credential", "Proxy-Authorization": "auth-credential", proxyAuthorizationHeader: "basic-credential", setCookieHeader: "set-cookie-credential" });
  assert.deepEqual(value, { APIKey: "[REDACTED]", openAIApiKey: "[REDACTED]", githubToken: "[REDACTED]", sessionToken: "[REDACTED]", dbPassword: "[REDACTED]", passwordHash: "[REDACTED]", password_digest: "[REDACTED]", api_key_value: "[REDACTED]", token_value: "[REDACTED]", signingSecret: "[REDACTED]", sessionCookie: "[REDACTED]", "Proxy-Authorization": "[REDACTED]", proxyAuthorizationHeader: "[REDACTED]", setCookieHeader: "[REDACTED]" });
});

test("redaction removes secrets from double-encoded JSON", () => {
  const value = redact(JSON.stringify(JSON.stringify({ apiKey: "custom-secret-value", password: "pw-secret" })));
  assert.doesNotMatch(value, /custom-secret-value|pw-secret/);
  assert.deepEqual(JSON.parse(JSON.parse(value)), { apiKey: "[REDACTED]", password: "[REDACTED]" });
});

test("redaction removes secrets from bounded multiply-encoded JSON", () => {
  const value = redact(JSON.stringify(JSON.stringify(JSON.stringify({ apiKey: "triple-secret" }))));
  assert.doesNotMatch(value, /triple-secret/);
  assert.deepEqual(JSON.parse(JSON.parse(JSON.parse(value))), { apiKey: "[REDACTED]" });
});

test("multiply-encoded JSON fails closed at the decode depth limit", () => {
  let value = JSON.stringify({ apiKey: "deep-secret" });
  for (let index = 0; index < 6; index += 1) value = JSON.stringify(value);
  const redacted = redact(value);
  assert.doesNotMatch(redacted, /deep-secret/);
  assert.match(redacted, /REDACTED_NESTED_JSON/);
});

test("oversized JSON is bounded before parsing", () => {
  const value = `{\"safe\":\"${"x".repeat(14100)}\"}`;
  assert.equal(redact(value), "[TRUNCATED_JSON]");
});

test("oversized whitespace-prefixed JSON is bounded before trimming", () => {
  const value = `${" ".repeat(14100)}{\"safe\":true}`;
  assert.equal(redact(value).length, 10000);
});

test("redaction bounds shared-reference graphs without invoking getters", () => {
  const shared = { value: "safe" };
  const graph = Object.fromEntries(Array.from({ length: 100 }, (_, index) => [`k${index}`, shared]));
  Object.defineProperty(graph, "danger", { enumerable: true, get() { throw new Error("getter invoked"); } });
  const result = redact(graph);
  assert.equal(result.danger, "[GETTER]");
  assert.equal(result.k1, "[REPEATED_REFERENCE]");
});

test("private keys are redacted before string truncation", () => {
  const key = `-----BEGIN PRIVATE KEY-----\n${"A".repeat(12000)}\n-----END PRIVATE KEY-----`;
  assert.equal(redact(key), "[REDACTED_PRIVATE_KEY]");
});

test("secrets crossing the truncation boundary are redacted before slicing", () => {
  const prefix = "x".repeat(9995);
  const value = redact(`${prefix} mail=alice@example.com token=sk-proj-abcdefghijklmnop`);
  assert.doesNotMatch(value, /alice@|sk-proj-|sk-p$/);
});

test("IPv6 addresses are redacted without replacing ordinary colon text", () => {
  const value = redact("full 2001:db8:0:0:0:0:0:1 compressed 2001:db8::1 mapped ::ffff:192.0.2.1 clock 12:30");
  assert.doesNotMatch(value, /2001:db8|::ffff/);
  assert.match(value, /clock 12:30/);
});

test("stable event IDs dedupe the same terminal event", () => {
  assert.equal(stableEventId("tool_success", "s", "c", "success"), stableEventId("tool_success", "s", "c", "success"));
  assert.notEqual(stableEventId("tool_success", "s", "c", "success"), stableEventId("tool_failure", "s", "c", "error"));
  assert.notEqual(stableEventId("tool_success", "parent", "c", "success"), stableEventId("tool_success", "child", "c", "success"));
});

test("metadata-only trace omits every content-bearing canonical field from disk", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ogmem-trace-"));
  const spool = createTraceSpool({ directory: dir, limit: 10, ttlSeconds: 3600, send: async () => { const error = new Error("offline"); error.retryable = true; throw error; }, logger: { warn() {} } });
  const trace = createTraceAdapter({ config: { traceCaptureContent: false, accountId: "a", userId: "u", agentId: "opencode" }, spool });
  const events = [
    ["user_message", "prompt"], ["assistant_message", "assistant_message"], ["tool_start", "tool_input"],
    ["tool_success", "tool_response"], ["tool_failure", "error"], ["turn_end", "assistant_message"], ["session_error", "error"],
  ];
  for (const [index, [type]] of events.entries()) {
    await trace.emit(type, "s", { objectId: `id-${index}`, terminalState: type.includes("failure") || type === "session_error" ? "error" : "success", content: `sentinel-${type} person@example.com`, metadata: { tool: "read" } });
  }
  await trace.flush();
  const names = (await readdir(dir)).filter((name) => name.endsWith(".json"));
  assert.equal(names.length, events.length);
  const envelopes = await Promise.all(names.map((name) => readFile(join(dir, name), "utf8").then(JSON.parse)));
  assert.doesNotMatch(JSON.stringify(envelopes), /sentinel-|person@example.com/);
  for (const [type, field] of events) {
    const envelope = envelopes.find((item) => item.event.event_type === type);
    assert.ok(envelope, type);
    assert.equal(field in envelope.event, false, `${type} must omit ${field}`);
  }
});

test("oversized trace content is bounded below the server limit", async () => {
  let saved;
  const trace = createTraceAdapter({
    config: { traceCaptureContent: true, accountId: "a", userId: "u", agentId: "opencode" },
    spool: { enqueue: async (value) => { saved = value; }, flush: async () => {} },
  });
  await trace.emit("tool_success", "s", { objectId: "c", content: Array.from({ length: 100 }, () => "x".repeat(8192)) });
  assert.ok(Buffer.byteLength(JSON.stringify(saved), "utf8") < 64 * 1024);
  assert.equal(saved.event.tool_response, "[TRUNCATED_EVENT]");
  assert.equal(saved.event.tool_name, "unknown_tool");
});

test("trace bounds structural and identity fields before spooling", async () => {
  let saved;
  const long = "password=hunter2 ".repeat(5000);
  const trace = createTraceAdapter({
    config: { traceCaptureContent: true, accountId: long, userId: long, agentId: long },
    spool: { enqueue: async (value) => { saved = value; }, flush: async () => {} },
  });
  assert.equal(await trace.emit("tool_success", long, { objectId: long, content: "ok", metadata: { tool: long } }), true);
  assert.ok(Buffer.byteLength(JSON.stringify(saved), "utf8") < 56 * 1024);
  assert.ok(saved.session_id.length <= 512);
  assert.ok(saved.event.tool_name.length <= 256);
  assert.ok(saved.event.tool_use_id.length <= 512);
  assert.doesNotMatch(JSON.stringify(saved), /hunter2/);
});

test("tool and turn events use canonical top-level fields", async () => {
  const saved = [];
  const trace = createTraceAdapter({ config: { traceCaptureContent: true, accountId: "a", userId: "u", agentId: "opencode" }, spool: { enqueue: async (value) => saved.push(value), flush: async () => {} } });
  await trace.emit("tool_start", "s", { objectId: "call", content: { path: "x" }, metadata: { tool: "read" } });
  await trace.emit("turn_end", "s", { objectId: "m2", terminalState: "idle", content: "done" });
  assert.equal(saved[0].event.tool_name, "read");
  assert.equal(saved[0].event.tool_use_id, "call");
  assert.deepEqual(saved[0].event.tool_input, { path: "x" });
  assert.equal(saved[1].event.assistant_message, "done");
});

test("extension metadata cannot overwrite the canonical Trace event type", async () => {
  let saved;
  const trace = createTraceAdapter({ config: { traceCaptureContent: false, accountId: "a", userId: "u", agentId: "opencode" }, spool: { enqueue: async (value) => { saved = value; }, flush: async () => {} } });
  await trace.emit("future_event", "s", { metadata: { event_type: "spoofed", extension_field: "kept" } });
  assert.equal(saved.event.event_type, "future_event");
  assert.equal(saved.event.extension_field, "kept");
});
