import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import test from "node:test";
import plugin, { resolveConfig, runCompactControlFlow } from "./index.js";

function host(config) {
  let factory;
  const hooks = [];
  plugin.register({
    pluginConfig: { mode: "remote", ...config },
    logger: { info() {}, warn() {}, error() {} },
    on: (name, handler) => hooks.push({ name, handler }),
    registerService() {},
    registerContextEngine: (_id, value) => { factory = value; },
  });
  return { engine: factory(), hooks };
}

test("capture/recall defaults, environment and explicit false precedence", () => {
  const names = ["OGMEM_AUTO_CAPTURE", "OGMEM_AUTO_RECALL"];
  const saved = names.map((name) => process.env[name]);
  try {
    for (const name of names) delete process.env[name];
    assert.equal(resolveConfig().autoCapture, true);
    assert.equal(resolveConfig().autoRecall, true);
    for (const name of names) process.env[name] = "false";
    assert.equal(resolveConfig().autoCapture, false);
    assert.equal(resolveConfig().autoRecall, false);
    assert.equal(resolveConfig({ autoCapture: true }).autoCapture, true);
    for (const name of names) process.env[name] = "true";
    assert.equal(resolveConfig({ autoCapture: false }).autoCapture, false);
    assert.equal(resolveConfig({ autoRecall: false }).autoRecall, false);
    const schema = JSON.parse(readFileSync(new URL("./openclaw.plugin.json", import.meta.url)));
    for (const key of ["autoCapture", "autoRecall"]) {
      assert.equal(schema.configSchema.properties[key].type, "boolean");
      // OpenClaw applies schema defaults before register(), which would mask env settings.
      assert.equal(Object.hasOwn(schema.configSchema.properties[key], "default"), false);
    }
  } finally {
    names.forEach((name, i) => {
      if (saved[i] === undefined) delete process.env[name];
      else process.env[name] = saved[i];
    });
  }
});

test("native plugin phase switches over real HTTP", async (t) => {
  const calls = [];
  const server = createServer(async (req, res) => {
    let text = "";
    for await (const chunk of req) text += chunk;
    const body = text ? JSON.parse(text) : {};
    calls.push({ path: req.url, method: req.method, headers: req.headers, body });
    const result = req.url.endsWith("/compose")
      ? { messages: body.messages, identityContext: "Remembered history", estimatedTokens: 20 }
      : { ok: true, ingested: true, disposed: true };
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(result));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const memoryApiBaseUrl = `http://127.0.0.1:${server.address().port}`;
  const messages = [{ role: "user", content: "QA question" }];
  const params = { sessionId: "phase-test", messages, prompt: "QA question" };
  try {
    for (const autoCapture of [true, false]) {
      for (const autoRecall of [true, false]) {
        await t.test(`capture=${autoCapture}, recall=${autoRecall}`, async () => {
          calls.length = 0;
          const { engine } = host({ memoryApiBaseUrl, autoCapture, autoRecall, prefetchEnabled: true });
          await engine.ingest(params);
          await engine.afterTurn(params);
          const assembled = await engine.assemble(params);
          const paths = calls.map((call) => call.path);
          assert.equal(paths.includes("/api/v1/ingest"), autoCapture);
          assert.equal(paths.includes("/api/v1/after_turn"), autoCapture);
          assert.equal(paths.includes("/api/v1/compose"), autoRecall);
          assert.equal(paths.includes("/api/v1/prefetch"), autoRecall);
          if (autoRecall) assert.match(JSON.stringify(assembled.messages), /Remembered history/);
          else assert.deepEqual(assembled, { messages, estimatedTokens: 0, systemPromptAddition: "" });
          await engine.dispose();
        });
      }
    }
    for (const externalization of [false, true]) {
      for (const reason of ["reset", "deleted"]) {
        await t.test(`capture off keeps cleanup identity: externalization=${externalization}, reason=${reason}`, async () => {
          calls.length = 0;
          const defaults = { accountId: "default-account", userId: "default-user", agentId: "default-agent" };
          const identity = { accountId: "host-account", userId: "host-user", agentId: "host-agent" };
          const sessionId = `dynamic-${externalization}-${reason}`;
          const { engine, hooks } = host({ memoryApiBaseUrl, ...defaults, autoCapture: false,
            toolResultExternalizationEnabled: externalization,
            toolResultFirstViewExternalizationEnabled: externalization });
          const input = { ...params, ...identity, sessionId };
          assert.deepEqual(await engine.ingest(input), { ingested: false });
          await engine.afterTurn(input);
          await engine.dispose();
          assert.equal(calls.length, 0, "identity registration must not capture or queue writes");
          const end = hooks.filter((hook) => hook.name === "session_end").at(-1).handler;
          await end({ sessionId, reason }, {});
          assert.deepEqual(calls.map(({ path, method }) => ({ path, method })), [
            { path: "/api/v1/dispose", method: "POST" },
            ...(reason === "deleted" ? [{ path: `/api/v1/sessions/${sessionId}`, method: "DELETE" }] : []),
          ]);
          for (const { body, headers } of calls) {
            assert.deepEqual({ accountId: body.accountId, userId: body.userId, agentId: body.agentId }, identity);
            assert.equal(headers["x-user-id"], identity.userId);
            assert.equal(body.messages, undefined);
          }
          // A different session only overrides userId; defaults must still be used for other fields.
          calls.length = 0;
          const partialId = sessionId + "-partial";
          await engine.afterTurn({ ...params, sessionId: partialId, userId: "partial-user" });
          await end({ sessionId: partialId, reason: "reset" }, {});
          assert.equal(calls.length, 1);
          assert.deepEqual(calls[0].body, { sessionId: partialId, ...defaults, userId: "partial-user" });
        });
      }
    }
    await t.test("capture off creates no after-turn retry work; cleanup still disposes existing buffers", async () => {
      calls.length = 0;
      const { engine, hooks } = host({ memoryApiBaseUrl, autoCapture: false,
        toolResultExternalizationEnabled: true, toolResultFirstViewExternalizationEnabled: false });
      assert.deepEqual(await engine.ingest(params), { ingested: false });
      await engine.afterTurn(params);
      await engine.dispose();
      assert.equal(calls.length, 0);
      // Last session_end handler is memory cleanup, not the independent trace hook.
      await hooks.filter((hook) => hook.name === "session_end").at(-1).handler(
        { sessionId: "previously-buffered", reason: "reset" }, {});
      assert.deepEqual(calls.map((call) => call.path), ["/api/v1/dispose"]);
      assert.equal(calls[0].body.messages, undefined);
    });
  } finally {
    const closed = once(server, "close");
    server.close();
    server.closeAllConnections?.();
    await closed;
  }
});

test("explicit compact remains available with automatic capture disabled", async () => {
  const calls = [];
  const result = await runCompactControlFlow({
    cfg: resolveConfig({ autoCapture: false, compactTakeoverEnabled: true }),
    params: { sessionId: "existing-history" },
    call: async (method) => {
      calls.push(method);
      return method === "prepare_compaction"
        ? { ok: true, prepareToken: "prepared-history" }
        : { ok: true, compacted: true, result: { summary: "existing history" } };
    },
    persistTakeoverCompactionBoundary: async () => ({}),
  });
  assert.equal(result.compacted, true);
  assert.deepEqual(calls, ["prepare_compaction", "compact"]);
});
