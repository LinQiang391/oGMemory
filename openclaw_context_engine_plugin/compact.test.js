import assert from "node:assert/strict";

import {
  delegateCompactionToRuntimeOrThrow,
  prepareCompactionWithTransport,
  resolveConfig,
  runCompactControlFlow,
} from "./index.js";

const resolved = resolveConfig({
  compactTakeoverEnabled: false,
  summaryMaxChars: 1234,
  shortTermIndexMode: "off",
});

assert.equal(resolved.compactTakeoverEnabled, false);
assert.equal(resolved.summaryMaxChars, 1234);
assert.equal(resolved.shortTermIndexMode, "off");

const prepareCalls = [];
const prepared = await prepareCompactionWithTransport({
  call: async (method, params) => {
    prepareCalls.push({ method, params });
    return { ok: true, prepared: true, prepareToken: "prepare-1" };
  },
  params: { sessionId: "sess-prepare" },
});

assert.deepEqual(prepared, { ok: true, prepared: true, prepareToken: "prepare-1" });
assert.deepEqual(prepareCalls, [
  {
    method: "prepare_compaction",
    params: { sessionId: "sess-prepare" },
  },
]);

const takeoverCalls = [];
const takeoverResult = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: true,
    summaryMaxChars: 88,
    shortTermIndexMode: "sync",
    accountId: "acct-1",
    userId: "user-1",
    agentId: "agent-1",
  }),
  params: { sessionId: "sess-takeover", tokenBudget: 4096 },
  call: async (method, params) => {
    takeoverCalls.push({ method, params });
    if (method === "prepare_compaction") return { ok: true, prepareToken: "prepare-2" };
    if (method === "compact") {
      return { ok: true, compacted: true, result: { summary: "takeover-summary" } };
    }
    throw new Error(`unexpected method: ${method}`);
  },
});

assert.deepEqual(takeoverResult, {
  ok: true,
  compacted: true,
  result: { summary: "takeover-summary" },
});
assert.deepEqual(takeoverCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-takeover",
      tokenBudget: 4096,
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
    },
  },
  {
    method: "compact",
    params: {
      sessionId: "sess-takeover",
      tokenBudget: 4096,
      accountId: "acct-1",
      userId: "user-1",
      agentId: "agent-1",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
      prepareToken: "prepare-2",
    },
  },
]);

const delegatedCalls = [];
const delegated = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: false,
    summaryMaxChars: 99,
    shortTermIndexMode: "async",
    accountId: "acct-2",
  }),
  params: { sessionId: "sess-delegate", tokenBudget: 2048 },
  call: async (method, params) => {
    delegatedCalls.push({ method, params });
    return { ok: true, prepareToken: "prepare-3" };
  },
  runtimeDelegate: async (params) => ({
    ok: true,
    compacted: true,
    result: { summary: `delegated:${params.sessionId}` },
  }),
});

assert.deepEqual(delegated, {
  ok: true,
  compacted: true,
  result: { summary: "delegated:sess-delegate" },
});
assert.deepEqual(delegatedCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-delegate",
      tokenBudget: 2048,
      accountId: "acct-2",
      userId: "",
      agentId: "",
      summaryMaxChars: 99,
      shortTermIndexMode: "async",
    },
  },
]);

await assert.rejects(
  delegateCompactionToRuntimeOrThrow({
    params: { sessionId: "sess-no-delegate" },
    loadDelegate: async () => null,
  }),
  /delegateCompactionToRuntime unavailable/,
);
