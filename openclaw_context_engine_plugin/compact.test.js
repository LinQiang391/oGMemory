import assert from "node:assert/strict";

import {
  appendTakeoverCompactionBoundary,
  delegateCompactionToRuntimeOrThrow,
  buildLayeredComposeMessages,
  prepareCompactionWithTransport,
  resolveConfig,
  runCompactControlFlow,
  spawnContextEngine,
} from "./index.js";

function messageEntry(id, role, content = "", extra = {}) {
  return {
    id,
    type: "message",
    parentId: null,
    message: {
      role,
      content,
      ...extra,
    },
  };
}

function compactionEntry(id, firstKeptEntryId, summary = "summary") {
  return {
    id,
    type: "compaction",
    parentId: null,
    summary,
    firstKeptEntryId,
    tokensBefore: 100,
  };
}

function createFakeSessionManager(branch, appendIds = "compaction-test", customIds = []) {
  const appendCalls = [];
  const customAppendCalls = [];
  const entries = [...branch];
  const ids = Array.isArray(appendIds) ? [...appendIds] : [appendIds];
  const customEntryIds = Array.isArray(customIds) ? [...customIds] : [customIds];
  const manager = {
    appendCalls,
    customAppendCalls,
    getBranch: () => entries,
    getEntries: () => entries,
    getHeader: () => ({ type: "session", id: "session-test" }),
    getLeafEntry: () => entries[entries.length - 1] ?? null,
    getLeafId: () => entries[entries.length - 1]?.id ?? null,
    appendCustomEntry: (customType, data) => {
      const appendId =
        customEntryIds.shift() ?? `custom-boundary-${customAppendCalls.length + 1}`;
      customAppendCalls.push({ customType, data });
      entries.push({
        id: appendId,
        type: "custom",
        parentId: entries[entries.length - 1]?.id ?? null,
        customType,
        data,
      });
      return appendId;
    },
    appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
      const appendId = ids.shift() ?? `compaction-${appendCalls.length + 1}`;
      appendCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
      entries.push({
        id: appendId,
        type: "compaction",
        parentId: entries[entries.length - 1]?.id ?? null,
        summary,
        firstKeptEntryId,
        tokensBefore,
        details,
      });
      return appendId;
    },
  };
  return manager;
}

const resolved = resolveConfig({
  compactTakeoverEnabled: false,
  summaryMaxChars: 1234,
  shortTermIndexMode: "off",
});

assert.equal(resolved.compactTakeoverEnabled, false);
assert.equal(resolved.summaryMaxChars, 1234);
assert.equal(resolved.shortTermIndexMode, "off");

const prefetchResolved = resolveConfig({
  prefetchEnabled: true,
  prefetchTopK: 7,
});
assert.equal(prefetchResolved.prefetchEnabled, true);
assert.equal(prefetchResolved.prefetchTopK, 7);

const spawnedEnv = {};
const spawned = spawnContextEngine(
  {
    ...resolveConfig({
      projectRoot: "/tmp/project",
      agfsPort: 1833,
      contextEnginePort: 8090,
      prefetchEnabled: true,
      prefetchTopK: 7,
    }),
  },
  { info: () => {}, debug: () => {}, warn: () => {} },
  {
    spawnFn: (_cmd, _args, options) => {
      Object.assign(spawnedEnv, options.env);
      return {
        stdout: { on: () => {} },
        stderr: { on: () => {} },
        on: () => {},
      };
    },
    findPython: () => "python",
  },
);
assert.ok(spawned);
assert.equal(spawnedEnv.OGMEM_PREFETCH_ENABLED, "true");
assert.equal(spawnedEnv.OGMEM_PREFETCH_TOP_K, "7");

const layered = buildLayeredComposeMessages(
  {
    identityContext: "profile",
    episodicContext: "archive",
    sessionContext: "session",
    retrievedEvidence: "retrieved",
    messages: [
      { role: "user", content: "server profile", _ogmem: true },
      { role: "user", content: "question" },
      { role: "user", content: "server retrieved", _ogmem: true },
    ],
  },
  [],
);
assert.deepEqual(layered.map((m) => m.content), [
  "[User Profile - background, identity, preferences, and stable traits]\nprofile",
  "[Episodic Memory - past conversations, events, and experiences]\narchive",
  "question",
  "[Current Session - recent messages and context from this conversation]\nsession",
  "[Retrieved Memories - relevant facts recalled from long-term memory]\nretrieved",
]);

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

const boundaryCalls = [];
const persistedBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session.jsonl",
  summary: "takeover-summary",
  tokensBefore: 99,
  details: { ok: true },
  loadSessionManagerModule: async () => ({
    SessionManager: {
      open: () => ({
        getLeafId: () => "entry-9",
        appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
          boundaryCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
          return "compaction-1";
        },
      }),
    },
  }),
});

assert.deepEqual(persistedBoundary, {
  firstKeptEntryId: "entry-9",
  compactionEntryId: "compaction-1",
  reused: false,
});
assert.deepEqual(boundaryCalls, [
  {
    summary: "takeover-summary",
    firstKeptEntryId: "entry-9",
    tokensBefore: 99,
    details: { ok: true },
  },
]);

const assistantAfterToolManager = createFakeSessionManager(
  [
    messageEntry("entry-user", "user", "read the large file"),
    messageEntry("entry-tool-call", "assistant", [
      { type: "text", text: "reading" },
      { type: "toolCall", id: "call-1", name: "read_file", arguments: {} },
    ]),
    messageEntry("entry-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
    messageEntry("entry-final-assistant", "assistant", "I checked it."),
  ],
  "compaction-assistant-tail",
);
const assistantAfterToolBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-assistant-tail.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => assistantAfterToolManager },
  }),
});
assert.equal(assistantAfterToolBoundary.firstKeptEntryId, "entry-final-assistant");
assert.equal(
  assistantAfterToolManager.appendCalls[0].firstKeptEntryId,
  "entry-final-assistant",
);

const toolLeafManager = createFakeSessionManager(
  [
    messageEntry("entry-user-tool-leaf", "user", "read the large file"),
    messageEntry("entry-tool-call-leaf", "assistant", [
      { type: "text", text: "reading" },
      { type: "tool_use", id: "call-2", name: "read_file", input: {} },
    ]),
    messageEntry("entry-tool-result-leaf", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
  ],
  "compaction-tool-final",
  "custom-tool-boundary",
);
const toolLeafBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-tool-leaf.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => toolLeafManager },
  }),
});
assert.equal(toolLeafBoundary.firstKeptEntryId, "custom-tool-boundary");
assert.equal(toolLeafBoundary.compactionEntryId, "compaction-tool-final");
assert.equal(toolLeafManager.customAppendCalls.length, 1);
assert.equal(toolLeafManager.appendCalls.length, 1);
assert.equal(toolLeafManager.appendCalls[0].firstKeptEntryId, "custom-tool-boundary");

const fallbackHardCheckpointCalls = [];
const fallbackHardCheckpointManager = {
  getBranch: () => [
    messageEntry("entry-fallback-user", "user", "read the large file"),
    messageEntry("entry-fallback-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
  ],
  getLeafId: () => "entry-fallback-tool-result",
  getLeafEntry: () => messageEntry("entry-fallback-tool-result", "toolResult"),
  appendCompaction: (summary, firstKeptEntryId, tokensBefore, details) => {
    const id =
      fallbackHardCheckpointCalls.length === 0
        ? "compaction-fallback-anchor"
        : "compaction-fallback-final";
    fallbackHardCheckpointCalls.push({ summary, firstKeptEntryId, tokensBefore, details });
    return id;
  },
};
const fallbackHardCheckpointBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-fallback-hard-checkpoint.jsonl",
  summary: "summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => fallbackHardCheckpointManager },
  }),
});
assert.equal(fallbackHardCheckpointBoundary.firstKeptEntryId, "compaction-fallback-anchor");
assert.equal(fallbackHardCheckpointBoundary.compactionEntryId, "compaction-fallback-final");
assert.equal(fallbackHardCheckpointCalls.length, 2);
assert.equal(fallbackHardCheckpointCalls[0].firstKeptEntryId, "entry-fallback-user");
assert.equal(
  fallbackHardCheckpointCalls[1].firstKeptEntryId,
  "compaction-fallback-anchor",
);

const unsafeExistingCompactionManager = createFakeSessionManager(
  [
    compactionEntry("entry-existing-summary", "entry-existing-summary", "old summary"),
    messageEntry("entry-existing-tool-call", "assistant", [
      { type: "tool_use", id: "call-existing", name: "read_file", input: {} },
    ]),
    messageEntry("entry-existing-tool-result", "toolResult", [
      { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
    ]),
    compactionEntry(
      "entry-existing-leaf-compaction",
      "entry-existing-summary",
      "unsafe leaf summary",
    ),
  ],
  "compaction-existing-hardened",
  "custom-existing-boundary",
);
const unsafeExistingCompactionBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-existing-compaction.jsonl",
  summary: "fresh summary",
  tokensBefore: 1000,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => unsafeExistingCompactionManager },
  }),
});
assert.equal(
  unsafeExistingCompactionBoundary.firstKeptEntryId,
  "custom-existing-boundary",
);
assert.equal(
  unsafeExistingCompactionBoundary.compactionEntryId,
  "compaction-existing-hardened",
);
assert.equal(unsafeExistingCompactionManager.customAppendCalls.length, 1);
assert.equal(unsafeExistingCompactionManager.appendCalls.length, 1);
assert.equal(
  unsafeExistingCompactionManager.appendCalls[0].firstKeptEntryId,
  "custom-existing-boundary",
);

const manualManager = createFakeSessionManager(
  [
    messageEntry("entry-manual-user", "user", "old request"),
    messageEntry("entry-manual-assistant", "assistant", "old response"),
  ],
  "compaction-manual-final",
  "custom-manual-boundary",
);
const manualBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-manual.jsonl",
  summary: "manual summary",
  tokensBefore: 100,
  params: {
    force: true,
    compactionTarget: "threshold",
    runtimeContext: { trigger: "manual" },
  },
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => manualManager },
  }),
});
assert.equal(manualBoundary.firstKeptEntryId, "custom-manual-boundary");
assert.equal(manualBoundary.compactionEntryId, "compaction-manual-final");
assert.equal(manualManager.customAppendCalls.length, 1);
assert.equal(manualManager.appendCalls.length, 1);
assert.equal(manualManager.appendCalls[0].firstKeptEntryId, "custom-manual-boundary");

const upstreamBoundaryManager = createFakeSessionManager(
  [
    messageEntry("entry-upstream-user", "user", "safe user"),
    messageEntry("entry-upstream-assistant", "assistant", "safe assistant"),
  ],
  "compaction-upstream",
);
const upstreamBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-upstream.jsonl",
  summary: "upstream summary",
  tokensBefore: 100,
  upstreamFirstKeptEntryId: "entry-upstream-user",
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => upstreamBoundaryManager },
  }),
});
assert.equal(upstreamBoundary.firstKeptEntryId, "entry-upstream-user");
assert.equal(
  upstreamBoundaryManager.appendCalls[0].firstKeptEntryId,
  "entry-upstream-user",
);

const upstreamToolTailManager = createFakeSessionManager(
  [
    messageEntry("entry-upstream-tool-call", "assistant", [
      { type: "toolCall", id: "call-upstream", name: "read_file", arguments: {} },
    ]),
    messageEntry("entry-upstream-tool-result", "toolResult", [
      { type: "text", text: "small result" },
    ]),
  ],
  "compaction-upstream-tool-tail",
);
const upstreamToolTailBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-upstream-tool-tail.jsonl",
  summary: "upstream tool tail summary",
  tokensBefore: 100,
  upstreamFirstKeptEntryId: "entry-upstream-tool-call",
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => upstreamToolTailManager },
  }),
});
assert.equal(upstreamToolTailBoundary.firstKeptEntryId, "entry-upstream-tool-call");
assert.equal(
  upstreamToolTailManager.appendCalls[0].firstKeptEntryId,
  "entry-upstream-tool-call",
);

for (const [name, branch] of [
  [
    "object-tool-call",
    [
      messageEntry("entry-object-tool-call", "assistant", {
        type: "toolCall",
        id: "call-object",
        name: "read_file",
        arguments: {},
      }),
      messageEntry("entry-object-tool-result", "toolResult", [
        { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
      ]),
    ],
  ],
  [
    "function-call",
    [
      messageEntry("entry-function-call", "assistant", {
        type: "function_call",
        id: "call-function",
        name: "read_file",
        arguments: "{}",
      }),
      messageEntry("entry-function-tool-result", "tool", [
        { type: "text", text: "TOOL_RESULT_BLOB_MARKER" },
      ]),
    ],
  ],
  [
    "stopreason-tooluse",
    [
      messageEntry("entry-stopreason-tooluse", "assistant", "calling tool", {
        stopReason: "toolUse",
      }),
    ],
  ],
]) {
  const customBoundaryId = `custom-${name}-boundary`;
  const manager = createFakeSessionManager(
    branch,
    `compaction-${name}-final`,
    customBoundaryId,
  );
  const boundary = await appendTakeoverCompactionBoundary({
    sessionFile: `/tmp/session-${name}.jsonl`,
    summary: `${name} summary`,
    tokensBefore: 100,
    loadSessionManagerModule: async () => ({
      SessionManager: { open: () => manager },
    }),
  });
  assert.equal(boundary.firstKeptEntryId, customBoundaryId);
  assert.equal(manager.customAppendCalls.length, 1);
  assert.equal(manager.appendCalls.length, 1);
  assert.equal(manager.appendCalls[0].firstKeptEntryId, customBoundaryId);
}

const emptySummaryManager = createFakeSessionManager(
  [messageEntry("entry-empty-summary-user", "user", "pending request")],
  "compaction-empty-summary",
);
const emptySummaryBoundary = await appendTakeoverCompactionBoundary({
  sessionFile: "/tmp/session-empty-summary.jsonl",
  summary: "",
  tokensBefore: 100,
  loadSessionManagerModule: async () => ({
    SessionManager: { open: () => emptySummaryManager },
  }),
});
assert.equal(emptySummaryBoundary.firstKeptEntryId, "entry-empty-summary-user");
assert.equal(emptySummaryManager.appendCalls[0].summary, "");

const takeoverWithBoundaryCalls = [];
const takeoverWithBoundaryResult = await runCompactControlFlow({
  cfg: resolveConfig({
    compactTakeoverEnabled: true,
    summaryMaxChars: 88,
    shortTermIndexMode: "sync",
  }),
  params: {
    sessionId: "sess-takeover-boundary",
    sessionFile: "/tmp/session.jsonl",
    tokenBudget: 4096,
  },
  call: async (method, params) => {
    takeoverWithBoundaryCalls.push({ method, params });
    if (method === "prepare_compaction") return { ok: true, prepareToken: "prepare-9" };
    if (method === "compact") {
      return {
        ok: true,
        compacted: true,
        result: {
          summary: "boundary-summary",
          tokensBefore: 123,
          tokensAfter: 45,
          details: { safe: true },
        },
      };
    }
    throw new Error(`unexpected method: ${method}`);
  },
  persistTakeoverCompactionBoundary: async ({ sessionFile, summary, tokensBefore, details }) => {
    takeoverWithBoundaryCalls.push({
      method: "persist_boundary",
      params: { sessionFile, summary, tokensBefore, details },
    });
    return {
      firstKeptEntryId: "entry-42",
      compactionEntryId: "compaction-42",
      reused: false,
    };
  },
});

assert.equal(takeoverWithBoundaryResult.result.firstKeptEntryId, "entry-42");
assert.deepEqual(takeoverWithBoundaryCalls, [
  {
    method: "prepare_compaction",
    params: {
      sessionId: "sess-takeover-boundary",
      sessionFile: "/tmp/session.jsonl",
      tokenBudget: 4096,
      accountId: "",
      userId: "",
      agentId: "",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
    },
  },
  {
    method: "compact",
    params: {
      sessionId: "sess-takeover-boundary",
      sessionFile: "/tmp/session.jsonl",
      tokenBudget: 4096,
      accountId: "",
      userId: "",
      agentId: "",
      summaryMaxChars: 88,
      shortTermIndexMode: "sync",
      prepareToken: "prepare-9",
    },
  },
  {
    method: "persist_boundary",
    params: {
      sessionFile: "/tmp/session.jsonl",
      summary: "boundary-summary",
      tokensBefore: 123,
      details: { safe: true },
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
