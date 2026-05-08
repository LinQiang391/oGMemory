import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  waitForHealth,
  waitForHealthOrExit,
  quickHealthCheck,
  prepareLocalPort,
  findGoBinary,
  findPythonBin,
} from "./process.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "..");

// ---------------------------------------------------------------------------
// Service defaults
// ---------------------------------------------------------------------------

function resolveBoolean(value, fallback) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(lowered)) return true;
    if (["0", "false", "no", "off"].includes(lowered)) return false;
  }
  return fallback;
}

function resolveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function resolveConfig(pluginConfig = {}) {
  return {
    mode: pluginConfig.mode || process.env.OGMEM_MODE || "remote",
    // AGFS
    agfsServerDir:
      pluginConfig.agfsServerDir ||
      process.env.AGFS_SERVER_DIR ||
      join(PROJECT_ROOT, "agfs"),
    agfsPort: pluginConfig.agfsPort || Number(process.env.AGFS_PORT) || 1833,
    agfsDataDir:
      pluginConfig.agfsDataDir ||
      process.env.AGFS_DATA_DIR ||
      join(PROJECT_ROOT, "..", "data"),
    // ContextEngine
    contextEnginePort:
      pluginConfig.contextEnginePort || Number(process.env.CONTEXTENGINE_PORT) || 8090,
    projectRoot: pluginConfig.projectRoot || process.env.CONTEXTENGINE_PROJECT_ROOT || PROJECT_ROOT,
    // HTTP mode (remote or locally-spawned)
    memoryApiBaseUrl:
      pluginConfig.memoryApiBaseUrl || process.env.OGMEM_API_URL || "http://127.0.0.1:8090",
    // Multi-tenant auth
    authApiKey: pluginConfig.authApiKey || process.env.OG_AUTH_API_KEY || "",
    authAccountId: pluginConfig.authAccountId || process.env.OG_AUTH_ACCOUNT_ID || "",
    // Timing
    startupTimeoutMs: pluginConfig.startupTimeoutMs || 60_000,
    commitTokenThreshold: pluginConfig.commitTokenThreshold ?? 0,
    // Identity
    accountId: pluginConfig.accountId || "",
    userId: pluginConfig.userId || "",
    agentId: pluginConfig.agentId || "",
    // LLM (passed as env vars to CE server)
    llmProvider: pluginConfig.llmProvider || process.env.CONTEXTENGINE_PROVIDER || "mock",
    llmApiKey: pluginConfig.llmApiKey || process.env.OPENAI_API_KEY || "",
    llmBaseUrl: pluginConfig.llmBaseUrl || process.env.OPENAI_BASE_URL || "",
    llmModel: pluginConfig.llmModel || process.env.OPENAI_LLM_MODEL || "gpt-4o-mini",
    // Embedding (defaults to LLM settings if not specified)
    embeddingProvider: pluginConfig.embeddingProvider || process.env.EMBEDDING_PROVIDER || "",
    embeddingApiKey: pluginConfig.embeddingApiKey || process.env.OPENAI_EMBEDDING_API_KEY || "",
    embeddingBaseUrl: pluginConfig.embeddingBaseUrl || process.env.OPENAI_EMBEDDING_BASE_URL || "",
    embeddingModel: pluginConfig.embeddingModel || process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-ada-002",
    compactTakeoverEnabled: resolveBoolean(
      pluginConfig.compactTakeoverEnabled ?? process.env.OGMEM_COMPACT_TAKEOVER_ENABLED,
      true,
    ),
    summaryMaxChars: resolveInteger(
      pluginConfig.summaryMaxChars ?? process.env.OGMEM_SUMMARY_MAX_CHARS,
      4000,
    ),
    shortTermIndexMode: ["sync", "async", "off"].includes(pluginConfig.shortTermIndexMode)
      ? pluginConfig.shortTermIndexMode
      : ["sync", "async", "off"].includes(process.env.OGMEM_SHORT_TERM_INDEX_MODE)
        ? process.env.OGMEM_SHORT_TERM_INDEX_MODE
        : "sync",
  };
}

// ---------------------------------------------------------------------------
// Service spawners
// ---------------------------------------------------------------------------

function spawnAgfs(cfg, logger) {
  const goCheck = findGoBinary();
  if (!goCheck.ok) {
    throw new Error(
      "Go is not installed. Install Go 1.22+ to run AGFS, or use mode=remote.\n" +
        "  → https://go.dev/dl/",
    );
  }
  logger.info(`agfs: using ${goCheck.version}`);

  // Ensure data dir exists
  mkdirSync(cfg.agfsDataDir, { recursive: true });

  const child = spawn("go", ["run", "cmd/server/main.go", "-addr", `:${cfg.agfsPort}`], {
    cwd: cfg.agfsServerDir,
    env: { ...process.env },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (d) => logger.debug?.(`[agfs:stdout] ${String(d).trim()}`));
  child.stderr?.on("data", (d) => logger.debug?.(`[agfs:stderr] ${String(d).trim()}`));
  child.on("exit", (code, sig) => {
    if (!child.killed) logger.warn(`agfs: exited unexpectedly (code=${code}, signal=${sig})`);
  });

  return child;
}

function spawnContextEngine(cfg, logger) {
  const pythonBin = findPythonBin(cfg.projectRoot);
  logger.info(`ce: using python=${pythonBin}`);

  const child = spawn(pythonBin, ["server/app.py"], {
    cwd: cfg.projectRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      AGFS_PORT: String(cfg.agfsPort),
      AGFS_BASE_URL: `http://127.0.0.1:${cfg.agfsPort}`,
      OGMEM_CONFIG: join(cfg.projectRoot, "config", "ogmem.yaml"),
      // LLM config
      CONTEXTENGINE_PROVIDER: cfg.llmProvider,
      OPENAI_API_KEY: cfg.llmApiKey,
      OPENAI_BASE_URL: cfg.llmBaseUrl,
      OPENAI_LLM_MODEL: cfg.llmModel,
      // Embedding config (fallback to LLM settings)
      EMBEDDING_PROVIDER: cfg.embeddingProvider || cfg.llmProvider,
      OPENAI_EMBEDDING_API_KEY: cfg.embeddingApiKey || cfg.llmApiKey,
      OPENAI_EMBEDDING_BASE_URL: cfg.embeddingBaseUrl || cfg.llmBaseUrl,
      OPENAI_EMBEDDING_MODEL: cfg.embeddingModel,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (d) => logger.debug?.(`[ce:stdout] ${String(d).trim()}`));
  child.stderr?.on("data", (d) => logger.debug?.(`[ce:stderr] ${String(d).trim()}`));
  child.on("exit", (code, sig) => {
    if (!child.killed) logger.warn(`ce: exited unexpectedly (code=${code}, signal=${sig})`);
  });

  return child;
}

// ---------------------------------------------------------------------------
// Transport: HTTP
// ---------------------------------------------------------------------------

function buildAuthHeaders(authCfg = {}, params = {}, contentType = null) {
  const headers = {};
  if (contentType) headers["Content-Type"] = contentType;
  if (authCfg.authApiKey) headers["X-API-Key"] = authCfg.authApiKey;
  if (authCfg.authAccountId) headers["X-Account-ID"] = authCfg.authAccountId;

  const userId = typeof params?.userId === "string" ? params.userId.trim() : "";
  if (userId) headers["X-User-ID"] = userId;

  return headers;
}

async function callHttp(baseUrl, method, params, cfg = {}) {
  try {
    const headers = buildAuthHeaders(cfg, params, "application/json");
    const resp = await fetch(`${baseUrl}/api/v1/${method}`, {
      method: "POST",
      headers,
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(method === "after_turn" ? 300_000 : 120_000),
    });
    if (!resp.ok) {
      const body = await resp.text();
      console.error(`[og-memory] HTTP ${resp.status} from ${method}, body: ${body}`);
      return null;
    }
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] HTTP ${method} error: ${err.message}`);
    return null;
  }
}

export async function loadRuntimeCompactionDelegate() {
  for (const specifier of ["openclaw/plugin-sdk", "openclaw/plugin-sdk/core"]) {
    try {
      const mod = await import(specifier);
      if (typeof mod.delegateCompactionToRuntime === "function") {
        return mod.delegateCompactionToRuntime;
      }
    } catch {
      // The plugin can still run its preflight JS path when the runtime SDK is absent.
    }
  }
  return null;
}

function buildCompactionParams(params, cfg) {
  return {
    ...params,
    accountId: params?.accountId || cfg.accountId,
    userId: params?.userId || cfg.userId,
    agentId: params?.agentId || cfg.agentId,
    summaryMaxChars: cfg.summaryMaxChars,
    shortTermIndexMode: cfg.shortTermIndexMode,
  };
}

function requireSuccessfulTransportResult(method, result) {
  if (!result) {
    throw new Error(`og-memory ${method} failed: empty transport response`);
  }
  if (result.error) {
    throw new Error(`og-memory ${method} failed: ${result.error}`);
  }
  return result;
}

export async function prepareCompactionWithTransport({ call, params }) {
  const result = await call("prepare_compaction", params);
  return requireSuccessfulTransportResult("prepare_compaction", result);
}

export async function delegateCompactionToRuntimeOrThrow({
  params,
  runtimeDelegate,
  loadDelegate = loadRuntimeCompactionDelegate,
}) {
  const delegate = runtimeDelegate ?? (await loadDelegate());
  if (typeof delegate === "function") {
    const result = await delegate(params);
    if (!result) {
      throw new Error("delegateCompactionToRuntime failed: empty runtime response");
    }
    return result;
  }
  throw new Error("delegateCompactionToRuntime unavailable");
}

export async function runCompactControlFlow({
  cfg,
  params,
  call,
  runtimeDelegate,
  loadDelegate = loadRuntimeCompactionDelegate,
}) {
  const enhancedParams = buildCompactionParams(params, cfg);
  const prepared = await prepareCompactionWithTransport({ call, params: enhancedParams });

  if (!cfg.compactTakeoverEnabled) {
    return delegateCompactionToRuntimeOrThrow({
      params,
      runtimeDelegate,
      loadDelegate,
    });
  }
  if (!prepared.prepareToken) {
    throw new Error("prepare_compaction failed: missing prepare token");
  }

  const result = await call("compact", {
    ...enhancedParams,
    prepareToken: prepared.prepareToken,
  });
  return requireSuccessfulTransportResult("compact", result);
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

export default {
  id: "og-memory-context-engine",
  name: "oG-Memory Context Engine",
  description: "oG-Memory integration: semantic search with hierarchical retrieval",
  kind: "context-engine",

  register(api) {
    const cfg = resolveConfig(api.pluginConfig);
    const isLocal = cfg.mode === "local";

    api.logger.info(
      `og-memory: mode=${isLocal ? "local" : "remote"}, url=${cfg.memoryApiBaseUrl}`,
    );

    // =======================================================================
    // Service lifecycle — start/stop AGFS + ContextEngine
    // =======================================================================

    let agfsProcess = null;
    let ceProcess = null;

    api.registerService({
      id: "contextengine-backend",
      start: async () => {
        if (!isLocal) {
          // Remote mode: just verify connectivity
          api.logger.info("og-memory: remote mode — verifying service connectivity...");
          const healthUrl = cfg.memoryApiBaseUrl
            ? `${cfg.memoryApiBaseUrl}/api/v1/health`
            : `http://127.0.0.1:${cfg.contextEnginePort}/api/v1/health`;
          const ceOk = await quickHealthCheck(
            healthUrl,
            5000,
          );
          if (!ceOk) {
            throw new Error(
              `Cannot reach ContextEngine at ${cfg.memoryApiBaseUrl || `http://127.0.0.1:${cfg.contextEnginePort}`}. ` +
              `Ensure the server is running, or switch to mode=local.`,
            );
          }
          api.logger.info("og-memory: remote services verified ✓");
          return;
        }

        // --- Local mode: spawn AGFS then ContextEngine ---

        // Check if AGFS server dir exists
        if (!existsSync(cfg.agfsServerDir)) {
          throw new Error(
            `AGFS server directory not found: ${cfg.agfsServerDir}\n` +
            `  Set agfsServerDir in plugin config or AGFS_SERVER_DIR env var.\n` +
            `  Or check that the agfs/ directory exists in the project.`,
          );
        }

        // 1. Start AGFS
        const agfsPort = await prepareLocalPort(
          cfg.agfsPort,
          (p) => `http://127.0.0.1:${p}/`,
          api.logger,
        );
        api.logger.info(`agfs: spawning on :${agfsPort} (dir: ${cfg.agfsServerDir})`);
        agfsProcess = spawnAgfs({ ...cfg, agfsPort }, api.logger);

        try {
          await waitForHealthOrExit(
            `http://127.0.0.1:${agfsPort}/`,
            cfg.startupTimeoutMs,
            500,
            agfsProcess,
            true, // AGFS returns 404 for / — accept any HTTP response
          );
          api.logger.info(`agfs: ready (port ${agfsPort}, pid ${agfsProcess.pid})`);
        } catch (err) {
          agfsProcess.kill("SIGTERM");
          agfsProcess = null;
          throw new Error(`AGFS failed to start: ${err.message}`);
        }

        // 2. Start ContextEngine
        const cePort = await prepareLocalPort(
          cfg.contextEnginePort,
          (p) => `http://127.0.0.1:${p}/api/v1/health`,
          api.logger,
        );
        const ceCfg = { ...cfg, agfsPort, contextEnginePort: cePort };
        api.logger.info(`ce: spawning on :${cePort} (root: ${cfg.projectRoot})`);
        ceProcess = spawnContextEngine(ceCfg, api.logger);

        try {
          await waitForHealthOrExit(
            `http://127.0.0.1:${cePort}/api/v1/health`,
            cfg.startupTimeoutMs,
            500,
            ceProcess,
          );
          api.logger.info(`ce: ready (port ${cePort}, pid ${ceProcess.pid})`);
        } catch (err) {
          ceProcess.kill("SIGTERM");
          ceProcess = null;
          agfsProcess.kill("SIGTERM");
          agfsProcess = null;
          throw new Error(`ContextEngine failed to start: ${err.message}`);
        }

        // Update port in config if auto-selected
        if (agfsPort !== cfg.agfsPort) cfg.agfsPort = agfsPort;
        if (cePort !== cfg.contextEnginePort) {
          cfg.contextEnginePort = cePort;
          cfg.memoryApiBaseUrl = `http://127.0.0.1:${cePort}`;
        }

        api.logger.info("og-memory: all services ready ✓");
      },

      stop: () => {
        if (ceProcess) {
          ceProcess.kill("SIGTERM");
          api.logger.info(`ce: stopped (pid ${ceProcess.pid})`);
          ceProcess = null;
        }
        if (agfsProcess) {
          agfsProcess.kill("SIGTERM");
          api.logger.info(`agfs: stopped (pid ${agfsProcess.pid})`);
          agfsProcess = null;
        }
        api.logger.info("og-memory: services stopped");
      },
    });

    // =======================================================================
    // Transport helpers
    // =======================================================================

    const ogmemUrl = cfg.memoryApiBaseUrl || (isLocal ? `http://127.0.0.1:${cfg.contextEnginePort}` : "");

    const authCfg = {
      authApiKey: cfg.authApiKey,
      authAccountId: cfg.authAccountId,
    };

    function call(method, params) {
      return callHttp(ogmemUrl, method, params, authCfg);
    }

    function callAsync(method, params) {
      return callHttp(ogmemUrl, method, params, authCfg);
    }

    function extractContentText(content) {
      if (typeof content === "string") return content;
      if (Array.isArray(content)) {
        return content
          .map((b) => (typeof b === "string" ? b : b?.text || ""))
          .join(" ")
          .trim();
      }
      return String(content || "");
    }

    function extractUserIdFromTag(content) {
      if (typeof content !== "string") return null;
      const match = content.match(/^\[ogmem-user:\s*([^\]]+)\]/i);
      return match ? match[1].trim() : null;
    }

    function extractUserIdFromMessages(messages) {
      if (!Array.isArray(messages)) return null;
      // Search user messages first
      for (const msg of messages) {
        if (msg?.role !== "user") continue;
        const content = extractContentText(msg.content || "");
        const userId = extractUserIdFromTag(content);
        if (userId) return userId;
      }
      // Fallback: search all messages
      for (const msg of messages) {
        const content = extractContentText(msg?.content || "");
        const userId = extractUserIdFromTag(content);
        if (userId) return userId;
      }
      return null;
    }

    // =======================================================================
    // Context engine registration
    // =======================================================================

    api.registerContextEngine("og-memory-context-engine", () => {
      let _instanceSessionId = null;

      return ({
      info: {
        id: "og-memory-context-engine",
        name: "oG-Memory Context Engine",
        version: "1.0.0",
        ownsCompaction: cfg.compactTakeoverEnabled,
      },

      async ingest(params) {
        return call("ingest", params);
      },

      async assemble(params) {
        const messages = params?.messages || [];

        const taggedUserId = extractUserIdFromMessages(messages);
        if (taggedUserId) {
          console.log(`[og-memory] assemble: extracted userId from tag: ${taggedUserId}`);
        }
        const enhancedParams = {
          ...params,
          userId: taggedUserId || params?.userId || cfg.userId,
          accountId: params?.accountId || cfg.accountId,
          agentId: params?.agentId || cfg.agentId,
        };

        const result = await call("compose", enhancedParams);

        console.log(`[og-memory] compose result keys: ${Object.keys(result || {}).join(',')}`);
        console.log(`[og-memory] assemble: identity=${(result?.identityContext||'').length}b, episodic=${(result?.episodicContext||'').length}b, session=${(result?.sessionContext||'').length}b, ws=${(result?.retrievedEvidence||'').length}b, msgs=${(result?.messages||[]).length}`);

        // Handle null result from callHttp errors
        if (!result) {
          console.error(`[og-memory] compose returned null - check CE server logs`);
          return {
            messages: params.messages || [],
            estimatedTokens: 0,
            systemPromptAddition: "",
          };
        }

        // Full compose snapshot for tests/e2e/eval.py (one JSON per session key, every assemble)
        // ALWAYS write snapshot, even if empty, for debugging
        try {
          const wsDir = "/tmp/ogmem_ws";
          mkdirSync(wsDir, { recursive: true });
          const wsKey = params?.sessionKey || params?.sessionId || "unknown";
          const composeSnapshot = {
            estimatedTokens: result.estimatedTokens ?? 0,
            archiveCount: result.archiveCount ?? 0,
            archiveIncluded: result.archiveIncluded ?? false,
            stats: result.stats ?? {},
            budgetUsedBySlot: result.budgetUsedBySlot ?? {},
            openLoops: result.openLoops ?? [],
            uncertainties: result.uncertainties ?? [],
            identityContext: result.identityContext ?? "",
            episodicContext: result.episodicContext ?? "",
            sessionContext: result.sessionContext ?? "",
            taskContext: result.taskContext ?? "",
            retrievedEvidence: result.retrievedEvidence ?? "",
            composeMessagesCount: Array.isArray(result.messages) ? result.messages.length : 0,
            systemPromptAddition: result.systemPromptAddition ?? "",
            systemPromptSuffix: result.systemPromptSuffix ?? "",
            memoryUserMessage: result.memoryUserMessage ?? "",
          };
          const snapshotPath = `${wsDir}/${wsKey}.compose.json`;
          writeFileSync(snapshotPath, JSON.stringify(composeSnapshot, null, 2), "utf-8");
          console.log(`[og-memory] Wrote compose snapshot: ${snapshotPath}`);
        } catch (e) {
          console.error(`[og-memory] Failed to write snapshot: ${e.message}`);
        }

        // All context layers as "user" role messages for KV-cache efficiency.
        // systemPromptAddition is empty — profile in user msg avoids invalidating
        // the stable system prompt KV-cache when profile updates.
        // OpenClaw's convertMessagesToInputItems drops "system" role messages,
        // so context must use "user" role to reach the LLM.

        // Build layered messages for KV-cache efficiency:
        // Stable prefix (profile, episodic) → conversation history → dynamic suffix (session, WS)
        // This maximizes KV-cache reuse: stable prefix + conversation prefix cached across turns.
        const layeredMessages = [];
        // --- Stable prefix (rarely changes, fully cached) ---
        if (result.identityContext) layeredMessages.push({ role: "user", content: `[User Profile — background, identity, preferences, and stable traits]\n${result.identityContext}` });
        if (result.episodicContext) layeredMessages.push({ role: "user", content: `[Episodic Memory — past conversations, events, and experiences]\n${result.episodicContext}` });
        // --- Conversation history (grows each turn, prefix cached) ---
        const origMessages = result.messages ?? params.messages ?? [];
        layeredMessages.push(...origMessages);
        // --- Dynamic suffix (changes every turn, only tail recomputed) ---
        if (result.sessionContext) layeredMessages.push({ role: "user", content: `[Current Session — recent messages and context from this conversation]\n${result.sessionContext}` });
        if (result.retrievedEvidence) layeredMessages.push({ role: "user", content: `[Retrieved Memories — relevant facts recalled from long-term memory]\n${result.retrievedEvidence}` });

        console.log(`[og-memory] assemble return: ${layeredMessages.length} msgs, ws_injected=${!!result.retrievedEvidence}, first_3_roles=${layeredMessages.slice(0,3).map(m=>m.role).join(',')}`);
        console.log(`[og-memory] assemble params keys: ${Object.keys(params || {}).join(',')}, sessionKey=${params?.sessionKey}, sessionId=${params?.sessionId}`);

        // Write WS to temp file for eval.py (backward compat: working-set text only)
        // ALWAYS write WS file, even if empty, for debugging
        try {
          const wsDir = "/tmp/ogmem_ws";
          mkdirSync(wsDir, { recursive: true });
          const wsKey = params?.sessionKey || params?.sessionId || "unknown";
          const wsPath = `${wsDir}/${wsKey}.txt`;
          writeFileSync(wsPath, result.retrievedEvidence || "", "utf-8");
          if (result.retrievedEvidence) {
            console.log(`[og-memory] Wrote working set: ${wsPath} (${result.retrievedEvidence.length} chars)`);
          } else {
            console.log(`[og-memory] Wrote empty working set: ${wsPath}`);
          }
        } catch (e) {
          console.error(`[og-memory] Failed to write working set: ${e.message}`);
        }

        return {
          messages: result.messages ?? params.messages ?? [],
          estimatedTokens: result.estimatedTokens ?? 0,
          systemPromptAddition: "",
        };
      },

      async compact(params) {
        return runCompactControlFlow({
          cfg,
          params,
          call,
        });
      },

      async afterTurn(params) {
        const messages = params?.messages || [];

        // EXTRACT userId from tagged message
        const taggedUserId = extractUserIdFromMessages(messages);
        if (taggedUserId) {
          console.log(`[og-memory] afterTurn: extracted userId from tag: ${taggedUserId}`);
        }

        if (params?.sessionId) _instanceSessionId = params.sessionId;

        console.log(`[og-memory] afterTurn called, sessionId=${params?.sessionId || "unknown"}, messages=${messages.length}, userId=${taggedUserId || params?.userId || "(none)"}`);

        // Always route through server's after_turn endpoint
        // It handles: session buffer → memory extraction → write AGFS → index → archive
        const enhancedParams = {
          ...params,
          userId: taggedUserId || params?.userId || cfg.userId,
          accountId: params?.accountId || cfg.accountId,
          agentId: params?.agentId || cfg.agentId,
          commitTokenThreshold: cfg.commitTokenThreshold,
        };

        return callAsync("after_turn", enhancedParams);
      },

      async dispose() {
        const sid = _instanceSessionId || "";
        if (sid) console.log(`[og-memory] dispose called, sessionId=${sid}`);
        return call("dispose", { sessionId: sid });
      },
    })});
  },
};
