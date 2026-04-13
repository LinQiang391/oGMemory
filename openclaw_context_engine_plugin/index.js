import { execFileSync, execFile, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync, mkdirSync } from "node:fs";
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
const MEMORY_API_PY = join(__dirname, "bridge", "memory_api.py");

// ---------------------------------------------------------------------------
// Service defaults
// ---------------------------------------------------------------------------

function resolveConfig(pluginConfig = {}) {
  return {
    mode: pluginConfig.mode || process.env.OGMEM_MODE || "remote",
    // AGFS
    agfsServerDir:
      pluginConfig.agfsServerDir ||
      process.env.AGFS_SERVER_DIR ||
      join(PROJECT_ROOT, "..", "OpenViking", "third_party", "agfs", "agfs-server"),
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
    // Timing
    startupTimeoutMs: pluginConfig.startupTimeoutMs || 60_000,
    // Subprocess bridge
    pythonBin: pluginConfig.pythonBin || process.env.OG_PYTHON_BIN || "python3",
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
      OGMEM_CONFIG: join(cfg.projectRoot, "ogmem.yaml"),
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
// Transport: subprocess (legacy) or HTTP (standalone oG-Memory)
// ---------------------------------------------------------------------------

function callPythonSync(pythonBin, method, params) {
  const raw = execFileSync(pythonBin, [MEMORY_API_PY, method, JSON.stringify(params)], {
    encoding: "utf-8",
    timeout: 120_000,
  });
  const lines = raw.trim().split("\n");
  try { return JSON.parse(lines[lines.length - 1]); } catch { return {}; }
}

function callPythonAsync(pythonBin, method, params) {
  return new Promise((resolve) => {
    execFile(
      pythonBin,
      [MEMORY_API_PY, method, JSON.stringify(params)],
      { encoding: "utf-8", timeout: 300_000 },
      (err, stdout, stderr) => {
        if (stderr) for (const l of stderr.trim().split("\n")) if (l.trim()) console.error(`[og-memory:${method}] ${l}`);
        if (stdout) for (const l of stdout.trim().split("\n").slice(0, -1)) if (l.trim()) console.log(`[og-memory:${method}] ${l}`);
        if (err) { console.error(`[og-memory] ${method} async error: ${err.message}`); resolve({}); return; }
        const lines = (stdout || "").trim().split("\n");
        try { resolve(JSON.parse(lines[lines.length - 1])); } catch { resolve({}); }
      },
    );
  });
}

async function callHttp(baseUrl, method, params) {
  try {
    const resp = await fetch(`${baseUrl}/api/v1/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(method === "after_turn" ? 300_000 : 120_000),
    });
    if (!resp.ok) {
      console.error(`[og-memory] HTTP ${resp.status} from ${method}`);
      return {};
    }
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] HTTP ${method} error: ${err.message}`);
    return {};
  }
}

// ---------------------------------------------------------------------------
// Session API helpers (HTTP mode)
// ---------------------------------------------------------------------------

async function sessionAddMessage(baseUrl, sessionId, role, content, params) {
  try {
    const resp = await fetch(`${baseUrl}/api/v1/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, content, ...params }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) return {};
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] sessionAddMessage error: ${err.message}`);
    return {};
  }
}

async function sessionGet(baseUrl, sessionId, params) {
  try {
    const qs = new URLSearchParams(params).toString();
    const resp = await fetch(`${baseUrl}/api/v1/sessions/${sessionId}?${qs}`, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return {};
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] sessionGet error: ${err.message}`);
    return {};
  }
}

async function sessionCommit(baseUrl, sessionId, { wait = false } = {}, params = {}) {
  try {
    const resp = await fetch(`${baseUrl}/api/v1/sessions/${sessionId}/commit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ wait, ...params }),
      signal: AbortSignal.timeout(wait ? 300_000 : 30_000),
    });
    if (!resp.ok) return {};
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] sessionCommit error: ${err.message}`);
    return {};
  }
}

async function sessionGetContext(baseUrl, sessionId, tokenBudget, params) {
  try {
    const qs = new URLSearchParams({ token_budget: String(tokenBudget), ...params }).toString();
    const resp = await fetch(`${baseUrl}/api/v1/sessions/${sessionId}/context?${qs}`, {
      signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) return {};
    return await resp.json();
  } catch (err) {
    console.error(`[og-memory] sessionGetContext error: ${err.message}`);
    return {};
  }
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
    const useHttp = !!cfg.memoryApiBaseUrl;
    const isLocal = cfg.mode === "local";

    api.logger.info(
      `og-memory: mode=${isLocal ? "local" : "remote"}, transport=${useHttp ? "http" : "subprocess"}, ` +
      (useHttp ? `url=${cfg.memoryApiBaseUrl}` : `pythonBin=${cfg.pythonBin}`),
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
            `  Or clone OpenViking: git clone https://github.com/c4pt0r/OpenViking.git`,
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

    function call(method, params) {
      return useHttp || isLocal
        ? callHttp(ogmemUrl, method, params)
        : callPythonSync(cfg.pythonBin, method, params);
    }

    function callAsync(method, params) {
      return useHttp || isLocal
        ? callHttp(ogmemUrl, method, params)
        : callPythonAsync(cfg.pythonBin, method, params);
    }

    function sessionParams(params) {
      return {
        accountId: params.accountId || cfg.accountId,
        userId: params.userId || cfg.userId,
        agentId: params.agentId || cfg.agentId,
      };
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

    api.registerContextEngine("og-memory-context-engine", () => ({
      info: {
        id: "og-memory-context-engine",
        name: "oG-Memory Context Engine",
        version: "1.0.0",
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

        // Backend already builds messages in the correct order:
        //   [user] Profile → original messages → [user] Working Set
        // Just pass through; no systemPromptAddition needed.
        return {
          messages: result.messages ?? params.messages ?? [],
          estimatedTokens: result.estimatedTokens ?? 0,
          systemPromptAddition: "",
        };
      },

      async compact(params) {
        const sessionId = params?.sessionId || "unknown";
        const tokenBudget = params?.tokenBudget || 128_000;

        if (useHttp || isLocal) {
          const sp = sessionParams(params);
          const commitResult = await sessionCommit(ogmemUrl, sessionId, { wait: true }, sp);
          if (!commitResult.archived) {
            return { ok: true, compacted: false, reason: commitResult.reason || "no_archive" };
          }
          const context = await sessionGetContext(ogmemUrl, sessionId, tokenBudget, sp);
          return {
            ok: true,
            compacted: true,
            result: {
              summary: context.latest_archive_overview || "",
              firstKeptEntryId: commitResult.archive_id || "",
              tokensBefore: params?.currentTokenCount ?? -1,
              tokensAfter: context.estimatedTokens ?? 0,
            },
          };
        }

        return call("compact", params);
      },

      async afterTurn(params) {
        const messages = params?.messages || [];

        // EXTRACT userId from tagged message
        const taggedUserId = extractUserIdFromMessages(messages);
        if (taggedUserId) {
          console.log(`[og-memory] afterTurn: extracted userId from tag: ${taggedUserId}`);
        }

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
        return call("dispose", {});
      },
    }));
  },
};
