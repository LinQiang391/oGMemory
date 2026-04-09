import { execFileSync, execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MEMORY_API_PY = join(__dirname, "bridge", "memory_api.py");

// ---------------------------------------------------------------------------
// Transport: subprocess (legacy all-in-one) or HTTP (standalone oG-Memory)
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
      }
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

// Session API helpers (HTTP mode only)

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
    const ogmemUrl = api.pluginConfig?.memoryApiBaseUrl || process.env.OGMEM_API_URL || "";
    const useHttp = !!ogmemUrl;
    const pythonBin = api.pluginConfig?.pythonBin || process.env.OG_PYTHON_BIN || "python3";

    api.logger.info(`[og-memory] mode=${useHttp ? "http" : "subprocess"} ${useHttp ? "url=" + ogmemUrl : "pythonBin=" + pythonBin}`);

    function call(method, params) {
      return useHttp ? callHttp(ogmemUrl, method, params) : callPythonSync(pythonBin, method, params);
    }

    function callAsync(method, params) {
      return useHttp ? callHttp(ogmemUrl, method, params) : callPythonAsync(pythonBin, method, params);
    }

    function sessionParams(params) {
      return {
        accountId: params.accountId || api.pluginConfig?.accountId || "",
        userId: params.userId || api.pluginConfig?.userId || "",
        agentId: params.agentId || api.pluginConfig?.agentId || "",
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

    api.registerContextEngine("og-memory-context-engine", () => ({
      info: {
        id: "og-memory-context-engine",
        name: "oG-Memory Context Engine",
        version: "0.2.0",
      },

      async ingest(params) {
        return call("ingest", params);
      },

      async assemble(params) {
        const result = await call("assemble", params);
        return {
          messages: result.messages ?? params.messages ?? [],
          estimatedTokens: result.estimatedTokens ?? 0,
          systemPromptAddition: result.systemPromptAddition ?? "",
          memoryUserMessage: result.memoryUserMessage ?? "",
        };
      },

      async compact(params) {
        const sessionId = params?.sessionId || "unknown";
        const tokenBudget = params?.tokenBudget || 128_000;

        if (useHttp) {
          // HTTP mode: use session API for real compact
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

        // Subprocess fallback: delegate to server
        return call("compact", params);
      },

      async afterTurn(params) {
        const sessionId = params?.sessionId || "unknown";
        const messages = params?.messages || [];
        const prePromptCount = params?.prePromptMessageCount || 0;
        console.log(`[og-memory] afterTurn called, sessionId=${sessionId}, messages=${messages.length}`);

        if (useHttp) {
          // HTTP mode: incremental message push via session API
          const sp = sessionParams(params);
          const newMessages = messages.slice(prePromptCount);

          for (const msg of newMessages) {
            const role = msg.role || "user";
            const content = extractContentText(msg.content);
            if (content) await sessionAddMessage(ogmemUrl, sessionId, role, content, sp);
          }

          // Check pending_tokens
          const session = await sessionGet(ogmemUrl, sessionId, sp);
          const pendingTokens = session.pending_tokens ?? 0;
          const threshold = api.pluginConfig?.commitTokenThreshold ?? 20_000;

          if (pendingTokens < threshold) {
            return { ok: true, status: "accumulating", pending_tokens: pendingTokens };
          }

          // Threshold reached → async commit
          const result = await sessionCommit(ogmemUrl, sessionId, { wait: false }, sp);
          return { ok: true, task_id: result.task_id, status: result.status };
        }

        // Subprocess fallback: pass full messages to server
        return callAsync("after_turn", params);
      },

      async dispose() {
        return call("dispose", {});
      },
    }));
  },
};
