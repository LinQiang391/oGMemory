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
        return { ok: true, compacted: false, reason: "delegated" };
      },

      async afterTurn(params) {
        console.log(`[og-memory] afterTurn called, sessionId=${params?.sessionId || "?"}, messages=${params?.messages?.length || 0}`);
        return callAsync("after_turn", params);
      },

      async dispose() {
        return call("dispose", {});
      },
    }));
  },
};
