import { execFileSync, execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MEMORY_API_PY = join(__dirname, "bridge", "memory_api.py");

function callPython(pythonBin, method, params) {
  const raw = execFileSync(pythonBin, [MEMORY_API_PY, method, JSON.stringify(params)], {
    encoding: "utf-8",
    timeout: 120_000,
  });
  const lines = raw.trim().split("\n");
  const jsonLine = lines[lines.length - 1];
  try {
    return JSON.parse(jsonLine);
  } catch {
    return {};
  }
}

function callPythonAsync(pythonBin, method, params) {
  return new Promise((resolve) => {
    execFile(
      pythonBin,
      [MEMORY_API_PY, method, JSON.stringify(params)],
      { encoding: "utf-8", timeout: 300_000 },
      (err, stdout, stderr) => {
        if (stderr) {
          for (const line of stderr.trim().split("\n")) {
            if (line.trim()) console.error(`[og-memory:${method}] ${line}`);
          }
        }
        if (stdout) {
          for (const line of stdout.trim().split("\n").slice(0, -1)) {
            if (line.trim()) console.log(`[og-memory:${method}] ${line}`);
          }
        }
        if (err) {
          console.error(`[og-memory] ${method} async error: ${err.message}`);
          resolve({});
          return;
        }
        const lines = (stdout || "").trim().split("\n");
        const jsonLine = lines[lines.length - 1];
        try {
          resolve(JSON.parse(jsonLine));
        } catch {
          resolve({});
        }
      }
    );
  });
}

export default definePluginEntry({
  id: "og-memory-context-engine",
  name: "oG-Memory Context Engine",
  description: "oG-Memory integration: semantic search with hierarchical retrieval",
  kind: "context-engine",

  register(api) {
    const pythonBin = api.pluginConfig?.pythonBin || process.env.OG_PYTHON_BIN || "python3";
    api.logger.info(`[og-memory] pythonBin=${pythonBin}`);

    api.registerContextEngine("og-memory-context-engine", () => ({
      info: {
        id: "og-memory-context-engine",
        name: "oG-Memory Context Engine",
        version: "0.1.0",
      },

      async ingest(params) {
        return callPython(pythonBin, "ingest", params);
      },

      async assemble(params) {
        const result = callPython(pythonBin, "assemble", params);
        return {
          messages: result.messages ?? params.messages ?? [],
          estimatedTokens: result.estimatedTokens ?? 0,
          systemPromptAddition: result.systemPromptAddition ?? "",
        };
      },

      async compact(params) {
        return { ok: true, compacted: false, reason: "delegated" };
      },

      async afterTurn(params) {
        console.log(`[og-memory] afterTurn called, sessionId=${params?.sessionId || "?"}, messages=${params?.messages?.length || 0}`);
        return callPythonAsync(pythonBin, "after_turn", params);
      },

      async dispose() {
        return callPython(pythonBin, "dispose", {});
      },
    }));
  },
});
