import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MOCK_ENGINE_PY = join(__dirname, "mock_engine.py");

/**
 * Call mock_engine.py with a method name and JSON-serialised params.
 * Returns the parsed JSON stdout from the Python process.
 */
function callPython(pythonBin, method, params) {
  const raw = execFileSync(pythonBin, [MOCK_ENGINE_PY, method, JSON.stringify(params)], {
    encoding: "utf-8",
    timeout: 30_000,
  });

  // The Python script prints a JSON line as the *last* line of stdout.
  // Everything before that is debug logging (also printed to stdout for
  // visibility; a real engine would use stderr).
  const lines = raw.trim().split("\n");
  const jsonLine = lines[lines.length - 1];
  try {
    return JSON.parse(jsonLine);
  } catch {
    // If Python didn't return valid JSON, return a safe default.
    return {};
  }
}

/**
 * OpenClaw plugin register function.
 * @param {import("openclaw/plugin-sdk").PluginAPI} api
 */
export default function register(api) {
  const pluginConfig = api.getConfig?.() ?? {};
  const pythonBin = pluginConfig.pythonBin || process.env.MOCK_ENGINE_PYTHON_BIN || "python3";

  api.registerContextEngine("mock-context-engine", () => ({
    // ── info ────────────────────────────────────────────────
    info: {
      id: "mock-context-engine",
      name: "Mock Context Engine",
      version: "0.1.0",
      ownsCompaction: true,
    },

    // ── required lifecycle ──────────────────────────────────

    async ingest(params) {
      return callPython(pythonBin, "ingest", params);
    },

    async assemble(params) {
      const result = callPython(pythonBin, "assemble", params);
      // Guarantee the contract expected by OpenClaw:
      return {
        messages: result.messages ?? params.messages ?? [],
        estimatedTokens: result.estimatedTokens ?? 0,
        systemPromptAddition: result.systemPromptAddition ?? "",
      };
    },

    async compact(params) {
      return callPython(pythonBin, "compact", params);
    },

    // ── optional lifecycle ──────────────────────────────────

    async bootstrap(params) {
      return callPython(pythonBin, "bootstrap", params);
    },

    // async ingestBatch(params) {
    //   return callPython(pythonBin, "ingest_batch", params);
    // },

    async afterTurn(params) {
      return callPython(pythonBin, "after_turn", params);
    },

    async prepareSubagentSpawn(params) {
      return callPython(pythonBin, "prepare_subagent_spawn", params);
    },

    async onSubagentEnded(params) {
      return callPython(pythonBin, "on_subagent_ended", params);
    },

    async dispose() {
      return callPython(pythonBin, "dispose", {});
    },
  }));
}
