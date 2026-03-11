import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as path from "node:path";
import * as fs from "node:fs";

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { ogMemoryConfigSchema, type ParsedOgMemoryConfig } from "./config.js";

const COMMAND_TIMEOUT_MS = 30_000;
const CACHE_TTL_MS = 60_000;

interface CacheEntry {
  content: string;
  timestamp: number;
}

const recentMemoriesCache = new Map<string, CacheEntry>();

function getProjectRoot(): string {
  const envRoot = process.env.OG_MEMORY_PROJECT_ROOT;
  if (envRoot && fs.existsSync(path.join(envRoot, "ogmemory_cli.py"))) {
    return path.resolve(envRoot);
  }
  const currentFile = fileURLToPath(import.meta.url);
  const dir = path.dirname(currentFile);
  const defaultRoot = path.resolve(dir, "..");
  if (fs.existsSync(path.join(defaultRoot, "ogmemory_cli.py"))) {
    return defaultRoot;
  }
  const dockerRoot = "/workspace/oG-Memory";
  if (fs.existsSync(path.join(dockerRoot, "ogmemory_cli.py"))) {
    return dockerRoot;
  }
  return defaultRoot;
}

function getPythonPathSeparator(): string {
  return process.platform === "win32" ? ";" : ":";
}

function getCacheKey(): string {
  return process.cwd();
}

function getCachedMemory(): CacheEntry | null {
  const key = getCacheKey();
  const entry = recentMemoriesCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
    recentMemoriesCache.delete(key);
    return null;
  }
  return entry;
}

function setCachedMemory(content: string): void {
  recentMemoriesCache.set(getCacheKey(), { content, timestamp: Date.now() });
}

function clearCache(): void {
  recentMemoriesCache.clear();
}

function buildEnv(cfg: ParsedOgMemoryConfig, projectRoot: string): Record<string, string | undefined> {
  const pathSep = getPythonPathSeparator();
  return {
    ...process.env,
    PYTHONPATH: `${projectRoot}${pathSep}${process.env.PYTHONPATH || ""}`,
    ...(cfg.dbHost && { OG_DB_HOST: cfg.dbHost }),
    OG_DB_PORT: String(cfg.dbPort),
    ...(cfg.dbName && { OG_DB_NAME: cfg.dbName }),
    ...(cfg.dbUser && { OG_DB_USER: cfg.dbUser }),
    ...(cfg.dbPassword && { OG_DB_PASSWORD: cfg.dbPassword }),
    ...(cfg.embeddingProvider && { OG_EMBEDDING_PROVIDER: cfg.embeddingProvider }),
    ...(cfg.embeddingModel && { OG_EMBEDDING_MODEL: cfg.embeddingModel }),
    ...(cfg.openaiApiKey && { OPENAI_API_KEY: cfg.openaiApiKey }),
    ...(cfg.openaiBaseUrl && { OPENAI_BASE_URL: cfg.openaiBaseUrl }),
    OG_CHUNK_SIZE: String(cfg.chunkSize),
    OG_CHUNK_OVERLAP: String(cfg.chunkOverlap),
  };
}

function validateConfig(cfg: ParsedOgMemoryConfig): boolean {
  if (!cfg.openaiApiKey && !process.env.OPENAI_API_KEY) {
    return false;
  }
  return true;
}

/** Run ogmemory_cli via spawn (avoids exec string/array bugs and shell escaping). */
async function runOgMemoryCommand(
  projectRoot: string,
  cfg: ParsedOgMemoryConfig,
  args: string[],
  logger: OpenClawPluginApi["logger"],
): Promise<string> {
  if (!validateConfig(cfg)) {
    throw new Error("Invalid configuration: OPENAI_API_KEY required");
  }
  const pythonPath = process.env.PYTHON_PATH || "python3";
  const cliPath = path.join(projectRoot, "ogmemory_cli.py");
  const env = buildEnv(cfg, projectRoot);

  logger.debug?.(`ogmemory: executing ${pythonPath} ${cliPath} ${args.join(" ")}`);

  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath, [cliPath, ...args], {
      cwd: projectRoot,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    child.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
    child.stderr?.on("data", (chunk: Buffer) => {
      stderrChunks.push(chunk);
      const s = chunk.toString().trim();
      if (s) logger.debug?.(`[ogmemory] ${s}`);
    });
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`ogmemory command timed out after ${COMMAND_TIMEOUT_MS}ms`));
    }, COMMAND_TIMEOUT_MS);
    child.on("error", (err: Error) => {
      clearTimeout(timeout);
      reject(err);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timeout);
      const stderrStr = Buffer.concat(stderrChunks).toString().trim();
      if (stderrStr) logger.warn?.(`ogmemory stderr: ${stderrStr}`);
      if (code === 0) {
        resolve(Buffer.concat(stdoutChunks).toString());
      } else {
        const errMsg = stderrStr
          ? `ogmemory command failed (code=${code}): ${stderrStr.slice(-500)}`
          : `ogmemory command failed (code=${code}${signal ? `, signal=${signal}` : ""})`;
        reject(new Error(errMsg));
      }
    });
  });
}

/** Check Python availability (spawn-based, no exec array bug). */
async function checkPythonExecutable(logger: OpenClawPluginApi["logger"]): Promise<boolean> {
  const pythonPath = process.env.PYTHON_PATH || "python3";
  return new Promise((resolve) => {
    const child = spawn(pythonPath, ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      resolve(false);
    }, 5000);
    child.on("error", () => {
      clearTimeout(timeout);
      resolve(false);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      resolve(code === 0);
    });
  });
}

const ogMemoryPlugin = {
  id: "ogmemory",
  name: "oG-Memory",
  description: "openGauss-based semantic memory for OpenClaw",
  kind: "memory" as const,
  configSchema: ogMemoryConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = ogMemoryConfigSchema.parse(api.pluginConfig);
    const projectRoot = getProjectRoot();

    if (!validateConfig(cfg)) {
      api.logger.warn?.("ogmemory: OPENAI_API_KEY not configured; tools may fail.");
    }

    api.registerTool(
      {
        name: "ogmemory_search",
        label: "oG-Memory Search",
        description: "Search semantic memory using openGauss vector database.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query text" }),
          topK: Type.Optional(
            Type.Number({ description: "Number of results to return (default: 5)" }),
          ),
          hybrid: Type.Optional(
            Type.Boolean({ description: "Use hybrid search vector + BM25 (default: true)" }),
          ),
        }),
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const { query } = params as { query: string };
          const topK =
            typeof (params as { topK?: number }).topK === "number"
              ? Math.max(1, Math.min(50, Math.floor((params as { topK: number }).topK)))
              : 5;
          const hybrid = (params as { hybrid?: boolean }).hybrid !== false;
          const args = ["search", query, "--json-output", "--top-k", String(topK)];
          if (hybrid) args.push("--hybrid");
          try {
            const result = await runOgMemoryCommand(projectRoot, cfg, args, api.logger);
            const parsed = JSON.parse(result) as Array<{ id?: string; text?: string; score?: number; source?: string }>;
            const lines =
              Array.isArray(parsed) && parsed.length > 0
                ? parsed
                    .map(
                      (r, i) =>
                        `${i + 1}. [${typeof r.score === "number" ? r.score.toFixed(4) : "—"}] ${(r.text ?? "").slice(0, 200)}${(r.text?.length ?? 0) > 200 ? "..." : ""}`,
                    )
                    .join("\n")
                : "No results.";
            return {
              content: [{ type: "text" as const, text: `oG-Memory search results:\n${lines}` }],
              details: { count: Array.isArray(parsed) ? parsed.length : 0, raw: parsed },
            };
          } catch (err) {
            api.logger.warn?.(`ogmemory_search failed: ${String(err)}`);
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Search failed or no results: ${err instanceof Error ? err.message : String(err)}`,
                },
              ],
              details: { error: String(err) },
            };
          }
        },
      },
      { name: "ogmemory_search" },
    );

    api.registerTool(
      {
        name: "ogmemory_add",
        label: "oG-Memory Add",
        description: "Add a memory to openGauss vector database.",
        parameters: Type.Object({
          text: Type.String({ description: "Memory text to add" }),
          metadata: Type.Optional(
            Type.Record(Type.String(), Type.Union([Type.String(), Type.Number(), Type.Boolean()]), {
              description: "Optional metadata key-value pairs",
            }),
          ),
        }),
        async execute(_toolCallId: string, params: Record<string, unknown>) {
          const { text } = params as { text: string };
          const metadata = (params as { metadata?: Record<string, string | number | boolean> }).metadata;
          const args = ["add", text];
          if (metadata && typeof metadata === "object") {
            for (const [k, v] of Object.entries(metadata)) {
              args.push("--metadata", `${k}=${String(v)}`);
            }
          }
          const result = await runOgMemoryCommand(projectRoot, cfg, args, api.logger);
          return {
            content: [{ type: "text" as const, text: result.trim() || "Memory added." }],
            details: { action: "add" },
          };
        },
      },
      { name: "ogmemory_add" },
    );

    api.registerTool(
      {
        name: "ogmemory_stats",
        label: "oG-Memory Stats",
        description: "Get statistics about the memory database.",
        parameters: Type.Object({}),
        async execute() {
          const result = await runOgMemoryCommand(projectRoot, cfg, ["stats"], api.logger);
          return {
            content: [{ type: "text" as const, text: result.trim() }],
            details: { action: "stats" },
          };
        },
      },
      { name: "ogmemory_stats" },
    );

    api.registerTool(
      {
        name: "ogmemory_clear_cache",
        label: "oG-Memory Clear Cache",
        description: "Clear the in-process memory cache used for file-based context.",
        parameters: Type.Object({}),
        async execute() {
          clearCache();
          return {
            content: [{ type: "text" as const, text: "Cache cleared." }],
            details: { action: "clear_cache" },
          };
        },
      },
      { name: "ogmemory_clear_cache" },
    );

    if (typeof api.registerCli === "function") {
      api.registerCli(
        (arg: { program: unknown }) => {
          const program = arg.program as {
            command: (n: string, d?: string) => {
              description: (d: string) => {
                command: (s: string, d?: string) => { option: (...a: unknown[]) => unknown; action: (fn: (...a: unknown[]) => void) => unknown };
                option: (...a: unknown[]) => unknown;
                action: (fn: (...a: unknown[]) => void) => unknown;
              };
            };
          };
          type SubCmd = { option: (...a: unknown[]) => SubCmd; action: (fn: (...a: unknown[]) => void) => SubCmd };
          type CliCmd = { command: (s: string, d?: string) => SubCmd };
          const ogmemoryCmd = program.command("ogmemory").description("oG-Memory commands") as CliCmd;
          ogmemoryCmd
            .command("search <query>", "Search memories")
            .option("-k, --top-k <n>", "Number of results", "5")
            .option("--hybrid", "Use hybrid search")
            .action(async (...a: unknown[]) => {
              const query = String(a[0] ?? "");
              const options = (a[1] ?? {}) as { topK?: string; hybrid?: boolean };
              const args = ["search", query];
              if (options?.topK) args.push("--top-k", String(options.topK));
              if (options?.hybrid) args.push("--hybrid");
              console.log(await runOgMemoryCommand(projectRoot, cfg, args, api.logger));
            });
          ogmemoryCmd
            .command("index <path>", "Index markdown files")
            .action(async (...a: unknown[]) => {
              const filePath = String(a[0] ?? "");
              console.log(await runOgMemoryCommand(projectRoot, cfg, ["index", filePath], api.logger));
            });
          ogmemoryCmd
            .command("stats", "Show statistics")
            .action(async () => {
              console.log(await runOgMemoryCommand(projectRoot, cfg, ["stats"], api.logger));
            });
          ogmemoryCmd
            .command("add <text>", "Add a memory")
            .option("-m, --metadata <items>", "Metadata as key=value pairs")
            .action(async (...a: unknown[]) => {
              const text = String(a[0] ?? "");
              const options = (a[1] ?? {}) as { metadata?: string[] };
              const args = ["add", text];
              if (options?.metadata?.length) {
                options.metadata.forEach((m: string) => args.push("--metadata", m));
              }
              console.log(await runOgMemoryCommand(projectRoot, cfg, args, api.logger));
            });
          ogmemoryCmd
            .command("reset", "Reset all data")
            .option("--yes", "Confirm reset")
            .action(async (...a: unknown[]) => {
              const options = (a[0] ?? {}) as { yes?: boolean };
              if (!options?.yes) {
                console.log("Error: --yes flag is required to reset");
                return;
              }
              console.log(await runOgMemoryCommand(projectRoot, cfg, ["reset", "--yes"], api.logger));
            });
          ogmemoryCmd
            .command("clear-cache", "Clear memory cache")
            .action(() => {
              clearCache();
              console.log("Cache cleared");
            });
        },
        { commands: ["ogmemory"] },
      );
    }

    if (typeof api.registerCommand === "function") {
      api.registerCommand({
        name: "memory",
        description: "Search oG-Memory for relevant context",
        acceptsArgs: true,
        requireAuth: true,
        handler: async (ctx: { args?: string }) => {
          const query = ctx.args?.trim();
          if (!query) return { text: "Usage: /memory <query>" };
          try {
            const result = await runOgMemoryCommand(
              projectRoot,
              cfg,
              ["search", query, "--top-k", "3"],
              api.logger,
            );
            return { text: result };
          } catch (err) {
            api.logger.error?.(`Memory search failed: ${String(err)}`);
            return { text: `Memory search failed: ${String(err)}` };
          }
        },
      });
    }

    api.on("before_agent_start", async (event?: unknown) => {
      const cached = getCachedMemory();
      if (cached?.content) {
        api.logger.debug?.("ogmemory: using cached file-based memories");
        return { prependContext: cached.content };
      }
      try {
        const memoryPath = path.resolve(process.cwd(), cfg.memoryDir);
        if (!fs.existsSync(memoryPath)) {
          setCachedMemory("");
          return {};
        }
        const allFiles = fs.readdirSync(memoryPath).filter((f) => f.endsWith(".md"));
        if (allFiles.length === 0) {
          setCachedMemory("");
          return {};
        }
        const recentFiles = allFiles
          .map((f) => ({
            name: f,
            mtime: fs.statSync(path.join(memoryPath, f)).mtimeMs,
          }))
          .sort((a, b) => b.mtime - a.mtime)
          .slice(0, 2)
          .map((f) => f.name);
        let recentMemories = "";
        for (const file of recentFiles) {
          const filePath = path.join(memoryPath, file);
          const content = fs.readFileSync(filePath, "utf-8");
          const lines = content.split("\n").slice(-30);
          recentMemories += `\n\nRecent memories from ${file}:\n${lines.join("\n")}`;
        }
        setCachedMemory(recentMemories);
        if (recentMemories) {
          api.logger.debug?.(`ogmemory: loaded ${recentFiles.length} memory files`);
          return { prependContext: recentMemories };
        }
      } catch (err) {
        api.logger.error?.(`ogmemory: failed to load file memories: ${String(err)}`);
        setCachedMemory("");
      }
      return {};
    });

    api.on("shutdown", () => {
      api.logger.info?.("ogmemory: cleaning up");
      clearCache();
    });

    checkPythonExecutable(api.logger)
      .then((ok) => {
        if (ok) api.logger.info?.("ogmemory: plugin loaded (Python available)");
        else api.logger.warn?.("ogmemory: plugin loaded but Python not found; tools may fail.");
      })
      .catch((err) => {
        api.logger.warn?.(`ogmemory: Python check failed: ${String(err)}`);
      });
  },
};

export default ogMemoryPlugin;
