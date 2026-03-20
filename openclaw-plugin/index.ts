export default function register(api: any) {
  api.registerTool({
    name: "og_memory_write",
    label: "Memory Write (oG-Memory)",
    description:
      "Write a memory entry to the oG-Memory graph storage (localfs-backed). " +
      "Use this when the user asks you to remember something, or when you " +
      "encounter important information that should be persisted.",
    parameters: {
      type: "object",
      properties: {
        content: { type: "string", description: "The memory content to store" },
        category: { type: "string", description: "Memory category: profile, preference, entity, event, etc." },
        tags: { type: "array", items: { type: "string" }, description: "Optional tags for categorizing the memory" },
      },
      required: ["content"],
    },
    async execute(_id: string, params: { content: string; category?: string; tags?: string[] }) {
      try {
        const result = await writeMemory(params.content, params.category, params.tags, api);
        return {
          content: [
            {
              type: "text" as const,
              text: `Memory stored successfully. ${result}`,
            },
          ],
        };
      } catch (err: any) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Failed to store memory: ${err.message}`,
            },
          ],
        };
      }
    },
  });

  // TODO: Phase N - Register og_memory_search tool
  // TODO: Phase N - Register og_memory_get tool
  // TODO: Phase N - Consider registering as memory slot plugin (kind: "memory")

  api.logger?.info?.("[og-memory] Plugin registered successfully");
}

async function writeMemory(
  content: string,
  category: string | undefined,
  tags: string[] | undefined,
  api: any
): Promise<string> {
  const config = api.config?.plugins?.entries?.["og-memory"]?.config ?? {};
  const pythonPath = config.pythonPath || "python";
  const projectRoot = config.projectRoot || "";
  const apiKey = config.apiKey || "";
  const openaiBaseUrl = config.openaiBaseUrl || process.env.OPENAI_BASE_URL || "";

  // Derive script path from projectRoot
  const scriptPath = `${projectRoot}/openclaw-plugin/bridge/write_memory.py`;

  const input = JSON.stringify({ content, category: category || "preference", tags: tags || [] });

  return new Promise((resolve, reject) => {
    const { execFile } = require("child_process");

    const child = execFile(
      pythonPath,
      [scriptPath],
      {
        env: {
          ...process.env,
          OG_MEMORY_ROOT: projectRoot,
          AGFS_BASE_URL: "http://localhost:8080",
          CONTEXTENGINE_PROVIDER: "openai",
          OPENAI_API_KEY: apiKey || process.env.OPENAI_API_KEY || "",
          OPENAI_BASE_URL: openaiBaseUrl,
          // Disable proxy for local AGFS requests
          http_proxy: "",
          https_proxy: "",
          HTTP_PROXY: "",
          HTTPS_PROXY: "",
        },
        timeout: 30000,
        maxBuffer: 1024 * 1024,
      },
      (error: any, stdout: string, stderr: string) => {
        if (stderr) {
          api.logger?.warn?.(`[og-memory] Python stderr: ${stderr}`);
        }
        if (error) {
          api.logger?.error?.(`[og-memory] Python exec failed: ${error.message} | code: ${error.code} | stderr: ${stderr}`);
          reject(new Error(`Python bridge failed: ${error.message}`));
          return;
        }
        api.logger?.info?.(`[og-memory] Python stdout: ${stdout.substring(0, 200)}`);
        try {
          const result = JSON.parse(stdout.trim());
          if (result.ok) {
            resolve(result.message || "Written successfully");
          } else {
            reject(new Error(result.error || "Unknown write error"));
          }
        } catch (parseErr: any) {
          api.logger?.error?.(`[og-memory] Parse failed: ${parseErr.message} | stdout: ${stdout}`);
          reject(new Error(`Failed to parse bridge output: ${stdout}`));
        }
      }
    );
    child.stdin?.on('error', (err: any) => {
      api.logger?.error?.(`[og-memory] Stdin error: ${err.message}`);
    });
    child.stdin?.write(input);
    child.stdin?.end();
  });
}
