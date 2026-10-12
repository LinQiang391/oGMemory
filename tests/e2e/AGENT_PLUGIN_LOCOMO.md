# Native agent LoCoMO evaluation

These runners execute the real agent host and load the repository's native plugin. They do not call plugin hook functions directly.

| Runner | Host boundary | Plugin boundary |
| --- | --- | --- |
| `run_openclaw_plugin_locomo.py` | OpenClaw gateway `/v1/responses` | `og-memory-context-engine` |
| `run_opencode_plugin_locomo.py` | `opencode run --format json` | OpenCode JavaScript plugin |
| `run_claude_code_plugin_locomo.py` | `claude -p --plugin-dir` | Claude Code hooks |
| `run_hermes_plugin_locomo.py` | `hermes -z` | Hermes memory provider |
| `run_dsh_plugin_locomo.py` | `dsh --profile headless` | DSH plugin package |
| `run_codex_plugin_locomo.py` | `codex exec --json --ephemeral` | Codex lifecycle hooks |

Each ingest session and each QA question uses a fresh host session. The same oGMemory account and user identity are retained across sessions. A run fails when the plugin write cannot be recalled from oGMemory.
Transient host failures are retried with bounded backoff; each attempt has a separate host log.
CLI hosts may use `--qa-parallel N`; every question still runs in a fresh real host process.
Rate limits use a fixed `--rate-limit-retry-delay` (30 seconds by default).
Ingest is writable. QA is read-only by default: compose reaches oGMemory, while lifecycle
writeback (`after_turn`, `commit`, `compact`, and `dispose`) is acknowledged without
mutating benchmark memory. `--allow-qa-writeback` is only for lifecycle debugging.

## Requirements

- Running oGMemory HTTP server with `retrieval.hybrid_enabled: false`
- OpenAI-compatible LLM and judge credentials
- Installed target host CLI
- `pnpm` for DSH plugin installation
- Codex CLI with plugin support for Codex evaluation

Set the LLM key without placing it on the command line:

```bash
export OGMEM_API_KEY='...'
```

Example:

```bash
python tests/e2e/run_openclaw_plugin_locomo.py locomo10.json \
  --output-dir results/openclaw-sample0 \
  --sample 0 \
  --ogmemory-url http://127.0.0.1:8090 \
  --llm-base-url https://api.example.com/v1 \
  --llm-model model-name \
  --judge-base-url https://api.example.com/v1 \
  --judge-model model-name
```

Replace the runner filename for another host. Use `--runtime-root` when the repository filesystem does not provide private Unix permissions required by OpenClaw.

Providers that expose separate OpenAI and Anthropic endpoints require the Claude endpoint explicitly:

```bash
python tests/e2e/run_claude_code_plugin_locomo.py locomo10.json \
  --output-dir results/claude-code-sample0 \
  --llm-base-url https://provider.example/openai/v1 \
  --claude-base-url https://provider.example/anthropic \
  --claude-auth-token \
  --llm-model model-name
```

Use `--openclaw-provider-api openai-completions` when the model provider's Responses API cannot accept OpenClaw's multi-message context payload.

Artifacts include the invocation manifest, redacted host logs, ingest replies, QA CSV, judged CSV, write verification, and accuracy summary.

## Evaluation boundary

[AntTrail](https://gitcode.com/datagallery/AntTrail/tree/dev)'s `tests/e2e/run_benchmark.sh` benchmarks oGMemory by direct HTTP. Its documented OpenClaw strategy delegates agent evaluation to [openclaw-eval](https://github.com/ZaynJarvis/openclaw-eval), which sends ingest and QA prompts through the real OpenClaw Responses API. These runners apply that real-host boundary to all six supported agents.
