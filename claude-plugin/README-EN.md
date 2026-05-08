# oG-Memory — Claude Code plugin (English)

Sync Claude Code sessions to oG-Memory, inject retrieved memory on each user prompt, report successful tool I/O to the same session buffer as the transcript, and **reuse one HTTP/identity module** for all hook scripts.

## What the hooks do

| Hook | Script | API | Notes |
|------|--------|-----|--------|
| `PostToolUse` | `call_add_session_message.py` | `POST /api/v1/sessions/…/messages` | Only **mutating** tools (`Write`, `Edit`, `MultiEdit`, `Bash`, `NotebookEdit`) — matcher + script allowlist so Read/Glob/Grep noise does not flood the session buffer. Appends a `role: tool` summary (truncated). |
| `UserPromptSubmit` | `call_compose.py` | `POST /api/v1/compose` | RAG + profile → `additionalContext` for the model. |
| `Stop` | `call_after_turn.py` (async) or `stop_detach.sh` | `POST /api/v1/after_turn` | Incremental JSONL → server (byte offset sidecar). Supports headless `-p` mode. |
| `PreCompact` | `call_after_turn.py` | `POST /api/v1/after_turn` | Same script; syncs before compaction. |

Shared code: `scripts/ogm_plugin_request.py` (`base_api_url`, `http_plugin_headers`, `base_ctx`) used by the three `call_*.py` entrypoints. Scripts also prepend the `scripts` directory to `sys.path` so `import ogm_plugin_request` works when the CWD is not the scripts folder.

## Config / auth (optional)

- **Base URL:** `OG_MEMORY_URL` (e.g. `http://127.0.0.1:8090`).

- **Same key as OpenClaw** (`openclaw_context_engine_plugin` → `authApiKey` / `process.env.OG_AUTH_API_KEY`):

  Set **`OG_AUTH_API_KEY`** in the **environment of the `claude` process** (shell export, service env, etc.).  
  When set, requests also send `X-API-Key`, `X-Account-ID`, and `X-User-ID` (identity from `OgMemConfig` when the repo is on `PYTHONPATH`, else `OG_*` / `OG_MEMORY_*` as implemented in `ogm_plugin_request.py`).

  If your server has no key-based auth, leave it unset. **Do not commit secrets.**

- Unit tests: `pytest tests/unit/claude_plugin/test_plugin_scripts.py -q` (from repo root).

## Skills (slash commands in Claude Code)

| Command | Role |
|--------|------|
| `/og-compose <query>` | Calls compose and prints full context (see skill). |
| `/og-add-history` | Imports project chat history into oG-Memory (see skill). |

## Install

**Marketplace:** e.g. add the opengauss `oG-Memory` repo, install `og-memory@og-memory`, then `/reload-plugins` (see marketplace docs in repo).

**Local dev:**

```bash
claude --plugin-dir ./claude-plugin
# optional: --debug
```

**Prereqs:** reachable oG-Memory HTTP API; Python 3; hook scripts are stdlib-only.

## Debugging

- `/reload-plugins` — should show the plugin and hook count.
- With `--debug`, search for `[call_compose]`, `[call_after_turn]`, and `[call_add_session_message]` in hook output.
- Ingest success is reflected in `*.jsonl.ingest-offset` next to the session transcript (see `call_after_turn` sidecar).
