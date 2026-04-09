# ENV.md — Environment Variables

> **⚠️ 重要**: oG-Memory 核心项目的环境变量配置请参阅上级目录的 [`../ENV.md`](../ENV.md)。
> 本文档仅说明 OpenClaw 插件特定的变量。

---

## Plugin Variables

| Variable | Default | Description |
|---|---|---|
| `MOCK_ENGINE_PYTHON_BIN` | `python3` | Python interpreter path. Override if using a virtualenv or specific version. |
| `MOCK_ENGINE_LOG_LEVEL` | `debug` | Log verbosity: `debug`, `info`, `warn`, `error`. |

> **注**: `AGFS_BASE_URL` 等核心变量由 oG-Memory 主项目管理，详见 [`../ENV.md`](../ENV.md)。

## OpenClaw Variables

These are standard OpenClaw env vars that affect plugin loading and runtime:

| Variable | Default | Description |
|---|---|---|
| `OPENCLAW_HOME` | `~/.openclaw` | Root config directory. `openclaw.json` lives here. |
| `OPENCLAW_GATEWAY_PORT` | `3000` | Port the OpenClaw gateway listens on. |
| `OPENCLAW_LOG_LEVEL` | `info` | Gateway-wide log level. |
| `OPENCLAW_PLUGIN_PATH` | *(none)* | Extra colon-separated directories to scan for plugins. Useful for local dev. |
| `OPENCLAW_DISABLE_TELEMETRY` | `false` | Set to `true` to disable anonymous telemetry. |

## Path Conventions

```
~/.openclaw/
├── openclaw.json              # Main config (plugin slots configured here)
├── plugins/
│   └── mock-context-engine/   # ← This plugin after `openclaw plugins install -l`
│       ├── index.js
│       ├── mock_engine.py
│       ├── bridge/
│       │   ├── __init__.py
│       │   └── memory_api.py
│       ├── openclaw.plugin.json
│       └── package.json
└── sessions/                  # Session data managed by OpenClaw
```

## Example `.env`

```bash
# .env (place in project root or export before starting gateway)
MOCK_ENGINE_PYTHON_BIN=/usr/bin/python3
MOCK_ENGINE_LOG_LEVEL=debug
MOCK_MEMORY_API_BASE_URL=http://localhost:9200
```
