# ENV.md — Environment Variables

> **⚠️ 重要**: oG-Memory 核心项目的环境变量配置请参阅上级目录的 [`../ENV.md`](../ENV.md)。
> 本文档仅说明 OpenClaw 插件特定的变量。

---

## Plugin Variables

| Variable | Default | Description |
|---|---|---|
| `MOCK_ENGINE_LOG_LEVEL` | `debug` | Log verbosity: `debug`, `info`, `warn`, `error`. |

> **注**: oG-Memory 核心环境变量（`OPENAI_API_KEY`, `CONTEXTENGINE_PROVIDER` 等）请参阅上级目录的 [`../ENV.md`](../ENV.md)。

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
├── extensions/
│   └── mock-context-engine/    # ← This plugin
│       ├── index.js            # Node.js entry point
│       ├── bridge/
│       │   └── memory_api.py   # Python bridge to oG-Memory (CLI entry + lifecycle + LLM)
│       ├── tests/
│       │   └── test_plugin_write.py
│       ├── openclaw.plugin.json
│       └── package.json
└── sessions/                  # Session data managed by OpenClaw
```

**Architecture:**
- `bridge/memory_api.py` handles CLI entry point, OpenClaw lifecycle methods, and LLM initialization
- Uses real OpenAI LLM for extraction (reads `OPENAI_API_KEY` from environment)
- `service/api.py` remains generic (no OpenClaw-specific logic)

## Example Configuration

### Via openclaw.json

编辑 `~/.openclaw/openclaw.json`，在插件配置中添加 `config.pythonBin` 字段：

```jsonc
{
  "plugins": {
    "entries": {
      "mock-context-engine": {
        "enabled": true,
        "config": {
          "pythonBin": "/home/aaa/miniconda3/envs/py11/bin/python"
        }
      }
    }
  }
}
```

然后重启 Gateway：
```bash
openclaw gateway restart
```

**注意事项：**
- Python 环境必须是 Python 3.11+ 且已安装 pyagfs 和 openai
- 如果未配置 `pythonBin`，将使用系统默认的 `python3`
- 可通过运行 `check-env.sh` 验证配置是否正确

## Verification

After configuration, verify the plugin is using the correct Python:

```bash
# Check gateway logs for Python path
journalctl --user -u openclaw-gateway -f | grep -i "python"

# Or check which Python is being used
openclaw agent --agent main -m "test message" 2>&1 | grep -i "python"
```

Look for successful oG-Memory integration logs:
```
[memory-api ...] [INFO] after_turn  committing N messages to oG-Memory
[memory-api ...] [INFO] commit_messages_to_memory session=... extracted=X completed=Y
```
