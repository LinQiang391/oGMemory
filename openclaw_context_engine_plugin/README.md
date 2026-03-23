# Mock Context Engine — OpenClaw Plugin

A minimal context engine plugin for OpenClaw. All lifecycle methods (`ingest`, `assemble`, `compact`, etc.) are implemented as **stubs that print every parameter to stdout**, making it easy to verify the integration is wired up before writing real logic.

## Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.10
- **OpenClaw** installed and running ([Install guide](https://docs.openclaw.ai/install))

## Install

### 1. Clone / copy the plugin

```bash
git clone <this-repo> mock-context-engine
cd mock-context-engine
```

### 2. Install into OpenClaw (local dev mode)

```bash
openclaw plugins install -l ./mock-context-engine
```

### 3. Configure `openclaw.json`

Open `~/.openclaw/openclaw.json` and set the context engine slot:

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "mock-context-engine"
    },
    "entries": {
      "mock-context-engine": {
        "enabled": true,
        "pythonBin": "python3",   // optional, defaults to python3
        "logLevel": "debug"       // optional
      }
    }
  }
}
```

### 4. Restart the gateway

```bash
openclaw restart
# or stop + start
```

### 5. Verify

```bash
openclaw doctor
```

You should see `mock-context-engine` listed as the active context engine. Send a message through any channel — the gateway logs will show the printed params from each lifecycle call.

## Project Structure

```
mock-context-engine/
├── index.js                   # OpenClaw plugin entry — registers the engine, bridges to Python
├── mock_engine.py             # Python mock — prints params, returns stub responses
├── bridge/
│   ├── __init__.py
│   └── memory_api.py          # Mock memory CRUD / search / embed methods
├── openclaw.plugin.json       # Plugin manifest (required by OpenClaw)
├── package.json
├── ENV.md                     # Environment variable reference
└── README.md                  # ← You are here
```

## How It Works

```
OpenClaw gateway
  │
  ├─ ingest()  ──►  index.js  ──►  python3 mock_engine.py ingest '{...}'
  ├─ assemble()──►  index.js  ──►  python3 mock_engine.py assemble '{...}'
  ├─ compact() ──►  index.js  ──►  python3 mock_engine.py compact '{...}'
  └─ ...           (same pattern for all lifecycle methods)
```

Each Python call:

1. Prints a timestamped log line with all received params.
2. Calls into `bridge/memory_api.py` mock methods where applicable (which also print their args).
3. Outputs a JSON result on the last line of stdout — `index.js` parses this as the return value.

## Switching Back

To revert to the built-in engine:

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "legacy"
    }
  }
}
```

Then restart the gateway.

## Next Steps

- Replace the print-only stubs in `mock_engine.py` with real logic (vector search, summarisation, etc.).
- Swap `bridge/memory_api.py` for calls to your actual memory backend (Qdrant, Pinecone, SQLite, etc.).
- For long-running workloads, consider replacing the `execFileSync` bridge with a persistent Python HTTP server or Unix socket.
