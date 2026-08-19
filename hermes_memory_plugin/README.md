# Hermes Agent integration

This directory provides a native Hermes `MemoryProvider` for oGMemory. It uses
oGMemory's own context composition, session buffering, memory extraction, and
commit lifecycle instead of emulating OpenViking's filesystem API.

## Install

Start oGMemory's HTTP server, then copy the provider into the active Hermes
profile:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins/ogmemory"
cp hermes_memory_plugin/ogmemory/* "${HERMES_HOME:-$HOME/.hermes}/plugins/ogmemory/"
```

Configure `${HERMES_HOME:-$HOME/.hermes}/config.yaml`:

```yaml
memory:
  provider: ogmemory
```

Configure `${HERMES_HOME:-$HOME/.hermes}/.env`:

```dotenv
OGMEMORY_ENDPOINT=http://127.0.0.1:8090
OGMEMORY_ACCOUNT_ID=default
OGMEMORY_USER_ID=default
OGMEMORY_AGENT_ID=hermes
# OGMEMORY_API_KEY=tenant-key
```

Verify:

```bash
hermes memory status
```

The provider recalls context before each turn, sends completed turns to
`/api/v1/after_turn`, and commits the oGMemory session on Hermes session end,
session switch, compression, or shutdown. It exposes `ogmemory_search` and
`ogmemory_remember` tools.
