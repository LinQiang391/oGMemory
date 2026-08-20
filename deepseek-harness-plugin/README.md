# oGMemory plugin for DeepSeek Harness

This plugin connects a user's DeepSeek Harness installation to an available
oGMemory HTTP service and enables automatic cross-session memory recall.

Prerequisites:

- DSH `0.1.0-rc.6` is installed with a working model provider configuration.
- The oGMemory HTTP service address is known and reachable from DSH.

## Install

```powershell
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

## Connect DSH to oGMemory

Set the oGMemory service address and the memory identity used by DSH:

```powershell
$env:OGMEM_URL='http://127.0.0.1:8090'
$env:OGMEM_ACCOUNT_ID='acct-demo'
$env:OGMEM_USER_ID='user-dsh'
$env:OGMEM_AGENT_ID='deepseek-harness'
dsh --profile web
```

If the oGMemory HTTP API requires authentication, also set:

```powershell
$env:OGMEM_DSH_API_KEY='<oGMemory API key>'
```

`OGMEM_DSH_API_KEY` is the oGMemory HTTP API credential, not a model-provider
credential.

Retryable write failures are persisted before delivery under
`~/.ogmemory/dsh-pending`, retried in session order while DSH remains running,
and replayed at the next start if shutdown happens first. Set
`OGMEM_DSH_PENDING_DIR` to use another private runtime directory. Session
identities use separate hashed subdirectories. Session shutdown drains queued
turns before calling oGMemory `/api/v1/dispose`.

Completed turns are force-extracted by default, making short memories available
to a new session immediately. Set `OGMEM_DSH_FORCE_EXTRACT=false` only when
server-side threshold batching is preferred over immediate cross-session recall.

## Native oGMemory tools

The plugin registers the native read-only `ogm_query_memory` and
`ogm_read_memory` tools. They use `ogmemory://` URIs and the same contracts as
the oGMemory MCP server. Identity and session fields always come from plugin
configuration and the calling DSH agent, never model arguments.
Set `OGMEM_DSH_TOOLS=false` when the same DSH profile already loads these tools
through an oGMemory MCP connection.

Model-initiated writes and deletes are disabled by default. Enable them with
`OGMEM_DSH_WRITE_TOOLS=true` or `OGMEM_DSH_DELETE_TOOLS=true`; the oGMemory
server must also enable the corresponding MCP write/delete toolset. Delete
accepts only an exact `ogmemory://` item URI and performs oGMemory soft archive.

Recall remains based on `/compose`: its identity, episodic, session, and
retrieved slots are injected on the first pre-step. The plugin does not create
a second profile/session model.

## Verify the connection

Confirm that DSH loaded the plugin:

```powershell
dsh --profile web --dump-config
```

The output must include the `@opengauss/ogmemory-dsh-plugin` layer and the
`ogmemory-lifecycle` entry.

Then verify cross-session recall in the DSH UI:

1. Create a session and ask DSH to remember a unique marker.
2. Wait for the reply, then create a new session.
3. Ask DSH for the marker from the previous session.
4. Confirm that the page shows `Context injection ogmemory-lifecycle` and that
   DSH returns the exact marker.

Both the oGMemory context injection and the correct cross-session answer must
be present for the connection to be considered successful.

Run unit and contract tests with `npm install && npm test`. Set
`OGMEM_E2E=1` to include the real-backend marker write/recall gate.

For Chinese instructions, see [INSTALL-ZH.md](./INSTALL-ZH.md).

## Uninstall

```powershell
dsh plugin --profile web remove @opengauss/ogmemory-dsh-plugin
```
