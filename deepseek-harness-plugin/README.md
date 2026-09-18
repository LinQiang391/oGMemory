# oGMemory plugin for DeepSeek Harness

This plugin connects a user's DeepSeek Harness installation to an available
oGMemory HTTP service and enables automatic cross-session memory recall.

Prerequisites:

- DSH `0.1.5-rc.2` is installed with a working model provider configuration.
- The oGMemory HTTP service address is known and reachable from DSH.

Plugin `0.1.1` supports DSH CLI `0.1.5-rc.2`, core packages `0.1.5-rc.2`,
and Cordis `4.0.2`. It automatically writes completed turns to oGMemory
and recalls relevant memories across sessions.

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

Completed turns trigger background extraction by default, including short
conversations. Keep DSH running until write-back, extraction, and indexing
finish. Once processing completes, the plugin automatically recalls relevant
memories in later sessions. Set `OGMEM_DSH_FORCE_EXTRACT=false` to use
server-side threshold batching.

## Verify the connection

Confirm that DSH loaded the plugin:

```powershell
dsh --profile web --dump-config
```

The output must include the `@opengauss/ogmemory-dsh-plugin` layer and the
`ogmemory-lifecycle` entry.

Then verify cross-session recall in the DSH UI:

1. Create a session and send: `Remember my test marker OGMEM-DSH-TEST-001.
   Reply only "Remembered".` Use a fresh marker for each test.
2. After the reply, keep DSH running for background extraction and indexing.
   Allow 1–2 minutes initially; processing time depends on the model service
   and indexing speed.
3. Create a new session with the same Account ID, User ID, and Agent ID. Send:
   `What test marker did I ask you to remember in the previous session?
   Reply with the marker only.`
4. Confirm that DSH returns `OGMEM-DSH-TEST-001` (or your chosen marker).
   Expand the reply's process section to see
   `Context injection ogmemory-lifecycle`. The `Trajectory` tab also shows
   the injected `<ogmemory-context>` content.

If background processing is still underway, allow more time and repeat step 3
in another new session.

Both the oGMemory context injection and the correct cross-session answer must
be present for the connection to be considered successful.

Run unit and contract tests with `npm install && npm test`. Set
`OGMEM_E2E=1` to include the real-backend marker write/recall gate.

For Chinese instructions, see [INSTALL-ZH.md](./INSTALL-ZH.md).

## Uninstall

```powershell
dsh plugin --profile web remove @opengauss/ogmemory-dsh-plugin
```
