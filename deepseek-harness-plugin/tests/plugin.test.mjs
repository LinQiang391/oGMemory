import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { apply } from '../index.js'

test('integrates compose context and after-turn writeback through lifecycle events', async t => {
  const pendingDir = await mkdtemp(join(tmpdir(), 'ogmemory-dsh-plugin-'))
  t.after(() => rm(pendingDir, { recursive: true, force: true }))
  const requests = []
  const originalFetch = globalThis.fetch
  globalThis.fetch = async (url, init) => {
    const body = JSON.parse(init.body)
    requests.push({ url: String(url), init, body })
    if (String(url).endsWith('/compose')) {
      return Response.json({
        identityContext: 'profile-memory',
        retrievedEvidence: 'retrieved-memory',
      })
    }
    return Response.json({ ok: true, status: 'accumulating' })
  }
  t.after(() => { globalThis.fetch = originalFetch })

  const harness = fakeContext()
  apply(harness.ctx, {
    baseUrl: 'http://127.0.0.1:8090',
    accountId: 'acct-test',
    userId: 'user-test',
    agentId: 'agent-test',
    pendingDir,
    composeTimeoutMs: 1000,
    afterTurnTimeoutMs: 1000,
    writebackInitialDelayMs: 1,
    writebackMaxDelayMs: 2,
  })

  const session = { id: 'session-1', events: [] }
  const agent = { session }
  harness.agents.set(session.id, agent)
  const decision = {
    kind: 'enter',
    messages: [{
      source: { kind: 'user' },
      content: [{ type: 'text', text: 'what did we decide?' }],
    }],
  }
  const recalledDecision = await harness.listener('agent/pre-step')(
    { agent, step: 1, signal: new AbortController().signal },
    async () => decision,
  )
  const context = recalledDecision.messages.at(-1).content[0].text
  assert.match(context, /profile-memory/)
  assert.match(context, /retrieved-memory/)
  assert.equal(recalledDecision.messages.at(-1).source.kind, 'plugin')
  assert.equal(recalledDecision.messages.at(-1).source.plugin, 'ogmemory-lifecycle')
  assert.equal(recalledDecision.messages.at(-1).role, 'user')
  assert.match(recalledDecision.messages.at(-1).id, /^[0-9a-f-]{36}$/)
  assert.equal(Object.isFrozen(recalledDecision.messages.at(-1)), true)

  session.events.push(
    event('turn/start', { turn: 1 }, 0),
    event('user/message', decision.messages[0], 1),
    event('assistant/message', {
      turn: 1,
      step: 1,
      message: {
        source: { kind: 'model', provider: 'p', model: 'm' },
        content: [{ type: 'text', text: 'the decision' }],
      },
    }, 2),
    event('turn/end', { turn: 1, reason: { kind: 'completed' } }, 3),
  )
  harness.listener('session/event')(session, session.events.at(-1))
  await harness.listener('session/flush')(session)

  assert.equal(requests.length, 2)
  assert.equal(requests[0].body.sessionId, 'session-1')
  assert.equal(requests[0].init.headers['X-API-Key'], 'test-key')
  assert.deepEqual(
    requests[1].body.messages.map(message => [message.role, message.content]),
    [['user', 'what did we decide?'], ['assistant', 'the decision']],
  )
  assert.equal(requests[1].body.forceExtract, true)
  assert.match(requests[1].body.client_request_id, /^[0-9a-f-]{36}$/)
  assert.equal(requests[1].body.wait, true)
  await harness.listener('session/disposed')(session)
  assert.equal(requests.at(-1).url.endsWith('/dispose'), true)
  assert.equal(requests.at(-1).body.sessionId, 'session-1')
  await harness.dispose()
})

function fakeContext() {
  const listeners = new Map()
  const effects = []
  const agents = new Map()
  const tools = new Map()
  const ctx = {
    agents: { get: id => agents.get(id) },
    credentials: { resolve: async ref => ref === 'OGMEM_DSH_API_KEY' ? { value: 'test-key', source: 'test' } : undefined },
    sessions: {},
    tools: { register: tool => tools.set(tool.name, tool) },
    logger: { debug() {}, warn() {}, error() {} },
    on(name, listener) {
      listeners.set(name, listener)
      return () => listeners.delete(name)
    },
    effect(register) {
      effects.push(register())
    },
  }
  return {
    ctx,
    agents,
    tools,
    listener: name => listeners.get(name),
    async dispose() {
      await Promise.all(effects.map(dispose => dispose()))
    },
  }
}

function event(type, data, seq) {
  return { type, data, seq, time: 1700000000000 + seq }
}
