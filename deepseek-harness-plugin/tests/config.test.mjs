import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveConfig } from '../lib/config.js'

test('resolves a loopback default configuration', () => {
  const config = resolveConfig()
  assert.equal(config.baseUrl, 'http://127.0.0.1:8090')
  assert.equal(config.apiKeyEnv, 'OGMEM_DSH_API_KEY')
  assert.equal(config.composeEnabled, true)
  assert.equal(config.writebackEnabled, true)
  assert.equal(config.disposeEnabled, true)
  assert.equal(config.toolsEnabled, true)
  assert.equal(config.writeToolsEnabled, false)
  assert.equal(config.deleteToolsEnabled, false)
  assert.equal(config.disposeTimeoutMs, 3000)
  assert.equal(config.shutdownTimeoutMs, 5000)
  assert.equal(config.captureToolResults, false)
  assert.equal(config.forceExtract, true)
  assert.equal(Object.isFrozen(config), true)
})

test('requires explicit opt-in for WSL address over plaintext HTTP', () => {
  assert.throws(
    () => resolveConfig({ baseUrl: 'http://172.24.16.1:8090' }),
    /allowInsecureHttp=true/,
  )
  assert.equal(resolveConfig({
    baseUrl: 'http://172.24.16.1:8090/',
    allowInsecureHttp: true,
  }).baseUrl, 'http://172.24.16.1:8090')
})

test('rejects invalid queue and identity configuration', () => {
  assert.throws(() => resolveConfig({ accountId: '' }), /accountId/)
  assert.throws(() => resolveConfig({ apiKeyEnv: 'not-an-env-ref' }), /apiKeyEnv/)
  assert.throws(() => resolveConfig({ pendingDir: 1 }), /pendingDir/)
  assert.throws(() => resolveConfig({ toolsEnabled: 'true' }), /toolsEnabled/)
  assert.throws(() => resolveConfig({ shutdownTimeoutMs: 0 }), /shutdownTimeoutMs/)
  assert.throws(
    () => resolveConfig({ writebackInitialDelayMs: 1000, writebackMaxDelayMs: 500 }),
    /must not exceed/,
  )
})
