'use strict';

// Standalone Node tests: no Python/AGFS fixtures, Docker daemon, or credentials.
// Run from the repository root:
//   node --test tests/unit/docker/test_openclaw_config.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '../../..');
const helper = path.join(root, 'docker/configure-openclaw.cjs');
const template = fs.readFileSync(path.join(root, 'config/openclaw/ogmem.template.json'), 'utf8');
const dummyEnv = {
  LLM_PROVIDER: 'test-provider',
  LLM_MODEL: 'test-model',
  LLM_API_KEY: 'test-key',
  LLM_BASE_URL: 'https://example.invalid/v1',
  GATEWAY_PORT: '48542',
  OG_AUTH_API_KEY: 'memory-key',
  OG_AUTH_ACCOUNT_ID: 'test-account',
};

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const templatePath = path.join(dir, 'template.json');
  const configPath = path.join(dir, 'home', 'openclaw.json');
  fs.writeFileSync(templatePath, template);
  const write = (value) => {
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(configPath, typeof value === 'string' ? value : JSON.stringify(value));
  };
  return {
    configPath, templatePath, write,
    read: () => JSON.parse(fs.readFileSync(configPath, 'utf8')),
    raw: () => fs.readFileSync(configPath, 'utf8'),
    backups: () => fs.readdirSync(path.dirname(configPath)).filter((name) => name.includes('.bak.')),
    run: (env = {}) => {
      const result = spawnSync(process.execPath, [helper, templatePath, configPath], {
        // Never inherit real DB/model credentials from the developer's environment.
        env: { ...dummyEnv, ...env }, encoding: 'utf8',
      });
      assert.ifError(result.error);
      assert.equal(result.signal, null);
      return result;
    },
  };
}

function succeeds(result) {
  assert.equal(result.status, 0, result.stderr);
}

test('template is valid JSON and fresh rendering produces a numeric gateway port', (t) => {
  assert.equal(JSON.parse(template).gateway.port, '__GATEWAY_PORT__');
  const f = fixture(t);
  succeeds(f.run());
  const cfg = f.read();
  assert.equal(cfg.gateway.mode, 'local');
  assert.equal(cfg.gateway.port, 48542);
  assert.equal(cfg.agents.defaults.model.primary, 'test-provider/test-model');
  assert.equal(cfg.models.providers['test-provider'].apiKey, 'test-key');
  assert.equal(cfg.plugins.entries['og-memory-context-engine'].config.authApiKey, 'memory-key');
  assert.doesNotMatch(f.raw(), /__[A-Z_]+__/);
  assert.equal(fs.statSync(f.configPath).mode & 0o777, 0o600);
});

test('JSON-aware rendering safely escapes strings and never re-renders injected tokens', (t) => {
  const f = fixture(t);
  const value = 'quote" backslash\\ ampersand& slash/ newline\n__LLM_MODEL__';
  succeeds(f.run({
    LLM_PROVIDER: 'provider/with"quotes',
    LLM_API_KEY: value,
    OG_AUTH_API_KEY: value,
    OG_AUTH_ACCOUNT_ID: value,
    OGMEM_URL: 'https://example.invalid/path?q="a"&b=\\value',
  }));
  const cfg = f.read();
  assert.equal(cfg.models.providers['provider/with"quotes'].apiKey, value);
  const plugin = cfg.plugins.entries['og-memory-context-engine'].config;
  assert.equal(plugin.authApiKey, value);
  assert.equal(plugin.authAccountId, value);
  assert.equal(plugin.memoryApiBaseUrl, 'https://example.invalid/path?q="a"&b=\\value');
});

test('rendering preserves URL fallback and default port behavior', (t) => {
  const f = fixture(t);
  succeeds(f.run({ GATEWAY_PORT: '', OGMEM_API_URL: 'http://example.invalid:8090' }));
  assert.equal(f.read().gateway.port, 18789);
  assert.equal(f.read().plugins.entries['og-memory-context-engine'].config.memoryApiBaseUrl,
    'http://example.invalid:8090');
});

test('valid existing user config is preserved byte-for-byte despite changed environment', (t) => {
  const f = fixture(t);
  const original = '{ "gateway": { "mode": "local", "port": 19000 }, "custom": true }\n';
  f.write(original);
  succeeds(f.run({ GATEWAY_PORT: '12345' }));
  assert.equal(f.raw(), original);
  assert.deepEqual(f.backups(), []);
});

test('missing gateway mode is repaired without losing custom settings', (t) => {
  const f = fixture(t);
  const cfg = {
    gateway: { port: 19000, bind: 'loopback', auth: { token: 'keep-token' } },
    channels: { custom: { enabled: true } },
    agents: { defaults: { model: { primary: 'custom/model' } } },
  };
  f.write(cfg);
  const original = f.raw();
  succeeds(f.run());
  assert.deepEqual(f.read(), { ...cfg, gateway: { ...cfg.gateway, mode: 'local' } });
  const [backup] = f.backups();
  const backupPath = path.join(path.dirname(f.configPath), backup);
  assert.equal(fs.readFileSync(backupPath, 'utf8'), original);
  assert.equal(fs.statSync(backupPath).mode & 0o777, 0o600);
  const repaired = f.raw();
  succeeds(f.run());
  assert.equal(f.raw(), repaired);
  assert.equal(f.backups().length, 1);
});

test('missing gateway object is added without replacing the remaining config', (t) => {
  const f = fixture(t);
  f.write({ channels: { custom: true } });
  succeeds(f.run());
  assert.deepEqual(f.read(), {
    channels: { custom: true }, gateway: { mode: 'local', port: 48542 },
  });
});

for (const legacy of [false, true]) {
  test(`existing ${legacy ? 'unquoted legacy' : 'quoted'} port placeholder is repaired in place`, (t) => {
    const f = fixture(t);
    const cfg = JSON.parse(template);
    cfg.channels = { preserve: 'custom-channel' };
    const text = JSON.stringify(cfg);
    f.write(legacy ? text.replace('"__GATEWAY_PORT__"', '__GATEWAY_PORT__') : text);
    const original = f.raw();
    succeeds(f.run());
    assert.equal(f.read().gateway.port, 48542);
    assert.deepEqual(f.read().channels, cfg.channels);
    const backup = path.join(path.dirname(f.configPath), f.backups()[0]);
    assert.equal(fs.readFileSync(backup, 'utf8'), original);
  });
}

test('missing plugin auth fields are added without replacing existing credentials or config', (t) => {
  const f = fixture(t);
  f.write({
    gateway: { mode: 'local', port: 19000 },
    plugins: { entries: { 'og-memory-context-engine': {
      config: { authApiKey: 'keep-existing', memoryApiBaseUrl: 'http://custom.invalid' },
    } } },
  });
  succeeds(f.run());
  assert.deepEqual(f.read().plugins.entries['og-memory-context-engine'].config, {
    authApiKey: 'keep-existing', authAccountId: 'test-account',
    memoryApiBaseUrl: 'http://custom.invalid',
  });
});

for (const port of ['0', '65536', 'abc', '1.5', '12junk']) {
  test(`invalid environment port ${port} cannot overwrite a pending config`, (t) => {
    const f = fixture(t);
    f.write(template);
    const result = f.run({ GATEWAY_PORT: port });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /GATEWAY_PORT/);
    assert.equal(f.raw(), template);
    assert.deepEqual(f.backups(), []);
    assert.deepEqual(fs.readdirSync(path.dirname(f.configPath)), ['openclaw.json']);
  });
}

for (const config of [
  '{ "apiKey": "do-not-leak-secret", ',
  '',
  '[]',
  '{"gateway":null}',
  '{"gateway":{"mode":"remote","port":18789}}',
  '{"gateway":{"mode":"local","port":"18789"}}',
  '{"gateway":{"mode":"local","port":70000}}',
]) {
  test(`invalid or incompatible existing config is preserved (${JSON.stringify(config)})`, (t) => {
    const f = fixture(t);
    f.write(config);
    const result = f.run();
    assert.notEqual(result.status, 0);
    assert.doesNotMatch(result.stderr, /do-not-leak-secret/);
    assert.equal(f.raw(), config);
    assert.deepEqual(f.backups(), []);
  });
}

test('malformed template never installs an active config', (t) => {
  const f = fixture(t);
  fs.writeFileSync(f.templatePath, '{"bad":');
  assert.notEqual(f.run().status, 0);
  assert.equal(fs.existsSync(f.configPath), false);
});

test('entrypoint uses the validated renderer and keeps credentials private', () => {
  const entrypoint = fs.readFileSync(path.join(root, 'docker/entrypoint-openclaw.sh'), 'utf8');
  const dockerfile = fs.readFileSync(path.join(root, 'docker/Dockerfile.openclaw'), 'utf8');
  assert.match(entrypoint, /node \/usr\/local\/bin\/configure-openclaw\.cjs "\$TEMPLATE" "\$CONFIG"/);
  assert.match(entrypoint, /chmod 600 "\$CONFIG"/);
  assert.doesNotMatch(entrypoint, /chmod a\+rw "\$CONFIG"/);
  assert.match(dockerfile, /COPY docker\/configure-openclaw.cjs \/usr\/local\/bin\/configure-openclaw.cjs/);
  assert.doesNotMatch(dockerfile, /cp .*openclaw\.template\.json .*openclaw\.json/);
});
