#!/usr/bin/env node
'use strict';

// Keep this dependency-free: the OpenClaw base image already provides Node.
const fs = require('node:fs');
const path = require('node:path');
const { randomBytes } = require('node:crypto');

const [templatePath, configPath] = process.argv.slice(2);

function object(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function portFromEnv() {
  const raw = process.env.GATEWAY_PORT || '18789';
  if (!/^\d+$/.test(raw)) throw new Error('GATEWAY_PORT must be an integer from 1 to 65535');
  const port = Number(raw);
  if (port < 1 || port > 65535) throw new Error('GATEWAY_PORT must be an integer from 1 to 65535');
  return port;
}

function parseConfig(text) {
  // Repair only the known legacy template syntax, not arbitrary malformed JSON.
  const normalized = text.replace(
    /("port"\s*:\s*)__GATEWAY_PORT__(?=\s*[,}])/g,
    '$1"__GATEWAY_PORT__"',
  );
  try {
    return JSON.parse(normalized);
  } catch {
    // Do not print parser errors: they can contain snippets of API keys.
    throw new Error('Config is not valid JSON; refusing to overwrite it. Restore or repair it first');
  }
}

function replacement(token) {
  const env = process.env;
  switch (token) {
    case '__LLM_PROVIDER__': return env.LLM_PROVIDER || 'openai';
    case '__LLM_MODEL__': return env.LLM_MODEL || 'gpt-4o-mini';
    case '__LLM_BASE_URL__': return env.LLM_BASE_URL || 'https://api.openai.com/v1';
    case '__LLM_API_KEY__': return env.LLM_API_KEY || '';
    case '__GATEWAY_PORT__': return portFromEnv();
    case '__OGMEM_URL__': return env.OGMEM_URL || env.OGMEM_API_URL || 'http://127.0.0.1:8090';
    case '__OG_AUTH_API_KEY__': return env.OG_AUTH_API_KEY || '';
    case '__OG_AUTH_ACCOUNT_ID__': return env.OG_AUTH_ACCOUNT_ID || '';
    default: return token;
  }
}

let changed = false;
function render(value) {
  if (typeof value === 'string') {
    if (value === '__GATEWAY_PORT__') {
      changed = true;
      return portFromEnv();
    }
    return value.replace(
      /__(?:LLM_PROVIDER|LLM_MODEL|LLM_BASE_URL|LLM_API_KEY|GATEWAY_PORT|OGMEM_URL|OG_AUTH_API_KEY|OG_AUTH_ACCOUNT_ID)__/g,
      (token) => {
        changed = true;
        return String(replacement(token));
      },
    );
  }
  if (Array.isArray(value)) return value.map(render);
  if (object(value)) {
    // Substitution applies to provider keys as well as values, but never
    // re-scans injected environment values or interpolates raw JSON text.
    const entries = Object.entries(value).map(([key, item]) => [render(key), render(item)]);
    if (new Set(entries.map(([key]) => key)).size !== entries.length) {
      throw new Error('Placeholder substitution would overwrite an existing config key');
    }
    return Object.fromEntries(entries);
  }
  return value;
}

function prepare(config) {
  if (!object(config)) throw new Error('Config must be a JSON object');
  // A fresh state directory has only the implicit "main" agent. Requests
  // naming another agent are rejected before any context-engine hook runs.
  // Register only operator-provided IDs; never infer agents from HTTP input.
  const requestedAgents = (process.env.OPENCLAW_AGENT_IDS || '')
    .split(',').map((id) => id.trim()).filter(Boolean);
  if (requestedAgents.some((id) => !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(id))) {
    throw new Error('OPENCLAW_AGENT_IDS must contain comma-separated lowercase agent IDs (1-64 letters, digits, underscores or hyphens)');
  }
  if (requestedAgents.length) {
    if (config.agents === undefined) config.agents = {};
    if (!object(config.agents)) throw new Error('agents must be a JSON object');
    if (config.agents.list === undefined) {
      config.agents.list = [{ id: 'main', default: true }];
      changed = true;
    }
    if (!Array.isArray(config.agents.list)) throw new Error('agents.list must be an array');
    for (const id of new Set(requestedAgents)) {
      if (!config.agents.list.some((agent) => agent?.id === id)) {
        config.agents.list.push({ id });
        changed = true;
      }
    }
  }
  if (config.gateway === undefined) {
    config.gateway = {};
    changed = true;
  }
  if (!object(config.gateway)) throw new Error('gateway must be a JSON object');
  if (config.gateway.mode === undefined) {
    config.gateway.mode = 'local';
    changed = true;
  }
  if (config.gateway.mode !== 'local') {
    throw new Error('gateway.mode must be "local" for this deployment; refusing to replace an explicit mode');
  }
  if (config.gateway.port === undefined) {
    config.gateway.port = portFromEnv();
    changed = true;
  }
  if (!Number.isInteger(config.gateway.port) || config.gateway.port < 1 || config.gateway.port > 65535) {
    throw new Error('gateway.port must be an integer from 1 to 65535');
  }

  // Older deployments lack these plugin fields. Add only missing fields;
  // never reset the rest of a user's config or replace their credentials.
  const plugin = config.plugins?.entries?.['og-memory-context-engine']?.config;
  if (object(plugin)) {
    // Container-level phase controls must also work with a persisted config.
    // Outside this Docker renderer the plugin retains config-over-env priority.
    for (const [key, envName] of [
      ['autoCapture', 'OGMEM_AUTO_CAPTURE'],
      ['autoRecall', 'OGMEM_AUTO_RECALL'],
    ]) {
      const raw = process.env[envName]?.trim().toLowerCase();
      if (!raw) continue;
      if (!['true', 'false', '1', '0', 'yes', 'no', 'on', 'off'].includes(raw)) {
        throw new Error(`${envName} must be a boolean (true/false, 1/0, yes/no, on/off)`);
      }
      const value = ['true', '1', 'yes', 'on'].includes(raw);
      if (plugin[key] !== value) {
        plugin[key] = value;
        changed = true;
      }
    }
    for (const [key, envName] of [
      ['authApiKey', 'OG_AUTH_API_KEY'],
      ['authAccountId', 'OG_AUTH_ACCOUNT_ID'],
    ]) {
      if (plugin[key] === undefined) {
        plugin[key] = process.env[envName] || '';
        changed = true;
      }
    }
  }
  return config;
}

try {
  if (!templatePath || !configPath) throw new Error('Usage: configure-openclaw.cjs TEMPLATE CONFIG');
  let original;
  try {
    if (!fs.lstatSync(configPath).isFile()) throw new Error('Config must be a regular file');
    original = fs.readFileSync(configPath, 'utf8');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const config = prepare(render(parseConfig(original ?? fs.readFileSync(templatePath, 'utf8'))));
  if (original !== undefined && !changed) {
    console.log('Existing OpenClaw config is valid; preserving it');
  } else {
    // Fully parse and validate before writing anything. Keep both the backup
    // and the temporary file private because they contain credentials.
    const serialized = `${JSON.stringify(config, null, 2)}\n`;
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    const suffix = `${Date.now()}-${process.pid}-${randomBytes(6).toString('hex')}`;
    const temporary = `${configPath}.tmp.${suffix}`;
    let created = false;
    try {
      fs.writeFileSync(temporary, serialized, { mode: 0o600, flag: 'wx' });
      created = true;
      if (original !== undefined) {
        const backup = `${configPath}.bak.${suffix}`;
        fs.writeFileSync(backup, original, { mode: 0o600, flag: 'wx' });
        console.log(`Backed up existing config to ${backup}`);
      }
      fs.renameSync(temporary, configPath);
    } finally {
      if (created && fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
    console.log(`Prepared validated OpenClaw config at ${configPath}`);
  }
} catch (error) {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
}
