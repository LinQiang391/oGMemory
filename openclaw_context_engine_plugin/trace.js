import { randomUUID as nodeRandomUUID } from "node:crypto";
import { isIP } from "node:net";

const MAX_TRACE_CONTENT_CHARS = 10_000;
const SECRET_KEY = /(?:api[_-]?key|authorization|cookie|password|secret|token)/i;
const SECRET_NAME = /(?:[a-z0-9]+[_-])*(?:api[_-]?key|password|secret(?:[_-]access[_-]key)?|token)|authorization|cookie/;
const QUOTED_SECRET = new RegExp(`\\b(${SECRET_NAME.source})(\\s*[:=]\\s*)(["'])(.*?)\\3`, "gi");
const INLINE_SECRET = new RegExp(`\\b(${SECRET_NAME.source})(\\s*[:=]\\s*)(?:(?:bearer|basic)\\s+)?([^\\s,;"']+)`, "gi");
const QUOTED_CLI_SECRET = /(--(?:api[-_]?key|password|secret|token)\s+)(["'])(.*?)\2/gi;
const CLI_SECRET = /(--(?:api[-_]?key|password|secret|token)\s+)([^\s,;"']+)/gi;
const CONNECTION_STRING = /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis):\/\/[^\s/@:]+:[^\s/@]+@[^\s]+/gi;
const BEARER_TOKEN = /(\bbearer\s+)([^\s,;"']+)/gi;
const EMAIL = /(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])/g;
const PHONE = /(?<!\d)1[3-9]\d{9}(?!\d)/g;
const ID_CARD = /(?<![A-Za-z0-9])\d{17}[0-9Xx](?![A-Za-z0-9])/g;
const PASSPORT = /(?<![A-Za-z0-9])[EeGg]\d{8}(?![A-Za-z0-9])/g;
const BANK_CARD = /(?<!\d)\d{16,19}(?!\d)/g;
const IPV6 = /(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{0,4})(?![\w:.])/g;
const IPV4 = /(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])/g;

function secretKind(name) {
  const normalized = String(name).toLowerCase().replaceAll("-", "_");
  if (normalized.includes("api_key") || normalized.includes("apikey")) return "api_key";
  if (normalized.includes("password")) return "password";
  if (normalized.includes("authorization")) return "authorization";
  if (normalized.includes("cookie")) return "cookie";
  if (normalized.includes("token")) return "token";
  return "secret";
}

class RedactionState {
  constructor() {
    this.placeholders = new Map();
    this.counts = new Map();
  }

  placeholder(kind, value) {
    const plaintext = String(value);
    if (!this.placeholders.has(plaintext)) {
      const count = (this.counts.get(kind) || 0) + 1;
      this.counts.set(kind, count);
      this.placeholders.set(plaintext, `{{${kind}_${count}}}`);
    }
    return this.placeholders.get(plaintext);
  }

  redactString(text) {
    let value = text.replace(CONNECTION_STRING, (match) => this.placeholder("connection_string", match));
    value = value.replace(QUOTED_SECRET, (_match, name, separator, quote, plaintext) => (
      `${name}${separator}${quote}${this.placeholder(secretKind(name), plaintext)}${quote}`
    ));
    value = value.replace(INLINE_SECRET, (match, name, separator, plaintext) => (
      `${name}${separator}${this.placeholder(/\bbearer\s/i.test(match) ? "token" : secretKind(name), plaintext)}`
    ));
    value = value.replace(BEARER_TOKEN, (_match, prefix, token) => (
      `${prefix}${this.placeholder("token", token)}`
    ));
    value = value.replace(QUOTED_CLI_SECRET, (_match, prefix, quote, plaintext) => (
      `${prefix}${quote}${this.placeholder(secretKind(prefix), plaintext)}${quote}`
    ));
    value = value.replace(CLI_SECRET, (_match, prefix, plaintext) => (
      `${prefix}${this.placeholder(secretKind(prefix), plaintext)}`
    ));
    value = value.replace(EMAIL, (match) => this.placeholder("email", match));
    value = value.replace(PHONE, (match) => this.placeholder("phone", match));
    value = value.replace(ID_CARD, (match) => this.placeholder("id_card", match));
    value = value.replace(PASSPORT, (match) => this.placeholder("passport", match));
    value = value.replace(BANK_CARD, (match) => this.placeholder("bank_card", match));
    value = value.replace(IPV6, (match) => (
      isIP(match) === 6 ? this.placeholder("ipv6", match) : match
    ));
    return value.replace(IPV4, (match) => (
      isIP(match) === 4 ? this.placeholder("ipv4", match) : match
    ));
  }

  redact(value) {
    if (Array.isArray(value)) return value.map((item) => this.redact(item));
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [
        key,
        SECRET_KEY.test(key)
          ? this.placeholder(secretKind(key), item)
          : this.redact(item),
      ]));
    }
    return typeof value === "string" ? this.redactString(value) : value;
  }
}

export function redactTraceValue(value) {
  return new RedactionState().redact(value);
}

function contentSize(value) {
  if (typeof value === "string") return value.length;
  try {
    return JSON.stringify(value).length;
  } catch {
    return String(value).length;
  }
}

function bounded(value) {
  const serialized = typeof value === "string" ? value : JSON.stringify(value);
  if (serialized.length <= MAX_TRACE_CONTENT_CHARS) return value;
  const suffix = `\n[${serialized.length - MAX_TRACE_CONTENT_CHARS} more chars omitted]`;
  return `${serialized.slice(0, MAX_TRACE_CONTENT_CHARS - suffix.length)}${suffix}`;
}

function textContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (typeof part === "string" ? part : part?.text || ""))
    .filter(Boolean)
    .join("\n");
}

function lastAssistantText(messages) {
  if (!Array.isArray(messages)) return "";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role !== "assistant") continue;
    const text = textContent(messages[index].content);
    if (text.trim()) return text;
  }
  return "";
}

function addRunId(result, event, ctx) {
  const runId = event?.runId || ctx?.runId;
  if (runId) result.run_id = String(runId);
}

export function openClawHookToEvent(hookName, event = {}, ctx = {}) {
  const state = new RedactionState();
  const result = {};

  switch (hookName) {
    case "session_start":
      result.event_type = "session_start";
      if (event.resumedFrom) result.resumed_from = String(event.resumedFrom);
      break;
    case "before_agent_run":
      result.event_type = "user_message";
      addRunId(result, event, ctx);
      result.prompt_size = contentSize(event.prompt || "");
      result.prompt = bounded(state.redact(String(event.prompt || "")));
      break;
    case "before_tool_call":
      result.event_type = "tool_start";
      addRunId(result, event, ctx);
      result.tool_name = String(event.toolName || "unknown_tool");
      if (event.toolCallId || ctx.toolCallId) result.tool_use_id = String(event.toolCallId || ctx.toolCallId);
      result.tool_input_size = contentSize(event.params || {});
      result.tool_input = bounded(state.redact(event.params || {}));
      break;
    case "after_tool_call": {
      const failed = event.error !== undefined && event.error !== null && event.error !== "";
      result.event_type = failed ? "tool_failure" : "tool_success";
      addRunId(result, event, ctx);
      result.tool_name = String(event.toolName || "unknown_tool");
      if (event.toolCallId || ctx.toolCallId) result.tool_use_id = String(event.toolCallId || ctx.toolCallId);
      result.tool_input_size = contentSize(event.params || {});
      result.tool_input = bounded(state.redact(event.params || {}));
      if (failed) {
        result.error_size = contentSize(event.error);
        result.error = bounded(state.redact(String(event.error)));
      } else if (event.result !== undefined) {
        result.tool_response_size = contentSize(event.result);
        result.tool_response = bounded(state.redact(event.result));
      }
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      break;
    }
    case "agent_end": {
      result.event_type = "turn_end";
      addRunId(result, event, ctx);
      const assistantMessage = lastAssistantText(event.messages);
      result.assistant_message_size = contentSize(assistantMessage);
      result.assistant_message = bounded(state.redact(assistantMessage));
      result.status = event.success ? "ok" : "error";
      if (event.error) {
        result.error_size = contentSize(event.error);
        result.error = bounded(state.redact(String(event.error)));
      }
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      break;
    }
    case "before_compaction":
      result.event_type = "pre_compact";
      addRunId(result, event, ctx);
      result.message_count = event.messageCount;
      if (Number.isFinite(event.compactingCount)) result.compacting_count = event.compactingCount;
      if (Number.isFinite(event.tokenCount)) result.token_count = event.tokenCount;
      break;
    case "session_end":
      result.event_type = "session_end";
      result.message_count = event.messageCount;
      if (Number.isFinite(event.durationMs)) result.duration_ms = event.durationMs;
      if (event.reason) result.reason = bounded(state.redact(String(event.reason)));
      break;
    default:
      return null;
  }
  return result;
}

function traceHeaders(cfg, envelope) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.authApiKey) headers["X-API-Key"] = cfg.authApiKey;
  if (cfg.authAccountId) headers["X-Account-ID"] = cfg.authAccountId;
  if (envelope.user_id) headers["X-User-ID"] = envelope.user_id;
  return headers;
}

export async function postTraceEvent(cfg, envelope, deps = {}) {
  const fetchFn = deps.fetchFn || fetch;
  const timeoutSignalFn = deps.timeoutSignalFn || ((timeoutMs) => AbortSignal.timeout(timeoutMs));
  const logger = deps.logger || console;
  const url = `${String(cfg.memoryApiBaseUrl || "").replace(/\/$/, "")}/api/v1/trace/events`;
  const body = JSON.stringify(envelope);
  let lastError;

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetchFn(url, {
        method: "POST",
        headers: traceHeaders(cfg, envelope),
        body,
        signal: timeoutSignalFn(cfg.traceTimeoutMs),
      });
      if (response.ok) return true;
      lastError = new Error(`HTTP ${response.status ?? "error"}`);
    } catch (error) {
      lastError = error;
    }
  }

  logger.error?.(`[og-memory] trace upload failed after 2 attempts: ${lastError?.message || lastError}`);
  return false;
}

const TRACE_HOOK_NAMES = [
  "session_start",
  // Non-bundled OpenClaw plugins need
  // plugins.entries.og-memory-context-engine.hooks.allowConversationAccess=true
  // for conversation-bearing before_agent_run and agent_end payloads.
  "before_agent_run",
  "before_tool_call",
  "after_tool_call",
  "agent_end",
  "before_compaction",
  "session_end",
];

export function registerOpenClawTraceHooks(api, cfg, deps = {}) {
  const randomUUID = deps.randomUUID || nodeRandomUUID;
  const now = deps.now || (() => new Date());
  const sessionIdentities = new Map();

  for (const hookName of TRACE_HOOK_NAMES) {
    api.on(hookName, async (event, ctx) => {
      const canonicalEvent = openClawHookToEvent(hookName, event, ctx);
      if (!canonicalEvent) return;
      const sessionId = event?.sessionId || ctx?.sessionId || event?.sessionKey || ctx?.sessionKey;
      if (!sessionId) {
        api.logger?.error?.(`[og-memory] trace ${canonicalEvent.event_type} skipped: missing session id`);
        return;
      }
      const cachedIdentity = sessionIdentities.get(sessionId) || {};
      const runtimeIdentity = {
        accountId: event?.accountId || ctx?.accountId,
        userId: event?.senderId || ctx?.senderId,
        agentId: event?.agentId || ctx?.agentId,
      };
      const nextIdentity = {
        accountId: runtimeIdentity.accountId || cachedIdentity.accountId,
        userId: runtimeIdentity.userId || cachedIdentity.userId,
        agentId: runtimeIdentity.agentId || cachedIdentity.agentId,
      };
      if (Object.values(nextIdentity).some(Boolean)) {
        sessionIdentities.set(sessionId, nextIdentity);
      }
      const eventId = randomUUID();
      const envelope = {
        request_id: eventId,
        timestamp: now().toISOString(),
        account_id: nextIdentity.accountId || cfg.accountId,
        user_id: nextIdentity.userId || cfg.userId,
        agent_id: nextIdentity.agentId || cfg.agentId,
        session_id: String(sessionId),
        event: canonicalEvent,
      };
      try {
        await postTraceEvent(cfg, envelope, {
          ...deps,
          logger: deps.logger || api.logger,
        });
      } finally {
        if (hookName === "session_end") sessionIdentities.delete(sessionId);
      }
    });
  }
}
