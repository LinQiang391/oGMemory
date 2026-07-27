const MAX_NODES = 2000;
const MAX_DEPTH = 16;
const MAX_STRING = 10000;
const REDACTION_OVERLAP = 4096;
const SECRET_KEY = /(?:api[_-]?key|authorization|cookie|password|secret|token)/i;
const RULES = [
  [/-----BEGIN [^-\r\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\r\n]*PRIVATE KEY-----/g, "[REDACTED_PRIVATE_KEY]"],
  [/\b(?:password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*(?:["'][^"']*["']|[^\s,;]+)/gi, "[REDACTED_SECRET]"],
  [/(--(?:api[-_]?key|password|secret|token)\s+)(?:["'][^"']*["']|[^\s,;]+)/gi, "$1[REDACTED]"],
  [/\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi, "Bearer [REDACTED]"],
  [/\b(cookie|set-cookie|authorization)\s*[:=]\s*[^\r\n]+/gi, "$1=[REDACTED]"],
  [/\bAKIA[0-9A-Z]{16}\b/g, "[REDACTED_AWS_KEY]"],
  [/\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g, "[REDACTED_JWT]"],
  [/\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|glpat-|xox[baprs]-)[A-Za-z0-9_-]{12,}\b/g, "[REDACTED_TOKEN]"],
  [/\b[a-z][a-z0-9+.-]*:\/\/[^\s/@:]+:[^\s/@]+@[^\s]+/gi, "[REDACTED_CONNECTION_STRING]"],
  [/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[REDACTED_EMAIL]"],
  [/\b(?:\+?\d[\d ()-]{7,}\d)\b/g, "[REDACTED_PHONE]"],
  [/\b(?:\d[ -]*?){13,19}\b/g, "[REDACTED_NUMBER]"],
  [/(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{0,4})(?![\w:.])/g, (value) => isIP(value) === 6 ? "[REDACTED_IP]" : value],
  [/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[REDACTED_IP]"],
];

export function redactString(value, maxLength = MAX_STRING) {
  let output;
  try {
    const input = String(value);
    output = (input.length > maxLength ? input.slice(0, maxLength + REDACTION_OVERLAP) : input)
      .replace(/-----BEGIN [^-\r\n]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-\r\n]*PRIVATE KEY-----|$)/g, "[REDACTED_PRIVATE_KEY]");
  }
  catch { return "[UNSERIALIZABLE]"; }
  for (const [pattern, replacement] of RULES) output = output.replace(pattern, replacement);
  output = output
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]*$/i, "[REDACTED_TRUNCATED]")
    .replace(/\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|glpat-|xox[baprs]-)[A-Za-z0-9_-]*$/i, "[REDACTED_TRUNCATED]");
  return output.slice(0, maxLength);
}

export function safeErrorMessage(error) {
  return redactString(error?.message || "unknown error", 1000);
}

export function redact(value, options = {}) {
  const state = options.state || { nodes: 0, seen: new WeakSet() };
  const depth = options.depth || 0;
  state.nodes += 1;
  if (state.nodes > (options.maxNodes || MAX_NODES)) return "[NODE_LIMIT]";
  if (depth > (options.maxDepth || MAX_DEPTH)) return "[DEPTH_LIMIT]";
  if (typeof value === "string") return redactString(value);
  if (typeof value === "bigint") return value.toString();
  if (["function", "symbol", "undefined"].includes(typeof value)) return undefined;
  if (Array.isArray(value)) {
    if (state.seen.has(value)) return "[REPEATED_REFERENCE]";
    state.seen.add(value);
    const result = [];
    for (const item of value) {
      if (state.nodes >= (options.maxNodes || MAX_NODES)) { result.push("[NODE_LIMIT]"); break; }
      result.push(redact(item, { ...options, state, depth: depth + 1 }));
    }
    return result;
  }
  if (value && typeof value === "object") {
    if (state.seen.has(value)) return "[REPEATED_REFERENCE]";
    state.seen.add(value);
    const result = {};
    for (const key of Object.keys(value)) {
      if (state.nodes >= (options.maxNodes || MAX_NODES)) { result.__truncated__ = "[NODE_LIMIT]"; break; }
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor) continue;
      if (!("value" in descriptor)) { result[key] = "[GETTER]"; continue; }
      const item = descriptor.value;
      const redacted = SECRET_KEY.test(key) ? "[REDACTED]" : redact(item, { ...options, state, depth: depth + 1 });
      if (redacted !== undefined) result[key] = redacted;
    }
    return result;
  }
  return value;
}
import { isIP } from "node:net";
