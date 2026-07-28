import { createHash } from "node:crypto";
import { redact, redactString, safeErrorMessage } from "./redaction.mjs";

const MAX_TRACE_BYTES = 56 * 1024;
const boundedField = (value, maxLength = 512) => redactString(value ?? "", maxLength);
const boundedIdentifier = (value, maxLength = 512) => String(value ?? "").slice(0, maxLength);

export function stableEventId(type, sessionId, objectId = "", terminalState = "") {
  const digest = createHash("sha256").update([type, sessionId, objectId, terminalState].join("\0")).digest("hex").slice(0, 48);
  return `oc_${digest}`;
}

function sizeOf(value) {
  try { return Buffer.byteLength(typeof value === "string" ? value : JSON.stringify(value), "utf8"); }
  catch { return 0; }
}

function canonicalEvent(type, options, captureContent) {
  const { content, metadata = {}, objectId = "", terminalState = "" } = options;
  const clean = redact(content);
  const event = { event_type: type };
  if (type === "user_message") {
    event.prompt_size = sizeOf(clean);
    if (captureContent) event.prompt = clean || "[EMPTY]";
  } else if (type === "assistant_message") {
    event.assistant_message_size = sizeOf(clean);
    if (captureContent) event.assistant_message = clean || "[EMPTY]";
  } else if (["tool_start", "tool_success", "tool_failure"].includes(type)) {
    event.tool_name = boundedField(metadata.tool || "unknown_tool", 256);
    if (objectId) event.tool_use_id = boundedField(objectId);
    if (type === "tool_start") {
      event.tool_input_size = sizeOf(clean);
      if (captureContent) event.tool_input = clean;
    } else if (type === "tool_success") {
      event.tool_response_size = sizeOf(clean);
      if (captureContent) event.tool_response = clean;
    } else {
      event.error_size = sizeOf(clean);
      if (captureContent) event.error = clean || "[UNKNOWN_ERROR]";
    }
  } else if (type === "turn_end") {
    event.status = terminalState === "error" ? "error" : "ok";
    event.assistant_message_size = sizeOf(clean);
    if (captureContent) event.assistant_message = clean || "[NO_ASSISTANT_TEXT]";
  } else if (type === "session_error") {
    event.error_size = sizeOf(clean);
    if (captureContent) event.error = clean || "[UNKNOWN_ERROR]";
  } else {
    const cleanMetadata = redact(metadata);
    if (metadata && Object.hasOwn(metadata, "parent_session_id")) cleanMetadata.parent_session_id = boundedIdentifier(metadata.parent_session_id);
    for (const [key, value] of Object.entries(cleanMetadata || {})) if (key !== "event_type") event[key] = value;
  }
  return event;
}

function truncateCanonical(event, captureContent) {
  const minimal = { event_type: event.event_type, truncated: true };
  for (const key of ["tool_name", "tool_use_id", "status", "prompt_size", "assistant_message_size", "tool_input_size", "tool_response_size", "error_size"]) {
    if (event[key] !== undefined) minimal[key] = event[key];
  }
  if (captureContent) {
    if (event.event_type === "user_message") minimal.prompt = "[TRUNCATED_EVENT]";
    if (["assistant_message", "turn_end"].includes(event.event_type)) minimal.assistant_message = "[TRUNCATED_EVENT]";
    if (event.event_type === "tool_start") minimal.tool_input = "[TRUNCATED_EVENT]";
    if (event.event_type === "tool_success") minimal.tool_response = "[TRUNCATED_EVENT]";
    if (["tool_failure", "session_error"].includes(event.event_type)) minimal.error = "[TRUNCATED_EVENT]";
  }
  return minimal;
}

export function createTraceAdapter({ config, spool, logger = console, now = () => new Date() }) {
  const envelopeFor = (type, sessionId, options = {}) => {
    let event = canonicalEvent(type, options, config.traceCaptureContent);
    const envelope = {
      request_id: stableEventId(type, sessionId, options.objectId, options.terminalState),
      session_id: boundedIdentifier(sessionId, 128),
      timestamp: now().toISOString(),
      account_id: boundedIdentifier(config.accountId),
      user_id: boundedIdentifier(config.userId),
      agent_id: boundedIdentifier(config.agentId),
      event,
    };
    if (Buffer.byteLength(JSON.stringify(envelope), "utf8") > MAX_TRACE_BYTES) {
      event = truncateCanonical(event, config.traceCaptureContent);
      envelope.event = event;
    }
    if (Buffer.byteLength(JSON.stringify(envelope), "utf8") > MAX_TRACE_BYTES) throw new RangeError("oGMemory trace envelope exceeded the canonical size limit");
    return envelope;
  };
  return {
    async emit(type, sessionId, options = {}) {
      if (!sessionId) return false;
      try {
        await spool.enqueue(envelopeFor(type, sessionId, options));
        return true;
      } catch (error) {
        logger.warn?.(`Unable to spool oGMemory trace event: ${safeErrorMessage(error)}`);
        return false;
      }
    },
    async emitMany(items) {
      const valid = (Array.isArray(items) ? items : []).filter((item) => item?.sessionId);
      if (!valid.length) return true;
      try {
        const envelopes = valid.map((item) => envelopeFor(item.type, item.sessionId, item.options));
        if (spool.enqueueMany) await spool.enqueueMany(envelopes);
        else await Promise.all(envelopes.map((envelope) => spool.enqueue(envelope)));
        return true;
      } catch (error) {
        logger.warn?.(`Unable to spool oGMemory trace event batch: ${safeErrorMessage(error)}`);
        return false;
      }
    },
    flush() { return spool.flush(); },
    close() { spool.close?.(); },
  };
}
