import { join } from "node:path";
import { detectCompatibility, DISPOSE_OPENCODE_VERSION, MINIMUM_OPENCODE_VERSION } from "./compatibility.mjs";
import { createComposeController } from "./compose.mjs";
import { loadConfig } from "./config.mjs";
import { createHttpClient } from "./http-client.mjs";
import { createMessageSync } from "./message-sync.mjs";
import { createSessionStore, mergeTurnBoundaries } from "./session-state.mjs";
import { createTraceAdapter } from "./trace-adapter.mjs";
import { createTraceSpool } from "./trace-spool.mjs";

const SESSION_EVENTS = new Set(["session.created", "session.error", "message.updated", "message.part.updated", "session.idle", "session.deleted"]);

function sessionIdOf(event) {
  const properties = event?.properties || {};
  return properties.sessionID || properties.sessionId || properties.part?.sessionID || properties.part?.sessionId || properties.info?.sessionID || properties.info?.sessionId || properties.info?.id || "";
}

function textParts(parts) {
  return (Array.isArray(parts) ? parts : []).filter((part) => part?.type === "text" && !part.synthetic && !part.ignored)
    .map((part) => part.text).filter((text) => typeof text === "string").join("\n").trim();
}

function createLogger(client) {
  const fallback = (level, message) => {
    const method = typeof console[level] === "function" ? console[level] : console.log;
    method(`[ogmemory] ${message}`);
  };
  const write = (level, message) => {
    try {
      const operation = client?.app?.log?.({ body: { service: "ogmemory", level, message } });
      if (operation) void Promise.resolve(operation).catch(() => fallback(level, message));
      else fallback(level, message);
    } catch { fallback(level, message); }
  };
  return { info: (message) => write("info", message), warn: (message) => write("warn", message), error: (message) => write("error", message) };
}

function safeRuntimeErrorDiagnostic(error) {
  const allowedNames = new Set(["Error", "TypeError", "RangeError", "SyntaxError", "AggregateError"]);
  const allowedCodes = new Set(["EACCES", "EIO", "EMFILE", "ENFILE", "ENOENT", "ENOSPC", "ENOTDIR", "EPERM", "EROFS", "ETIMEDOUT"]);
  const name = allowedNames.has(error?.name) ? error.name : "Error";
  const code = allowedCodes.has(error?.code) ? `, code=${error.code}` : "";
  return `${name}${code}`;
}

function configurationDiagnostic(error) {
  const allowedReasons = new Set(["INVALID_ENDPOINT_POLICY", "INVALID_ENDPOINT_URL", "INVALID_FIELD_RANGE", "INVALID_FIELD_TYPE", "PARSE_ERROR", "UNKNOWN_FIELD"]);
  return allowedReasons.has(error?.configReason) ? error.configReason : "INVALID_CONFIGURATION";
}

async function updateDeletionFallback(store, sessionId, { expectedCursor, messages, cursor }) {
  if (store.updateDeletionFallback) return store.updateDeletionFallback(sessionId, { expectedCursor, messages, cursor });
  const state = store.get?.(sessionId) || {};
  if (expectedCursor !== undefined && state.deletionFallbackCursor !== expectedCursor) return false;
  if (!store.set) return false;
  await store.set(sessionId, { deletionFallbackMessages: messages, deletionFallbackCursor: cursor });
  return true;
}

async function updateTurnBoundaries(store, sessionId, options) {
  const current = store.get?.(sessionId)?.pendingTurnBoundaries || [];
  const guardedOptions = options.preferExisting && options.expected === undefined
    ? { ...options, expected: current }
    : options;
  if (store.updateTurnBoundaries) return store.updateTurnBoundaries(sessionId, guardedOptions);
  const pendingTurnBoundaries = mergeTurnBoundaries(current, guardedOptions);
  await store.set?.(sessionId, { pendingTurnBoundaries });
  return pendingTurnBoundaries;
}

export async function OgMemoryPlugin({ client }, dependencies = {}) {
  const logger = createLogger(client);
  let config;
  try { config = await (dependencies.loadConfig || loadConfig)(); }
  catch (error) { logger.error(`oGMemory configuration is invalid (reason=${configurationDiagnostic(error)}); automatic integration disabled`); return {}; }
  if (!config.enabled) { logger.info("oGMemory OpenCode integration is disabled by configuration"); return {}; }

  const compatibility = await (dependencies.detectCompatibility || detectCompatibility)();
  if (compatibility.supported === false) {
    logger.error(`oGMemory requires OpenCode >= ${MINIMUM_OPENCODE_VERSION}; automatic integration disabled`);
    return {};
  }
  if (compatibility.supported === null) logger.warn(`Could not determine OpenCode version; continuing fail-open (requires >= ${MINIMUM_OPENCODE_VERSION})`);
  if (compatibility.hasDispose === false) logger.warn(`OpenCode < ${DISPOSE_OPENCODE_VERSION} does not call plugin dispose; disk durability remains active but graceful shutdown flushing is unavailable`);

  try {
    const http = dependencies.createHttpClient ? dependencies.createHttpClient(config) : createHttpClient(config);
    const store = (dependencies.createSessionStore || createSessionStore)(join(config.runtimeDir, "sessions.json"), {
      limit: config.sessionStateLimit,
      ttlSeconds: config.sessionStateTtlSeconds,
      maxBytes: config.sessionStateMaxBytes,
      logger,
    });
    await store.load();
    const spool = (dependencies.createTraceSpool || createTraceSpool)({
      directory: join(config.runtimeDir, "trace-spool"),
      limit: config.traceSpoolLimit,
      ttlSeconds: config.traceSpoolTtlSeconds,
      send: http.trace,
      logger,
    });
    void spool.drain().catch((error) => logger.warn(`oGMemory trace replay failed open (${safeRuntimeErrorDiagnostic(error)})`));
    const trace = createTraceAdapter({ config, spool, logger });
    const compose = createComposeController({ http, logger, ttlMs: config.composeCacheTtlMs });
    const sync = createMessageSync({
      client, http, store, logger, timeoutMs: config.transcriptTimeoutMs, limit: config.transcriptLimit,
      onMessage: (sessionId, message) => trace.emit(`${message.role}_message`, sessionId, { objectId: message.id, content: message.content }),
      onMessages: (sessionId, messages) => trace.emitMany(messages.map((message) => ({
        type: `${message.role}_message`, sessionId, options: { objectId: message.id, content: message.content },
      }))),
    });
    void replayStoredSessions({ store, sync, trace }).catch((error) => logger.warn(`oGMemory pending-session recovery failed open (${safeRuntimeErrorDiagnostic(error)})`));

    return createHooks({ logger, trace, compose, sync, store, http, disposeTimeoutMs: config.disposeTimeoutMs });
  } catch (error) {
    logger.error(`oGMemory initialization failed open (${safeRuntimeErrorDiagnostic(error)}); automatic integration disabled`);
    return {};
  }
}

function deletionRecoveryPatch(state, sessionId) {
  const fallback = Array.isArray(state.deletionFallbackMessages) ? state.deletionFallbackMessages : [];
  const pending = Array.isArray(state.pendingMessages) ? state.pendingMessages : [];
  const boundaries = Array.isArray(state.pendingBoundaries) ? state.pendingBoundaries : [];
  const cursor = state.deletionFallbackCursor || sessionId;
  const boundaryIncluded = boundaries.some((boundary) => boundary.turnKey === cursor || boundary.cursor === cursor);
  const messageIncluded = fallback.length && pending.length > 0 && (state.pendingCursor === cursor || boundaryIncluded);
  const boundary = { cursor, turnKey: cursor, lastAssistant: "", count: fallback.length };
  return {
    pendingMessages: fallback.length && !messageIncluded ? [...pending, ...fallback] : pending,
    pendingCursor: fallback.length && !messageIncluded ? cursor : state.pendingCursor,
    pendingBoundaries: fallback.length && !boundaryIncluded ? [...boundaries, boundary] : boundaries,
    activeTurnKey: fallback.length && !messageIncluded ? cursor : state.activeTurnKey,
    pendingSessionEnd: true,
    pendingAt: new Date().toISOString(),
  };
}

export async function replayStoredSessions({ store, sync, trace }) {
  for (const [sessionId, listedState] of store.list()) {
    let state = listedState;
    const boundaries = Array.isArray(state.pendingTurnBoundaries) ? [...state.pendingTurnBoundaries] : [];
    if (state.pendingSessionEnd) {
      const result = await sync.sync(sessionId);
      boundaries.push(...(result.boundaries || []));
      const resolvedIds = result.resolvedMessageIds || result.committedMessageIds;
      if (Array.isArray(resolvedIds) && resolvedIds.includes(state.deletionFallbackCursor)) {
        await updateDeletionFallback(store, sessionId, { expectedCursor: state.deletionFallbackCursor, messages: [], cursor: null });
      }
      if (result.transcriptUnavailable && !result.transcriptGone) continue;
      state = store.get ? store.get(sessionId) : state;
      if (Array.isArray(state.pendingMessages) && state.pendingMessages.length) continue;
      const fallback = Array.isArray(state.deletionFallbackMessages) ? state.deletionFallbackMessages : [];
      if (fallback.length) {
        const patch = deletionRecoveryPatch(state, sessionId);
        await store.set(sessionId, patch);
        state = store.get ? store.get(sessionId) : { ...state, ...patch };
        const replayed = await sync.replayPending(sessionId);
        if (!replayed.success) continue;
        boundaries.push(...replayed.boundaries);
        if (!Array.isArray(replayed.resolvedMessageIds) || !replayed.resolvedMessageIds.includes(state.deletionFallbackCursor)) continue;
        await updateDeletionFallback(store, sessionId, { expectedCursor: state.deletionFallbackCursor, messages: [], cursor: null });
        state = store.get ? store.get(sessionId) : { ...state, deletionFallbackMessages: [], deletionFallbackCursor: null };
      }
      if (Array.isArray(state.pendingMessages) && state.pendingMessages.length) continue;
    } else if (Array.isArray(state.pendingMessages) && state.pendingMessages.length) {
      const result = await sync.replayPending(sessionId);
      if (!result.success) continue;
      boundaries.push(...result.boundaries);
    }
    const boundaryResult = await emitTurnBoundaries(trace, sessionId, boundaries);
    if (!boundaryResult.success) {
      await updateTurnBoundaries(store, sessionId, { retained: boundaryResult.boundaries, preferExisting: true });
      continue;
    }
    if (state.pendingSessionEnd) {
      if (!await trace.emit("session_end", sessionId, { objectId: sessionId, terminalState: "deleted" })) continue;
      await store.remove(sessionId);
    } else if (boundaries.length) {
      await updateTurnBoundaries(store, sessionId, { delivered: boundaryResult.boundaries });
    }
  }
}

async function emitTurnBoundaries(trace, sessionId, boundaries) {
  const unique = new Map();
  for (const boundary of boundaries || []) {
    if (!boundary?.cursor) continue;
    const key = boundary.turnKey || boundary.cursor;
    const existing = unique.get(key);
    if (existing?.terminalState === "error" && boundary.terminalState !== "error") continue;
    unique.set(key, boundary);
  }
  const normalized = [...unique.values()];
  let success = true;
  for (const boundary of normalized) {
    if (!await trace.emit("turn_end", sessionId, { objectId: boundary.cursor, terminalState: boundary.terminalState || "idle", content: boundary.lastAssistant })) success = false;
  }
  return { success, boundaries: normalized };
}

export function createHooks({ logger, trace, compose, sync, store, http, disposeTimeoutMs = 3000 }) {
  const eventLocks = new Map();
  const enqueueEvent = (sessionId, operation) => {
    const previous = eventLocks.get(sessionId) || Promise.resolve();
    const next = previous.catch(() => {}).then(operation).finally(() => {
      if (eventLocks.get(sessionId) === next) eventLocks.delete(sessionId);
    });
    eventLocks.set(sessionId, next);
    return next;
  };
  const stageBoundary = async (sessionId, result, terminalState = "idle") => {
    const expected = store.get?.(sessionId)?.pendingTurnBoundaries || [];
    const incoming = result.boundaries.map((boundary) => ({ ...boundary, terminalState: boundary.terminalState || "idle" }));
    const errorBoundary = terminalState === "error" ? incoming.at(-1) || expected.at(-1) : null;
    const errorTurnKey = errorBoundary?.turnKey || errorBoundary?.cursor;
    await updateTurnBoundaries(store, sessionId, { retained: incoming, expected, preferExisting: true, errorTurnKey });
  };
  const finalizeBoundaries = async (sessionId, result) => {
    const staged = store.get(sessionId).pendingTurnBoundaries || [];
    const emitted = await emitTurnBoundaries(trace, sessionId, [...staged, ...result.boundaries]);
    if (emitted.success) await updateTurnBoundaries(store, sessionId, { delivered: emitted.boundaries });
    else await updateTurnBoundaries(store, sessionId, { retained: emitted.boundaries, preferExisting: true });
    return emitted.success;
  };
  const clearCoveredDeletionFallback = async (sessionId, result) => {
    const state = store.get(sessionId);
    const fallbackCursor = state.deletionFallbackCursor;
    if (!fallbackCursor) return;
    const explicitResolved = Array.isArray(result?.resolvedMessageIds) ? result.resolvedMessageIds : null;
    if (!explicitResolved && !result?.success && !result?.committed) return;
    const resolvedIds = explicitResolved || result.committedMessageIds;
    const covered = Array.isArray(resolvedIds) && resolvedIds.includes(fallbackCursor);
    if (covered) await updateDeletionFallback(store, sessionId, { expectedCursor: fallbackCursor, messages: [], cursor: null });
  };
  const commitDeletionFallback = async (sessionId, result, observedFallback) => {
    const operation = async () => {
      const state = store.get(sessionId);
      const fallback = Array.isArray(state.deletionFallbackMessages) ? state.deletionFallbackMessages : [];
      const currentPending = Array.isArray(state.pendingMessages) && state.pendingMessages.length > 0;
      if (!fallback.length) {
        return { status: observedFallback ? "already-committed" : "skipped", state };
      }
      if (currentPending || !(result.success || result.transcriptGone) || !http?.afterTurn) {
        return { status: "skipped", state };
      }
      const cursor = state.deletionFallbackCursor || sessionId;
      const boundary = { cursor, turnKey: cursor, lastAssistant: "", count: fallback.length };
      try {
        await http.afterTurn(sessionId, fallback);
        const pending = Array.isArray(state.pendingTurnBoundaries) ? state.pendingTurnBoundaries : [];
        const byTurn = new Map(pending.map((candidate) => [candidate.turnKey || candidate.cursor, candidate]));
        const existing = byTurn.get(boundary.turnKey);
        if (existing?.terminalState !== "error") byTurn.set(boundary.turnKey, boundary);
        await store.set(sessionId, {
          deletionFallbackMessages: [], deletionFallbackCursor: null,
          pendingTurnBoundaries: [...byTurn.values()],
        });
        return { status: "committed", boundary, count: fallback.length, state: store.get(sessionId) };
      } catch {
        await store.set(sessionId, deletionRecoveryPatch(store.get(sessionId), sessionId));
        return { status: "failed", state: store.get(sessionId) };
      }
    };
    return store.runExclusive ? store.runExclusive(sessionId, operation) : operation();
  };
  return {
    "chat.message": async (input, output) => {
      const prompt = textParts(output?.parts);
      const messageId = output?.message?.id || input.messageID;
      await trace.emit("user_message", input.sessionID, { objectId: messageId, content: prompt });
      if (prompt && messageId) {
        try { await updateDeletionFallback(store, input.sessionID, { messages: [{ role: "user", content: prompt }], cursor: messageId }); }
        catch { logger.warn("oGMemory could not persist the deletion fallback; continuing fail-open"); }
      }
      await compose.recall(input.sessionID, messageId, prompt);
    },
    "experimental.chat.system.transform": async (input, output) => { compose.inject(input.sessionID, output); },
    "tool.execute.before": async (input, output) => {
      await trace.emit("tool_start", input.sessionID, { objectId: input.callID, content: output?.args, metadata: { tool: input.tool } });
    },
    "tool.execute.after": async (input, output) => {
      await trace.emit("tool_success", input.sessionID, { objectId: input.callID, terminalState: "success", content: output, metadata: { tool: input.tool } });
    },
    "experimental.session.compacting": async (input) => {
      try {
        const result = await sync.sync(input.sessionID);
        await clearCoveredDeletionFallback(input.sessionID, result);
        await trace.emit("pre_compact", input.sessionID, { objectId: result.cursor || input.sessionID, metadata: { message_count: result.count } });
      } catch {
        logger.warn("oGMemory pre-compact synchronization failed open");
      }
    },
    event: async ({ event }) => {
      const sessionId = sessionIdOf(event);
      if (!sessionId) {
        if (SESSION_EVENTS.has(event?.type)) logger.warn(`oGMemory skipped ${event.type} event without a session ID`);
        return;
      }
      return enqueueEvent(sessionId, async () => {
        try {
        const properties = event?.properties || {};
        if (event?.type === "session.created") {
          const parentSessionId = properties.info?.parentID;
          await trace.emit("session_start", sessionId, { objectId: sessionId, metadata: parentSessionId ? { parent_session_id: String(parentSessionId).slice(0, 512) } : {} });
        }
        if (event?.type === "session.error") {
          const state = store.get(sessionId);
          const errorSequence = store.nextErrorSequence
            ? await store.nextErrorSequence(sessionId)
            : Number(state.errorSequence || 0) + 1;
          if (!store.nextErrorSequence) await store.set(sessionId, { errorSequence });
          await trace.emit("session_error", sessionId, { objectId: `error-${errorSequence}`, terminalState: "error", content: properties.error });
          const result = await sync.sync(sessionId);
          await stageBoundary(sessionId, result, "error");
          await clearCoveredDeletionFallback(sessionId, result);
        }
        if (event?.type === "message.updated" && properties.info?.role === "assistant" && properties.info?.time?.completed) {
          const result = await sync.sync(sessionId);
          await stageBoundary(sessionId, result);
          await clearCoveredDeletionFallback(sessionId, result);
        }
        if (event?.type === "message.part.updated" && properties.part?.type === "tool" && properties.part?.state?.status === "error") {
          await trace.emit("tool_failure", sessionId, { objectId: properties.part.callID || properties.part.id, terminalState: "error", content: properties.part.state, metadata: { tool: properties.part.tool } });
        }
        if (event?.type === "session.idle") {
          const result = await sync.sync(sessionId);
          compose.clear(sessionId);
          await finalizeBoundaries(sessionId, result);
          await clearCoveredDeletionFallback(sessionId, result);
        }
        if (event?.type === "session.deleted") {
          try { await store.set(sessionId, { pendingSessionEnd: true, pendingAt: new Date().toISOString() }); }
          catch { logger.warn("oGMemory deletion tombstone persistence failed; continuing best-effort synchronization"); }
          let result = await sync.sync(sessionId);
          await clearCoveredDeletionFallback(sessionId, result);
          let fallbackState = store.get(sessionId);
          let fallback = Array.isArray(fallbackState.deletionFallbackMessages) ? fallbackState.deletionFallbackMessages : [];
          let currentPending = Array.isArray(fallbackState.pendingMessages) && fallbackState.pendingMessages.length > 0;
          if (result.transcriptUnavailable && !result.transcriptGone) {
            await stageBoundary(sessionId, result);
            await store.set(sessionId, { pendingSessionEnd: true, pendingAt: new Date().toISOString() });
            return;
          }
          const fallbackCommit = await commitDeletionFallback(sessionId, result, fallback.length > 0);
          fallbackState = fallbackCommit.state || store.get(sessionId);
          fallback = Array.isArray(fallbackState.deletionFallbackMessages) ? fallbackState.deletionFallbackMessages : [];
          currentPending = Array.isArray(fallbackState.pendingMessages) && fallbackState.pendingMessages.length > 0;
          if (fallbackCommit.status === "committed") {
              const boundary = fallbackCommit.boundary;
              result = {
                ...result, success: true, committed: true,
                committedMessageIds: [...new Set([...(result.committedMessageIds || []), boundary.cursor])],
                resolvedMessageIds: [...new Set([...(result.resolvedMessageIds || []), boundary.cursor])],
                count: Number(result.count || 0) + fallbackCommit.count, cursor: boundary.cursor,
                boundaries: [...(result.boundaries || []), boundary],
              };
          } else if (fallbackCommit.status === "already-committed") {
            result = { ...result, success: true, committed: true };
          } else if (fallbackCommit.status === "failed") {
            await stageBoundary(sessionId, result);
            return;
          } else if ((result.committed || result.resolvedMessageIds?.length) && !fallback.length && !currentPending) {
            result = { ...result, success: true };
          }
          if (!result.success && fallback.length) {
            const current = store.get(sessionId);
            await stageBoundary(sessionId, result);
            await store.set(sessionId, deletionRecoveryPatch(current, sessionId));
            return;
          }
          const finalized = await finalizeBoundaries(sessionId, result);
          compose.clear(sessionId);
          if (result.success && finalized) {
            if (await trace.emit("session_end", sessionId, { objectId: sessionId, terminalState: "deleted" })) await store.remove(sessionId);
            else await store.set(sessionId, { pendingSessionEnd: true, pendingAt: new Date().toISOString() });
          } else {
            await store.set(sessionId, { pendingSessionEnd: true, pendingAt: new Date().toISOString() });
          }
        }
        } catch (error) {
          logger.warn(`oGMemory event hook failed open (${safeRuntimeErrorDiagnostic(error)})`);
        }
      });
    },
    dispose: async () => {
      let timer;
      const flushInOrder = async () => {
        await Promise.allSettled([...eventLocks.values()]);
        await Promise.allSettled([Promise.resolve().then(() => sync.flush())]);
        await Promise.allSettled([Promise.resolve().then(() => trace.flush())]);
      };
      await Promise.race([
        flushInOrder(),
        new Promise((resolve) => { timer = setTimeout(resolve, disposeTimeoutMs); }),
      ]).finally(() => { clearTimeout(timer); trace.close?.(); });
    },
  };
}

export default OgMemoryPlugin;
