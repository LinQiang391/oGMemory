import { safeErrorMessage } from "./redaction.mjs";

const MAX_MESSAGE_CHARS = 100000;
const MAX_DELTA_CHARS = 1000000;
const MAX_TRANSCRIPT_PAGES = 10;

function unwrap(result) { return result?.data ?? result ?? []; }
function throwResolvedSdkError(result) {
  if (!result?.error) return;
  const error = new Error("OpenCode SDK returned an error response");
  error.status = Number(result?.response?.status || result?.error?.status || 0);
  throw error;
}
function created(message) { return message?.info?.time?.created ?? message?.time?.created ?? 0; }
function idOf(message) { return message?.info?.id ?? message?.id ?? ""; }
function isGenerated(message) {
  const info = message?.info || message || {};
  const metadata = info.metadata || {};
  const isAssistantSummary = info.role === "assistant" && info.summary === true;
  return Boolean(info.synthetic || info.ignored || info.compaction || isAssistantSummary || metadata.compaction || metadata.compaction_continue || metadata.compactionContinue);
}

function contentOf(message) {
  const parts = Array.isArray(message?.parts) ? message.parts : [];
  const content = [];
  let length = 0;
  for (const part of parts) {
    const text = part?.text;
    if (part?.type !== "text" || part.synthetic || part.ignored || typeof text !== "string") continue;
    const separator = content.length ? 1 : 0;
    const available = MAX_MESSAGE_CHARS - length - separator;
    if (available <= 0) break;
    content.push(text.slice(0, available));
    length += separator + Math.min(text.length, available);
    if (text.length > available) break;
  }
  return content.join("\n").trim();
}

export function normalizeTranscript(messages, cursor = null, cursorCreatedAt = 0) {
  const sorted = [...(Array.isArray(messages) ? messages : [])].sort((a, b) => created(a) - created(b) || idOf(a).localeCompare(idOf(b)));
  const index = cursor ? sorted.findIndex((message) => idOf(message) === cursor) : -1;
  const delta = !cursor ? sorted : index >= 0 ? sorted.slice(index + 1) : cursorCreatedAt
    ? sorted.filter((message) => created(message) > cursorCreatedAt || (created(message) === cursorCreatedAt && idOf(message) > cursor))
    : [];
  let remaining = MAX_DELTA_CHARS;
  let cursorMessage = null;
  const records = [];
  for (const message of delta) {
    if (isGenerated(message)) { cursorMessage = message; continue; }
    const role = message?.info?.role ?? message?.role;
    if (role !== "user" && role !== "assistant") { cursorMessage = message; continue; }
    const content = contentOf(message);
    if (!content) { cursorMessage = message; continue; }
    if (content.length > remaining) break;
    records.push({ id: idOf(message), role, content });
    remaining -= content.length;
    cursorMessage = message;
  }
  return {
    records,
    messages: records.map(({ role, content }) => ({ role, content })),
    cursor: cursorMessage ? idOf(cursorMessage) : cursor,
    cursorCreatedAt: cursorMessage ? created(cursorMessage) : cursorCreatedAt,
  };
}

function withTimeout(promise, timeoutMs) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`OpenCode transcript request timed out after ${timeoutMs}ms`)), timeoutMs); }),
  ]).finally(() => clearTimeout(timer));
}

function buildTurnBoundaries(records, activeTurnKey) {
  const boundaries = [];
  let current = null;
  for (const record of records) {
    if (record.role === "user") {
      if (current) boundaries.push(current);
      current = { cursor: record.id, lastAssistant: "", count: 1, turnKey: record.id };
      continue;
    }
    if (!current) current = { cursor: record.id, lastAssistant: "", count: 0, turnKey: activeTurnKey || record.id };
    current.cursor = record.id;
    current.lastAssistant = record.content;
    current.count += 1;
  }
  if (current) boundaries.push(current);
  return boundaries;
}

function boundaryCount(boundary) {
  const count = Number(boundary?.count);
  return Number.isSafeInteger(count) && count >= 0 ? count : 0;
}

function mergeBoundaries(...groups) {
  const merged = new Map();
  for (const boundary of groups.flat()) {
    if (!boundary?.cursor) continue;
    const key = boundary.turnKey || boundary.cursor;
    const existing = merged.get(key);
    if (existing?.terminalState === "error" && boundary.terminalState !== "error") continue;
    merged.set(key, boundary);
  }
  return [...merged.values()];
}

export function createMessageSync({ client, http, store, logger = console, onMessage = () => {}, onMessages = null, timeoutMs = 5000, limit = 500 }) {
  const locks = new Map();
  let sdkMode = null;
  function enqueueSession(sessionId, operation) {
    const previous = locks.get(sessionId) || Promise.resolve();
    const next = previous.catch(() => {}).then(operation).finally(() => {
      if (locks.get(sessionId) === next) locks.delete(sessionId);
    });
    locks.set(sessionId, next);
    return next;
  }
  async function requestPage(sessionId, before) {
    if (sdkMode === "legacy") {
      if (before) throw new Error("Legacy OpenCode SDK cannot paginate transcript gaps; cursor retained for safe recovery");
      const result = await client.session.messages({ path: { id: sessionId }, query: { limit } });
      throwResolvedSdkError(result);
      const page = unwrap(result);
      if (!Array.isArray(page)) throw new Error("Legacy OpenCode SDK returned an invalid transcript response");
      return page;
    }
    if (sdkMode === "flat") {
      const result = await client.session.messages({ sessionID: sessionId, limit, ...(before ? { before } : {}) });
      throwResolvedSdkError(result);
      const page = unwrap(result);
      if (!Array.isArray(page)) throw new Error("OpenCode SDK returned an invalid transcript response");
      return page;
    }
    let flatError;
    try {
      const result = await client.session.messages({ sessionID: sessionId, limit, ...(before ? { before } : {}) });
      throwResolvedSdkError(result);
      const page = unwrap(result);
      if (Array.isArray(page)) { sdkMode = "flat"; return page; }
    } catch (error) { flatError = error; }
    try {
      const result = await client.session.messages({ path: { id: sessionId }, query: { limit } });
      throwResolvedSdkError(result);
      const page = unwrap(result);
      if (Array.isArray(page)) { sdkMode = "legacy"; return page; }
    } catch (error) {
      const gone = (value) => [404, 410].includes(Number(value?.status || value?.response?.status || 0));
      const failure = flatError || error;
      const combined = new Error(safeErrorMessage(failure));
      combined.transcriptGone = Boolean(flatError && gone(flatError) && gone(error));
      throw combined;
    }
    throw flatError || new Error("OpenCode SDK returned an invalid transcript response");
  }
  async function fetchTranscript(sessionId, prior) {
    const collected = [];
    let before;
    for (let pageNumber = 0; pageNumber < MAX_TRANSCRIPT_PAGES; pageNumber += 1) {
      const page = await requestPage(sessionId, before);
      if (!Array.isArray(page) || !page.length) return { messages: collected, gap: false };
      collected.push(...page);
      if (!prior.cursor || page.some((message) => idOf(message) === prior.cursor) || page.length < limit) return { messages: collected, gap: false };
      if (sdkMode === "legacy") {
        logger.warn?.(`Legacy OpenCode transcript gap exceeded ${limit} messages; dropping the unavailable prefix and resuming from the bounded window`);
        return { messages: collected, gap: true };
      }
      const oldest = [...page].sort((a, b) => created(a) - created(b) || idOf(a).localeCompare(idOf(b)))[0];
      const nextBefore = idOf(oldest);
      if (!nextBefore || nextBefore === before) throw new Error("OpenCode transcript pagination did not advance");
      before = nextBefore;
    }
    logger.warn?.(`OpenCode transcript gap exceeded ${MAX_TRANSCRIPT_PAGES * limit} messages; dropping the unavailable prefix and resuming from the bounded window`);
    return { messages: collected, gap: true };
  }
  function recordProgress(progress, boundaries, committed, resolvedIds = []) {
    if (!progress) return;
    progress.boundaries.push(...boundaries);
    for (const boundary of boundaries) {
      if (boundary?.turnKey) progress.resolvedMessageIds.add(boundary.turnKey);
      if (boundary?.cursor) progress.resolvedMessageIds.add(boundary.cursor);
    }
    for (const id of resolvedIds) if (id) progress.resolvedMessageIds.add(id);
    if (!committed) return;
    progress.committed = true;
    for (const boundary of boundaries) {
      if (boundary?.turnKey) progress.committedMessageIds.add(boundary.turnKey);
      if (boundary?.cursor) progress.committedMessageIds.add(boundary.cursor);
    }
    for (const id of resolvedIds) if (id) progress.committedMessageIds.add(id);
  }
  async function sendPending(sessionId, prior, progress) {
    if (!Array.isArray(prior.pendingMessages) || !prior.pendingMessages.length) return { boundaries: [], committed: false };
    const boundaries = Array.isArray(prior.pendingBoundaries) ? prior.pendingBoundaries : [];
    try { await http.afterTurn(sessionId, prior.pendingMessages); }
    catch (error) {
      const isConfigurationError = error?.status === 401 || error?.status === 403;
      if (error?.retryable !== false || isConfigurationError) throw error;
      logger.warn?.(`Dropping permanently rejected pending oGMemory write payload (status=${error.status || "unknown"}); later messages remain enabled`);
      recordProgress(progress, boundaries, false, [prior.pendingCursor]);
      await store.set(sessionId, {
        cursor: prior.pendingCursor ?? prior.cursor,
        cursorCreatedAt: prior.pendingCursorCreatedAt ?? prior.cursorCreatedAt,
        pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
        rejectedWriteCount: Number(prior.rejectedWriteCount || 0) + 1,
        lastRejectedStatus: error.status || 0,
        lastRejectedAt: new Date().toISOString(),
        pendingTurnBoundaries: mergeBoundaries(prior.pendingTurnBoundaries || [], boundaries),
      });
      return { boundaries, committed: false };
    }
    recordProgress(progress, boundaries, true, [prior.pendingCursor]);
    await store.set(sessionId, {
      cursor: prior.pendingCursor ?? prior.cursor,
      cursorCreatedAt: prior.pendingCursorCreatedAt ?? prior.cursorCreatedAt,
      pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
      pendingTurnBoundaries: mergeBoundaries(prior.pendingTurnBoundaries || [], boundaries),
    });
    return { boundaries, committed: true };
  }
  async function run(sessionId, progress) {
    let prior = store.get(sessionId);
    await sendPending(sessionId, prior, progress);
    prior = store.get(sessionId);
    let transcript;
    try { transcript = await withTimeout(fetchTranscript(sessionId, prior), timeoutMs); }
    catch (error) {
      const failure = new Error(safeErrorMessage(error));
      const status = Number(error?.status || error?.response?.status || 0);
      failure.transcriptUnavailable = true;
      failure.transcriptGone = Boolean(error?.transcriptGone || (sdkMode && (status === 404 || status === 410)));
      throw failure;
    }
    const raw = transcript.messages;
    if (transcript.gap) await store.set(sessionId, { transcriptGapCount: Number(prior.transcriptGapCount || 0) + 1, lastTranscriptGapAt: new Date().toISOString() });
    const delta = normalizeTranscript(raw, prior.cursor, prior.cursorCreatedAt);
    const currentBoundaries = buildTurnBoundaries(delta.records, prior.activeTurnKey);
    const turnKey = currentBoundaries.at(-1)?.turnKey || prior.activeTurnKey || delta.cursor;
    if (onMessages) {
      if (await onMessages(sessionId, delta.records) === false) throw new Error("Unable to durably spool recovered oGMemory message traces");
    } else await Promise.all(delta.records.map((record) => onMessage(sessionId, record)));
    if (delta.messages.length) {
      try {
        await http.afterTurn(sessionId, delta.messages);
        recordProgress(progress, currentBoundaries, true, delta.records.map((record) => record.id));
      }
      catch (error) {
        const isConfigurationError = error?.status === 401 || error?.status === 403;
        if (error?.retryable === false && !isConfigurationError) {
          logger.warn?.(`Dropping permanently rejected oGMemory write payload (status=${error.status || "unknown"}); later messages remain enabled`);
          recordProgress(progress, currentBoundaries, false, delta.records.map((record) => record.id));
          await store.set(sessionId, {
            cursor: delta.cursor, cursorCreatedAt: delta.cursorCreatedAt,
            pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
            rejectedWriteCount: Number(prior.rejectedWriteCount || 0) + 1,
            lastRejectedStatus: error.status || 0,
            lastRejectedAt: new Date().toISOString(),
            pendingTurnBoundaries: mergeBoundaries(prior.pendingTurnBoundaries || [], currentBoundaries),
          });
        } else {
          await store.set(sessionId, { pendingMessages: delta.messages, pendingCursor: delta.cursor, pendingCursorCreatedAt: delta.cursorCreatedAt, pendingBoundaries: currentBoundaries, activeTurnKey: turnKey, pendingAt: new Date().toISOString() });
          throw error;
        }
      }
    }
    if (delta.cursor !== prior.cursor) await store.set(sessionId, {
      cursor: delta.cursor, cursorCreatedAt: delta.cursorCreatedAt, activeTurnKey: turnKey, updatedAt: new Date().toISOString(), pendingMessages: [], pendingAt: null,
      pendingTurnBoundaries: mergeBoundaries(store.get(sessionId).pendingTurnBoundaries || [], currentBoundaries),
    });
    const boundaries = progress.boundaries;
    return {
      success: true, committed: progress.committed,
      committedMessageIds: [...progress.committedMessageIds],
      resolvedMessageIds: [...progress.resolvedMessageIds],
      count: boundaries.reduce((sum, boundary) => sum + boundaryCount(boundary), 0),
      cursor: delta.cursor, lastAssistant: boundaries.at(-1)?.lastAssistant || "", boundaries,
    };
  }
  return {
    sync(sessionId) {
      if (!sessionId) return Promise.resolve({ success: false, count: 0, cursor: null, lastAssistant: "", boundaries: [] });
      const progress = { committed: false, committedMessageIds: new Set(), resolvedMessageIds: new Set(), boundaries: [] };
      const operation = () => run(sessionId, progress);
      return enqueueSession(sessionId, () => store.runExclusive ? store.runExclusive(sessionId, operation) : operation()).catch((error) => {
        logger.warn?.(`oGMemory after_turn failed; cursor retained: ${safeErrorMessage(error)}`);
        const current = store.get(sessionId);
        const boundaries = progress.boundaries;
        return {
          success: false, committed: progress.committed,
          committedMessageIds: [...progress.committedMessageIds],
          resolvedMessageIds: [...progress.resolvedMessageIds],
          count: boundaries.reduce((sum, boundary) => sum + boundaryCount(boundary), 0),
          cursor: current.cursor, lastAssistant: boundaries.at(-1)?.lastAssistant || "", boundaries,
          transcriptUnavailable: Boolean(error.transcriptUnavailable), transcriptGone: Boolean(error.transcriptGone),
        };
      });
    },
    async replayPending(sessionId) {
      return enqueueSession(sessionId, async () => {
        const progress = { committed: false, committedMessageIds: new Set(), resolvedMessageIds: new Set(), boundaries: [] };
        const operation = async () => {
          const replayed = await sendPending(sessionId, store.get(sessionId), progress);
          return {
            success: true, committed: replayed.committed,
            committedMessageIds: [...progress.committedMessageIds], resolvedMessageIds: [...progress.resolvedMessageIds],
            boundaries: replayed.boundaries,
          };
        };
        try {
          return await (store.runExclusive ? store.runExclusive(sessionId, operation) : operation());
        } catch (error) {
          logger.warn?.(`oGMemory pending after_turn replay failed: ${safeErrorMessage(error)}`);
          return {
            success: false, committed: progress.committed,
            committedMessageIds: [...progress.committedMessageIds], resolvedMessageIds: [...progress.resolvedMessageIds],
            boundaries: progress.boundaries,
          };
        }
      });
    },
    async flush() { await Promise.allSettled([...locks.values()]); },
  };
}
