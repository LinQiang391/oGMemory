import { safeErrorMessage } from "./redaction.mjs";

const MAX_MESSAGE_CHARS = 100000;
const MAX_DELTA_CHARS = 1000000;
const MAX_TRANSCRIPT_PAGES = 10;

function unwrap(result) { return result?.data ?? result ?? []; }
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
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`OpenCode transcript request timed out after ${timeoutMs}ms`)), timeoutMs); timer.unref?.(); }),
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

export function createMessageSync({ client, http, store, logger = console, onMessage = () => {}, timeoutMs = 5000, limit = 500 }) {
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
      const page = unwrap(result);
      if (!Array.isArray(page)) throw new Error("Legacy OpenCode SDK returned an invalid transcript response");
      return page;
    }
    if (sdkMode === "flat") {
      const page = unwrap(await client.session.messages({ sessionID: sessionId, limit, ...(before ? { before } : {}) }));
      if (!Array.isArray(page)) throw new Error("OpenCode SDK returned an invalid transcript response");
      return page;
    }
    let flatError;
    try {
      const page = unwrap(await client.session.messages({ sessionID: sessionId, limit, ...(before ? { before } : {}) }));
      if (Array.isArray(page)) { sdkMode = "flat"; return page; }
    } catch (error) { flatError = error; }
    try {
      const page = unwrap(await client.session.messages({ path: { id: sessionId }, query: { limit } }));
      if (Array.isArray(page)) { sdkMode = "legacy"; return page; }
    } catch (error) { throw flatError || error; }
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
  async function sendPending(sessionId, prior) {
    if (!Array.isArray(prior.pendingMessages) || !prior.pendingMessages.length) return null;
    const boundaries = Array.isArray(prior.pendingBoundaries) ? prior.pendingBoundaries : [];
    try { await http.afterTurn(sessionId, prior.pendingMessages); }
    catch (error) {
      const isConfigurationError = error?.status === 401 || error?.status === 403;
      if (error?.retryable !== false || isConfigurationError) throw error;
      logger.warn?.(`Dropping permanently rejected pending oGMemory write payload (status=${error.status || "unknown"}); later messages remain enabled`);
      await store.set(sessionId, {
        cursor: prior.pendingCursor ?? prior.cursor,
        cursorCreatedAt: prior.pendingCursorCreatedAt ?? prior.cursorCreatedAt,
        pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
        rejectedWriteCount: Number(prior.rejectedWriteCount || 0) + 1,
        lastRejectedStatus: error.status || 0,
        lastRejectedAt: new Date().toISOString(),
      });
      return boundaries;
    }
    await store.set(sessionId, {
      cursor: prior.pendingCursor ?? prior.cursor,
      cursorCreatedAt: prior.pendingCursorCreatedAt ?? prior.cursorCreatedAt,
      pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
    });
    return boundaries;
  }
  async function run(sessionId) {
    let prior = store.get(sessionId);
    const replayedBoundaries = await sendPending(sessionId, prior) || [];
    prior = store.get(sessionId);
    const transcript = await withTimeout(fetchTranscript(sessionId, prior), timeoutMs);
    const raw = transcript.messages;
    if (transcript.gap) await store.set(sessionId, { transcriptGapCount: Number(prior.transcriptGapCount || 0) + 1, lastTranscriptGapAt: new Date().toISOString() });
    const delta = normalizeTranscript(raw, prior.cursor, prior.cursorCreatedAt);
    const currentBoundaries = buildTurnBoundaries(delta.records, prior.activeTurnKey);
    const turnKey = currentBoundaries.at(-1)?.turnKey || prior.activeTurnKey || delta.cursor;
    await Promise.all(delta.records.map((record) => onMessage(sessionId, record)));
    if (delta.messages.length) {
      try { await http.afterTurn(sessionId, delta.messages); }
      catch (error) {
        const isConfigurationError = error?.status === 401 || error?.status === 403;
        if (error?.retryable === false && !isConfigurationError) {
          logger.warn?.(`Dropping permanently rejected oGMemory write payload (status=${error.status || "unknown"}); later messages remain enabled`);
          await store.set(sessionId, {
            cursor: delta.cursor, cursorCreatedAt: delta.cursorCreatedAt,
            pendingMessages: [], pendingCursor: null, pendingCursorCreatedAt: null, pendingBoundaries: [], pendingAt: null,
            rejectedWriteCount: Number(prior.rejectedWriteCount || 0) + 1,
            lastRejectedStatus: error.status || 0,
            lastRejectedAt: new Date().toISOString(),
          });
        } else {
          await store.set(sessionId, { pendingMessages: delta.messages, pendingCursor: delta.cursor, pendingCursorCreatedAt: delta.cursorCreatedAt, pendingBoundaries: currentBoundaries, activeTurnKey: turnKey, pendingAt: new Date().toISOString() });
          throw error;
        }
      }
    }
    if (delta.cursor !== prior.cursor) await store.set(sessionId, { cursor: delta.cursor, cursorCreatedAt: delta.cursorCreatedAt, activeTurnKey: turnKey, updatedAt: new Date().toISOString(), pendingMessages: [], pendingAt: null });
    const boundaries = [...replayedBoundaries, ...currentBoundaries];
    return { success: true, count: boundaries.reduce((sum, boundary) => sum + boundary.count, 0), cursor: delta.cursor, lastAssistant: boundaries.at(-1)?.lastAssistant || "", boundaries };
  }
  return {
    sync(sessionId) {
      if (!sessionId) return Promise.resolve({ success: false, count: 0, cursor: null, lastAssistant: "", boundaries: [] });
      return enqueueSession(sessionId, () => run(sessionId)).catch((error) => {
        logger.warn?.(`oGMemory after_turn failed; cursor retained: ${safeErrorMessage(error)}`);
        return { success: false, count: 0, cursor: store.get(sessionId).cursor, lastAssistant: "", boundaries: [] };
      });
    },
    async replayPending(sessionId) {
      return enqueueSession(sessionId, async () => {
        try {
          const boundaries = await sendPending(sessionId, store.get(sessionId));
          return { success: true, boundaries: boundaries || [] };
        } catch (error) {
          logger.warn?.(`oGMemory pending after_turn replay failed: ${safeErrorMessage(error)}`);
          return { success: false, boundaries: [] };
        }
      });
    },
    async flush() { await Promise.allSettled([...locks.values()]); },
  };
}
