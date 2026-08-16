/**
 * DeepSeek Harness lifecycle integration for oGMemory.
 *
 * Recall is resolved at the first pre-step of each turn and appended to that
 * same model request as a source-attributed snapshot message. Write-back
 * starts only after the durable turn/end event commits, then runs on a
 * per-session ordered queue.
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'

import { resolveConfig } from './lib/config.js'
import { OgMemoryClient } from './lib/client.js'
import { scopedPendingDir } from './lib/pending-queue.js'
import { extractDirectUserPrompt, renderComposeContext } from './lib/recall.js'
import { registerOgMemoryTools } from './lib/tools.js'
import { guardOgMemoryUri } from './lib/uri-guard.js'
import { collectTurnMessages, OrderedWriteQueue } from './lib/writeback.js'

export const name = 'ogmemory-lifecycle'

export const inject = ['agents', 'credentials', 'sessions', 'tools']

export function apply(ctx, inputConfig = {}) {
  const config = resolveConfig(inputConfig)
  const client = new OgMemoryClient(ctx, config)
  const activeSessions = new Set()
  const queue = new OrderedWriteQueue({
    logger: ctx.logger,
    maxAttempts: config.writebackMaxAttempts,
    initialDelayMs: config.writebackInitialDelayMs,
    maxDelayMs: config.writebackMaxDelayMs,
    pendingDir: scopedPendingDir(config.pendingDir, config),
    shutdownTimeoutMs: config.shutdownTimeoutMs,
    send: operation => operation.kind === 'dispose'
      ? client.dispose(operation)
      : client.afterTurn(operation),
  })
  void queue.start()
  registerOgMemoryTools(ctx, client, config)
  if (config.toolsEnabled) ctx.on('tools/pre-execute', guardOgMemoryUri)

  ctx.on('agent/session-start', ({ agent }) => {
    activeSessions.add(String(agent.session.id))
  })

  ctx.on('agent/pre-step', async ({ agent, step, signal }, next) => {
    const decision = await next()
    if (!config.composeEnabled || decision.kind === 'reject' || step !== 1 || signal.aborted) {
      return decision
    }

    const prompt = extractDirectUserPrompt(decision.messages, config.maxPromptChars)
    if (prompt.length === 0) {
      return decision
    }

    try {
      const result = await client.compose({
        sessionId: String(agent.session.id),
        prompt,
        tokenBudget: config.tokenBudget,
      }, signal)
      if (signal.aborted) return decision
      const context = renderComposeContext(result, config.maxContextChars)
      ctx.logger.debug(
        `ogmemory recall session=${String(agent.session.id)} chars=${context.length}`,
      )
      if (context.length === 0) return decision
      return {
        kind: 'enter',
        messages: [
          ...decision.messages,
          createUserMessage({
            source: {
              kind: 'plugin',
              plugin: name,
              form: 'snapshot',
              sections: [{ name: 'ogmemory', text: context }],
            },
            content: [{ type: 'text', text: context }],
          }),
        ],
      }
    } catch (error) {
      if (!signal.aborted) {
        ctx.logger.warn(
          `ogmemory recall failed session=${String(agent.session.id)}: ${errorMessage(error)}`,
        )
      }
    }
    return decision
  }, { prepend: true })

  ctx.on('session/event', (session, event) => {
    if (!config.writebackEnabled || event.type !== 'turn/end') return
    const messages = collectTurnMessages(session.events, event.data.turn, {
      captureToolResults: config.captureToolResults,
      maxMessageChars: config.maxMessageChars,
      maxToolResultChars: config.maxToolResultChars,
      maxTurnMessages: config.maxTurnMessages,
    })
    if (messages.length === 0) return
    queue.enqueue(String(session.id), {
      kind: 'afterTurn',
      sessionId: String(session.id),
      messages,
      prePromptMessageCount: 0,
      forceExtract: config.forceExtract,
      wait: true,
    })
  })

  ctx.on('session/flush', session => queue.flush(String(session.id)))

  ctx.on('session/disposed', async session => {
    const sessionId = String(session.id)
    if (config.disposeEnabled) queue.enqueue(sessionId, { kind: 'dispose', sessionId })
    await queue.retire(sessionId)
    activeSessions.delete(sessionId)
  })

  ctx.effect(() => async () => {
    if (config.disposeEnabled) {
      for (const sessionId of activeSessions) {
        queue.enqueue(sessionId, { kind: 'dispose', sessionId })
      }
    }
    await queue.dispose()
  }, 'ogmemory writeback queue')
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export { resolveConfig } from './lib/config.js'
export { OgMemoryClient, OgMemoryHttpError } from './lib/client.js'
export { extractDirectUserPrompt, renderComposeContext } from './lib/recall.js'
export { collectTurnMessages, OrderedWriteQueue } from './lib/writeback.js'
