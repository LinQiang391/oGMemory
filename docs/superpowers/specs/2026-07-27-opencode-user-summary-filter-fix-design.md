# OpenCode user summary filtering fix

## Problem

The OpenCode plugin drops normal user messages before calling oGMemory
`/api/v1/after_turn`. OpenCode 1.18.6 represents normal user messages with a
`summary` object containing diff metadata, while assistant compaction summaries
use `summary: true`. The plugin currently treats every truthy `summary` value as
generated content, so ordinary user messages are excluded and archives retain
only assistant messages.

## Scope

Change only transcript generated-message classification in
`opencode-plugin/lib/message-sync.mjs`. Preserve the existing filtering of
synthetic, ignored, compaction, and compaction-continuation messages. Do not
change the oGMemory HTTP API or archive implementation.

## Behavior

- A user message with `summary: { diffs: [...] }` is treated as a normal user
  message and is sent to `/api/v1/after_turn`.
- An assistant message with `summary: true` remains excluded as a generated
  compaction summary.
- Existing explicit generated-message flags remain excluded.
- Cursor, retry, pagination, and turn-boundary behavior remain unchanged.

## Verification

1. Add a regression fixture matching the OpenCode 1.18.6 user-message shape and
   prove that normalization returns both user and assistant messages.
2. Prove that an assistant `summary: true` message is still filtered.
3. Run the complete OpenCode plugin test suite and package check.
4. Run a real WSL OpenCode session through the plugin and verify that the
   oGMemory archive contains both roles and produces retrievable memory index
   data when extraction succeeds.

## Failure handling

The existing fail-open, durable retry, and cursor retention behavior is not
modified. If the WSL extraction backend fails independently of transcript
normalization, archive role verification remains the acceptance criterion for
this defect and the backend failure is reported separately.
