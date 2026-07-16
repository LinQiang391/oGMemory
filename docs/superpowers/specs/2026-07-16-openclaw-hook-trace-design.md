# OpenClaw Hook Trace Design

## Goal

Capture OpenClaw lifecycle and tool traces alongside the existing Claude Code
traces. Each agent converts its native hook payload into the same internal event
shape before sending it to the existing trace endpoint. Once converted, the
trace pipeline does not branch on agent type.

Trace content is retained for later analysis, but supported credentials and PII
must be irreversibly replaced with placeholders before the event leaves the
hook-to-event adapter.

## Scope

This change will:

- register trace-producing hooks in the existing OpenClaw context-engine plugin;
- give Claude Code and OpenClaw explicit hook-to-event adapters;
- apply equivalent irreversible redaction in both adapters;
- send both adapters' canonical events to `POST /api/v1/trace/events`;
- preserve the existing SQL `trace_events` and `trace_sessions` schemas;
- keep existing Claude Code hook requests compatible.

This change will not introduce an agent registry, agent-specific fields in the
canonical event, reversible secret storage, an OpenTelemetry pipeline, or a new
trace endpoint.

## Architecture

```text
Claude Code hook JSON -> claudeHookToEvent() --+
                                                +-> canonical event -> existing trace endpoint
OpenClaw typed hook  -> openClawHookToEvent() --+
```

Agent-specific hook names and payload layouts are consumed only by their
adapter. The canonical event contains semantic event data, not the producer's
identity. The existing HTTP identity envelope (`account_id`, `user_id`,
`agent_id`, `session_id`, `request_id`, and `timestamp`) remains unchanged.

The adapters send events through the existing non-blocking trace transport. A
trace upload failure must not fail an agent turn.

## Canonical Events

The current event object remains the compatibility baseline. Adapters produce
one of these semantic event types:

| Meaning | Canonical `event_type` |
| --- | --- |
| Session opened | `session_start` |
| User input | `user_message` |
| Tool will run | `tool_start` |
| Tool completed | `tool_success` |
| Tool failed | `tool_failure` |
| Assistant turn completed | `turn_end` |
| Compaction will run | `pre_compact` |
| Session closed | `session_end` |

Relevant generic fields such as `run_id`, `tool_name`, `tool_use_id`,
`tool_input`, `tool_response`, `error`, `assistant_message`, `reason`, sizes,
and status may accompany the event. `run_id` is correlation data rather than an
agent identifier and is included when the hook provides it.

Claude Code's existing `hook_type` may remain in legacy requests. New internal
logic must use canonical semantics after the HTTP compatibility boundary and
must not branch on Claude or OpenClaw.

## Hook Mapping

| Claude Code hook | OpenClaw hook | Canonical event |
| --- | --- | --- |
| `SessionStart` | `session_start` | `session_start` |
| `UserPromptSubmit` | agent-run input hook | `user_message` |
| `PreToolUse` | `before_tool_call` | `tool_start` |
| `PostToolUse` | successful `after_tool_call` | `tool_success` |
| `PostToolUseFailure` | failed `after_tool_call` | `tool_failure` |
| `Stop` | `agent_end` | `turn_end` |
| `PreCompact` | `before_compaction` | `pre_compact` |
| `SessionEnd` | `session_end` | `session_end` |

Only hooks supported by the repository's OpenClaw plugin SDK/runtime will be
registered. The implementation will verify exact event and context fields
against the installed SDK before binding them.

## Irreversible Redaction

Redaction happens inside each hook-to-event adapter, before transport. It is
recursive across strings, objects, arrays, tool inputs, tool results, errors,
prompts, and assistant output.

The first version detects:

- API keys, tokens, passwords, authorization values, cookies, and secret-like
  named fields;
- email addresses and phone numbers;
- national identity card, passport, and bank-card-like values;
- IPv4 and IPv6 addresses;
- credentials embedded in common connection strings.

Detected values are replaced with typed placeholders such as `{{email_1}}` and
`{{api_key_1}}`. Equal sensitive values repeated within one event use the same
placeholder. Numbering and mappings do not cross event boundaries, and the
original values are never persisted for restoration.

The Claude and OpenClaw implementations must conform to the same fixture-based
redaction contract even though one adapter is Python and the other is
JavaScript.

## Existing Trace Integration

Both adapters post to `POST /api/v1/trace/events`. The endpoint continues to
authenticate the identity envelope and persist through `MemoryService` into
`trace_events`. Completed turns continue to aggregate into `trace_sessions`.

The server keeps compatibility with existing Claude events that identify a turn
boundary through `hook_type == "Stop"`. New adapters identify the same boundary
with `event_type == "turn_end"`; the completed assistant text remains in the
generic `assistant_message` field. No agent discriminator or SQL migration is
required.

## Reliability and Limits

- Hook-to-event conversion is deterministic and bounded.
- Existing body-size and field-size limits remain in force.
- OpenClaw trace delivery uses a short timeout and bounded retry/spooling
  behavior appropriate to the plugin runtime.
- Trace delivery errors are observable through plugin logs but do not block
  model, tool, or session execution.
- Malformed or unauthenticated events continue to be rejected by the server.

## Testing

Tests will prove:

1. Claude and OpenClaw equivalents produce the same canonical event semantics.
2. Every supported sensitive class becomes a typed placeholder and plaintext is
   absent from the outbound payload.
3. Repeated values within one event reuse a placeholder and no reversible map is
   retained.
4. OpenClaw success, failure, assistant completion, compaction, and session
   lifecycle hooks send trace events without interrupting the agent on failure.
5. Existing Claude hook behavior and legacy HTTP requests remain compatible.
6. Canonical OpenClaw events enter the current SQL event and turn aggregation
   flow without a schema change.

Implementation will follow test-driven development: each adapter, redaction
case, compatibility behavior, and integration boundary receives a failing test
before production code is changed.
