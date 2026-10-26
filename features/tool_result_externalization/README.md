# Tool-result externalization and Turn Ledger

Large tool-result externalization is opt-in. With the feature disabled, the
service does not create the externalizer, Turn Ledger, Artifact state, or
externalization state, and the legacy message/SessionBuffer path is unchanged.

## Session-scoped Artifact storage

Filesystem and AGFS backends store new tool-result Artifacts under the owning
account and user using oGMemory's plain-text Session hierarchy:

```text
<root-or-mount>/accounts/<account_id>/user/<user_id>/agents/<agent_id>/sessions/<session_id>/tool-results/<artifact_id>/
  output.txt
  metadata.json
```

No hashed Session directory or legacy-path fallback is used. PostgreSQL keeps
the equivalent account, Session, user, agent, and Artifact identity in table
columns. The public `ogmemory://tool-results/<artifact_id>` reference remains
stable; its Stub carries `session_id`, while the authenticated request context
and Artifact metadata provide the owning account.

Explicit Session deletion owns Artifact reclamation. The
OpenClaw adapter keeps ordinary `session_end`/`dispose` non-destructive, but a
typed `session_end` with `reason=deleted` calls the authenticated Session DELETE
endpoint after flushing pending checkpoints. Compaction, reset, and Gateway
shutdown never use this destructive path. There is no separate lifecycle
callback or generic-dispatch cleanup entrypoint.

Explicit deletion remains available when externalization is currently disabled,
so Artifacts created by an earlier enabled configuration still follow the
OpenClaw Session lifecycle and are reclaimed.

## Multi-resolution v2 stubs

New Artifacts persist a bounded deterministic candidate pool extracted from the
complete decoded tool result. No per-result LLM call is made. The scanner gives
priority to errors/exceptions, status codes, exact identifiers, important
numbers, and explicit constraints. Each candidate retains literal UTF-8 byte
offsets and a SHA-256-bound evidence snippet.

The model-facing v2 stub adds safely JSON-encoded `key_facts`,
`selected_evidence`, `retained_records`, and `coverage` fields beside
`synopsis`. Candidate ranking uses only the current user question and terminal
assistant text; that ranking context is never written to Artifact metadata.
`coverage` records the complete
scan range, whether the candidate or stub projection was truncated, and what
was omitted. The existing `synopsis_max_chars` budget is shared by `synopsis`,
`key_facts`, `selected_evidence`, `retained_records`, and `coverage`; no second
Artifact summary budget is introduced. Unique critical raw records are the only
intentional soft-overflow exception. `stub_summary_turn_chars` bounds that
combined summary across one governed Turn. The v2 Stub is the sole canonical
tool-result Stub format; older v1 Stub variants are not generated or accepted.

`retained_records` contains original records rather than generated summaries.
For JSON object arrays it preserves the original array path and record object
shape; CSV rows remain literal. Repetitive string/log records may use a
reversible `{"template":"...[]...","rows":[[...]]}` encoding: repeated literal
fragments occur once and each row stores only its variable values. Expanding
the template reproduces every retained record exactly. Template discovery uses
at most 128 distributed retained-record samples, then validates every retained
record. Structurally different rows remain literal in `unmatched_records`;
`unmatched_positions` preserves their source order. Identical complete records
are still deduplicated by first occurrence and the final projection is restored
to source order. The defaults explicitly keep JSON structure, set
`include_summaries=False`, use a 2.0 standard-deviation anomaly threshold, and
reserve at most 12 position anchors with the first 3 and last 2 considered
first.

Errors/exceptions/failures/critical timeouts, rare fields, rare status values,
numeric outliers, numeric change points with ±1 neighbors, and query-matching
records are critical. All unique critical records are retained even when they
alone exceed `synopsis_max_chars`; `coverage.critical_records_over_budget`
reports that intentional exception. Remaining room is filled with front-heavy
search anchors, back-heavy log anchors, balanced time-series anchors, or
distributed generic-array anchors.

Facts already represented by `retained_records` are not repeated in
`key_facts`, and their matching `selected_evidence` entries are omitted as
well. Facts from records that were not retained remain eligible for
`key_facts` and evidence. When no record projection exists, fact and evidence
generation follows the original path unchanged. `coverage.covered_key_facts`
and `coverage.covered_evidence` report this non-lossy deduplication.

### Cost gates

The record path uses deterministic short circuits before expensive work:

- normal externalization thresholds reject small tool results before this code;
- invalid JSON falls back to line handling, and collections with fewer than five
  records skip record analysis;
- template discovery samples at most 128 retained string records, with at most
  4,096 characters per discovery row and 65,536 discovery characters total;
- a cheap shared 4-gram probe must find repeated literal structure before the
  sequence matcher runs;
- the discovered template is checked linearly against every selected retained
  record; matching rows become variables and mismatches remain literal;
- the combined template plus unmatched literals must round-trip every row and
  reduce the UTF-8 cost proxy by at least 15 percent, otherwise all retained
  records remain literal;
- retained-record rendering and fact-coverage results are cached for each
  selected-index set while fitting the shared stub budget.

These gates never discard critical records. The complete Artifact remains the
authoritative lossless source, and no extra LLM call is introduced.

Answers should use the stub and existing assistant analysis first. If an exact
fact is absent, `search_tool_result_artifacts` scans with the configured 16 KiB
locator chunks but returns only an exact log line, JSON object/member, CSV/TSV
row, code function, or short paragraph plus small context. Results include
exact UTF-8 offsets, Artifact/content hashes, match offsets, and fragment type.
Only when that precise fragment is insufficient should a caller continue with
`read_tool_result_artifact`; missing identifiers, numbers, statuses, and errors
must never be guessed.

Recall-tool results have separate model-facing and memory-facing lifecycles:

- verified `search_tool_result_artifacts` and bounded
  `read_tool_result_artifact` fragments remain verbatim in the current model
  request, but persistent governance projects them back to their original
  Artifact stubs without creating nested Artifacts;
- an attested `list_tool_result_artifacts` metadata result remains visible
  during the current Turn, then its uniquely paired assistant tool call and
  tool result are removed after a valid terminal assistant response, before
  Turn Ledger registration, SessionBuffer storage, extraction, or Archive;
- missing or ambiguous call IDs, invalid HMAC/owner binding, tampering, or a
  non-terminal checkpoint fail open and retain the original messages.

Memory extraction does not automatically search or read Artifact contents.
Phase 2 keeps the bounded multiresolution Stub in its focused context and uses
the terminal assistant response as ordinary supporting context. Exact Artifact
search and range reads remain explicit model-facing tools for answering user
questions; verified recall results persist back as their original Stub rather
than being copied into SessionBuffer or Archive.

## First-view externalization

Hosts with a trusted post-tool/pre-model middleware can govern a qualifying
single result before its first model request. The formal OpenClaw adapter calls
`/api/v1/govern_first_view_tool_result` with trusted session, Turn, tool-call,
tool-name, and agent identity. The server derives strong source IDs, commits the
Artifact synchronously, and returns the multiresolution stub without writing a
second SessionBuffer message. The complete `afterTurn` later validates and
persists that same stub, so retries reuse the Artifact instead of duplicating
it. Store, timeout, transport, or incomplete-identity failures retain the raw
tool result.

This path uses `single_result_chars`. Results that are individually smaller
than the threshold keep their original first view.

## User-message lifecycle

OpenClaw 2026.6.10 uses one outer user message as one complete Agent Turn:

1. `assemble` runs once before the agent/tool loop. Qualifying large outputs
   produced later in that Turn are synchronously replaced by the formal
   adapter before their first model request; smaller outputs remain raw.
2. The final `afterTurn` contains the complete
   `user -> assistant tool call -> tool result -> terminal assistant` delta and
   `prePromptMessageCount`.
3. In explicit `semanticTurnMode=user_message`, oGMemory removes the trusted
   pre-prompt prefix, validates the complete ordered Turn, and persists
   `runtime_extensions.tool_result_turn_ledger` before issuing strong identity.
4. A first-view Artifact is reused when its valid stub reaches `afterTurn`.
   Remaining single results at or above `single_result_chars` are persisted as
   strong Artifacts.
5. A later `compose` restores identity only when one complete ordered projection
   matches exactly and uniquely. It can then return the normal Artifact stub.

The Ledger is owner-bound to account, user, agent, and session. It persists a
monotonic `next_turn_sequence`, up to 1,024 completed Turns, ordered message
projection digests, result descriptors, server-issued HMAC `ogt_*` Turn ids,
server-issued HMAC `ogm_*` message ids, and verified Artifact ids. Exact
`afterTurn` retries reuse the same Turn and Artifact ids.

Caller `_ogmem_source` fields are never accepted as Ledger identity. Generic
ingest routes receive fresh server-owned fallback Buffer ids, which remain
`identity_strength=fallback`. Those ids only bind server-generated mixed
`stub + inline` Buffer projections so a later safety scan does not create an
Artifact whose content is another Artifact stub.

Any unknown schema, owner mismatch, invalid allocation high-water, malformed or
duplicate entry, non-canonical call mapping, partial/ambiguous projection, or
overlapping candidate fails open for model-facing replay: the original tool
text is retained. Persistence initially remains fail closed while the adapter
performs its bounded externalization retries. If those retries are exhausted,
the server-issued HMAC fallback token permits only that exact owner-bound
checkpoint to persist its original messages through the default SessionBuffer
and extraction path. The fallback checkpoint is durably excluded from later
governance scans; subsequent checkpoints remain eligible for externalization.

Session deletion removes the persisted state and Artifacts. Ordinary oGMemory
compaction does not blindly clear the Ledger,
because local Buffer archival does not prove that the host transcript has
already removed the old Turn; retention is bounded and incomplete compacted
projections simply fail open.

## TTL cache-delay lifecycle

When `tool_result_optimization.cache_delay.enabled=true`, medium results at or
above `delayed_result_chars` and below `single_result_chars` have two views.
SessionBuffer and Archive receive the verified stub immediately, while the
model-facing transcript keeps the raw result until an idle TTL epoch expires.

The first verified raw compose records `hot_seen_at`. A later request after
`idle_ttl_seconds` advances a monotonic cutoff and rewrites only strong,
owner-bound Artifacts observed before that cutoff. Read/search source fragments
are excluded, so their current model-facing projection remains verbatim;
list-tool navigation is removed by its terminal lifecycle rule. Fallback
Artifacts are never TTL-rewritten.

Turn-Ledger and cache-delay mutations use the fixed per-session lock order:
Turn Ledger first, then cache delay. State-save failure keeps the caller's hot
view and rolls back new cold eligibility.

## Current limits

- `user_message` is the only currently supported semantic Turn lifecycle.
  The configuration field remains available so future host lifecycles can be
  added explicitly without content-based mode guessing.
- The persisted Ledger is trusted server state. HMAC ids are server-issued but
  are not an at-rest signature over SessionState; a future stable-key/key-id
  design could add tamper detection and rotation.
- Locks are process-local. Multi-worker deployments need session affinity or a
  distributed compare-and-swap/lock before sharing one session.
