# Tool-result cache-delay optimization

This optional extension requires
`tool_result_externalization.enabled=true`. It is disabled by default and never
implicitly enables the main feature.

## Lazy TTL cutover

`cache_delay.enabled` maintains two verified views:

- SessionBuffer, extraction and Archive externalize results at or above
  `delayed_result_chars` immediately, so the durable memory view is always a
  bounded Artifact stub.
- Model-facing replay keeps results below `single_result_chars` raw while the
  session is active, preserving an existing provider KV-cache prefix.
- A stable medium Artifact must first survive into a model-facing `assemble`
  projection as raw content. Its Artifact id and first hot-projection time are
  then recorded in versioned SessionState.
- On the first later `assemble` after `idle_ttl_seconds`, the previous request
  time becomes a monotonic cold cutoff. Only verified medium Artifacts observed
  at or before that cutoff are replaced with stubs.
- A result that becomes cold never returns to raw. Results first observed after
  the cutoff stay hot until a later idle epoch advances the cutoff.

There is no background thread: TTL is evaluated lazily on the next request.
State is stored under
`runtime_extensions.tool_result_cache_delay` with
`last_model_request_at`, `cold_cutover_before`, and `hot_seen_artifacts`.
State load/save failure preserves the raw model view. Artifact persistence
failure retains the main feature's fail-closed durable behavior.

Only stable account/user/agent/session/Turn/message/tool-call/part identities
may be cold-rewritten. Fallback identity, source-aware read/search results, and
list metadata are excluded. In `user_message` mode, Ledger-bound Artifacts use
the Ledger's persisted `artifact_id` and `hot_seen_at` and do not consume the
fallback observation table. That table tracks at most 2,048 active non-Ledger
Artifact ids; cold entries that are no longer referenced by the current
projection are reclaimed before new observations are admitted. If every slot
is still active, overflow remains hot rather than risking a non-monotonic
rewrite.

`cache_hit_price_ratio` is observability and cost-planning metadata only. It is
not a correctness gate. Character limits are a provider-neutral token proxy,
and a cold stub is never substituted when it is not smaller than the raw result.
