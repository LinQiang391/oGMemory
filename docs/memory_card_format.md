# Entity and preference cards

The extraction YAML defines the Markdown layout. Enable prewrite layout validation:

```yaml
memory:
  extraction:
    card_format_enabled: true
```

`OGMEM_CARD_FORMAT_ENABLED=true` is the environment equivalent. The setting applies to the HTTP memory service. It defaults to false for existing deployments.

Both types use a single-line abstract and an overview with one core fact per `- ` bullet. Entity content has one H1 display name, an optional introductory sentence, and named H2 sections with fact bullets. Preference content has one H1 topic and preference bullets. Preserve exact facts and source provenance; the layout check does not establish factual correctness.

With `node_processing_enabled: true`, eager uses one native extraction pass and lazy uses one ReAct run, followed by entity/preference node processing. Otherwise the configured extraction passes and candidate reconciliation apply. Validation runs after candidate selection and before checkpoint acceptance, and again on the built node before storage. A malformed card enters the existing durable source recovery task. The synchronous turn receives `ok=true, status=queued, reason=memory_card_format`; queued acknowledges the durable task, and the source completes when recovery finishes. An asynchronous turn retains its normal `processing` acknowledgement and registers recovery in its worker after a format failure.

Turns without a client request ID receive a fresh identity, persisted with their source batch. Identical text in a later turn has its own identity. Background recovery reuses the saved identity and source, with at most two attempts. The validation error is included in the retry's extraction prompt. Accepted candidates and completed writes retain their existing checkpoint/receipt protection. Inspect completion through `wait_until_idle` and its recovery task status.

Scope: ordinary `after_turn` source extraction. Compaction and dispose retain their lifecycle recovery contract.
