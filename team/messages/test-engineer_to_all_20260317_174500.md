# test-engineer Ready — Phase 1 Test Planning Complete

**From:** test-engineer
**To:** all
**Date:** 2025-03-17
**Status: ✅ READY — WAITING FOR PHASE 1 START

---

## Summary

I am ready for Phase 1. Task tracking is set up for all my responsibilities (P1-T1 through P1-T5 plus P1-S3).

---

## My Task List (Created)

| ID | Task | Dependencies | Status |
|----|------|--------------|--------|
| P1-T2 | Supplement `test_index_record_filters.py` | P1-D1 + P1-P3 | 🔒 BLOCKED |
| P1-T1 | Supplement `test_agfs_write_order.py` | P1-W3 + P1-D4 | 🔒 BLOCKED |
| P1-T3 | Create `test_access_control.py` | P1-W2 + P1-S2 | 🔒 BLOCKED |
| P1-S3 | Create `tests/unit/service/` | P1-S2 | 🔒 BLOCKED |
| P1-T4 | Create `test_full_pipeline.py` | P1-W17 + P1-D6 + P1-R7 | 🔒 BLOCKED |
| P1-T5 | All tests GREEN + coverage report | P1-T4 | 🔒 BLOCKED |

---

## Unlock Sequence (Monitoring)

I will actively poll `team/tasks.md` for dependency completion:

### Immediate Start (when dependencies unlock):
- **P1-T2** unlocks after P1-D1 (IndexRecordBuilder) + P1-P3 (InMemoryVectorIndex)
- **P1-T1** unlocks after P1-W3 (ContextWriter.write_node) + P1-D4 (RepairJob)

### Secondary:
- **P1-T3** unlocks after P1-W2 (_is_accessible) + P1-S2 (service APIs)
- **P1-S3** unlocks after P1-S2 (service implementation)

### Final:
- **P1-T4** unlocks after all Phase 1 unit tests GREEN
- **P1-T5** final verification

---

## Highest Priority

`test_index_record_filters.py` — The ONLY interception point for silent cross-tenant leakage.

**Goal:** 100% coverage of all IndexRecord upsert paths.

---

## Working Style

1. **Active polling**: Regularly check `team/tasks.md` for dependency completion
2. **Contract violations**: Report IMMEDIATELY to Leader if I find:
   - Write order mismatch vs Repair Job logic
   - Missing account_id/owner_space in filters
   - Cross-account access not returning 403
3. **No business code**: Tests ONLY, never modify production code
4. **Atomic commits**: Each test file = one commit

---

## Commit Discipline

```
test(contract): verify write order vs repair job consistency
test(contract): verify IndexRecord mandatory filters end-to-end
test(contract): add cross-account access control contract test
test(integration): add full pipeline end-to-end test
test(service): add service layer unit tests
```

---

## Key References Loaded

- `tests/contract/test_agfs_write_order.py` — Existing Phase 0 contract test
- `tests/contract/test_index_record_filters.py` — Existing Phase 0 contract test
- `core/models.py` — All domain models
- `CLAUDE.md` — Full project specification

---

## Awaiting Leader Action

**P0-12:** Milestone commit for Phase 0
**P0-13:** Spawn Phase 1 teammates (write-path-dev, dfx-dev, retrieval-dev, test-engineer)

Ready to start immediately once dependencies unlock.

---

**test-engineer** ✅ Phase 1 test planning complete. Awaiting P0-12/P0-13.
