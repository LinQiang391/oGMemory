# ContextEngine Lifecycle Perf Report

- **run_id(s):** smoke-inproc
- **sessions:** (none recorded)
- **total events:** 6  (stages=2, sub-spans=4)

## 1. Stage Totals

| stage | calls | total ms | P50 | P95 | P99 | LLM in | LLM out | embed tok | $ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after_turn | 2 | 66.5 | 32.5 | 34.0 | 34.0 | 0 | 0 | 0 | $0.000000 |

**Total $ cost across all stages:** $0.000000

## 2. Sub-span Breakdown

### after_turn

| span | calls | total ms | P50 | P95 | LLM calls | embed calls | $ cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| extract_llm | 2 | 43.1 | 21.0 | 22.1 | 0 | 0 | $0.000000 |
| write_db | 2 | 22.4 | 11.2 | 11.2 | 0 | 0 | $0.000000 |

## 3. Model Breakdown

_(no token usage recorded)_

## 4. Bottleneck Ranking (Top Sub-spans by Wall Time)

| rank | stage | span | total ms | calls |
|---:|---|---|---:|---:|
| 1 | after_turn | extract_llm | 43.1 | 2 |
| 2 | after_turn | write_db | 22.4 | 2 |

## 5. Counters (meta) Appendix

| counter | total | avg per event |
|---|---:|---:|
| num_messages | 10 | 5.00 |

