# ContextEngine Lifecycle Perf Report

- **run_id(s):** docker-mode-smoke
- **sessions:** perf-smoke-sess, perf-smoke-sess-2
- **total events:** 4  (stages=2, sub-spans=2)

## 1. Stage Totals

| stage | calls | total ms | P50 | P95 | P99 | LLM in | LLM out | embed tok | $ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after_turn | 1 | 34.0 | 34.0 | 34.0 | 34.0 | 0 | 0 | 0 | $0.000000 |
| compose | 1 | 10.8 | 10.8 | 10.8 | 10.8 | 0 | 0 | 0 | $0.000000 |

**Total $ cost across all stages:** $0.000000

## 2. Sub-span Breakdown

### after_turn

| span | calls | total ms | P50 | P95 | LLM calls | embed calls | $ cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| http_post_after_turn | 1 | 33.2 | 33.2 | 33.2 | 0 | 0 | $0.000000 |

### compose

| span | calls | total ms | P50 | P95 | LLM calls | embed calls | $ cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| http_post_compose | 1 | 9.9 | 9.9 | 9.9 | 0 | 0 | $0.000000 |

## 3. Model Breakdown

_(no token usage recorded)_

## 4. Bottleneck Ranking (Top Sub-spans by Wall Time)

| rank | stage | span | total ms | calls |
|---:|---|---|---:|---:|
| 1 | after_turn | http_post_after_turn | 33.2 | 1 |
| 2 | compose | http_post_compose | 9.9 | 1 |

## 5. Counters (meta) Appendix

| counter | total | avg per event |
|---|---:|---:|
| http_status | 400 | 200.00 |
| n_msgs | 2 | 2.00 |

