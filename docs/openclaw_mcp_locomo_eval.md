# OpenClaw MCP LoCoMO evaluation

Configure independent judge credentials, then run the complete evaluation with
the LoCoMO dataset path as its only positional argument:

```text
LOCOMO_JUDGE_API_KEY=...
LOCOMO_JUDGE_BASE_URL=https://example.invalid/v1
LOCOMO_JUDGE_MODEL=judge-model
```

```shell
python tests/e2e/run_openclaw_mcp_locomo.py /path/to/locomo.json
```

The runner validates the dataset, discovers an authenticated oGMemory MCP
endpoint, temporarily configures OpenClaw, ingests every conversation,
runs QA with the shared LoCoMO prompt, judges each answer, and writes
accuracy and timing reports. It starts an isolated temporary OpenClaw gateway;
the user's running gateway and configuration are not modified or restarted. The
temporary gateway copies only the main agent's authentication and model provider state;
the original sessions and memory indexes remain untouched.

Results are written under `tests/e2e/result/openclaw_mcp_locomo_<timestamp>_<nonce>/`.
An explicit `--output-dir` must point to a new or empty directory. Generated files:

- `summary.md` and `summary.json`: accuracy and ingest, QA, and judge durations
- `qa_judged.csv`: question-level answers and labels
- `qa.csv`: merged OpenClaw answers
- `api_logs/`: request-level evaluation logs

Check connectivity without ingesting or evaluating:

```shell
python tests/e2e/run_openclaw_mcp_locomo.py /path/to/locomo.json --preflight-only
```

Optional environment overrides:

- `OGMEMORY_MCP_URL`, `OGMEMORY_MCP_API_KEY`
- `LOCOMO_JUDGE_BASE_URL`, `LOCOMO_JUDGE_API_KEY` (judge stage only)
- `LOCOMO_JUDGE_MODEL`, `LOCOMO_JUDGE_API_MODE` (judge stage only)
- `LOCOMO_OGMEM_CONFIG`, `LOCOMO_MCP_TIMEOUT_SECONDS`
- `LOCOMO_OPENCLAW_CONFIG`, `LOCOMO_OPENCLAW_STATE_DIR`
