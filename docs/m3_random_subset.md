# Reproducible M3-Bench full-video pilot

Run from the repository checkout on the experiment server. The wrapper samples
10 videos uniformly without replacement from annotated videos with questions,
using seed 42. Candidates are sorted before sampling, so JSON key order does not
affect selection. All questions for each selected video are evaluated.

## Preview, then run

```bash
# Resolve every selected video and pin the cohort; no pipeline, DB, or judge calls.
bash scripts/run_m3_random10.sh --selection-only

cat data/m3_runs_yuanjian/m3bench_random10_seed42/sample_manifest.json

# Optional: inspect the per-video commands without executing them.
bash scripts/run_m3_random10.sh --dry-run

# Run all ten videos sequentially, then aggregate and judge.
bash scripts/run_m3_random10.sh
```

The manifest records the seed, candidate IDs, exclusions, selected IDs, annotation
content hash, and question counts. Missing selected files cause an error before
any video runs; the script never silently substitutes another video.

Repeating the launch command skips videos whose QA output is already complete.
Incomplete videos rebuild evidence with stage force flags and smart resume
disabled. To rerun completed videos too, pass `--force`. This flag does **not**
allow replacing the pinned cohort.

Use a new `RUN_ROOT` when changing the seed, exclusions, annotations, or experiment
settings. A changed sample/annotation manifest is rejected in an existing root.
Runtime settings are not locked by the manifest: changing them while resuming
could mix experimental conditions.

## Server configuration

Defaults follow the successful `bedroom_03` full-video setup:

- Full duration (`--seconds 0`), SQL graph, `video_episodic_full.yaml`.
- No graph-reuse shortcut; refresh ASR, face detection, speaker processing,
  and episodic memory for videos that execute.
- QA: six iterations, 180-second timeout, candidate retrieval 20,
  transcript retrieval 20 with RRF, context 8, and neighbor inclusion.
- Raw audio enabled; judge enabled. Debug logging is off to avoid large
  base64 audio payloads.

Supported environment overrides:

| Variable | Default / purpose |
| --- | --- |
| `SAMPLE_SIZE` | `10` |
| `SEED` | `42` |
| `RUN_ROOT` | `data/m3_runs_yuanjian/m3bench_random${SAMPLE_SIZE}_seed${SEED}` |
| `CONFIG` | `data/ogmem_test_lyj.resolved.yaml` |
| `ANNOTATIONS` | `data/m3/annotations/robot.json` |
| `VIDEOS_ROOT` | `data/m3/videos` |
| `CV_SERVICE_URL` | `http://localhost:8081` |
| `ASR_BASE_URL` | Unset: retain the pipeline ASR backend. Set to your OpenAI-compatible ASR base URL to select `openai_http`. |
| `NO_AUDIO` | `false`; set `true` if the episodic LLM cannot accept raw audio. ASR text remains enabled. |
| `EXCLUDE` | Optional comma-separated annotated video IDs |

For a fresh holdout-style pilot excluding the already-debugged video:

```bash
EXCLUDE=bedroom_03 RUN_ROOT=data/m3_runs_yuanjian/random10_seed42_holdout \
  bash scripts/run_m3_random10.sh --selection-only

# Keep the same environment values when launching.
EXCLUDE=bedroom_03 RUN_ROOT=data/m3_runs_yuanjian/random10_seed42_holdout \
  bash scripts/run_m3_random10.sh
```

The default includes all eligible videos; exclusion is an explicit experiment
choice. Fix the cohort before looking at scores.

This script does not provision NPU services or automatically move models to an
NPU. Configure the LLM, embedding, CV, and optional ASR endpoints on the server
before launching. Without `ASR_BASE_URL`, the full pipeline retains its
`faster_whisper` backend; other local speaker dependencies also need to work on
the target machine.

A fresh output directory does not isolate the SQL graph. Reusing the same
account/database can update shared evidence; use a dedicated test database or
account when isolation is needed. Additional runner flags can be appended, e.g.
`--account-id <experiment-account>`.

## Outputs and interpretation

- `sample_manifest.json`: cohort identity and expected question counts.
- `driver.log`: appended driver log.
- `<video_id>/pipeline.log`: pipeline output for each executed video.
- `_aggregate/run_report.json`: per-video execution status and sample metadata.
- `_aggregate/all_m3_qa_results.jsonl`: combined QA records.
- Judge outputs in `_aggregate/`, including `m3_eval_summary.json`.

Check execution failures and actual answer/judge coverage against the manifest
before interpreting scores. Ten videos need not contain exactly 100 questions.
The aggregate mean is question-weighted; a per-video mean averaged across videos
can be reported separately to give videos equal weight. This is a reproducible
pilot, not a definitive full-benchmark result.
