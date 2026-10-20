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
| `CV_SERVICE_URL` | Optional explicit override; otherwise use `cv_service.base_url` in the selected YAML (fallback `http://127.0.0.1:8081`). |
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

## 模型过载重试与人脸聚类加速

### Episodic memory：有上限的退避

默认对明确的 HTTP 408、429、500、502、503、504、529 错误额外重试最多
3 次。退避从 10 秒开始指数增长，带少量随机抖动，每次最多 60 秒。
这只是 episodic driver 的重试上限；模型 SDK 自己的重试和请求超时仍然
可能增加总耗时。401、400、输出格式错误等不会被当作过载重试。
重试耗尽后仍记录失败并阻止后续阶段，不会把旧记忆当成生成成功。

可以在原来的运行命令末尾追加：

```bash
--set episodic_memory.params.max_retries=3 \
--set episodic_memory.params.retry_base_seconds=10 \
--set episodic_memory.params.retry_max_seconds=60
```

`max_retries=0` 可关闭额外重试。CLI 对应参数为
`--max-retries`、`--retry-base-seconds`、`--retry-max-seconds`。

### 单个视频失败后的定向恢复

保留同一个视频的 manifests、graph/account 和输出目录，在 **episodic 阶段**
使用 `--retry-failed`（pipeline 参数为
`episodic_memory.params.retry_failed=true`）。
它读取同目录的 `episodic_memory_failures.json` 和输出 JSONL 对应的
`.summary.json`，要求 summary 明确记录相同的 `account_id`，且失败列表非空。
不支持直接用没有这些诊断信息的旧实验目录猜测失败片段。

- 上次失败的 clip 强制重新生成，即使数据库保留着旧的 completed 记忆。
- 其他 clip 不强刷，即使同时指定 `force=true`；仍校验现有 transcript
  等缓存证据，证据过期或缺失时会重建。
- 输出包含本轮生成和复用的完整结果，summary 区分 generated/resumed/failed。
  `recovered_retries` 只统计最终成功的调用经历的重试次数，不包含耗尽的重试。
- 失败列表中的 clip 不在当前 manifest/video 范围内时会报错，不能静默漏掉。

**不要直接把 `retry_failed=true` 加到整个随机抽样 wrapper 上作为通用续跑。**
已经成功的视频没有失败列表；wrapper 默认还会强刷部分上游证据，
而完成 QA 的视频可能被外层跳过。定向恢复应针对失败视频的 episodic 阶段，
保留原来的输入和模型配置；恢复成功后再继续下游阶段。
新一轮完整实验不需要开启 `retry_failed`，默认退避已生效。

### 人脸聚类

- 仍对整个 account 的人脸做跨视频聚类，不通过缩小到当前视频来换速度。
- 加载人脸不再受默认 4096 条限制；已有身份关系只查询端点，不逐个加载邻居。
- SQL 批量插入缺失关系，已有关系不再重复更新，也不反复累加 PageRank。
  新关系的 PageRank 按本批次发生变化的 anchor 聚合更新，不再逐条重复扫描。
- 相同 account、阈值及完整人脸输入复用纯聚类计算；输入变化或缓存损坏时重算。
  **缓存命中仍查数据库并补齐缺失关系**，不会把计算缓存当成写库成功凭据。
- pipeline 默认共享缓存目录为 `ctx.work_dir.parent/_cache/face_clustering`，
  可用 `face_clustering.params.cache_dir` 指定。单独运行模块可用 `--cache-dir`。
- `face_clusters.summary.json` 记录加载、聚类计算、关系查询、写入耗时以及
  `cache_hit`、人脸数、新增关系数，用于核对瓶颈是否真的改善。

这两项优化不能保证整轮耗时缩短到某个值：首次聚类仍需计算，模型服务持续
过载时重试反而需要等待，ASR/检测等其他阶段也可能占用主要时间。
建议先验证一个视频的成功/失败计数及阶段耗时，再跑相同 cohort 的三个视频。
不需要为启用这两项功能清空数据库；对照实验若要隔离旧证据，应另用测试
account/数据库，单换输出目录并不隔离图数据。
