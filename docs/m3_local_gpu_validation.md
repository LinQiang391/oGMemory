# Local GPU validation (2026-10-22)

## Reproduction requirements

The local deployment lives in the sibling `deploy-llm-on-gpu` checkout.
Keep credentials in ignored configuration files, not this document.
Use separate accounts, vector tables, and output directories for each model.
Do not interpret a successful process exit or a healthy API as benchmark success.

For the validation on this host:

- LLM endpoint: loopback port 8000, OpenAI-compatible.
- Text embeddings: loopback port 8001, `BAAI/bge-m3`, 1024 dimensions,
  client batch size 8.
- Face detection: loopback port 18081, real InsightFace GPU service.
- All of `llm`, `m3.episodic_llm`, `m3.qa_llm`, and `m3.judge_llm`
  must use the local model and matching API key.
- The actual `/v1/models` context limit, application context declaration,
  and output token budget must agree. Compose defaults do not prove what
  an already-running container uses.
- OpenClaw's persisted provider API key can differ from its container
  environment. A 401 from `/v1/responses` can be an upstream model error;
  inspect the response body and gateway logs before changing gateway auth.

The full pipeline uses real faster-whisper, pyannote, and SpeechBrain, not
mock replacements. Export the authorized Hugging Face token before running.
The installed CTranslate2 needs CUDA 12 cuBLAS even when host PyTorch uses
CUDA 13. On this host, the missing dependency was installed into `.venv`
with `uv pip install --python .venv/bin/python nvidia-cublas-cu12`; the
runtime paths are:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export HUGGINGFACE_TOKEN="$HF_TOKEN"
```

Do not overwrite an existing environment or reuse a different model's
completed episodic memories when comparing models.

## Header-only audio regression

A 60-second stream-copy trim of `living_room_06` produced three clips.
The final clip had a 78-byte WAV container with **no sample data**.
Its streaming header advertised an unknown length (`0xffffffff`), so
both `bool(file_bytes)` and `wave.getnframes()` incorrectly suggested
usable audio. Omni returned HTTP 400 on this input.

The episodic operator now reads one actual WAV frame before attaching
audio. Header-only WAVs are omitted with a warning while image and
transcript evidence remains available. Silent but nonempty audio is
still attached. Unsupported/corrupt containers retain the previous
provider-side validation rather than being silently discarded.

Regression tests cover bytes, disk paths, and base64 content, with
ordinary and streaming WAV headers, and both empty and nonempty samples.
`audio_attached`/`has_audio` stay consistent with the evidence actually sent.

## Required tool-call output errors

The local vLLM 0.10.2 service can return HTTP 400 when its output parser
expects a `FunctionDefinition` array but the model generates a single
object or non-JSON tool markup. This is distinct from invalid request
schemas or context overflow.
The OpenAI provider now recognizes that specific response validation error
under `tool_choice=required` as `ToolCallProtocolError`, on both tool APIs.
It does not mark the endpoint as globally unsupported or fabricate tool calls.
The existing extraction layer can then use its schema-validated JSON
fallback only after both native attempts fail the protocol.

Other HTTP 400 errors remain `LLMCallError`. Unit tests cover this boundary
and the actual service error was used for the regression fixture. Check
extraction logs for fallback use; do not describe fallback extraction as
native tool-call success.

## LoCoMo bundled messages and archive completion

Each LoCoMo session is sent as one user message containing multiple dialogue
lines. Without explicit bounds, Omni-7B enumerated nonexistent message
indices until its output was truncated. Span identification now states the
actual message count and valid index range in the prompt, and supplies a
per-request schema with bounded indices and `maxItems`. The shared schema
is not mutated. A live replay changed the first conversation from a
truncated, unbounded span list to one valid `[0, 0]` span.

The same model sometimes returns an object for the session compressor's
`overview` string. The compressor now checks both summary field types
before returning them, using its existing deterministic compression
fallback for invalid types. Original messages remain in the archive.
This prevents a later PostgreSQL `can't adapt type 'dict'` failure.

The background `after_turn ... extract done` marker is emitted only after
snapshot commit succeeds. LoCoMo's existing log-based completion check
therefore cannot proceed on a marker emitted before a failed archive write.
Tests cover both successful and failed commits. Actual memory rows, vector
rows, archive coverage, and outbox state must still be audited separately.

## Account isolation and truthful QA coverage

The voice-clustering CLI previously hardcoded the default M3 account,
and its pipeline wrapper did not forward the experiment account. This
was exposed by both pilots reading the same 72 historical speaker records.
Both diagnostic runs emitted zero new edges in that old account (34
existing links were skipped). The fix forwards `--account-id` end to end;
the 7B pilot now clusters its own 38 ECAPA records into 15 clusters.
Cross-account tests verify that unrelated identities are neither read
into the output nor linked.

The QA validator also rejects unexecuted `Search(...)`/`Answer(...)`,
tool XML, and JSON function objects in the final answer. These are not
successful answers just because they arrived over HTTP 200. The prompt
uses plain function names and explicitly asks for the function-call
interface rather than encouraging literal bracketed protocol tags.
Failed diagnostic results are retained separately from the corrected run.

```bash
.venv/bin/python -m pytest tests/unit/m3 \
  tests/unit/extraction/operators/test_episodic_memory_generator.py -q
.venv/bin/python -m pytest tests/unit/extraction/operators -q
.venv/bin/python -m pytest tests/unit/providers/llm/test_openai_llm.py \
  tests/unit/extraction -q
.venv/bin/python -m pytest tests/unit/session \
  tests/unit/server/test_memory_service_threading.py \
  tests/integration/core/test_provenance_pipeline.py -q
```

## Workload and result validation

The 3B baseline uses the first 60 seconds of `living_room_06`, with only
the question at 00:43. It is a smoke test, not a full-video benchmark.
The complete-video pilot selects `living_room_22` because it is the
shortest locally available annotated video (440.135 seconds, 13 questions),
not because of its score. It uses all 15 clips with four sampled images
per clip and raw audio; four-frame sampling is a resource-limited setting,
not the pipeline's default 16-frame setting.

Example full-video invocation after creating an ignored local YAML:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m m3.run_video_pipeline \
  --config data/local_gpu_7b_20261022.yaml \
  --video data/m3/videos/robot/living_room_22.mp4 \
  --video-id living_room_22 --seconds 0 \
  --work-dir data/m3_runs_yuanjian/local_gpu_7b_20261022 \
  --pipeline config/m3/pipelines/video_episodic_full.yaml \
  --graph-store sql --account-id local-gpu-7b-20261022 \
  --no-smart-resume --max-frames 4 --episodic-max-frames 4 \
  --set voice_asr.params.device=cuda \
  --set speaker_diarization.params.device=cuda \
  --set speaker_embedding.params.device=cuda \
  --set qa.params.annotations=/data/m3_data/annotations/robot.json \
  --set qa.params.timeout_seconds=180 \
  --set qa.params.temperature=0 --set qa.params.top_k=8 \
  --set qa.params.candidate_k=20 --set qa.params.transcript_top_k=20 \
  --set qa.params.transcript_fusion=rrf --set qa.params.context_k=8 \
  --set qa.params.include_neighbors=1
```

GPU 1 also hosts the lightweight embedding service; run these stages
sequentially and check available memory. The local 7B deployment uses
tensor parallelism on GPU 0 and GPU 3, leaving face detection on GPU 2.

After running, check **every** stage's status and failure counts, the
expected clip/question coverage, nonempty predictions, actual retrieval
history, and `judge_error_count`. A judge API failure is not a wrong
answer and its nominal zero score must not be reported as model quality.
Same-model local judging is a diagnostic, not an independent quality
assessment.

Host-specific commands, logs, GPU sampling, model checksums, and the
final validation report are stored under
`deploy-llm-on-gpu/results/local-e2e-20261022/`. M3 artifacts are under
`data/m3_runs_yuanjian/local_gpu_{3b,7b}_20261022/`. LoCoMo uses only
ignored `configs/ogmem-local-*.toml` files and `test_results/`; its tracked
source/configuration files are not modified.

LoCoMo's `small` input has 50 questions, but this runner excludes category
5, so the expected evaluation coverage is **35 questions**, not 50.
The verified test uses a dedicated oGMemory instance on loopback port 4832
and gateway on port 48544. Storage and vector indexing both use the fresh
`ogmem_local_gpu_20261022_verified` database, not the existing application
database. The account is `local-locomo-7b-20261022-v3`; the gateway state
directory is `data/local_locomo_openclaw_20261022_verified`. Failed attempts
remain in separate databases/state directories and are not mixed into the
verified run.

The test agent uses the minimal tool profile; remote automatic memory
capture/recall remains enabled. Extraction uses local Omni-7B with
temperature 0, JSON mode, 2048 output tokens, and a 600-second timeout.
The ignored LoCoMo config is `configs/ogmem-local-7b.toml`, with results in
`test_results/ogmem-local-7b-20261022-verified/`. Its judge override is local
even though the unrelated default `env.toml` still has an external judge.
The sibling deployment's `compose.evaluation.yaml` describes this isolated
stack; its ignored env files contain credentials and must not be published.
