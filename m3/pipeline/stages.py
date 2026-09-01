"""Stage implementations for the declarative M3 pipeline runner."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from core.models import RequestContext
from m3.config import M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.pipeline.context import PipelineContext, StageResult
from m3.pipeline.spec import StageSpec

if TYPE_CHECKING:
    from service.graph_service import GraphService

logger = logging.getLogger(__name__)

StageFn = Callable[[PipelineContext, StageSpec], StageResult]


# ---------------------------------------------------------------------------
# Smart-resume probe protocol (Layer 2)
# ---------------------------------------------------------------------------
#
# Each pipeline stage may declare an optional probe that decides whether
# the stage's work is already complete given the current graph + artefact
# state. The runner calls the probe BEFORE invoking the stage function; if
# the probe says ``complete=True`` the stage is marked as a no-op skip and
# the expensive work (Whisper, ECAPA, pyannote, multimodal LLM) is
# skipped on retry.
#
# Probes are registered in :data:`DEFAULT_STAGE_PROBES` (parallel to
# :data:`DEFAULT_STAGE_REGISTRY`). Stages without a probe always run.
#
# Per-AUDIO-node / per-clip skip is handled INSIDE each driver (Layer 1).
# The probe is the coarser outer check that short-circuits the entire
# stage when nothing changed; the driver skip is the fine-grained check
# that saves work mid-stage when only some units finished last time.


@dataclass(frozen=True)
class ProbeResult:
    """Result of a stage probe call.

    Attributes:
        complete: True when the runner should skip the stage. ``reason``
            must be set so the pipeline summary stays informative.
        reason: Short human-readable explanation (e.g. ``"all clips
            have completed episodic_memory_generator ExtractedNodes"``).
        count: Optional stat surfaced in the pipeline summary (e.g.
            ``5`` clips already done, ``23`` segments cached). Defaults
            to 0 when not relevant.
    """

    complete: bool
    reason: str = ""
    count: int = 0


ProbeFn = Callable[[PipelineContext, StageSpec, "GraphService"], ProbeResult]


def probe_skip(
    *, reason: str = "stage disabled", count: int = 0
) -> ProbeResult:
    """Helper for the common ``complete=True`` case."""
    return ProbeResult(complete=True, reason=reason, count=count)


def probe_continue(*, reason: str = "no probe matched", count: int = 0) -> ProbeResult:
    """Helper for the common ``complete=False`` case."""
    return ProbeResult(complete=False, reason=reason, count=count)


def _ctx(account_id: str) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3-pipeline",
        trace_id=f"m3-pipeline-{uuid.uuid4().hex}",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_graph(store_kind: str, connection_string: str | None):
    from m3.load_clips import build_graph_service
    graph, _ = build_graph_service(store_kind, connection_string)
    return graph


def create_memory_nodes(
    *,
    episodic_jsonl: Path,
    graph_store: str,
    connection_string: str | None,
    account_id: str = "m3bench",
    memory_topic: str | None = None,
    graph=None,
    ctx: RequestContext | None = None,
) -> dict[str, int]:
    """Create one MemoryNode per episodic-memory record."""
    graph = graph or _build_graph(graph_store, connection_string)
    ctx = ctx or _ctx(account_id)
    created = skipped = missing_support = empty = 0

    for rec in _read_jsonl(episodic_jsonl):
        video_id = rec.get("video_id")
        clip_id = rec.get("clip_id")
        ext_id = rec.get("extracted_node_id")
        description = (rec.get("description") or "").strip()
        if not description:
            empty += 1
            continue
        if not ext_id or graph.get_extracted_node(ext_id, ctx) is None:
            logger.warning(
                "Missing supporting ExtractedNode: video=%s clip=%s ext=%s",
                video_id, clip_id, ext_id,
            )
            missing_support += 1
            continue

        memory_id = hashlib.sha1(
            f"memory/m3/episodic/{video_id}/{clip_id}".encode()
        ).hexdigest()[:32]
        if graph.get_memory_node(memory_id, ctx) is not None:
            skipped += 1
            continue

        content = {
            "video_id": video_id,
            "clip_id": clip_id,
            "description": description,
            "frame_count": rec.get("frame_count"),
            "has_audio": rec.get("has_audio"),
            "source_extracted_node_id": ext_id,
            "video_raw_node_id": rec.get("video_raw_node_id"),
        }
        graph.create_memory_node(
            summary=description[:240],
            content=json.dumps(content, ensure_ascii=False, indent=2),
            supporting_node_ids=[ext_id],
            topic=memory_topic or f"m3:{video_id}",
            tags=["m3", "video", "episodic", str(video_id)],
            node_id=memory_id,
            ctx=ctx,
        )
        created += 1

    return {
        "created_memory_nodes": created,
        "skipped": skipped,
        "missing_support": missing_support,
        "empty_description": empty,
    }


def _ok(stage: StageSpec, *, outputs=None, counts=None, warnings=None) -> StageResult:
    return StageResult(
        stage_id=stage.id,
        stage_type=stage.type,
        ok=True,
        outputs=outputs or {},
        counts=counts or {},
        warnings=warnings or [],
    )


def trim_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    selected = ctx.artifact("selected_video")
    selected.parent.mkdir(parents=True, exist_ok=True)
    seconds = int(stage.params.get("seconds", ctx.seconds) or 0)
    ffmpeg_bin = str(stage.params.get("ffmpeg_bin", ctx.variables.get("ffmpeg_bin", "ffmpeg")))
    log_level = str(stage.params.get("ffmpeg_log_level", ctx.variables.get("ffmpeg_log_level", "error")))
    if seconds > 0:
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", log_level,
            "-i", str(ctx.video), "-t", str(seconds), "-c", "copy", str(selected),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg trim failed for {ctx.video}: rc={proc.returncode} stderr={proc.stderr.strip()}"
            )
    elif selected.resolve() != ctx.video:
        selected.write_bytes(ctx.video.read_bytes())
    return _ok(stage, outputs={"selected_video": str(selected)}, counts={"seconds": seconds})


def write_single_annotation_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    path = ctx.artifact("annotations")
    path.parent.mkdir(parents=True, exist_ok=True)
    annotation = {
        ctx.video_id: {
            "video_path": str(ctx.artifact("selected_video")),
            "qa_list": [],
        }
    }
    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    return _ok(stage, outputs={"annotations": str(path)}, counts={"videos": 1})


def preprocess_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.preprocess import parse_args, run
    argv = [
        "--annotations", str(ctx.artifact("annotations")),
        "--videos-root", str(ctx.work_dir / "videos"),
        "--clips-root", str(ctx.artifact("clips_root")),
        "--clip-seconds", str(stage.params.get("clip_seconds", 30)),
        "--ffmpeg-bin", str(stage.params.get("ffmpeg_bin", ctx.variables.get("ffmpeg_bin", "ffmpeg"))),
        "--ffprobe-bin", str(stage.params.get("ffprobe_bin", ctx.variables.get("ffprobe_bin", "ffprobe"))),
        "--project-root", str(ctx.work_dir),
    ]
    if ctx.variables.get("force"):
        argv.append("--force")
    rc = run(parse_args(argv))
    if rc != 0:
        raise RuntimeError(f"m3.preprocess returned {rc}")
    return _ok(stage, outputs={"clips_root": str(ctx.artifact("clips_root"))})


def load_clips_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.load_clips import parse_args, run
    argv = [
        "--annotations", str(ctx.artifact("annotations")),
        "--clips-root", str(ctx.artifact("clips_root")),
        "--videos-root", str(ctx.work_dir / "videos"),
        "--output-jsonl", str(ctx.artifact("data_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
        "--clip-seconds", str(stage.params.get("clip_seconds", 30)),
        "--ffprobe-bin", str(stage.params.get("ffprobe_bin", ctx.variables.get("ffprobe_bin", "ffprobe"))),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    # Pass the runner-managed graph through so load_clips writes VIDEO
    # RawNodes into the SAME store extract_frames / extract_audio /
    # episodic_memory read from. Without this each stage's parse_args/run
    # builds its own MemoryGraphStore and the graph state is invisible.
    rc = run(parse_args(argv), graph=ctx.graph)
    if rc != 0:
        raise RuntimeError(f"m3.load_clips returned {rc}")
    rows = _read_jsonl(ctx.artifact("data_jsonl"))
    clips = sum(len(r.get("raw_nodes", [])) for r in rows)
    return _ok(stage, outputs={"data_jsonl": str(ctx.artifact("data_jsonl"))}, counts={"videos": len(rows), "clips": clips})


def extract_audio_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Run :mod:`m3.extract_audio` on every VIDEO RawNode.

    Sits between ``load_clips`` (which persists VIDEO ``RawNode``s) and
    ``episodic_memory`` (which walks the graph for AUDIO children via
    REFERENCES edges). With ``--graph-store memory`` the in-memory
    store does not survive across stages, so the AUDIO nodes written
    here will not be visible downstream — run with ``--graph-store sql``
    for cross-stage visibility.
    """
    from m3.extract_audio import parse_args, run
    argv = [
        "--manifest", str(ctx.artifact("data_jsonl")),
        "--audio-root", str(ctx.artifact("audio_root")),
        "--output-jsonl", str(ctx.artifact("audio_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
        "--sample-rate", str(stage.params.get("sample_rate", 16000)),
        "--channels", str(stage.params.get("channels", 1)),
        "--ffmpeg-bin", str(stage.params.get(
            "ffmpeg_bin", ctx.variables.get("ffmpeg_bin", "ffmpeg"),
        )),
        "--ffmpeg-log-level", str(stage.params.get(
            "ffmpeg_log_level", ctx.variables.get("ffmpeg_log_level", "error"),
        )),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    rc = run(parse_args(argv), graph=ctx.graph, op_service=ctx.op_service)
    # 2 = partial failures (some clips failed but others succeeded).
    # Treat it as ok so a single broken clip doesn't abort the pipeline.
    if rc not in (0, 2):
        raise RuntimeError(f"m3.extract_audio returned {rc}")
    rows = _read_jsonl(ctx.artifact("audio_jsonl"))
    new_records = sum(1 for r in rows if not r.get("resumed"))
    return _ok(
        stage,
        outputs={
            "audio_root": str(ctx.artifact("audio_root")),
            "audio_jsonl": str(ctx.artifact("audio_jsonl")),
        },
        counts={"records": len(rows), "new_records": new_records},
    )


def extract_frames_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.extract_frames import parse_args, run
    argv = [
        "--manifest", str(ctx.artifact("data_jsonl")),
        "--frames-root", str(ctx.artifact("frames_root")),
        "--output-jsonl", str(ctx.artifact("frames_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
        "--fps", str(stage.params.get("fps", 1)),
        "--max-frames", str(stage.params.get("max_frames", 8)),
        "--resize", str(stage.params.get("resize", 512)),
        "--jpeg-quality", str(stage.params.get("jpeg_quality", 80)),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("keyframes_only"):
        argv.append("--keyframes-only")
    rc = run(parse_args(argv), graph=ctx.graph, op_service=ctx.op_service)
    if rc != 0:
        raise RuntimeError(f"m3.extract_frames returned {rc}")
    rows = _read_jsonl(ctx.artifact("frames_jsonl"))
    return _ok(stage, outputs={"frames_jsonl": str(ctx.artifact("frames_jsonl"))}, counts={"records": len(rows), "frames": sum(int(r.get("child_count") or 0) for r in rows)})


def detect_faces_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.detect_faces import parse_args, run
    argv = [
        "--frames-manifest", str(ctx.artifact("frames_jsonl")),
        "--output-jsonl", str(ctx.artifact("faces_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
        "--backend", str(stage.params.get("backend", "http")),
        "--min-confidence", str(stage.params.get("min_confidence", 0.5)),
        "--max-faces", str(stage.params.get("max_faces", 20)),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("cv_service_url"):
        argv.extend(["--cv-service-url", str(stage.params["cv_service_url"])])
    if stage.params.get("cv_timeout") is not None:
        argv.extend(["--cv-timeout", str(stage.params["cv_timeout"])])
    if stage.params.get("force"):
        argv.append("--force")
    rc = run(parse_args(argv), graph=ctx.graph, op_service=ctx.op_service)
    if rc != 0:
        raise RuntimeError(f"m3.detect_faces returned {rc}")
    rows = _read_jsonl(ctx.artifact("faces_jsonl"))
    return _ok(stage, outputs={"faces_jsonl": str(ctx.artifact("faces_jsonl"))}, counts={"records": len(rows), "faces": sum(int(r.get("face_count") or 0) for r in rows)})


def episodic_memory_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.episodic_memory import parse_args, run
    argv = [
        "--config", str(ctx.config_path),
        "--video-records-manifest", str(ctx.artifact("data_jsonl")),
        "--frames-manifest", str(ctx.artifact("frames_jsonl")),
        "--output-jsonl", str(ctx.artifact("episodic_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
        "--max-frames", str(stage.params.get("max_frames", 8)),
    ]
    if stage.params.get("llm"):
        argv.extend(["--llm", str(stage.params["llm"])])
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    # Audio handling: when the spec declares an ``audio_jsonl`` artifact
    # (i.e. the extract_audio stage is registered, even if disabled or
    # skipped), forward it to the episodic driver so the multimodal LLM
    # can attach AUDIO children. ``no_audio`` flips to ``True`` only when
    # the caller explicitly opted out — default ``False`` matches the
    # intent of having audio available.
    audio_manifest: Path | None = None
    if "audio_jsonl" in ctx.artifacts:
        candidate = ctx.artifact("audio_jsonl")
        if candidate.exists():
            audio_manifest = candidate
    no_audio = bool(stage.params.get("no_audio", False))
    if audio_manifest is not None and not no_audio:
        argv.extend(["--audio-manifest", str(audio_manifest)])
    elif no_audio:
        argv.append("--no-audio")
    rc = run(parse_args(argv), graph=ctx.graph, op_service=ctx.op_service)
    if rc != 0:
        raise RuntimeError(f"m3.episodic_memory returned {rc}")
    rows = _read_jsonl(ctx.artifact("episodic_jsonl"))
    return _ok(stage, outputs={"episodic_jsonl": str(ctx.artifact("episodic_jsonl"))}, counts={"records": len(rows), "description_chars": sum(len(r.get("description") or "") for r in rows), "audio_attached": 0 if audio_manifest is None or no_audio else 1})


def memory_from_episodic_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    # Pass the runner-managed graph so create_memory_nodes writes
    # MemoryNodes into the SAME store load_clips wrote VIDEO RawNodes
    # into. Without this the in-memory store would be fresh per call
    # and the "episodic MemoryNode exists?" check would always miss.
    counts = create_memory_nodes(
        episodic_jsonl=ctx.artifact("episodic_jsonl"),
        graph_store=ctx.graph_store,
        connection_string=ctx.connection_string,
        account_id=ctx.account_id,
        memory_topic=stage.params.get("memory_topic"),
        graph=ctx.graph,
    )
    return _ok(stage, counts=counts)


def index_episodic_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.index_episodic import parse_args, run
    argv = [
        "--config", str(ctx.config_path),
        "--episodic-jsonl", str(ctx.artifact("episodic_jsonl")),
        "--output-index", str(ctx.artifact("index_jsonl")),
        "--graph-store", ctx.graph_store,
        "--account-id", ctx.account_id,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("embedder"):
        argv.extend(["--embedder", str(stage.params["embedder"])])
    if stage.params.get("embedding_dimension") is not None:
        argv.extend(["--embedding-dimension", str(stage.params["embedding_dimension"])])
    # When voice_asr has produced a transcript manifest, pass it through
    # so the per-clip retrieval text is augmented with the speech
    # content. The artefact path follows the naming convention used
    # in video_episodic_full.yaml.
    voice_asr_path = ctx.artifact("voice_asr", optional=True)
    if voice_asr_path is not None:
        argv.extend(["--asr-manifest", str(voice_asr_path)])
    # transcript_first defaults to True in the index_episodic CLI; opt
    # out via params.transcript_first=false for vision-only smoke runs.
    if stage.params.get("transcript_first") is False:
        argv.append("--no-transcript-first")
    # Pass the runner-managed graph so each index record's embedding is
    # mirrored back into the corresponding ExtractedNode, making
    # GraphService.search_by_embedding(...) usable downstream.
    rc = run(parse_args(argv), graph=ctx.graph)
    if rc != 0:
        raise RuntimeError(f"m3.index_episodic returned {rc}")
    rows = _read_jsonl(ctx.artifact("index_jsonl"))
    return _ok(stage, outputs={"index_jsonl": str(ctx.artifact("index_jsonl"))}, counts={"records": len(rows)})


def qa_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Adapter-driven QA: drives OGMemoryControlAdapter per question.

    End-to-end QA pipeline stage — wraps ``m3.qa.run_react_qa`` (the
    adapter-based driver). Reads the annotations file from the
    ``annotations`` artifact (the auto-generated ``one_video.json``
    written by ``write_single_annotation``) UNLESS the stage param
    ``annotations`` overrides it — call sites that have a real QA
    list (e.g. M3-Bench's ``robot.json``) should pass it via
    ``--set qa.params.annotations=...``.

    Position: must run after ``build_index`` so the EpisractedNode
    embeddings are populated in the graph (otherwise
    ``search_by_embedding`` returns no hits and the adapter loops
    without producing an answer).
    """
    from m3.qa import run_react_qa, parse_react_qa_args

    annotations = stage.params.get("annotations") or str(ctx.artifact("annotations"))
    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "m3_qa_results.jsonl"
    )

    argv = [
        "--annotations", annotations,
        "--output-jsonl", output_jsonl,
        "--graph-store", ctx.graph_store,
        "--config", str(ctx.config_path),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("embedder"):
        argv.extend(["--embedder", str(stage.params["embedder"])])
    if stage.params.get("embedding_dimension") is not None:
        argv.extend(["--embedding-dimension", str(stage.params["embedding_dimension"])])
    if stage.params.get("llm"):
        argv.extend(["--llm", str(stage.params["llm"])])
    if stage.params.get("temperature") is not None:
        argv.extend(["--temperature", str(stage.params["temperature"])])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("limit") is not None:
        argv.extend(["--limit", str(stage.params["limit"])])
    if stage.params.get("max_iterations") is not None:
        argv.extend(["--max-iterations", str(stage.params["max_iterations"])])
    if stage.params.get("default_video_id"):
        argv.extend(["--default-video-id", str(stage.params["default_video_id"])])
    if stage.params.get("top_k") is not None:
        argv.extend(["--top-k", str(stage.params["top_k"])])
    if stage.params.get("include_neighbors") is not None:
        argv.extend(["--include-neighbors", str(stage.params["include_neighbors"])])
    if stage.params.get("ignore_before_clip"):
        argv.append("--ignore-before-clip")
    if stage.params.get("resume"):
        argv.append("--resume")
    if stage.params.get("verbose"):
        argv.append("--verbose")

    # Pass the runner-managed graph so the adapter runs against the
    # same store that build_index wrote embeddings into.
    rc = run_react_qa(parse_react_qa_args(argv), graph=ctx.graph)
    # rc=2 means partial failures but the run still produced records;
    # we surface it as a non-fatal warning so a single broken question
    # doesn't abort the whole pipeline.
    rows = _read_jsonl(Path(output_jsonl)) if Path(output_jsonl).exists() else []
    answered = sum(1 for r in rows if not r.get("error"))
    failed = sum(1 for r in rows if r.get("error"))
    # Defense in depth: catch rows whose ``predicted_answer`` still
    # contains [Search]/[Answer] protocol scaffolding, even if the
    # driver didn't mark them with ``error``. The driver in
    # ``m3.qa.run_react_qa`` already detects this on writes, but a
    # legacy JSONL from a pre-hardening run could carry such rows
    # and make the summary look green when it isn't.
    from m3.qa import _has_protocol_scaffolding
    invalid = sum(
        1 for r in rows
        if not r.get("error")
        and _has_protocol_scaffolding(str(r.get("predicted_answer") or ""))
    )
    skipped = 0  # not surfaced separately; resume-mode knows
    warnings: list[str] = []
    if rc not in (0, 2):
        raise RuntimeError(f"m3.qa.run_react_qa returned {rc}")
    if rc == 2:
        warnings.append(
            f"qa stage ran with partial failures: {failed}/{len(rows)} "
            f"questions failed"
        )
    if invalid:
        warnings.append(
            f"qa stage detected {invalid}/{len(rows)} predictions with "
            f"[Search]/[Answer] protocol scaffolding; these are not "
            f"real answers and are excluded from the answered count"
        )
    return _ok(
        stage,
        outputs={"qa_results": output_jsonl},
        counts={
            "records": len(rows),
            # Exclude scaffolding-flagged rows from ``answered`` so the
            # summary doesn't show false green: a stage with 100 rows,
            # 30 failed, 5 scaffolding, 65 real answers should report
            # ``answered=65`` not ``answered=70``.
            "answered": answered - invalid,
            "failed": failed,
            "invalid": invalid,
            "skipped": skipped,
        },
        warnings=warnings,
    )


def eval_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """End-to-end M3-Bench evaluation: drives ``m3.eval``.

    Wraps ``m3.eval.run`` so a declarative pipeline can include the
    judge pass as a real stage. The graph is reused from the runner
    context (``ctx.graph``) so eval runs against the same store that
    ``build_index`` populated — no separate graph rebuild. By default
    uses the same ``annotations`` artifact as the ``qa`` stage (which
    is the bench-scale annotations file in production, or the
    auto-generated ``one_video.json`` for the smoke fixture); override
    via ``--set eval.params.annotations=...``.

    Typical params (all optional, all forward to ``m3.eval``'s CLI):

    * ``annotations``: path to M3-Bench JSON (default: ctx artifact)
    * ``output_dir``: where to write ``m3_eval_results.jsonl`` +
      ``m3_eval_summary.json`` (default: ``{work_dir}/m3_eval``)
    * ``embedder``, ``embedding_dimension``: forwarded to embedder
    * ``llm``, ``temperature``: forwarded to the answer LLM
    * ``judge_base_url``, ``judge_model``, ``judge_api_key``: forwarded
      to the judge LLM. Pick any OpenAI-compatible endpoint (DeepSeek,
      vLLM, …). Defaults to MiniMax-M3 when none are set.
    * ``input_qa_results``: path to a pre-existing predictions JSONL
      (typically ``{work_dir}/m3_qa_results.jsonl``). When set, the
      stage runs in judge-only mode and skips the adapter entirely —
      useful for rejudging an earlier QA pass with a different judge
      model without paying for a second adapter run.
    * ``max_iterations``, ``timeout_seconds``: adapter driver config
    * ``video``, ``type_filter``, ``limit``: question filtering
    * ``resume``: forward ``--resume`` for resumable eval
    * ``default_video_id``: forwarded to the adapter

    Returns ``_ok`` with ``summary_json`` and ``results_jsonl``
    outputs pointing into ``output_dir``, plus a ``counts`` dict that
    surfaces ``total_questions``, ``answered``, ``failed``,
    ``judge_error_count``, and ``mean_score`` straight from the
    summary JSON so the pipeline_summary.json viewer can show
    bench-scale accuracy without opening the eval subdir.
    """
    from m3.eval import parse_args as parse_eval_args
    from m3.eval import run as run_eval

    annotations = stage.params.get("annotations") or str(ctx.artifact("annotations"))
    output_dir = Path(
        stage.params.get("output_dir") or (ctx.work_dir / "m3_eval")
    )

    argv: list[str] = [
        "--annotations", annotations,
        "--output-dir", str(output_dir),
        "--graph-store", ctx.graph_store,
        "--config", str(ctx.config_path),
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    # Embedder
    if stage.params.get("embedder"):
        argv.extend(["--embedder", str(stage.params["embedder"])])
    if stage.params.get("embedding_dimension") is not None:
        argv.extend(["--embedding-dimension", str(stage.params["embedding_dimension"])])
    # Answer LLM
    if stage.params.get("llm"):
        argv.extend(["--llm", str(stage.params["llm"])])
    if stage.params.get("temperature") is not None:
        argv.extend(["--temperature", str(stage.params["temperature"])])
    # Judge LLM — any OpenAI-compatible (base_url, model, api_key)
    if stage.params.get("judge_base_url"):
        argv.extend(["--judge-base-url", str(stage.params["judge_base_url"])])
    if stage.params.get("judge_model"):
        argv.extend(["--judge-model", str(stage.params["judge_model"])])
    if stage.params.get("judge_api_key"):
        argv.extend(["--judge-api-key", str(stage.params["judge_api_key"])])
    # Judge-only mode: rejudge an existing predictions JSONL without
    # running the adapter. Useful when you want to swap judge models
    # after a bench-scale QA pass has already finished.
    if stage.params.get("input_qa_results"):
        argv.extend(["--input-qa-results", str(stage.params["input_qa_results"])])
    # Adapter driver
    if stage.params.get("max_iterations") is not None:
        argv.extend(["--max-iterations", str(stage.params["max_iterations"])])
    if stage.params.get("timeout_seconds") is not None:
        argv.extend(["--timeout-seconds", str(stage.params["timeout_seconds"])])
    if stage.params.get("default_video_id"):
        argv.extend(["--default-video-id", str(stage.params["default_video_id"])])
    # Filtering
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("type_filter"):
        argv.extend(["--type-filter", str(stage.params["type_filter"])])
    if stage.params.get("limit") is not None:
        argv.extend(["--limit", str(stage.params["limit"])])
    if stage.params.get("resume"):
        argv.append("--resume")

    eval_args = parse_eval_args(argv)
    rc = run_eval(eval_args)

    summary_json = output_dir / "m3_eval_summary.json"
    results_jsonl = output_dir / "m3_eval_results.jsonl"

    counts: dict[str, Any] = {"records": 0}
    warnings: list[str] = []
    if summary_json.exists():
        try:
            summary = json.loads(summary_json.read_text())
        except json.JSONDecodeError:
            warnings.append(f"could not parse eval summary at {summary_json}")
            summary = {}
        counts = {
            "total_questions": summary.get("total_questions", 0),
            "answered": summary.get("answered", 0),
            "failed": summary.get("failed", 0),
            "judge_error_count": summary.get("judge_error_count", 0),
            "mean_score": summary.get("overall_mean_score", 0.0),
            "missing_from_input": summary.get("missing_from_input", 0),
        }
    if results_jsonl.exists():
        rows = _read_jsonl(results_jsonl)
        counts["records"] = len(rows)

    if rc not in (0, 2):
        raise RuntimeError(f"m3.eval.run returned {rc}")
    if rc == 2:
        warnings.append(
            f"eval ran with partial failures: "
            f"{counts.get('failed', 0)}/{counts.get('total_questions', 0)} "
            f"questions failed"
        )

    return _ok(
        stage,
        outputs={
            "eval_summary": str(summary_json),
            "eval_results": str(results_jsonl),
        },
        counts=counts,
        warnings=warnings,
    )


def face_clustering_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Cross-clip face identity clustering.

    Walks the graph for every face ExtractedNode with an embedding
    and emits ``same_identity`` edges between cross-clip matches.
    Must run AFTER ``detect_faces`` (the source of the embeddings)
    and AFTER ``build_index`` (so the EpisrictedNode embeddings are
    already in the graph — we reuse the same store). Algorithm
    details in ``m3/face_clustering.py``.
    """
    from m3.face_clustering import run_face_clustering, parse_args as parse_fc_args

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "face_clusters.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("threshold") is not None:
        argv.extend(["--threshold", str(stage.params["threshold"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    rc = run_face_clustering(parse_fc_args(argv), graph=ctx.graph)
    if rc != 0:
        raise RuntimeError(f"m3.face_clustering returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    cluster_count = len({r["anchor_node_id"] for r in rows if r.get("anchor_node_id")})
    return _ok(
        stage,
        outputs={"face_clusters": output_jsonl},
        counts={"faces": len(rows), "clusters": cluster_count},
    )


def voice_segmentation_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Voice activity detection on every AUDIO RawNode.

    Wraps ``extraction.operators.builtin.voice_segmentation.VoiceSegmentationExtractor``
    via the registry. Walks the graph for every AUDIO node with a
    ``source_path`` and runs the operator through OperatorService —
    each call fans out to one VOICE_SEGMENTATION_MATERIALIZATION
    child per detected voice run.

    Position: must run after ``extract_audio`` so AUDIO RawNodes
    with ``source_path`` exist.
    """
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from service.operator_service import OperatorService

    from m3.voice_segmentation import (
        run_voice_segmentation as run_vs,
        parse_args as parse_vs_args,
    )

    # Build a per-stage registry (re-registered defaults; harmless
    # duplicate). The runner-managed graph + op_service would be
    # ideal but the driver pattern doesn't accept an op_service
    # injection yet, so we build one here that targets the shared
    # graph.
    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(ctx.graph, registry=registry)

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "voice_segments.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("energy_threshold_dbfs") is not None:
        argv.extend(["--energy-threshold-dbfs", str(stage.params["energy_threshold_dbfs"])])
    if stage.params.get("frame_sec") is not None:
        argv.extend(["--frame-sec", str(stage.params["frame_sec"])])
    if stage.params.get("min_segment_sec") is not None:
        argv.extend(["--min-segment-sec", str(stage.params["min_segment_sec"])])
    if stage.params.get("max_segment_sec") is not None:
        argv.extend(["--max-segment-sec", str(stage.params["max_segment_sec"])])
    if stage.params.get("pad_sec") is not None:
        argv.extend(["--pad-sec", str(stage.params["pad_sec"])])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    if stage.params.get("force"):
        argv.append("--force")

    rc = run_vs(parse_vs_args(argv), graph=ctx.graph, op_service=op_service)
    if rc not in (0, 2):
        raise RuntimeError(f"m3.voice_segmentation returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return _ok(
        stage,
        outputs={"voice_segments": output_jsonl},
        counts={"segments": len(rows)},
        warnings=[] if rc == 0 else [
            "voice_segmentation ran with partial failures",
        ],
    )


def voice_asr_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Run :mod:`m3.voice_asr` on every AUDIO RawNode.

    Thin wrapper around the ``m3.voice_asr`` driver CLI. Builds
    argv from ``stage.params`` and forwards the runner-managed
    graph. Must run after ``extract_audio`` so AUDIO RawNodes
    with ``source_path`` exist.

    Default backend is ``faster_whisper`` (CPU). For NPU boxes
    pass ``params.backend=openai_http`` and a
    ``params.asr_base_url`` pointing at a Whisper server.
    """
    from m3.voice_asr import (
        parse_args as parse_va_args, run_voice_asr as run_va,
    )

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "voice_asr.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("backend"):
        argv.extend(["--backend", str(stage.params["backend"])])
    if stage.params.get("model"):
        argv.extend(["--model", str(stage.params["model"])])
    if stage.params.get("language"):
        argv.extend(["--language", str(stage.params["language"])])
    if stage.params.get("beam_size") is not None:
        argv.extend(["--beam-size", str(stage.params["beam_size"])])
    if stage.params.get("no_word_timestamps"):
        argv.append("--no-word-timestamps")
    if stage.params.get("no_vad_filter"):
        argv.append("--no-vad-filter")
    if stage.params.get("asr_base_url"):
        argv.extend(["--base-url", str(stage.params["asr_base_url"])])
    if stage.params.get("asr_api_key"):
        argv.extend(["--api-key", str(stage.params["asr_api_key"])])
    if stage.params.get("device"):
        argv.extend(["--device", str(stage.params["device"])])
    if stage.params.get("timeout_sec") is not None:
        argv.extend(["--timeout-sec", str(stage.params["timeout_sec"])])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    # Pass --force through so a ``--set voice_asr.params.force=true``
    # override (or a runner-level probe override) can disable the
    # per-AUDIO-node skip and re-transcribe from scratch.
    if stage.params.get("force"):
        argv.append("--force")

    rc = run_va(parse_va_args(argv), graph=ctx.graph)
    if rc not in (0, 2):
        raise RuntimeError(f"m3.voice_asr returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    # Hard gate: when an ASR backend is configured (not mock) and the
    # driver produced zero segments, that's a loud error — typically
    # caused by the silent-drop bug (list_raw_nodes returning 0 AUDIO
    # children) or by missing ``extract_audio``. Surface it as an
    # error-stage result so the pipeline summary flags it and the
    # downstream QA stage can short-circuit on ``has_audio=False``.
    # The driver exit code (rc=0 even on 0-row output, since the driver
    # doesn't fail itself) is not enough to detect this — the file size
    # is the truth.
    backend = (stage.params.get("backend") or "faster_whisper").lower()
    segment_count = len(rows)
    if segment_count == 0 and backend != "mock":
            raise RuntimeError(
                f"voice_asr produced 0 segments (backend={backend!r}). "
                "This usually means list_raw_nodes returned 0 AUDIO "
                "children — verify extract_audio ran and AUDIO RawNodes "
                "are in the graph. Pass --set voice_asr.params.backend=mock "
                "to bypass this gate for smoke runs."
            )
    return _ok(
        stage,
        outputs={"voice_asr": output_jsonl},
        counts={"segments": segment_count},
        warnings=[] if rc == 0 else ["voice_asr ran with partial failures"],
    )


def speaker_diarization_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Run :mod:`m3.speaker_diarization` on every AUDIO RawNode.

    Reads ``voice_asr`` ExtractedNodes from the graph so it can
    stamp ``speaker_id`` onto each ASR word. Must run after
    ``voice_asr`` for that alignment to fire; falls back to
    time-based turns when ASR is missing.
    """
    from m3.speaker_diarization import (
        parse_args as parse_sd_args, run_speaker_diarization as run_sd,
    )

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "speaker_turns.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("backend"):
        argv.extend(["--backend", str(stage.params["backend"])])
    if stage.params.get("model"):
        argv.extend(["--model", str(stage.params["model"])])
    if stage.params.get("hf_token"):
        argv.extend(["--hf-token", str(stage.params["hf_token"])])
    elif "HUGGINGFACE_TOKEN" in os.environ:
        argv.extend(["--hf-token", os.environ["HUGGINGFACE_TOKEN"]])
    if stage.params.get("min_speakers") is not None:
        argv.extend(["--min-speakers", str(stage.params["min_speakers"])])
    if stage.params.get("max_speakers") is not None:
        argv.extend(["--max-speakers", str(stage.params["max_speakers"])])
    if stage.params.get("device"):
        argv.extend(["--device", str(stage.params["device"])])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    if stage.params.get("force"):
        argv.append("--force")

    rc = run_sd(parse_sd_args(argv), graph=ctx.graph)
    if rc not in (0, 2):
        raise RuntimeError(f"m3.speaker_diarization returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return _ok(
        stage,
        outputs={"speaker_turns": output_jsonl},
        counts={"turns": len(rows)},
        warnings=[] if rc == 0 else [
            "speaker_diarization ran with partial failures",
        ],
    )


def speaker_embedding_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Run :mod:`m3.speaker_embedding` on every AUDIO RawNode.

    Reads ``speaker_diarization`` ExtractedNodes from the graph
    so it knows which ``speaker_id``s to embed. Must run after
    ``speaker_diarization``.
    """
    from m3.speaker_embedding import (
        parse_args as parse_se_args, run_speaker_embedding as run_se,
    )

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "speaker_embeddings.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("backend"):
        argv.extend(["--backend", str(stage.params["backend"])])
    if stage.params.get("model"):
        argv.extend(["--model", str(stage.params["model"])])
    if stage.params.get("device"):
        argv.extend(["--device", str(stage.params["device"])])
    if stage.params.get("embedding_dim") is not None:
        argv.extend(["--embedding-dim", str(stage.params["embedding_dim"])])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    if stage.params.get("force"):
        argv.append("--force")

    rc = run_se(parse_se_args(argv), graph=ctx.graph)
    if rc not in (0, 2):
        raise RuntimeError(f"m3.speaker_embedding returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return _ok(
        stage,
        outputs={"speaker_embeddings": output_jsonl},
        counts={"speakers": len(rows)},
        warnings=[] if rc == 0 else [
            "speaker_embedding ran with partial failures",
        ],
    )


def voice_clustering_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Cross-clip voice identity clustering.

    Mirrors ``face_clustering_stage`` for voice segments. Walks the
    graph for every ``voice_segmentation`` ExtractedNode with a
    ``voice_embedding``, runs single-linkage cosine clustering, and
    emits ``same_identity`` edges between cluster anchor and members.

    Must run AFTER ``voice_segmentation`` (source of embeddings) and
    AFTER ``build_index`` (so the EpisrictedNode embeddings are
    already in the graph — same store, same instance).
    """
    from m3.voice_clustering import run_voice_clustering, parse_args as parse_vc_args

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "voice_clusters.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("threshold") is not None:
        argv.extend(["--threshold", str(stage.params["threshold"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")
    rc = run_voice_clustering(parse_vc_args(argv), graph=ctx.graph)
    if rc != 0:
        raise RuntimeError(f"m3.voice_clustering returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    cluster_count = len({r["anchor_node_id"] for r in rows if r.get("anchor_node_id")})
    return _ok(
        stage,
        outputs={"voice_clusters": output_jsonl},
        counts={"voices": len(rows), "clusters": cluster_count},
    )


def semantic_memory_consolidator_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    """Per-video consolidation of episodic_memory_generator nodes.

    For each video in the graph, collects every EpisrictedNode and
    consolidates them into a single MemoryNode (merged description,
    union of characters / events / entities). Each source EpisrictedNode
    gets an EVOLVES edge pointing at the consolidated MemoryNode.

    Must run AFTER ``episodic_memory`` (the source of nodes to
    consolidate) and AFTER ``build_index`` (so the EpisrictedNode
    embeddings are in the graph — the [Search] tool reads from the
    graph, and the consolidated MemoryNode needs to be discoverable
    alongside its sources).
    """
    from m3.semantic_memory_consolidator import (
        run_semantic_memory_consolidator,
        parse_args as parse_smc_args,
    )

    output_jsonl = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "semantic_memories.jsonl"
    )
    argv = [
        "--graph-store", ctx.graph_store,
        "--output-jsonl", output_jsonl,
    ]
    if ctx.connection_string:
        argv.extend(["--connection-string", ctx.connection_string])
    if stage.params.get("video"):
        argv.extend(["--video", str(stage.params["video"])])
    if stage.params.get("verbose"):
        argv.append("--verbose")

    rc = run_semantic_memory_consolidator(parse_smc_args(argv), graph=ctx.graph)
    if rc != 0:
        raise RuntimeError(f"m3.semantic_memory_consolidator returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    memory_count = len(rows)
    edge_count = sum(r.get("evolves_edge_count", 0) for r in rows)
    return _ok(
        stage,
        outputs={"semantic_memories": output_jsonl},
        counts={"memories": memory_count, "evolves_edges": edge_count},
    )


DEFAULT_STAGE_REGISTRY: dict[str, StageFn] = {
    "video.trim": trim_stage,
    "m3.write_single_annotation": write_single_annotation_stage,
    "m3.preprocess": preprocess_stage,
    "m3.load_clips": load_clips_stage,
    "m3.extract_audio": extract_audio_stage,
    "m3.extract_frames": extract_frames_stage,
    "m3.detect_faces": detect_faces_stage,
    "m3.episodic_memory": episodic_memory_stage,
    "m3.memory.from_episodic": memory_from_episodic_stage,
    "m3.index_episodic": index_episodic_stage,
    "m3.qa": qa_stage,
    "m3.face_clustering": face_clustering_stage,
    "m3.voice_segmentation": voice_segmentation_stage,
    "m3.voice_clustering": voice_clustering_stage,
    "m3.voice_asr": voice_asr_stage,
    "m3.speaker_diarization": speaker_diarization_stage,
    "m3.speaker_embedding": speaker_embedding_stage,
    "m3.semantic_memory_consolidator": semantic_memory_consolidator_stage,
    "m3.eval": eval_stage,
}


# ---------------------------------------------------------------------------
# Per-stage probes for smart resume (Layer 2)
# ---------------------------------------------------------------------------
#
# Each probe is called by :meth:`PipelineRunner.run` BEFORE the stage
# function. Returning ``ProbeResult(complete=True, ...)`` causes the
# runner to skip the stage (logged as ``ok=True, counts={"skipped":1}``).
# The probe sees the runner-managed graph and the expected artefact
# paths via :class:`PipelineContext`. ``--no-smart-resume`` on the CLI
# disables the whole probe layer (runner-side flag).
#
# The probes below complement the per-source-node Layer-1 skips inside
# each driver. Layer 1 saves work mid-stage; Layer 2 short-circuits the
# whole stage when nothing changed.


def _classify_extracted_node(
    ext: Any,
    *,
    require_embeddings: bool = False,
) -> str | None:
    """Classify a single ExtractedNode's completion status.

    Shared by :func:`_probe_from_extracted_nodes` (probe side) and
    :func:`m3.resume.video_fully_indexed` (legacy graph-aware resume).
    Both need the same "is this node done?" judgement; previously the
    logic was duplicated and could drift if either side added a new
    "done" criterion (e.g. embedding requirement).

    Returns ``None`` when the node counts as completed (with the
    embedding populated if ``require_embeddings`` is set); otherwise a
    short tag the caller can bucket by:
    - ``"missing"`` — ext is ``None``
    - ``"unfinished"`` — extraction_status != ``"completed"``
    - ``"missing_embedding"`` — completed but ``metadata.embedding``
      is empty when ``require_embeddings`` is set
    """
    if ext is None:
        return "missing"
    if ext.extraction_status != "completed":
        return "unfinished"
    if require_embeddings and not (ext.metadata or {}).get("embedding"):
        return "missing_embedding"
    return None


def _probe_from_extracted_nodes(
    ctx: PipelineContext,
    extractor_type: str,
    *,
    require_embeddings: bool = False,
) -> ProbeResult:
    """Common probe: complete when every relevant ExtractedNode is finished.

    Walks the graph for the given ``extractor_type`` and returns
    complete iff every node has ``extraction_status == "completed"``.
    Use for stages whose "done" state is fully captured in the graph
    (``episodic_memory``, ``voice_asr``, ``extract_frames``,
    ``detect_faces``, ``voice_segmentation``, ``speaker_diarization``,
    ``speaker_embedding``).

    ``require_embeddings=True`` additionally checks the ``embedding``
    metadata field is populated (used for ``index_episodic`` /
    ``episodic_memory`` so QA can search). The per-node status
    classification is delegated to
    :func:`_classify_extracted_node` so the legacy
    :func:`m3.resume.video_fully_indexed` helper shares the same
    judgement.
    """
    graph = ctx.graph
    if graph is None:
        return probe_continue(reason="no shared graph")

    request_ctx = _ctx(ctx.account_id)
    try:
        nodes = graph.list_extracted_nodes(extractor_type, request_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe %s: list_extracted_nodes failed: %s", extractor_type, exc)
        return probe_continue(reason=f"probe error: {exc}")

    if not nodes:
        return probe_continue(reason=f"no {extractor_type} ExtractedNodes yet")

    incomplete = [
        n for n in nodes
        if _classify_extracted_node(n, require_embeddings=require_embeddings) is not None
    ]
    if incomplete:
        return probe_continue(
            reason=f"{len(incomplete)}/{len(nodes)} {extractor_type} not completed",
            count=len(nodes) - len(incomplete),
        )
    return probe_skip(
        reason=f"all {len(nodes)} {extractor_type} ExtractedNodes completed",
        count=len(nodes),
    )


def _artefact_exists(path: Path | None, *, min_bytes: int = 1) -> ProbeResult:
    """Common probe: complete when an artefact file exists and is non-empty."""
    if path is None or not path.exists():
        return probe_continue(reason="artefact path missing")
    try:
        size = path.stat().st_size
    except OSError:
        return probe_continue(reason="artefact unreadable")
    if size < min_bytes:
        return probe_continue(reason=f"artefact too small ({size}B)")
    return probe_skip(reason=f"artefact present ({size}B)", count=1)


# Cheap artefact-based probes -------------------------------------------------


def _probe_video_trim(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    return _artefact_exists(ctx.artifact("selected_video"))


def _probe_write_single_annotation(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _artefact_exists(ctx.artifact("annotations"), min_bytes=2)


def _probe_preprocess(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    clips_root = ctx.artifact("clips_root", optional=True)
    if clips_root is None or not clips_root.exists():
        return probe_continue(reason="clips_root missing")
    files = list(clips_root.rglob("*.mp4"))
    if not files:
        return probe_continue(reason="no clip files in clips_root")
    return probe_skip(reason=f"{len(files)} clips on disk", count=len(files))


def _probe_load_clips(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    path = ctx.artifact("data_jsonl", optional=True)
    if path is None or not path.exists():
        return probe_continue(reason="data.jsonl missing")
    return _artefact_exists(path, min_bytes=10)


def _probe_extract_frames(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "video_frame_extractor")


def _probe_extract_audio(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "audio_extractor")


def _probe_detect_faces(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "face_detection")


def _probe_voice_segmentation(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "voice_segmentation")


def _probe_voice_asr(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "voice_asr")


def _probe_speaker_diarization(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "speaker_diarization")


def _probe_speaker_embedding(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    return _probe_from_extracted_nodes(ctx, "speaker_embedding")


def _probe_episodic_memory(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    # ``require_embeddings=False`` because the index_episodic stage is
    # the one that stamps the embedding; the probe only short-circuits
    # episodic_memory when every ExtractedNode is ``completed``.
    return _probe_from_extracted_nodes(ctx, "episodic_memory_generator")


def _probe_index_episodic(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    # Index_episodic only matters if every episodic_memory_generator node
    # has an embedding. If the graph has no episodic nodes the stage is
    # a no-op; otherwise check embeddings are present.
    return _probe_from_extracted_nodes(
        ctx, "episodic_memory_generator", require_embeddings=True,
    )


def _probe_semantic_memory_consolidator(
    ctx: PipelineContext, stage: StageSpec, graph
) -> ProbeResult:
    path = ctx.artifact("semantic_memories", optional=True)
    if path is None or not path.exists():
        return probe_continue(reason="semantic_memories.jsonl missing")
    return _artefact_exists(path, min_bytes=10)


def _probe_qa(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
    # QA stage is already handled by ``m3.qa.run_react_qa --resume``;
    # the probe is conservative and only short-circuits when the
    # output JSONL is present AND non-empty AND every expected
    # question_id is covered. ``expected_qids`` resolution lives in
    # ``m3.qa.run_react_qa`` itself; here we just confirm the artefact
    # exists.
    output = stage.params.get("output_jsonl") or str(
        ctx.work_dir / "m3_qa_results.jsonl"
    )
    p = Path(output)
    if not p.exists():
        return probe_continue(reason="qa_results.jsonl missing")
    return _artefact_exists(p, min_bytes=10)


DEFAULT_STAGE_PROBES: dict[str, ProbeFn] = {
    "video.trim": _probe_video_trim,
    "m3.write_single_annotation": _probe_write_single_annotation,
    "m3.preprocess": _probe_preprocess,
    "m3.load_clips": _probe_load_clips,
    "m3.extract_audio": _probe_extract_audio,
    "m3.extract_frames": _probe_extract_frames,
    "m3.detect_faces": _probe_detect_faces,
    "m3.episodic_memory": _probe_episodic_memory,
    "m3.index_episodic": _probe_index_episodic,
    "m3.qa": _probe_qa,
    "m3.voice_segmentation": _probe_voice_segmentation,
    "m3.voice_asr": _probe_voice_asr,
    "m3.speaker_diarization": _probe_speaker_diarization,
    "m3.speaker_embedding": _probe_speaker_embedding,
    "m3.semantic_memory_consolidator": _probe_semantic_memory_consolidator,
}


def _safe_probe(probe: ProbeFn | None) -> ProbeFn | None:
    """Return a wrapper that turns probe exceptions into ``ProbeResult(complete=False)``."""
    if probe is None:
        return None

    def _wrapped(ctx: PipelineContext, stage: StageSpec, graph) -> ProbeResult:
        try:
            return probe(ctx, stage, graph)
        except Exception as exc:  # noqa: BLE001 — probe failures are best-effort
            logger.warning(
                "probe for stage %s raised %s — running stage anyway",
                stage.id, exc,
            )
            return probe_continue(reason=f"probe error: {exc}")

    return _wrapped
