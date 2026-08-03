"""Stage implementations for the declarative M3 pipeline runner."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.models import RequestContext
from m3.config import M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.pipeline.context import PipelineContext, StageResult
from m3.pipeline.spec import StageSpec

logger = logging.getLogger(__name__)

StageFn = Callable[[PipelineContext, StageSpec], StageResult]


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
    skipped = 0  # not surfaced separately; resume-mode knows
    warnings: list[str] = []
    if rc not in (0, 2):
        raise RuntimeError(f"m3.qa.run_react_qa returned {rc}")
    if rc == 2:
        warnings.append(
            f"qa stage ran with partial failures: {failed}/{len(rows)} "
            f"questions failed"
        )
    return _ok(
        stage,
        outputs={"qa_results": output_jsonl},
        counts={"records": len(rows), "answered": answered,
                "failed": failed, "skipped": skipped},
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
    from m3.voice_asr import parse_args as parse_va_args, run as run_va

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

    rc = run_va(parse_va_args(argv), graph=ctx.graph)
    if rc not in (0, 2):
        raise RuntimeError(f"m3.voice_asr returned {rc}")
    rows: list[dict] = []
    if Path(output_jsonl).exists():
        with Path(output_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return _ok(
        stage,
        outputs={"voice_asr": output_jsonl},
        counts={"segments": len(rows)},
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
        parse_args as parse_sd_args, run as run_sd,
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
        parse_args as parse_se_args, run as run_se,
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
}
