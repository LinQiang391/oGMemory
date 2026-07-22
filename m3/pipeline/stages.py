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
    rc = run(parse_args(argv))
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
    rc = run(parse_args(argv))
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
    rc = run(parse_args(argv))
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
    rc = run(parse_args(argv))
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
    if stage.params.get("no_audio", True):
        argv.append("--no-audio")
    rc = run(parse_args(argv))
    if rc != 0:
        raise RuntimeError(f"m3.episodic_memory returned {rc}")
    rows = _read_jsonl(ctx.artifact("episodic_jsonl"))
    return _ok(stage, outputs={"episodic_jsonl": str(ctx.artifact("episodic_jsonl"))}, counts={"records": len(rows), "description_chars": sum(len(r.get("description") or "") for r in rows)})


def memory_from_episodic_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    counts = create_memory_nodes(
        episodic_jsonl=ctx.artifact("episodic_jsonl"),
        graph_store=ctx.graph_store,
        connection_string=ctx.connection_string,
        account_id=ctx.account_id,
        memory_topic=stage.params.get("memory_topic"),
    )
    return _ok(stage, counts=counts)


def index_episodic_stage(ctx: PipelineContext, stage: StageSpec) -> StageResult:
    from m3.index_episodic import parse_args, run
    argv = [
        "--config", str(ctx.config_path),
        "--episodic-jsonl", str(ctx.artifact("episodic_jsonl")),
        "--output-index", str(ctx.artifact("index_jsonl")),
    ]
    if stage.params.get("embedder"):
        argv.extend(["--embedder", str(stage.params["embedder"])])
    if stage.params.get("embedding_dimension") is not None:
        argv.extend(["--embedding-dimension", str(stage.params["embedding_dimension"])])
    rc = run(parse_args(argv))
    if rc != 0:
        raise RuntimeError(f"m3.index_episodic returned {rc}")
    rows = _read_jsonl(ctx.artifact("index_jsonl"))
    return _ok(stage, outputs={"index_jsonl": str(ctx.artifact("index_jsonl"))}, counts={"records": len(rows)})


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
}
