"""Run a one-video M3 analysis pipeline and create graph MemoryNodes.

This is a convenience wrapper for trying the current M3 stack on a local video:

1. Optionally trim the first N seconds from the video.
2. Slice the selected video into fixed-length clips.
3. Load clips as VIDEO RawNodes.
4. Extract sampled frames as IMAGE RawNodes.
5. Generate per-clip episodic memories via the configured multimodal LLM.
6. Create one MemoryNode per episodic clip, supported by that ExtractedNode.
7. Optionally build an episodic retrieval index.

Example::

    python -m m3.run_video_pipeline \
      --video data/m3/videos/robot/bedroom_03.mp4 \
      --seconds 120 \
      --config deploy/ogmemory.yaml \
      --graph-store sql \
      --connection-string "host=127.0.0.1 port=12456 dbname=postgres user=gaussdb password=..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from core.models import RequestContext
from m3.config import (
    M3BENCH_ACCOUNT_ID,
    M3BENCH_AGENT_ID,
    M3BENCH_USER_ID,
    load_m3_runtime_config,
)
from m3.extract_frames import parse_args as parse_extract_frames_args
from m3.extract_frames import run as run_extract_frames
from m3.index_episodic import parse_args as parse_index_args
from m3.index_episodic import run as run_index
from m3.load_clips import parse_args as parse_load_args
from m3.load_clips import run as run_load
from m3.preprocess import parse_args as parse_preprocess_args
from m3.preprocess import run as run_preprocess
from m3.episodic_memory import parse_args as parse_episodic_args
from m3.episodic_memory import run as run_episodic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg_trim(src: Path, dst: Path, *, seconds: int, ffmpeg_bin: str, log_level: str) -> None:
    """Write the first ``seconds`` seconds of ``src`` to ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        log_level,
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-c",
        "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed for {src}: rc={proc.returncode} stderr={proc.stderr.strip()}"
        )


def _write_single_annotation(path: Path, *, video_id: str, video_path: Path) -> None:
    """Write a minimal M3 annotation for one video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotation = {
        video_id: {
            "video_path": str(video_path),
            "qa_list": [],
        }
    }
    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_graph(store_kind: str, connection_string: str | None):
    from m3.load_clips import build_graph_service
    graph, _ = build_graph_service(store_kind, connection_string)
    return graph


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3-video-pipeline",
        trace_id=f"m3-video-pipeline-{uuid.uuid4().hex}",
    )


def create_memory_nodes(
    *,
    episodic_jsonl: Path,
    graph_store: str,
    connection_string: str | None,
    memory_topic: str | None = None,
    graph=None,
    ctx: RequestContext | None = None,
) -> dict:
    """Create one MemoryNode per episodic-memory record.

    Each MemoryNode is supported by the corresponding episodic ExtractedNode.
    Deterministic MemoryNode IDs make the step idempotent.
    """
    graph = graph or _build_graph(graph_store, connection_string)
    ctx = ctx or _ctx()
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict:
    video = Path(args.video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")

    if args.graph_store == "sql" and not args.connection_string:
        args.connection_string = load_m3_runtime_config(args.config).graph_connection_string
        if not args.connection_string:
            raise RuntimeError(
                "--graph-store sql requires --connection-string or vector_db.connection_string in ogmemory.yaml"
            )

    video_id = args.video_id or video.stem
    work_dir = Path(args.work_dir).resolve()
    selected_video = work_dir / "videos" / "robot" / f"{video_id}.mp4"
    annotations = work_dir / "annotations" / "one_video.json"
    clips_root = work_dir / "clips"
    frames_root = work_dir / "frames"
    data_jsonl = work_dir / "data.jsonl"
    frames_jsonl = work_dir / "frames.jsonl"
    episodic_jsonl = work_dir / "episodic_memory.jsonl"
    index_jsonl = work_dir / "episodic_index.jsonl"

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing video %s -> %s", video, selected_video)
    if args.seconds and args.seconds > 0:
        _run_ffmpeg_trim(
            video,
            selected_video,
            seconds=args.seconds,
            ffmpeg_bin=args.ffmpeg_bin,
            log_level=args.ffmpeg_log_level,
        )
    else:
        selected_video.parent.mkdir(parents=True, exist_ok=True)
        if selected_video.resolve() != video:
            selected_video.write_bytes(video.read_bytes())

    _write_single_annotation(
        annotations,
        video_id=video_id,
        video_path=selected_video,
    )

    preprocess_args = parse_preprocess_args([
        "--annotations", str(annotations),
        "--videos-root", str(work_dir / "videos"),
        "--clips-root", str(clips_root),
        "--clip-seconds", str(args.clip_seconds),
        "--ffmpeg-bin", args.ffmpeg_bin,
        "--ffprobe-bin", args.ffprobe_bin,
        "--project-root", str(work_dir),
    ] + (["--force"] if args.force else []))
    rc = run_preprocess(preprocess_args)
    if rc not in (0,):
        raise RuntimeError(f"m3.preprocess returned {rc}")

    load_args = parse_load_args([
        "--annotations", str(annotations),
        "--clips-root", str(clips_root),
        "--videos-root", str(work_dir / "videos"),
        "--output-jsonl", str(data_jsonl),
        "--graph-store", args.graph_store,
        "--account-id", args.account_id,
        "--clip-seconds", str(args.clip_seconds),
        "--ffprobe-bin", args.ffprobe_bin,
    ] + (["--connection-string", args.connection_string] if args.connection_string else []))
    rc = run_load(load_args)
    if rc not in (0,):
        raise RuntimeError(f"m3.load_clips returned {rc}")

    frame_argv = [
        "--manifest", str(data_jsonl),
        "--frames-root", str(frames_root),
        "--output-jsonl", str(frames_jsonl),
        "--graph-store", args.graph_store,
        "--account-id", args.account_id,
        "--fps", str(args.fps),
        "--max-frames", str(args.max_frames),
        "--resize", str(args.resize),
        "--jpeg-quality", str(args.jpeg_quality),
    ]
    if args.connection_string:
        frame_argv.extend(["--connection-string", args.connection_string])
    if args.keyframes_only:
        frame_argv.append("--keyframes-only")
    frame_args = parse_extract_frames_args(frame_argv)
    rc = run_extract_frames(frame_args)
    if rc not in (0,):
        raise RuntimeError(f"m3.extract_frames returned {rc}")

    episodic_argv = [
        "--config", args.config,
        "--video-records-manifest", str(data_jsonl),
        "--frames-manifest", str(frames_jsonl),
        "--output-jsonl", str(episodic_jsonl),
        "--graph-store", args.graph_store,
        "--account-id", args.account_id,
        "--max-frames", str(args.episodic_max_frames),
    ]
    if args.llm:
        episodic_argv.extend(["--llm", args.llm])
    if args.connection_string:
        episodic_argv.extend(["--connection-string", args.connection_string])
    if args.no_audio:
        episodic_argv.append("--no-audio")
    episodic_args = parse_episodic_args(episodic_argv)
    rc = run_episodic(episodic_args)
    if rc not in (0,):
        raise RuntimeError(f"m3.episodic_memory returned {rc}")

    memory_summary = create_memory_nodes(
        episodic_jsonl=episodic_jsonl,
        graph_store=args.graph_store,
        connection_string=args.connection_string,
        memory_topic=args.memory_topic,
    )

    index_summary = None
    if args.build_index:
        index_argv = [
            "--config", args.config,
            "--episodic-jsonl", str(episodic_jsonl),
            "--output-index", str(index_jsonl),
        ]
        if args.embedder:
            index_argv.extend(["--embedder", args.embedder])
        if args.embedding_dimension is not None:
            index_argv.extend(["--embedding-dimension", str(args.embedding_dimension)])
        index_args = parse_index_args(index_argv)
        rc = run_index(index_args)
        if rc not in (0,):
            raise RuntimeError(f"m3.index_episodic returned {rc}")
        index_summary = {"index_jsonl": str(index_jsonl)}

    summary = {
        "video_id": video_id,
        "selected_video": str(selected_video),
        "annotations": str(annotations),
        "data_jsonl": str(data_jsonl),
        "frames_jsonl": str(frames_jsonl),
        "episodic_jsonl": str(episodic_jsonl),
        "memory": memory_summary,
        "index": index_summary,
    }
    summary_path = work_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-video M3 analysis and create MemoryNodes.")
    p.add_argument("--video", required=True)
    p.add_argument("--video-id", default=None)
    p.add_argument("--seconds", type=int, default=120,
                   help="Analyze only the first N seconds; <=0 means full video (default: %(default)s)")
    p.add_argument("--work-dir", default="data/m3_runs/one_video")
    p.add_argument("--config", default="deploy/ogmemory.yaml")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None)
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--llm", choices=("minimax", "openai", "mock"), default=None,
                   help="Override m3.episodic_llm.provider from config")
    p.add_argument("--no-audio", action="store_true", default=True,
                   help="Current pipeline is vision-only by default")
    p.add_argument("--clip-seconds", type=int, default=30)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=8,
                   help="Max frames to extract per clip")
    p.add_argument("--episodic-max-frames", type=int, default=8,
                   help="Max extracted frames to send to the episodic LLM per clip")
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--keyframes-only", action="store_true")
    p.add_argument("--memory-topic", default=None)
    p.add_argument("--build-index", action="store_true")
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Override m3.embedding.provider when --build-index is set")
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("--ffmpeg-log-level", default="error")
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.resize == 0:
        args.resize = None
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        logger.exception("M3 one-video pipeline failed: %s", exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
