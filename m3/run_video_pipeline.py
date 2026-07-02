"""Run a declarative M3 video pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from m3.config import M3BENCH_ACCOUNT_ID
from m3.pipeline import PipelineRunner, load_pipeline_spec
from m3.pipeline.stages import create_memory_nodes

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "config/m3/pipelines/video_episodic.yaml"


def _legacy_set_overrides(args: argparse.Namespace) -> list[str]:
    """Map legacy convenience flags onto declarative stage param overrides."""
    out: list[str] = []
    out.extend([
        f"preprocess.params.clip_seconds={args.clip_seconds}",
        f"load_clips.params.clip_seconds={args.clip_seconds}",
        f"extract_frames.params.fps={args.fps}",
        f"extract_frames.params.max_frames={args.max_frames}",
        f"extract_frames.params.resize={args.resize}",
        f"extract_frames.params.jpeg_quality={args.jpeg_quality}",
        f"extract_frames.params.keyframes_only={str(args.keyframes_only).lower()}",
        f"episodic_memory.params.max_frames={args.episodic_max_frames}",
        f"episodic_memory.params.no_audio={str(args.no_audio).lower()}",
    ])
    if args.llm:
        out.append(f"episodic_memory.params.llm={args.llm}")
    if args.face_backend:
        out.append(f"detect_faces.params.backend={args.face_backend}")
    if args.cv_service_url:
        out.append(f"detect_faces.params.cv_service_url={args.cv_service_url}")
    if args.cv_timeout is not None:
        out.append(f"detect_faces.params.cv_timeout={args.cv_timeout}")
    out.extend([
        f"detect_faces.params.min_confidence={args.face_min_confidence}",
        f"detect_faces.params.max_faces={args.face_max_faces}",
    ])
    if args.memory_topic:
        out.append(f"create_memory_nodes.params.memory_topic={args.memory_topic}")
    if args.embedder:
        out.append(f"build_index.params.embedder={args.embedder}")
    if args.embedding_dimension is not None:
        out.append(f"build_index.params.embedding_dimension={args.embedding_dimension}")
    if args.ffmpeg_bin:
        out.append(f"trim.params.ffmpeg_bin={args.ffmpeg_bin}")
        out.append(f"preprocess.params.ffmpeg_bin={args.ffmpeg_bin}")
    if args.ffprobe_bin:
        out.append(f"preprocess.params.ffprobe_bin={args.ffprobe_bin}")
        out.append(f"load_clips.params.ffprobe_bin={args.ffprobe_bin}")
    if args.ffmpeg_log_level:
        out.append(f"trim.params.ffmpeg_log_level={args.ffmpeg_log_level}")
    return out


def run_pipeline(args: argparse.Namespace) -> dict:
    """Load the requested YAML spec and execute it."""
    video = Path(args.video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")

    video_id = args.video_id or video.stem
    work_dir = Path(args.work_dir).resolve()
    config_path = Path(args.config).resolve()
    spec = load_pipeline_spec(args.pipeline)
    runner = PipelineRunner()

    enable = list(args.enable_stage or [])
    disable = list(args.disable_stage or [])
    if args.detect_faces:
        enable.append("detect_faces")
    if args.build_index:
        enable.append("build_index")
    set_values = _legacy_set_overrides(args) + list(args.set or [])
    runner.apply_overrides(
        spec,
        enable_stages=enable,
        disable_stages=disable,
        set_values=set_values,
    )

    ctx = runner.build_context(
        spec,
        video=video,
        video_id=video_id,
        seconds=args.seconds,
        work_dir=work_dir,
        config_path=config_path,
        graph_store=args.graph_store,
        connection_string=args.connection_string,
        account_id=args.account_id,
        variables={
            "force": args.force,
            "ffmpeg_bin": args.ffmpeg_bin,
            "ffprobe_bin": args.ffprobe_bin,
            "ffmpeg_log_level": args.ffmpeg_log_level,
        },
    )
    return runner.run(spec, ctx)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-video M3 analysis and create MemoryNodes.")
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                   help="YAML pipeline spec (default: %(default)s)")
    p.add_argument("--enable-stage", action="append", default=[])
    p.add_argument("--disable-stage", action="append", default=[])
    p.add_argument("--set", action="append", default=[],
                   help="Override a stage value, e.g. extract_frames.params.max_frames=16")
    p.add_argument("--video", required=True)
    p.add_argument("--video-id", default=None)
    p.add_argument("--seconds", type=int, default=120,
                   help="Analyze only the first N seconds; <=0 means full video (default: %(default)s)")
    p.add_argument("--work-dir", default="data/m3_runs/one_video")
    p.add_argument("--config", default="deploy/ogmemory.yaml")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None)
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)

    # Backward-compatible convenience overrides for the default spec.
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
    p.add_argument("--detect-faces", action="store_true",
                   help="Optional stage: run m3.detect_faces after frame extraction")
    p.add_argument("--face-backend", choices=("http", "mock"), default="http")
    p.add_argument("--cv-service-url", default=None)
    p.add_argument("--cv-timeout", type=float, default=30.0)
    p.add_argument("--face-min-confidence", type=float, default=0.5)
    p.add_argument("--face-max-faces", type=int, default=20)
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
