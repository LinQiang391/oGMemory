"""Phase 0: Cut M3-Bench videos into fixed-length clips.

Reads the M3-Bench annotation JSON (e.g. ``data/annotations/robot.json``)
and for each video runs ``ffmpeg`` to split it into ``clip_seconds``
second clips under ``clips_root/<video_id>/<clip_id>.mp4``.

Idempotent: videos whose expected clips already exist on disk are
skipped unless ``--force`` is given.

Usage::

    # from m3-agent working dir (where data/ lives):
    python -m m3.preprocess --annotations data/annotations/robot.json

    # custom paths:
    python -m m3.preprocess \\
        --annotations /abs/path/to/robot.json \\
        --videos-root /abs/videos \\
        --clips-root /abs/clips \\
        --clip-seconds 30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from m3.config import PreprocessConfig, resolve_video_path, video_id_from_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def ffprobe_duration(video_path: Path, ffprobe_bin: str) -> float:
    """Return video duration in seconds via ffprobe.

    Raises RuntimeError if ffprobe fails or returns unparsable output.
    """
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: rc={proc.returncode} stderr={proc.stderr.strip()}"
        )
    out = proc.stdout.strip()
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration {out!r} for {video_path}"
        ) from exc


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def expected_clip_count(duration_seconds: float, clip_seconds: int) -> int:
    """Number of clips produced by ffmpeg when cutting at ``clip_seconds``.

    M3-Agent's reference bash uses ``ceil(duration/30)`` segments and
    the last segment is shorter than ``clip_seconds`` if the duration is
    not a multiple. We replicate that behaviour so downstream code can
    match by integer clip_id.
    """
    return max(1, math.ceil(duration_seconds / clip_seconds))


def cut_clip(
    src: Path,
    dst: Path,
    *,
    start_seconds: int,
    clip_seconds: int,
    ffmpeg_bin: str,
    log_level: str,
) -> None:
    """Run ``ffmpeg -ss <start> -i <src> -t <clip_seconds> -c copy <dst>``.

    ``-c copy`` is stream-copy (no re-encode) — fastest path that the
    M3-Agent reference bash script uses. Works because keyframes are
    reasonably close. For videos with very long GOPs, falling back to
    re-encode (``-c:v libx264``) may be needed; not enabled by default.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", log_level,
        "-ss", str(start_seconds),
        "-i", str(src),
        "-t", str(clip_seconds),
        "-c", "copy",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src} -> {dst}: rc={proc.returncode} stderr={proc.stderr.strip()}"
        )


def slice_video(
    video_path: Path,
    clips_dir: Path,
    *,
    clip_seconds: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    log_level: str,
    force: bool = False,
) -> int:
    """Slice one video into ``clips_dir/<video_id>/<clip_id>.mp4``.

    Returns the number of clips written (excluding pre-existing ones
    when ``force`` is False). Pre-existing clips are detected by file
    presence; ffmpeg is not re-invoked for them.
    """
    video_id = video_id_from_path(str(video_path))
    out_dir = clips_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(video_path, ffprobe_bin)
    n_expected = expected_clip_count(duration, clip_seconds)

    written = 0
    for clip_id in range(n_expected):
        out_path = out_dir / f"{clip_id}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            continue
        cut_clip(
            video_path, out_path,
            start_seconds=clip_id * clip_seconds,
            clip_seconds=clip_seconds,
            ffmpeg_bin=ffmpeg_bin,
            log_level=log_level,
        )
        written += 1
    logger.info(
        "Sliced %s (%.1fs, %d clips, %d newly written)",
        video_id, duration, n_expected, written,
    )
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def list_video_paths(
    annotations_file: Path,
    videos_root: Path,
    project_root: Path | None = None,
) -> list[Path]:
    """Resolve video paths declared in the annotation JSON.

    Delegates the per-path resolution to :func:`m3.config.resolve_video_path`
    so the loader script and this slicer always agree. Videos whose
    source cannot be resolved are logged and skipped.
    """
    with annotations_file.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    root = project_root or Path.cwd()
    paths: list[Path] = []
    for _vid, info in annotations.items():
        declared = info["video_path"]
        resolved = resolve_video_path(
            declared, project_root=root, videos_root=videos_root,
        )
        if resolved is None:
            logger.warning("Video file not found, skipping: %s", declared)
            continue
        paths.append(resolved)
    return paths


def run(args: argparse.Namespace) -> int:
    cfg = PreprocessConfig(
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        videos_root=args.videos_root,
        clips_root=args.clips_root,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )
    annotations = Path(args.annotations).resolve()
    videos_root = Path(cfg.videos_root).resolve()
    clips_root = Path(cfg.clips_root).resolve()

    if not annotations.exists():
        logger.error("Annotations file not found: %s", annotations)
        return 1
    if not videos_root.exists():
        logger.error("Videos root not found: %s", videos_root)
        return 1

    clips_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    video_paths = list_video_paths(annotations, videos_root, project_root=project_root)
    logger.info("Found %d videos to slice", len(video_paths))

    failures: list[tuple[str, str]] = []
    total_clips_written = 0
    for vp in video_paths:
        try:
            total_clips_written += slice_video(
                vp, clips_root,
                clip_seconds=cfg.clip_seconds,
                ffmpeg_bin=cfg.ffmpeg_bin,
                ffprobe_bin=cfg.ffprobe_bin,
                log_level=cfg.ffmpeg_log_level,
                force=args.force,
            )
        except Exception as exc:
            vid = video_id_from_path(str(vp))
            logger.exception("Failed to slice %s: %s", vid, exc)
            failures.append((vid, str(exc)))

    logger.info(
        "Done. %d clips newly written across %d videos, %d failures.",
        total_clips_written, len(video_paths), len(failures),
    )
    if failures:
        # Write failures to a JSON for inspection; the eval harness can
        # skip these videos later.
        failures_path = clips_root.parent / "preprocess_failures.json"
        with failures_path.open("w", encoding="utf-8") as f:
            json.dump([{"video_id": v, "error": e} for v, e in failures], f, indent=2)
        logger.warning("Failures written to %s", failures_path)
        return 2 if len(failures) == len(video_paths) else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice M3-Bench videos into fixed-length clips.")
    p.add_argument("--annotations", required=True,
                   help="Path to M3-Bench annotation JSON (e.g. data/annotations/robot.json)")
    p.add_argument("--videos-root", default="data/videos",
                   help="Root directory containing the video files (default: %(default)s)")
    p.add_argument("--clips-root", default="data/clips",
                   help="Output directory for sliced clips (default: %(default)s)")
    p.add_argument("--clip-seconds", type=int, default=30,
                   help="Length of each output clip in seconds (default: %(default)s)")
    p.add_argument("--fps", type=int, default=5,
                   help="Frame sampling rate inside each clip (default: %(default)s, "
                        "metadata only — actual frame extraction is done by operators later)")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("--project-root", default=None,
                   help="Directory that contains data/ (defaults to cwd)")
    p.add_argument("--force", action="store_true",
                   help="Re-slice even if output clips already exist")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
