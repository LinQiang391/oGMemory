"""Shared configuration for M3-Bench scripts.

Centralises paths, account identifiers, and clip defaults so individual
scripts can be invoked with sensible defaults when run from the m3-agent
working directory (where ``data/annotations/robot.json`` and
``data/videos/robot/`` live).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Tenant id used for all M3-Bench data inside oG-Memory's graph.
# All videos for a benchmark run share one account so multi-tenant
# isolation does not block retrieval across videos.
M3BENCH_ACCOUNT_ID = "m3bench"

# Default agent / user ids — operators need them for RequestContext.
M3BENCH_AGENT_ID = "m3bench-agent"
M3BENCH_USER_ID = "m3bench-user"


@dataclass(frozen=True)
class PreprocessConfig:
    """Defaults for video → clip slicing.

    Attributes:
        clip_seconds: Length of each output clip (matches M3-Agent default).
        fps: Frame sampling rate inside each clip (M3-Agent default).
        videos_root: Directory containing video files.
        clips_root: Directory where sliced clips are written.
        ffmpeg_bin: ffmpeg executable name or path.
        ffprobe_bin: ffprobe executable name or path.
        ffmpeg_log_level: ``-loglevel`` value for ffmpeg invocation.
    """
    clip_seconds: int = 30
    fps: int = 5
    videos_root: str = "data/videos"
    clips_root: str = "data/clips"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_log_level: str = "error"


@dataclass(frozen=True)
class LoadConfig:
    """Defaults for clip → RawNode batch loading.

    Attributes:
        annotations_file: M3-Bench annotations JSON.
        clips_root: Where sliced clips live.
        videos_root: Where the original videos live (used to resolve
            ``info["video_path"]`` in the manifest and for source
            metadata on the RawNode).
        output_jsonl: Where to write per-video sample records (M3-Agent
            compatible format).
        account_id: Tenant id; defaults to M3BENCH_ACCOUNT_ID.
        clip_seconds: Clip length in seconds — must match the value the
            slicer used, otherwise ``start_sec`` on every clip will be
            wrong. Defaults to the M3-Agent value (30).
        only_annotated: If True (default) only videos listed in the
            annotation JSON are processed; otherwise every ``*.mp4``
            under ``clips_root`` is processed.
    """
    annotations_file: str = "data/annotations/robot.json"
    clips_root: str = "data/clips"
    videos_root: str = "data/videos"
    output_jsonl: str = "data/data.jsonl"
    account_id: str = M3BENCH_ACCOUNT_ID
    clip_seconds: int = 30
    only_annotated: bool = True


def video_id_from_path(video_path: str) -> str:
    """Extract the M3 video id from a path like ``data/videos/robot/living_room_06.mp4``.

    Returns the filename stem. Works for both ``robot`` and ``web`` subsets.
    """
    return os.path.splitext(os.path.basename(video_path))[0]


def resolve_video_path(
    declared: str,
    *,
    project_root: Path,
    videos_root: Path,
) -> Path | None:
    """Resolve a video path declared in the M3 annotation JSON to a real file.

    Tries, in order:
    1. ``<project_root> / declared`` — the canonical M3 layout where the
       script is invoked from the project root (data/videos/... lives
       under cwd).
    2. ``<videos_root> / <subset> / <stem>.mp4`` — keeps the
       ``videos_root`` argument useful as an override for relocated data
       (e.g. downloaded to a different drive).
    3. ``<videos_root> / <stem>.mp4`` — last-resort flat fallback for
       annotations that don't use a subset directory.

    Returns the first existing candidate, or ``None`` if none match.
    """
    stem = video_id_from_path(declared)
    parts = Path(declared).parts
    subset: str | None = None
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "videos":
        subset = parts[2] if len(parts) >= 3 else None

    candidates: list[Path | None] = [
        project_root / declared,
        videos_root / subset / f"{stem}.mp4" if subset else None,
        videos_root / f"{stem}.mp4",
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None

