"""Shared configuration for M3-Bench scripts.

Centralises paths, account identifiers, and clip defaults so individual
scripts can be invoked with sensible defaults when run from the m3-agent
working directory (where ``data/annotations/robot.json`` and
``data/videos/robot/`` live).
"""

from __future__ import annotations

from dataclasses import dataclass


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
        output_jsonl: Where to write per-video sample records (M3-Agent
            compatible format).
        account_id: Tenant id; defaults to M3BENCH_ACCOUNT_ID.
    """
    annotations_file: str = "data/annotations/robot.json"
    clips_root: str = "data/clips"
    output_jsonl: str = "data/data.jsonl"
    account_id: str = M3BENCH_ACCOUNT_ID
    # If True, only videos in annotations are processed; else process every
    # *.mp4 under clips_root.
    only_annotated: bool = True


def video_id_from_path(video_path: str) -> str:
    """Extract the M3 video id from a path like ``data/videos/robot/living_room_06.mp4``.

    Returns the filename stem. Works for both ``robot`` and ``web`` subsets.
    """
    import os
    return os.path.splitext(os.path.basename(video_path))[0]
