"""Conservative temporal checks for complete evidence (absolute video seconds)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def seconds(value: Any) -> float:
    """Require finite, nonnegative seconds; never turn bad bounds into no bound."""
    if isinstance(value, bool):
        raise ValueError("seconds must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seconds must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("seconds must be a finite nonnegative number")
    return result


def temporal_metadata(content: Mapping[str, Any]) -> dict[str, float]:
    """Normalize a source interval. Missing/invalid provenance stays unknown."""
    try:
        start = seconds(content.get("source_start_sec", content.get("start_sec")))
        duration = seconds(content.get("source_duration_sec", content.get("duration_sec")))
        end = seconds(start + duration)
        if "source_end_sec" in content and seconds(content["source_end_sec"]) != end:
            return {}
    except ValueError:
        return {}
    return {"source_start_sec": start, "source_duration_sec": duration, "source_end_sec": end}


def contains_time(content: Mapping[str, Any], timestamp: float) -> bool:
    """Half-open source interval; no guessed clip duration or annotation offset."""
    timestamp = seconds(timestamp)
    interval = temporal_metadata(content)
    return bool(interval and interval["source_start_sec"] <= timestamp < interval["source_end_sec"])


def current_clip_key(content: Mapping[str, Any]) -> tuple:
    """Prefer the latest start in overlapping container intervals.

    At an exact next-clip start, the new clip wins. Ties prefer the smaller
    ordinal; callers use stable node/URI ordering for duplicate extractions.
    """
    from core.clip_lookup import clip_ordinal
    clip = clip_ordinal(content.get("source_clip_id", content.get("_clip_id", content.get("clip_id"))))
    return (-temporal_metadata(content)["source_start_sec"], clip if clip is not None else float("inf"))


def within_time(content: Mapping[str, Any], before_sec: float | None) -> bool:
    if before_sec is None:
        return True
    cutoff = seconds(before_sec)
    interval = temporal_metadata(content)
    if not interval or interval["source_end_sec"] > cutoff:
        return False
    # ASR timestamps are clip-relative. A malformed/out-of-clip transcript
    # makes the full summary unsafe too: it may already incorporate that ASR.
    transcript = content.get("speech_transcript") or []
    if not isinstance(transcript, (list, tuple)):
        return False
    for segment in transcript:
        try:
            start = seconds(segment.get("start_sec"))
            end = seconds(segment.get("end_sec"))
            if end < start or end > interval["source_duration_sec"]:
                return False
        except (ValueError, AttributeError):
            return False
    return True


def within_scope(
    content: Mapping[str, Any], video_id=None, before_clip=None, before_sec=None
) -> bool:
    video = content.get("source_video_id", content.get("_video_id", content.get("video_id")))
    clip = content.get("source_clip_id", content.get("_clip_id", content.get("clip_id")))
    if video_id is not None and str(video) != str(video_id):
        return False
    if before_clip is not None:
        try:
            if isinstance(clip, bool) or int(clip) != float(clip) or int(clip) > before_clip:
                return False
        except (ValueError, TypeError, OverflowError):
            return False
    return within_time(content, before_sec)


def boundary_transcript(content: Mapping[str, Any], before_sec: float) -> dict:
    """Project a crossing clip to completed speech only, never its summary.

    This is a read-time projection, not a replacement for the persisted node.
    Speaker identities may have been inferred using later context, so they
    are deliberately omitted as well as visual and derived search fields.
    """
    cutoff = seconds(before_sec)
    interval = temporal_metadata(content)
    if not interval or not interval["source_start_sec"] < cutoff < interval["source_end_sec"]:
        return {}
    segments = content.get("speech_transcript")
    if not isinstance(segments, list):
        return {}
    safe = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        try:
            start, end = seconds(segment.get("start_sec")), seconds(segment.get("end_sec"))
        except ValueError:
            continue
        text = segment.get("text")
        if (start <= end <= interval["source_duration_sec"]
                and interval["source_start_sec"] + end <= cutoff
                and isinstance(text, str) and text.strip()):
            safe.append({"start_sec": start, "end_sec": end, "text": text})
    if not safe:
        return {}
    safe.sort(key=lambda segment: (segment["start_sec"], segment["end_sec"]))
    return {
        "source_video_id": content.get("source_video_id", content.get("_video_id", content.get("video_id"))),
        "source_clip_id": content.get("source_clip_id", content.get("_clip_id", content.get("clip_id"))),
        "source_start_sec": interval["source_start_sec"],
        "source_duration_sec": cutoff - interval["source_start_sec"],
        "source_end_sec": cutoff,
        "evidence_role": "boundary_transcript",
        "speech_transcript": safe,
    }
