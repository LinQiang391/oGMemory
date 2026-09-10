"""Translate M3 annotation boundaries once, outside generic retrieval services."""

from __future__ import annotations

from core.temporal import seconds


def annotation_scope(before_clip=None, timestamp=None, *, enabled=True):
    """Return (inclusive clip bound, absolute-video cutoff in seconds).

    With a timestamp the question's clip may be a candidate, but its *entire*
    source interval must end before the cutoff. Without a timestamp only the
    annotated previous clip and earlier clips are safe. No clip duration is
    guessed. Explicit ignore mode disables both temporal dimensions.
    """
    if not enabled:
        return None, None
    clip = None
    if before_clip is not None and before_clip != "":
        try:
            clip = int(before_clip)
            if isinstance(before_clip, bool) or clip != float(before_clip):
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"invalid before_clip: {before_clip!r}") from exc
    cutoff = None
    if timestamp is not None and timestamp != "":
        if isinstance(timestamp, str) and ":" in timestamp:
            parts = timestamp.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid timestamp: {timestamp!r}")
            values = [seconds(p) for p in parts]
            if any(v >= 60 for v in values[1:]) or any(v != int(v) for v in values[:-1]):
                raise ValueError(f"invalid timestamp: {timestamp!r}")
            cutoff = seconds(sum(v * 60**i for i, v in enumerate(reversed(values))))
        else:
            cutoff = seconds(timestamp)
    return (clip + 1 if clip is not None and cutoff is not None else clip), cutoff
