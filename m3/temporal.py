"""Translate M3 annotation boundaries once, outside generic retrieval services."""

from __future__ import annotations

from core.clip_lookup import clip_ordinal
from core.temporal import contains_time, seconds, temporal_metadata

TEMPORAL_POLICIES = ("strict", "current_clip", "lookahead", "unrestricted")


def add_temporal_arguments(parser):
    parser.add_argument(
        "--temporal-policy", choices=TEMPORAL_POLICIES, default="current_clip",
        help="QA timestamp policy: current_clip includes the whole local current clip; "
             "strict is causal; lookahead allows bounded future speech; unrestricted removes time bounds.",
    )
    parser.add_argument(
        "--lookahead-seconds", type=seconds, default=0.0,
        help="Future allowance for --temporal-policy lookahead (default 0).",
    )


def resolve_question_scope(
    before_clip=None, timestamp=None, *, policy="current_clip",
    lookahead_seconds=0.0, enabled=True, clip_lookup=None,
):
    """Resolve annotation policy into ordinary strict retrieval bounds + audit.

    A timestamp is authoritative over annotation ordinals, which may refer to
    different segmentation. The callback takes absolute seconds and returns
    validated local clip provenance, or None. Missing provenance never opens
    the future: fall back to the exact requested cutoff.
    """
    if policy not in TEMPORAL_POLICIES:
        raise ValueError(f"unknown temporal policy: {policy!r}")
    policy = policy if enabled else "unrestricted"
    grace = seconds(lookahead_seconds) if enabled else 0.0
    if grace and policy != "lookahead":
        raise ValueError("lookahead_seconds requires temporal_policy='lookahead'")
    clip, question_sec = annotation_scope(
        before_clip, timestamp, enabled=policy != "unrestricted",
    )
    cutoff = question_sec
    current = None
    resolution = "unrestricted" if policy == "unrestricted" else "annotation_clip_only"
    if question_sec is not None:
        cutoff = seconds(question_sec + grace) if policy == "lookahead" else question_sec
        # Discard the raw ordinal even if no local interval is found.
        clip = None
        if clip_lookup is None:
            raise ValueError("timestamped QA requires a source-interval lookup")
        current = clip_lookup(cutoff)
        if current is not None:
            ordinal = clip_ordinal(current.get("clip_id"))
            if ordinal is None or not contains_time(current, cutoff):
                raise ValueError("invalid current-clip provenance")
            current = {**temporal_metadata(current), "clip_id": ordinal}
            clip = ordinal
            resolution = "source_interval"
            if policy == "current_clip":
                cutoff = current["source_end_sec"]
        else:
            resolution = "no_containing_clip_strict_fallback"
    audit = {
        "policy": policy,
        "annotation_before_clip": before_clip,
        "annotation_timestamp": timestamp,
        "question_sec": question_sec,
        "resolved_before_clip": clip,
        "effective_before_sec": cutoff,
        "lookahead_seconds": grace if policy == "lookahead" else 0.0,
        "future_allowance_sec": max(0.0, cutoff - question_sec) if question_sec is not None else None,
        "current_clip": current,
        "resolution": resolution,
    }
    return clip, cutoff, audit


def temporal_prompt(audit):
    """Make retrospective evidence explicit without changing the search query."""
    if not audit:
        return ""
    return (
        f"Temporal evidence policy: {audit['policy']}; question time: "
        f"{audit['question_sec']} seconds; evidence cutoff: "
        f"{audit['effective_before_sec']} seconds; inclusive local clip bound: "
        f"{audit['resolved_before_clip']}. "
        "Evidence after the question time may provide retrospective context, "
        "but does not prove an action had already happened at the question time. "
        "For questions about the start or another exact moment, use evidence "
        "describing that moment, not a later state."
    )


def annotation_scope(before_clip=None, timestamp=None, *, enabled=True):
    """Legacy annotation parser; QA uses resolve_question_scope above.

    Return (inclusive annotation clip bound, absolute cutoff in seconds).

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
