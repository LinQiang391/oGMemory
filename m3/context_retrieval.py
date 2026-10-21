"""Shared deterministic neighborhood planning, separate from query ranking."""
from core.clip_lookup import clip_ordinal


def validate_context_options(radius: int, context_k: int) -> None:
    for name, value in (("include_neighbors", radius), ("context_k", context_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")
    if context_k and not radius:
        raise ValueError("context_k > 0 requires include_neighbors > 0")


def neighbor_plan(seeds, *, radius, before_clip=None):
    """Map (video, clip) to (distance, seed rank, supporting seed identity).

    Nearest clips first, then stronger main evidence; no invented clip IDs for
    malformed provenance. Existing main clips are left out of context requests.
    """
    normalized = [(video, clip_ordinal(clip), identity) for video, clip, identity in seeds]
    occupied = {(video, clip) for video, clip, _ in normalized if clip is not None}
    planned = {}
    for rank, (video, clip, identity) in enumerate(normalized):
        if not video or clip is None:
            continue
        for distance in range(1, radius + 1):
            for target in (clip - distance, clip + distance):
                key = (video, target)
                if target < 0 or key in occupied or (before_clip is not None and target > before_clip):
                    continue
                priority = (distance, rank, identity)
                if key not in planned or priority[:2] < planned[key][:2]:
                    planned[key] = priority
    return dict(sorted(planned.items(), key=lambda item: (*item[1][:2], item[0])))
