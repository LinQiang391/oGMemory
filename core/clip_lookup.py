"""Scope validation for optional exact clip reads (no embedding required)."""
from typing import Any


def clip_ordinal(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        ordinal = int(value)
        return ordinal if ordinal >= 0 and ordinal == float(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def validate_clip_lookup(filters: dict, clip_ids: list[int]) -> list[int]:
    for key in ("account_id", "video_id"):
        if not isinstance(filters.get(key), str) or not filters[key].strip():
            raise ValueError(f"exact clip lookup requires a non-empty {key}")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in clip_ids):
        raise ValueError("clip_ids must contain nonnegative integers")
    return list(dict.fromkeys(clip_ids))
