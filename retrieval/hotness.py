"""Hotness scoring for cold/hot memory lifecycle management.

Pure function — ``score = sigmoid(log1p(active_count)) * exp_decay(age)``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

DEFAULT_HALF_LIFE_DAYS: float = 7.0


def hotness_score(
    active_count: int,
    updated_at: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)

    freq = 1.0 / (1.0 + math.exp(-math.log1p(max(active_count, 0))))

    if updated_at is None:
        return 0.0

    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = max((now - updated_at).total_seconds() / 86_400.0, 0.0)
    decay_rate = math.log(2) / max(half_life_days, 1e-9)
    recency = math.exp(-decay_rate * age_days)

    return freq * recency
