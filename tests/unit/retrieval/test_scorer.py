"""Unit tests for retrieval.hotness.hotness_score."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from retrieval.hotness import hotness_score


class TestHotnessScore:
    def test_zero_count_zero_date(self):
        assert hotness_score(0, None) == 0.0

    def test_recent_high_count(self):
        now = datetime.now(timezone.utc)
        s = hotness_score(100, now, now=now)
        assert s > 0.5

    def test_old_decays(self):
        now = datetime.now(timezone.utc)
        recent = hotness_score(10, now - timedelta(hours=1), now=now)
        old = hotness_score(10, now - timedelta(days=30), now=now)
        assert recent > old

    def test_more_active_higher(self):
        now = datetime.now(timezone.utc)
        low = hotness_score(1, now, now=now)
        high = hotness_score(100, now, now=now)
        assert high > low

    def test_half_life(self):
        now = datetime.now(timezone.utc)
        at_half = now - timedelta(days=7)
        s_fresh = hotness_score(10, now, now=now, half_life_days=7.0)
        s_half = hotness_score(10, at_half, now=now, half_life_days=7.0)
        ratio = s_half / s_fresh
        assert 0.45 <= ratio <= 0.55
