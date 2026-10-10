"""Tests for :mod:`m3.resume_status`."""

from __future__ import annotations

from core.graph_enums import ExtractionStatus
from m3.resume_status import is_completed_extraction


def test_is_completed_for_enum():
    assert is_completed_extraction(ExtractionStatus.COMPLETED) is True
    assert is_completed_extraction(ExtractionStatus.PENDING) is False
    assert is_completed_extraction(ExtractionStatus.IN_PROGRESS) is False
    assert is_completed_extraction(ExtractionStatus.FAILED) is False


def test_is_completed_for_string_value():
    """SQL payload form stores the raw ``.value`` string."""
    assert is_completed_extraction("completed") is True
    assert is_completed_extraction("pending") is False
    assert is_completed_extraction("failed") is False


def test_is_completed_for_none_or_unknown():
    assert is_completed_extraction(None) is False
    assert is_completed_extraction("") is False
    assert is_completed_extraction(42) is False
