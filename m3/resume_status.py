"""Idempotency helpers shared by M3 drivers.

Both :mod:`m3.detect_faces` and :mod:`m3.extract_frames` short-circuit
on rerun when an ``ExtractedNode`` already exists with a successful
result. The comparison must accept multiple storage representations
because the graph layer may serialize the enum value as a plain string
(``"completed"``) before going to disk and deserialize it back into a
:class:`core.graph_enums.ExtractionStatus` StrEnum.

A direct ``== ExtractionStatus.COMPLETED.value`` works in practice
because StrEnum is a string subclass, but it is brittle and opaque.
Centralize the comparison here so future storage backends (which may
e.g. persist as int) only need to adapt this one helper.
"""

from __future__ import annotations

from typing import Any

from core.graph_enums import ExtractionStatus

__all__ = ["is_completed_extraction"]


def is_completed_extraction(status: Any) -> bool:
    """Return ``True`` when ``status`` represents a finished extraction.

    Accepts:
      * :class:`ExtractionStatus` (the in-memory enum form).
      * The raw enum value as a string (``"completed"``), which is what
        the SQL graph store writes to ``payload``.
      * ``None`` and any other value — returned as ``False``.
    """
    if status is None:
        return False
    if isinstance(status, ExtractionStatus):
        return status == ExtractionStatus.COMPLETED
    if isinstance(status, str):
        return status == ExtractionStatus.COMPLETED.value
    return False
