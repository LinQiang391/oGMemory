"""Canonical text normalization — re-export from ``core``.

This module is a stable alias for provider code to depend on so the rule
"providers → core only" is not re-stated at every call site.

See ``core.models.normalize_text`` for the authoritative definition.
Duplicating the normalization logic here would re-introduce the hash /
cache-key drift this effort is eliminating (see VECTORIZATION_PLAN §0).
"""

from core.models import normalize_text

__all__ = ["normalize_text"]