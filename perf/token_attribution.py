"""Token / model attribution helpers.

Diffs ``providers.token_tracker.TokenTracker`` snapshots around a span so
we can attribute LLM / embedding token usage to specific lifecycle stages
(and sub-spans) without touching provider code.

Design:
    - ``snapshot_service(service)`` reads both ``service._llm.token_tracker``
      and ``service._embedder.token_tracker`` (when present) and returns a
      plain dict.
    - ``diff(after, before)`` subtracts counter fields to produce the
      delta for the span.
    - Model names are read from the providers' public ``.model`` property
      when available; otherwise the class name is used.

This module touches no provider internals and never raises. Missing
providers yield zeroed counters with ``token_source="missing"`` flagged
by the caller.
"""

from __future__ import annotations

from typing import Any


def _snap_tracker(tracker: Any) -> dict[str, int]:
    """Snapshot one ``TokenTracker`` into a flat dict of counters."""
    if tracker is None:
        return {}
    snap = tracker.snapshot()
    return {
        "input_tokens": snap.input_tokens,
        "output_tokens": snap.output_tokens,
        "cache_read": snap.cache_read,
        "cache_write": snap.cache_write,
        "llm_calls": snap.llm_calls,
        "embed_tokens": snap.embed_tokens,
        "embed_calls": snap.embed_calls,
        "local_cache_hits": snap.local_cache_hits,
        "local_cache_misses": snap.local_cache_misses,
        "local_cache_saved_tokens": snap.local_cache_saved_tokens,
    }


def _model_of(obj: Any) -> str | None:
    """Best-effort model identifier from a provider object."""
    if obj is None:
        return None
    model = getattr(obj, "model", None)
    if isinstance(model, str) and model:
        return model
    return obj.__class__.__name__


def snapshot_service(service: Any) -> dict[str, Any]:
    """Capture token counters + model names for the service's providers.

    Returns a dict shaped like::

        {
            "llm": {"input_tokens": ..., ...},
            "embed": {"embed_tokens": ..., ...},
            "llm_model": "gpt-4o-mini",
            "embed_model": "text-embedding-3-small",
        }

    Missing providers yield empty dicts and ``None`` model names.
    """
    llm = getattr(service, "_llm", None)
    embedder = getattr(service, "_embedder", None)

    llm_tracker = getattr(llm, "token_tracker", None) if llm else None
    embed_tracker = getattr(embedder, "token_tracker", None) if embedder else None

    return {
        "llm": _snap_tracker(llm_tracker),
        "embed": _snap_tracker(embed_tracker),
        "llm_model": _model_of(llm),
        "embed_model": _model_of(embedder),
    }


def diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Subtract ``before`` counters from ``after``.

    Model names are preserved from ``after`` (provider identity is stable
    across a single span).
    """
    def _sub(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
        if not a:
            return {}
        out: dict[str, int] = {}
        for k, v in a.items():
            out[k] = int(v) - int(b.get(k, 0))
        return out

    llm_delta = _sub(after.get("llm", {}), before.get("llm", {}) if before else {})
    embed_delta = _sub(after.get("embed", {}), before.get("embed", {}) if before else {})

    return {
        "llm": llm_delta,
        "embed": embed_delta,
        "llm_model": after.get("llm_model"),
        "embed_model": after.get("embed_model"),
    }


def token_source(snapshot: dict[str, Any]) -> str:
    """Classify where the token attribution came from.

    Returns one of ``"tracker"`` (provider has a tracker), or
    ``"missing"`` (no tracker available). Reports use this to decide
    whether to display cost.
    """
    if snapshot.get("llm") or snapshot.get("embed"):
        return "tracker"
    return "missing"


def detect_counter_reset(after: dict[str, Any], before: dict[str, Any], field: str = "input_tokens") -> bool:
    """Return True if a counter appears to have reset between two snapshots.

    Detection heuristic: if ``after[field] < before[field] * 0.5`` (after is
    less than half of before), the counter likely rolled over or was reset.
    """
    a = after.get(field, 0)
    b = before.get(field, 0)
    return bool(a and b and a < b * 0.5)
