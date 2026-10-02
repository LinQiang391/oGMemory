"""Conservative equality for event drafts before occurrence reconciliation."""

import json
from dataclasses import asdict, replace

from core.models import CandidateMemory


def event_draft_signature(candidate: CandidateMemory) -> str:
    """Only confidence and source IDs may differ in an exact duplicate.

    A shared routing key is not evidence of a shared occurrence. Keep different
    bodies, occurrence fields and attribution for the later semantic review.
    """
    fields = asdict(candidate)
    fields.pop("confidence")
    fields.pop("provenance_ids")
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def merge_duplicate_event_sources(
    existing: CandidateMemory, candidate: CandidateMemory,
) -> CandidateMemory:
    """Merge evidence IDs for equal drafts without mutating either input."""
    winner = candidate if candidate.confidence > existing.confidence else existing
    return replace(winner, provenance_ids=list(dict.fromkeys(
        [*existing.provenance_ids, *candidate.provenance_ids])))
