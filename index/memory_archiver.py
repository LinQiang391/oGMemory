"""Cold-storage archival for stale memories based on hotness scoring.

Moves memories with low hotness scores to an archive directory,
reducing token consumption from stale abstracts and overviews during
retrieval. Archived memories can be restored to their original location.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.interfaces import ContextFS, VectorIndex
from core.models import RequestContext, SeedHit
from retrieval.hotness import hotness_score, DEFAULT_HALF_LIFE_DAYS

# Directory name for archived memories within each scope
ARCHIVE_DIR = "_archive"


@dataclass
class ArchivalCandidate:
    """A memory that qualifies for archival."""

    uri: str
    active_count: int
    updated_at: Optional[datetime]
    score: float
    context_type: str = ""
    parent_uri: str = ""
    category: str = ""
    owner_space: str = ""


@dataclass
class ArchivalResult:
    """Summary of an archival operation."""

    scanned: int = 0
    archived: int = 0
    skipped: int = 0
    errors: int = 0
    candidates: list[ArchivalCandidate] = field(default_factory=list)


class MemoryArchiver:
    """Archives cold memories based on hotness scoring.

    Uses ``hotness_score()`` from retrieval.hotness to identify memories
    whose access frequency and recency have fallen below a threshold.
    Moves them to ``{scope}/_archive/`` using ``ContextFS.move_node()`` so
    they remain recoverable but are excluded from default retrieval.
    """

    DEFAULT_THRESHOLD: float = 0.1
    DEFAULT_MIN_AGE_DAYS: int = 7
    DEFAULT_BATCH_SIZE: int = 100

    def __init__(
        self,
        fs: ContextFS,
        vector_index: VectorIndex,
        threshold: float = DEFAULT_THRESHOLD,
        min_age_days: int = DEFAULT_MIN_AGE_DAYS,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ):
        """Initialize the archiver.

        Args:
            fs: ContextFS instance for filesystem operations.
            vector_index: VectorIndex for scanning memories.
            threshold: Hotness score below which memories are archived.
            min_age_days: Skip memories updated within this many days.
            half_life_days: Half-life for recency decay in hotness scoring.
        """
        self.fs = fs
        self.vector_index = vector_index
        self.threshold = threshold
        self.min_age_days = min_age_days
        self.half_life_days = half_life_days

    @staticmethod
    def hotness_score(
        active_count: int,
        updated_at: Optional[datetime],
        now: Optional[datetime] = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> float:
        """Compute a 0.0–1.0 hotness score.

        Delegates to retrieval.hotness.hotness_score for consistency.
        Formula: score = sigmoid(log1p(active_count)) * exp_decay(age)

        Args:
            active_count: Number of times this context was retrieved/accessed.
            updated_at: Last update / access timestamp (preferably UTC).
            now: Current time override (useful for deterministic tests).
            half_life_days: Half-life for the recency decay, in days.

        Returns:
            A float in [0.0, 1.0].
        """
        return hotness_score(
            active_count=active_count,
            updated_at=updated_at,
            now=now,
            half_life_days=half_life_days,
        )

    def scan(
        self,
        scope_uri: str,
        ctx: RequestContext,
        now: Optional[datetime] = None,
    ) -> list[ArchivalCandidate]:
        """Scan a scope for cold memories.

        Queries the vector index for all L2 memories under *scope_uri*,
        computes their hotness score, and returns those below the threshold
        that are older than ``min_age_days``.

        Args:
            scope_uri: Root URI to scan (e.g. ``ctx://acme/users/alice/memories/``).
            ctx: Request context for tenant isolation.
            now: Override current time (for deterministic tests).

        Returns:
            List of candidates eligible for archival, sorted by score
            ascending (coldest first).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        candidates: list[ArchivalCandidate] = []

        # Build filters for L2 content only
        # Only scan L2 content -- never archive L0 abstracts or L1 overviews
        filters = {
            "level": 2,
            **ctx.__dict__,  # Include account_id for tenant isolation
        }

        try:
            # Query vector index for memories in scope
            # Use a dummy query vector to get all matching records
            dummy_vector = [0.0] * 1536  # Common embedding dimension

            hits = self.vector_index.search_by_vector(
                query_vector=dummy_vector,
                filters=filters,
                top_k=1000,  # Large batch to scan
            )

            for hit in hits:
                uri = hit.uri

                # Skip entries already in an archive directory
                if f"/{ARCHIVE_DIR}/" in uri:
                    continue

                # Skip entries outside the requested scope
                if not uri.startswith(scope_uri):
                    continue

                # Respect minimum age
                if hit.updated_at is not None:
                    updated_at = hit.updated_at
                    if isinstance(updated_at, str):
                        try:
                            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        except ValueError:
                            updated_at = None

                    if updated_at is not None:
                        age_days = (now - updated_at).total_seconds() / 86400.0
                        if age_days < self.min_age_days:
                            continue

                score = self.hotness_score(
                    active_count=hit.active_count,
                    updated_at=hit.updated_at,
                    now=now,
                    half_life_days=self.half_life_days,
                )

                if score < self.threshold:
                    candidates.append(
                        ArchivalCandidate(
                            uri=uri,
                            active_count=hit.active_count,
                            updated_at=hit.updated_at,
                            score=score,
                            context_type=hit.context_type,
                            parent_uri=hit.parent_uri or "",
                            category=hit.category,
                            owner_space=hit.owner_space,
                        )
                    )

        except Exception:
            # If vector index query fails, return empty candidates
            pass

        # Coldest first
        candidates.sort(key=lambda c: c.score)

        return candidates

    def archive(
        self,
        candidates: list[ArchivalCandidate],
        ctx: RequestContext,
        dry_run: bool = False,
    ) -> ArchivalResult:
        """Archive the given candidates.

        Moves each candidate to ``{parent}/_archive/{filename}`` using
        ``ContextFS.move_node()``, which atomically updates the vector index.

        Args:
            candidates: Output of ``scan()``.
            ctx: Request context for tenant isolation.
            dry_run: If True, return what would happen without moving.

        Returns:
            Summary of the operation.
        """
        result = ArchivalResult(scanned=len(candidates), candidates=candidates)

        for candidate in candidates:
            archive_uri = _build_archive_uri(candidate.uri)

            if dry_run:
                result.skipped += 1
                continue

            try:
                self.fs.move_node(candidate.uri, archive_uri, ctx)
                result.archived += 1
            except Exception:
                result.errors += 1

        return result

    def restore(
        self,
        archived_uri: str,
        ctx: RequestContext,
    ) -> bool:
        """Restore an archived memory to its original location.

        The original location is derived by removing the ``_archive/``
        path segment from the URI.

        Args:
            archived_uri: URI of the archived memory.
            ctx: Request context for tenant isolation.

        Returns:
            True if the memory was restored successfully.
        """
        original_uri = _build_restore_uri(archived_uri)
        if original_uri is None:
            return False

        try:
            self.fs.move_node(archived_uri, original_uri, ctx)
            return True
        except Exception:
            return False

    def scan_and_archive(
        self,
        scope_uri: str,
        ctx: RequestContext,
        dry_run: bool = False,
        now: Optional[datetime] = None,
    ) -> ArchivalResult:
        """Convenience method: scan then archive in one call."""
        candidates = self.scan(scope_uri, ctx, now=now)
        return self.archive(candidates, ctx, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_archive_uri(uri: str) -> str:
    """Insert ``_archive/`` before the filename in a URI.

    ``ctx://acme/users/alice/memories/preferences/coding_style``
    -> ``ctx://acme/users/alice/memories/preferences/_archive/coding_style``
    """
    last_slash = uri.rfind("/")
    if last_slash == -1:
        return f"{ARCHIVE_DIR}/{uri}"

    parent = uri[:last_slash]
    filename = uri[last_slash + 1:]
    return f"{parent}/{ARCHIVE_DIR}/{filename}"


def _build_restore_uri(archived_uri: str) -> Optional[str]:
    """Remove the ``_archive/`` segment to recover the original URI.

    ``ctx://acme/users/alice/memories/preferences/_archive/coding_style``
    -> ``ctx://acme/users/alice/memories/preferences/coding_style``

    Returns None if the URI does not contain ``_archive/``.
    """
    marker = f"/{ARCHIVE_DIR}/"
    idx = archived_uri.find(marker)
    if idx == -1:
        return None

    parent = archived_uri[:idx]
    filename = archived_uri[idx + len(marker) :]
    return f"{parent}/{filename}"
