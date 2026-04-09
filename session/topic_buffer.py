"""Session-level topic buffer for RCA pipeline.

Provides caching for stable slots and prefetch results from PrefetchManager.
This is a key component of Phase 0 (prefetch) and Phase 1 (assembly) of the
RuntimeContextAssembly (RCA) pipeline.
"""

from dataclasses import dataclass, field
from typing import Any
from core.models import SeedHit


@dataclass
class SlotContent:
    """Content stored in a named slot.

    Slots are used to cache stable content like identity, skills_summary,
    and archive_history that doesn't change frequently during a session.
    """
    content: str
    uris: list[str] = field(default_factory=list)
    tokens: int = 0
    cached_at: str = ""

    def __post_init__(self):
        from datetime import datetime, timezone
        if not self.cached_at:
            self.cached_at = datetime.now(timezone.utc).isoformat()


class SessionTopicBuffer:
    """Session-level cache for fixed slots and prefetch results.

    Manages two types of data:
    1. Stable slots (identity, skills_summary, archive_history) - cached across turns
    2. Prefetch results (pending_injection) - populated by PrefetchManager

    Also tracks which URIs have been injected to avoid duplicates in the
    working set across multiple assembly calls.
    """

    def __init__(self, session_id: str):
        """Initialize a new buffer for a session.

        Args:
            session_id: Unique session identifier
        """
        self._session_id = session_id
        self._slots: dict[str, SlotContent] = {}
        self._pending_injection: list[SeedHit] | None = None
        self._injected_uris: set[str] = set()

    # ---------------------------------------------------------------------
    # Stable Base caching
    # ---------------------------------------------------------------------

    def get_cached_slot(self, name: str) -> SlotContent | None:
        """Get cached content for a named slot.

        Args:
            name: Slot name (e.g., "identity", "skills_summary", "archive_history")

        Returns:
            SlotContent if cached, None otherwise
        """
        return self._slots.get(name)

    def set_cached_slot(self, name: str, content: SlotContent) -> None:
        """Cache content for a named slot.

        Args:
            name: Slot name to cache
            content: SlotContent to store
        """
        self._slots[name] = content

    def invalidate(self, name: str) -> None:
        """Invalidate a cached slot.

        Args:
            name: Slot name to invalidate
        """
        if name in self._slots:
            del self._slots[name]

    def clear_all_slots(self) -> None:
        """Clear all cached slots."""
        self._slots.clear()

    # ---------------------------------------------------------------------
    # Prefetch results
    # ---------------------------------------------------------------------

    def get_pending_injection(self) -> list[SeedHit] | None:
        """Get pending prefetch results waiting to be injected.

        Returns:
            List of SeedHit from prefetch, or None if no prefetch pending
        """
        return self._pending_injection

    def set_pending_injection(self, hits: list[SeedHit]) -> None:
        """Set pending prefetch results from PrefetchManager.

        Args:
            hits: List of SeedHit from L0 vector recall
        """
        self._pending_injection = hits

    def clear_pending_injection(self) -> None:
        """Clear pending prefetch results after consumption."""
        self._pending_injection = None

    # ---------------------------------------------------------------------
    # Injection tracking
    # ---------------------------------------------------------------------

    def mark_injected(self, uris: list[str]) -> None:
        """Mark URIs as injected to avoid duplicates.

        Args:
            uris: List of URIs that were included in the working set
        """
        self._injected_uris.update(uris)

    def was_injected(self, uri: str) -> bool:
        """Check if a URI was already injected in a previous turn.

        Args:
            uri: URI to check

        Returns:
            True if URI was already injected, False otherwise
        """
        return uri in self._injected_uris

    def clear_injected_tracking(self) -> None:
        """Clear injected URI tracking (e.g., for new session)."""
        self._injected_uris.clear()

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Get the session ID for this buffer."""
        return self._session_id

    def get_slot_names(self) -> list[str]:
        """Get list of currently cached slot names."""
        return list(self._slots.keys())

    def __repr__(self) -> str:
        return f"SessionTopicBuffer(session_id={self._session_id}, slots={len(self._slots)}, pending={len(self._pending_injection) if self._pending_injection else 0})"
