"""Data models for session archival system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ArchiveEntry:
    """A single archived session.

    Represents a compressed snapshot of a session's conversation history.
    """

    archive_id: str  # Unique identifier (uuid4 or timestamp-based)
    session_id: str  # Original session identifier
    overview: str  # Structured overview of the session (→ overview.md)
    abstract: str  # Brief summary (≤100 chars) (→ abstract.md)
    messages: list[dict]  # Full message history (→ messages.json)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveWriteResult:
    """Result of writing an archive to storage."""

    archive_id: str
    session_id: str
    uri: str  # Full AGFS URI where archive was written
    success: bool
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
