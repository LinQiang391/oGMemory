"""Data models for session archival system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SessionMessage:
    """A single message in the session buffer."""

    id: str  # msg_{uuid4_hex}
    role: str  # "user" | "assistant"
    content: str  # message text
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.content) // 4)


@dataclass
class SessionMeta:
    """Session metadata persisted alongside the buffer."""

    session_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    message_count: int = 0
    commit_count: int = 0
    last_commit_at: str = ""


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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveWriteResult:
    """Result of writing an archive to storage."""

    archive_id: str
    session_id: str
    uri: str  # Full AGFS URI where archive was written
    success: bool
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionWindowState:
    """Rolling compressed window for session state layer (Layer 2).

    Structured state object for cognitive state assembly, not just
    chat history compression. Updated every N turns (not every turn like
    Layer 3, not never like Layer 1). Provides the LLM awareness of recent
    conversation flow without re-sending all messages. Goes into
    systemPromptSuffix for KV cache friendliness.
    """

    compressed_text: str = ""  # Fallback: full summary text
    turn_count_at_last_compress: int = 0  # Turns when last compressed
    token_count_at_last_compress: int = 0  # Token count when last compressed
    skills_text: str = ""  # Skills retrieved from compressed content

    # Structured cognitive state fields
    active_task: str = ""  # Current task or sub-task
    confirmed_constraints: list[str] = field(default_factory=list)  # Known constraints
    recent_decisions: list[str] = field(default_factory=list)  # Decisions made this session
    open_loops: list[str] = field(default_factory=list)  # Unresolved items
    uncertainties: list[str] = field(default_factory=list)  # Things we're unsure about
    # TODO: Add last_accessed_at field for working set hit tracking (R6)
