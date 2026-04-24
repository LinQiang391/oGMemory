"""Session-level task state tracking for RCA pipeline.

S1 of the RuntimeContextAssembly pipeline: manages session-level task state
and commitments. This is the foundation for context-aware assembly.

Implementation uses in-memory storage (no AGFS persistence) per design spec.
Task state and commitments are tracked per session and used by the
assembly pipeline to provide relevant context.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TaskState:
    """Task state for a session.

    Tracks the current objective, stage, next step, and any blockers
    for the task being worked on in a session.

    Fields:
        objective: Overall task goal (e.g., "Debug Python performance issue")
        current_stage: Current stage of work (e.g., "Investigating root cause")
        next_step: Planned next action (e.g., "Profile memory usage")
        blockers: List of blocking items preventing progress
    """

    objective: str | None = None
    current_stage: str | None = None
    next_step: str | None = None
    blockers: list[str] = field(default_factory=list)


@dataclass
class Commitment:
    """A commitment made during a session.

    Commitments are promises or action items that emerge during conversation.
    They can be open (pending), fulfilled (completed), or expired (no longer relevant).

    Fields:
        content: Description of the commitment
        status: Current status - "open" | "fulfilled" | "expired"
        created_at: ISO timestamp when commitment was made
        resolved_at: ISO timestamp when commitment was resolved (for fulfilled)
    """

    content: str
    status: str = "open"  # open | fulfilled | expired
    created_at: str = ""
    resolved_at: str | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class SessionState:
    """S1: Session-level task state tracking.

    Manages task state and commitments for sessions using in-memory storage.
    This is the foundation for context-aware assembly in the RCA pipeline.

    The session state provides:
    - Task context: What the user is working on
    - Commitments: Promises made during conversation
    - Open loops: Unresolved commitments that need attention
    """

    def __init__(self):
        """Initialize session state manager."""
        self._task_states: dict[str, TaskState] = {}
        self._commitments: dict[str, list[Commitment]] = {}

    def get_task_state(self, session_id: str) -> TaskState:
        """Get task state for a session.

        Returns existing TaskState or creates a new empty one.
        Never returns None.

        Args:
            session_id: Session identifier

        Returns:
            TaskState for the session (empty if not previously set)
        """
        if session_id not in self._task_states:
            self._task_states[session_id] = TaskState()
        return self._task_states[session_id]

    def get_commitments(
        self, session_id: str, status: str = "open"
    ) -> list[Commitment]:
        """Get commitments for a session, optionally filtered by status.

        Args:
            session_id: Session identifier
            status: Filter by status ("open", "fulfilled", "expired").
                    If None, returns all commitments.

        Returns:
            List of commitments matching the status filter
        """
        if session_id not in self._commitments:
            return []

        commitments = self._commitments[session_id]

        if status is None:
            return commitments

        return [c for c in commitments if c.status == status]

    def update_task_state(self, session_id: str, state: TaskState) -> None:
        """Update task state for a session.

        Merges with existing state, only overwriting non-None fields.
        This allows partial updates without losing existing information.

        Args:
            session_id: Session identifier
            state: New task state (partial or complete)
        """
        current = self.get_task_state(session_id)

        # Merge: only overwrite non-None fields
        if state.objective is not None:
            current.objective = state.objective
        if state.current_stage is not None:
            current.current_stage = state.current_stage
        if state.next_step is not None:
            current.next_step = state.next_step
        if state.blockers:
            # Replace blockers list entirely
            current.blockers = list(state.blockers)

        self._task_states[session_id] = current

    def add_commitment(self, session_id: str, commitment: Commitment) -> None:
        """Add a commitment to a session.

        Args:
            session_id: Session identifier
            commitment: Commitment to add
        """
        if session_id not in self._commitments:
            self._commitments[session_id] = []

        self._commitments[session_id].append(commitment)

    def resolve_commitment(self, session_id: str, content: str) -> None:
        """Resolve a commitment by marking it as fulfilled.

        Finds commitment by content match (exact or substring) and sets
        status to "fulfilled" with resolved_at timestamp.

        Args:
            session_id: Session identifier
            content: Content string to match commitment
        """
        if session_id not in self._commitments:
            return

        now = datetime.now(timezone.utc).isoformat()

        for commitment in self._commitments[session_id]:
            if commitment.status == "open" and content in commitment.content:
                commitment.status = "fulfilled"
                commitment.resolved_at = now

    def get_open_loops(self, session_id: str) -> list[Commitment]:
        """Get open (unresolved) commitments for a session.

        Open loops are commitments with status="open" that represent
        unresolved promises or action items.

        Args:
            session_id: Session identifier

        Returns:
            List of open commitments
        """
        return self.get_commitments(session_id, status="open")

    def clear_session(self, session_id: str) -> None:
        """Clear all state for a session.

        Useful for session cleanup or testing.

        Args:
            session_id: Session identifier
        """
        if session_id in self._task_states:
            del self._task_states[session_id]
        if session_id in self._commitments:
            del self._commitments[session_id]
