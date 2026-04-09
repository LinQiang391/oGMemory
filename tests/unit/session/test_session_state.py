"""Tests for session.session_state module."""

import pytest
from session.session_state import SessionState, TaskState, Commitment


class TestTaskState:
    """Tests for TaskState dataclass."""

    def test_task_state_defaults(self):
        """Test TaskState with default values."""
        state = TaskState()
        assert state.objective is None
        assert state.current_stage is None
        assert state.next_step is None
        assert state.blockers == []

    def test_task_state_with_values(self):
        """Test TaskState with all fields set."""
        state = TaskState(
            objective="Debug performance issue",
            current_stage="Investigating",
            next_step="Profile memory",
            blockers=["Missing logs", "No access to prod"],
        )
        assert state.objective == "Debug performance issue"
        assert state.current_stage == "Investigating"
        assert state.next_step == "Profile memory"
        assert len(state.blockers) == 2

    def test_task_state_blockers_mutable(self):
        """Test that blockers list is mutable."""
        state = TaskState()
        state.blockers.append("Blocker 1")
        assert len(state.blockers) == 1


class TestCommitment:
    """Tests for Commitment dataclass."""

    def test_commitment_defaults(self):
        """Test Commitment with default values."""
        commitment = Commitment(content="Send email to team")
        assert commitment.content == "Send email to team"
        assert commitment.status == "open"
        assert commitment.created_at != ""
        assert commitment.resolved_at is None

    def test_commitment_with_status(self):
        """Test Commitment with explicit status."""
        commitment = Commitment(
            content="Review PR", status="fulfilled", resolved_at="2025-01-01T00:00:00Z"
        )
        assert commitment.status == "fulfilled"
        assert commitment.resolved_at == "2025-01-01T00:00:00Z"

    def test_commitment_auto_timestamp(self):
        """Test that created_at is auto-generated."""
        commitment = Commitment(content="Test")
        assert commitment.created_at != ""
        # Verify it's a valid ISO format timestamp
        from datetime import datetime

        datetime.fromisoformat(commitment.created_at)


class TestSessionState:
    """Tests for SessionState."""

    def test_initialization(self):
        """Test SessionState initialization."""
        state = SessionState()
        assert state._task_states == {}
        assert state._commitments == {}

    def test_get_task_state_new_session(self):
        """Test getting task state for new session returns empty state."""
        session_state = SessionState()
        task_state = session_state.get_task_state("new-session")

        assert task_state is not None
        assert task_state.objective is None
        assert task_state.current_stage is None
        assert task_state.next_step is None
        assert task_state.blockers == []

    def test_get_task_state_existing_session(self):
        """Test getting task state for existing session."""
        session_state = SessionState()

        # Create task state
        state1 = session_state.get_task_state("session-1")
        state1.objective = "Test objective"

        # Get again - should return same instance
        state2 = session_state.get_task_state("session-1")
        assert state2 is state1
        assert state2.objective == "Test objective"

    def test_update_task_state_new_session(self):
        """Test updating task state for new session."""
        session_state = SessionState()
        new_state = TaskState(
            objective="Deploy to production", current_stage="Testing"
        )

        session_state.update_task_state("session-1", new_state)

        retrieved = session_state.get_task_state("session-1")
        assert retrieved.objective == "Deploy to production"
        assert retrieved.current_stage == "Testing"

    def test_update_task_state_merge_partial(self):
        """Test that update merges with existing state."""
        session_state = SessionState()

        # Initial state
        initial = TaskState(
            objective="Debug issue",
            current_stage="Investigating",
            next_step="Check logs",
            blockers=["Blocker 1"],
        )
        session_state.update_task_state("session-1", initial)

        # Partial update (only update objective and next_step)
        partial = TaskState(objective="Debug performance issue", next_step="Profile")
        session_state.update_task_state("session-1", partial)

        retrieved = session_state.get_task_state("session-1")
        # Updated fields
        assert retrieved.objective == "Debug performance issue"
        assert retrieved.next_step == "Profile"
        # Preserved fields
        assert retrieved.current_stage == "Investigating"
        assert len(retrieved.blockers) == 1

    def test_update_task_state_replace_blockers(self):
        """Test that blockers list is replaced on update."""
        session_state = SessionState()

        initial = TaskState(blockers=["Old blocker 1", "Old blocker 2"])
        session_state.update_task_state("session-1", initial)

        updated = TaskState(blockers=["New blocker"])
        session_state.update_task_state("session-1", updated)

        retrieved = session_state.get_task_state("session-1")
        assert len(retrieved.blockers) == 1
        assert retrieved.blockers[0] == "New blocker"

    def test_update_task_state_empty_blockers_no_replace(self):
        """Test that empty blockers list doesn't replace existing blockers."""
        session_state = SessionState()

        initial = TaskState(blockers=["Blocker 1"])
        session_state.update_task_state("session-1", initial)

        # Update with empty blockers
        partial = TaskState(objective="New objective")
        session_state.update_task_state("session-1", partial)

        retrieved = session_state.get_task_state("session-1")
        # Blockers should be preserved (empty list means "don't update")
        assert len(retrieved.blockers) == 1

    def test_add_commitment(self):
        """Test adding commitment to session."""
        session_state = SessionState()
        commitment = Commitment(content="Send summary email")

        session_state.add_commitment("session-1", commitment)

        commitments = session_state.get_commitments("session-1")
        assert len(commitments) == 1
        assert commitments[0].content == "Send summary email"
        assert commitments[0].status == "open"

    def test_add_multiple_commitments(self):
        """Test adding multiple commitments."""
        session_state = SessionState()

        session_state.add_commitment("session-1", Commitment(content="Task 1"))
        session_state.add_commitment("session-1", Commitment(content="Task 2"))
        session_state.add_commitment("session-1", Commitment(content="Task 3"))

        commitments = session_state.get_commitments("session-1")
        assert len(commitments) == 3

    def test_get_commitments_no_session(self):
        """Test getting commitments for non-existent session."""
        session_state = SessionState()
        commitments = session_state.get_commitments("nonexistent")
        assert commitments == []

    def test_get_commitments_filter_by_status(self):
        """Test filtering commitments by status."""
        session_state = SessionState()

        session_state.add_commitment(
            "session-1", Commitment(content="Open task", status="open")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Fulfilled task", status="fulfilled")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Expired task", status="expired")
        )

        open_only = session_state.get_commitments("session-1", status="open")
        assert len(open_only) == 1
        assert open_only[0].content == "Open task"

        fulfilled_only = session_state.get_commitments("session-1", status="fulfilled")
        assert len(fulfilled_only) == 1

    def test_get_commitments_no_filter(self):
        """Test getting all commitments without status filter."""
        session_state = SessionState()

        session_state.add_commitment(
            "session-1", Commitment(content="Task 1", status="open")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Task 2", status="fulfilled")
        )

        all_commitments = session_state.get_commitments("session-1", status=None)
        assert len(all_commitments) == 2

    def test_resolve_commitment_exact_match(self):
        """Test resolving commitment with exact content match."""
        session_state = SessionState()

        commitment = Commitment(content="Send weekly report")
        session_state.add_commitment("session-1", commitment)

        session_state.resolve_commitment("session-1", "Send weekly report")

        updated = session_state.get_commitments("session-1", status=None)[0]
        assert updated.status == "fulfilled"
        assert updated.resolved_at is not None

    def test_resolve_commitment_substring_match(self):
        """Test resolving commitment with substring match."""
        session_state = SessionState()

        commitment = Commitment(content="Review the pull request for feature X")
        session_state.add_commitment("session-1", commitment)

        # Resolve with substring
        session_state.resolve_commitment("session-1", "pull request")

        updated = session_state.get_commitments("session-1", status=None)[0]
        assert updated.status == "fulfilled"

    def test_resolve_commitment_no_match(self):
        """Test resolving non-existent commitment doesn't crash."""
        session_state = SessionState()

        commitment = Commitment(content="Task 1")
        session_state.add_commitment("session-1", commitment)

        # Try to resolve non-existent commitment
        session_state.resolve_commitment("session-1", "Non-existent task")

        # Original commitment should still be open
        retrieved = session_state.get_commitments("session-1")[0]
        assert retrieved.status == "open"

    def test_resolve_commitment_only_open(self):
        """Test that only open commitments are resolved."""
        session_state = SessionState()

        # Add fulfilled commitment
        session_state.add_commitment(
            "session-1", Commitment(content="Already done", status="fulfilled")
        )
        # Add open commitment
        session_state.add_commitment("session-1", Commitment(content="To do", status="open"))

        session_state.resolve_commitment("session-1", "done")

        # Fulfilled one stays fulfilled
        commitments = session_state.get_commitments("session-1", status=None)
        assert commitments[0].status == "fulfilled"
        assert commitments[1].status == "open"

    def test_get_open_loops(self):
        """Test getting open loops (unresolved commitments)."""
        session_state = SessionState()

        session_state.add_commitment(
            "session-1", Commitment(content="Open task 1", status="open")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Fulfilled task", status="fulfilled")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Open task 2", status="open")
        )

        open_loops = session_state.get_open_loops("session-1")
        assert len(open_loops) == 2
        assert all(c.status == "open" for c in open_loops)

    def test_get_open_loops_empty_session(self):
        """Test getting open loops for session with no commitments."""
        session_state = SessionState()
        open_loops = session_state.get_open_loops("new-session")
        assert open_loops == []

    def test_clear_session(self):
        """Test clearing session state."""
        session_state = SessionState()

        # Add task state and commitments
        task = TaskState(objective="Test")
        session_state.update_task_state("session-1", task)
        session_state.add_commitment("session-1", Commitment(content="Task"))

        # Verify state exists
        assert session_state.get_task_state("session-1").objective == "Test"
        assert len(session_state.get_commitments("session-1")) == 1

        # Clear session
        session_state.clear_session("session-1")

        # Verify state is cleared
        new_task = session_state.get_task_state("session-1")
        assert new_task.objective is None
        assert session_state.get_commitments("session-1") == []

    def test_clear_nonexistent_session(self):
        """Test clearing non-existent session doesn't crash."""
        session_state = SessionState()
        session_state.clear_session("nonexistent")  # Should not raise

    def test_multiple_sessions_independent(self):
        """Test that multiple sessions have independent state."""
        session_state = SessionState()

        # Session 1
        session_state.update_task_state("session-1", TaskState(objective="Task 1"))
        session_state.add_commitment("session-1", Commitment(content="Commitment 1"))

        # Session 2
        session_state.update_task_state("session-2", TaskState(objective="Task 2"))
        session_state.add_commitment("session-2", Commitment(content="Commitment 2"))

        # Verify independence
        state1 = session_state.get_task_state("session-1")
        state2 = session_state.get_task_state("session-2")
        assert state1.objective == "Task 1"
        assert state2.objective == "Task 2"

        commits1 = session_state.get_commitments("session-1")
        commits2 = session_state.get_commitments("session-2")
        assert commits1[0].content == "Commitment 1"
        assert commits2[0].content == "Commitment 2"

    def test_complex_workflow(self):
        """Test a realistic workflow combining all features."""
        session_state = SessionState()

        # S1: User starts task
        task = TaskState(
            objective="Fix authentication bug",
            current_stage="Investigating",
            next_step="Check auth logs",
            blockers=["No access to staging"],
        )
        session_state.update_task_state("session-123", task)

        # S2: Make commitment during conversation
        session_state.add_commitment(
            "session-123", Commitment(content="Share findings with team")
        )

        # S3: Add more commitments
        session_state.add_commitment(
            "session-123", Commitment(content="Create test case")
        )
        session_state.add_commitment(
            "session-123", Commitment(content="Document fix")
        )

        # S4: Check open loops
        open_loops = session_state.get_open_loops("session-123")
        assert len(open_loops) == 3

        # S5: Resolve one commitment
        session_state.resolve_commitment("session-123", "test case")

        # S6: Open loops reduced
        open_loops = session_state.get_open_loops("session-123")
        assert len(open_loops) == 2

        # S7: Update task progress
        updated_task = TaskState(
            current_stage="Implementing fix", next_step="Write unit tests"
        )
        session_state.update_task_state("session-123", updated_task)

        # S8: Verify merge preserved objective and blockers
        final_state = session_state.get_task_state("session-123")
        assert final_state.objective == "Fix authentication bug"
        assert final_state.current_stage == "Implementing fix"
        assert final_state.next_step == "Write unit tests"
        assert len(final_state.blockers) == 1

    def test_commitment_statuses(self):
        """Test all commitment status values."""
        session_state = SessionState()

        session_state.add_commitment(
            "session-1", Commitment(content="Open", status="open")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Fulfilled", status="fulfilled")
        )
        session_state.add_commitment(
            "session-1", Commitment(content="Expired", status="expired")
        )

        all_commitments = session_state.get_commitments("session-1", status=None)
        assert len(all_commitments) == 3

        open_only = session_state.get_commitments("session-1", status="open")
        assert len(open_only) == 1

        fulfilled_only = session_state.get_commitments("session-1", status="fulfilled")
        assert len(fulfilled_only) == 1

        expired_only = session_state.get_commitments("session-1", status="expired")
        assert len(expired_only) == 1
