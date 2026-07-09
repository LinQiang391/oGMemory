from session.models import SessionWindowState
from session.session_state import Commitment, SessionState, TaskState
from session.session_state_bridge import apply_session_state_bridge


def test_apply_session_state_bridge_maps_all_prd_fields():
    state = SessionState()
    state.update_task_state(
        "sess-bridge",
        TaskState(
            objective="Complete compact PRD",
            current_stage="bridge",
            blockers=["Need sync frequency"],
        ),
    )
    state.add_commitment(
        "sess-bridge",
        Commitment(content="Keep fallback opt-in", status="fulfilled", kind="constraint"),
    )
    state.add_commitment(
        "sess-bridge",
        Commitment(content="Use soft merge markers", status="fulfilled", kind="decision"),
    )
    state.add_commitment(
        "sess-bridge",
        Commitment(content="Wire prefetch", status="open"),
    )
    window = SessionWindowState()

    changed = apply_session_state_bridge(window, state, "sess-bridge", turn_count=5)

    assert changed is True
    assert window.active_task == "Complete compact PRD - bridge"
    assert window.confirmed_constraints == ["Keep fallback opt-in"]
    assert window.recent_decisions == ["Use soft merge markers"]
    assert window.open_loops == ["Wire prefetch"]
    assert window.uncertainties == ["Need sync frequency"]
    assert window.session_state_version == state.get_version("sess-bridge")
    assert window.session_state_synced_at
    assert window.session_state_sync_turn_count == 5


def test_apply_session_state_bridge_noops_when_version_already_synced():
    state = SessionState()
    state.update_task_state("sess-bridge", TaskState(objective="First task"))
    window = SessionWindowState()
    apply_session_state_bridge(window, state, "sess-bridge", turn_count=1)
    first_synced_at = window.session_state_synced_at

    changed = apply_session_state_bridge(window, state, "sess-bridge", turn_count=2)

    assert changed is False
    assert window.active_task == "First task"
    assert window.session_state_synced_at == first_synced_at


def test_apply_session_state_bridge_clears_stale_task_after_state_clear():
    state = SessionState()
    state.update_task_state("sess-clear", TaskState(objective="Old task"))
    window = SessionWindowState()
    apply_session_state_bridge(window, state, "sess-clear", turn_count=1)
    assert window.active_task == "Old task"

    state.clear_session("sess-clear")
    changed = apply_session_state_bridge(window, state, "sess-clear", turn_count=2)

    assert changed is True
    assert window.active_task == ""
    assert window.confirmed_constraints == []
    assert window.recent_decisions == []
    assert window.open_loops == []
    assert window.uncertainties == []
