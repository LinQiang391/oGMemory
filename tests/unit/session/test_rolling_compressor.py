"""Tests for rolling session compressor."""

from unittest.mock import Mock

from session.models import SessionMessage, SessionWindowState
from session.rolling_compressor import RollingCompressor
from session.session_state import Commitment, SessionState, TaskState


def _msg(role: str, content: str) -> SessionMessage:
    return SessionMessage(id=f"{role}-1", role=role, content=content)


def test_llm_compress_returns_structured_state():
    llm = Mock()
    llm.complete_json.return_value = {
        "active_task": "write compressor tests",
        "confirmed_constraints": ["use pytest"],
        "recent_decisions": ["test direct compressor behavior"],
        "open_loops": ["run coverage"],
        "uncertainties": ["none"],
        "summary": "The user is adding direct RollingCompressor tests.",
    }
    compressor = RollingCompressor(llm=llm)

    result = compressor._llm_compress([_msg("user", "Please test rolling compression")])

    assert result == {
        "active_task": "write compressor tests",
        "confirmed_constraints": ["use pytest"],
        "recent_decisions": ["test direct compressor behavior"],
        "open_loops": ["run coverage"],
        "uncertainties": ["none"],
        "summary": "The user is adding direct RollingCompressor tests.",
    }
    prompt, schema = llm.complete_json.call_args.args
    assert "Analyze this conversation" in prompt
    assert "Please test rolling compression" in prompt
    assert schema["required"] == ["summary"]


def test_compress_noops_when_messages_empty():
    llm = Mock()
    state = SessionWindowState(active_task="keep me")
    compressor = RollingCompressor(llm=llm)

    result = compressor.compress([], state)

    assert result is state
    assert state.active_task == "keep me"
    llm.complete_json.assert_not_called()


def test_compress_noops_when_llm_unavailable():
    state = SessionWindowState(
        active_task="keep task",
        compressed_text="keep summary",
        turn_count_at_last_compress=7,
        token_count_at_last_compress=99,
    )
    compressor = RollingCompressor(llm=None)

    result = compressor.compress([_msg("user", "Compress this")], state)

    assert result is state
    assert state.active_task == "keep task"
    assert state.compressed_text == "keep summary"
    assert state.turn_count_at_last_compress == 7
    assert state.token_count_at_last_compress == 99


def test_compress_noops_when_llm_returns_empty(caplog):
    llm = Mock()
    llm.complete_json.return_value = {}
    state = SessionWindowState(active_task="keep me")
    compressor = RollingCompressor(llm=llm)

    with caplog.at_level("WARNING", logger="ogmem.session"):
        result = compressor.compress([_msg("user", "Compress this")], state)

    assert result is state
    assert state.active_task == "keep me"
    assert "LLM returned empty result" in caplog.text


def test_llm_exception_logs_warning_and_noops(caplog):
    llm = Mock()
    llm.complete_json.side_effect = RuntimeError("llm down")
    state = SessionWindowState(active_task="keep me")
    compressor = RollingCompressor(llm=llm)

    with caplog.at_level("WARNING", logger="ogmem.session"):
        result = compressor.compress([_msg("user", "Compress this")], state)

    assert result is state
    assert state.active_task == "keep me"
    assert "RollingCompressor LLM failed: llm down" in caplog.text


def test_compress_populates_all_structured_fields_and_counters():
    llm = Mock()
    llm.complete_json.return_value = {
        "active_task": "cover rolling compressor",
        "confirmed_constraints": ["only touch relevant files"],
        "recent_decisions": ["use direct unit tests"],
        "open_loops": ["commit passing tests"],
        "uncertainties": ["coverage tooling availability"],
        "summary": "RollingCompressor gets direct unit coverage.",
    }
    messages = [
        _msg("user", "u" * 40),
        _msg("assistant", "a" * 80),
        _msg("user", "done"),
    ]
    state = SessionWindowState()
    compressor = RollingCompressor(llm=llm)

    result = compressor.compress(messages, state)

    assert result is state
    assert state.active_task == "cover rolling compressor"
    assert state.confirmed_constraints == ["only touch relevant files"]
    assert state.recent_decisions == ["use direct unit tests"]
    assert state.open_loops == ["commit passing tests"]
    assert state.uncertainties == ["coverage tooling availability"]
    assert state.compressed_text == "RollingCompressor gets direct unit coverage."
    assert state.turn_count_at_last_compress == 2
    assert state.token_count_at_last_compress == 31


def test_compress_merges_session_state_into_window_state():
    llm = Mock()
    llm.complete_json.return_value = {
        "active_task": "llm inferred task",
        "open_loops": ["llm inferred loop"],
        "summary": "rolling summary only",
    }
    session_state = SessionState()
    session_state.update_task_state("sess-bridge", TaskState(objective="Fix compact bridge"))
    session_state.add_commitment("sess-bridge", Commitment(content="Add durable open loops"))

    state = SessionWindowState()
    result = RollingCompressor(llm=llm).compress(
        [_msg("user", "hello")],
        state,
        session_state=session_state,
        session_id="sess-bridge",
    )

    assert result.active_task == "Fix compact bridge"
    assert result.open_loops == ["Add durable open loops"]
    assert result.compressed_text == "rolling summary only"


def test_compress_maps_full_session_state_into_window_state():
    llm = Mock()
    llm.complete_json.return_value = {
        "active_task": "llm task",
        "confirmed_constraints": [],
        "recent_decisions": [],
        "open_loops": [],
        "uncertainties": [],
        "summary": "rolling summary",
    }
    session_state = SessionState()
    session_state.update_task_state(
        "sess-full-bridge",
        TaskState(
            objective="Ship compact work",
            current_stage="P1 bridge",
            blockers=["Need archive merge decision"],
        ),
    )
    session_state.add_commitment(
        "sess-full-bridge",
        Commitment(content="Constraint: keep fallback opt-in", status="fulfilled"),
    )
    session_state.add_commitment(
        "sess-full-bridge",
        Commitment(content="Decision: merge oldest archives first", status="fulfilled"),
    )
    session_state.add_commitment(
        "sess-full-bridge",
        Commitment(content="Finish TopicDetector tests", status="open"),
    )

    result = RollingCompressor(llm=llm).compress(
        [_msg("user", "hello")],
        SessionWindowState(),
        session_state=session_state,
        session_id="sess-full-bridge",
    )

    assert result.active_task == "Ship compact work - P1 bridge"
    assert result.open_loops == ["Finish TopicDetector tests"]
    assert result.confirmed_constraints == ["keep fallback opt-in"]
    assert result.recent_decisions == ["merge oldest archives first"]
    assert result.uncertainties == ["Need archive merge decision"]


def test_compress_noops_without_llm_when_fallback_disabled():
    state = SessionWindowState(active_task="keep")

    result = RollingCompressor(llm=None, fallback_enabled=False).compress(
        [_msg("user", "Can you finish archive merging?")],
        state,
    )

    assert result.active_task == "keep"
    assert result.compressed_text == ""


def test_compress_uses_rule_fallback_without_llm_when_enabled():
    state = SessionWindowState()

    result = RollingCompressor(llm=None, fallback_enabled=True).compress(
        [
            _msg("user", "Please implement archive merge support. What remains blocked?"),
            _msg("assistant", "I will inspect the archive store."),
        ],
        state,
    )

    assert result.active_task == "Please implement archive merge support."
    assert result.open_loops == ["What remains blocked?"]
    assert result.compressed_text
    assert result.turn_count_at_last_compress == 1


def test_rule_fallback_summary_truncation_drops_partial_leading_word():
    text = "partial" + ("x" * 20) + " " + ("keep " * 140)
    result = RollingCompressor(llm=None, fallback_enabled=True).compress(
        [_msg("user", text)],
        SessionWindowState(),
    )

    assert len(result.compressed_text) <= 600
    assert result.compressed_text
    assert not result.compressed_text.startswith("x")
    assert result.compressed_text.startswith("keep")


def test_rule_fallback_summary_truncation_preserves_long_token_when_no_boundary_exists():
    long_word = "prefix" + ("x" * 610)
    result = RollingCompressor(llm=None, fallback_enabled=True).compress(
        [_msg("user", long_word)],
        SessionWindowState(),
    )

    assert 0 < len(result.compressed_text) <= 600


class _FailingLLM:
    def complete_json(self, prompt, schema):
        raise RuntimeError("llm unavailable")


def test_compress_uses_rule_fallback_when_llm_raises_and_fallback_enabled():
    result = RollingCompressor(llm=_FailingLLM(), fallback_enabled=True).compress(
        [_msg("user", "Please fix the archive merge path. What remains?")],
        SessionWindowState(),
    )

    assert result.active_task == "Please fix the archive merge path."
    assert result.open_loops == ["What remains?"]
    assert result.compressed_text


def test_compress_uses_rule_fallback_when_llm_returns_empty_and_fallback_enabled():
    llm = Mock()
    llm.complete_json.return_value = {}

    result = RollingCompressor(llm=llm, fallback_enabled=True).compress(
        [_msg("user", "Please finish topic detection. What is missing?")],
        SessionWindowState(),
    )

    assert result.active_task == "Please finish topic detection."
    assert result.open_loops == ["What is missing?"]
    assert result.compressed_text
