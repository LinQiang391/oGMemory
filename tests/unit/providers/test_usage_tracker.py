from providers.token_tracker import TokenTracker, UsageTracker


def test_usage_tracker_correlates_tool_tokens_by_tool_category_and_session():
    tracker = UsageTracker()

    llm_call_id = tracker.record_llm(
        120,
        30,
        session_id="session-a",
    )
    tracker.record_tool_call(
        tool_name="read_memory_node",
        category="tool",
        session_id="session-a",
        status="success",
        duration_ms=25.5,
        prompt_tokens=100,
        completion_tokens=20,
        llm_call_id=llm_call_id,
    )
    tracker.record_tool_call(
        tool_name="read_memory_node",
        category="tool",
        session_id="session-a",
        status="error",
        duration_ms=None,
        prompt_tokens=10,
        completion_tokens=5,
    )

    snapshot = tracker.snapshot()

    by_tool = snapshot.tool_stats["by_tool"]["read_memory_node"]
    assert by_tool["call_count"] == 2
    assert by_tool["success_count"] == 1
    assert by_tool["fail_count"] == 1
    assert by_tool["total_prompt_tokens"] == 110
    assert by_tool["total_completion_tokens"] == 25
    assert by_tool["total_tokens"] == 135
    assert by_tool["llm_call_ids"] == [llm_call_id]

    by_category = snapshot.tool_stats["by_category"]["tool"]
    assert by_category["call_count"] == 2
    assert by_category["total_tokens"] == 135

    by_session = snapshot.tool_stats["by_session"]["session-a"]
    assert by_session["llm_calls"] == 1
    assert by_session["tool_calls"] == 2
    assert by_session["input_tokens"] == 120
    assert by_session["output_tokens"] == 30
    assert by_session["tool_tokens"] == 135


def test_token_tracker_keeps_legacy_snapshot_and_supports_tool_calls():
    tracker = TokenTracker()
    tracker.record_llm(10, 5)
    tracker.record_embed(7)
    tracker.record_tool_call(
        tool_name="calculator",
        category="tool",
        session_id="session-b",
        status="completed",
        prompt_tokens=3,
        completion_tokens=2,
    )

    snapshot = tracker.snapshot()

    assert snapshot.input_tokens == 10
    assert snapshot.output_tokens == 5
    assert snapshot.embed_tokens == 7
    assert snapshot.tool_stats["by_tool"]["calculator"]["total_tokens"] == 5
