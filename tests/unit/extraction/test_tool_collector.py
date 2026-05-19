from __future__ import annotations

from extraction.tool_collector import (
    ToolUsageSummary,
    format_tool_stats_text,
    parse_message_parts,
    update_buffer_stats,
)


def test_parse_plain_text_message_has_no_tool_calls():
    parsed = parse_message_parts("hello")

    assert parsed == {"text": "hello", "tool_calls": []}


def test_parse_openclaw_structured_content_splits_text_and_tool_calls():
    parsed = parse_message_parts([
        {"type": "text", "text": "before"},
        {
            "type": "tool_call",
            "tool_name": "bash",
            "tool_status": "success",
            "duration_ms": None,
            "prompt_tokens": 12,
            "completion_tokens": 3,
        },
        "after",
    ])

    assert parsed["text"] == "before after"
    assert parsed["tool_calls"] == [{
        "tool_name": "bash",
        "tool_input": {},
        "tool_output": "",
        "tool_status": "success",
        "duration_ms": 0,
        "prompt_tokens": 12,
        "completion_tokens": 3,
    }]


def test_update_buffer_stats_aggregates_success_failure_and_tokens():
    stats: dict[str, dict] = {}

    update_buffer_stats(stats, [
        {"tool_name": "bash", "tool_status": "success", "duration_ms": 100, "prompt_tokens": 10, "completion_tokens": 5},
        {"tool_name": "bash", "tool_status": "failed", "duration_ms": 50, "prompt_tokens": 4, "completion_tokens": 1},
        {"tool_name": "read", "tool_status": "completed", "duration_ms": 20, "prompt_tokens": 2, "completion_tokens": 8},
    ])

    assert stats["bash"] == {
        "call_count": 2,
        "success_count": 1,
        "fail_count": 1,
        "total_duration_ms": 150.0,
        "total_prompt_tokens": 14,
        "total_completion_tokens": 6,
    }
    assert stats["read"]["success_count"] == 1


def test_format_tool_stats_text_sorts_by_call_count():
    text = format_tool_stats_text({
        "read": {
            "call_count": 1,
            "success_count": 1,
            "fail_count": 0,
            "total_duration_ms": 10,
            "total_prompt_tokens": 1,
            "total_completion_tokens": 1,
        },
        "bash": {
            "call_count": 2,
            "success_count": 1,
            "fail_count": 1,
            "total_duration_ms": 30,
            "total_prompt_tokens": 2,
            "total_completion_tokens": 2,
        },
    })

    lines = text.splitlines()
    assert lines[0] == "**Tool Usage Statistics (this session):**"
    assert lines[1].startswith("- Tool 'bash': 2 calls")
    assert lines[2].startswith("- Tool 'read': 1 calls")


def test_tool_usage_summary_handles_zero_calls():
    summary = ToolUsageSummary(tool_name="noop")

    assert summary.success_rate == 0.0
    assert summary.avg_duration_ms == 0.0
    assert "0 calls" in summary.to_extraction_text()


def test_parse_non_string_or_list_content():
    assert parse_message_parts(None) == {"text": "", "tool_calls": []}
    assert parse_message_parts({"text": "dict"}) == {"text": "{'text': 'dict'}", "tool_calls": []}


def test_format_tool_stats_text_accepts_tracker_like_object():
    class Tracker:
        tool_stats = {
            "bash": {
                "call_count": 1,
                "success_count": 1,
                "fail_count": 0,
                "total_duration_ms": 10,
                "total_prompt_tokens": 1,
                "total_completion_tokens": 1,
            }
        }

    assert "Tool 'bash'" in format_tool_stats_text(Tracker())


def test_format_tool_stats_text_returns_empty_for_no_stats():
    assert format_tool_stats_text({}) == ""


def test_update_buffer_stats_supports_usage_tracker_objects():
    calls = []

    class Tracker:
        def record_tool_call(self, **kwargs):
            calls.append(kwargs)

    update_buffer_stats(
        Tracker(),
        [{
            "tool_name": "bash",
            "tool_status": "success",
            "duration_ms": None,
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "category": "shell",
            "llm_call_id": "llm-1",
        }],
        session_id="sess",
    )

    assert calls == [{
        "tool_name": "bash",
        "category": "shell",
        "session_id": "sess",
        "status": "success",
        "duration_ms": 0,
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "llm_call_id": "llm-1",
    }]


def test_update_buffer_stats_skips_empty_tool_names():
    stats: dict[str, dict] = {}

    update_buffer_stats(stats, [{"tool_name": "", "tool_status": "success"}])

    assert stats == {}
