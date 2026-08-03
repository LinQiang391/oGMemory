import json

from skills.sanitizer import StreamingTraceSanitizer, prompt_payload, sanitize_trace
from skills.sedimentation import _activated_skill_names
from skills.trace_compaction import TraceEvidenceCompactor


def _events(output="saved"):
    return [
        {"event_type": "user_message", "prompt": "Transform the workbook."},
        {
            "event_type": "tool_start", "tool_name": "spreadsheet",
            "tool_call_id": "private-call-id", "input": {"operation": "transform"},
        },
        {
            "event_type": "tool_success", "tool_name": "spreadsheet",
            "tool_call_id": "private-call-id", "output": output,
        },
        {"event_type": "assistant_message", "assistant_message": "Saved and verified."},
    ]


def _stream(events, *, max_chars, split=None):
    sanitizer = StreamingTraceSanitizer(allow_failed=True)
    compactor = TraceEvidenceCompactor(max_chars=max_chars)
    for ordinal, event in enumerate(events, start=1):
        cleaned = sanitizer.feed(event)
        compactor.feed(ordinal=ordinal, raw_event=event, sanitized_event=cleaned)
        if split == ordinal:
            checkpoint = compactor.checkpoint(
                sanitizer=sanitizer.checkpoint(), done=False,
            )
            sanitizer = StreamingTraceSanitizer(
                allow_failed=True, checkpoint=checkpoint["sanitizer"],
            )
            compactor = TraceEvidenceCompactor(max_chars=max_chars, checkpoint=checkpoint)
    return sanitizer.finish(), compactor


def test_streaming_sanitizer_matches_whole_trace_across_tool_page_boundary():
    events = _events()
    whole = sanitize_trace(events, allow_failed=True)
    streamed, _ = _stream(events, max_chars=32768, split=2)

    assert streamed.tools == whole.tools
    assert streamed.failed_tools == whole.failed_tools
    assert streamed.failed == whole.failed


def test_short_trace_preserves_current_assessor_payload_exactly():
    events = _events()
    expected = prompt_payload(sanitize_trace(events, allow_failed=True).events)
    _, compactor = _stream(events, max_chars=32768)

    assert prompt_payload(compactor.events()) == expected


def test_long_trace_is_bounded_and_keeps_success_and_final_verification():
    events = _events(output="search noise " * 10000)
    _, compactor = _stream(events, max_chars=12000)
    compacted = compactor.events()
    payload = prompt_payload(compacted)

    assert len(payload) <= 12000
    assert "spreadsheet" in payload
    assert "tool_success" in payload
    assert "Saved and verified" in payload
    assert "TRUNCATED" in payload
    assert compactor.public_metrics(len(payload))["trace_compacted"] is True


def test_required_messages_are_truncated_to_the_total_budget():
    events = [
        {"event_type": "user_message", "prompt": "a" * 4000},
        {"event_type": "user_message", "prompt": "b" * 4000},
        {"event_type": "assistant_message", "assistant_message": "c" * 4000},
    ]

    _, compactor = _stream(events, max_chars=1024)
    compacted = compactor.events()

    assert len(json.dumps(compacted, ensure_ascii=False, sort_keys=True)) <= 1024
    assert [event["event_type"] for event in compacted] == [
        "user_message", "user_message", "assistant_message",
    ]
    assert all("TRUNCATED" in json.dumps(event) for event in compacted)


def test_checkpoint_resume_is_byte_identical_and_contains_no_call_id():
    events = _events(output="large result " * 2000)
    _, continuous = _stream(events, max_chars=9000)
    _, resumed = _stream(events, max_chars=9000, split=2)

    assert prompt_payload(resumed.events()) == prompt_payload(continuous.events())
    checkpoint = resumed.checkpoint(sanitizer={}, done=True)
    serialized = json.dumps(checkpoint, sort_keys=True)
    assert "private-call-id" not in serialized


def test_wide_and_deep_objects_are_structurally_bounded():
    wide = {f"field-{index}": index for index in range(100)}
    deep = "leaf"
    for _ in range(40):
        deep = {"child": deep}
    event = {
        "event_type": "user_message",
        "prompt": "Bound structured input.",
        "wide": wide,
        "deep": deep,
    }
    compactor = TraceEvidenceCompactor(max_chars=32768)

    compactor.feed(ordinal=1, raw_event=event, sanitized_event=event)

    serialized = json.dumps(compactor.events(), sort_keys=True)
    assert "truncated_keys" in serialized
    assert "TRUNCATED type=dict depth=16" in serialized
    assert compactor.public_metrics(len(serialized))["truncated_field_count"] >= 2


def test_checkpoint_bounds_unresolved_tool_starts_before_persisting():
    compactor = TraceEvidenceCompactor(max_chars=1024)
    for ordinal in range(1, 5):
        event = {
            "event_type": "tool_start",
            "tool_name": f"tool-{ordinal}",
            "call_id": f"call-{ordinal}",
            "input": {f"field-{index}": "x" * 4000 for index in range(32)},
        }
        compactor.feed(ordinal=ordinal, raw_event=event, sanitized_event=event)

    checkpoint = compactor.checkpoint(sanitizer={}, done=False)

    assert len(json.dumps(checkpoint).encode()) <= 8192
    assert checkpoint["accumulator"]["pending"] == {}
    assert checkpoint["accumulator"]["pending_dropped"] == {}


def test_evicted_unique_calls_do_not_grow_checkpoint_tombstones():
    compactor = TraceEvidenceCompactor(max_chars=1024)
    for ordinal in range(1, 1001):
        event = {
            "event_type": "tool_start",
            "tool_name": "shell",
            "call_id": f"call-{ordinal}",
            "input": {"command": "x" * 4000},
        }
        compactor.feed(ordinal=ordinal, raw_event=event, sanitized_event=event)

    checkpoint = compactor.checkpoint(sanitizer={}, done=False)

    assert checkpoint["accumulator"]["pending"] == {}
    assert checkpoint["accumulator"]["pending_dropped"] == {}
    assert len(json.dumps(checkpoint).encode()) <= 8192


def test_sanitizer_checkpoint_budget_is_independent_from_assessment_budget():
    sanitizer = StreamingTraceSanitizer(allow_failed=True)
    compactor = TraceEvidenceCompactor(max_chars=1024)
    for ordinal in range(1, sanitizer.MAX_ACTIVE_TOOL_CALLS + 1):
        event = {
            "event_type": "tool_start",
            "tool_name": "shell",
            "call_id": f"call-{ordinal}",
        }
        compactor.feed(
            ordinal=ordinal,
            raw_event=event,
            sanitized_event=sanitizer.feed(event),
        )

    checkpoint = compactor.checkpoint(
        sanitizer=sanitizer.checkpoint(), done=False,
    )

    assert len(checkpoint["sanitizer"]["started"]) == sanitizer.MAX_ACTIVE_TOOL_CALLS


def test_repeated_unidentified_calls_remain_checkpoint_bounded_and_pair_in_order():
    sanitizer = StreamingTraceSanitizer(allow_failed=True)
    compactor = TraceEvidenceCompactor(max_chars=32768)
    start = {
        "event_type": "tool_start", "tool_name": "shell",
        "arguments": {"command": "x" * 4000},
    }
    success = {
        "event_type": "tool_success", "tool_name": "shell", "result": "ok",
    }
    sanitizer.feed({"event_type": "user_message", "prompt": "run the tools"})
    for ordinal in range(1, 101):
        compactor.feed(
            ordinal=ordinal, raw_event=start,
            sanitized_event=sanitizer.feed(start),
        )

    checkpoint = compactor.checkpoint(sanitizer=sanitizer.checkpoint(), done=False)
    assert len(checkpoint["accumulator"]["pending"]["shell|none"]) == 4
    assert checkpoint["accumulator"]["pending_dropped"]["shell|none"] == 96

    sanitizer = StreamingTraceSanitizer(
        allow_failed=True, checkpoint=checkpoint["sanitizer"],
    )
    compactor = TraceEvidenceCompactor(max_chars=32768, checkpoint=checkpoint)
    for ordinal in range(101, 201):
        compactor.feed(
            ordinal=ordinal, raw_event=success,
            sanitized_event=sanitizer.feed(success),
        )
    assistant = {"event_type": "assistant_message", "assistant_message": "done"}
    compactor.feed(
        ordinal=201, raw_event=assistant,
        sanitized_event=sanitizer.feed(assistant),
    )

    assert sanitizer.finish().tools == {"shell"}
    assert compactor.pending == {}
    assert compactor.pending_dropped == {}
    compactor.checkpoint(sanitizer=sanitizer.checkpoint(), done=True)


def _long_failed_trace(*skill_names):
    events = [{"event_type": "user_message", "prompt": "Run the workflow."}]
    for index, skill_name in enumerate(skill_names):
        events.extend([
            {
                "event_type": "tool_start", "tool_name": "ogm_read_skill",
                "tool_call_id": f"skill-{index}", "arguments": {"name": skill_name},
            },
            {
                "event_type": "tool_success", "tool_name": "ogm_read_skill",
                "tool_call_id": f"skill-{index}", "result": "loaded",
            },
        ])
    for index in range(25):
        events.extend([
            {
                "event_type": "tool_start", "tool_name": "other_tool",
                "tool_call_id": f"call-{index}", "arguments": {"index": index},
            },
            {
                "event_type": "tool_success", "tool_name": "other_tool",
                "tool_call_id": f"call-{index}", "result": "ok",
            },
        ])
    events.extend([
        {
            "event_type": "tool_start", "tool_name": "other_tool",
            "tool_call_id": "failed", "arguments": {},
        },
        {
            "event_type": "tool_failure", "tool_name": "other_tool",
            "tool_call_id": "failed", "error": "boom",
        },
        {"event_type": "assistant_message", "assistant_message": "The task failed."},
    ])
    return events


def test_long_failed_trace_keeps_early_skill_activation_across_checkpoint_resume():
    events = _long_failed_trace("repeat-workflow")

    _, compactor = _stream(events, max_chars=4096, split=3)

    assert _activated_skill_names(compactor.events()) == {"repeat-workflow"}


def test_long_failed_trace_keeps_two_distinct_activations_as_ambiguous():
    events = _long_failed_trace("first-skill", "second-skill")

    _, compactor = _stream(events, max_chars=4096)

    assert _activated_skill_names(compactor.events()) == {
        "first-skill", "second-skill",
    }


def test_long_failed_trace_keeps_failure_before_success_pairs_fill_budget():
    events = [{"event_type": "user_message", "prompt": "Run the workflow."}]
    for index in range(4):
        events.extend([
            {
                "event_type": "tool_start", "tool_name": "successful-tool",
                "tool_call_id": f"success-{index}", "arguments": "x" * 40,
            },
            {
                "event_type": "tool_success", "tool_name": "successful-tool",
                "tool_call_id": f"success-{index}", "result": "y" * 40,
            },
        ])
    events.extend([
        {
            "event_type": "tool_start", "tool_name": "failing-tool",
            "tool_call_id": "failure", "arguments": "z" * 40,
        },
        {
            "event_type": "tool_failure", "tool_name": "failing-tool",
            "tool_call_id": "failure", "error": "root-cause-marker" + "e" * 40,
        },
        {"event_type": "assistant_message", "assistant_message": "The task failed."},
    ])

    _, compactor = _stream(events, max_chars=1024)
    compacted = compactor.events()

    assert any(event["event_type"] == "tool_failure" for event in compacted)
    assert "root-cause-marker" in json.dumps(compacted)


def test_oversized_failure_is_projected_into_remaining_budget():
    events = [
        {"event_type": "user_message", "prompt": "Run the workflow."},
        {
            "event_type": "tool_start", "tool_name": "failing-tool",
            "tool_call_id": "failure", "arguments": "z" * 40,
        },
        {
            "event_type": "tool_failure", "tool_name": "failing-tool",
            "tool_call_id": "failure", "error": "root-cause-marker" + "e" * 2000,
        },
        {"event_type": "assistant_message", "assistant_message": "The task failed."},
    ]

    _, compactor = _stream(events, max_chars=1024)
    compacted = compactor.events()

    assert len(json.dumps(compacted, ensure_ascii=False, sort_keys=True)) <= 1024
    assert any(event["event_type"] == "tool_failure" for event in compacted)
    assert "root-cause-marker" in json.dumps(compacted)
    assert "TRUNCATED" in json.dumps(compacted)


def test_failure_keeps_structured_error_when_only_start_exceeds_budget():
    events = [
        {"event_type": "user_message", "prompt": "Run the workflow."},
        {
            "event_type": "tool_start", "tool_name": "failing-tool",
            "tool_call_id": "failure", "arguments": "z" * 2000,
        },
        {
            "event_type": "tool_failure", "tool_name": "failing-tool",
            "tool_call_id": "failure",
            "error": {"code": "E_WRITE", "reason": "root-cause-marker"},
        },
        {"event_type": "assistant_message", "assistant_message": "The task failed."},
    ]

    _, compactor = _stream(events, max_chars=1024)
    compacted = compactor.events()
    failure = next(event for event in compacted if event["event_type"] == "tool_failure")

    assert failure["error"] == {"code": "E_WRITE", "reason": "root-cause-marker"}
