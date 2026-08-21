from __future__ import annotations

import json

import pytest

from features.tool_result_externalization.fragments import precise_fragment
from features.tool_result_externalization.records import (
    RecordPreservationConfig,
    _factor_template_rows,
    extract_record_candidates,
    record_selection,
    retained_record_values,
    select_retained_records,
)
from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder


def test_comma_rich_build_log_is_not_misclassified_as_csv():
    payload = "\n".join(
        f"[2024-01-01 00:00:{index % 60:02d}] INFO compile.rs:42 "
        f"building module foo_{index} (crate=workspace-{index // 10}, "
        f"deps=[serde={index}, tokio={index}, regex={index % 7}])"
        for index in range(400)
    )
    detected_type, _synopsis = ToolResultSynopsisBuilder(500).build(payload)
    records, layout = extract_record_candidates(
        payload,
        detected_type=detected_type,
        tool_name="run_cargo_build",
        ranking_context="build status",
    )

    assert detected_type == "text"
    assert layout is not None and layout["kind"] == "log"
    assert layout["record_count"] == 400
    assert (records[0]["index"], records[-1]["index"]) == (0, 399)


def test_jsonl_with_commas_remains_line_preserving_text():
    payload = "\n".join([
        '{"id":1,"status":"ok"}',
        '{"id":2,"status":"warn"}',
        '{"id":3,"status":"ok"}',
        '{"id":4,"status":"ok"}',
        '{"id":5,"status":"ok"}',
    ])
    detected_type, _synopsis = ToolResultSynopsisBuilder(500).build(payload)
    records, layout = extract_record_candidates(
        payload,
        detected_type=detected_type,
        tool_name="read_jsonl",
        ranking_context="warn",
    )

    assert detected_type == "text"
    assert layout is not None and layout["record_count"] == 5
    assert records[0]["value"] == '{"id":1,"status":"ok"}'


@pytest.mark.parametrize(
    ("detected_type", "delimiter", "line_ending"),
    [
        ("csv", ",", "\n"),
        ("csv", ",", "\r\n"),
        ("tsv", "\t", "\n"),
        ("tsv", "\t", "\r\n"),
    ],
)
def test_delimited_records_preserve_quoted_newlines_and_exact_spans(
    detected_type,
    delimiter,
    line_ending,
):
    header = delimiter.join(("id", "status", "message"))
    rows = [
        delimiter.join(("1", "ok", "normal")),
        delimiter.join(("2", "failed", f'"line one{line_ending}line two"')),
        delimiter.join(("3", "ok", "normal")),
        delimiter.join(("4", "ok", "normal")),
        delimiter.join(("5", "ok", "normal")),
    ]
    payload = line_ending.join((header, *rows))

    candidates, layout = extract_record_candidates(
        payload,
        detected_type=detected_type,
        tool_name="read_table",
        ranking_context="failed line two",
    )

    assert layout is not None
    assert layout["record_count"] == 5
    assert "_record_spans" not in layout
    multiline = next(row for row in candidates if "line two" in row["value"])
    assert multiline["value"] == rows[1]
    assert payload[
        multiline["source_start_char"]:multiline["source_end_char"]
    ] == rows[1]

    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="failed line two",
        max_chars=1_000,
    )
    retained = record_selection(selected, critical, layout).value
    assert retained.startswith(header + line_ending)
    assert retained_record_values(retained, layout) == rows

    match = payload.encode().find(b"line two")
    fragment = precise_fragment(
        payload,
        match_offset_bytes=match,
        match_length_bytes=len(b"line two"),
        detected_type=detected_type,
        max_chars=200,
    )
    assert fragment["content"] == rows[1]


def test_malformed_csv_falls_back_without_creating_partial_logical_records():
    payload = "id,status,message\n1,ok,normal\n2,failed,\"unterminated\n3,ok,next\n4,ok,next\n5,ok,next"

    candidates, layout = extract_record_candidates(
        payload,
        detected_type="csv",
        tool_name="read_table",
        ranking_context="failed",
    )

    assert layout is not None
    assert layout["kind"] == "text"
    assert layout["record_count"] == 6
    assert all("source_start_char" not in row for row in candidates)


def test_python_fragment_stops_before_module_level_statements():
    payload = (
        "def target():\n"
        "    value = 'needle'\n"
        "    return value\n\n"
        "MODULE_LEVEL = 99\n"
        "print(MODULE_LEVEL)\n"
    )
    fragment = precise_fragment(
        payload,
        match_offset_bytes=payload.encode().find(b"needle"),
        match_length_bytes=len(b"needle"),
        detected_type="code",
        max_chars=500,
    )

    assert fragment["fragment_type"] == "code_function"
    assert "def target" in fragment["content"]
    assert "MODULE_LEVEL" not in fragment["content"]


def test_template_discovery_samples_then_validates_all_retained_rows():
    rows = [
        "当前时间是2026.8.4，正在运行第"
        f"{index}次，当前结果为{'failed' if index == 2 else 'pass'}"
        for index in range(1, 1_001)
    ]
    rows[499] = "系统崩溃：FatalException"

    factorized = _factor_template_rows(
        rows,
        RecordPreservationConfig(template_discovery_max_records=4),
    )

    assert factorized is not None
    assert factorized["template"] == "当前时间是[]，正在运行第[]次，当前结果为[]"
    assert len(factorized["rows"]) == 999
    assert factorized["unmatched_positions"] == [499]
    assert retained_record_values(factorized, {"kind": "log"}) == rows


def test_position_anchors_keep_first_three_last_two_and_middle_bias():
    payload = json.dumps([f"row-{index}" for index in range(20)])
    candidates, layout = extract_record_candidates(
        payload,
        detected_type="json",
        tool_name="read_array",
        ranking_context="",
    )
    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="",
        max_chars=1_000,
    )
    selection = record_selection(selected, critical, layout)

    assert selection.pattern == "array"
    assert {0, 1, 2, 18, 19}.issubset(selection.selected_indices)
    assert any(5 <= index <= 14 for index in selection.selected_indices)


def test_alternating_low_cardinality_values_stay_within_projection_budget():
    payload = json.dumps([
        {"timestamp": index, "state": index % 2}
        for index in range(1_000)
    ])
    candidates, layout = extract_record_candidates(
        payload,
        detected_type="json",
        tool_name="read_metrics",
        ranking_context="",
    )
    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="",
        max_chars=300,
    )
    selection = record_selection(selected, critical, layout)

    assert not any(
        reason.startswith("change_point:")
        for row in candidates
        for reason in row["reasons"]
    )
    assert critical == set()
    assert selection.serialized_chars <= 300
    assert len(selection.selected_indices) < 20


def test_statistical_change_points_are_globally_bounded():
    rows = [
        {
            "timestamp": index,
            "state": (index // 5) % 2,
            "phase": (index // 7) % 3,
        }
        for index in range(1_000)
    ]
    candidates, layout = extract_record_candidates(
        json.dumps(rows),
        detected_type="json",
        tool_name="read_metrics",
        ranking_context="",
        config=RecordPreservationConfig(max_hard_change_points=3),
    )
    change_rows = {
        int(row["index"])
        for row in candidates
        if any(reason.startswith("change_point:") for reason in row["reasons"])
    }

    assert len(change_rows) <= 6
    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="",
        max_chars=500,
    )
    assert change_rows.issubset(critical)
    assert len(selected) < 30


def test_one_persistent_state_transition_remains_hard_evidence():
    rows = [
        {"timestamp": index, "state": 0 if index < 50 else 1}
        for index in range(100)
    ]
    candidates, layout = extract_record_candidates(
        json.dumps(rows),
        detected_type="json",
        tool_name="read_metrics",
        ranking_context="",
    )
    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="",
        max_chars=500,
    )

    assert {49, 50}.issubset(critical)
    assert {49, 50}.issubset({int(row["index"]) for row in selected})


def test_skewed_binary_metric_does_not_mark_every_minority_value_hard():
    rows = [
        {
            "timestamp": index,
            "metric": 1 if 400 <= index < 500 else 0,
        }
        for index in range(1_000)
    ]
    candidates, layout = extract_record_candidates(
        json.dumps(rows),
        detected_type="json",
        tool_name="read_metrics",
        ranking_context="",
    )
    metric_reasons = [
        reason
        for row in candidates
        for reason in row["reasons"]
        if reason.endswith(":metric")
    ]

    assert not any(reason.startswith("numeric_anomaly:") for reason in metric_reasons)
    assert len([
        reason for reason in metric_reasons
        if reason.startswith("change_point:")
    ]) <= 4

    selected, critical = select_retained_records(
        candidates,
        layout,
        ranking_context="",
        max_chars=500,
    )
    assert len(critical) <= 4
    assert len(selected) < 20
