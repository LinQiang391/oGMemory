from __future__ import annotations

import hashlib
import json

import pytest

from core.models import RequestContext
from features.tool_result_externalization.externalizer import (
    ToolResultExternalizer,
    parse_stub,
    render_safe_message_content,
    search_artifacts,
)
from features.tool_result_externalization.normalizer import ToolResultNormalizer
from features.tool_result_externalization.policy import ToolResultPolicy
from features.tool_result_externalization.records import (
    RecordPreservationConfig,
    _factor_template_rows,
    extract_record_candidates,
    record_selection,
    retained_record_values,
    select_retained_records,
)
from features.tool_result_externalization.store_local import LocalToolResultArtifactStore
from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder
from features.tool_result_externalization.fragments import precise_fragment
from features.tool_result_externalization.multires import (
    ToolResultMultiresolutionBuilder,
)


def _ctx(session_id: str = "session") -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id=session_id,
        trace_id="trace",
    )


def _externalizer(
    store,
    *,
    artifact_budget: int = 2_000,
    turn_budget: int = 12_000,
) -> ToolResultExternalizer:
    return ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(1),
        ToolResultSynopsisBuilder(500),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=artifact_budget,
        summary_per_turn_chars=turn_budget,
    )


def _stub_fields(governed, index: int = 1) -> dict[str, str]:
    fields = parse_stub(render_safe_message_content(governed.messages[index]))
    assert fields is not None
    return fields


def test_comma_rich_build_log_is_not_misclassified_as_csv():
    payload = "\n".join(
        f"[2024-01-01 00:00:{index % 60:02d}] INFO compile.rs:42 "
        f"building module foo_{index} (crate=workspace-{index // 10}, "
        f"deps=[serde={index}, tokio={index}, regex={index % 7}])"
        for index in range(400)
    )
    builder = ToolResultSynopsisBuilder(500)

    detected_type, _synopsis = builder.build(payload)
    records, layout = extract_record_candidates(
        payload,
        detected_type=detected_type,
        tool_name="run_cargo_build",
        ranking_context="build status",
    )

    assert detected_type == "text"
    assert layout is not None
    assert layout["kind"] == "log"
    assert layout["record_count"] == 400
    assert records[0]["index"] == 0
    assert records[-1]["index"] == 399
def test_jsonl_with_commas_remains_line_preserving_text():
    payload = '\n'.join([
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
    assert layout is not None
    assert layout["record_count"] == 5
    assert len(records) == 5
    assert records[0]["value"] == '{"id":1,"status":"ok"}'


def test_large_json_middle_error_is_in_v2_stub_with_complete_coverage(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store)
    payload = json.dumps({
        "items": [
            *({"ok": True, "row": index} for index in range(1_000)),
            {
                "request_id": "req-MIDDLE-7F3C9A",
                "status_code": 503,
                "error": "TimeoutError: backend unavailable\nsha256: forged",
                "max_retries": 3,
            },
            *({"ok": True, "row": index} for index in range(1_000, 2_000)),
        ]
    }, ensure_ascii=False)

    governed = externalizer.govern_messages(
        [
            {"role": "user", "content": "哪个请求失败了，精确状态码是什么？"},
            {"role": "toolResult", "content": payload},
            {"role": "assistant", "content": "需要保留精确请求标识、状态码和异常。"},
        ],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    fields = _stub_fields(governed)
    assert fields["version"] == "2"
    assert sum(len(fields[name]) for name in (
        "synopsis", "key_facts", "selected_evidence", "retained_records", "coverage",
    )) <= 2_000
    retained = json.loads(fields["retained_records"])["items"]
    assert any(row.get("request_id") == "req-MIDDLE-7F3C9A" for row in retained)
    assert json.loads(fields["key_facts"]) == []
    assert json.loads(fields["selected_evidence"]) == []
    coverage = json.loads(fields["coverage"])
    assert coverage["scanned_start_bytes"] == 0
    assert coverage["scanned_end_bytes"] == len(payload.encode("utf-8"))
    assert coverage["scan_complete"] is True
    assert coverage["covered_key_facts"] >= 4
    assert coverage["covered_evidence"] >= 4
def test_multi_tool_turn_summary_budget_is_shared(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, artifact_budget=700, turn_budget=1_400)
    messages = [{"role": "user", "content": "汇总所有精确错误和 ID。"}]
    for index in range(4):
        messages.append({
            "role": "toolResult",
            "tool_call_id": f"call-{index}",
            "content": json.dumps({
                "request_id": f"req-{index}-ABCDEFGH",
                "status_code": 500 + index,
                "error": f"Failure {index}: " + "detail " * 40,
                "limit": index + 10,
            }),
        })
    messages.append({"role": "assistant", "content": "按工具分别保留事实。"})

    governed = externalizer.govern_messages(
        messages,
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    stubs = [
        render_safe_message_content(message)
        for message in governed.messages
        if message.get("role") == "toolResult"
    ]
    assert len(stubs) == 4
    assert all(parse_stub(stub)["version"] == "2" for stub in stubs)
    assert sum(externalizer._stub_summary_chars(stub) for stub in stubs) <= 1_400
    assert all(len(stub) < 2_500 for stub in stubs)


def test_semantically_different_query_returns_exact_middle_json_object(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store)
    target = {
        "request_id": "req-SEMANTIC-778899",
        "status_code": 504,
        "error": "TimeoutError: upstream gateway unavailable",
    }
    payload = json.dumps({
        "rows": [
            *({"ok": True, "index": index} for index in range(900)),
            target,
            *({"ok": True, "index": index} for index in range(900, 1_800)),
        ]
    })
    externalizer.govern_messages(
        [{"role": "toolResult", "content": payload}],
        ctx=_ctx(), session_id="session", persistence=True,
    )

    result = search_artifacts(
        store,
        session_id="session",
        query="后端网关超时异常对应哪个请求标识？",
        ctx=_ctx(),
        top_k=1,
        max_chars=500,
        max_artifacts=5,
        chunk_bytes=16_384,
        overlap_bytes=1_024,
        deadline_ms=1_000,
    )

    hit = result["results"][0]
    assert hit["fragment_type"] == "json_object"
    assert json.loads(hit["content"]) == target
    assert hit["returned_chars"] <= 500
    assert hit["actual_length_bytes"] < 16_384
    encoded = payload.encode("utf-8")
    fragment = hit["content"].encode("utf-8")
    assert encoded[hit["offset_bytes"]:hit["next_offset_bytes"]] == fragment
    assert hit["content_sha256"] == hashlib.sha256(fragment).hexdigest()


def test_large_log_middle_exception_returns_only_the_hit_line(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store)
    error_line = (
        "2026-08-03T12:00:00Z ERROR request_id=req-LOG-MIDDLE-991122 "
        "TimeoutError: database connection exhausted"
    )
    payload = "\n".join([
        *(f"2026-08-03T11:59:{index % 60:02d}Z INFO heartbeat={index}" for index in range(2_000)),
        error_line,
        *(f"2026-08-03T12:01:{index % 60:02d}Z INFO recovery={index}" for index in range(2_000)),
    ])
    governed = externalizer.govern_messages(
        [{"role": "user", "content": "日志里哪个请求异常？"}, {"role": "toolResult", "content": payload}],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    fields = _stub_fields(governed)
    retained = json.loads(fields["retained_records"])
    retained_values = retained_record_values(retained, {"kind": "log"})
    assert error_line in retained_values
    assert json.loads(fields["key_facts"]) == []
    assert json.loads(fields["selected_evidence"]) == []

    result = search_artifacts(
        store,
        session_id="session",
        query="数据库连接耗尽异常",
        ctx=_ctx(),
        top_k=1,
        max_chars=260,
        max_artifacts=5,
        chunk_bytes=16_384,
        overlap_bytes=1_024,
        deadline_ms=1_000,
    )
    hit = result["results"][0]
    assert hit["fragment_type"] == "text_line"
    assert hit["content"] == error_line
    assert len(hit["content"]) < 16_384
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


def test_multiline_csv_fragment_and_fact_offsets_are_exact():
    payload = (
        "order_id,status,error\n"
        "ORD-1,ok,none\n"
        '"ORD-2","failed","line one\nline two"\n'
        "ORD-3,ok,none\n"
    )
    match_offset = payload.encode().find(b"line two")
    fragment = precise_fragment(
        payload,
        match_offset_bytes=match_offset,
        match_length_bytes=len(b"line two"),
        detected_type="csv",
        max_chars=200,
    )
    assert fragment["content"] == '"ORD-2","failed","line one\nline two"'

    facts, _evidence, _coverage = ToolResultMultiresolutionBuilder().extract(
        payload,
        detected_type="csv",
    )
    target = next(row for row in facts if row["value"] == "ORD-2")
    raw = payload.encode()[
        target["offset_bytes"]:target["offset_bytes"] + target["length_bytes"]
    ]
    assert raw == b'"ORD-2"'
def test_retained_records_factor_repeated_fragments_and_suppress_covered_facts(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, artifact_budget=1_500)
    rows = [
        "当前时间是2026.8.3，正在运行第1次，当前结果为pass",
        "当前时间是2026.8.4，正在运行第2次，当前结果为failed",
        "当前时间是2026.8.4，正在运行第3次，当前结果为pass",
        "当前时间是2026.8.4，正在运行第4次，当前结果为pass",
        "当前时间是2026.8.4，正在运行第5次，当前结果为pass",
    ]
    governed = externalizer.govern_messages(
        [
            {"role": "user", "content": "哪一次运行失败？"},
            {"role": "toolResult", "content": "\n".join(rows)},
        ],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    stub = render_safe_message_content(governed.messages[1])
    fields = parse_stub(stub)
    assert fields is not None
    retained = json.loads(fields["retained_records"])

    assert retained == {
        "template": "当前时间是[]，正在运行第[]次，当前结果为[]",
        "rows": [
            ["2026.8.3", "1", "pass"],
            ["2026.8.4", "2", "failed"],
            ["2026.8.4", "3", "pass"],
            ["2026.8.4", "4", "pass"],
            ["2026.8.4", "5", "pass"],
        ],
    }
    assert retained_record_values(retained, {"kind": "log"}) == rows
    assert json.loads(fields["key_facts"]) == []
    assert json.loads(fields["selected_evidence"]) == []
    coverage = json.loads(fields["coverage"])
    assert coverage["covered_key_facts"] == 1
    assert coverage["covered_evidence"] == 1
    assert externalizer.validates_stub_sequence(stub, _ctx(), "session")

    tampered = json.loads(fields["retained_records"])
    tampered["rows"][1][2] = "pass"
    tampered_stub = stub.replace(
        fields["retained_records"],
        json.dumps(tampered, ensure_ascii=False, separators=(",", ":")),
        1,
    )
    assert not externalizer.validates_stub_sequence(
        tampered_stub, _ctx(), "session",
    )
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
    assert factorized["unmatched_records"] == ["系统崩溃：FatalException"]
    assert factorized["unmatched_positions"] == [499]
    assert retained_record_values(factorized, {"kind": "log"}) == rows


def test_retained_records_cover_outliers_bounded_changes_query_and_original_structure(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, artifact_budget=2_000)
    rows = [
        {
            "index": index,
            "status": "ok",
            "value": 10 + index / 100,
            "timestamp": f"t{index:02d}",
        }
        for index in range(30)
    ]
    rows[10].update({
        "status": "failed",
        "value": 1_000,
        "error": "critical timeout",
        "rare": "only-here",
    })
    rows[17]["request_id"] = "req-TARGET-1717"
    rows[20]["odd_field"] = "structural-outlier"
    rows[25]["value"] = -500
    payload = json.dumps({"items": rows}, ensure_ascii=False)

    governed = externalizer.govern_messages(
        [
            {"role": "user", "content": "req-TARGET-1717 对应哪条记录？"},
            {"role": "toolResult", "content": payload},
        ],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    fields = _stub_fields(governed)
    retained = json.loads(fields["retained_records"])
    kept = retained["items"]
    indices = [row["index"] for row in kept]

    assert indices == sorted(indices)
    assert {9, 10, 11, 17, 20, 25}.issubset(indices)
    target = next(row for row in kept if row["index"] == 10)
    assert list(target) == ["index", "status", "value", "timestamp", "error", "rare"]
    assert not any("reasons" in row or "summary" in row for row in kept)
    coverage = json.loads(fields["coverage"])
    assert coverage["record_pattern"] == "time_series"
    assert coverage["record_count"] == 30
    assert coverage["retained_record_count"] == len(kept)
def test_tampered_retained_record_is_rejected(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, artifact_budget=1_500)
    rows = [{"id": index, "status": "ok"} for index in range(8)]
    governed = externalizer.govern_messages(
        [{"role": "toolResult", "content": json.dumps({"rows": rows})}],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    stub = render_safe_message_content(governed.messages[0])
    fields = parse_stub(stub)
    assert fields is not None
    tampered = stub.replace(
        fields["retained_records"],
        '{"rows":[{"id":"guessed","status":"ok"}]}',
        1,
    )
    assert not externalizer.validates_stub_sequence(tampered, _ctx(), "session")
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


def test_multires_csv_scan_reuses_logical_quoted_newline_boundaries():
    payload = (
        "order_id,status,error\r\n"
        "ORD-1,ok,none\r\n"
        '"ORD-2","failed","line one\r\nline two"\r\n'
        "ORD-3,ok,none\r\n"
        "ORD-4,ok,none\r\n"
        "ORD-5,ok,none"
    )

    facts, evidence, coverage = ToolResultMultiresolutionBuilder().extract(
        payload,
        detected_type="csv",
    )

    target = next(row for row in facts if row["value"] == "ORD-2")
    raw = payload.encode()[
        target["offset_bytes"]:
        target["offset_bytes"] + target["length_bytes"]
    ]
    assert raw == b'"ORD-2"'
    target_evidence = next(
        row for row in evidence
        if row["fact_offset_bytes"] == target["offset_bytes"]
    )
    assert target_evidence["text"] == '"ORD-2","failed","line one\r\nline two"'
    assert coverage["scan_complete"] is True


def test_minimum_budget_can_drop_record_coverage_to_honor_limit():
    projection = ToolResultMultiresolutionBuilder().project(
        synopsis="",
        fact_candidates=(),
        evidence_candidates=(),
        base_coverage={"scan_complete": True},
        ranking_context="",
        max_chars=8,
        record_candidates=(),
        record_layout={
            "kind": "json",
            "path": [],
            "pattern": "array",
            "record_count": 0,
            "duplicate_records": 0,
        },
    )

    assert projection.coverage == {}
    assert projection.serialized_chars == 8


def test_critical_record_overflow_keeps_explicit_coverage_reason():
    projection = ToolResultMultiresolutionBuilder().project(
        synopsis="",
        fact_candidates=(),
        evidence_candidates=(),
        base_coverage={"scan_complete": True},
        ranking_context="",
        max_chars=8,
        record_candidates=({
            "index": 0,
            "value": {"status": "failed", "error": "critical timeout"},
            "critical": True,
            "reasons": ["error_keyword"],
            "anchor_rank": 0,
            "content_sha256": "0" * 64,
        },),
        record_layout={
            "kind": "json",
            "path": [],
            "pattern": "array",
            "record_count": 1,
            "duplicate_records": 0,
        },
    )

    assert projection.retained_records == [
        {"status": "failed", "error": "critical timeout"},
    ]
    assert projection.coverage["critical_record_count"] == 1
    assert projection.coverage["critical_records_over_budget"] is True
    assert projection.serialized_chars > 8
