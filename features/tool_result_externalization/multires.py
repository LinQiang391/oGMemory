"""Deterministic multi-resolution summaries for untrusted tool output.

The builder scans the complete decoded Artifact once, persists a bounded pool
of literal candidates, and renders a smaller context-ranked projection into a
stub.  Ranking context is request-local and is never stored in Artifact
metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .fragments import iter_delimited_records
from .records import (
    RecordPreservationConfig,
    record_covers_fact,
    record_selection,
    select_retained_records,
)


_ERROR_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_])(?:traceback|error|exception|failed|failure|critical|"
    r"fatal|panic|denied|timeout)(?![A-Za-z0-9_])|错误|异常|失败|严重|致命|恐慌|拒绝|超时)",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"(?:\b(?:must|required|requires?|cannot|can't|should not|at least|at most|"
    r"maximum|minimum|limit(?:ed)?|only)\b|必须|不得|不能|不可|至少|至多|最多|最少|"
    r"限制|仅限|只能)",
    re.IGNORECASE,
)
_KEY_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z_][\w.-]{0,80})\s*[:=]\s*"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^,;\s}\]]{1,256})"
)
_JSON_SCALAR_RE = re.compile(
    r'"(?P<key>(?:\\.|[^"\\]){1,160})"\s*:\s*'
    r'(?P<value>"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)',
    re.DOTALL,
)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_ULID_RE = re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b")
_HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP/\d(?:\.\d)?\s+|\b(?:status(?:_code)?|http_status|response_code|"
    r"exit_code|code)\s*[:=]\s*)([1-5]\d{2})\b",
    re.IGNORECASE,
)
_LOG_LIKE_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}|\[[A-Z]+\]|(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\b)",
    re.IGNORECASE,
)

_ID_LABELS = {
    "id", "uuid", "requestid", "request_id", "traceid", "trace_id",
    "correlationid", "correlation_id", "sessionid", "session_id",
    "taskid", "task_id", "jobid", "job_id", "runid", "run_id",
    "messageid", "message_id", "artifactid", "artifact_id", "orderid",
    "order_id", "transactionid", "transaction_id", "eventid", "event_id",
    "spanid", "span_id", "toolcallid", "tool_call_id",
}
_STATUS_LABELS = {
    "status", "statuscode", "status_code", "httpstatus", "http_status",
    "responsecode", "response_code", "exitcode", "exit_code", "code",
}
_ERROR_LABEL_PARTS = (
    "error", "exception", "failure", "failed", "critical", "panic", "fatal",
)
_EMPTY_ERROR_VALUES = {
    "", "0", "false", "n/a", "nil", "no error", "none", "null", "ok", "success",
}
_NUMBER_LABEL_PARTS = (
    "count", "total", "size", "bytes", "length", "limit", "quota", "timeout",
    "duration", "elapsed", "latency", "retry", "attempt", "port", "rate",
    "percent", "percentage", "threshold", "minimum", "maximum", "min", "max",
)
_TEXT_LABELS = {
    "action", "command", "component", "description", "endpoint", "file",
    "filename", "language", "level", "message", "method", "name", "operation",
    "path", "resource", "service", "tool", "type",
}
_LABEL_PRIORITY = {
    "description": 150,
    "error": 140,
    "error_code": 140,
    "message": 130,
    "trace_id": 125,
    "request_id": 120,
    "path": 115,
    "service": 110,
    "status_code": 105,
    "name": 100,
    "component": 95,
    "endpoint": 90,
    "operation": 85,
    "resource": 80,
    "action": 75,
    "command": 70,
    "method": 65,
    "level": 60,
    "type": 55,
    "language": 20,
}
_CONTEXT_STOPWORDS = {
    "about", "after", "also", "and", "are", "does", "for", "from", "how",
    "into", "its", "only", "that", "the", "this", "what", "when", "which",
    "with", "what's", "whichfile",
}
_UNSCOPED_REPEAT_LABELS = {
    "language", "level", "size", "status", "type",
}

_KIND_PRIORITY = {
    "error": 100,
    "status": 90,
    "id": 80,
    "constraint": 70,
    "number": 60,
    "text": 50,
}
_MIN_SUMMARY_JSON_CHARS = 8  # JSON encodings for "", [], [], and {}


@dataclass(frozen=True)
class _RawFact:
    kind: str
    label: str
    value: str
    char_start: int
    char_end: int
    evidence_start: int
    evidence_end: int
    value_truncated: bool = False


@dataclass(frozen=True)
class SummaryProjection:
    synopsis: str
    key_facts: tuple[dict[str, Any], ...]
    selected_evidence: tuple[dict[str, Any], ...]
    retained_records: Any
    coverage: dict[str, Any]
    serialized_chars: int


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", value.casefold())


def _safe_value(value: Any, limit: int = 512) -> tuple[str, bool]:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, (int, float)):
        rendered = str(value)
    else:
        rendered = str(value).strip()
    rendered = rendered.replace("\x00", "�")
    if len(rendered) <= limit:
        return rendered, False
    return rendered[: max(1, limit - 1)] + "…", True


def _classify(label: str, value: Any) -> str | None:
    normalized = _normalized_label(label)
    lowered_value = str(value).casefold()
    if any(part in normalized for part in _ERROR_LABEL_PARTS):
        if lowered_value.strip() in _EMPTY_ERROR_VALUES:
            return None
        return "error"
    if normalized in _STATUS_LABELS or normalized.endswith("statuscode"):
        return "status"
    raw_label = label.strip()
    if (
        normalized in _ID_LABELS
        or re.search(r"(?:^|[_.-])id$", raw_label, re.IGNORECASE)
        or raw_label.endswith(("Id", "ID"))
    ):
        return "id"
    if any(part in normalized for part in _NUMBER_LABEL_PARTS) and isinstance(
        value, (int, float)
    ):
        return "number"
    if _ERROR_RE.search(lowered_value):
        return "error"
    if _CONSTRAINT_RE.search(lowered_value):
        return "constraint"
    if normalized in _TEXT_LABELS:
        return "text"
    return None


def _json_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw.strip("\"'")


def _byte_offsets(text: str, positions: Iterable[int]) -> dict[int, int]:
    ordered = sorted(set(max(0, min(len(text), position)) for position in positions))
    result: dict[int, int] = {}
    last_char = 0
    last_byte = 0
    for position in ordered:
        last_byte += len(text[last_char:position].encode("utf-8"))
        result[position] = last_byte
        last_char = position
    return result


def _delimited_field_spans(record: str, delimiter: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    in_quotes = False
    index = 0
    while index < len(record):
        char = record[index]
        if char == '"':
            if in_quotes and index + 1 < len(record) and record[index + 1] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
        elif char == delimiter and not in_quotes:
            spans.append((start, index))
            start = index + 1
        index += 1
    spans.append((start, len(record)))
    return spans


def _json_object_spans(text: str) -> list[tuple[int, int]]:
    """Return exact JSON object spans without materializing the payload."""
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append((char, index))
        elif char in "]}" and stack:
            opener, start = stack.pop()
            if (opener, char) not in {("[", "]"), ("{", "}")}:
                continue
            if opener == "{":
                spans.append((start, index + 1))
    return spans


def _enclosing_json_object(
    spans: Iterable[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    matches = [
        span for span in spans
        if span[0] <= char_start and char_end <= span[1]
    ]
    return min(matches, key=lambda span: (span[1] - span[0], span[0])) if matches else None


def _context_profile(context: str) -> tuple[set[str], set[str]]:
    lowered = context.casefold()
    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_.:/-]{1,}", lowered)
        if term not in _CONTEXT_STOPWORDS
    }
    kinds: set[str] = set()
    aliases = {
        "error": ("error", "exception", "failure", "failed", "错误", "异常", "失败", "报错"),
        "status": ("status", "code", "http", "状态码", "返回码", "响应码"),
        "id": (" id", "identifier", "uuid", "request", "trace", "标识", "编号", "请求号", "请求"),
        "constraint": ("must", "required", "limit", "constraint", "必须", "限制", "约束", "不能"),
        "number": ("count", "total", "size", "how many", "数值", "数量", "多少", "大小"),
    }
    for kind, values in aliases.items():
        if any(value in lowered for value in values):
            kinds.add(kind)
    return terms, kinds


def _fact_score(fact: dict[str, Any], context: str) -> tuple[int, int, int]:
    terms, kinds = _context_profile(context)
    haystack = f"{fact.get('label', '')} {fact.get('value', '')}".casefold()
    lexical = sum(32 for term in terms if term and term in haystack)
    exact_value = str(fact.get("value", "")).casefold()
    normalized_label = _normalized_label(str(fact.get("label", "")))
    if (
        exact_value
        and len(exact_value) >= 6
        and normalized_label not in _UNSCOPED_REPEAT_LABELS
        and exact_value in context.casefold()
    ):
        lexical += 80
    kind_bonus = 40 if fact.get("kind") in kinds else 0
    return (
        _KIND_PRIORITY.get(str(fact.get("kind")), 0) + kind_bonus + lexical,
        -int(fact.get("offset_bytes", 0)),
        -len(str(fact.get("value", ""))),
    )


def _summary_json_chars(
    synopsis: str,
    facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    retained_records: Any,
    coverage: dict[str, Any],
) -> int:
    encoded = [
        json.dumps(synopsis, ensure_ascii=False, separators=(",", ":")),
        json.dumps(facts, ensure_ascii=False, separators=(",", ":")),
        json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
        json.dumps(coverage, ensure_ascii=False, separators=(",", ":")),
    ]
    if retained_records is not None:
        encoded.append(json.dumps(
            retained_records, ensure_ascii=False, separators=(",", ":"),
        ))
    return sum(len(value) for value in encoded)


def _balanced_candidate_pool(
    candidates: list[_RawFact],
    max_candidates: int,
) -> list[_RawFact]:
    """Keep deterministic coverage across fact kinds before request-local ranking.

    A pure priority prefix can fill the complete persisted pool with early
    errors and permanently discard every status/ID/constraint/number.  Split
    the bounded pool across the kinds that actually occur, and sample each
    kind across the complete Artifact.  The pool remains content-only and
    deterministic; the current question is used later only by ``project``.
    """
    if len(candidates) <= max_candidates:
        return list(candidates)
    def sample_evenly(rows: list[_RawFact], quota: int) -> list[_RawFact]:
        quota = min(len(rows), max(0, quota))
        if quota <= 0:
            return []
        if quota == 1:
            return [rows[0]]
        indices = [
            round(index * (len(rows) - 1) / (quota - 1))
            for index in range(quota)
        ]
        return [rows[index] for index in dict.fromkeys(indices)]

    groups: dict[tuple[str, str], list[_RawFact]] = {}
    for candidate in candidates:
        key = (candidate.kind, _normalized_label(candidate.label))
        groups.setdefault(key, []).append(candidate)
    group_order = sorted(
        groups,
        key=lambda key: (
            -(
                _KIND_PRIORITY.get(key[0], 0)
                + _LABEL_PRIORITY.get(key[1], 0)
            ),
            -_KIND_PRIORITY.get(key[0], 0),
            key[0],
            key[1],
        ),
    )
    tiers: dict[int, list[tuple[str, str]]] = {}
    for key in group_order:
        score = _KIND_PRIORITY.get(key[0], 0) + _LABEL_PRIORITY.get(key[1], 0)
        tiers.setdefault(score, []).append(key)
    selected_groups: list[tuple[str, str]] = []
    remaining = max_candidates
    for score in sorted(tiers, reverse=True):
        if remaining <= 0:
            break
        tier = sorted(tiers[score])
        if len(tier) <= remaining:
            selected_groups.extend(tier)
            remaining -= len(tier)
            continue
        if remaining == 1:
            indices = [len(tier) - 1]
        else:
            indices = [
                round(index * (len(tier) - 1) / (remaining - 1))
                for index in range(remaining)
            ]
        selected_groups.extend(tier[index] for index in dict.fromkeys(indices))
        remaining = 0
    quotas = {key: 1 for key in selected_groups}
    while remaining > 0:
        progressed = False
        for key in selected_groups:
            if quotas[key] >= len(groups[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    selected = [
        item
        for key in selected_groups
        for item in sample_evenly(groups[key], quotas[key])
    ]
    return sorted(
        selected[:max_candidates],
        key=lambda fact: (-_KIND_PRIORITY.get(fact.kind, 0), fact.char_start, fact.label),
    )


class ToolResultMultiresolutionBuilder:
    """Extract and project bounded literal facts without an LLM call."""

    def __init__(
        self,
        max_candidates: int = 96,
        max_evidence_chars: int = 700,
        record_config: RecordPreservationConfig | None = None,
    ):
        if max_candidates < 1 or max_evidence_chars < 32:
            raise ValueError("multi-resolution limits must be positive")
        self.max_candidates = max_candidates
        self.max_evidence_chars = max_evidence_chars
        self.record_config = record_config or RecordPreservationConfig()

    def extract(
        self,
        text: str,
        *,
        detected_type: str,
        ranking_context: str = "",
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
        raw_facts: list[_RawFact] = []
        self._scan_json_scalars(text, raw_facts)
        self._scan_lines(text, raw_facts, detected_type=detected_type)
        if detected_type in {"csv", "tsv"}:
            self._scan_delimited(text, raw_facts, "\t" if detected_type == "tsv" else ",")

        deduplicated: dict[tuple[str, str, str, int], _RawFact] = {}
        for fact in raw_facts:
            key = (fact.kind, fact.label.casefold(), fact.value, fact.char_start)
            deduplicated.setdefault(key, fact)
        ordered_raw = sorted(
            deduplicated.values(),
            key=lambda fact: (-_KIND_PRIORITY.get(fact.kind, 0), fact.char_start, fact.label),
        )
        total_candidates = len(ordered_raw)
        # ``ranking_context`` is deliberately not used while constructing the
        # persisted pool.  Request-local ranking happens only in ``project``.
        _ = ranking_context
        pool_raw = _balanced_candidate_pool(ordered_raw, self.max_candidates)
        json_objects = _json_object_spans(text) if detected_type == "json" else []
        evidence_spans: dict[tuple[int, int], tuple[int, int]] = {}
        for fact in pool_raw:
            span = _enclosing_json_object(
                json_objects, fact.char_start, fact.char_end,
            )
            evidence_spans[(fact.char_start, fact.char_end)] = span or (
                fact.evidence_start, fact.evidence_end,
            )
        positions = [
            position
            for fact in pool_raw
            for position in (
                fact.char_start,
                fact.char_end,
                *evidence_spans[(fact.char_start, fact.char_end)],
            )
        ]
        offsets = _byte_offsets(text, positions)

        facts: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for fact in pool_raw:
            source_start = offsets[fact.char_start]
            source_end = offsets[fact.char_end]
            evidence_char_start, evidence_char_end = evidence_spans[
                (fact.char_start, fact.char_end)
            ]
            item = {
                "kind": fact.kind,
                "label": fact.label,
                "value": fact.value,
                "offset_bytes": source_start,
                "length_bytes": max(0, source_end - source_start),
            }
            if fact.value_truncated:
                item["value_truncated"] = True
            facts.append(item)

            fact_char_length = fact.char_end - fact.char_start
            fact_complete = fact_char_length <= self.max_evidence_chars
            literal = text[evidence_char_start:evidence_char_end]
            if len(literal) > self.max_evidence_chars:
                available = self.max_evidence_chars
                if fact_complete:
                    before = max(0, (available - fact_char_length) // 2)
                    centered_start = fact.char_start - before
                else:
                    centered_start = fact.char_start
                evidence_char_start = max(
                    evidence_char_start,
                    min(centered_start, evidence_char_end - available),
                )
                evidence_char_end = min(
                    evidence_char_end,
                    evidence_char_start + available,
                )
                literal = text[evidence_char_start:evidence_char_end]
                literal_truncated = True
            else:
                literal_truncated = False
            evidence_start = _byte_offsets(
                text, (evidence_char_start,),
            )[evidence_char_start]
            encoded = literal.encode("utf-8")
            evidence_item = {
                "offset_bytes": evidence_start,
                "length_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "text": literal,
                "fact_offset_bytes": source_start,
            }
            if literal_truncated:
                evidence_item["truncated"] = True
            if not fact_complete:
                evidence_item["fact_complete"] = False
            evidence.append(evidence_item)

        evidence_by_offset = {item["fact_offset_bytes"]: item for item in evidence}
        evidence = list(evidence_by_offset.values())

        total_bytes = len(text.encode("utf-8"))
        coverage = {
            "scanned_start_bytes": 0,
            "scanned_end_bytes": total_bytes,
            "scan_complete": True,
            "candidate_pool_truncated": total_candidates > len(facts),
            "uncovered_candidates": max(0, total_candidates - len(facts)),
        }
        return tuple(facts), tuple(evidence), coverage

    @staticmethod
    def _scan_json_scalars(text: str, output: list[_RawFact]) -> None:
        for match in _JSON_SCALAR_RE.finditer(text):
            try:
                label = json.loads(f'"{match.group("key")}"')
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            value = _json_scalar(match.group("value"))
            kind = _classify(str(label), value)
            if kind is None:
                continue
            rendered, truncated = _safe_value(value)
            output.append(_RawFact(
                kind=kind,
                label=str(label)[:160],
                value=rendered,
                char_start=match.start("value"),
                char_end=match.end("value"),
                evidence_start=match.start(),
                evidence_end=match.end(),
                value_truncated=truncated,
            ))

    @staticmethod
    def _scan_lines(
        text: str,
        output: list[_RawFact],
        *,
        detected_type: str,
    ) -> None:
        structured = detected_type in {"json", "csv", "tsv"}
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                cursor += len(raw_line)
                continue
            line_start = cursor + (len(line) - len(line.lstrip()))
            line_end = cursor + len(line)

            for match in (() if structured else _KEY_VALUE_RE.finditer(line)):
                label = match.group("label")
                raw_value = match.group("value")
                value = _json_scalar(raw_value) if raw_value.startswith('"') else raw_value.strip("'")
                kind = _classify(label, value)
                if kind is None:
                    continue
                rendered, truncated = _safe_value(value)
                output.append(_RawFact(
                    kind=kind,
                    label=label[:160],
                    value=rendered,
                    char_start=cursor + match.start("value"),
                    char_end=cursor + match.end("value"),
                    evidence_start=line_start,
                    evidence_end=line_end,
                    value_truncated=truncated,
                ))

            if not structured and _ERROR_RE.search(stripped):
                rendered, truncated = _safe_value(stripped)
                output.append(_RawFact(
                    "error", "message", rendered, line_start, line_end,
                    line_start, line_end, truncated,
                ))
            if not structured and _CONSTRAINT_RE.search(stripped):
                rendered, truncated = _safe_value(stripped)
                output.append(_RawFact(
                    "constraint", "constraint", rendered, line_start, line_end,
                    line_start, line_end, truncated,
                ))
            for match in (() if structured else _UUID_RE.finditer(line)):
                output.append(_RawFact(
                    "id", "uuid", match.group(0), cursor + match.start(), cursor + match.end(),
                    line_start, line_end,
                ))
            for match in (() if structured else _ULID_RE.finditer(line)):
                output.append(_RawFact(
                    "id", "ulid", match.group(0), cursor + match.start(), cursor + match.end(),
                    line_start, line_end,
                ))
            if not structured and (
                _LOG_LIKE_RE.search(stripped) or "status" in stripped.casefold()
            ):
                for match in _HTTP_STATUS_RE.finditer(line):
                    output.append(_RawFact(
                        "status", "status_code", match.group(1),
                        cursor + match.start(1), cursor + match.end(1), line_start, line_end,
                    ))
            cursor += len(raw_line)

    @staticmethod
    def _scan_delimited(text: str, output: list[_RawFact], delimiter: str) -> None:
        try:
            records = list(iter_delimited_records(text, delimiter))
        except csv.Error:
            return
        if len(records) < 2:
            return
        header = records[0].fields
        columns = [
            (index, name, _classify(name, 0))
            for index, name in enumerate(header)
        ]
        for logical_record in records[1:]:
            row = logical_record.fields
            record = logical_record.raw
            spans = _delimited_field_spans(record, delimiter)
            for index, label, header_kind in columns:
                if index >= len(row) or index >= len(spans) or not row[index].strip():
                    continue
                kind = header_kind or _classify(label, row[index])
                if kind is None and (
                    _UUID_RE.fullmatch(row[index].strip())
                    or _ULID_RE.fullmatch(row[index].strip())
                ):
                    kind = "id"
                if kind is None:
                    continue
                value, truncated = _safe_value(row[index])
                local_start, local_end = spans[index]
                output.append(_RawFact(
                    kind, label[:160], value,
                    logical_record.char_start + local_start,
                    logical_record.char_start + local_end,
                    logical_record.char_start,
                    logical_record.char_end,
                    truncated,
                ))

    def project(
        self,
        *,
        synopsis: str,
        fact_candidates: Iterable[dict[str, Any]],
        evidence_candidates: Iterable[dict[str, Any]],
        base_coverage: dict[str, Any] | None,
        ranking_context: str,
        max_chars: int,
        record_candidates: Iterable[dict[str, Any]] = (),
        record_layout: dict[str, Any] | None = None,
    ) -> SummaryProjection:
        if max_chars < _MIN_SUMMARY_JSON_CHARS:
            raise ValueError(
                f"summary projection budget must be at least {_MIN_SUMMARY_JSON_CHARS} chars"
            )
        ranked_facts = sorted(
            (dict(item) for item in fact_candidates if isinstance(item, dict)),
            key=lambda item: _fact_score(item, ranking_context),
            reverse=True,
        )
        evidence_by_fact = {
            int(item.get("fact_offset_bytes", -1)): dict(item)
            for item in evidence_candidates
            if isinstance(item, dict)
        }
        rendered_synopsis = synopsis
        selected_record_rows, critical_record_indices = select_retained_records(
            record_candidates,
            record_layout,
            ranking_context=ranking_context,
            max_chars=max_chars,
            config=self.record_config,
        )

        record_selection_cache: dict[tuple[int, ...], Any] = {}

        def selection_for(rows: Iterable[dict[str, Any]]):
            materialized = list(rows)
            key = tuple(int(row.get("index", -1)) for row in materialized)
            if key not in record_selection_cache:
                record_selection_cache[key] = record_selection(
                    materialized,
                    critical_record_indices,
                    record_layout,
                    self.record_config,
                )
            return record_selection_cache[key]

        def current_record_selection():
            return selection_for(selected_record_rows)

        covered_offsets_cache: dict[tuple[int, ...], set[int]] = {}

        def covered_fact_offsets() -> set[int]:
            key = tuple(
                int(row.get("index", -1)) for row in selected_record_rows
            )
            if key in covered_offsets_cache:
                return covered_offsets_cache[key]
            covered_offsets_cache[key] = {
                int(fact.get("offset_bytes", -1))
                for fact in ranked_facts
                if any(
                    record_covers_fact(row.get("value"), fact)
                    for row in selected_record_rows
                )
            }
            return covered_offsets_cache[key]

        initially_covered = covered_fact_offsets()
        uncovered_ranked_facts = [
            fact
            for fact in ranked_facts
            if int(fact.get("offset_bytes", -1)) not in initially_covered
        ]
        selected_facts = uncovered_ranked_facts[:16]
        selected_evidence: list[dict[str, Any]] = []
        selected_evidence_keys: set[tuple[int, int, str]] = set()
        for item in selected_facts:
            offset = int(item.get("offset_bytes", -2))
            if offset not in evidence_by_fact:
                continue
            evidence_item = evidence_by_fact[offset]
            evidence_key = (
                int(evidence_item.get("offset_bytes", -1)),
                int(evidence_item.get("length_bytes", -1)),
                str(evidence_item.get("sha256", "")),
            )
            if evidence_key in selected_evidence_keys:
                continue
            selected_evidence_keys.add(evidence_key)
            selected_evidence.append(evidence_item)
            if len(selected_evidence) >= 4:
                break

        def coverage_for(*, compact: bool = False) -> dict[str, Any]:
            records = current_record_selection()
            critical_rows = [
                row for row in selected_record_rows
                if int(row.get("index", -1)) in critical_record_indices
            ]
            critical = selection_for(critical_rows)
            coverage = {} if compact else dict(base_coverage or {})
            covered_offsets = covered_fact_offsets()
            uncovered_facts = [
                fact
                for fact in ranked_facts
                if int(fact.get("offset_bytes", -1)) not in covered_offsets
            ]
            omitted_facts = max(0, len(uncovered_facts) - len(selected_facts))
            covered_evidence = sum(
                int(item.get("fact_offset_bytes", -2)) in covered_offsets
                for item in evidence_by_fact.values()
            )
            eligible_evidence = len({
                int(item.get("offset_bytes", -2))
                for item in selected_facts
                if int(item.get("offset_bytes", -2)) in evidence_by_fact
            })
            omitted_evidence = max(0, eligible_evidence - len(selected_evidence))
            evidence_truncated = any(
                bool(item.get("truncated"))
                or item.get("fact_complete") is False
                for item in selected_evidence
            )
            summary_truncated = bool(
                omitted_facts
                or omitted_evidence
                or evidence_truncated
                or coverage.get("candidate_pool_truncated")
                or rendered_synopsis != synopsis
                or records.omitted_records > 0
            )
            record_coverage = (
                {
                    "record_pattern": records.pattern,
                    "record_count": records.total_records,
                    "retained_record_count": len(records.selected_indices),
                    "critical_record_count": len(records.critical_indices),
                    "deduplicated_records": records.duplicate_records,
                    "omitted_records": records.omitted_records,
                    "critical_records_over_budget": critical.serialized_chars > max_chars,
                }
                if record_layout is not None
                else {}
            )
            if compact:
                coverage.update({
                    "scan_complete": bool((base_coverage or {}).get("scan_complete", False)),
                    "candidate_pool_truncated": bool(
                        (base_coverage or {}).get("candidate_pool_truncated", False)
                    ),
                    "evidence_truncated": evidence_truncated,
                    "summary_truncated": True,
                    "coverage_compact": True,
                    **record_coverage,
                })
                return coverage
            coverage.update({
                "summary_truncated": summary_truncated,
                "omitted_key_facts": omitted_facts,
                "omitted_evidence": omitted_evidence,
                "evidence_truncated": evidence_truncated,
                **(
                    {"covered_key_facts": len(covered_offsets)}
                    if covered_offsets
                    else {}
                ),
                **(
                    {"covered_evidence": covered_evidence}
                    if covered_evidence
                    else {}
                ),
                **record_coverage,
                "uncovered": (
                    "lower-ranked facts, evidence, records, or ordinary content omitted"
                    if summary_truncated
                    else "ordinary content is represented by retained records and synopsis"
                ),
            })
            return coverage

        compact_coverage = False
        coverage = coverage_for()
        while _summary_json_chars(
            rendered_synopsis,
            selected_facts,
            selected_evidence,
            current_record_selection().value if selected_record_rows else None,
            coverage,
        ) > max_chars:
            if not compact_coverage:
                compact_coverage = True
                coverage = coverage_for(compact=True)
                continue
            elif len(selected_facts) > max(4, len(selected_evidence)):
                protected_offsets = {
                    int(item.get("fact_offset_bytes", -2))
                    for item in selected_evidence
                }
                removable = next((
                    index for index in range(len(selected_facts) - 1, -1, -1)
                    if int(selected_facts[index].get("offset_bytes", -1))
                    not in protected_offsets
                ), None)
                if removable is None:
                    selected_evidence.pop()
                else:
                    selected_facts.pop(removable)
                coverage = coverage_for(compact=True)
            elif selected_evidence:
                selected_evidence.pop()
                coverage = coverage_for(compact=True)
            elif selected_facts:
                selected_facts.pop()
                coverage = coverage_for(compact=True)
            elif coverage and not critical_record_indices:
                coverage = {}
            elif rendered_synopsis:
                overflow = _summary_json_chars(
                    rendered_synopsis,
                    selected_facts,
                    selected_evidence,
                    current_record_selection().value if selected_record_rows else None,
                    coverage,
                ) - max_chars
                keep = max(0, len(rendered_synopsis) - max(1, overflow))
                rendered_synopsis = (
                    rendered_synopsis[: max(0, keep - 1)] + "…" if keep else ""
                )
                if coverage:
                    coverage = coverage_for(compact=True)
            else:
                noncritical = [
                    row for row in selected_record_rows
                    if int(row.get("index", -1)) not in critical_record_indices
                ]
                if noncritical:
                    drop = max(
                        noncritical,
                        key=lambda row: (
                            int(row.get("anchor_rank", -1) or -1),
                            int(row.get("index", -1)),
                        ),
                    )
                    selected_record_rows.remove(drop)
                    coverage = coverage_for(compact=True)
                else:
                    # Critical raw records are the only intentional exception
                    # to the shared max_chars budget.
                    break

        if compact_coverage and coverage:
            coverage["evidence_truncated"] = any(
                bool(item.get("truncated"))
                or item.get("fact_complete") is False
                for item in selected_evidence
            )

        records = current_record_selection()
        retained_records = records.value if selected_record_rows else None
        serialized_chars = _summary_json_chars(
            rendered_synopsis,
            selected_facts,
            selected_evidence,
            retained_records,
            coverage,
        )
        return SummaryProjection(
            synopsis=rendered_synopsis,
            key_facts=tuple(selected_facts),
            selected_evidence=tuple(selected_evidence),
            retained_records=retained_records,
            coverage=coverage,
            serialized_chars=serialized_chars,
        )


def ranking_context_from_messages(
    messages: Iterable[Any],
    *,
    prompt: str = "",
) -> str:
    """Return only the current question and terminal assistant text.

    Tool payloads are excluded by consuming ``ordinary_text_parts`` from the
    provider-neutral normalizer.  OpenClaw supplies the current inbound user
    prompt separately from historical ``messages``; when present, that prompt
    is authoritative and stale history must not influence the projection.
    The returned string is request-local.
    """
    current_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if current_prompt:
        return current_prompt
    last_user = ""
    last_user_index = -1
    last_assistant = ""
    last_assistant_index = -1
    for index, message in enumerate(messages):
        role = str(getattr(message, "role", "")).casefold()
        ordinary = getattr(message, "ordinary_text_parts", ())
        text = "\n".join(str(part) for part in ordinary if str(part).strip()).strip()
        if not text:
            continue
        if role == "user":
            last_user = text
            last_user_index = index
        elif role == "assistant":
            last_assistant = text
            last_assistant_index = index
    same_turn_assistant = (
        last_assistant
        if last_assistant_index > last_user_index
        else ""
    )
    return "\n".join(
        value for value in (last_user, same_turn_assistant) if value
    ).strip()
