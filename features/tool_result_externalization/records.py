"""Deterministic raw-record preservation for multi-resolution stubs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from .fragments import (
    best_query_match,
    expanded_query_terms,
    iter_delimited_records,
)


_ERROR_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_])(?:traceback|error|exception|failed|failure|critical|"
    r"fatal|panic|denied|timeout)(?![A-Za-z0-9_])|错误|异常|失败|严重|致命|拒绝|超时)",
    re.IGNORECASE,
)
_STATUS_KEYS = {
    "status", "state", "result", "outcome", "level", "severity",
    "status_code", "statuscode", "http_status", "response_code", "exit_code",
}
_TIME_KEYS = {
    "time", "timestamp", "ts", "date", "datetime", "created_at", "updated_at",
    "event_time", "start_time", "end_time",
}
_LOG_KEYS = {
    "level", "severity", "message", "msg", "logger", "log_level", "loglevel",
}
_VALUE_TOKEN_EDGE_RE = re.compile(r"[A-Za-z0-9_.:/+-]+$")
_VALUE_TOKEN_START_RE = re.compile(r"^[A-Za-z0-9_.:/+-]+")
_TEMPLATE_SLOT = "[]"
_TEMPLATE_DYNAMIC_RE = re.compile(
    r"(?:\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|"
    r"\b[0-9A-HJKMNP-TV-Z]{26}\b|"
    r"(?<![A-Za-z0-9_])(?:failed|failure|pass(?:ed)?|ok|success|error|"
    r"exception|critical|fatal|timeout)(?![A-Za-z0-9_])|"
    r"\d+(?:[.:/+_-]\d+)*)",
    re.IGNORECASE,
)

_QUERY_STOP_TERMS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at",
    "be", "before", "by", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "of", "on", "or",
    "our", "please", "that", "the", "their", "them", "then", "there", "these",
    "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}
_QUERY_CONTROL_TERMS = {
    "artifact", "artifacts", "constraint", "count", "digest", "evidence", "exact",
    "first", "inspect", "last", "limit", "list", "maximum", "minimum", "must",
    "ogmemory", "order", "original", "read", "recall", "required", "search",
    "source", "stub", "summary", "tool", "tools", "total", "verification", "verify",
    "不得", "大小", "必须", "数值", "数量", "约束", "至少", "限制", "至多",
}
_SHORT_QUERY_VALUE_TERMS = {
    "ok", "id",
}
_MAX_QUERY_VALUE_DOCUMENT_FRACTION = 0.25
_MIN_CORPUS_FOR_COMMON_VALUE_FILTER = 8


@dataclass(frozen=True)
class RetainedRecordSelection:
    value: Any
    selected_indices: tuple[int, ...]
    critical_indices: tuple[int, ...]
    total_records: int
    duplicate_records: int
    pattern: str
    serialized_chars: int

    @property
    def omitted_records(self) -> int:
        return max(0, self.total_records - self.duplicate_records - len(self.selected_indices))


@dataclass(frozen=True)
class RecordPreservationConfig:
    """Headroom-compatible deterministic preservation defaults."""

    variance_threshold: float = 2.0
    max_hard_change_points: int = 8
    max_anchor_slots: int = 12
    first_items: int = 3
    last_items: int = 2
    dedup_identical_items: bool = True
    preserve_original_json_structure: bool = True
    include_summaries: bool = False
    template_min_records: int = 3
    template_discovery_max_records: int = 128
    template_discovery_max_record_chars: int = 4_096
    template_discovery_max_chars: int = 65_536
    template_min_savings_ratio: float = 0.15


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_text(value: Any) -> str:
    return value if isinstance(value, str) else _canonical(value)


def _fact_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value).strip()


def _named_scalars(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _named_scalars(child)
            else:
                yield str(key), child
    elif isinstance(value, list):
        for child in value:
            yield from _named_scalars(child)


def _query_value_text(value: Any) -> str:
    """Render only scalar values for record relevance.

    Schema keys such as ``request_id``, ``status``, ``size``, and ``service``
    describe every row in many tool results.  Matching those keys is useful for
    locating a field inside one fragment, but it is not evidence that every
    record is relevant to the current question.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return "\n".join(
            _fact_scalar(child)
            for _key, child in _named_scalars(value)
            if _fact_scalar(child)
        )
    return _fact_scalar(value)


def _query_term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    if re.search(r"[A-Za-z0-9]", term):
        # Treat underscore, slash, colon, and hyphen as token separators so a
        # query for ``spike`` can match the value ``traffic_spike`` while short
        # stop words cannot match inside unrelated words.
        return re.compile(
            rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    return re.compile(escaped, re.IGNORECASE)


def _query_relevance_scores(
    rows: Iterable[dict[str, Any]],
    ranking_context: str,
) -> dict[int, float]:
    """Return request-local, corpus-aware scores for soft record selection.

    Query relevance is deliberately computed from values rather than schema
    keys.  Terms present in a large fraction of the candidate corpus are
    suppressed: they identify the collection (for example ``python`` or
    ``file``), not the few records that answer the current question.
    """
    materialized = [dict(row) for row in rows if isinstance(row, dict)]
    if not materialized or not ranking_context.strip():
        return {}
    _phrase, expanded_terms = expanded_query_terms(ranking_context)
    direct_terms = {
        term.casefold()
        for term in re.findall(
            r"[a-z0-9][a-z0-9_.:/-]{1,}", ranking_context.casefold(),
        )
    }
    terms: list[str] = []
    for raw_term in expanded_terms:
        term = raw_term.strip().casefold()
        if (
            not term
            or term in _QUERY_STOP_TERMS
            or term in _QUERY_CONTROL_TERMS
            or (
                len(term) < 3
                and not term.isdigit()
                and term not in _SHORT_QUERY_VALUE_TERMS
            )
        ):
            continue
        terms.append(term)
    if not terms:
        return {}

    value_texts = {
        int(row.get("index", -1)): _query_value_text(row.get("value"))
        for row in materialized
    }
    term_documents: dict[str, set[int]] = {}
    for term in terms:
        pattern = _query_term_pattern(term)
        matched = {
            index for index, text in value_texts.items()
            if text and pattern.search(text)
        }
        if matched:
            term_documents[term] = matched

    corpus_size = len(value_texts)
    scores: dict[int, float] = defaultdict(float)
    for term, indices in term_documents.items():
        document_fraction = len(indices) / max(1, corpus_size)
        if (
            corpus_size >= _MIN_CORPUS_FOR_COMMON_VALUE_FILTER
            and document_fraction > _MAX_QUERY_VALUE_DOCUMENT_FRACTION
        ):
            continue
        inverse_frequency = math.log((corpus_size + 1) / (len(indices) + 1)) + 1.0
        weight = inverse_frequency * (2.0 if term in direct_terms else 1.0)
        weight += min(len(term), 16) / 16
        for index in indices:
            scores[index] += weight
    return dict(scores)


def record_covers_fact(record: Any, fact: dict[str, Any]) -> bool:
    """Return whether a retained source record already carries one fact."""
    target = str(fact.get("value", ""))
    if not target:
        return False
    label = _normalized_key(fact.get("label", ""))
    for key, value in _named_scalars(record):
        if _normalized_key(key) == label and _fact_scalar(value) == target:
            return True
    if not isinstance(record, str):
        return len(target) >= 4 and target in _record_text(record)
    if record == target or (len(target) >= 4 and target in record):
        return True
    if not label:
        return False
    return bool(re.search(
        rf"(?:^|\b){re.escape(str(fact.get('label', '')))}\s*[:=]\s*"
        rf"[\"']?{re.escape(target)}(?:[\"']?\b|$)",
        record,
        re.IGNORECASE,
    ))


def _extract_template_row(template: str, value: str) -> list[str] | None:
    literals = template.split(_TEMPLATE_SLOT)
    if len(literals) < 2 or not value.startswith(literals[0]):
        return None
    cursor = len(literals[0])
    variables: list[str] = []
    for literal in literals[1:]:
        position = value.find(literal, cursor) if literal else len(value)
        if position < cursor:
            return None
        variables.append(value[cursor:position])
        cursor = position + len(literal)
    if cursor != len(value):
        return None
    rebuilt = [literals[0]]
    for variable, literal in zip(variables, literals[1:]):
        rebuilt.extend((variable, literal))
    return variables if "".join(rebuilt) == value else None


def _template_values(value: Any) -> list[str] | None:
    if not isinstance(value, dict):
        return None
    allowed = {"template", "rows", "unmatched_records", "unmatched_positions"}
    if not {"template", "rows"}.issubset(value) or not set(value).issubset(allowed):
        return None
    template = value.get("template")
    rows = value.get("rows")
    unmatched = value.get("unmatched_records", [])
    positions = value.get("unmatched_positions", [])
    if not isinstance(template, str) or not isinstance(rows, list):
        return None
    if not isinstance(unmatched, list) or not all(isinstance(item, str) for item in unmatched):
        return None
    if not isinstance(positions, list) or len(positions) != len(unmatched) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in positions
    ):
        return None
    slot_count = template.count(_TEMPLATE_SLOT)
    if slot_count < 1:
        return None
    literals = template.split(_TEMPLATE_SLOT)
    matched: list[str] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != slot_count or not all(
            isinstance(item, str) for item in row
        ):
            return None
        parts: list[str] = [literals[0]]
        for item, literal in zip(row, literals[1:]):
            parts.extend((item, literal))
        matched.append("".join(parts))
    total = len(matched) + len(unmatched)
    if positions != sorted(set(positions)) or any(
        position < 0 or position >= total for position in positions
    ):
        return None
    unmatched_by_position = dict(zip(positions, unmatched))
    expanded: list[str] = []
    matched_index = 0
    for position in range(total):
        if position in unmatched_by_position:
            expanded.append(unmatched_by_position[position])
        elif matched_index < len(matched):
            expanded.append(matched[matched_index])
            matched_index += 1
        else:
            return None
    return expanded if matched_index == len(matched) else None


def _distributed_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    return list(dict.fromkeys(
        round(index * (count - 1) / max(1, limit - 1))
        for index in range(limit)
    ))


def _template_signature(value: str) -> str:
    return _TEMPLATE_DYNAMIC_RE.sub(_TEMPLATE_SLOT, value)


def _factor_template_rows(
    values: list[Any],
    config: RecordPreservationConfig | None = None,
) -> dict[str, Any] | None:
    """Factor repeated literal fragments into one reversible text template."""
    config = config or RecordPreservationConfig()
    if len(values) < config.template_min_records or not all(
        isinstance(value, str) for value in values
    ):
        return None
    rows = [str(value) for value in values]
    if any(_TEMPLATE_SLOT in row for row in rows):
        return None
    discovery_rows: list[str] = []
    discovery_chars = 0
    for index in _distributed_indices(
        len(rows), config.template_discovery_max_records,
    ):
        row = rows[index]
        if len(row) > config.template_discovery_max_record_chars:
            continue
        if discovery_chars + len(row) > config.template_discovery_max_chars:
            continue
        discovery_rows.append(row)
        discovery_chars += len(row)
    if len(discovery_rows) < config.template_min_records:
        return None

    signature_groups: dict[str, list[str]] = {}
    for row in discovery_rows:
        signature_groups.setdefault(_template_signature(row), []).append(row)
    dominant_rows = max(signature_groups.values(), key=len)
    if len(dominant_rows) >= config.template_min_records:
        discovery_rows = dominant_rows

    signature_template = _template_signature(discovery_rows[0])
    template = (
        signature_template
        if signature_template.count(_TEMPLATE_SLOT) > 0
        and all(
            _extract_template_row(signature_template, row) is not None
            for row in discovery_rows
        )
        else None
    )
    if template is None:
        base = discovery_rows[0]
        if not base:
            return None
        common_ngrams = {
            base[index:index + 4]
            for index in range(max(0, len(base) - 3))
        }
        for row in discovery_rows[1:]:
            if len(common_ngrams) < 2:
                return None
            common_ngrams.intersection_update(
                row[index:index + 4]
                for index in range(max(0, len(row) - 3))
            )
        if len(common_ngrams) < 2:
            return None
        common = [True] * len(base)
        for row in discovery_rows[1:]:
            matched = [False] * len(base)
            for block in SequenceMatcher(None, base, row, autojunk=False).get_matching_blocks():
                for index in range(block.a, block.a + block.size):
                    if index < len(matched):
                        matched[index] = True
            common = [left and right for left, right in zip(common, matched)]

        runs: list[tuple[int, int, str]] = []
        start: int | None = None
        for index, is_common in enumerate([*common, False]):
            if is_common and start is None:
                start = index
            elif not is_common and start is not None:
                text = base[start:index]
                if start > 0:
                    text = _VALUE_TOKEN_START_RE.sub("", text)
                if index < len(base):
                    text = _VALUE_TOKEN_EDGE_RE.sub("", text)
                if len(text) >= 2:
                    runs.append((start, index, text))
                start = None
        literals = [text for _start, _end, text in runs]
        if not literals or sum(map(len, literals)) < 8:
            return None

        extracted: list[list[str]] = []
        for row in discovery_rows:
            cursor = 0
            gaps: list[str] = []
            for literal in literals:
                position = row.find(literal, cursor)
                if position < 0:
                    return None
                gaps.append(row[cursor:position])
                cursor = position + len(literal)
            gaps.append(row[cursor:])
            extracted.append(gaps)

        variable_gaps = [
            index
            for index in range(len(literals) + 1)
            if any(row[index] for row in extracted)
        ]
        if not variable_gaps:
            return None
        template_parts: list[str] = []
        for index, literal in enumerate([*literals, ""]):
            if index in variable_gaps:
                template_parts.append(_TEMPLATE_SLOT)
            template_parts.append(literal)
        template = "".join(template_parts)
    factorized = {"template": template, "rows": []}
    unmatched: list[str] = []
    unmatched_positions: list[int] = []
    for position, row in enumerate(rows):
        variables = _extract_template_row(template, row)
        if variables is None:
            unmatched.append(row)
            unmatched_positions.append(position)
        else:
            factorized["rows"].append(variables)
    if len(factorized["rows"]) < config.template_min_records:
        return None
    if unmatched:
        factorized["unmatched_records"] = unmatched
        factorized["unmatched_positions"] = unmatched_positions
    if _template_values(factorized) != rows:
        return None
    # UTF-8 bytes are a closer cheap proxy for model-token cost than Python's
    # Unicode code-point count, especially for repeated CJK literals.
    original_cost = len(_canonical(rows).encode("utf-8"))
    factorized_cost = len(_canonical(factorized).encode("utf-8"))
    savings_ratio = (
        (original_cost - factorized_cost) / original_cost
        if original_cost
        else 0.0
    )
    return factorized if savings_ratio >= config.template_min_savings_ratio else None


def _query_matches(value: Any, ranking_context: str) -> bool:
    if not ranking_context.strip():
        return False
    match = best_query_match(_record_text(value), ranking_context)
    return bool(
        match is not None
        and (match.exact_phrase_hits > 0 or match.score >= 12.0)
    )


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value).casefold())


def _find_primary_array(value: Any, path: tuple[Any, ...] = ()) -> tuple[tuple[Any, ...], list] | None:
    candidates: list[tuple[tuple[Any, ...], list]] = []
    if isinstance(value, list):
        candidates.append((path, value))
    elif isinstance(value, dict):
        for key, child in value.items():
            found = _find_primary_array(child, (*path, key))
            if found is not None:
                candidates.append(found)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (len(item[1]), -len(item[0])))


def _json_records(text: str) -> tuple[list[Any], dict[str, Any]] | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    found = _find_primary_array(value)
    if found is None:
        return None
    path, records = found
    return list(records), {"kind": "json", "path": list(path)}


def _line_records(text: str, detected_type: str) -> tuple[list[Any], dict[str, Any]] | None:
    if detected_type in {"csv", "tsv"}:
        delimiter = "\t" if detected_type == "tsv" else ","
        try:
            logical = [
                record
                for record in iter_delimited_records(text, delimiter)
                if record.raw.strip() and any(field.strip() for field in record.fields)
            ]
        except csv.Error:
            logical = []
        if len(logical) >= 2:
            record_separator = text[
                logical[0].char_end:logical[1].char_start
            ]
            if record_separator not in {"\n", "\r\n", "\r"}:
                record_separator = "\n"
            return [record.raw for record in logical[1:]], {
                "kind": detected_type,
                "header": logical[0].raw,
                "delimiter": delimiter,
                "record_separator": record_separator,
                "_record_spans": [
                    (record.char_start, record.char_end)
                    for record in logical[1:]
                ],
            }

    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    log_like = sum(
        bool(re.match(
            r"^(?:"
            r"\d{4}-\d{2}-\d{2}"
            r"|\[[A-Z]+\]"
            r"|\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*"
            r"(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b"
            r"|DEBUG|INFO|WARN|WARNING|ERROR|FATAL"
            r")",
            line,
            re.I,
        ))
        for line in lines
    )
    return lines, {"kind": "log" if log_like >= max(1, len(lines) // 5) else "text"}


def _numeric_fields(records: list[Any]) -> dict[str, list[tuple[int, float]]]:
    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                values[str(key)].append((index, number))
    return values


def _structural_outliers(records: list[Any]) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = defaultdict(set)
    dict_rows = [(index, record) for index, record in enumerate(records) if isinstance(record, dict)]
    if len(dict_rows) < 5:
        return reasons
    key_counts = Counter(str(key) for _index, row in dict_rows for key in row)
    rare_limit = max(1, math.floor(len(dict_rows) * 0.1))
    for index, row in dict_rows:
        if any(key_counts[str(key)] <= rare_limit for key in row):
            reasons[index].add("rare_field")

    for key in {str(key) for _index, row in dict_rows for key in row}:
        normalized = _normalized_key(key)
        if normalized not in _STATUS_KEYS and not normalized.endswith("status"):
            continue
        values = [
            (index, _canonical(row[key]))
            for index, row in dict_rows
            if key in row
        ]
        counts = Counter(value for _index, value in values)
        if len(counts) < 2:
            continue
        common = max(counts.values())
        for index, value in values:
            if counts[value] <= rare_limit and counts[value] < common:
                reasons[index].add("rare_status")
    return reasons


def _numeric_outliers(
    records: list[Any],
    threshold: float = 2.0,
    max_change_points: int = 8,
) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = defaultdict(set)
    change_candidates: list[tuple[float, float, str, int, int]] = []
    for field, values in _numeric_fields(records).items():
        if len(values) < 5:
            continue
        numbers = [value for _index, value in values]
        low_cardinality = len(set(numbers)) <= 5
        deviation = statistics.pstdev(numbers)
        if deviation > 0 and not low_cardinality:
            mean = statistics.fmean(numbers)
            for index, value in values:
                if abs(value - mean) > threshold * deviation:
                    reasons[index].add(f"numeric_anomaly:{field}")

        deltas = [abs(values[pos][1] - values[pos - 1][1]) for pos in range(1, len(values))]
        if not deltas:
            continue
        delta_mean = statistics.fmean(deltas)
        delta_deviation = statistics.pstdev(deltas)
        for pos, delta in enumerate(deltas, start=1):
            anomalous_delta = delta_deviation > 0 and delta > delta_mean + threshold * delta_deviation
            if not anomalous_delta:
                continue
            score = (delta - delta_mean) / delta_deviation
            change_candidates.append((
                score,
                delta,
                field,
                values[pos - 1][0],
                values[pos][0],
            ))

    for _score, _delta, field, previous_index, current_index in sorted(
        change_candidates,
        key=lambda item: (-item[0], -item[1], item[2], item[4]),
    )[:max(0, max_change_points)]:
        reasons[previous_index].add(f"change_point:{field}")
        reasons[current_index].add(f"change_point:{field}")
    return reasons


def _detect_pattern(
    records: list[Any],
    layout: dict[str, Any],
    tool_name: str | None,
) -> str:
    tool = str(tool_name or "").casefold()
    if any(token in tool for token in ("search", "grep", "query", "find")):
        return "search"
    if any(token in tool for token in ("log", "tail", "journal")):
        return "log"
    if layout.get("kind") == "log":
        return "log"
    dict_rows = [record for record in records if isinstance(record, dict)]
    if dict_rows:
        time_rows = sum(
            any(_normalized_key(key) in _TIME_KEYS for key in row)
            for row in dict_rows
        )
        numeric_rows = sum(
            any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row.values())
            for row in dict_rows
        )
        log_rows = sum(
            any(_normalized_key(key) in _LOG_KEYS for key in row)
            for row in dict_rows
        )
        if log_rows >= max(1, len(dict_rows) * 3 // 4):
            return "log"
        if time_rows >= max(1, len(dict_rows) * 3 // 4) and numeric_rows >= max(1, len(dict_rows) // 2):
            return "time_series"
        if any(
            _normalized_key(key) in {"score", "rank", "relevance", "distance"}
            for row in dict_rows
            for key in row
        ):
            return "search"
    return "array"


def _anchor_order(
    count: int,
    pattern: str,
    config: RecordPreservationConfig,
) -> list[int]:
    if count <= 0:
        return []
    guaranteed = [
        *range(min(config.first_items, count)),
        *range(max(0, count - config.last_items), count),
    ]
    order = list(dict.fromkeys(guaranteed))
    remaining = max(0, config.max_anchor_slots - len(order))
    if remaining <= 0:
        return order
    if pattern == "log":
        candidates = list(range(count - 1, -1, -1))
    elif pattern == "search":
        front = list(range(count))
        back = list(range(count - 1, -1, -1))
        candidates = []
        front_slots = max(1, round(remaining * 0.75))
        candidates.extend(front[:front_slots])
        candidates.extend(back[: max(1, remaining - front_slots)])
    else:
        slots = min(count, config.max_anchor_slots)
        candidates = [
            round(index * (count - 1) / max(1, slots - 1))
            for index in range(slots)
        ]
        if pattern == "time_series":
            candidates = [
                value
                for pair in zip(candidates, reversed(candidates))
                for value in pair
            ]
    for index in candidates:
        if index not in order:
            order.append(index)
        if len(order) >= config.max_anchor_slots:
            break
    return order


def extract_record_candidates(
    text: str,
    *,
    detected_type: str,
    tool_name: str | None,
    ranking_context: str,
    config: RecordPreservationConfig | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    config = config or RecordPreservationConfig()
    extracted = _json_records(text) if detected_type == "json" else None
    if extracted is None:
        extracted = _line_records(text, detected_type)
    if extracted is None:
        return (), None
    records, layout = extracted
    record_spans = layout.pop("_record_spans", None)
    # Match SmartCrusher's default minimum: tiny collections are not sampled
    # into a stub because one large scalar/record could otherwise echo the
    # complete externalized payload.
    if len(records) < 5:
        return (), None
    pattern = _detect_pattern(records, layout, tool_name)
    layout = {
        **layout,
        "pattern": pattern,
        "record_count": len(records),
        "include_summaries": config.include_summaries,
        "preserve_original_structure": config.preserve_original_json_structure,
    }
    canonical_records = [_canonical(record) for record in records]
    record_texts = [
        record if isinstance(record, str) else canonical_records[index]
        for index, record in enumerate(records)
    ]

    reasons: dict[int, set[str]] = defaultdict(set)
    for index in range(len(records)):
        if _ERROR_RE.search(record_texts[index]):
            reasons[index].add("error_keyword")
        if _query_matches(record_texts[index], ranking_context):
            reasons[index].add("query_match")
    for source in (
        _structural_outliers(records),
        _numeric_outliers(
            records,
            config.variance_threshold,
            config.max_hard_change_points,
        ),
    ):
        for index, values in source.items():
            reasons[index].update(values)

    anchor_rank = {
        index: rank
        for rank, index in enumerate(_anchor_order(len(records), pattern, config))
    }
    first_by_content: dict[str, int] = {}
    merged_reasons: dict[int, set[str]] = defaultdict(set)
    merged_anchor_rank: dict[int, int] = {}
    duplicate_records = 0
    for index in range(len(records)):
        content = canonical_records[index]
        identity = content if config.dedup_identical_items else f"{content}\0{index}"
        first = first_by_content.setdefault(identity, index)
        if first != index:
            duplicate_records += 1
        merged_reasons[first].update(reasons.get(index, ()))
        if index in anchor_rank:
            merged_anchor_rank[first] = min(
                anchor_rank[index],
                merged_anchor_rank.get(first, anchor_rank[index]),
            )

    candidates: list[dict[str, Any]] = []
    for _identity, index in first_by_content.items():
        content = canonical_records[index]
        record_reasons = sorted(merged_reasons.get(index, ()))
        if not record_reasons and index not in merged_anchor_rank:
            continue
        candidates.append({
            "index": index,
            "value": records[index],
            **(
                {"raw_json": json.dumps(records[index], ensure_ascii=False, separators=(",", ":"))}
                if layout.get("kind") == "json"
                else {}
            ),
            "critical": bool(record_reasons),
            "reasons": record_reasons,
            "anchor_rank": merged_anchor_rank.get(index),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            **(
                {
                    "source_start_char": int(record_spans[index][0]),
                    "source_end_char": int(record_spans[index][1]),
                }
                if record_spans is not None and index < len(record_spans)
                else {}
            ),
        })
    layout["duplicate_records"] = duplicate_records
    return tuple(sorted(candidates, key=lambda item: int(item["index"]))), layout


def render_retained_records(
    rows: Iterable[dict[str, Any]],
    layout: dict[str, Any] | None,
    config: RecordPreservationConfig | None = None,
) -> Any:
    selected = [dict(row) for row in sorted(rows, key=lambda item: int(item["index"]))]
    values = [
        json.loads(row["raw_json"])
        if (layout or {}).get("kind") == "json" and isinstance(row.get("raw_json"), str)
        else row["value"]
        for row in selected
    ]
    factorized = _factor_template_rows(values, config)
    if not layout:
        return factorized or values
    kind = layout.get("kind")
    if kind == "json":
        result: Any = factorized or values
        for key in reversed(list(layout.get("path") or [])):
            result = {str(key): result}
        return result
    if kind in {"csv", "tsv"}:
        lines = [str(layout.get("header", "")), *(str(value) for value in values)]
        separator = str(layout.get("record_separator") or "\n")
        return separator.join(line for line in lines if line)
    return factorized or values


def retained_record_values(value: Any, layout: dict[str, Any] | None) -> list[Any] | None:
    if not layout:
        expanded = _template_values(value)
        if expanded is not None:
            return expanded
        return list(value) if isinstance(value, list) else None
    kind = layout.get("kind")
    if kind == "json":
        current = value
        for key in list(layout.get("path") or []):
            if not isinstance(current, dict) or str(key) not in current:
                return None
            current = current[str(key)]
        expanded = _template_values(current)
        if expanded is not None:
            return expanded
        return list(current) if isinstance(current, list) else None
    if kind in {"csv", "tsv"}:
        if not isinstance(value, str):
            return None
        delimiter = str(layout.get("delimiter") or ("\t" if kind == "tsv" else ","))
        try:
            records = [
                record
                for record in iter_delimited_records(value, delimiter)
                if record.raw.strip() and any(field.strip() for field in record.fields)
            ]
        except csv.Error:
            return None
        if not records or records[0].raw != str(layout.get("header", "")):
            return None
        return [record.raw for record in records[1:]]
    expanded = _template_values(value)
    if expanded is not None:
        return expanded
    return list(value) if isinstance(value, list) else None


def is_hard_critical_record(row: dict[str, Any]) -> bool:
    """Return whether one attested candidate must survive every projection.

    Trustworthy reason lists distinguish persistent structural/error/anomaly
    evidence from request-local ``query_match`` relevance.  Legacy metadata
    without such a list keeps the prior fail-safe ``critical`` behavior.
    """
    reasons = row.get("reasons")
    if isinstance(reasons, (list, tuple)) and all(
        isinstance(reason, str) for reason in reasons
    ):
        return any(reason != "query_match" for reason in reasons)
    return row.get("critical") is True


def select_retained_records(
    candidates: Iterable[dict[str, Any]],
    layout: dict[str, Any] | None,
    *,
    ranking_context: str,
    max_chars: int,
    config: RecordPreservationConfig | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    rows = [dict(item) for item in candidates if isinstance(item, dict)]
    hard_critical_indices: set[int] = set()
    for row in rows:
        index = int(row.get("index", -1))
        if is_hard_critical_record(row):
            hard_critical_indices.add(index)

    selected = [
        row for row in rows
        if int(row.get("index", -1)) in hard_critical_indices
    ]
    query_scores = _query_relevance_scores(rows, ranking_context)
    query_rows = sorted(
        (
            row for row in rows
            if int(row.get("index", -1)) not in hard_critical_indices
            and int(row.get("index", -1)) in query_scores
        ),
        key=lambda item: (
            -query_scores[int(item.get("index", -1))],
            (
                int(item["anchor_rank"])
                if item.get("anchor_rank") is not None
                else 1 << 30
            ),
            int(item.get("index", -1)),
        ),
    )
    # Request-local relevance is soft: add the best discriminative values only
    # while the retained-record projection remains within the shared budget.
    # Structural/error/anomaly/legacy rows above remain the sole intentional
    # over-budget exception.
    for row in query_rows:
        candidate = [*selected, row]
        rendered = render_retained_records(candidate, layout, config)
        if len(_canonical(rendered)) <= max_chars:
            selected.append(row)

    selected_indices = {int(row.get("index", -1)) for row in selected}
    anchors = sorted(
        (
            row for row in rows
            if int(row.get("index", -1)) not in selected_indices
            and row.get("anchor_rank") is not None
        ),
        key=lambda item: (int(item["anchor_rank"]), int(item["index"])),
    )
    for row in anchors:
        candidate = [*selected, row]
        rendered = render_retained_records(candidate, layout, config)
        if len(_canonical(rendered)) <= max_chars:
            selected.append(row)
    return (
        sorted(selected, key=lambda item: int(item["index"])),
        hard_critical_indices,
    )


def record_selection(
    rows: Iterable[dict[str, Any]],
    critical_indices: set[int],
    layout: dict[str, Any] | None,
    config: RecordPreservationConfig | None = None,
) -> RetainedRecordSelection:
    selected = sorted((dict(row) for row in rows), key=lambda item: int(item["index"]))
    rendered = render_retained_records(selected, layout, config)
    return RetainedRecordSelection(
        value=rendered,
        selected_indices=tuple(int(row["index"]) for row in selected),
        critical_indices=tuple(
            int(row["index"]) for row in selected if int(row["index"]) in critical_indices
        ),
        total_records=int((layout or {}).get("record_count", len(selected))),
        duplicate_records=int((layout or {}).get("duplicate_records", 0)),
        pattern=str((layout or {}).get("pattern", "array")),
        serialized_chars=len(_canonical(rendered)),
    )
