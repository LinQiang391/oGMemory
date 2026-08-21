"""Locate with coarse chunks, then return exact structural fragments."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import math
import re
import time
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator


_LOG_LINE_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}|\[[A-Z]+\]|(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\b)",
    re.IGNORECASE,
)
_PY_DEF_RE = re.compile(r"^(?P<indent>[ \t]*)(?:async\s+def|def|class)\s+[A-Za-z_$][\w$]*")
_BRACE_DEF_RE = re.compile(
    r"(?:\bfunction\s+[A-Za-z_$][\w$]*|\bfunc\s+(?:\([^)]*\)\s*)?[A-Za-z_$][\w$]*|"
    r"\b(?:public|private|protected|static|async|export|const|let|var|[A-Za-z_$][\w$<>]*)\s+"
    r"[A-Za-z_$][\w$]*\s*\([^;]*\)\s*\{)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryMatch:
    char_start: int
    char_end: int
    score: float
    exact_phrase_hits: int


@dataclass(frozen=True)
class QueryTerm:
    value: str
    kind: str
    direct: bool
    category: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    phrase: str
    terms: tuple[QueryTerm, ...]
    categories: tuple[str, ...]


@dataclass(frozen=True)
class QueryOccurrence:
    value: str
    kind: str
    char_start: int
    char_end: int
    exact_phrase: bool = False


@dataclass(frozen=True)
class DelimitedRecord:
    """One parsed CSV/TSV record plus its exact raw-text character span."""

    char_start: int
    char_end: int
    raw: str
    fields: tuple[str, ...]


_QUERY_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at",
    "be", "before", "by", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "of", "on", "or",
    "our", "please", "that", "the", "their", "them", "then", "there", "these",
    "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}
_QUERY_CATEGORY_ALIASES = {
    "error": (
        "error", "exception", "traceback", "failed", "failure", "panic", "fatal",
        "错误", "异常", "失败", "报错", "故障",
    ),
    "status": (
        "status", "status_code", "http_status", "response_code", "exit_code",
        "状态码", "返回码", "响应码",
    ),
    "id": (
        "request_id", "trace_id", "correlation_id", "session_id", "task_id",
        "job_id", "uuid", "identifier", "标识", "编号", "请求号",
    ),
    "constraint": (
        "must", "required", "limit", "minimum", "maximum", "constraint",
        "必须", "不得", "限制", "约束", "至少", "至多",
    ),
    "number": (
        "count", "total", "size", "bytes", "duration", "timeout", "limit",
        "数量", "数值", "大小", "多少",
    ),
}
_QUERY_CATEGORY_INDICATORS = {
    "error": ("error", "exception", "错误", "异常", "失败", "报错", "故障"),
    "status": ("status", "code", "状态码", "返回码", "响应码"),
    "id": (" id", "identifier", "uuid", "标识", "编号", "请求号", "哪个请求"),
    "constraint": ("must", "required", "constraint", "限制", "约束", "必须", "不能"),
    "number": ("count", "total", "how many", "数量", "数值", "多少", "大小"),
}
_QUOTED_ENTITY_RE = re.compile(r'''["'`](?P<value>[^"'`\r\n]{1,256})["'`]''')
_PRECISE_ENTITY_RES = (
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I),
    re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b"),
    re.compile(r"https?://[^\s\]\[{}()<>\"']+", re.I),
    re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:)?[/\\][A-Za-z0-9_.~+%:@-]+(?:[/\\][A-Za-z0-9_.~+%:@-]+)+"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[_:/.-][A-Za-z0-9]+)+(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:ms|s|m|h|kb|mb|gb|%|px)?(?![A-Za-z0-9_.])", re.I),
)
_ORDINARY_QUERY_TERM_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{1,}|"
    r"[\u3040-\u30ff\u31f0-\u31ffー]{2,}|"
    r"[\u0400-\u04ff]{2,}|"
    r"[\uac00-\ud7af]{2,}"
)
_CRITICAL_CATEGORY_PATTERNS = {
    "error": re.compile(
        r"(?<![A-Za-z0-9_])(?:traceback|error|exception|failed|failure|critical|"
        r"fatal|panic|denied|timeout)(?![A-Za-z0-9_])|"
        r"\b[A-Za-z][A-Za-z0-9_]*(?:Error|Exception)\b|"
        r"错误|异常|失败|严重|致命|拒绝|超时",
        re.I,
    ),
    "constraint": re.compile(
        r"\b(?:must|required|cannot|can't|should not|at least|at most|maximum|"
        r"minimum|limited?|only)\b|必须|不得|不能|不可|至少|至多|最多|最少|限制|仅限|只能",
        re.I,
    ),
    "status": re.compile(
        r"(?:\b(?:status(?:_code)?|http_status|response_code|exit_code|code)\s*[:=]\s*"
        r"(?:[45]\d{2}|failed|failure|error|denied|timeout)\b|"
        r"\b(?:failed|failure|denied|timeout)\b)",
        re.I,
    ),
}


def _literal_pattern(value: str) -> re.Pattern[str]:
    escaped = re.escape(value)
    if re.search(r"[A-Za-z0-9]", value):
        return re.compile(
            rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    return re.compile(escaped, re.IGNORECASE)


def build_query_plan(query: str, *, max_terms: int = 64) -> QueryPlan:
    """Extract a full phrase, ordinary terms, precise entities, and category aliases."""
    phrase = query.strip()
    lowered = phrase.casefold()
    categories = tuple(
        category
        for category, markers in _QUERY_CATEGORY_INDICATORS.items()
        if any(marker in lowered for marker in markers)
    )
    selected: dict[str, QueryTerm] = {}

    def add(value: str, kind: str, direct: bool, category: str | None = None) -> None:
        normalized = value.strip().casefold()
        if not normalized or len(normalized) > 256:
            return
        current = selected.get(normalized)
        priority = {"category": 0, "keyword": 1, "entity": 2}
        if current is None or priority.get(kind, 0) > priority.get(current.kind, 0):
            selected[normalized] = QueryTerm(value.strip(), kind, direct, category)

    for match in _QUOTED_ENTITY_RE.finditer(phrase):
        add(match.group("value"), "entity", True)
    for pattern in _PRECISE_ENTITY_RES:
        for match in pattern.finditer(phrase):
            add(match.group(0), "entity", True)
    for match in _ORDINARY_QUERY_TERM_RE.finditer(phrase):
        value = match.group(0)
        if value.casefold() not in _QUERY_STOPWORDS:
            add(value, "keyword", True)
    for category in categories:
        for alias in _QUERY_CATEGORY_ALIASES[category]:
            add(alias, "category", False, category)

    ordered = sorted(
        selected.values(),
        key=lambda term: (
            -int(term.kind == "entity"),
            -int(term.direct),
            -len(term.value),
            term.value.casefold(),
        ),
    )[:max_terms]
    return QueryPlan(phrase=phrase, terms=tuple(ordered), categories=categories)


def query_occurrences(
    content: str,
    plan: QueryPlan,
    *,
    max_positions_per_term: int = 4,
    max_phrase_positions: int = 4,
    deadline: float | None = None,
) -> tuple[QueryOccurrence, ...]:
    """Keep bounded positions for every query term instead of one winner per chunk."""
    occurrences: list[QueryOccurrence] = []
    if plan.phrase:
        for match in islice(
            _literal_pattern(plan.phrase).finditer(content),
            max_phrase_positions,
        ):
            if deadline is not None and time.monotonic() >= deadline:
                return tuple(occurrences)
            occurrences.append(QueryOccurrence(
                plan.phrase, "phrase", match.start(), match.end(), True,
            ))
    for term in plan.terms:
        if deadline is not None and time.monotonic() >= deadline:
            break
        for match in islice(
            _literal_pattern(term.value).finditer(content),
            max_positions_per_term,
        ):
            if deadline is not None and time.monotonic() >= deadline:
                return tuple(occurrences)
            occurrences.append(QueryOccurrence(
                term.value, term.kind, match.start(), match.end(), False,
            ))
    return tuple(occurrences)


def _document_length(content: str) -> int:
    return max(1, len(re.findall(
        r"[A-Za-z0-9_]+|[\u3040-\u30ff\u31f0-\u31ffー]+|"
        r"[\u0400-\u04ff]+|[\uac00-\ud7af]+|[\u3400-\u9fff]",
        content,
    )))


def _minimum_position_window(position_groups: list[list[int]]) -> int | None:
    if len(position_groups) < 2:
        return None
    events = sorted(
        (position, group_index)
        for group_index, positions in enumerate(position_groups)
        for position in positions
    )
    counts: dict[int, int] = {}
    covered = 0
    left = 0
    best: int | None = None
    for right, (right_position, right_group) in enumerate(events):
        counts[right_group] = counts.get(right_group, 0) + 1
        if counts[right_group] == 1:
            covered += 1
        while covered == len(position_groups) and left <= right:
            left_position, left_group = events[left]
            width = max(0, right_position - left_position)
            best = width if best is None else min(best, width)
            counts[left_group] -= 1
            if counts[left_group] == 0:
                covered -= 1
            left += 1
    return best


def rank_structural_fragments(
    fragments: list[dict[str, Any]],
    plan: QueryPlan,
    *,
    deadline: float | None = None,
    max_positions_per_term: int = 16,
) -> list[dict[str, Any]]:
    """Rank one global structural-fragment pool with BM25 and bounded bonuses."""
    if not fragments:
        return []
    term_patterns = {
        term.value.casefold(): _literal_pattern(term.value)
        for term in plan.terms
    }
    phrase_pattern = _literal_pattern(plan.phrase) if plan.phrase else None
    documents: list[dict[str, Any]] = []
    document_frequency = {key: 0 for key in term_patterns}
    for fragment in fragments:
        if deadline is not None and time.monotonic() >= deadline:
            break
        content = str(fragment.get("content", ""))
        matches: dict[str, tuple[int, tuple[int, ...]]] = {}
        for key, pattern in term_patterns.items():
            if deadline is not None and time.monotonic() >= deadline:
                break
            frequency = 0
            positions: list[int] = []
            for match in pattern.finditer(content):
                frequency += 1
                if len(positions) < max_positions_per_term:
                    positions.append(match.start())
            if frequency:
                matches[key] = (frequency, tuple(positions))
                document_frequency[key] += 1
        phrase_hits = 0
        phrase_first: tuple[int, int] | None = None
        if phrase_pattern is not None:
            for match in phrase_pattern.finditer(content):
                phrase_hits += 1
                if phrase_first is None:
                    phrase_first = (match.start(), match.end())
        documents.append({
            "fragment": fragment,
            "content": content,
            "matches": matches,
            "phrase_hits": phrase_hits,
            "phrase_first": phrase_first,
            "length": _document_length(content),
        })

    document_count = len(documents)
    average_length = sum(doc["length"] for doc in documents) / max(1, document_count)
    idf = {
        key: math.log(
            1.0 + (document_count - frequency + 0.5) / (frequency + 0.5),
        )
        for key, frequency in document_frequency.items()
        if frequency
    }
    direct_terms = [term for term in plan.terms if term.direct]
    ranked: list[dict[str, Any]] = []
    for document in documents:
        if deadline is not None and time.monotonic() >= deadline:
            break
        fragment = dict(document["fragment"])
        matches = document["matches"]
        score = 0.0
        best_anchor: tuple[float, int, int] | None = None
        covered_direct: list[QueryTerm] = []
        direct_positions: list[list[int]] = []
        entity_hits = 0
        max_direct_idf = 0.0
        for term in plan.terms:
            key = term.value.casefold()
            term_stats = matches.get(key)
            if term_stats is None:
                continue
            frequency, positions = term_stats
            weight = 2.4 if term.kind == "entity" else (1.0 if term.direct else 0.35)
            length_norm = 1.2 * (
                1.0 - 0.75 + 0.75 * document["length"] / max(1.0, average_length)
            )
            saturated_tf = frequency * 2.2 / (frequency + length_norm)
            term_score = idf.get(key, 0.0) * saturated_tf * weight
            score += term_score
            anchor_priority = term_score + (8.0 if term.kind == "entity" else 0.0)
            first_position = positions[0]
            anchor = (
                anchor_priority,
                first_position,
                first_position + len(term.value),
            )
            if best_anchor is None or anchor[0] > best_anchor[0]:
                best_anchor = anchor
            if term.direct:
                covered_direct.append(term)
                direct_positions.append(list(positions))
                max_direct_idf = max(max_direct_idf, idf.get(key, 0.0))
            if term.kind == "entity":
                entity_hits += 1

        phrase_hits = int(document["phrase_hits"])
        if phrase_hits and document["phrase_first"] is not None:
            phrase_start, phrase_end = document["phrase_first"]
            best_anchor = (1_000.0, phrase_start, phrase_end)
            score += 40.0 + min(phrase_hits, 4) * 8.0
        if direct_terms:
            coverage = len(covered_direct) / len(direct_terms)
            score += len(covered_direct) * 0.5 + coverage * 1.5
        proximity_window = _minimum_position_window(direct_positions)
        if proximity_window is not None:
            score += (
                2.0 * max(1, len(direct_positions) - 1)
                / (1.0 + proximity_window / 64.0)
            )
        score += entity_hits * 4.0

        fragment_categories = {
            category
            for category, pattern in _CRITICAL_CATEGORY_PATTERNS.items()
            if pattern.search(document["content"])
        }
        score += len(fragment_categories.intersection(plan.categories)) * 4.0
        if fragment_categories and covered_direct:
            # A rare query hit inside an error/constraint/abnormal-status
            # fragment must not lose to a nearby pair of very common terms.
            score += min(6.0, max_direct_idf * 4.0)
        score += min(3.0, max(0.0, float(fragment.get("_metadata_score", 0.0))) * 0.02)

        if best_anchor is None:
            continue
        anchor_start = best_anchor[1]
        anchor_end = best_anchor[2]
        fragment["match_offset_bytes"] = int(fragment.get("offset_bytes", 0)) + len(
            document["content"][:anchor_start].encode("utf-8")
        )
        fragment["match_length_bytes"] = max(
            1,
            len(document["content"][anchor_start:anchor_end].encode("utf-8")),
        )
        fragment["relevance_score"] = round(score, 6)
        fragment["_exact_phrase_hits"] = phrase_hits
        fragment["_covered_query_terms"] = len(covered_direct)
        ranked.append(fragment)

    return sorted(
        ranked,
        key=lambda item: (
            -int(item.get("_exact_phrase_hits", 0) > 0),
            -float(item.get("relevance_score", 0.0)),
            -int(item.get("_covered_query_terms", 0)),
            str(item.get("ref", "")),
            int(item.get("offset_bytes", 0)),
        ),
    )


def expanded_query_terms(query: str) -> tuple[str, tuple[str, ...]]:
    phrase = query.strip()
    lowered = phrase.casefold()
    terms = set(re.findall(r"[a-z0-9][a-z0-9_.:/-]{1,}", lowered))
    aliases = {
        "error": ("error", "exception", "traceback", "failed", "failure", "panic", "fatal", "错误", "异常", "失败", "报错", "故障"),
        "status": ("status", "status_code", "http_status", "response_code", "exit_code", "状态码", "返回码", "响应码"),
        "id": ("request_id", "trace_id", "correlation_id", "session_id", "task_id", "job_id", "uuid", "identifier", "标识", "编号", "请求号"),
        "constraint": ("must", "required", "limit", "minimum", "maximum", "constraint", "必须", "不得", "限制", "约束", "至少", "至多"),
        "number": ("count", "total", "size", "bytes", "duration", "timeout", "limit", "数量", "数值", "大小", "多少"),
    }
    indicators = {
        "error": ("error", "exception", "错误", "异常", "失败", "报错", "故障"),
        "status": ("status", "code", "状态码", "返回码", "响应码"),
        "id": (" id", "identifier", "uuid", "标识", "编号", "请求号", "哪个请求"),
        "constraint": ("must", "required", "constraint", "限制", "约束", "必须", "不能"),
        "number": ("count", "total", "how many", "数量", "数值", "多少", "大小"),
    }
    for kind, markers in indicators.items():
        if any(marker in lowered for marker in markers):
            terms.update(aliases[kind])
    return phrase, tuple(sorted(terms, key=lambda value: (-len(value), value)))


def best_query_match(content: str, query: str) -> QueryMatch | None:
    phrase, terms = expanded_query_terms(query)
    phrase_pattern = re.compile(re.escape(phrase), re.IGNORECASE) if phrase else None
    phrase_matches = list(phrase_pattern.finditer(content)) if phrase_pattern else []
    if phrase_matches:
        first = phrase_matches[0]
        return QueryMatch(first.start(), first.end(), 100.0 + len(phrase_matches) * 10, len(phrase_matches))

    lowered = content.casefold()
    best: tuple[float, int, int] | None = None
    total_hits = 0
    for term in terms:
        if not term:
            continue
        start = lowered.find(term.casefold())
        if start < 0:
            continue
        hits = lowered.count(term.casefold())
        total_hits += hits
        score = len(term) * 2 + min(hits, 20)
        candidate = (score, start, start + len(term))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return QueryMatch(best[1], best[2], best[0] + total_hits / max(1, len(content)), 0)


def _char_from_byte(text: str, offset_bytes: int) -> int:
    encoded = text.encode("utf-8")
    offset = max(0, min(len(encoded), offset_bytes))
    while offset > 0 and offset < len(encoded) and encoded[offset] & 0xC0 == 0x80:
        offset -= 1
    return len(encoded[:offset].decode("utf-8"))


def _byte_from_char(text: str, char_offset: int) -> int:
    return len(text[: max(0, min(len(text), char_offset))].encode("utf-8"))


def _line_span(text: str, char_offset: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, char_offset) + 1
    end = text.find("\n", char_offset)
    return start, len(text) if end < 0 else end


def iter_delimited_records(
    text: str,
    delimiter: str,
) -> Iterator[DelimitedRecord]:
    """Yield exact logical CSV/TSV records, including quoted newlines.

    ``csv.reader.line_num`` reports the last physical line consumed for each
    logical record. Combining it with original line endings preserves exact
    quoting and embedded newlines instead of reconstructing normalized rows.
    """
    if len(delimiter) != 1:
        raise ValueError("delimiter must be one character")
    physical_ends: list[int] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\r":
            cursor += 2 if cursor + 1 < len(text) and text[cursor + 1] == "\n" else 1
            physical_ends.append(cursor)
        elif text[cursor] == "\n":
            cursor += 1
            physical_ends.append(cursor)
        else:
            cursor += 1
    if text and (not physical_ends or physical_ends[-1] != len(text)):
        physical_ends.append(len(text))

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=delimiter,
        strict=True,
    )
    record_start = 0
    for fields in reader:
        consumed_end = (
            physical_ends[reader.line_num - 1]
            if reader.line_num and reader.line_num <= len(physical_ends)
            else len(text)
        )
        record_end = consumed_end
        if text[max(record_start, record_end - 2):record_end] == "\r\n":
            record_end -= 2
        elif record_end > record_start and text[record_end - 1] in "\r\n":
            record_end -= 1
        yield DelimitedRecord(
            char_start=record_start,
            char_end=record_end,
            raw=text[record_start:record_end],
            fields=tuple(fields),
        )
        record_start = consumed_end


def _delimited_row_span(
    text: str,
    char_offset: int,
    delimiter: str,
) -> tuple[int, int]:
    """Return one logical CSV/TSV record, including quoted newlines."""
    try:
        for record in iter_delimited_records(text, delimiter):
            if record.char_start <= char_offset <= record.char_end:
                return record.char_start, record.char_end
    except csv.Error:
        pass
    return _line_span(text, char_offset)


def _paragraph_span(text: str, char_offset: int) -> tuple[int, int]:
    before = list(re.finditer(r"\n\s*\n", text[:char_offset]))
    start = before[-1].end() if before else 0
    after = re.search(r"\n\s*\n", text[char_offset:])
    end = char_offset + after.start() if after else len(text)
    return start, end


def _json_spans(text: str, char_offset: int) -> list[tuple[int, int, str]]:
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int, str]] = []
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
            end = index + 1
            if start <= char_offset < end:
                spans.append((start, end, "json_object" if opener == "{" else "json_array"))
    return sorted(spans, key=lambda item: (item[1] - item[0], item[0]))


def _json_member_span(text: str, char_offset: int, container: tuple[int, int, str]) -> tuple[int, int]:
    start, end, _kind = container
    depth = 0
    in_string = False
    escaped = False
    member_start = start + 1
    member_end = end - 1
    for index in range(start + 1, end - 1):
        char = text[index]
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
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            if index < char_offset:
                member_start = index + 1
            elif index >= char_offset:
                member_end = index
                break
    while member_start < member_end and text[member_start].isspace():
        member_start += 1
    while member_end > member_start and text[member_end - 1].isspace():
        member_end -= 1
    return member_start, member_end


def _line_table(text: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        end = cursor + len(raw_line.rstrip("\r\n"))
        rows.append((cursor, end, raw_line.rstrip("\r\n")))
        cursor += len(raw_line)
    if not rows and text:
        rows.append((0, len(text), text))
    return rows


def _python_function_span(text: str, char_offset: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        tree = None
    if tree is not None:
        lines = text.splitlines(keepends=True)
        line_starts: list[int] = []
        cursor = 0
        for line in lines:
            line_starts.append(cursor)
            cursor += len(line)

        def position(line_number: int, utf8_column: int) -> int:
            if not lines:
                return 0
            line_index = max(0, min(len(lines) - 1, line_number - 1))
            raw = lines[line_index].encode("utf-8")
            byte_index = max(0, min(len(raw), utf8_column))
            while byte_index > 0 and byte_index < len(raw) and raw[byte_index] & 0xC0 == 0x80:
                byte_index -= 1
            return line_starts[line_index] + len(raw[:byte_index].decode("utf-8"))

        candidates: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            decorators = getattr(node, "decorator_list", ()) or ()
            start_line = min(
                [node.lineno, *(decorator.lineno for decorator in decorators)],
            )
            start = position(start_line, 0)
            end = position(
                int(getattr(node, "end_lineno", node.lineno)),
                int(getattr(node, "end_col_offset", 0)),
            )
            if start <= char_offset < end:
                candidates.append((start, end))
        if candidates:
            return min(candidates, key=lambda item: (item[1] - item[0], item[0]))

    rows = _line_table(text)
    containing = next((index for index, row in enumerate(rows) if row[0] <= char_offset <= row[1]), None)
    if containing is None:
        return None
    for index in range(containing, -1, -1):
        match = _PY_DEF_RE.match(rows[index][2])
        if match is None:
            continue
        indent = len(match.group("indent").replace("\t", "    "))
        end = len(text)
        for later in range(index + 1, len(rows)):
            line = rows[later][2]
            if not line.strip():
                continue
            later_indent = len(line) - len(line.lstrip(" \t"))
            if later_indent <= indent:
                end = rows[later][0]
                break
        if rows[index][0] <= char_offset < end:
            return rows[index][0], end
    return None


def _brace_function_span(text: str, char_offset: int) -> tuple[int, int] | None:
    candidates = [match for match in _BRACE_DEF_RE.finditer(text[: char_offset + 1])]
    for match in reversed(candidates):
        brace = text.find("{", match.start(), min(len(text), char_offset + 1))
        if brace < 0:
            continue
        depth = 0
        in_string: str | None = None
        in_line_comment = False
        in_block_comment = False
        in_regex = False
        escaped = False
        for index in range(brace, len(text)):
            char = text[index]
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if in_line_comment:
                if char == "\n":
                    in_line_comment = False
                continue
            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                continue
            if in_regex:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "/":
                    in_regex = False
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue
            if char == "/" and next_char == "/":
                in_line_comment = True
                continue
            if char == "/" and next_char == "*":
                in_block_comment = True
                continue
            if char == "/":
                previous = text[:index].rstrip()[-1:] or "="
                if previous in "=(:,![{;?":
                    in_regex = True
                    continue
            if char in {'"', "'", "`"}:
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    if match.start() <= char_offset < end:
                        line_start = text.rfind("\n", 0, match.start()) + 1
                        return line_start, end
                    break
    return None


def _bounded_span(
    start: int,
    end: int,
    match_start: int,
    match_end: int,
    max_chars: int,
) -> tuple[int, int, bool]:
    if end - start <= max_chars:
        return start, end, False
    match_length = max(1, match_end - match_start)
    available = max(0, max_chars - match_length)
    bounded_start = max(start, match_start - available // 2)
    bounded_end = min(end, bounded_start + max_chars)
    bounded_start = max(start, bounded_end - max_chars)
    return bounded_start, bounded_end, True


def precise_fragment(
    text: str,
    *,
    match_offset_bytes: int,
    match_length_bytes: int,
    detected_type: str,
    max_chars: int,
    context_chars: int = 120,
) -> dict[str, Any]:
    """Return one exact contiguous structural unit plus small literal context."""
    if max_chars < 1:
        raise ValueError("fragment max_chars must be positive")
    match_start = _char_from_byte(text, match_offset_bytes)
    match_end = _char_from_byte(text, match_offset_bytes + max(1, match_length_bytes))
    match_end = max(match_start + 1, match_end)

    fragment_type = "text_line"
    start, end = _line_span(text, match_start)
    if detected_type == "json":
        spans = _json_spans(text, match_start)
        if spans:
            start, end, fragment_type = spans[0]
            if end - start > max_chars:
                start, end = _json_member_span(text, match_start, spans[0])
                fragment_type = "json_member"
    elif detected_type in {"csv", "tsv"}:
        start, end = _delimited_row_span(
            text,
            match_start,
            "\t" if detected_type == "tsv" else ",",
        )
        fragment_type = "csv_row" if detected_type == "csv" else "tsv_row"
    elif detected_type == "code":
        function = _python_function_span(text, match_start) or _brace_function_span(text, match_start)
        if function is not None:
            start, end = function
            fragment_type = "code_function"
    else:
        line_text = text[start:end].lstrip()
        if not _LOG_LINE_RE.search(line_text):
            paragraph = _paragraph_span(text, match_start)
            if paragraph[1] - paragraph[0] <= max_chars:
                start, end = paragraph
                fragment_type = "text_paragraph"

    content_budget = max(1, max_chars - min(context_chars * 2, max_chars // 3))
    start, end, fragment_truncated = _bounded_span(
        start, end, match_start, match_end, content_budget,
    )
    content = text[start:end]
    remaining = max(0, max_chars - len(content))
    before_budget = min(context_chars, remaining // 2)
    after_budget = min(context_chars, remaining - before_budget)
    before_start = max(0, start - before_budget)
    after_end = min(len(text), end + after_budget)
    context_before = text[before_start:start]
    context_after = text[end:after_end]

    content_bytes = content.encode("utf-8")
    before_bytes = context_before.encode("utf-8")
    after_bytes = context_after.encode("utf-8")
    offset_bytes = _byte_from_char(text, start)
    before_offset = _byte_from_char(text, before_start)
    after_offset = _byte_from_char(text, end)
    return {
        "content": content,
        "offset_bytes": offset_bytes,
        "actual_length_bytes": len(content_bytes),
        "next_offset_bytes": offset_bytes + len(content_bytes),
        "decoded_chars": len(content),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "chunk_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "fragment_type": fragment_type,
        "fragment_complete": not fragment_truncated,
        "match_offset_bytes": match_offset_bytes,
        "match_length_bytes": max(1, match_length_bytes),
        "context_before": context_before,
        "context_before_offset_bytes": before_offset,
        "context_before_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "context_after": context_after,
        "context_after_offset_bytes": after_offset,
        "context_after_sha256": hashlib.sha256(after_bytes).hexdigest(),
        "returned_chars": len(content) + len(context_before) + len(context_after),
    }


def precise_fragment_from_window(
    window_text: str,
    *,
    window_offset_bytes: int,
    artifact_total_bytes: int,
    match_offset_bytes: int,
    match_length_bytes: int,
    detected_type: str,
    max_chars: int,
    context_chars: int = 120,
) -> dict[str, Any]:
    """Build a fragment from a bounded Artifact window using absolute offsets."""
    if window_offset_bytes < 0 or artifact_total_bytes < 0:
        raise ValueError("window and Artifact byte offsets must be non-negative")
    window_length_bytes = len(window_text.encode("utf-8"))
    window_end_bytes = min(
        artifact_total_bytes,
        window_offset_bytes + window_length_bytes,
    )
    if not (
        window_offset_bytes
        <= match_offset_bytes
        < max(window_offset_bytes + 1, window_end_bytes)
    ):
        raise ValueError("match offset must be contained in the Artifact window")

    relative_match_offset = match_offset_bytes - window_offset_bytes
    fragment = precise_fragment(
        window_text,
        match_offset_bytes=relative_match_offset,
        match_length_bytes=match_length_bytes,
        detected_type=detected_type,
        max_chars=max_chars,
        context_chars=context_chars,
    )
    local_offset = int(fragment["offset_bytes"])
    local_next_offset = int(fragment["next_offset_bytes"])
    full_window = window_offset_bytes == 0 and window_end_bytes >= artifact_total_bytes
    touches_truncated_edge = (
        (window_offset_bytes > 0 and local_offset == 0)
        or (
            window_end_bytes < artifact_total_bytes
            and local_next_offset >= window_length_bytes
        )
    )
    structural_types = {
        "json": {"json_object", "json_array", "json_member"},
        "code": {"code_function"},
    }
    structure_not_yet_visible = (
        detected_type in structural_types
        and fragment["fragment_type"] not in structural_types[detected_type]
    )
    needs_expansion = not full_window and (
        touches_truncated_edge or structure_not_yet_visible
    )

    for field in (
        "offset_bytes",
        "next_offset_bytes",
        "context_before_offset_bytes",
        "context_after_offset_bytes",
    ):
        fragment[field] = int(fragment[field]) + window_offset_bytes
    fragment["match_offset_bytes"] = match_offset_bytes
    fragment["fragment_complete"] = (
        bool(fragment["fragment_complete"]) and not needs_expansion
    )
    fragment["_needs_window_expansion"] = needs_expansion
    return fragment
