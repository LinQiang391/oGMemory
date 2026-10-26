"""Bounded deterministic synopsis generation."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any


class ToolResultSynopsisBuilder:
    def __init__(self, max_chars: int = 2_000):
        if max_chars < 1:
            raise ValueError("synopsis_max_chars must be positive")
        self.max_chars = max_chars

    def build(self, text: str, mime_type: str | None = None) -> tuple[str, str]:
        detected = self._detect(text, mime_type)
        try:
            synopsis = getattr(self, f"_build_{detected}")(text)
        except Exception:
            detected = "text"
            synopsis = self._build_text(text)
        return detected, self._bound(synopsis)

    def _bound(self, value: str) -> str:
        if len(value) <= self.max_chars:
            return value
        return value[: max(1, self.max_chars - 1)] + "…"

    @staticmethod
    def _looks_like_headered_delimited(text: str, delimiter: str) -> bool:
        """Conservatively auto-detect only tables with a real header row.

        The downstream record projection treats row zero as a header. Merely
        seeing commas in a log line is therefore insufficient: build logs and
        stack traces commonly contain stable comma-separated fragments inside
        otherwise free-form text. Explicit CSV/TSV MIME types still bypass this
        heuristic.
        """
        lines = [line for line in text.splitlines() if line.strip()][:50]
        if len(lines) < 2 or delimiter not in lines[0]:
            return False
        sample = "\n".join(lines)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiter)
            rows = list(csv.reader(io.StringIO(sample), dialect=dialect))
            widths = {len(row) for row in rows}
            return (
                dialect.delimiter == delimiter
                and len(widths) == 1
                and next(iter(widths), 0) >= 2
                and csv.Sniffer().has_header(sample)
            )
        except (csv.Error, StopIteration):
            return False

    @staticmethod
    def _detect(text: str, mime_type: str | None) -> str:
        mime = (mime_type or "").casefold()
        stripped = text.lstrip()
        if "json" in mime or stripped.startswith(("{", "[")):
            try:
                json.loads(text)
                return "json"
            except Exception:
                pass
        if "csv" in mime:
            return "csv"
        if "tab-separated" in mime or "tsv" in mime:
            return "tsv"
        if "xml" in mime or stripped.startswith("<"):
            try:
                ET.fromstring(text)
                return "xml"
            except Exception:
                pass
        if "yaml" in mime or stripped.startswith("---"):
            return "yaml"
        if re.search(r"(^|\n)\s*(def |class |function |func |package |import )", text):
            return "code"
        first = text.splitlines()[0] if text.splitlines() else ""
        if ToolResultSynopsisBuilder._looks_like_headered_delimited(text, ","):
            return "csv"
        if ToolResultSynopsisBuilder._looks_like_headered_delimited(text, "\t"):
            return "tsv"
        if re.match(r"^[\w.-]+\s*:\s*", first):
            return "yaml"
        return "text"

    @staticmethod
    def _scalar_samples(value: Any, limit: int = 5) -> list[str]:
        samples: list[str] = []
        queue = [value]
        while queue and len(samples) < limit:
            current = queue.pop(0)
            if isinstance(current, dict):
                queue.extend(current.values())
            elif isinstance(current, list):
                queue.extend(current[:limit])
            elif current is not None:
                samples.append(repr(current)[:120])
        return samples

    def _build_json(self, text: str) -> str:
        value = json.loads(text)
        if isinstance(value, dict):
            keys = sorted(str(key) for key in value)[:30]
            shape = f"object with {len(value)} keys: {', '.join(keys)}"
        elif isinstance(value, list):
            shape = f"array with {len(value)} items"
        else:
            shape = f"JSON {type(value).__name__}"
        samples = self._scalar_samples(value)
        return f"{shape}; scalar samples: {', '.join(samples) if samples else '(none)'}"

    @staticmethod
    def _build_delimited(text: str, delimiter: str) -> str:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        width = max((len(row) for row in rows), default=0)
        header = rows[0] if rows else []
        samples = rows[1:4]
        return (
            f"table with {max(0, len(rows) - 1)} data rows and {width} columns; "
            f"header: {header}; samples: {samples}"
        )

    def _build_csv(self, text: str) -> str:
        return self._build_delimited(text, ",")

    def _build_tsv(self, text: str) -> str:
        return self._build_delimited(text, "\t")

    @staticmethod
    def _build_xml(text: str) -> str:
        root = ET.fromstring(text)
        nodes = list(root.iter())
        tags = Counter(node.tag for node in nodes[1:])
        common = ", ".join(f"{tag}:{count}" for tag, count in tags.most_common(10))
        return f"XML root {root.tag!r}, {len(nodes)} nodes; common child tags: {common}"

    def _build_yaml(self, text: str) -> str:
        import yaml

        value = yaml.safe_load(text)
        if isinstance(value, dict):
            keys = sorted(str(key) for key in value)[:30]
            shape = f"YAML mapping with {len(value)} keys: {', '.join(keys)}"
        elif isinstance(value, list):
            shape = f"YAML sequence with {len(value)} items"
        else:
            shape = f"YAML {type(value).__name__}"
        samples = self._scalar_samples(value)
        return f"{shape}; scalar samples: {', '.join(samples) if samples else '(none)'}"

    @staticmethod
    def _edge_preview(text: str, max_each: int) -> str:
        """Return bounded, non-complete edge samples for a synopsis."""
        sample_budget = min(max_each * 2, max(0, len(text) - 1))
        if sample_budget <= 0:
            return "preview omitted"
        head_chars = min(max_each, (sample_budget + 1) // 2)
        tail_chars = min(max_each, sample_budget - head_chars)
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars else ""
        return f"start: {head!r}; end: {tail!r}"

    @staticmethod
    def _build_code(text: str) -> str:
        lines = text.splitlines()
        names = re.findall(r"^\s*(?:def|class|function|func)\s+([\w$]+)", text, re.MULTILINE)
        return (
            f"code with {len(lines)} lines; definitions: {', '.join(names[:20]) or '(none)'}; "
            f"{ToolResultSynopsisBuilder._edge_preview(text, 300)}"
        )

    @staticmethod
    def _build_text(text: str) -> str:
        return (
            f"text with {len(text.splitlines())} lines and {len(text)} characters; "
            f"{ToolResultSynopsisBuilder._edge_preview(text, 400)}"
        )
