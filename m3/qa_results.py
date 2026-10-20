"""Completion semantics shared by QA drivers, probes and summaries."""
import json
from pathlib import Path


def completed_answer(row):
    from m3.qa import _has_protocol_scaffolding
    answer = row.get("predicted_answer")
    return (not row.get("error") and isinstance(answer, str)
            and bool(answer.strip()) and not _has_protocol_scaffolding(answer))


def latest_answers(path):
    rows = {}
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(row, dict) or row.get("video_id") is None or row.get("question_id") is None:
            continue
        rows[(str(row["video_id"]), str(row["question_id"]))] = row
    return rows


def completed_keys(path):
    return {key for key, row in latest_answers(path).items() if completed_answer(row)}


def separate_partial_line(path):
    """Keep an interrupted final JSON line from swallowing the retry row."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
