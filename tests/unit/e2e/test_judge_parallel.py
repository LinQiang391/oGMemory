from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from judge_parallel import build_judge_signature, judge_row, main, merge_existing  # noqa: E402


def _row() -> dict[str, str]:
    return {
        "sample_id": "sample-0",
        "qi": "0",
        "question": "What did Alice buy?",
        "expected": "A shell necklace",
        "response": "A shell necklace",
    }


def test_judge_failure_retries_and_remains_unjudged() -> None:
    with patch("judge_parallel.get_client", side_effect=RuntimeError("temporary outage")) as client:
        index, label, reasoning, had_error, rate_limited = judge_row(
            0,
            _row(),
            "dummy",
            "https://example.invalid/v1",
            "judge-model",
            "chat",
            max_attempts=3,
            retry_delay=0,
        )

    assert index == 0
    assert label == ""
    assert had_error is True
    assert rate_limited is False
    assert "attempt 3/3" in reasoning
    assert client.call_count == 3


def test_malformed_judge_response_retries_and_remains_unjudged() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
    )
    with patch("judge_parallel.get_client", return_value=client):
        _, label, reasoning, had_error, rate_limited = judge_row(
            0,
            _row(),
            "dummy",
            "https://example.invalid/v1",
            "judge-model",
            "chat",
            max_attempts=2,
            retry_delay=0,
        )

    assert label == ""
    assert had_error is True
    assert rate_limited is False
    assert "attempt 2/2" in reasoning
    assert client.chat.completions.create.call_count == 2


def test_chat_judge_does_not_require_response_format_support() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"CORRECT"}'))]
    )
    with patch("judge_parallel.get_client", return_value=client):
        _, label, _, had_error, rate_limited = judge_row(
            0,
            _row(),
            "dummy",
            "https://example.invalid/v1",
            "judge-model",
            "chat",
            retry_delay=0,
        )

    assert label == "CORRECT"
    assert had_error is False
    assert rate_limited is False
    assert "response_format" not in client.chat.completions.create.call_args.kwargs


def test_main_fails_when_a_row_cannot_be_judged(tmp_path: Path) -> None:
    input_path = tmp_path / "qa.csv"
    output_path = tmp_path / "judged.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerow(_row())

    argv = [
        "judge_parallel.py",
        str(input_path),
        "--output",
        str(output_path),
        "--api-base",
        "https://example.invalid/v1",
        "--judge-model",
        "judge-model",
        "--api-key",
        "dummy",
        "--retry-delay",
        "0",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("judge_parallel.judge_row", return_value=(0, "", "[ERROR] outage", True, False)),
    ):
        result = main()

    judged = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert result == 1
    assert judged[0]["is_correct"] == ""
    assert judged[0]["reasoning"] == "[ERROR] outage"


def test_judge_retries_rate_limit() -> None:
    with patch("judge_parallel.get_client", side_effect=RuntimeError("429 Too Many Requests")) as client:
        _, label, _, had_error, rate_limited = judge_row(
            0,
            _row(),
            "dummy",
            "https://example.invalid/v1",
            "judge-model",
            "chat",
            max_attempts=3,
            retry_delay=0,
        )

    assert label == ""
    assert had_error is True
    assert rate_limited is True
    assert client.call_count == 3


def test_judge_stops_retrying_rate_limit_when_requested() -> None:
    stop_event = threading.Event()
    with patch("judge_parallel.get_client", side_effect=RuntimeError("429 Too Many Requests")) as client:
        _, label, _, had_error, rate_limited = judge_row(
            0,
            _row(),
            "dummy",
            "https://example.invalid/v1",
            "judge-model",
            "chat",
            max_attempts=3,
            retry_delay=1,
            stop_event=stop_event,
        )

    assert label == ""
    assert had_error is True
    assert rate_limited is True
    assert stop_event.is_set()
    assert client.call_count == 1


def test_main_returns_without_waiting_for_running_tasks_on_rate_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "qa.csv"
    output_path = tmp_path / "judged.csv"
    rows = [_row(), _row() | {"qi": "1"}]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerows(rows)

    slow_task_started = threading.Event()

    def fake_judge(index, *_args, **_kwargs):
        if index == 0:
            slow_task_started.wait(timeout=1)
            return 0, "", "[ERROR] 429 Too Many Requests", True, True
        slow_task_started.set()
        time.sleep(1.2)
        return 1, "CORRECT", "", False, False

    argv = [
        "judge_parallel.py",
        str(input_path),
        "--output",
        str(output_path),
        "--api-base",
        "https://example.invalid/v1",
        "--judge-model",
        "judge-model",
        "--api-key",
        "dummy",
        "--parallel",
        "2",
        "--fail-fast-rate-limit",
    ]
    started = time.monotonic()
    with patch.object(sys, "argv", argv), patch("judge_parallel.judge_row", side_effect=fake_judge):
        result = main()

    assert result == 2
    assert time.monotonic() - started < 0.6


def test_resume_ignores_judgment_for_changed_response(tmp_path: Path) -> None:
    output_path = tmp_path / "judged.csv"
    signature = build_judge_signature("https://example.invalid/v1", "judge-model", "chat")
    prior = _row() | {
        "response": "old answer",
        "is_correct": "CORRECT",
        "reasoning": "old",
        "judge_signature": signature,
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prior))
        writer.writeheader()
        writer.writerow(prior)

    current = [_row() | {"response": "new answer"}]
    merge_existing(current, str(output_path), signature)

    assert current[0].get("is_correct", "") == ""
    assert current[0].get("reasoning", "") == ""


def test_resume_ignores_judgment_from_different_judge_config(tmp_path: Path) -> None:
    output_path = tmp_path / "judged.csv"
    old_signature = build_judge_signature("https://example.invalid/v1", "old-model", "chat")
    prior = _row() | {
        "is_correct": "CORRECT",
        "reasoning": "old",
        "judge_signature": old_signature,
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prior))
        writer.writeheader()
        writer.writerow(prior)

    current = [_row()]
    new_signature = build_judge_signature("https://example.invalid/v1", "new-model", "chat")
    merge_existing(current, str(output_path), new_signature)

    assert current[0].get("is_correct", "") == ""


def test_resume_reuses_matching_judgment(tmp_path: Path) -> None:
    output_path = tmp_path / "judged.csv"
    signature = build_judge_signature("https://example.invalid/v1", "judge-model", "chat")
    prior = _row() | {
        "is_correct": "CORRECT",
        "reasoning": "matches",
        "judge_signature": signature,
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prior))
        writer.writeheader()
        writer.writerow(prior)

    current = [_row()]
    merge_existing(current, str(output_path), signature)

    assert current[0]["is_correct"] == "CORRECT"
    assert current[0]["reasoning"] == "matches"


def test_main_rejects_zero_checkpoint_interval() -> None:
    argv = [
        "judge_parallel.py",
        "input.csv",
        "--output",
        "output.csv",
        "--api-base",
        "https://example.invalid/v1",
        "--judge-model",
        "judge-model",
        "--checkpoint-every",
        "0",
    ]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
