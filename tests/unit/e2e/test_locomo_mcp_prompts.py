from __future__ import annotations

import os
import sys
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from eval import (  # noqa: E402
    build_memory_session_id,
    extract_response_text,
    is_successful_ingest_reply,
    run_judge,
    save_api_logs,
    save_ingest_record,
    send_message,
    wait_for_ogmemory_idle,
)
from locomo_mcp_prompts import (  # noqa: E402
    LOCOMO_ACCURACY_PROMPT,
    OGMEMORY_SPEC,
    OPENVIKING_SPEC,
    build_ingest_input_message,
    build_mcp_instructions,
    build_qa_input_message,
)


def test_ogmemory_prompts_use_only_underscore_projected_tool_names() -> None:
    assert (
        OGMEMORY_SPEC.write_tool,
        OGMEMORY_SPEC.search_tool,
        OGMEMORY_SPEC.read_tool,
    ) == (
        "ogmemory__ogm_write_memory",
        "ogmemory__ogm_query_memory",
        "ogmemory__ogm_read_memory",
    )

    prompts = (
        build_mcp_instructions(OGMEMORY_SPEC, "eval-user", "session-1"),
        build_ingest_input_message(OGMEMORY_SPEC, "Alice: hi", "session-1"),
        build_qa_input_message(OGMEMORY_SPEC, "Where did Alice go?", enable_read=True),
    )

    assert all("ogmemory__ogm-" not in prompt for prompt in prompts)


def test_extract_response_text_rejects_missing_text() -> None:
    with pytest.raises(RuntimeError, match="Could not extract text"):
        extract_response_text({"output": [{"content": []}]})


def test_send_message_logs_response_parse_failure() -> None:
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"output": []}

    with (
        patch("eval.api_logs", []) as logs,
        patch("eval.requests.post", return_value=response),
        pytest.raises(RuntimeError, match="Could not extract text"),
    ):
        send_message("http://gateway", "token", "user", "hello")

    assert len(logs) == 1
    assert logs[0]["response_status"] == "200"
    assert logs[0]["response_body"] == '{"output": []}'
    assert logs[0]["error"].startswith("Response parse error:")


def test_eval_judge_uses_shared_judge_implementation(tmp_path: Path) -> None:
    input_path = tmp_path / "qa.csv"
    output_path = tmp_path / "judged.csv"
    input_path.write_text(
        "question,expected,response\nQ,A,A\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        api_key="test-key",
        api_base="https://example.invalid/v1",
        judge_model="judge-model",
    )

    with patch("eval.judge_row", return_value=(0, "CORRECT", "same", False)) as judge:
        run_judge(args)

    judge.assert_called_once()
    call = judge.call_args.args
    assert call[0] == 0
    assert call[1]["response"] == "A"
    assert call[2:] == (
        "test-key",
        "https://example.invalid/v1",
        "judge-model",
        "chat",
    )
    assert "CORRECT" in output_path.read_text(encoding="utf-8")


def test_ingest_record_creates_configured_parent_directory(tmp_path: Path) -> None:
    record_path = tmp_path / "nested" / "ingest.json"

    save_ingest_record({"session": {"success": True}}, str(record_path))

    assert record_path.is_file()


def test_api_logs_do_not_overwrite_within_one_second(tmp_path: Path) -> None:
    with (
        patch.dict(os.environ, {"LOCOMO_API_LOG_DIR": str(tmp_path)}),
        patch("eval.api_logs", [{"timestamp": "now"}]),
        patch("eval.datetime") as clock,
    ):
        clock.now.side_effect = [
            real_datetime(2026, 8, 5, 1, 2, 3, 1),
            real_datetime(2026, 8, 5, 1, 2, 3, 2),
        ]
        save_api_logs(SimpleNamespace())
        save_api_logs(SimpleNamespace())

    assert len(list(tmp_path.glob("api_log_*.csv"))) == 2


def test_ogmemory_qa_prompt_uses_only_ogmemory_query_arguments() -> None:
    prompt = build_qa_input_message(
        OGMEMORY_SPEC,
        "Where did Alice go?",
        enable_read=True,
        min_score="0.2",
    )

    assert "ogmemory__ogm_query_memory" in prompt
    assert "query=<concise question-focused search string>, top_k=15, score_threshold=0.2" in prompt
    assert "Do not pass types, session_id, agent_id, or limit" in prompt
    assert "min_score" not in prompt
    assert "openviking__search" not in prompt


def test_openviking_qa_prompt_uses_only_openviking_search_arguments() -> None:
    prompt = build_qa_input_message(
        OPENVIKING_SPEC,
        "Where did Alice go?",
        enable_read=True,
        min_score="0.2",
    )

    assert "openviking__search" in prompt
    assert "query=<concise question-focused search string>, limit=5, min_score=0.2" in prompt
    assert "Do not pass types, session_id, agent_id, top_k, or score_threshold" in prompt
    assert "top_k=5" not in prompt
    assert "ogmemory__ogm_query_memory" not in prompt


def test_locomo_judge_schema_is_fixed() -> None:
    assert 'key as "label"' in LOCOMO_ACCURACY_PROMPT
    assert "May 7th" in LOCOMO_ACCURACY_PROMPT


def test_ingest_prompts_keep_backend_specific_write_shapes() -> None:
    ogmemory_prompt = build_ingest_input_message(
        OGMEMORY_SPEC,
        "Alice: hi",
        "ingest-sample0-session1",
    )
    openviking_prompt = build_ingest_input_message(
        OPENVIKING_SPEC,
        "Alice: hi",
        "ingest-sample0-session1",
    )

    assert (
        '{"text": <the full Transcript text below exactly as written>, '
        '"session_id": "ingest-sample0-session1", "wait": true}'
    ) in ogmemory_prompt
    assert (
        '"messages": [{"role": "user", "content": <the full Transcript text below exactly as written>}]'
        in openviking_prompt
    )
    assert '"wait": true' in ogmemory_prompt
    assert '"wait": true' not in openviking_prompt


def test_memory_session_id_is_isolated_by_eval_user() -> None:
    session_id = build_memory_session_id(
        "locomo-small-cleanasync-20260703_173338",
        "locomo2x-r2-conv-26",
        "session_1",
    )

    assert session_id == (
        "ingest-locomo-small-cleanasync-20260703_173338-locomo2x-r2-conv-26-session_1"
    )


def test_ingest_success_reply_requires_exact_ack() -> None:
    assert is_successful_ingest_reply("memory written")
    assert is_successful_ingest_reply("Memory written.")
    assert not is_successful_ingest_reply("memory not written")
    assert not is_successful_ingest_reply("write failed before memory written")


def test_ogmemory_idle_wait_rejects_missing_session() -> None:
    args = SimpleNamespace(
        ogmemory_url="http://127.0.0.1:8091",
        ogmemory_token="root-key",
        ogmemory_account_id="acct-demo",
        ogmemory_agent_id="",
        ogmemory_idle_timeout=5,
        user="eval-user",
    )
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "idle": True,
        "reason": "session_not_found",
        "extraction_in_progress": False,
    }

    with patch("eval.requests.post", return_value=response):
        try:
            wait_for_ogmemory_idle(args, "missing-session")
        except RuntimeError as exc:
            assert "could not find session missing-session" in str(exc)
        else:
            raise AssertionError("wait_for_ogmemory_idle should reject session_not_found")


def test_ogmemory_idle_wait_polls_outbox_without_draining() -> None:
    args = SimpleNamespace(
        ogmemory_url="http://127.0.0.1:8091",
        ogmemory_token="root-key",
        ogmemory_account_id="acct-demo",
        ogmemory_agent_id="",
        ogmemory_idle_timeout=5,
        user="eval-user",
    )
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"idle": True, "reason": "idle"}

    with patch("eval.requests.post", return_value=response) as post:
        wait_for_ogmemory_idle(args, "session-1")

    assert post.call_args.kwargs["json"]["drainOutbox"] is False
    assert post.call_args.kwargs["json"]["waitOutbox"] is True
