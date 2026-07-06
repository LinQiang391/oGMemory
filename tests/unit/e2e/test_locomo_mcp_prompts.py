from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))

from locomo_mcp_prompts import (  # noqa: E402
    OGMEMORY_SPEC,
    OPENVIKING_SPEC,
    build_ingest_input_message,
    build_qa_input_message,
)
from eval import build_memory_session_id, wait_for_ogmemory_idle  # noqa: E402


def test_ogmemory_qa_prompt_uses_only_ogmemory_query_arguments() -> None:
    prompt = build_qa_input_message(
        OGMEMORY_SPEC,
        "Where did Alice go?",
        enable_read=True,
        min_score="0.2",
    )

    assert "ogmemory__ogm-query_memory" in prompt
    assert "query=<concise question-focused search string>, top_k=5, score_threshold=0.2" in prompt
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
    assert "ogmemory__ogm-query_memory" not in prompt


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
    assert '"messages": [{"role": "user", "content": <the full Transcript text below exactly as written>}]' in openviking_prompt
    assert '"wait": true' in ogmemory_prompt
    assert '"wait": true' not in openviking_prompt


def test_memory_session_id_is_isolated_by_eval_user() -> None:
    session_id = build_memory_session_id(
        "locomo-small-cleanasync-20260703_173338",
        "locomo2x-r2-conv-26",
        "session_1",
    )

    assert session_id == (
        "ingest-locomo-small-cleanasync-20260703_173338-"
        "locomo2x-r2-conv-26-session_1"
    )


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
