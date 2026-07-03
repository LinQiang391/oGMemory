"""LoCoMo evaluation wrapper for OpenViking MCP.

This reuses tests/e2e/eval.py for parsing, OpenClaw calls, judging, and stats.
Only the MCP instructions are swapped from oGMemory tools to OpenViking tools.
"""

from __future__ import annotations

import os

import eval as locomo_eval
from locomo_mcp_prompts import OPENVIKING_SPEC
from locomo_mcp_prompts import build_ingest_input_message as build_backend_ingest_input_message
from locomo_mcp_prompts import build_mcp_instructions
from locomo_mcp_prompts import build_qa_input_message as build_backend_qa_input_message


def build_openviking_mcp_instructions(user: str, session_key: str | None) -> str:
    return build_mcp_instructions(OPENVIKING_SPEC, user, session_key)


def build_qa_input_message(question: str, question_time: str | None = None) -> str:
    return build_backend_qa_input_message(
        OPENVIKING_SPEC,
        question,
        question_time=question_time,
        enable_read=os.environ.get("OPENVIKING_QA_ENABLE_READ", "1") == "1",
        min_score=os.environ.get("OPENVIKING_QA_MIN_SCORE", ""),
    )


def build_ingest_input_message(conversation_text: str, memory_session_id: str) -> str:
    return build_backend_ingest_input_message(OPENVIKING_SPEC, conversation_text, memory_session_id)


locomo_eval.build_ogmemory_mcp_instructions = build_openviking_mcp_instructions
locomo_eval.build_qa_input_message = build_qa_input_message
locomo_eval.build_ingest_input_message = build_ingest_input_message


if __name__ == "__main__":
    locomo_eval.main()
