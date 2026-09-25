"""Capture messages submitted through the MCP memory-write boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import uuid4

from core.models import RequestContext
from skills.sanitizer import sanitize_trace_text

logger = logging.getLogger(__name__)
_CONTENT_CHARS = 10_000


def capture_write_trace(
    service: Any,
    ctx: RequestContext,
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    client_request_id: str | None = None,
) -> None:
    """Persist an ordered, bounded write batch and finalize its turn."""
    try:
        request_id = client_request_id or uuid4().hex
        identity = [ctx.account_id, ctx.user_id, ctx.agent_id, session_id, request_id]
        batch = hashlib.sha256(
            json.dumps([identity, messages], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        events = []
        assistant = ""
        for message in messages:
            role = message["role"]
            content = sanitize_trace_text(message["content"])
            event = {"event_type": f"{role}_message", "content_size": len(content)}
            field = {"user": "prompt", "assistant": "assistant_message"}.get(role, "content")
            event[field] = content[:_CONTENT_CHARS]
            if len(content) > _CONTENT_CHARS:
                event["truncated"] = True
            if role == "assistant":
                assistant = event[field]
            events.append(event)

        end = {"event_type": "turn_end", "status": "ok", "boundary": "mcp_write"}
        if assistant:
            end["assistant_message"] = assistant
        events.append(end)
        for index, event in enumerate(events):
            stored = _append(service, ctx, batch, index, event, session_id=session_id)
        result = service.finalize_session_trace({
            "_ctx": ctx,
            "sessionId": session_id,
            "stopEventIndex": stored["event_index"],
        })
        if not result.get("ok"):
            raise RuntimeError("trace finalization unavailable")
    except Exception as exc:
        logger.warning("MCP write trace capture failed: %s", type(exc).__name__)


def _append(
    service: Any, ctx: RequestContext, batch: str, index: int, event: dict[str, Any],
    *, session_id: str,
) -> dict[str, Any]:
    event_id = hashlib.sha256(f"mcp-write:{batch}:{index}".encode()).hexdigest()
    result = service.add_session_trace({
        "_ctx": ctx,
        "sessionId": session_id,
        "eventId": event_id,
        "hookType": event["event_type"],
        "event": {**event, "write_batch_id": batch},
    })
    if not result.get("ok"):
        raise RuntimeError("trace persistence unavailable")
    return result
