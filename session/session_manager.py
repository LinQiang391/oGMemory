"""Server-side session lifecycle manager.

Manages per-session message buffers with optional AGFS persistence.
Follows OpenViking's session model: accumulate → threshold commit → archive + extract.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.models import RequestContext
from session.models import SessionMessage, SessionMeta, SessionWindowState

logger = logging.getLogger("ogmem.session")

# ---------------------------------------------------------------------------
# SessionBuffer — in-memory message accumulator
# ---------------------------------------------------------------------------


@dataclass
class SessionBuffer:
    """In-memory buffer for a single session's messages."""

    session_id: str
    messages: list[SessionMessage] = field(default_factory=list)
    meta: SessionMeta = field(default_factory=SessionMeta)
    commit_in_progress: bool = False
    turn_count: int = 0
    window_state: SessionWindowState = field(default_factory=SessionWindowState)
    tool_usage_stats: dict[str, dict] = field(default_factory=dict)
    # Format: {"tool_name": {"call_count": N, "success_count": M, "total_duration_ms": F}}

    # Incremental extraction tracking
    extraction_watermark: int = 0   # Index into self.messages after last extraction
    extraction_summary: str = ""    # Summary of already-extracted content (~500 tokens max)

    @property
    def pending_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self.messages)

    def add(self, role: str, content: str, created_at: str | None = None) -> SessionMessage:
        kwargs: dict = dict(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            role=role,
            content=content,
        )
        if created_at:
            kwargs["created_at"] = created_at
        msg = SessionMessage(**kwargs)
        self.messages.append(msg)
        if role == "user":
            self.turn_count += 1
            # Read-write closure: update active_task from user message
            # Simple heuristic: if message looks like a task statement, capture it
            if content and len(content.strip()) > 0:
                # Take first 200 chars as potential task statement
                task_candidate = content.strip()[:200]
                # Basic heuristic: starts with action verb or contains "need to"
                action_words = ["help", "need", "want", "please", "can you", "could you"]
                if any(word in task_candidate.lower() for word in action_words):
                    self.window_state.active_task = task_candidate
        self.meta.message_count = len(self.messages)
        self.meta.updated_at = datetime.now(timezone.utc).isoformat()
        return msg

    def should_compress(
        self,
        turn_threshold: int = 10,
        token_threshold: int = 10_000,
    ) -> bool:
        """Check if window state needs recompression."""
        turns_since = self.turn_count - self.window_state.turn_count_at_last_compress
        tokens_since = self.pending_tokens - self.window_state.token_count_at_last_compress
        return turns_since >= turn_threshold or tokens_since >= token_threshold

    def snapshot_and_clear(self) -> list[SessionMessage]:
        """Return a copy of messages and clear the buffer."""
        snap = list(self.messages)
        self.messages.clear()
        self.turn_count = 0
        self.window_state = SessionWindowState()
        self.extraction_watermark = 0
        self.extraction_summary = ""
        self.meta.message_count = 0
        self.meta.updated_at = datetime.now(timezone.utc).isoformat()
        return snap


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Server-side session lifecycle manager.

    In-memory buffer + optional AGFS persistence.
    Provides: get_or_create, add_message, get_session, commit, get_context.
    """

    def __init__(
        self,
        get_llm: Callable[[], Any] | None = None,
        get_write_api: Callable[[], Any] | None = None,
        get_agfs: Callable[[], Any] | None = None,
    ):
        self._sessions: dict[str, SessionBuffer] = {}
        self._get_llm = get_llm
        self._get_write_api = get_write_api
        self._get_agfs = get_agfs

        # Background task tracking
        self._tasks: dict[str, dict] = {}

    # -- Buffer lifecycle ---------------------------------------------------

    def get_or_create(self, session_id: str) -> SessionBuffer:
        """Get existing buffer or create a new one."""
        if session_id not in self._sessions:
            meta = SessionMeta(session_id=session_id)
            self._sessions[session_id] = SessionBuffer(
                session_id=session_id, meta=meta
            )
            logger.debug("Created new session buffer: %s", session_id)
        return self._sessions[session_id]

    def get_window_state(self, session_id: str) -> SessionWindowState:
        """Get current window state for a session."""
        buf = self.get_or_create(session_id)
        return buf.window_state

    def update_window_state(
        self,
        session_id: str,
        window_state: SessionWindowState,
    ) -> None:
        """Update window state after compression."""
        buf = self.get_or_create(session_id)
        buf.window_state = window_state

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        ctx: RequestContext,
        created_at: str | None = None,
    ) -> dict:
        """Add a message to the session buffer."""
        buf = self.get_or_create(session_id)
        msg = buf.add(role, content, created_at=created_at)
        return {
            "ok": True,
            "message_id": msg.id,
            "pending_tokens": buf.pending_tokens,
        }

    def get_session(self, session_id: str, ctx: RequestContext) -> dict:
        """Return session meta + pending_tokens."""
        buf = self.get_or_create(session_id)
        return {
            "ok": True,
            "session_id": session_id,
            "pending_tokens": buf.pending_tokens,
            "message_count": buf.meta.message_count,
            "commit_count": buf.meta.commit_count,
            "last_commit_at": buf.meta.last_commit_at,
            "created_at": buf.meta.created_at,
            "updated_at": buf.meta.updated_at,
        }

    # -- Commit (archive + extract) -----------------------------------------

    def commit(
        self,
        session_id: str,
        ctx: RequestContext,
        wait: bool = False,
        skip_extraction: bool = False,
        session_time=None,
    ) -> dict:
        """Two-phase commit.

        Phase 1 (sync): snapshot messages, clear buffer, write raw archive.
        Phase 2 (background thread): generate overview/abstract, extract memories.
        Returns { task_id, archived, archive_uri, status }.

        Args:
            skip_extraction: If True, skip memory extraction in phase 2
                (used when caller already performed extraction, e.g. after_turn).
            session_time: Optional datetime for temporal resolution in extraction.
        """
        buf = self.get_or_create(session_id)

        if not buf.messages:
            return {"ok": True, "archived": False, "reason": "empty_buffer"}

        if buf.commit_in_progress:
            return {"ok": True, "archived": False, "reason": "commit_in_progress"}

        # Phase 1: snapshot + clear
        buf.commit_in_progress = True
        snapshot = buf.snapshot_and_clear()

        archive_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task_info = {
            "task_id": task_id,
            "session_id": session_id,
            "archive_id": archive_id,
            "status": "processing",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._tasks[task_id] = task_info

        if wait:
            self._run_commit_phase2(
                snapshot, session_id, archive_id, task_info, ctx,
                skip_extraction=skip_extraction,
                session_time=session_time,
            )
            buf.commit_in_progress = False
            buf.meta.commit_count += 1
            buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
            return {
                "ok": True,
                "archived": task_info.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_info.get("status", "completed"),
            }

        # Fire background thread
        def _bg():
            try:
                self._run_commit_phase2(
                    snapshot, session_id, archive_id, task_info, ctx,
                    skip_extraction=skip_extraction,
                    session_time=session_time,
                )
            except Exception as exc:
                logger.error(
                    "commit phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                task_info["status"] = "failed"
                task_info["error"] = str(exc)
            finally:
                buf.commit_in_progress = False
                buf.meta.commit_count += 1
                buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()

        t = threading.Thread(target=_bg, daemon=True, name=f"commit-{session_id[:8]}")
        t.start()

        return {
            "ok": True,
            "archived": False,
            "archive_id": archive_id,
            "task_id": task_id,
            "status": "processing",
        }

    def _run_commit_phase2(
        self,
        snapshot: list[SessionMessage],
        session_id: str,
        archive_id: str,
        task_info: dict,
        ctx: RequestContext,
        skip_extraction: bool = False,
        session_time=None,
    ) -> None:
        """Background work: compress → write archive → extract memories."""
        messages_dicts = [
            {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at}
            for m in snapshot
        ]

        overview, abstract = self._compress(messages_dicts)
        archive_result = self._write_archive(
            session_id, archive_id, overview, abstract, messages_dicts, ctx
        )

        task_info["archived"] = archive_result.get("success", False)
        task_info["archive_uri"] = archive_result.get("uri", "")

        if not skip_extraction:
            extraction_stats = self._extract_memories(messages_dicts, ctx, session_time=session_time)
            task_info["extraction"] = extraction_stats
        task_info["status"] = "completed"

    def _compress(self, messages: list[dict]) -> tuple[str, str]:
        """Compress messages into overview + abstract."""
        try:
            if self._get_llm is not None:
                llm = self._get_llm()
                if llm is not None:
                    from session.compressor import SessionCompressor
                    compressor = SessionCompressor(llm=llm)
                    return compressor.compress(messages)
        except Exception as exc:
            logger.warning("LLM compress failed, using fallback: %s", exc)

        # Fallback: simple compression
        from session.compressor import SessionCompressor
        return SessionCompressor(llm=None).compress(messages)

    def _write_archive(
        self,
        session_id: str,
        archive_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
    ) -> dict:
        """Write archive to AGFS if available, fall back to in-memory."""
        try:
            if self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)
                    result = store.write_archive(
                        session_id=session_id,
                        overview=overview,
                        abstract=abstract,
                        messages=messages,
                        ctx=ctx,
                        archive_id=archive_id,
                    )
                    if result.success:
                        return {
                            "success": True,
                            "uri": result.uri,
                            "archive_id": result.archive_id,
                        }
                    logger.warning("AGFS write failed: %s", result.error)
        except Exception as exc:
            logger.warning("AGFS archive write failed: %s", exc, exc_info=True)

        # In-memory fallback — archive still exists in memory
        return {
            "success": True,
            "uri": f"memory://{session_id}/{archive_id}",
            "archive_id": archive_id,
        }

    def _extract_memories(
        self, messages: list[dict], ctx: RequestContext, session_time=None,
    ) -> dict:
        """Extract candidate memories from committed messages."""
        try:
            if self._get_write_api is not None:
                write_api = self._get_write_api()
                if write_api is not None:
                    result = write_api.commit_session(
                        messages, ctx, confidence_threshold=0.5, wait=True,
                        session_time=session_time,
                    )
                    return {
                        "candidates_extracted": result.get("candidates_extracted", 0),
                        "writes_completed": result.get("writes_completed", 0),
                    }
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc, exc_info=True)
        return {"candidates_extracted": 0, "writes_completed": 0}

    # -- Context assembly ---------------------------------------------------

    def get_context(
        self,
        session_id: str,
        token_budget: int,
        ctx: RequestContext,
    ) -> dict:
        """Assemble context: profile + archives + active messages."""
        buf = self.get_or_create(session_id)

        # Collect archives from AGFS
        latest_archive_overview = ""
        archive_refs: list[dict] = []

        try:
            if self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)
                    entries = store.list_archives(session_id, ctx)
                    if entries:
                        latest = entries[0]
                        latest_archive_overview = latest.overview or ""
                        for entry in entries:
                            archive_refs.append({
                                "archive_id": entry.archive_id,
                                "abstract": entry.abstract,
                            })
        except Exception as exc:
            logger.warning("get_context archive collection failed: %s", exc)

        # Estimate tokens for active messages
        active_tokens = buf.pending_tokens

        return {
            "ok": True,
            "session_id": session_id,
            "pending_tokens": active_tokens,
            "estimatedTokens": active_tokens,
            "active_message_count": len(buf.messages),
            "archive_count": len(archive_refs),
            "latest_archive_overview": latest_archive_overview,
            "archives": archive_refs,
        }

    # -- Task polling -------------------------------------------------------

    def get_task(self, task_id: str) -> dict | None:
        """Poll background task status."""
        return self._tasks.get(task_id)

    # -- Cleanup ------------------------------------------------------------

    def remove_session(self, session_id: str) -> None:
        """Remove session buffer (for dispose/cleanup)."""
        self._sessions.pop(session_id, None)
