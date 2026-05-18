"""Server-side session lifecycle manager.

Manages per-session message buffers with optional AGFS persistence.
Accumulate → threshold commit → archive + extract session model.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.models import RequestContext
from session.models import SessionMessage, SessionMeta, SessionWindowState

logger = logging.getLogger("ogmem.session")


def generate_archive_id() -> str:
    """Generate a unique archive_id.

    Format: {timestamp}_{uuid8} e.g., '20260515_100000_a1b2c3d4'
    """
    return (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )

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
    compaction_prepare_token: str = ""
    compaction_prepare_message_count: int = 0
    compaction_prepare_watermark: int = 0

    # Background extraction coordination: set when after_turn spawns a
    # background thread; cleared (and event set) when the thread finishes.
    # dispose() waits on this to avoid the race where the session is removed
    # before background commit completes.
    extraction_in_progress: bool = False
    extraction_done_event: object = None  # threading.Event — typed as object to avoid import
    extraction_active_count: int = 0
    extraction_lock: object = field(default_factory=threading.Lock, repr=False)

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
        """Return a copy of messages and clear the buffer.

        Note: extraction_watermark and extraction_summary are preserved across
        snapshots because they track which memories have been extracted, not
        which messages are in the buffer.
        """
        snap = list(self.messages)
        extracted_count = self.extraction_watermark
        summary = self.extraction_summary
        self.messages.clear()
        self.turn_count = 0
        self.window_state = SessionWindowState()
        self.meta.message_count = 0
        self.meta.updated_at = datetime.now(timezone.utc).isoformat()
        # Preserve extraction state — watermark is relative to messages,
        # so adjust for the cleared buffer
        self.extraction_watermark = 0
        self.extraction_summary = summary
        self.compaction_prepare_token = ""
        self.compaction_prepare_message_count = 0
        self.compaction_prepare_watermark = 0
        return snap

    def begin_extraction(self):
        """Mark a background extraction as active and return the shared done event."""
        with self.extraction_lock:
            event = self.extraction_done_event
            if event is None or not self.extraction_in_progress or getattr(event, "is_set", lambda: False)():
                event = threading.Event()
                self.extraction_done_event = event
            self.extraction_active_count += 1
            self.extraction_in_progress = True
            return event

    def end_extraction(self) -> None:
        """Mark one background extraction as finished."""
        with self.extraction_lock:
            if self.extraction_active_count > 0:
                self.extraction_active_count -= 1
            if self.extraction_active_count <= 0:
                self.extraction_active_count = 0
                self.extraction_in_progress = False
                event = self.extraction_done_event
                if event is not None:
                    event.set()

    def remove_messages_by_id(self, message_ids: set[str]) -> int:
        """Remove archived snapshot messages without touching newer messages."""
        if not message_ids:
            return 0

        kept: list[SessionMessage] = []
        removed = 0
        removed_before_watermark = 0
        for idx, msg in enumerate(self.messages):
            if msg.id in message_ids:
                removed += 1
                if idx < self.extraction_watermark:
                    removed_before_watermark += 1
            else:
                kept.append(msg)

        if removed:
            self.messages = kept
            self.turn_count = sum(1 for msg in self.messages if msg.role == "user")
            self.extraction_watermark = max(
                0,
                min(len(self.messages), self.extraction_watermark - removed_before_watermark),
            )
            self.meta.message_count = len(self.messages)
            self.meta.updated_at = datetime.now(timezone.utc).isoformat()
        return removed

    def rewind_watermark_to_ids(self, message_ids: set[str]) -> bool:
        """Rewind extraction watermark to the first remaining failed message."""
        if not message_ids:
            return False
        for idx, msg in enumerate(self.messages):
            if msg.id in message_ids:
                if idx < self.extraction_watermark:
                    self.extraction_watermark = idx
                    self.meta.updated_at = datetime.now(timezone.utc).isoformat()
                    return True
                return False
        return False


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
        get_archive_store: Callable[[], Any] | None = None,
        archive_store_required: bool = False,
    ):
        self._sessions: dict[str, SessionBuffer] = {}
        self._get_llm = get_llm
        self._get_write_api = get_write_api
        self._get_agfs = get_agfs
        self._get_archive_store = get_archive_store
        self._archive_store_required = archive_store_required

        # Background task tracking
        self._tasks: dict[str, dict] = {}

    # -- Buffer lifecycle ---------------------------------------------------

    def get_or_create(self, session_id: str, ctx: RequestContext | None = None) -> SessionBuffer:
        """Get existing buffer or create a new one.

        On first creation, binds identity (account_id, user_id, agent_id)
        from *ctx* into the SessionMeta so subsequent requests can inherit it.
        """
        if session_id not in self._sessions:
            meta = SessionMeta(session_id=session_id)
            if ctx is not None:
                meta.account_id = ctx.account_id
                meta.user_id = ctx.user_id
                meta.agent_id = ctx.agent_id
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

    def issue_compaction_prepare_token(self, session_id: str) -> str:
        buf = self.get_or_create(session_id)
        token = uuid.uuid4().hex
        buf.compaction_prepare_token = token
        buf.compaction_prepare_message_count = len(buf.messages)
        buf.compaction_prepare_watermark = buf.extraction_watermark
        return token

    def consume_compaction_prepare_token(self, session_id: str, token: str) -> bool:
        buf = self.get_or_create(session_id)
        token_matches = bool(token) and token == buf.compaction_prepare_token
        message_count_matches = len(buf.messages) == buf.compaction_prepare_message_count
        watermark_matches = buf.extraction_watermark == buf.compaction_prepare_watermark
        is_valid = token_matches and message_count_matches and watermark_matches
        if is_valid:
            self.clear_compaction_prepare_token(session_id)
        return is_valid

    def clear_compaction_prepare_token(self, session_id: str) -> None:
        buf = self.get_or_create(session_id)
        buf.compaction_prepare_token = ""
        buf.compaction_prepare_message_count = 0
        buf.compaction_prepare_watermark = 0

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
        Phase 2 (background thread): generate overview/abstract from archive.
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

        archive_id = generate_archive_id()

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
            self._process_snapshot(
                snapshot, session_id, archive_id, task_info, ctx
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
                "error": task_info.get("error"),
            }

        # Fire background thread
        def _commit_thread():
            try:
                self._process_snapshot(
                    snapshot, session_id, archive_id, task_info, ctx
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

        t = threading.Thread(target=_commit_thread, daemon=True, name=f"commit-{session_id[:8]}")
        t.start()

        return {
            "ok": True,
            "archived": False,
            "archive_id": archive_id,
            "task_id": task_id,
            "status": "processing",
        }

    def commit_snapshot(
        self,
        session_id: str,
        snapshot: list[SessionMessage],
        ctx: RequestContext,
        wait: bool = True,
        archive_id: str | None = None,
    ) -> dict:
        """Archive a fixed message snapshot without reading/clearing live buffer."""
        if not snapshot:
            return {"ok": True, "archived": False, "reason": "empty_snapshot"}

        buf = self.get_or_create(session_id)
        if archive_id is None:
            raise ValueError(
                "archive_id is required — callers (e.g. after_turn) pre-generate "
                "it so that commit_session and commit_snapshot share the same ID"
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

        snapshot_copy = list(snapshot)

        if wait:
            self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx)
            buf.meta.commit_count += 1
            buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
            return {
                "ok": True,
                "archived": task_info.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_info.get("status", "completed"),
            }

        def _commit_snapshot_thread():
            try:
                self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx)
            except Exception as exc:
                logger.error(
                    "commit_snapshot phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                task_info["status"] = "failed"
                task_info["error"] = str(exc)
            finally:
                buf.meta.commit_count += 1
                buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()

        t = threading.Thread(
            target=_commit_snapshot_thread,
            daemon=True,
            name=f"commit-snapshot-{session_id[:8]}",
        )
        t.start()

        return {
            "ok": True,
            "archived": False,
            "archive_id": archive_id,
            "task_id": task_id,
            "status": "processing",
        }

    def _process_snapshot(
        self,
        snapshot: list[SessionMessage],
        session_id: str,
        archive_id: str,
        task_info: dict,
        ctx: RequestContext,
        skip_extraction: bool = False,
        session_time=None,
    ) -> None:
        """Background work: compress → write archive.

        Extraction is NOT done here — it is the caller's responsibility
        (e.g. after_turn or compact in MemoryService) to extract memories
        before calling commit().  This prevents double extraction.
        """
        messages_dicts = [
            {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at}
            for m in snapshot
        ]

        prev_overview, prev_abstract = self._get_latest_archive_context(
            session_id, ctx,
        )
        overview, abstract = self._compress(
            messages_dicts,
            prev_overview=prev_overview,
            prev_abstract=prev_abstract,
        )
        archive_result = self._write_archive(
            session_id, archive_id, overview, abstract, messages_dicts, ctx
        )

        task_info["archived"] = archive_result.get("success", False)
        task_info["archive_uri"] = archive_result.get("uri", "")
        if not archive_result.get("success", False):
            task_info["status"] = "failed"
            task_info["error"] = archive_result.get("error", "archive write failed")
            return

        task_info["status"] = "completed"

    def _compress(
        self,
        messages: list[dict],
        prev_overview: str = "",
        prev_abstract: str = "",
    ) -> tuple[str, str]:
        """Compress messages into overview + abstract, fusing with previous archive."""
        try:
            if self._get_llm is not None:
                llm = self._get_llm()
                if llm is not None:
                    from session.compressor import SessionCompressor
                    compressor = SessionCompressor(llm=llm)
                    return compressor.compress(
                        messages,
                        prev_overview=prev_overview,
                        prev_abstract=prev_abstract,
                    )
        except Exception as exc:
            logger.warning("LLM compress failed, using fallback: %s", exc)

        from session.compressor import SessionCompressor
        return SessionCompressor(llm=None).compress(
            messages,
            prev_overview=prev_overview,
            prev_abstract=prev_abstract,
        )

    def _write_archive(
        self,
        session_id: str,
        archive_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
    ) -> dict:
        """Write archive via archive store if available.

        When ``archive_store_required`` is enabled, archive persistence
        failures are surfaced instead of silently degrading to an in-memory
        URI.
        """
        try:
            store = None
            if self._get_archive_store is not None:
                store = self._get_archive_store()
                if store is None and self._archive_store_required:
                    return {
                        "success": False,
                        "uri": "",
                        "archive_id": archive_id,
                        "error": "archive store unavailable",
                    }
            elif self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)

            if store is not None:
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
                logger.warning("Archive write failed: %s", result.error)
                if self._archive_store_required:
                    return {
                        "success": False,
                        "uri": result.uri,
                        "archive_id": archive_id,
                        "error": result.error or "archive write failed",
                    }
        except Exception as exc:
            logger.warning("Archive write failed: %s", exc, exc_info=True)
            if self._archive_store_required:
                return {
                    "success": False,
                    "uri": "",
                    "archive_id": archive_id,
                    "error": str(exc),
                }

        # In-memory fallback — archive still exists in memory
        return {
            "success": True,
            "uri": f"memory://{session_id}/{archive_id}",
            "archive_id": archive_id,
        }

    def _get_latest_archive_context(
        self, session_id: str, ctx: RequestContext,
    ) -> tuple[str, str]:
        """Fetch overview and abstract from the most recent archive for fusion.

        Returns:
            (prev_overview, prev_abstract) — empty strings if none found.
        """
        def _latest_from_entries(entries) -> tuple[str, str]:
            if not entries:
                return "", ""
            entries.sort(key=lambda e: e.created_at or "", reverse=True)
            latest = entries[0]
            return latest.overview or "", latest.abstract or ""

        if self._get_archive_store is not None:
            try:
                store = self._get_archive_store()
                if store is not None:
                    overview, abstract = _latest_from_entries(
                        store.list_archives(session_id, ctx),
                    )
                    if overview or abstract:
                        return overview, abstract
            except Exception as exc:
                logger.warning(
                    "Failed to fetch latest archive for fusion via archive store: %s",
                    exc,
                )

        if self._get_agfs is not None:
            try:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore

                    store = SessionArchiveStore(fs=agfs)
                    return _latest_from_entries(store.list_archives(session_id, ctx))
            except Exception as exc:
                logger.warning(
                    "Failed to fetch latest archive for fusion via AGFS fallback: %s",
                    exc,
                )
        return "", ""

    def _extract_memories(
        self, messages: list[dict], ctx: RequestContext,
        buf: SessionBuffer | None = None,
    ) -> dict:
        """Extract candidate memories from committed messages."""
        try:
            if self._get_write_api is not None:
                write_api = self._get_write_api()
                if write_api is not None:
                    session_summary = buf.extraction_summary if buf else ""
                    result = write_api.commit_session(
                        messages, ctx, confidence_threshold=0.5, wait=True,
                        session_summary=session_summary,
                    )
                    # Update buffer's extraction_summary after successful extraction
                    if buf and result.get("candidates_extracted", 0) > 0:
                        new_items = [
                            f"- [{p['action']}] {p['target_uri']}"
                            for p in result.get("plans", [])
                            if p["action"] != "skip"
                        ]
                        if new_items:
                            appended = (buf.extraction_summary + "\n" + "\n".join(new_items)).strip()
                            buf.extraction_summary = appended[-2000:]
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

        # Collect archives
        latest_archive_overview = ""
        archive_refs: list[dict] = []

        try:
            store = None
            if self._get_archive_store is not None:
                store = self._get_archive_store()
            elif self._get_agfs is not None:
                agfs = self._get_agfs()
                if agfs is not None:
                    from session.archive_store import SessionArchiveStore
                    store = SessionArchiveStore(fs=agfs)

            if store is not None:
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

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def remove_session(self, session_id: str) -> None:
        """Remove session buffer (for dispose/cleanup)."""
        removed = self._sessions.pop(session_id, None)
        if removed:
            logger.debug("Removed session buffer: %s", session_id)

    def get_pending_session_ids(self) -> list[str]:
        """Return IDs of sessions with unextracted messages."""
        return [
            sid for sid, buf in self._sessions.items()
            if buf.extraction_watermark < len(buf.messages)
        ]
