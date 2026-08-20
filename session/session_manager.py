"""Server-side session lifecycle manager.

Manages per-session message buffers with optional AGFS persistence.
Accumulate → threshold commit → archive + extract session model.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable, Optional

from core.models import ContextNode, RequestContext
from providers.token_tracker import UsageTracker
from session.models import ArchiveEntry, SessionMessage, SessionMeta, SessionWindowState
from session.topic_buffer import SessionTopicBuffer
from session.session_state import Commitment, SessionState, TaskState

logger = logging.getLogger("ogmem.session")

_SESSION_STATE_MISSING_TTL_SECONDS = 5.0


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
    usage_stats: UsageTracker = field(default_factory=UsageTracker)
    # Unified tracker for provider/tool usage. ``tool_usage_stats`` remains
    # as a dict property for older callers and prompt rendering.

    # Incremental extraction tracking
    extraction_watermark: int = 0   # Index into self.messages after last extraction
    extraction_summary: str = ""    # Summary of already-extracted content (~500 tokens max)
    completed_request_ids: list[str] = field(default_factory=list)
    active_request_ids: set[str] = field(default_factory=set, repr=False)
    compaction_prepare_token: str = ""
    compaction_prepare_archive_id: str = ""
    compaction_prepare_message_count: int = 0
    compaction_prepare_watermark: int = 0
    compaction_prepare_token_created_at: str = ""

    # Background extraction coordination: set when after_turn spawns a
    # background thread; cleared (and event set) when the thread finishes.
    # dispose() waits on this to avoid the race where the session is removed
    # before background commit completes.
    extraction_in_progress: bool = False
    extraction_done_event: object = None  # threading.Event — typed as object to avoid import
    extraction_active_count: int = 0
    extraction_lock: object = field(default_factory=threading.RLock, repr=False)
    dispose_in_progress: bool = False

    @property
    def pending_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self.messages)

    @property
    def tool_usage_stats(self) -> dict[str, dict]:
        return self.usage_stats.tool_stats

    @tool_usage_stats.setter
    def tool_usage_stats(self, stats: dict[str, dict]) -> None:
        self.usage_stats.merge_tool_stats(stats)

    def add(self, role: str, content: str, created_at: str | None = None) -> SessionMessage:
        with self.extraction_lock:
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

    def claim_incremental_extraction(self) -> dict:
        """Atomically claim the current unextracted message range.

        The snapshot, watermark advance, and active-extraction marker must move
        together. Otherwise concurrent ``after_turn`` requests can both observe
        the same watermark and extract/archive the same messages twice.
        """
        with self.extraction_lock:
            original_watermark = self.extraction_watermark
            incremental = list(self.messages[original_watermark:])
            extraction_in_progress_before = self.extraction_in_progress
            extraction_started = bool(
                incremental and any(message.role != "assistant" for message in incremental)
            )

            if incremental:
                self.extraction_watermark = len(self.messages)
            if extraction_started:
                self.begin_extraction()

            return {
                "incremental": incremental,
                "original_watermark": original_watermark,
                "extraction_in_progress_before": extraction_in_progress_before,
                "extraction_started": extraction_started,
                "extraction_summary": self.extraction_summary,
            }

    def request_completed(self, request_id: str) -> bool:
        with self.extraction_lock:
            return request_id in self.completed_request_ids

    def claim_request(self, request_id: str) -> str:
        with self.extraction_lock:
            if request_id in self.active_request_ids:
                return "processing"
            if request_id in self.completed_request_ids:
                return "completed"
            self.active_request_ids.add(request_id)
            return "claimed"

    def release_request(self, request_id: str) -> None:
        with self.extraction_lock:
            self.active_request_ids.discard(request_id)

    def mark_request_completed(self, request_id: str, max_entries: int = 256) -> None:
        with self.extraction_lock:
            if request_id in self.completed_request_ids:
                return
            self.completed_request_ids.append(request_id)
            if len(self.completed_request_ids) > max_entries:
                del self.completed_request_ids[:-max_entries]

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
        self.compaction_prepare_archive_id = ""
        self.compaction_prepare_message_count = 0
        self.compaction_prepare_watermark = 0
        self.compaction_prepare_token_created_at = ""
        return snap

    def capture_rollback_state(self) -> dict:
        """Capture buffer fields that snapshot_and_clear mutates."""
        return {
            "messages": list(self.messages),
            "turn_count": self.turn_count,
            "window_state": deepcopy(self.window_state),
            "meta_message_count": self.meta.message_count,
            "meta_updated_at": self.meta.updated_at,
            "extraction_watermark": self.extraction_watermark,
            "extraction_summary": self.extraction_summary,
            "compaction_prepare_token": self.compaction_prepare_token,
            "compaction_prepare_archive_id": self.compaction_prepare_archive_id,
            "compaction_prepare_message_count": self.compaction_prepare_message_count,
            "compaction_prepare_watermark": self.compaction_prepare_watermark,
            "compaction_prepare_token_created_at": self.compaction_prepare_token_created_at,
        }

    def restore_rollback_state(self, state: dict) -> None:
        """Restore buffer fields after a failed archive commit."""
        original_messages = list(state.get("messages", []))
        original_ids = {msg.id for msg in original_messages}
        new_messages = [msg for msg in self.messages if msg.id not in original_ids]
        self.messages = original_messages + new_messages
        self.turn_count = sum(1 for msg in self.messages if msg.role == "user")
        self.window_state = deepcopy(state.get("window_state", SessionWindowState()))
        self.meta.message_count = len(self.messages)
        self.meta.updated_at = state.get("meta_updated_at", self.meta.updated_at)
        self.extraction_watermark = min(
            int(state.get("extraction_watermark", 0) or 0),
            len(self.messages),
        )
        self.extraction_summary = state.get("extraction_summary", "")
        self.compaction_prepare_token = state.get("compaction_prepare_token", "")
        self.compaction_prepare_archive_id = state.get("compaction_prepare_archive_id", "")
        self.compaction_prepare_message_count = int(
            state.get("compaction_prepare_message_count", 0) or 0
        )
        self.compaction_prepare_watermark = int(
            state.get("compaction_prepare_watermark", 0) or 0
        )
        self.compaction_prepare_token_created_at = state.get(
            "compaction_prepare_token_created_at", ""
        )

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

        with self.extraction_lock:
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
                self.window_state.session_state_sync_turn_count = min(
                    self.window_state.session_state_sync_turn_count,
                    self.turn_count,
                )
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
        with self.extraction_lock:
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
        get_context_fs: Callable[[], Any] | None = None,
        session_state: SessionState | None = None,
        archive_max_count: int = 10,
        archive_merge_threshold: int = 10,
        compression_quality_enabled: bool = False,
        compression_quality_persist_metadata: bool = False,
    ):
        self._sessions: dict[str, SessionBuffer] = {}
        self._get_llm = get_llm
        self._get_write_api = get_write_api
        self._get_agfs = get_agfs
        self._get_archive_store = get_archive_store
        self._archive_store_required = archive_store_required
        self._get_context_fs = get_context_fs
        self._session_state = session_state or SessionState()
        self._session_state_loaded: set[str] = set()
        self._session_state_missing_until: dict[str, float] = {}
        self._session_load_events: dict[str, threading.Event] = {}
        self._topic_buffers: dict[str, SessionTopicBuffer] = {}
        self._archive_max_count = max(1, int(archive_max_count or 10))
        self._archive_merge_threshold = max(1, int(archive_merge_threshold or 10))
        self._archive_merge_max_messages = 200
        self._compression_quality_enabled = compression_quality_enabled
        self._compression_quality_persist_metadata = compression_quality_persist_metadata

        # Background task tracking
        self._tasks: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._save_lock = threading.RLock()

    def get_session_state(self) -> SessionState:
        return self._session_state

    @staticmethod
    def _session_state_context(
        session_id: str,
        meta: SessionMeta | None,
        fallback_ctx: RequestContext,
    ) -> RequestContext:
        account_id = meta.account_id if meta and meta.account_id else fallback_ctx.account_id
        user_id = meta.user_id if meta and meta.user_id else fallback_ctx.user_id
        agent_id = meta.agent_id if meta and meta.agent_id else fallback_ctx.agent_id
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            trace_id=fallback_ctx.trace_id,
            role=fallback_ctx.role,
            visible_owner_spaces=(),
        )

    def load_session_state(self, session_id: str, ctx: RequestContext) -> bool:
        with self._lock:
            cached = self._cached_session_state_load(session_id)
            if cached is not None:
                return cached
        if self._get_context_fs is None:
            return False
        try:
            fs = self._get_context_fs()
            if fs is None:
                return False
            status, payload = self._session_state.load_with_payload_status(
                session_id, fs, ctx
            )
            if status == "loaded":
                self._restore_runtime_state_from_payload(session_id, payload or {})
                with self._lock:
                    self._session_state_loaded.add(session_id)
            elif status == "missing":
                with self._lock:
                    self._session_state_missing_until[session_id] = (
                        monotonic() + _SESSION_STATE_MISSING_TTL_SECONDS
                    )
            return status == "loaded"
        except Exception as exc:
            logger.warning("load_session_state failed session=%s: %s", session_id, exc)
            return False

    def _cached_session_state_load(self, session_id: str) -> bool | None:
        if session_id in self._session_state_loaded:
            return True
        if self._session_state_missing_until.get(session_id, 0.0) > monotonic():
            return False
        self._session_state_missing_until.pop(session_id, None)
        return None

    def save_session_state(self, session_id: str, ctx: RequestContext) -> bool:
        if self._get_context_fs is None:
            return False
        try:
            fs = self._get_context_fs()
            if fs is None:
                return False
            with self._save_lock:
                with self._lock:
                    buf = self._sessions.get(session_id)
                    if buf is not None:
                        with buf.extraction_lock:
                            self._sync_runtime_state_to_session_state(session_id, buf)
                            payload = self._session_state.to_dict(
                                session_id,
                                session_meta=deepcopy(buf.meta),
                                window_state=deepcopy(buf.window_state),
                                extraction_summary=buf.extraction_summary,
                            )
                            payload["completed_request_ids"] = list(buf.completed_request_ids)
                            payload["buffer_messages"] = [
                                {
                                    "id": message.id,
                                    "role": message.role,
                                    "content": message.content,
                                    "created_at": message.created_at,
                                }
                                for message in buf.messages
                            ]
                            payload["extraction_watermark"] = buf.extraction_watermark
                    else:
                        payload = self._session_state.to_dict(session_id)
                now = datetime.now(timezone.utc).isoformat()
                node = ContextNode(
                    uri=self._session_state.state_uri(ctx.account_id, session_id),
                    context_type="RESOURCE",
                    category="state",
                    level=0,
                    owner_space=f"session:{session_id}",
                    abstract="Session task state and commitments",
                    overview="Session task state and commitments",
                    content=json.dumps(payload, ensure_ascii=False, indent=2),
                    metadata={
                        "session_id": session_id,
                        "updated_at": now,
                    },
                )
                fs.write_node(node, ctx)
                with self._lock:
                    self._session_state_loaded.add(session_id)
                    self._session_state_missing_until.pop(session_id, None)
            return True
        except Exception as exc:
            logger.warning("save_session_state failed session=%s: %s", session_id, exc)
            return False

    def _sync_runtime_state_to_session_state(
        self,
        session_id: str,
        buf: SessionBuffer,
    ) -> None:
        """Promote runtime window state into durable SessionState before save."""
        window_state = buf.window_state
        current = self._session_state.get_task_state(session_id)
        task_update = TaskState()
        if not current.objective and window_state.active_task:
            task_update.objective = window_state.active_task
        if not current.blockers and window_state.uncertainties:
            task_update.blockers = list(window_state.uncertainties)
        if task_update.objective is not None or task_update.blockers:
            self._session_state.update_task_state(session_id, task_update)

        existing = {
            (item.content, item.status, item.kind)
            for item in self._session_state.get_commitments(session_id, status=None)
        }

        def add_missing(content: str, status: str, kind: str) -> None:
            content = str(content or "").strip()
            if not content:
                return
            key = (content, status, kind)
            if key in existing:
                return
            self._session_state.add_commitment(
                session_id,
                Commitment(content=content, status=status, kind=kind),
            )
            existing.add(key)

        for content in window_state.open_loops:
            add_missing(content, "open", "loop")
        for content in window_state.confirmed_constraints:
            add_missing(content, "fulfilled", "constraint")
        for content in window_state.recent_decisions:
            add_missing(content, "fulfilled", "decision")

    @staticmethod
    def _restore_dataclass_fields(target: Any, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        for item in dataclass_fields(target):
            name = item.name
            if name not in payload:
                continue
            value = payload.get(name)
            current = getattr(target, name)
            if isinstance(current, list):
                setattr(target, name, list(value) if isinstance(value, list) else [])
            elif isinstance(current, int):
                try:
                    setattr(target, name, int(value or 0))
                except (TypeError, ValueError):
                    setattr(target, name, 0)
            elif isinstance(current, str):
                setattr(target, name, str(value or ""))
            else:
                setattr(target, name, value)

    def _restore_runtime_state_from_payload(self, session_id: str, payload: dict) -> None:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return
            self._restore_dataclass_fields(buf.meta, payload.get("session_meta") or {})
            buf.meta.session_id = session_id
            self._restore_dataclass_fields(buf.window_state, payload.get("window_state") or {})
            extraction_summary = payload.get("extraction_summary")
            if isinstance(extraction_summary, str):
                buf.extraction_summary = extraction_summary
            completed_request_ids = payload.get("completed_request_ids")
            if isinstance(completed_request_ids, list):
                buf.completed_request_ids = [
                    str(value)
                    for value in completed_request_ids[-256:]
                    if value and len(str(value)) <= 128
                ]
            buffer_messages = payload.get("buffer_messages")
            if isinstance(buffer_messages, list):
                restored_messages = []
                for value in buffer_messages:
                    if not isinstance(value, dict):
                        continue
                    message_id = value.get("id")
                    role = value.get("role")
                    content = value.get("content")
                    created_at = value.get("created_at")
                    if not all(
                        isinstance(item, str)
                        for item in (message_id, role, content, created_at)
                    ):
                        continue
                    restored_messages.append(SessionMessage(
                        id=message_id,
                        role=role,
                        content=content,
                        created_at=created_at,
                    ))
                buf.messages = restored_messages
                buf.turn_count = sum(1 for message in restored_messages if message.role == "user")
                buf.meta.message_count = len(restored_messages)
                try:
                    extraction_watermark = int(payload.get("extraction_watermark", 0) or 0)
                except (TypeError, ValueError):
                    extraction_watermark = 0
                buf.extraction_watermark = max(0, min(extraction_watermark, len(restored_messages)))

            if self._session_state.has_state(session_id):
                try:
                    from session.session_state_bridge import apply_session_state_bridge

                    apply_session_state_bridge(
                        buf.window_state,
                        self._session_state,
                        session_id,
                        turn_count=buf.turn_count,
                        force=True,
                    )
                except Exception as exc:
                    logger.warning("restore session state bridge failed session=%s: %s", session_id, exc)

    def get_topic_buffer(self, session_id: str) -> SessionTopicBuffer:
        """Get or create the per-session topic buffer."""
        with self._lock:
            if session_id not in self._topic_buffers:
                self._topic_buffers[session_id] = SessionTopicBuffer(session_id)
            return self._topic_buffers[session_id]

    def _maybe_merge_archives(
        self,
        session_id: str,
        ctx: RequestContext,
        wait: bool = False,
    ) -> None:
        if self._get_archive_store is None:
            return

        def _run() -> None:
            try:
                self._merge_archives_once(session_id, ctx)
            except Exception as exc:
                logger.warning("archive auto-merge failed session=%s: %s", session_id, exc)

        if wait:
            _run()
            return

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"archive-merge-{session_id[:8]}",
        )
        thread.start()

    def _merge_archives_once(self, session_id: str, ctx: RequestContext) -> bool:
        store = self._get_archive_store()
        if store is None:
            return False

        entries = store.list_archives(session_id, ctx)
        if len(entries) < 2 or len(entries) < self._archive_merge_threshold:
            return False

        sorted_entries = sorted(entries, key=lambda entry: entry.created_at or "")
        merge_count = max(0, len(sorted_entries) - self._archive_max_count + 1)
        merge_count = max(2, merge_count)
        to_merge_entries = sorted_entries[:merge_count]

        full_entries = []
        for entry in to_merge_entries:
            merged_entry = store.read_archive(session_id, entry.archive_id, ctx)
            full_entries.append(merged_entry if isinstance(merged_entry, ArchiveEntry) else entry)

        from session.archive_merger import ArchiveMerger

        merged = ArchiveMerger(max_messages=self._archive_merge_max_messages).merge(full_entries)
        if not merged.overview and not merged.messages:
            return False

        merged_archive_id = (
            f"merged_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        result = store.write_archive(
            session_id=session_id,
            overview=merged.overview,
            abstract=merged.abstract,
            messages=merged.messages,
            ctx=ctx,
            archive_id=merged_archive_id,
            metadata=merged.metadata,
        )
        if not getattr(result, "success", False):
            return False

        merged_archive_id = getattr(result, "archive_id", merged_archive_id) or merged_archive_id
        marked_archive_ids: list[str] = []
        if hasattr(store, "mark_archive_merged"):
            for entry in to_merge_entries:
                try:
                    ok = store.mark_archive_merged(
                        session_id,
                        entry.archive_id,
                        ctx,
                        merged_archive_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "archive merge mark failed session=%s archive=%s: %s",
                        session_id,
                        entry.archive_id,
                        exc,
                    )
                    ok = False
                if not ok:
                    if hasattr(store, "unmark_archive_merged"):
                        for marked_id in marked_archive_ids:
                            try:
                                store.unmark_archive_merged(
                                    session_id,
                                    marked_id,
                                    ctx,
                                    merged_archive_id,
                                )
                            except Exception:
                                logger.warning(
                                    "archive merge unmark failed session=%s archive=%s",
                                    session_id,
                                    marked_id,
                                )
                    if hasattr(store, "delete_archive"):
                        try:
                            store.delete_archive(session_id, merged_archive_id, ctx)
                        except Exception:
                            logger.warning(
                                "archive merge cleanup delete failed session=%s archive=%s",
                                session_id,
                                merged_archive_id,
                            )
                    return False
                marked_archive_ids.append(entry.archive_id)
        self.invalidate_topic_slot(session_id, "archive_history")
        return True

    @staticmethod
    def _commit_succeeded(task_info: dict) -> bool:
        return (
            task_info.get("status") == "completed"
            and task_info.get("archived", False) is True
        )

    def _update_task(self, task_info: dict, **fields) -> dict:
        with self._lock:
            task_info.update(fields)
            return dict(task_info)

    def _task_snapshot(self, task_info: dict) -> dict:
        with self._lock:
            return dict(task_info)

    # -- Buffer lifecycle ---------------------------------------------------

    def get_or_create(self, session_id: str, ctx: RequestContext | None = None) -> SessionBuffer:
        """Get existing buffer or create a new one.

        On first creation, binds identity (account_id, user_id, agent_id)
        from *ctx* into the SessionMeta so subsequent requests can inherit it.
        """
        load_event = None
        should_load = False
        wait_for_load = False
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                meta = SessionMeta(session_id=session_id)
                if ctx is not None:
                    meta.account_id = ctx.account_id
                    meta.user_id = ctx.user_id
                    meta.agent_id = ctx.agent_id
                buf = SessionBuffer(
                    session_id=session_id, meta=meta
                )
                self._sessions[session_id] = buf
                logger.debug("Created new session buffer: %s", session_id)
            elif ctx is not None:
                self._ensure_session_visible(buf, ctx)
            load_event = self._session_load_events.get(session_id)
            if load_event is not None and not load_event.is_set():
                wait_for_load = True
            elif ctx is not None and self._cached_session_state_load(session_id) is None:
                should_load = True
                load_event = threading.Event()
                self._session_load_events[session_id] = load_event
        if should_load and load_event is not None:
            try:
                self.load_session_state(session_id, ctx)
            finally:
                with self._lock:
                    pending_event = self._session_load_events.pop(session_id, None)
                    if pending_event is not None:
                        pending_event.set()
        elif wait_for_load and load_event is not None:
            load_event.wait()
        if ctx is not None:
            self._ensure_session_visible(buf, ctx)
        return buf

    def _ensure_session_visible(self, buf: SessionBuffer, ctx: RequestContext) -> None:
        if not self._session_visible_to_ctx(buf, ctx):
            raise PermissionError("not_found_or_not_visible")

    @staticmethod
    def _session_visible_to_ctx(buf: SessionBuffer, ctx: RequestContext) -> bool:
        meta = buf.meta
        if meta.account_id and meta.account_id != ctx.account_id:
            return False
        if meta.user_id and meta.user_id != ctx.user_id:
            return False
        if meta.agent_id and meta.agent_id != ctx.agent_id:
            return False
        return True

    def get_window_state(self, session_id: str) -> SessionWindowState:
        """Get current window state for a session."""
        buf = self.get_or_create(session_id)
        with self._lock:
            return buf.window_state

    def update_window_state(
        self,
        session_id: str,
        window_state: SessionWindowState,
    ) -> None:
        """Update window state after compression."""
        buf = self.get_or_create(session_id)
        with self._lock:
            buf.window_state = window_state

    def invalidate_topic_slot(self, session_id: str, name: str) -> None:
        with self._lock:
            buffer = self._topic_buffers.get(session_id)
            if buffer is not None:
                buffer.invalidate(name)

    def list_session_working_set(self) -> list[dict]:
        with self._lock:
            rows: list[dict] = []
            for session_id, buf in self._sessions.items():
                rows.append({
                    "session_id": session_id,
                    "last_accessed_at": buf.window_state.last_accessed_at,
                    "message_count": len(buf.messages),
                    "pending_tokens": buf.pending_tokens,
                    "active_task": buf.window_state.active_task,
                })
        return sorted(
            rows,
            key=lambda row: row.get("last_accessed_at") or "",
            reverse=True,
        )

    def evict_idle_sessions(
        self,
        max_idle_seconds: int,
        now_iso: str | None = None,
        ctx: RequestContext | None = None,
    ) -> list[str]:
        if max_idle_seconds <= 0:
            return []
        now = (
            datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            if now_iso
            else datetime.now(timezone.utc)
        )
        evicted: list[str] = []
        with self._lock:
            candidates = [
                (
                    session_id,
                    buf.window_state.last_accessed_at,
                    buf.commit_in_progress,
                    buf.extraction_in_progress,
                    len(buf.messages),
                    buf.extraction_watermark,
                    deepcopy(buf.meta),
                )
                for session_id, buf in self._sessions.items()
            ]
        for (
            session_id,
            last,
            commit_in_progress,
            extraction_in_progress,
            message_count,
            extraction_watermark,
            session_meta,
        ) in candidates:
            if not last:
                continue
            try:
                accessed = datetime.fromisoformat(last.replace("Z", "+00:00"))
            except Exception:
                continue
            if (now - accessed).total_seconds() > max_idle_seconds:
                if (
                    commit_in_progress
                    or extraction_in_progress
                    or message_count > extraction_watermark
                ):
                    continue
                if ctx is not None:
                    self.save_session_state(
                        session_id,
                        self._session_state_context(session_id, session_meta, ctx),
                    )
                with self._lock:
                    buf = self._sessions.get(session_id)
                    if buf is None:
                        continue
                    if (
                        buf.commit_in_progress
                        or buf.extraction_in_progress
                        or len(buf.messages) > buf.extraction_watermark
                    ):
                        continue
                    self._sessions.pop(session_id, None)
                    self._topic_buffers.pop(session_id, None)
                    self._session_state_loaded.discard(session_id)
                    self._session_state_missing_until.pop(session_id, None)
                logger.debug("Removed session buffer: %s", session_id)
                evicted.append(session_id)
        return evicted

    def issue_compaction_prepare_token(self, session_id: str, archive_id: str = "") -> str:
        buf = self.get_or_create(session_id)
        with self._lock:
            token = uuid.uuid4().hex
            buf.compaction_prepare_token = token
            buf.compaction_prepare_archive_id = archive_id
            buf.compaction_prepare_message_count = len(buf.messages)
            buf.compaction_prepare_watermark = buf.extraction_watermark
            buf.compaction_prepare_token_created_at = datetime.now(timezone.utc).isoformat()
            return token

    def get_compaction_prepare_archive_id(self, session_id: str) -> str:
        buf = self.get_or_create(session_id)
        with self._lock:
            return buf.compaction_prepare_archive_id

    def get_compaction_prepare_state(
        self,
        session_id: str,
        ttl_seconds: int = 300,
    ) -> dict | None:
        buf = self.get_or_create(session_id)
        with self._lock:
            token = buf.compaction_prepare_token
            if not token:
                return None
            if not self._compaction_prepare_token_is_fresh(
                buf.compaction_prepare_token_created_at,
                ttl_seconds,
            ):
                self._clear_compaction_prepare_token_locked(buf)
                return None
            if (
                len(buf.messages) != buf.compaction_prepare_message_count
                or buf.extraction_watermark != buf.compaction_prepare_watermark
            ):
                self._clear_compaction_prepare_token_locked(buf)
                return None
            return {
                "prepareToken": token,
                "archive_id": buf.compaction_prepare_archive_id or "",
            }

    def consume_compaction_prepare_token(
        self,
        session_id: str,
        token: str,
        ttl_seconds: int = 300,
    ) -> bool:
        buf = self.get_or_create(session_id)
        with self._lock:
            token_matches = bool(token) and token == buf.compaction_prepare_token
            message_count_matches = len(buf.messages) == buf.compaction_prepare_message_count
            watermark_matches = buf.extraction_watermark == buf.compaction_prepare_watermark
            ttl_matches = self._compaction_prepare_token_is_fresh(
                buf.compaction_prepare_token_created_at,
                ttl_seconds,
            )
            if buf.compaction_prepare_token and not ttl_matches:
                self._clear_compaction_prepare_token_locked(buf)
                return False
            is_valid = token_matches and message_count_matches and watermark_matches and ttl_matches
            if is_valid:
                self._clear_compaction_prepare_token_locked(buf)
            return is_valid

    def clear_compaction_prepare_token(self, session_id: str) -> None:
        buf = self.get_or_create(session_id)
        with self._lock:
            self._clear_compaction_prepare_token_locked(buf)

    @staticmethod
    def _clear_compaction_prepare_token_locked(buf: SessionBuffer) -> None:
        buf.compaction_prepare_token = ""
        buf.compaction_prepare_archive_id = ""
        buf.compaction_prepare_message_count = 0
        buf.compaction_prepare_watermark = 0
        buf.compaction_prepare_token_created_at = ""

    @staticmethod
    def _compaction_prepare_token_is_fresh(created_at: str, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return False
        if not created_at:
            return False
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo or timezone.utc)
            return (now - created).total_seconds() < ttl_seconds
        except Exception:
            return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        ctx: RequestContext,
        created_at: str | None = None,
    ) -> dict:
        """Add a message to the session buffer."""
        while True:
            buf = self.get_or_create(session_id, ctx=ctx)
            with self._lock:
                # A concurrent dispose may remove the buffer after get_or_create()
                # returns. Never append to an object that is no longer managed.
                if self._sessions.get(session_id) is not buf:
                    continue
                msg = buf.add(role, content, created_at=created_at)
                pending_tokens = buf.pending_tokens
                return {
                    "ok": True,
                    "message_id": msg.id,
                    "pending_tokens": pending_tokens,
                }

    def get_session(self, session_id: str, ctx: RequestContext) -> dict:
        """Return session meta + pending_tokens."""
        buf = self.get_or_create(session_id, ctx=ctx)
        with self._lock:
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
        archive_id: str | None = None,
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
        buf = self.get_or_create(session_id, ctx=ctx)

        with self._lock:
            if not buf.messages:
                return {"ok": True, "archived": False, "reason": "empty_buffer"}

            if buf.commit_in_progress:
                return {"ok": True, "archived": False, "reason": "commit_in_progress"}

            # Phase 1: snapshot + clear
            buf.commit_in_progress = True
            self._sync_runtime_state_to_session_state(session_id, buf)
            rollback_state = buf.capture_rollback_state()
            snapshot = buf.snapshot_and_clear()

            if archive_id is None:
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
            try:
                self._process_snapshot(
                    snapshot, session_id, archive_id, task_info, ctx
                )
            except Exception as exc:
                logger.error(
                    "commit phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            with self._lock:
                buf.commit_in_progress = False
                buf.meta.commit_count += 1
                buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                task_snapshot = dict(task_info)
                should_save = self._commit_succeeded(task_snapshot)
                if task_snapshot.get("status") == "failed":
                    buf.restore_rollback_state(rollback_state)
            if should_save:
                self.save_session_state(session_id, ctx)
            return {
                "ok": True,
                "archived": task_snapshot.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_snapshot.get("status", "completed"),
                "error": task_snapshot.get("error"),
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
                self._update_task(task_info, status="failed", error=str(exc))
            finally:
                with self._lock:
                    buf.commit_in_progress = False
                    buf.meta.commit_count += 1
                    buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                    task_snapshot = dict(task_info)
                    should_save = self._commit_succeeded(task_snapshot)
                    if task_snapshot.get("status") == "failed":
                        buf.restore_rollback_state(rollback_state)
                if should_save:
                    self.save_session_state(session_id, ctx)

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
        with self._lock:
            if archive_id is None:
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
            snapshot_copy = list(snapshot)

        if wait:
            try:
                self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx)
            except Exception as exc:
                logger.error(
                    "commit_snapshot phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            with self._lock:
                buf.meta.commit_count += 1
                buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                task_snapshot = dict(task_info)
                should_save = self._commit_succeeded(task_snapshot)
            if should_save:
                self.save_session_state(session_id, ctx)
            return {
                "ok": True,
                "archived": task_snapshot.get("archived", False),
                "archive_id": archive_id,
                "task_id": task_id,
                "status": task_snapshot.get("status", "completed"),
                "error": task_snapshot.get("error"),
            }

        def _commit_snapshot_thread():
            try:
                self._process_snapshot(snapshot_copy, session_id, archive_id, task_info, ctx)
            except Exception as exc:
                logger.error(
                    "commit_snapshot phase2 failed session=%s: %s",
                    session_id, exc, exc_info=True,
                )
                self._update_task(task_info, status="failed", error=str(exc))
            finally:
                with self._lock:
                    buf.meta.commit_count += 1
                    buf.meta.last_commit_at = datetime.now(timezone.utc).isoformat()
                    task_snapshot = dict(task_info)
                    should_save = self._commit_succeeded(task_snapshot)
                if should_save:
                    self.save_session_state(session_id, ctx)

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
        archive_metadata: dict = {}
        if self._compression_quality_enabled:
            try:
                from session.compression_quality import CompressionQualityEvaluator

                metrics = CompressionQualityEvaluator().evaluate(
                    messages_dicts,
                    overview,
                    abstract,
                )
                self._update_task(task_info, compression_quality=metrics)
                if self._compression_quality_persist_metadata:
                    archive_metadata["compression_quality"] = metrics
            except Exception as exc:
                logger.warning(
                    "compression quality evaluation failed session=%s: %s",
                    session_id,
                    exc,
                )
        archive_result = self._write_archive(
            session_id,
            archive_id,
            overview,
            abstract,
            messages_dicts,
            ctx,
            metadata=archive_metadata,
        )

        self._update_task(
            task_info,
            archived=archive_result.get("success", False),
            archive_uri=archive_result.get("uri", ""),
        )
        if not archive_result.get("success", False):
            self._update_task(
                task_info,
                status="failed",
                error=archive_result.get("error", "archive write failed"),
            )
            return

        self._update_task(task_info, status="completed")
        self.invalidate_topic_slot(session_id, "archive_history")
        self._maybe_merge_archives(session_id, ctx)

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
        metadata: dict | None = None,
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
                    metadata=metadata or {},
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
                    with self._lock:
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
                            with self._lock:
                                appended = (
                                    buf.extraction_summary + "\n" + "\n".join(new_items)
                                ).strip()
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
        with self._lock:
            active_tokens = buf.pending_tokens
            active_message_count = len(buf.messages)

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

        return {
            "ok": True,
            "session_id": session_id,
            "pending_tokens": active_tokens,
            "estimatedTokens": active_tokens,
            "active_message_count": active_message_count,
            "archive_count": len(archive_refs),
            "latest_archive_overview": latest_archive_overview,
            "archives": archive_refs,
        }

    # -- Task polling -------------------------------------------------------

    def get_task(self, task_id: str) -> dict | None:
        """Poll background task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task is not None else None

    # -- Cleanup ------------------------------------------------------------

    def has_session(self, session_id: str, ctx: RequestContext | None = None) -> bool:
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return False
            if ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return False
            return True

    def remove_session(self, session_id: str, ctx: RequestContext | None = None) -> None:
        """Remove session buffer (for dispose/cleanup)."""
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is not None and ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return
            removed = self._sessions.pop(session_id, None)
            self._topic_buffers.pop(session_id, None)
            self._session_state_loaded.discard(session_id)
            self._session_state_missing_until.pop(session_id, None)
        if removed:
            logger.debug("Removed session buffer: %s", session_id)

    def remove_session_if_empty(
        self,
        session_id: str,
        expected_buffer: SessionBuffer,
        ctx: RequestContext | None = None,
    ) -> bool:
        """Remove only the expected buffer when no unextracted work remains.

        A dispose background task must not remove a replacement buffer or a
        buffer that received newer messages after its snapshot was claimed.
        """
        with self._lock:
            buf = self._sessions.get(session_id)
            if buf is not expected_buffer:
                return False
            if ctx is not None and not self._session_visible_to_ctx(buf, ctx):
                return False
            with buf.extraction_lock:
                if (
                    buf.extraction_in_progress
                    or len(buf.messages) > buf.extraction_watermark
                ):
                    return False
                self._sessions.pop(session_id, None)
                self._topic_buffers.pop(session_id, None)
                self._session_state_loaded.discard(session_id)
                self._session_state_missing_until.pop(session_id, None)
        logger.debug("Removed empty session buffer: %s", session_id)
        return True

    def get_pending_session_ids(self) -> list[str]:
        """Return IDs of sessions with unextracted messages."""
        with self._lock:
            return [
                sid for sid, buf in self._sessions.items()
                if buf.extraction_watermark < len(buf.messages)
            ]
