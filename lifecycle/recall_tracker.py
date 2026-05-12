"""Thread-safe recall event tracker with batch buffering.

Buffers recall events in memory and flushes them to SQLRecallStore
when the buffer reaches a configurable size or on manual flush.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from core.models import RequestContext
from lifecycle.recall_store import RecallEvent

logger = logging.getLogger(__name__)


class RecallTracker:
    """Thread-safe recall event tracker with batch buffering.

    Buffers recall events in memory and automatically flushes to
    SQLRecallStore when buffer_size is reached. Also supports
    manual flushing via stop() or flush().
    """

    def __init__(
        self,
        buffer_size: int = 100,
        store=None,
    ):
        """Initialize the recall tracker.

        Args:
            buffer_size: Number of events to buffer before auto-flush
            store: SQLRecallStore instance (if None, flush becomes no-op)
        """
        self._buffer: list[RecallEvent] = []
        self._buffer_size = buffer_size
        self._store = store
        self._lock = threading.Lock()
        self._stopped = False

    def record_recalls(
        self,
        result: Any,
        query: str,
        ctx: RequestContext,
    ) -> None:
        """Record recall events from a retrieval result.

        Args:
            result: Retrieval result object (expected to have hits/results attribute)
            query: The search query text
            ctx: Request context with account_id and owner_space
        """
        if self._stopped:
            logger.warning("Cannot record recalls: tracker is stopped")
            return

        # Extract hits from the result object
        hits = []
        if hasattr(result, "hits"):
            hits = result.hits
        elif hasattr(result, "results"):
            hits = result.results
        elif isinstance(result, list):
            hits = result
        else:
            logger.debug(f"Unknown result type: {type(result)}")
            return

        if not hits:
            return

        events = []
        for hit in hits:
            # Extract URI from hit
            uri = None
            if hasattr(hit, "uri"):
                uri = hit.uri
            elif hasattr(hit, "node") and hasattr(hit.node, "uri"):
                uri = hit.node.uri
            elif isinstance(hit, dict):
                uri = hit.get("uri") or hit.get("node", {}).get("uri")

            if not uri:
                continue

            # Extract score from hit
            score = None
            if hasattr(hit, "score"):
                score = hit.score
            elif isinstance(hit, dict):
                score = hit.get("score")

            # Extract category from hit
            category = None
            if hasattr(hit, "category"):
                category = hit.category
            elif hasattr(hit, "node") and hasattr(hit.node, "category"):
                category = hit.node.category
            elif isinstance(hit, dict):
                category = hit.get("category") or hit.get("node", {}).get("category")

            events.append(
                RecallEvent(
                    uri=uri,
                    account_id=ctx.account_id,
                    owner_space=ctx.user_space_name(),
                    query_text=query,
                    score=score,
                    category=category,
                )
            )

        if not events:
            return

        # Thread-safe buffering
        with self._lock:
            if self._stopped:
                return
            self._buffer.extend(events)
            logger.debug(f"Buffered {len(events)} recall events (total: {len(self._buffer)})")

            # Auto-flush if buffer is full
            if len(self._buffer) >= self._buffer_size:
                self._flush_unlocked()

    def flush(self, ctx: RequestContext) -> bool:
        """Manually flush buffered events to store.

        Returns True if successful, False otherwise.
        """
        with self._lock:
            return self._flush_unlocked(ctx)

    def _flush_unlocked(self, ctx: RequestContext | None = None) -> bool:
        """Internal flush method (must hold lock).

        Args:
            ctx: Request context (required for first flush, optional for auto-flush)

        Returns:
            True if successful, False otherwise
        """
        if not self._buffer:
            return True

        if self._store is None:
            logger.debug("No store configured, discarding buffered events")
            self._buffer.clear()
            return True

        # Use context from first event if ctx not provided
        if ctx is None and self._buffer:
            # Create a minimal context from the first event
            from core.models import RequestContext
            first_event = self._buffer[0]
            ctx = RequestContext(
                account_id=first_event.account_id,
                user_id=first_event.owner_space.replace("user:", "") if first_event.owner_space.startswith("user:") else first_event.owner_space,
                agent_id="dream",
                session_id="dream-flush",
                trace_id="0",
            )

        if ctx is None:
            logger.warning("Cannot flush: no context available")
            return False

        events = self._buffer.copy()
        self._buffer.clear()

        try:
            success = self._store.batch_insert(events, ctx)
            if success:
                logger.debug(f"Flushed {len(events)} recall events to store")
            else:
                logger.error(f"Failed to flush {len(events)} recall events")
            return success
        except Exception as exc:
            logger.error(f"Error flushing recall events: {exc}")
            return False

    def stop(self, ctx: RequestContext) -> None:
        """Stop the tracker and flush remaining events.

        Args:
            ctx: Request context for final flush
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._flush_unlocked(ctx)

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def is_stopped(self) -> bool:
        """Check if tracker is stopped."""
        with self._lock:
            return self._stopped
