"""DreamService - Main orchestration for dreaming subsystem.

Coordinates LightDreaming phase and RecallTracker with APScheduler
for periodic memory consolidation.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, JobEvent
from apscheduler.schedulers.background import BackgroundScheduler

from core.interfaces import LLM, VectorIndex, Embedder
from core.models import RequestContext
from lifecycle.config import DreamConfig
from lifecycle.phases.light import LightDreaming, DreamReport
from lifecycle.recall_tracker import RecallTracker
from lifecycle.recall_store import SQLRecallStore

logger = logging.getLogger(__name__)


@dataclass
class DreamServiceMetrics:
    """Runtime metrics for DreamService."""
    service_id: str
    last_light_run: datetime | None = None
    last_light_report: DreamReport | None = None
    total_light_runs: int = 0
    total_memories_extracted: int = 0
    total_memories_written: int = 0
    total_recall_events_buffered: int = 0
    consecutive_errors: int = 0
    is_running: bool = False


class DreamService:
    """Main service for dreaming subsystem.

    Orchestrates LightDreaming phase and RecallTracker with periodic
    scheduling via APScheduler. Thread-safe singleton pattern.

    Usage:
        service = DreamService(
            llm=llm,
            write_api=write_api,
            archive_store=archive_store,
            embedder=embedder,
            vector_index=vector_index,
            connection_pool=pool,
            config=DreamConfig.from_config(config),
        )
        service.start()

        # Later...
        service.stop(ctx)
    """

    def __init__(
        self,
        llm: LLM,
        write_api: Any,
        archive_store: Any,
        embedder: Embedder,
        vector_index: VectorIndex,
        connection_pool: Any,
        config: DreamConfig,
    ):
        """Initialize the DreamService.

        Args:
            llm: LLM instance for extraction
            write_api: MemoryWriteAPI instance for writing memories
            archive_store: SQLSessionArchiveStore for reading archives
            embedder: Embedder instance for duplicate detection
            vector_index: VectorIndex instance for duplicate detection
            connection_pool: PostgreSQL connection pool for recall storage
            config: DreamConfig configuration
        """
        self._llm = llm
        self._write_api = write_api
        self._archive_store = archive_store
        self._embedder = embedder
        self._vector_index = vector_index
        self._connection_pool = connection_pool
        self._config = config

        self._service_id = f"dream-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._lock = threading.Lock()
        self._scheduler: BackgroundScheduler | None = None
        self._last_light_run: datetime | None = None

        # Initialize LightDreaming
        self._light_dreaming = LightDreaming(
            llm=llm,
            write_api=write_api,
            archive_store=archive_store,
            embedder=embedder,
            vector_index=vector_index,
        )

        # Initialize RecallTracker with SQLRecallStore
        self._recall_store = SQLRecallStore(
            connection_pool=connection_pool,
        )
        self._recall_tracker = RecallTracker(
            buffer_size=config.recall_buffer_size,
            store=self._recall_store,
        )

        self._metrics = DreamServiceMetrics(service_id=self._service_id)

    @property
    def service_id(self) -> str:
        """Return the service ID."""
        return self._service_id

    @property
    def metrics(self) -> DreamServiceMetrics:
        """Return current metrics."""
        return self._metrics

    @property
    def recall_tracker(self) -> RecallTracker:
        """Return the RecallTracker instance for pipeline integration."""
        return self._recall_tracker

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    def start(self) -> None:
        """Start the dreaming service.

        Starts the APScheduler for LightDreaming if enabled.
        """
        with self._lock:
            if self._running:
                logger.warning(
                    f"DreamService already running, service_id={self._service_id}"
                )
                return

            if not self._config.enabled:
                logger.info("DreamService disabled in config, not starting")
                return

            if self._config.light_enabled:
                self._scheduler = BackgroundScheduler()
                self._scheduler.add_job(
                    func=self._run_light_wrapped,
                    trigger="interval",
                    seconds=self._config.light_interval_seconds,
                    max_instances=1,
                    misfire_grace_time=300,  # 5 minutes
                    coalesce=True,
                    id="light_dreaming",
                    name=f"light_dreaming_{self._service_id}",
                )

                self._scheduler.add_listener(
                    self._on_job_error,
                    EVENT_JOB_ERROR | EVENT_JOB_MISSED
                )

                self._scheduler.start()
                logger.info(
                    f"LightDreaming scheduler started, service_id={self._service_id}, "
                    f"interval={self._config.light_interval_seconds}s"
                )

            self._running = True
            self._metrics.is_running = True
            logger.info(f"DreamService started, service_id={self._service_id}")

    def stop(self, ctx: RequestContext) -> None:
        """Stop the dreaming service.

        Args:
            ctx: RequestContext for final recall tracker flush
        """
        with self._lock:
            if not self._running:
                return

            # Stop scheduler
            if self._scheduler:
                self._scheduler.shutdown(wait=True)
                self._scheduler = None
                logger.info(f"Scheduler stopped, service_id={self._service_id}")

            # Flush recall tracker
            try:
                self._recall_tracker.stop(ctx)
                logger.info(f"RecallTracker flushed and stopped, service_id={self._service_id}")
            except Exception as exc:
                logger.error(
                    f"Error flushing RecallTracker: {exc}",
                    exc_info=True
                )

            self._running = False
            self._metrics.is_running = False
            logger.info(f"DreamService stopped, service_id={self._service_id}")

    def trigger_light(self, ctx: RequestContext) -> DreamReport:
        """Manually trigger LightDreaming (useful for testing).

        Args:
            ctx: RequestContext for this run

        Returns:
            DreamReport with statistics
        """
        logger.info(f"Manual LightDreaming triggered, service_id={self._service_id}")
        return self._run_light(ctx)

    def _run_light_wrapped(self) -> None:
        """Wrap LightDreaming run with logging and metrics tracking."""
        try:
            # Build RequestContext from config
            ctx = RequestContext(
                account_id=self._config.account_id,
                user_id=self._config.user_id,
                agent_id=self._config.agent_id,
                session_id=f"dream-{self._service_id}",
                trace_id=uuid.uuid4().hex,
            )

            report = self._run_light(ctx)

            # Update metrics
            self._metrics.last_light_run = datetime.now(timezone.utc)
            self._metrics.last_light_report = report
            self._metrics.total_light_runs += 1
            self._metrics.total_memories_extracted += report.candidates_extracted
            self._metrics.total_memories_written += report.candidates_written
            self._metrics.consecutive_errors = 0

            logger.info(
                f"LightDreaming completed: archives={report.archives_processed}, "
                f"extracted={report.candidates_extracted}, "
                f"deduplicated={report.candidates_deduplicated}, "
                f"written={report.candidates_written}, "
                f"errors={len(report.errors)}, "
                f"service_id={self._service_id}"
            )

        except Exception as exc:
            self._metrics.consecutive_errors += 1
            logger.error(
                f"LightDreaming failed: {exc}, service_id={self._service_id}",
                exc_info=True
            )
            raise

    def _run_light(self, ctx: RequestContext) -> DreamReport:
        """Run LightDreaming since last run.

        Args:
            ctx: RequestContext for this run

        Returns:
            DreamReport with statistics
        """
        # Determine time range
        since = self._last_light_run or (datetime.now(timezone.utc) - timedelta(hours=24))

        logger.info(
            f"Running LightDreaming since {since.isoformat()}, "
            f"account_id={ctx.account_id}, "
            f"service_id={self._service_id}"
        )

        report = self._light_dreaming.run(ctx, since)

        # Update last run time on success
        if not report.errors or report.candidates_written > 0:
            self._last_light_run = datetime.now(timezone.utc)

        return report

    def _on_job_error(self, event: JobEvent) -> None:
        """Handle job execution errors."""
        if event.exception:
            logger.error(
                f"Job {event.job_id} failed: {event.exception}",
                exc_info=event.exception
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"Job {event.job_id} missed, service_id={self._service_id}")
