"""Context writer for persisting ContextNodes.

Orchestrates the write process: extraction → planning → building → writing → outbox.
"""

import time
from typing import Optional

from core.models import RequestContext, CandidateMemory, ContextNode, WritePlan
from core.interfaces import ContextFS, MergePolicy, LLM
from core.logging_config import get_logger
from core.errors import ConcurrentModificationError
from commit.policy_router import PolicyRouter
from commit.archive_builder import ArchiveBuilder
from commit.outbox_store import OutboxStore

logger = get_logger(__name__)


class ContextWriter:
    """Writes context memories to storage.

    Orchestrates the full write pipeline:
    1. Plan: Determine write action via MergePolicy
    2. Build: Construct ContextNode via ArchiveBuilder
    3. Write: Persist via ContextFS
    4. Outbox: Register event for async index synchronization
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        policy_router: Optional[PolicyRouter] = None,
        archive_builder: Optional[ArchiveBuilder] = None,
        outbox_store: Optional[OutboxStore] = None,
    ):
        """Initialize ContextWriter.

        Args:
            fs: ContextFS for persisting nodes
            llm: LLM instance for semantic merge (required, passed to ArchiveBuilder)
            policy_router: PolicyRouter for selecting merge policies
            archive_builder: ArchiveBuilder for building ContextNode
            outbox_store: OutboxStore for registering index events (optional)
        """
        self._fs = fs
        self._llm = llm
        self._policy_router = policy_router or PolicyRouter(fs)
        self._archive_builder = archive_builder or ArchiveBuilder(llm=llm)
        self._outbox_store = outbox_store

    def write_candidate(
        self,
        candidate: CandidateMemory,
        ctx: RequestContext
    ) -> WritePlan:
        """Write a single candidate memory.

        Args:
            candidate: CandidateMemory to write
            ctx: RequestContext for this operation

        Returns:
            WritePlan showing what action was taken
        """
        # Retry loop for concurrent modification (optimistic locking)
        MAX_RETRIES = 3
        RETRY_DELAYS = [0.1, 0.2, 0.4]  # Exponential backoff in seconds

        for attempt in range(MAX_RETRIES):
            try:
                # Step 1: Plan - determine write action
                plan = self._policy_router.plan(candidate, ctx)

                # Skip if action is skip
                if plan.action == "skip":
                    return plan

                # Step 2: Build - construct ContextNode
                node = self._archive_builder.build(candidate, plan, ctx)

                # Step 3: Write - persist to storage
                self._fs.write_node(node, ctx)

                # Step 4: Outbox - register event for async index synchronization
                if self._outbox_store and plan.action in ("create", "merge"):
                    try:
                        self._outbox_store.register_write(node, ctx)
                    except Exception as e:
                        # Log error but don't fail the write
                        logger.warning(
                            "Failed to register outbox event for %s: %s",
                            node.uri, e
                        )

                return plan

            except ConcurrentModificationError as e:
                if attempt < MAX_RETRIES - 1:
                    # Retry with exponential backoff
                    delay = RETRY_DELAYS[attempt]
                    logger.info(
                        "Concurrent modification detected for %s (attempt %d), "
                        "retrying after %.1fs: expected v%d, found v%d",
                        candidate.routing_key, attempt + 1, delay, e.expected_version, e.actual_version
                    )
                    time.sleep(delay)
                    # Re-plan on retry (will read fresh version)
                    continue
                else:
                    # Max retries exceeded
                    logger.error(
                        "Max retries exceeded for concurrent modification at %s",
                        candidate.routing_key
                    )
                    raise

    def write_candidates(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext
    ) -> list[WritePlan]:
        """Write multiple candidate memories.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations

        Returns:
            List of WritePlan showing what action was taken for each
        """
        plans = []

        for candidate in candidates:
            try:
                plan = self.write_candidate(candidate, ctx)
                plans.append(plan)
            except Exception as e:
                # Log error but continue with other candidates
                logger.error(
                    "Failed to write candidate %s/%s: %s",
                    candidate.category, candidate.routing_key, e,
                    exc_info=True
                )

        return plans

    def write_candidates_parallel(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext,
        max_workers: int = 4
    ) -> list[WritePlan]:
        """Write multiple candidates in parallel.

        Args:
            candidates: List of CandidateMemory to write
            ctx: RequestContext for these operations
            max_workers: Maximum parallel write operations

        Returns:
            List of WritePlan showing what action was taken for each
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        plans = []
        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Submit all tasks
            futures = {
                executor.submit(self.write_candidate, candidate, ctx): candidate
                for candidate in candidates
            }

            # Collect results
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    plan = future.result()
                    plans.append(plan)
                except Exception as e:
                    logger.error(
                        "Failed to write candidate %s/%s: %s",
                        candidate.category, candidate.routing_key, e,
                        exc_info=True
                    )

        finally:
            # Ensure proper shutdown even if interrupted
            # cancel_futures=True cancels any pending futures
            executor.shutdown(wait=True, cancel_futures=True)

        return plans
