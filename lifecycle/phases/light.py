"""Light Dreaming phase for ContextEngine.

Periodically extracts missed memories from session archives using
the Extractor pipeline, with duplicate detection via vector search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.interfaces import LLM, VectorIndex, Embedder
from core.models import RequestContext, CandidateMemory
from extraction.tools import Extractor

logger = logging.getLogger(__name__)


@dataclass
class DreamReport:
    """Report from a dreaming run.

    Attributes:
        archives_processed: Number of archives processed
        candidates_extracted: Total candidates extracted
        candidates_deduplicated: Candidates removed as duplicates
        candidates_written: New candidates written to storage
        errors: List of error messages
        started_at: When the dream run started
        completed_at: When the dream run completed
    """
    archives_processed: int = 0
    candidates_extracted: int = 0
    candidates_deduplicated: int = 0
    candidates_written: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class LightDreaming:
    """Light Dreaming phase for memory consolidation.

    Processes session archives that were created since a given timestamp,
    extracts candidate memories using the Extractor, and writes new memories
    to storage with duplicate detection via vector search.

    This is a SYNCHRONOUS operation — no background threads or async processing.
    """

    # Cosine similarity threshold for duplicate detection
    _DUPLICATE_THRESHOLD = 0.9

    def __init__(
        self,
        llm: LLM,
        write_api: Any,
        archive_store: Any,
        embedder: Embedder,
        vector_index: VectorIndex,
    ):
        """Initialize LightDreaming.

        Args:
            llm: LLM instance for extraction
            write_api: MemoryWriteAPI instance for writing memories
            archive_store: SQLSessionArchiveStore instance for reading archives
            embedder: Embedder instance for duplicate detection
            vector_index: VectorIndex instance for duplicate detection
        """
        self._llm = llm
        self._write_api = write_api
        self._archive_store = archive_store
        self._embedder = embedder
        self._vector_index = vector_index

        # Create Extractor with minimal dependencies (only LLM required)
        # PromptManager will be auto-created from default templates dir
        self._extractor = Extractor(llm)

    def run(self, ctx: RequestContext, since: datetime) -> DreamReport:
        """Run light dreaming on archives created since the given timestamp.

        Args:
            ctx: RequestContext for this dreaming run
            since: Only process archives created after this timestamp

        Returns:
            DreamReport with statistics and any errors
        """
        report = DreamReport(started_at=datetime.now(timezone.utc))

        try:
            # Step 1: Fetch archives since the given timestamp
            logger.info(
                "LightDreaming: fetching archives since %s for account %s",
                since.isoformat(),
                ctx.account_id,
            )
            archives = self._archive_store.list_archives_since(since, ctx, limit=50)
            report.archives_processed = len(archives)

            if not archives:
                logger.info("LightDreaming: no archives found since %s", since.isoformat())
                report.completed_at = datetime.now(timezone.utc)
                return report

            logger.info("LightDreaming: processing %d archives", len(archives))

            # Step 2: Extract candidates from each archive
            all_candidates: list[CandidateMemory] = []
            for archive in archives:
                try:
                    if not archive.messages:
                        logger.debug(
                            "LightDreaming: skipping archive %s (no messages)",
                            archive.archive_id,
                        )
                        continue

                    logger.debug(
                        "LightDreaming: extracting from archive %s (%d messages)",
                        archive.archive_id,
                        len(archive.messages),
                    )

                    candidates = self._extractor.extract(
                        archive.messages,
                        ctx,
                        session_time=self._parse_archive_time(archive),
                        session_summary=archive.abstract or "",
                    )

                    logger.info(
                        "LightDreaming: extracted %d candidates from archive %s",
                        len(candidates),
                        archive.archive_id,
                    )
                    all_candidates.extend(candidates)

                except Exception as exc:
                    error_msg = (
                        f"Failed to extract from archive {archive.archive_id}: {exc}"
                    )
                    logger.error(error_msg, exc_info=True)
                    report.errors.append(error_msg)

            report.candidates_extracted = len(all_candidates)

            if not all_candidates:
                logger.info("LightDreaming: no candidates extracted")
                report.completed_at = datetime.now(timezone.utc)
                return report

            # Step 3: Filter duplicates using vector search
            new_candidates = self._filter_duplicates(all_candidates, ctx)
            report.candidates_deduplicated = len(all_candidates) - len(new_candidates)

            logger.info(
                "LightDreaming: duplicate filtering removed %d candidates, %d remain",
                report.candidates_deduplicated,
                len(new_candidates),
            )

            if not new_candidates:
                logger.info("LightDreaming: all candidates were duplicates")
                report.completed_at = datetime.now(timezone.utc)
                return report

            # Step 4: Write new candidates to storage
            try:
                result = self._write_api.write_memories(new_candidates, ctx)
                report.candidates_written = len(result) if isinstance(result, list) else 0

                logger.info(
                    "LightDreaming: wrote %d memories to storage",
                    report.candidates_written,
                )

            except Exception as exc:
                error_msg = f"Failed to write memories: {exc}"
                logger.error(error_msg, exc_info=True)
                report.errors.append(error_msg)

        except Exception as exc:
            error_msg = f"LightDreaming run failed: {exc}"
            logger.error(error_msg, exc_info=True)
            report.errors.append(error_msg)

        report.completed_at = datetime.now(timezone.utc)
        return report

    def _filter_duplicates(
        self,
        candidates: list[CandidateMemory],
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Filter out duplicate candidates using vector search.

        A candidate is considered a duplicate if vector search finds a similar
        memory with cosine similarity >= 0.9.

        Args:
            candidates: List of candidates to filter
            ctx: RequestContext for filtering

        Returns:
            List of non-duplicate candidates
        """
        if not candidates:
            return []

        non_duplicates: list[CandidateMemory] = []

        for candidate in candidates:
            # Embed the candidate's abstract
            try:
                embedding = self._embedder.embed(candidate.abstract)
            except Exception as exc:
                logger.warning(
                    "Failed to embed candidate (category=%s, routing_key=%s): %s",
                    candidate.category,
                    candidate.routing_key,
                    exc,
                )
                # Include candidate if embedding fails (better to over-write than miss)
                non_duplicates.append(candidate)
                continue

            # Search for similar memories
            try:
                from core.models import SeedHit
                hits: list[SeedHit] = self._vector_index.search_by_vector(
                    query_vector=embedding,
                    filters={
                        "account_id": ctx.account_id,
                        "category": candidate.category,
                    },
                    top_k=1,
                )

                if hits and hits[0].score >= self._DUPLICATE_THRESHOLD:
                    logger.debug(
                        "Duplicate found for candidate (category=%s, routing_key=%s, "
                        "similarity=%.2f, existing_uri=%s)",
                        candidate.category,
                        candidate.routing_key,
                        hits[0].score,
                        hits[0].uri,
                    )
                    # Skip this candidate as it's a duplicate
                    continue

            except Exception as exc:
                logger.warning(
                    "Vector search failed for candidate (category=%s, routing_key=%s): %s",
                    candidate.category,
                    candidate.routing_key,
                    exc,
                )
                # Include candidate if search fails (better to over-write than miss)

            non_duplicates.append(candidate)

        return non_duplicates

    def _parse_archive_time(self, archive: Any) -> datetime | None:
        """Parse creation time from archive metadata.

        Args:
            archive: ArchiveEntry with metadata

        Returns:
            datetime or None if parsing fails
        """
        try:
            if archive.created_at:
                # Try ISO format parsing
                if isinstance(archive.created_at, str):
                    return datetime.fromisoformat(archive.created_at.replace("Z", "+00:00"))
                elif isinstance(archive.created_at, datetime):
                    return archive.created_at
        except Exception as exc:
            logger.debug("Failed to parse archive time: %s", exc)

        return None
