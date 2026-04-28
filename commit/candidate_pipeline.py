"""Candidate extraction pipeline.

Runs all CandidateExtractor implementations in parallel and aggregates results.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from core.models import RequestContext, CandidateMemory
from core.interfaces import CandidateExtractor
from core.logging_config import get_logger
from core.validation import validate_candidate, validate_attribution, ValidationError

logger = get_logger(__name__)


class CandidatePipeline:
    """Pipeline for parallel candidate extraction.

    Runs all extractors concurrently and aggregates results.
    """

    def __init__(self, extractors: list[CandidateExtractor] | None = None):
        """Initialize CandidatePipeline.

        Args:
            extractors: List of CandidateExtractor instances.
                       If None, must be set before calling extract().
        """
        self._extractors = extractors or []

    def set_extractors(self, extractors: list[CandidateExtractor]) -> None:
        """Set the extractors to use.

        Args:
            extractors: List of CandidateExtractor instances
        """
        self._extractors = extractors

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
        parallel: bool = True,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> list[CandidateMemory]:
        """Extract candidates from messages using all extractors.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            parallel: If True, run extractors in parallel (default: True)
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text

        Returns:
            List of CandidateMemory from all extractors
        """
        if parallel:
            return self._extract_parallel(messages, ctx, session_time, session_summary, tool_stats_text)
        else:
            return self._extract_serial(messages, ctx, session_time, session_summary, tool_stats_text)

    def _extract_parallel(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> list[CandidateMemory]:
        """Run extractors in parallel using ThreadPoolExecutor.

        Args:
            messages: List of message dicts
            ctx: RequestContext
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text

        Returns:
            List of CandidateMemory from all extractors
        """
        all_candidates = []

        with ThreadPoolExecutor(max_workers=len(self._extractors)) as executor:
            # Submit all extraction jobs
            futures = {
                executor.submit(
                    extractor.extract, messages, ctx, session_time,
                    session_summary, tool_stats_text,
                ): extractor
                for extractor in self._extractors
            }

            # Collect results as they complete
            for future in as_completed(futures):
                extractor = futures[future]
                try:
                    candidates = future.result()
                    all_candidates.extend(candidates)
                except Exception as e:
                    # Log error but continue with other extractors
                    logger.error(
                        "Extractor %s failed: %s",
                        extractor.__class__.__name__, e,
                        exc_info=True
                    )

        # Validate all candidates
        validated_candidates = []
        for candidate in all_candidates:
            try:
                candidate = validate_attribution(candidate, user_id=ctx.user_id if ctx else None)
                validated_candidates.append(validate_candidate(candidate))
            except ValidationError as e:
                logger.warning("Rejected invalid candidate: %s", e)

        return validated_candidates

    def _extract_serial(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time=None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> list[CandidateMemory]:
        """Run extractors serially.

        Args:
            messages: List of message dicts
            ctx: RequestContext
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text

        Returns:
            List of CandidateMemory from all extractors
        """
        all_candidates = []

        for extractor in self._extractors:
            try:
                candidates = extractor.extract(
                    messages, ctx, session_time,
                    session_summary, tool_stats_text,
                )
                all_candidates.extend(candidates)
            except Exception as e:
                # Log error but continue
                logger.error(
                    "Extractor %s failed: %s",
                    extractor.__class__.__name__, e,
                    exc_info=True
                )

        # Validate all candidates
        validated_candidates = []
        for candidate in all_candidates:
            try:
                candidate = validate_attribution(candidate, user_id=ctx.user_id if ctx else None)
                validated_candidates.append(validate_candidate(candidate))
            except ValidationError as e:
                logger.warning("Rejected invalid candidate: %s", e)

        return validated_candidates

    def filter_by_confidence(
        self,
        candidates: list[CandidateMemory],
        threshold: float = 0.5
    ) -> list[CandidateMemory]:
        """Filter candidates by confidence threshold.

        Args:
            candidates: List of CandidateMemory
            threshold: Minimum confidence score (default: 0.5)

        Returns:
            Filtered list of CandidateMemory
        """
        return [c for c in candidates if c.confidence >= threshold]

    def deduplicate(
        self,
        candidates: list[CandidateMemory]
    ) -> list[CandidateMemory]:
        """Deduplicate candidates by (category, routing_key), keeping highest confidence.

        EXPERIMENT: B GROUP - DEDUP DISABLED
        """
        return candidates
