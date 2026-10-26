"""Lazy TTL cutover for medium tool results while preserving a stub buffer."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.models import RequestContext
from features.tool_result_externalization.externalizer import ToolResultExternalizer
from features.tool_result_externalization.models import ArtifactMetadata
from features.tool_result_externalization.multires import ranking_context_from_messages
from features.tool_result_externalization.normalizer import (
    ToolResultNormalizer,
    canonical_tool_call_id,
)
from features.tool_result_externalization.store_base import ToolResultArtifactStore


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CacheDelayState:
    schema_version: int = 2
    last_model_request_at: str = ""
    cold_cutover_before: str = ""
    hot_seen_artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CacheDelayState":
        value = value if isinstance(value, Mapping) else {}
        hot_seen: dict[str, str] = {}
        raw_seen = value.get("hot_seen_artifacts")
        if isinstance(raw_seen, Mapping):
            for artifact_id, seen_at in raw_seen.items():
                artifact_id = str(artifact_id)
                seen_at = str(seen_at or "")
                if (
                    len(hot_seen) >= ToolResultCacheDelay.MAX_TRACKED_ARTIFACTS
                    or not artifact_id.startswith("tr_")
                    or _parse_time(seen_at) is None
                ):
                    continue
                hot_seen[artifact_id] = seen_at
        return cls(
            schema_version=2,
            last_model_request_at=str(value.get("last_model_request_at") or ""),
            cold_cutover_before=str(value.get("cold_cutover_before") or ""),
            hot_seen_artifacts=hot_seen,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_model_request_at": self.last_model_request_at,
            "cold_cutover_before": self.cold_cutover_before,
            "hot_seen_artifacts": dict(self.hot_seen_artifacts),
        }


@dataclass(frozen=True)
class CacheDelayDecision:
    state: CacheDelayState
    active_cutoff: str = ""
    transitioned: bool = False
    observed_idle_seconds: float = 0.0
    cache_hit_price_ratio: float = 0.1


@dataclass(frozen=True)
class CacheDelayRewrite:
    messages: list[dict]
    replaced_count: int = 0


@dataclass(frozen=True)
class CacheDelayObservation:
    state: CacheDelayState
    observed_count: int = 0
    overflow_count: int = 0
    pruned_count: int = 0


class ToolResultCacheDelay:
    """Maintain a hot caller view and a monotonic cold Artifact/stub view."""

    STATE_KEY = "tool_result_cache_delay"
    MAX_TRACKED_ARTIFACTS = 2_048
    _ARTIFACT_ID = re.compile(r"^tr_[a-f0-9_]+$")
    _ARTIFACT_REF = re.compile(r"^ogmemory://tool-results/(tr_[a-f0-9_]+)$")

    def __init__(
        self,
        *,
        persistence_externalizer: ToolResultExternalizer,
        replay_externalizer: ToolResultExternalizer,
        store: ToolResultArtifactStore,
        delayed_result_chars: int = 2_000,
        immediate_result_chars: int = 20_000,
        idle_ttl_seconds: int = 300,
        cache_hit_price_ratio: float = 0.1,
        clock: Callable[[], datetime] | None = None,
        normalizer: ToolResultNormalizer | None = None,
    ) -> None:
        if delayed_result_chars < 2:
            raise ValueError("delayed_result_chars must be at least 2")
        if immediate_result_chars < 1:
            raise ValueError("immediate_result_chars must be positive")
        if delayed_result_chars >= immediate_result_chars:
            raise ValueError("delayed_result_chars must be below immediate_result_chars")
        if idle_ttl_seconds < 1:
            raise ValueError("idle_ttl_seconds must be positive")
        if not 0 < cache_hit_price_ratio <= 1:
            raise ValueError("cache_hit_price_ratio must be in (0, 1]")
        self.persistence_externalizer = persistence_externalizer
        self.replay_externalizer = replay_externalizer
        self.store = store
        self.delayed_result_chars = int(delayed_result_chars)
        self.immediate_result_chars = int(immediate_result_chars)
        self.idle_ttl_seconds = int(idle_ttl_seconds)
        self.cache_hit_price_ratio = float(cache_hit_price_ratio)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.normalizer = normalizer or ToolResultNormalizer()

    def govern_persistent_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        complete_turn: bool = False,
        ranking_prompt: str = "",
    ):
        """Persist the compact view; failures inherit main fail-closed behavior."""
        return self.persistence_externalizer.govern_messages(
            messages,
            ctx=ctx,
            session_id=session_id,
            persistence=True,
            ranking_prompt=ranking_prompt,
        )

    def govern_replay_messages(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        complete_turn: bool = False,
        ranking_prompt: str = "",
    ):
        """Keep medium results raw until TTL; apply only the single-result limit."""
        return self.replay_externalizer.govern_messages(
            messages,
            ctx=ctx,
            session_id=session_id,
            persistence=False,
            ranking_prompt=ranking_prompt,
        )

    def advance(self, value: Mapping[str, Any] | None) -> CacheDelayDecision:
        """Advance lazy TTL state using the request-start clock."""
        previous = CacheDelayState.from_mapping(value)
        now = _utc(self.clock())
        previous_request = _parse_time(previous.last_model_request_at)
        previous_cutoff = _parse_time(previous.cold_cutover_before)
        observed_idle = 0.0
        transitioned = False
        next_cutoff = previous_cutoff

        if previous_request is not None and now >= previous_request:
            observed_idle = (now - previous_request).total_seconds()
            if observed_idle >= self.idle_ttl_seconds:
                if next_cutoff is None or previous_request > next_cutoff:
                    next_cutoff = previous_request
                    transitioned = True

        # A backwards wall clock must not move either monotonic boundary back.
        next_request = max(
            (item for item in (previous_request, now) if item is not None),
            default=now,
        )
        state = CacheDelayState(
            schema_version=2,
            last_model_request_at=next_request.isoformat(),
            cold_cutover_before=next_cutoff.isoformat() if next_cutoff else "",
            hot_seen_artifacts=dict(previous.hot_seen_artifacts),
        )
        return CacheDelayDecision(
            state=state,
            active_cutoff=state.cold_cutover_before,
            transitioned=transitioned,
            observed_idle_seconds=observed_idle,
            cache_hit_price_ratio=self.cache_hit_price_ratio,
        )

    def anchor_turn_completion(
        self,
        value: Mapping[str, Any] | None,
        completed_at: str,
    ) -> CacheDelayState:
        """Move the active TTL epoch to the completed user Turn boundary."""
        current = CacheDelayState.from_mapping(value)
        completed = _parse_time(completed_at)
        if completed is None:
            return current
        previous_request = _parse_time(current.last_model_request_at)
        anchored = max(
            (item for item in (previous_request, completed) if item is not None),
            default=completed,
        )
        return CacheDelayState(
            schema_version=2,
            last_model_request_at=anchored.isoformat(),
            cold_cutover_before=current.cold_cutover_before,
            hot_seen_artifacts=dict(current.hot_seen_artifacts),
        )

    @staticmethod
    def _stable_artifact_ids(
        normalized_message,
        part,
        ctx: RequestContext,
        session_id: str,
    ) -> tuple[str, ...]:
        strong = normalized_message.identity_is_stable and (
            normalized_message.source_message_id
            or (normalized_message.turn_id and part.tool_call_id)
        )
        if not strong:
            return ()
        fields = (
            ctx.account_id,
            ctx.user_id,
            ctx.agent_id,
            session_id,
            normalized_message.turn_id or "",
            normalized_message.source_message_id or "",
            canonical_tool_call_id(part.tool_call_id),
            part.part_path,
            str(part.occurrence_index),
        )
        base = ToolResultExternalizer.strong_artifact_id(*fields)
        digest = hashlib.sha256(part.output_text.encode("utf-8")).hexdigest()
        return base, f"{base}_{digest[:12]}"

    @staticmethod
    def _artifact_matches(
        metadata: ArtifactMetadata,
        output: str,
        normalized_message,
        part,
    ) -> bool:
        digest = hashlib.sha256(part.output_text.encode("utf-8")).hexdigest()
        return bool(
            metadata.identity_strength == "strong"
            and metadata.source_message_id == normalized_message.source_message_id
            and metadata.turn_id == normalized_message.turn_id
            and canonical_tool_call_id(metadata.tool_call_id)
            == canonical_tool_call_id(part.tool_call_id)
            and metadata.part_path == part.part_path
            and metadata.occurrence_index == part.occurrence_index
            and metadata.original_chars == part.original_chars
            and metadata.sha256 == digest
            and output == part.output_text
        )

    def rewrite_cold_view(
        self,
        messages: list,
        *,
        ctx: RequestContext,
        session_id: str,
        cutoff: str,
        hot_seen_artifacts: Mapping[str, str],
        ranking_prompt: str = "",
    ) -> CacheDelayRewrite:
        """Replace only verified medium results seen before the sticky cutoff."""
        cutoff_time = _parse_time(cutoff)
        copied = [deepcopy(message) for message in messages]
        if cutoff_time is None:
            return CacheDelayRewrite(copied, 0)

        normalized = self.normalizer.normalize(copied)
        deadline = (
            time.monotonic()
            + self.replay_externalizer.write_timeout_ms / 1000
        )
        rewritten = [message.original for message in normalized]
        eligible: list[tuple[Any, Any, ArtifactMetadata]] = []
        for message in normalized:
            for part in message.tool_results:
                if part.source_ref or part.source_collection:
                    continue
                if not (
                    self.delayed_result_chars <= part.original_chars
                    < self.immediate_result_chars
                ):
                    continue
                for artifact_id in self._stable_artifact_ids(
                    message, part, ctx, session_id,
                ):
                    hot_seen_at = _parse_time(
                        str(hot_seen_artifacts.get(artifact_id) or "")
                    )
                    if hot_seen_at is None or hot_seen_at > cutoff_time:
                        continue
                    try:
                        metadata, output = self.replay_externalizer.io_runner.run(
                            lambda: self.store.read_all(
                                artifact_id, session_id, ctx,
                            ),
                            deadline=deadline,
                            operation_key=session_id,
                        )
                    except Exception:
                        continue
                    if not self._artifact_matches(
                        metadata, output, message, part,
                    ):
                        continue
                    eligible.append((message, part, metadata))
                    break
        stubs = self.replay_externalizer.build_stub_sequence(
            (metadata for _message, _part, metadata in eligible),
            ranking_context=ranking_context_from_messages(
                normalized,
                prompt=ranking_prompt,
            ),
        )
        replaced = 0
        for (message, part, _metadata), stub in zip(eligible, stubs):
            # The delayed path is a cost optimization, so never replace a hot
            # raw result with a larger cold projection at the 1k boundary.
            if len(stub) >= part.original_chars:
                continue
            self.normalizer.set_payload(
                rewritten[message.message_index],
                part.payload_path,
                stub,
            )
            replaced += 1
        return CacheDelayRewrite(rewritten, replaced)

    def observe_hot_view(
        self,
        messages: list,
        *,
        state: Mapping[str, Any] | None,
        ctx: RequestContext,
        session_id: str,
        observed_at: str,
        ledger_hot_seen_artifacts: Mapping[str, str] | None = None,
    ) -> CacheDelayObservation:
        """Record verified raw medium Artifacts that reached a compose projection."""
        current = CacheDelayState.from_mapping(state)
        observed_time = _parse_time(observed_at)
        if observed_time is None:
            return CacheDelayObservation(current)
        ledger_seen = {
            str(artifact_id): str(seen_at)
            for artifact_id, seen_at in (ledger_hot_seen_artifacts or {}).items()
            if self._ARTIFACT_ID.fullmatch(str(artifact_id)) is not None
            and _parse_time(str(seen_at)) is not None
        }
        seen = {
            artifact_id: seen_at
            for artifact_id, seen_at in current.hot_seen_artifacts.items()
            if artifact_id not in ledger_seen
        }
        observed_count = 0
        overflow_count = 0
        pruned_count = len(current.hot_seen_artifacts) - len(seen)
        normalized = self.normalizer.normalize(messages)
        deadline = (
            time.monotonic()
            + self.replay_externalizer.write_timeout_ms / 1000
        )
        active_artifact_ids: set[str] = set()
        verified_hot_ids: list[str] = []
        for message in normalized:
            for part in message.tool_results:
                if part.source_ref:
                    match = self._ARTIFACT_REF.fullmatch(part.source_ref)
                    if match is not None:
                        active_artifact_ids.add(match.group(1))
                    continue
                if part.source_collection:
                    for source_ref in part.source_refs:
                        match = self._ARTIFACT_REF.fullmatch(source_ref)
                        if match is not None:
                            active_artifact_ids.add(match.group(1))
                    continue
                if not (
                    self.delayed_result_chars <= part.original_chars
                    < self.immediate_result_chars
                ):
                    continue
                candidate_ids = self._stable_artifact_ids(
                    message, part, ctx, session_id,
                )
                active_artifact_ids.update(candidate_ids)
                for artifact_id in candidate_ids:
                    try:
                        metadata, output = self.replay_externalizer.io_runner.run(
                            lambda: self.store.read_all(
                                artifact_id, session_id, ctx,
                            ),
                            deadline=deadline,
                            operation_key=session_id,
                        )
                    except Exception:
                        continue
                    if not self._artifact_matches(metadata, output, message, part):
                        continue
                    active_artifact_ids.add(artifact_id)
                    if artifact_id not in ledger_seen:
                        verified_hot_ids.append(artifact_id)
                    break

        cutoff_time = _parse_time(current.cold_cutover_before)
        if cutoff_time is not None:
            for artifact_id, seen_at in list(seen.items()):
                first_seen = _parse_time(seen_at)
                if (
                    artifact_id not in active_artifact_ids
                    and first_seen is not None
                    and first_seen <= cutoff_time
                ):
                    seen.pop(artifact_id, None)
                    pruned_count += 1

        for artifact_id in dict.fromkeys(verified_hot_ids):
            if artifact_id in seen:
                continue
            if len(seen) >= self.MAX_TRACKED_ARTIFACTS:
                overflow_count += 1
            else:
                seen[artifact_id] = observed_time.isoformat()
                observed_count += 1
        return CacheDelayObservation(
            CacheDelayState(
                schema_version=2,
                last_model_request_at=current.last_model_request_at,
                cold_cutover_before=current.cold_cutover_before,
                hot_seen_artifacts=seen,
            ),
            observed_count=observed_count,
            overflow_count=overflow_count,
            pruned_count=pruned_count,
        )
