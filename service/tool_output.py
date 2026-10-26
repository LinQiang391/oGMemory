"""Read/Bash output filtering with direct, non-indexed ContextFS storage."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from threading import RLock
from typing import Any, cast

from core.enums import ContextType
from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import ContextNode, RequestContext
from core.resource_identity import ResourceIdentityError, normalize_posix_relative_path
from service.tool_output_filtering import (
    cap_reduction_output,
    legacy_bash_reduction,
    legacy_read_reduction,
    passthrough_reduction,
    reduce_bash_output,
    reduce_read_output,
    shorten_head_tail,
)

_TOOL_OUTPUT_URI_RE = re.compile(
    r"^ctx://(?P<account>[^/\s]+)/agents/(?P<agent>[^/\s]+)/"
    r"memories/tool_outputs/(?P<slug>[^/?#\s]+)$"
)
_SUPPORTED_BASH_CATEGORIES = {"test_runner", "diff", "python_error"}
_DEFAULT_MAX_ORIGINAL_BYTES = 1024 * 1024
_DEFAULT_MAX_RETAINED_NODES = 64
_UTF8_SIZE_CHUNK_CHARS = 64 * 1024
_RETENTION_SUMMARY_PAGE_SIZE = 128
_FILTER_DIAGNOSTIC_REASONS = frozenset(
    {
        "feature_disabled",
        "missing_path_or_workspace",
        "missing_content_or_locator",
        "unsafe_locator",
        "outside_workspace",
        "not_a_file",
        "explicit_read_range",
        "unsupported_content",
        "below_min_chars",
        "below_min_size",
        "not_shorter",
        "output_cap_too_small",
        "missing_command_or_output",
        "not_whitelisted",
        "original_too_large",
        "unchanged",
    }
)
_UNSUPPORTED_FILTER_REASONS = frozenset({"unsupported_content", "not_whitelisted"})
_SKIPPED_FILTER_REASONS = frozenset(
    {
        "missing_path_or_workspace",
        "missing_content_or_locator",
        "unsafe_locator",
        "outside_workspace",
        "not_a_file",
        "explicit_read_range",
        "below_min_chars",
        "below_min_size",
        "not_shorter",
        "original_too_large",
    }
)
_SAFE_DIAGNOSTIC_ENUM_RE = re.compile(r"^[a-z0-9_]{1,80}$")
# ContextFS implementations in older deployments may not expose a shared
# mutation lock.  Keep a process-wide re-entrant fallback for those doubles and
# local stores; production AGFS/SQL adapters provide their own cross-process
# implementation below.
_PROCESS_TOOL_OUTPUT_MUTATION_LOCK = RLock()


def _hash_text_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if index:
            digest.update(b"\x00")
        digest.update(part.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _bounded_utf8_size(text: str, *, max_bytes: int) -> int | None:
    """Return the UTF-8 byte size, stopping without a full-size encoded copy."""

    total = 0
    for start in range(0, len(text), _UTF8_SIZE_CHUNK_CHARS):
        chunk = text[start : start + _UTF8_SIZE_CHUNK_CHARS]
        total += len(chunk.encode("utf-8", errors="replace"))
        if total > max_bytes:
            return None
    return total


def parse_tool_output_uri(uri: str) -> tuple[str, str, str] | None:
    """Return ``(account_id, agent_id, slug)`` for an exact tool-output URI."""

    match = _TOOL_OUTPUT_URI_RE.fullmatch(uri or "")
    if match is None:
        return None
    return match.group("account"), match.group("agent"), match.group("slug")


def _shorten_head_tail(text: str, *, max_lines: int, label: str) -> str:
    """Compatibility alias for the historical reducer used by callers/tests."""

    return cast(str, shorten_head_tail(text, max_lines=max_lines, label=label))


class ToolOutputRepository:
    """Persist tool-output originals directly in ContextFS without Outbox writes."""

    def __init__(
        self,
        context_fs,
        *,
        max_original_bytes: int = _DEFAULT_MAX_ORIGINAL_BYTES,
        max_retained_nodes: int = _DEFAULT_MAX_RETAINED_NODES,
    ) -> None:
        if (
            isinstance(max_original_bytes, bool)
            or not isinstance(max_original_bytes, int)
            or max_original_bytes < 1
        ):
            raise ValueError("max_original_bytes must be a positive integer")
        if (
            isinstance(max_retained_nodes, bool)
            or not isinstance(max_retained_nodes, int)
            or max_retained_nodes < 1
        ):
            raise ValueError("max_retained_nodes must be a positive integer")
        self._fs = context_fs
        # Callers may use lower limits for isolated deployments/tests, but no
        # caller can widen the service-owned storage safety bounds.
        self._max_original_bytes = min(
            max_original_bytes,
            _DEFAULT_MAX_ORIGINAL_BYTES,
        )
        self._max_retained_nodes = min(
            max_retained_nodes,
            _DEFAULT_MAX_RETAINED_NODES,
        )
        self._mutation_lock = RLock()

    def measure_original_bytes(self, original: str) -> int | None:
        """Return the accepted byte size, or ``None`` when the fixed cap is exceeded."""

        return _bounded_utf8_size(original, max_bytes=self._max_original_bytes)

    @staticmethod
    def uri_for(ctx: RequestContext, slug: str) -> str:
        if not slug or "/" in slug:
            raise ValueError("invalid tool-output slug")
        return f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tool_outputs/{slug}"

    def store_original(
        self,
        ctx: RequestContext,
        *,
        slug: str,
        content_digest: str,
        original: str,
        filtered: str,
        tool_name: str,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not ctx.agent_id:
            raise ValueError("agentId is required")
        if not original:
            raise ValueError("original output is required")
        if not re.fullmatch(r"[0-9a-f]{64}", content_digest):
            raise ValueError("content digest must be a full sha256 hex digest")

        original_bytes = self.measure_original_bytes(original)
        if original_bytes is None:
            raise ValueError(
                f"original output exceeds {self._max_original_bytes}-byte limit"
            )

        uri = self.uri_for(ctx, slug)
        with self._mutation_lock, self._tool_output_mutation_scope(ctx) as ensure_owned:
            ensure_owned()
            if self._fs.exists(uri, ctx):
                ensure_owned()
                existing = self._fs.read_node(uri, ctx)
                ensure_owned()
                existing_digest = str(
                    (existing.metadata or {}).get("content_digest") or ""
                )
                if existing_digest != content_digest or existing.content != original:
                    raise ValueError("tool-output identity collision")
                refreshed_metadata = dict(existing.metadata or {})
                refreshed_metadata["stored_at"] = datetime.now(UTC).isoformat()
                refreshed = replace(existing, metadata=refreshed_metadata)
                ensure_owned()
                self._fs.write_node(refreshed, ctx)
                ensure_owned()
                self._prune_retained(
                    ctx,
                    limit=self._max_retained_nodes,
                    preserve_uri=uri,
                    ensure_owned=ensure_owned,
                )
                return uri

            safe_metadata = dict(metadata or {})
            safe_metadata.update(
                {
                    "content_digest": content_digest,
                    "tool_name": tool_name,
                    "strategy": strategy,
                    "indexing_disabled": True,
                    "original_chars": len(original),
                    "original_bytes": original_bytes,
                    "filtered_chars": len(filtered),
                    "stored_at": datetime.now(UTC).isoformat(),
                }
            )
            overview_fields = (
                "tool_name",
                "strategy",
                "bash_category",
                "exit_code",
                "original_chars",
                "original_bytes",
                "filtered_chars",
                "original_lines",
                "filtered_lines",
                "was_shortened",
            )
            overview = "\n".join(
                f"{key}: {safe_metadata[key]}"
                for key in overview_fields
                if key in safe_metadata and safe_metadata[key] is not None
            )
            node = ContextNode(
                uri=uri,
                context_type=ContextType.SKILL.value,
                category="tool_outputs",
                level=4,
                owner_space=ctx.agent_space_name(),
                abstract=(filtered or original)[:200],
                overview=overview,
                content=original,
                metadata=safe_metadata,
            )
            ensure_owned()
            self._fs.write_node(node, ctx)
            try:
                ensure_owned()
                self._prune_retained(
                    ctx,
                    limit=self._max_retained_nodes,
                    preserve_uri=uri,
                    ensure_owned=ensure_owned,
                )
            except Exception:
                try:
                    ensure_owned()
                    self._fs.delete_node(uri, ctx)
                    ensure_owned()
                except Exception:
                    pass
                raise
            return uri

    @contextmanager
    def _tool_output_mutation_scope(
        self,
        ctx: RequestContext,
    ) -> Iterator[Callable[[], None]]:
        """Serialize one agent's complete retention scope across instances.

        Tool-output retention deletes a set selected from a snapshot, so a
        lock scoped only to the newly written URI is insufficient: another
        writer could refresh one of the selected stale URIs between selection
        and deletion.  The shared lock therefore uses the agent's collection
        URI as its key and covers the write, snapshot, deletes, and rollback.
        """

        lock_uri = (
            f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tool_outputs"
        )
        lock_factory = getattr(self._fs, "tool_output_mutation_lock", None)
        if not callable(lock_factory):
            # AGFS already exposes the durable cross-process resource lock.  It
            # is safe for the collection URI and avoids duplicating its lease
            # and stale-owner recovery implementation here.
            lock_factory = getattr(self._fs, "resource_mutation_lock", None)
        if callable(lock_factory):
            with lock_factory(lock_uri, ctx) as lease:
                ensure_owned = getattr(lease, "ensure_owned", None)
                if not callable(ensure_owned):
                    def ensure_owned() -> None:
                        return None
                ensure_owned()
                yield ensure_owned
                ensure_owned()
            return

        with _PROCESS_TOOL_OUTPUT_MUTATION_LOCK:
            yield lambda: None

    def _prune_retained(
        self,
        ctx: RequestContext,
        *,
        limit: int,
        preserve_uri: str | None = None,
        ensure_owned: Callable[[], None] | None = None,
    ) -> None:
        check_ownership = ensure_owned or (lambda: None)
        check_ownership()
        _, total, stale_uris = self._retention_keep_set(
            ctx,
            limit=limit,
            preserve_uri=preserve_uri,
            ensure_owned=check_ownership,
        )
        check_ownership()
        if total <= limit:
            return

        deleted = 0
        # Delete only records selected as stale by the first (immutable)
        # snapshot.  A second live scan could observe a node written by a
        # different repository instance after the snapshot and incorrectly
        # evict that concurrent writer's freshly returned URI.
        for uri in sorted(stale_uris):
            check_ownership()
            try:
                self._fs.delete_node(uri, ctx)
            except NodeNotFoundError:
                pass
            check_ownership()
            deleted += 1
        expected = len(stale_uris)
        if deleted < expected:
            raise RuntimeError("tool-output retention could not remove all stale nodes")

    def _retention_keep_set(
        self,
        ctx: RequestContext,
        *,
        limit: int,
        preserve_uri: str | None = None,
        ensure_owned: Callable[[], None] | None = None,
    ) -> tuple[set[str], int, set[str]]:
        check_ownership = ensure_owned or (lambda: None)
        newest: list[tuple[str, str]] = []
        stale_uris: set[str] = set()
        total = 0
        preserve_record: tuple[str, str] | None = None
        for record in self._iter_retention_records(ctx, ensure_owned=check_ownership):
            check_ownership()
            total += 1
            if preserve_uri is not None and record[1] == preserve_uri:
                preserve_record = record
            newest.append(record)
            newest.sort()
            if len(newest) > limit:
                stale_uris.add(newest.pop(0)[1])

        if preserve_record is None and preserve_uri is not None:
            # The just-written node may be briefly absent from a paged backend
            # view. Reserve its slot so a second pass can never evict it.
            preserve_record = ("\uffff", preserve_uri)
            total += 1
        newest_uris = {uri for _, uri in newest}
        if preserve_record is not None and preserve_record[1] not in newest_uris:
            newest.append(preserve_record)
            newest.sort()
            if len(newest) > limit:
                removable = [
                    record for record in newest if record[1] != preserve_record[1]
                ]
                if not removable:
                    raise RuntimeError("tool-output retention preserve set is invalid")
                stale_uris.add(newest.pop(newest.index(min(removable)))[1])
            # A just-written node can be evicted while selecting the newest
            # records when its timestamp is unusual.  It is explicitly
            # reserved above, so remove it from the stale snapshot set.
            stale_uris.discard(preserve_record[1])
        keep_uris = {uri for _, uri in newest}
        stale_uris.difference_update(keep_uris)
        return keep_uris, total, stale_uris

    def _iter_retention_records(
        self,
        ctx: RequestContext,
        *,
        ensure_owned: Callable[[], None] | None = None,
    ) -> Iterator[tuple[str, str]]:
        check_ownership = ensure_owned or (lambda: None)
        parent_uri = (
            f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/tool_outputs"
        )
        page_method = getattr(self._fs, "list_node_summaries_page", None)
        if callable(page_method):
            after_uri: str | None = None
            while True:
                check_ownership()
                page = page_method(
                    parent_uri,
                    ctx,
                    limit=_RETENTION_SUMMARY_PAGE_SIZE,
                    after_uri=after_uri,
                )
                if not page:
                    break
                ordered = sorted(page, key=lambda node: str(node.uri))
                next_after = str(ordered[-1].uri)
                if after_uri is not None and next_after <= after_uri:
                    raise RuntimeError("tool-output summary pagination did not advance")
                for node in ordered:
                    check_ownership()
                    record = self._retention_record(ctx, node)
                    if record is not None:
                        yield record
                after_uri = next_after
                if len(page) < _RETENTION_SUMMARY_PAGE_SIZE:
                    break
            return

        for uri in self._fs.list_children(parent_uri, ctx):
            check_ownership()
            try:
                node = self._fs.read_node(uri, ctx)
            except NodeNotFoundError:
                continue
            record = self._retention_record(ctx, node)
            if record is not None:
                yield record

    @staticmethod
    def _retention_record(
        ctx: RequestContext,
        node: ContextNode,
    ) -> tuple[str, str] | None:
        parsed = parse_tool_output_uri(node.uri)
        if (
            parsed is None
            or parsed[:2] != (ctx.account_id, ctx.agent_id)
            or node.context_type != ContextType.SKILL.value
            or node.category != "tool_outputs"
            or node.owner_space != ctx.agent_space_name()
        ):
            return None
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        stored_at = str(
            metadata.get("stored_at")
            or metadata.get("created_at")
            or metadata.get("updated_at")
            or ""
        )
        return stored_at, node.uri

    def read_original(self, ctx: RequestContext, uri: str) -> str:
        parsed = parse_tool_output_uri(uri)
        if parsed is None:
            raise ValueError("memory_uri is not a tool_outputs node")
        account_id, agent_id, _ = parsed
        if account_id != ctx.account_id:
            raise AccessDeniedError(uri, ctx.account_id, "account mismatch")
        if not ctx.agent_id or agent_id != ctx.agent_id:
            raise AccessDeniedError(uri, ctx.account_id, "agent mismatch")

        node = self._fs.read_node(uri, ctx)
        if node.context_type != ContextType.SKILL.value or node.category != "tool_outputs":
            raise ValueError("memory_uri is not a tool_outputs node")
        if node.owner_space != ctx.agent_space_name():
            raise AccessDeniedError(uri, ctx.account_id, "owner space mismatch")
        if not isinstance(node.content, str):
            raise ValueError("tool-output original content is invalid")
        return node.content


class ToolOutputFilterService:
    """Apply deterministic filters and return references to stored originals."""

    def __init__(self, repository: ToolOutputRepository, config) -> None:
        self._repository = repository
        self._cfg = config

    def filter_read(self, params: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        if not self._filter_enabled("tool_output_filter_read_enabled"):
            return self._with_diagnostics(
                {"ok": False, "reason": "feature_disabled"}, operation="filter"
            )

        relative_path_raw = str(
            params.get("relative_path") or params.get("relativePath") or ""
        ).strip()
        original = params.get("content")
        if not relative_path_raw or not isinstance(original, str):
            return self._with_diagnostics(
                {"ok": False, "reason": "missing_content_or_locator"}, operation="filter"
            )
        try:
            relative_path = normalize_posix_relative_path(relative_path_raw)
        except ResourceIdentityError:
            return self._with_diagnostics(
                {"ok": False, "reason": "unsafe_locator"}, operation="filter"
            )
        if self._has_explicit_read_range(params):
            return self._with_diagnostics(
                {"ok": False, "reason": "explicit_read_range"}, operation="filter"
            )

        if self._repository.measure_original_bytes(original) is None:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "reason": "original_too_large",
                    "original_chars": len(original),
                },
                operation="filter",
            )

        if self._unsupported_read_content(relative_path, original):
            return self._with_diagnostics(
                {
                    "ok": False,
                    "reason": "unsupported_content",
                    "original_chars": len(original),
                },
                operation="filter",
            )
        min_chars = self._filter_limit("tool_output_filter_read_min_chars", 8000)
        if len(original) < min_chars:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "reason": "below_min_chars",
                    "original_chars": len(original),
                },
                operation="filter",
            )

        try:
            outcome = reduce_read_output(
                original,
                relative_path=relative_path,
                config=self._cfg,
            )
        except Exception:
            outcome = legacy_read_reduction(original, self._cfg)
        if outcome.text == original:
            return self._with_diagnostics(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "unchanged",
                    "original_chars": len(original),
                    "filtered_chars": len(original),
                    "original_lines": outcome.original_lines,
                    "filtered_lines": outcome.filtered_lines,
                    "detected": outcome.detected,
                    "strategy": outcome.strategy,
                    "was_shortened": False,
                    "reduction": outcome.reduction_metadata(),
                },
                operation="filter",
            )

        outcome = cap_reduction_output(
            outcome,
            max_chars=self._filter_limit("tool_output_filter_output_max_chars", 50000),
        )
        filtered = outcome.text
        if len(filtered) >= len(original):
            return self._with_diagnostics(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "not_shorter",
                    "strategy": outcome.strategy,
                    "was_shortened": False,
                    "original_chars": len(original),
                    "filtered_chars": len(original),
                    "original_lines": outcome.original_lines,
                    "filtered_lines": outcome.original_lines,
                    "detected": outcome.detected,
                    "reduction": outcome.reduction_metadata(),
                },
                operation="filter",
            )

        digest = _hash_text_parts(relative_path, original)
        slug = f"tool_output_read_{digest[:32]}"
        stored_memory_uri = self._repository.store_original(
            ctx,
            slug=slug,
            content_digest=digest,
            original=original,
            filtered=filtered,
            tool_name="Read",
            strategy=outcome.strategy,
            metadata={
                "source": "post_tool_use",
                "workspace_relative_path": relative_path,
                "detected": outcome.detected,
                "original_lines": outcome.original_lines,
                "filtered_lines": outcome.filtered_lines,
                "was_shortened": True,
                "reduction": outcome.reduction_metadata(),
            },
        )
        return self._with_diagnostics(
            {
                "ok": True,
                "filtered_output": filtered,
                "memory_uri": stored_memory_uri,
                "detected": outcome.detected,
                "strategy": outcome.strategy,
                "original_chars": len(original),
                "filtered_chars": len(filtered),
                "original_lines": outcome.original_lines,
                "filtered_lines": outcome.filtered_lines,
                "was_shortened": True,
                "reduction": outcome.reduction_metadata(),
            },
            operation="filter",
        )

    def filter_bash(self, params: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        if not self._filter_enabled("tool_output_filter_bash_enabled"):
            return self._with_diagnostics(
                {"ok": False, "reason": "feature_disabled"}, operation="filter"
            )

        command = str(params.get("command") or "").strip()
        output = str(params.get("output") or "")
        pre_category = str(params.get("pre_category") or "").strip().lower()
        if not command or not output:
            return self._with_diagnostics(
                {"ok": False, "reason": "missing_command_or_output"}, operation="filter"
            )

        if self._repository.measure_original_bytes(output) is None:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "reason": "original_too_large",
                    "original_chars": len(output),
                },
                operation="filter",
            )

        category = self.classify_bash_output(command, output, pre_category)
        if category not in _SUPPORTED_BASH_CATEGORIES:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "reason": "not_whitelisted",
                    "category": category or pre_category,
                },
                operation="filter",
            )

        min_chars = self._filter_limit("tool_output_filter_bash_min_chars", 8000)
        min_lines = self._filter_limit("tool_output_filter_bash_min_lines", 100)
        original_lines = len(output.splitlines())
        should_shorten = len(output) >= min_chars or original_lines >= min_lines
        if not should_shorten:
            outcome = passthrough_reduction(
                output,
                detected={
                    "test_runner": "test_log",
                    "diff": "diff",
                    "python_error": "python_error",
                }.get(category, "generic"),
                strategy="bash_passthrough_below_min_size",
            )
            return self._with_diagnostics(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "below_min_size",
                    "category": category,
                    "strategy": outcome.strategy,
                    "was_shortened": False,
                    "original_chars": len(output),
                    "filtered_chars": len(output),
                    "original_lines": outcome.original_lines,
                    "filtered_lines": outcome.filtered_lines,
                    "detected": outcome.detected,
                    "reduction": outcome.reduction_metadata(),
                },
                operation="filter",
            )

        try:
            outcome = reduce_bash_output(command, output, category, self._cfg)
        except Exception:
            outcome = legacy_bash_reduction(output, category, self._cfg)
        if outcome.text == output:
            return self._with_diagnostics(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "unchanged",
                    "category": category,
                    "strategy": outcome.strategy,
                    "was_shortened": False,
                    "original_chars": len(output),
                    "filtered_chars": len(output),
                    "original_lines": outcome.original_lines,
                    "filtered_lines": outcome.filtered_lines,
                    "detected": outcome.detected,
                    "reduction": outcome.reduction_metadata(),
                },
                operation="filter",
            )

        exit_code = params.get("exit_code")
        digest = _hash_text_parts(command, output, str(exit_code), pre_category)
        slug = f"tool_output_bash_{digest[:32]}"
        outcome = cap_reduction_output(
            outcome,
            max_chars=self._filter_limit("tool_output_filter_output_max_chars", 50000),
        )
        filtered = outcome.text
        strategy = outcome.strategy
        if len(filtered) >= len(output):
            return self._with_diagnostics(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "not_shorter",
                    "category": category,
                    "strategy": strategy,
                    "was_shortened": False,
                    "original_chars": len(output),
                    "filtered_chars": len(output),
                    "original_lines": outcome.original_lines,
                    "filtered_lines": original_lines,
                    "detected": outcome.detected,
                    "reduction": outcome.reduction_metadata(),
                },
                operation="filter",
            )
        stored_memory_uri = self._repository.store_original(
            ctx,
            slug=slug,
            content_digest=digest,
            original=output,
            filtered=filtered,
            tool_name="Bash",
            strategy=strategy,
            metadata={
                "source": "post_tool_use",
                "bash_category": category,
                "exit_code": exit_code,
                "pre_category": pre_category,
                "detected": outcome.detected,
                "original_lines": outcome.original_lines,
                "filtered_lines": outcome.filtered_lines,
                "was_shortened": True,
                "reduction": outcome.reduction_metadata(),
            },
        )
        return self._with_diagnostics(
            {
                "ok": True,
                "filtered_output": filtered,
                "memory_uri": stored_memory_uri,
                "category": category,
                "strategy": strategy,
                "was_shortened": True,
                "original_chars": len(output),
                "filtered_chars": len(filtered),
                "original_lines": outcome.original_lines,
                "filtered_lines": outcome.filtered_lines,
                "detected": outcome.detected,
                "reduction": outcome.reduction_metadata(),
            },
            operation="filter",
        )

    def recover_original(
        self,
        params: dict[str, Any],
        ctx: RequestContext,
    ) -> dict[str, Any]:
        memory_uri = str(
            params.get("memory_uri") or params.get("target_uri") or params.get("uri") or ""
        ).strip()
        if not memory_uri:
            return self._with_diagnostics(
                {"ok": False, "error": "memory_uri is required"}, operation="recovery"
            )

        try:
            original = self._repository.read_original(ctx, memory_uri)
        except NodeNotFoundError:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "error": "original output not found",
                    "memory_uri": memory_uri,
                },
                operation="recovery",
            )
        if not original:
            return self._with_diagnostics(
                {
                    "ok": False,
                    "error": "original output not found",
                    "memory_uri": memory_uri,
                },
                operation="recovery",
            )

        cap = self._server_recovery_cap()
        requested = self._positive_client_budget(params.get("max_chars"))
        budget = min(requested, cap) if requested is not None else cap
        returned = original
        truncated = False
        if len(original) > budget:
            returned = self._bounded_original_excerpt(original, budget)
            truncated = True

        return self._with_diagnostics(
            {
                "ok": True,
                "memory_uri": memory_uri,
                "original": returned,
                "original_chars": len(original),
                "returned_chars": len(returned),
                "truncated": truncated,
                "max_chars": budget,
            },
            operation="recovery",
        )

    def _with_diagnostics(
        self,
        result: dict[str, Any],
        *,
        operation: str,
    ) -> dict[str, Any]:
        """Add a bounded, non-sensitive projection to a terminal result.

        The normal result remains the compatibility contract. Diagnostics only
        derive state, counts, configured/effective budgets, and known enum
        values; they never copy paths, URIs, command/output text, or errors.
        """

        def count(value: object) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int) and value >= 0:
                return value
            return None

        def enum(value: object) -> str | None:
            if isinstance(value, str) and _SAFE_DIAGNOSTIC_ENUM_RE.fullmatch(value):
                return value
            return None

        projected = dict(result)
        if operation == "recovery":
            recovered = bool(projected.get("ok"))
            diagnostics: dict[str, Any] = {
                "state": "recovered" if recovered else "recovery_failed",
                "reason": None if recovered else "not_found_or_not_visible",
                "input_chars": count(projected.get("original_chars")),
                "output_chars": count(projected.get("returned_chars")),
                "truncated": bool(projected.get("truncated")) if recovered else False,
                "budget": {
                    "configured_max_chars": self._server_recovery_cap(),
                    "effective_max_chars": count(projected.get("max_chars")),
                },
            }
        else:
            raw_reason = projected.get("reason")
            reason = raw_reason if raw_reason in _FILTER_DIAGNOSTIC_REASONS else None
            reduction = projected.get("reduction")
            reduction_mapping = reduction if isinstance(reduction, dict) else {}
            fallback = enum(reduction_mapping.get("fallback"))
            if bool(projected.get("ok")):
                if bool(projected.get("skipped")):
                    state = "skipped"
                elif fallback not in {None, "none", "passthrough"}:
                    state = "fallback"
                else:
                    state = "filtered"
            elif reason == "feature_disabled":
                state = "disabled"
            elif reason in _UNSUPPORTED_FILTER_REASONS:
                state = "unsupported"
            elif reason in _SKIPPED_FILTER_REASONS:
                state = "skipped"
            else:
                state = "error"
            diagnostics = {
                "state": state,
                "reason": reason,
                "fallback": fallback,
                "input_chars": count(projected.get("original_chars")),
                "output_chars": count(projected.get("filtered_chars")),
                "detected": enum(projected.get("detected")) if bool(projected.get("ok")) else None,
                "strategy": enum(projected.get("strategy")) if bool(projected.get("ok")) else None,
                "was_shortened": bool(projected.get("was_shortened")),
                "budget": {
                    "configured_max_chars": self._filter_limit(
                        "tool_output_filter_output_max_chars", 50000
                    ),
                    "effective_max_chars": count(reduction_mapping.get("output_max_chars")),
                },
                "reduction": {
                    "omitted_lines": count(reduction_mapping.get("omitted_lines")),
                    "omitted_chars": count(reduction_mapping.get("omitted_chars")),
                    "preserved_blocks": count(reduction_mapping.get("preserved_blocks")),
                    "char_cap_applied": bool(reduction_mapping.get("char_cap_applied")),
                },
            }
        projected["diagnostics"] = diagnostics
        return projected

    def _filter_enabled(self, child_flag: str) -> bool:
        return bool(
            getattr(self._cfg, "tool_output_filter_enabled", False)
            and getattr(self._cfg, child_flag, True)
        )

    def _server_recovery_cap(self) -> int:
        try:
            value = int(getattr(self._cfg, "tool_output_recovery_max_chars", 50000))
        except (TypeError, ValueError):
            value = 50000
        return value if value > 0 else 50000

    @staticmethod
    def _bounded_original_excerpt(original: str, budget: int) -> str:
        """Return a recovery excerpt whose rendered body never exceeds ``budget``."""

        marker = f"\n\n... [original output clipped: {len(original)} chars omitted] ...\n\n"
        for _ in range(4):
            source_budget = budget - len(marker)
            if source_budget < 2:
                break
            omitted = len(original) - source_budget
            next_marker = f"\n\n... [original output clipped: {omitted} chars omitted] ...\n\n"
            if next_marker == marker:
                head = source_budget // 2
                tail = source_budget - head
                return original[:head] + marker + original[-tail:]
            marker = next_marker

        # Very small caller budgets cannot contain both an unambiguous marker
        # and any source text. Preserve a bounded source prefix rather than
        # exceed the caller's explicit ceiling.
        return original[:budget]

    def _filter_limit(self, name: str, fallback: int) -> int:
        value = getattr(self._cfg, name, fallback)
        if isinstance(value, bool):
            return fallback
        try:
            value = int(value)
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    @staticmethod
    def _has_explicit_read_range(params: dict[str, Any]) -> bool:
        return any(
            key in params and params[key] is not None and str(params[key]).strip() != ""
            for key in ("offset", "limit")
        )

    @staticmethod
    def _unsupported_read_content(relative_path: str, content: str) -> bool:
        if "\x00" in content:
            return True
        suffix = "." + relative_path.rsplit(".", 1)[-1].lower() if "." in relative_path else ""
        return suffix in {
            ".avif",
            ".bmp",
            ".gif",
            ".heic",
            ".ico",
            ".ipynb",
            ".jpeg",
            ".jpg",
            ".pdf",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }

    @staticmethod
    def _positive_client_budget(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def classify_bash_output(command: str, output: str, pre_category: str = "") -> str:
        lower = (command or "").lower()
        normalized_pre = (pre_category or "").strip().lower()
        if normalized_pre in _SUPPORTED_BASH_CATEGORIES:
            return normalized_pre
        if normalized_pre == "diff_maybe":
            if re.search(r"^diff --git a/", output, re.MULTILINE) or re.search(
                r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@",
                output,
                re.MULTILINE,
            ):
                return "diff"
            return ""
        if normalized_pre == "python_maybe":
            if re.search(
                r"Traceback \(most recent call last\)|\b\w*(Error|Exception):",
                output,
            ):
                return "python_error"
            return ""
        if re.search(
            r"(^|[\s;&|()])(?:python(?:\d+(?:\.\d+)?)?\s+-m\s+)?pytest\b",
            lower,
        ):
            return "test_runner"
        if re.search(r"(^|[\s;&|()])(?:tox|npm\s+(?:run\s+)?test)\b", lower):
            return "test_runner"
        if re.search(r"(^|[\s;&|()])git\s+(?:diff|show|log\s+-p)\b", lower):
            if "diff --git " in output or "@@ " in output:
                return "diff"
            return ""
        if re.search(r"(^|[\s;&|()])(?:hg|svn)\s+diff\b", lower):
            return "diff" if "@@ " in output else ""
        if re.search(
            r"Traceback \(most recent call last\)|\b\w*(Error|Exception):",
            output,
        ):
            return "python_error"
        return ""

    @staticmethod
    def shorten_bash_output(command: str, output: str, category: str) -> tuple[str, str]:
        del command
        if category == "test_runner":
            kept: list[str] = []
            patterns = (
                r"=+\s+FAILURES\s+=+",
                r"=+\s+ERRORS\s+=+",
                r"FAILED\s+",
                r"ERROR\s+",
                r"Traceback \(most recent call last\)",
                r"AssertionError",
                r"\b\d+\s+failed\b",
                r"\b\d+\s+passed\b",
                r"short test summary info",
            )
            for line in output.splitlines():
                if any(re.search(pattern, line) for pattern in patterns):
                    kept.append(line)
            if kept:
                return "\n".join(kept[:120]), "bash_test_runner"
            return (
                _shorten_head_tail(output, max_lines=80, label="test output folded"),
                "bash_test_runner_head_tail",
            )
        if category == "diff":
            return (
                _shorten_head_tail(output, max_lines=100, label="diff folded"),
                "bash_diff_head_tail",
            )
        if category == "python_error":
            lines = output.splitlines()
            traceback_start = 0
            for index, line in enumerate(lines):
                if "Traceback (most recent call last)" in line:
                    traceback_start = index
            return "\n".join(lines[traceback_start:][-100:]), "bash_python_error_tail"
        return _shorten_head_tail(output, max_lines=80, label="output folded"), "bash_head_tail"
