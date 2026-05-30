# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Eager prefetch for existing memories before LLM extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional

from core.interfaces import ContextFS, VectorIndex, Embedder
from core.models import RequestContext, SeedHit
logger = getLogger(__name__)


@dataclass
class PrefetchResult:
    """Result of prefetching existing memories for a memory type."""
    messages: list[str] = field(default_factory=list)
    read_uris: set[str] = field(default_factory=set)
    listed_uris: set[str] = field(default_factory=set)
    # Lazy mode: URI hints without full content
    uri_hints: list[dict] = field(default_factory=list)  # [{"uri": str, "abstract": str, "score": float}]
    is_lazy: bool = False


class MemoryPrefetcher:
    """Prefetch existing memories before LLM extraction.

    By operation_mode:
    - single-file upsert (profile): read whole file
    - multi-file upsert (preference/entity/pattern/skill/tool): vector search + read top hits
    - add_only (event/case): list recent entries, read overviews only
    """

    def __init__(
        self,
        fs: ContextFS,
        vector_index: VectorIndex,
        embedder: Embedder,
        registry: "SchemaRegistry",
        uri_resolver: "URIResolver",
    ):
        self._fs = fs
        self._vector_index = vector_index
        self._embedder = embedder
        self._registry = registry
        self._uri_resolver = uri_resolver

    def _prefetch(
        self,
        memory_type: str,
        ctx: RequestContext,
        *,
        conversation_text: str | None = None,
        lazy: bool = False,
    ) -> PrefetchResult:
        """Prefetch existing memories for a given memory type (internal method).

        Args:
            memory_type: Memory type identifier (e.g., "profile", "preference")
            ctx: RequestContext for multi-tenant isolation
            conversation_text: Recent conversation text for semantic query.
            lazy: If True, collect URI hints only (no content reading).

        Returns:
            PrefetchResult with messages (eager) or uri_hints (lazy)
        """
        schema = self._registry.get_compatible(memory_type)
        if schema is None:
            logger.warning(f"Cannot prefetch unknown or incompatible type: {memory_type}")
            return PrefetchResult(is_lazy=lazy)

        if schema.operation_mode == "add_only":
            return self._prefetch_add_only(memory_type, ctx, lazy=lazy)
        elif schema.operation_mode == "upsert":
            if schema.is_single_file:
                return self._prefetch_single_file(memory_type, ctx, lazy=lazy)
            else:
                return self._prefetch_multi_file(memory_type, ctx, conversation_text=conversation_text, lazy=lazy)
        else:
            logger.warning(f"Unknown operation_mode '{schema.operation_mode}' for {memory_type}")
            return PrefetchResult(is_lazy=lazy)

    def prefetch_for_span(
        self,
        categories: list[str],
        ctx: RequestContext,
        conversation_text: str | None = None,
        lazy: bool = False,
    ) -> PrefetchResult:
        """Prefetch memories for multiple categories.

        Args:
            categories: List of memory types to prefetch
            ctx: RequestContext
            conversation_text: Recent conversation text for semantic query.
            lazy: If True, collect URI hints only.

        Returns:
            Combined PrefetchResult from all categories
        """
        combined = PrefetchResult(is_lazy=lazy)
        for cat in categories:
            try:
                result = self._prefetch(cat, ctx, conversation_text=conversation_text, lazy=lazy)
                combined.messages.extend(result.messages)
                combined.read_uris.update(result.read_uris)
                combined.listed_uris.update(result.listed_uris)
                combined.uri_hints.extend(result.uri_hints)
            except Exception as e:
                logger.error(f"Prefetch failed for {cat}: {e}")
        return combined

    
    def _prefetch_single_file(self, memory_type: str, ctx: RequestContext, *, lazy: bool = False) -> PrefetchResult:
        """Prefetch for single-file schemas (e.g., profile)."""
        result = PrefetchResult(is_lazy=lazy)
        try:
            uri = self._uri_resolver.resolve(memory_type, {}, ctx)
            result.listed_uris.add(uri)

            if not self._fs.exists(uri, ctx):
                logger.debug(f"No existing {memory_type} at {uri}")
                return result

            if lazy:
                # Lazy: just get abstract (from vector index or minimal read)
                abstract = self._get_abstract_for_uri(uri, memory_type, ctx)
                result.uri_hints.append({"uri": uri, "abstract": abstract, "score": 1.0})
                logger.info(f"Lazy prefetch single-file {memory_type}: URI hint")
            else:
                # Eager: read full content
                node = self._fs.read_node(uri, ctx)
                result.read_uris.add(uri)
                result.messages.append(
                    f"== Existing {memory_type} memory ==\n"
                    f"URI: {uri}\n"
                    f"Abstract: {node.abstract}\n"
                    f"Overview: {node.overview}\n"
                    f"Content: {node.content}\n"
                )
                logger.info(f"Prefetched single-file {memory_type} from {uri}")

        except Exception as e:
            logger.error(f"Failed to prefetch single-file {memory_type}: {e}")
        return result

    def _prefetch_multi_file(
        self,
        memory_type: str,
        ctx: RequestContext,
        *,
        conversation_text: str | None = None,
        lazy: bool = False,
    ) -> PrefetchResult:
        """Prefetch for multi-file schemas using vector search."""
        result = PrefetchResult(is_lazy=lazy)
        try:
            query_text = conversation_text if conversation_text is not None else memory_type
            vectors = self._embedder.embed_texts([query_text])
            query_vector = vectors[0]

            filters = self._build_filters(memory_type, ctx)
            hits = self._vector_index.search_by_vector(query_vector=query_vector, filters=filters, top_k=10)

            if not hits:
                logger.debug(f"No vector hits for {memory_type}")
                return result

            for hit in hits:
                node_uri = hit.uri
                if node_uri.endswith("/content.md"):
                    node_uri = node_uri[: -len("/content.md")]
                result.listed_uris.add(node_uri)

                if lazy:
                    # Lazy: use abstract from hit directly
                    result.uri_hints.append({"uri": node_uri, "abstract": hit.abstract, "score": hit.score})
                else:
                    # Eager: read full content
                    try:
                        node = self._fs.read_node(node_uri, ctx)
                        result.read_uris.add(node_uri)
                        result.messages.append(
                            f"[{memory_type}] URI: {node_uri}\n"
                            f"Abstract: {node.abstract}\n"
                            f"Overview: {node.overview}\n"
                            f"Content: {node.content}\n"
                            f"(similarity: {hit.score:.2f})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read {hit.uri}: {e}")
                        result.messages.append(
                            f"[{memory_type}] URI: {hit.uri}\n"
                            f"Abstract: {hit.abstract}\n"
                            f"(similarity: {hit.score:.2f}, read failed)"
                        )

            count = len(result.uri_hints) if lazy else len(result.read_uris)
            logger.info(f"{'Lazy prefetch' if lazy else 'Prefetched'} {memory_type}: {count} items")

        except Exception as e:
            logger.error(f"Failed to prefetch multi-file {memory_type}: {e}")
        return result

    def _prefetch_add_only(self, memory_type: str, ctx: RequestContext, *, lazy: bool = False) -> PrefetchResult:
        """Prefetch recent entries for add-only schemas (event, case)."""
        result = PrefetchResult(is_lazy=lazy)
        try:
            dir_uri = self._uri_resolver.get_directory_uri(memory_type, ctx)
            children = self._fs.list_children(dir_uri, ctx)

            if not children:
                logger.debug(f"No children for {memory_type} at {dir_uri}")
                return result

            children_sorted = sorted(children, reverse=True)
            recent = children_sorted[:10]

            for uri in recent:
                result.listed_uris.add(uri)

                if lazy:
                    # Lazy: no abstract available from list, add empty hint
                    result.uri_hints.append({"uri": uri, "abstract": "", "score": 0.0})
                else:
                    # Eager: read overview
                    try:
                        node = self._fs.read_node(uri, ctx)
                        result.read_uris.add(uri)
                        result.messages.append(
                            f"[Recent {memory_type}] URI: {uri}\n"
                            f"Abstract: {node.abstract}\n"
                            f"Overview: {node.overview}\n"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read recent {memory_type} {uri}: {e}")

            count = len(result.uri_hints) if lazy else len(result.read_uris)
            logger.info(f"{'Lazy prefetch' if lazy else 'Prefetched'} {memory_type}: {count} recent entries")

        except Exception as e:
            logger.error(f"Failed to prefetch add-only {memory_type}: {e}")
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_filters(self, memory_type: str, ctx: RequestContext) -> dict:
        """Build vector search filters for a memory type."""
        filters = {"category": memory_type, "account_id": ctx.account_id}
        schema = self._registry.get(memory_type)
        if schema and schema.owner_scope == "user":
            filters["owner_space"] = f"user:{ctx.user_id}"
        elif schema and schema.owner_scope == "agent":
            filters["owner_space"] = f"agent:{ctx.agent_id}"
        return filters

    def _get_abstract_for_uri(self, uri: str, memory_type: str, ctx: RequestContext) -> str:
        """Get abstract for a URI (for lazy single-file prefetch)."""
        # Try vector index first
        try:
            filters = self._build_filters(memory_type, ctx)
            vectors = self._embedder.embed_texts([memory_type])
            hits = self._vector_index.search_by_vector(query_vector=vectors[0], filters=filters, top_k=1)
            if hits and hits[0].uri.rstrip("/content.md") == uri:
                return hits[0].abstract
        except Exception:
            pass
        # Fallback: minimal read
        node = self._fs.read_node(uri, ctx)
        return node.abstract