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

    def prefetch(self, memory_type: str, ctx: RequestContext) -> PrefetchResult:
        """Prefetch existing memories for a given memory type.

        Args:
            memory_type: Memory type identifier (e.g., "profile", "preference")
            ctx: RequestContext for multi-tenant isolation

        Returns:
            PrefetchResult with formatted messages and tracked URIs
        """
        schema = self._registry.get(memory_type)
        if schema is None:
            logger.warning(f"Cannot prefetch unknown type: {memory_type}")
            return PrefetchResult()

        # Route based on operation_mode and file structure
        if schema.operation_mode == "add_only":
            return self._prefetch_add_only(memory_type, ctx)
        elif schema.operation_mode == "upsert":
            if schema.is_single_file:
                return self._prefetch_single_file(memory_type, ctx)
            else:
                return self._prefetch_multi_file(memory_type, ctx)
        else:
            logger.warning(f"Unknown operation_mode '{schema.operation_mode}' for {memory_type}")
            return PrefetchResult()

    def prefetch_for_span(
        self,
        categories: list[str],
        ctx: RequestContext,
    ) -> PrefetchResult:
        """Prefetch memories for multiple categories (e.g., from a span).

        Args:
            categories: List of memory types to prefetch
            ctx: RequestContext

        Returns:
            Combined PrefetchResult from all categories
        """
        combined = PrefetchResult()
        for cat in categories:
            try:
                result = self.prefetch(cat, ctx)
                combined.messages.extend(result.messages)
                combined.read_uris.update(result.read_uris)
                combined.listed_uris.update(result.listed_uris)
            except Exception as e:
                logger.error(f"Prefetch failed for {cat}: {e}")
        return combined

    def _prefetch_single_file(self, memory_type: str, ctx: RequestContext) -> PrefetchResult:
        """Prefetch for single-file schemas (e.g., profile)."""
        result = PrefetchResult()
        try:
            uri = self._uri_resolver.resolve(memory_type, {}, ctx)
            result.listed_uris.add(uri)

            if self._fs.exists(uri, ctx):
                node = self._fs.read_node(uri, ctx)
                result.read_uris.add(uri)
                msg = (
                    f"== Existing {memory_type} memory ==\n"
                    f"URI: {uri}\n"
                    f"Abstract: {node.abstract}\n"
                    f"Overview: {node.overview}\n"
                    f"Content: {node.content}\n"
                )
                result.messages.append(msg)
                logger.info(f"Prefetched single-file {memory_type} from {uri}")
            else:
                logger.debug(f"No existing {memory_type} at {uri}")
        except Exception as e:
            logger.error(f"Failed to prefetch single-file {memory_type}: {e}")
        return result

    def _prefetch_multi_file(self, memory_type: str, ctx: RequestContext) -> PrefetchResult:
        """Prefetch for multi-file schemas using vector search."""
        result = PrefetchResult()
        try:
            # Embed memory_type name as query (simple but effective)
            vectors = self._embedder.embed_texts([memory_type])
            query_vector = vectors[0]

            # Search for existing memories of this type
            filters = {
                "category": memory_type,
                "account_id": ctx.account_id,
            }
            # Add owner_space filter based on schema
            schema = self._registry.get(memory_type)
            if schema and schema.owner_scope == "user":
                filters["owner_space"] = f"user:{ctx.user_id}"
            elif schema and schema.owner_scope == "agent":
                filters["owner_space"] = f"agent:{ctx.agent_id}"

            hits = self._vector_index.search_by_vector(
                query_vector=query_vector,
                filters=filters,
                top_k=10,
            )

            if not hits:
                logger.debug(f"No vector hits for {memory_type}")
                return result

            # Read full content for each hit
            for hit in hits:
                result.listed_uris.add(hit.uri)
                node_uri = hit.uri
                if node_uri.endswith("/content.md"):
                    node_uri = node_uri[: -len("/content.md")]
                try:
                    node = self._fs.read_node(node_uri, ctx)
                    result.read_uris.add(node_uri)
                    msg = (
                        f"[{memory_type}] URI: {node_uri}\n"
                        f"Abstract: {node.abstract}\n"
                        f"Overview: {node.overview}\n"
                        f"Content: {node.content}\n"
                        f"(similarity: {hit.score:.2f})"
                    )
                    result.messages.append(msg)
                except Exception as e:
                    logger.warning(f"Failed to read {hit.uri}: {e}")
                    # Still include abstract from SeedHit
                    msg = (
                        f"[{memory_type}] URI: {hit.uri}\n"
                        f"Abstract: {hit.abstract}\n"
                        f"(similarity: {hit.score:.2f}, read failed)"
                    )
                    result.messages.append(msg)

            logger.info(f"Prefetched {len(result.read_uris)} multi-file {memory_type} memories")

        except Exception as e:
            logger.error(f"Failed to prefetch multi-file {memory_type}: {e}")
        return result

    def _prefetch_add_only(self, memory_type: str, ctx: RequestContext) -> PrefetchResult:
        """Prefetch recent entries for add-only schemas (event, case)."""
        result = PrefetchResult()
        try:
            dir_uri = self._uri_resolver.get_directory_uri(memory_type, ctx)

            children = self._fs.list_children(dir_uri, ctx)
            if not children:
                logger.debug(f"No children for {memory_type} at {dir_uri}")
                return result

            # Sort by URI (timestamps in URI allow lexicographic sort)
            children_sorted = sorted(children, reverse=True)
            recent = children_sorted[:10]

            for uri in recent:
                result.listed_uris.add(uri)
                try:
                    node = self._fs.read_node(uri, ctx)
                    result.read_uris.add(uri)
                    msg = (
                        f"[Recent {memory_type}] URI: {uri}\n"
                        f"Abstract: {node.abstract}\n"
                        f"Overview: {node.overview}\n"
                    )
                    result.messages.append(msg)
                except Exception as e:
                    logger.warning(f"Failed to read recent {memory_type} {uri}: {e}")

            logger.info(f"Prefetched {len(result.read_uris)} recent {memory_type} entries")

        except Exception as e:
            logger.error(f"Failed to prefetch add-only {memory_type}: {e}")
        return result
