"""Assembly Service - assembles retrieved memory blocks.

Combines SeedHit from L0 with L1/L2 content from MemoryReadService.
Phase 2: Adds relation edges from RelationStore for graph expansion.

See CLAUDE.md §7 for retrieval chain rules.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.interfaces import ContextFS, RelationStore
from core.models import RequestContext, SeedHit, RelationEdge


@dataclass
class RetrievedBlock:
    """A fully retrieved memory block with content at requested level.

    Phase 2: Contains uri, abstract, overview, content, and relations.
    """
    uri: str
    score: float
    abstract: str
    overview: Optional[str] = None
    content: Optional[str] = None
    relations: list[RelationEdge] = field(default_factory=list)

    @property
    def has_overview(self) -> bool:
        return self.overview is not None

    @property
    def has_content(self) -> bool:
        return self.content is not None

    @property
    def has_relations(self) -> bool:
        return len(self.relations) > 0


@dataclass
class RetrievalResult:
    """Complete retrieval result with metadata.

    Returned by AssemblyService.assemble() for full results.
    """
    query: str
    blocks: list[RetrievedBlock]
    total_count: int
    account_id: str

    @property
    def has_results(self) -> bool:
        return len(self.blocks) > 0


class AssemblyService:
    """Assembles RetrievedBlock from SeedHit with additional content.

    Takes L0 SeedHit results and optionally enriches with L1/L2 content.
    Phase 2: Adds relation edges from RelationStore when available.
    """

    def __init__(
        self,
        fs: ContextFS,
        relation_store: Optional[RelationStore] = None,
    ):
        """Initialize AssemblyService.

        Args:
            fs: ContextFS for reading node content
            relation_store: Optional RelationStore for fetching relations
        """
        self._fs = fs
        self._relation_store = relation_store

    def assemble(
        self,
        seed_hits: list[SeedHit],
        include_overview: bool = False,
        include_content: bool = False,
        include_relations: bool = False,
        ctx: Optional[RequestContext] = None,
    ) -> list[RetrievedBlock]:
        """Assemble RetrievedBlock from SeedHit results.

        Args:
            seed_hits: L0 retrieval results from VectorIndex
            include_overview: If True, fetch L1 overview for each hit
            include_content: If True, fetch L2 content for each hit
            include_relations: If True, fetch relation edges (requires RelationStore and ctx)
            ctx: RequestContext for access control (required if including content or relations)

        Returns:
            List of RetrievedBlock with requested content levels

        Note:
            If ctx is None, only abstract is included (no L1/L2/relations fetch).
            This allows cheap L0-only results without access control overhead.
        """
        blocks = []

        for hit in seed_hits:
            block = RetrievedBlock(
                uri=hit.uri,
                score=hit.score,
                abstract=hit.abstract,
            )

            # Skip content fetch if no context
            if ctx is None:
                blocks.append(block)
                continue

            # Fetch relations if requested and RelationStore is available
            if include_relations and self._relation_store is not None:
                try:
                    block.relations = self._relation_store.get_edges(hit.uri, ctx)
                except Exception:
                    # Relation fetch failed, keep empty list
                    block.relations = []

            # Fetch overview if requested and available
            if include_overview and hit.has_overview:
                try:
                    from retrieval.memory_read_service import MemoryReadService
                    reader = MemoryReadService(self._fs)
                    block.overview = reader.read(hit.uri, "L1", ctx)
                except Exception:
                    # Node may have been deleted or access denied
                    # Keep the block without overview
                    pass

            # Fetch content if requested and available
            if include_content and hit.has_content:
                try:
                    from retrieval.memory_read_service import MemoryReadService
                    reader = MemoryReadService(self._fs)
                    block.content = reader.read(hit.uri, "L2", ctx)
                except Exception:
                    # Node may have been deleted or access denied
                    # Keep the block without content
                    pass

            blocks.append(block)

        return blocks

    def assemble_result(
        self,
        query: str,
        seed_hits: list[SeedHit],
        account_id: str,
        include_overview: bool = False,
        include_content: bool = False,
        include_relations: bool = False,
        ctx: Optional[RequestContext] = None,
    ) -> RetrievalResult:
        """Assemble complete RetrievalResult with metadata.

        Args:
            query: Original query text
            seed_hits: L0 retrieval results
            account_id: Account ID for the query
            include_overview: Whether to include L1 overview
            include_content: Whether to include L2 content
            include_relations: Whether to include relation edges
            ctx: RequestContext for access control

        Returns:
            RetrievalResult with blocks and metadata
        """
        blocks = self.assemble(
            seed_hits=seed_hits,
            include_overview=include_overview,
            include_content=include_content,
            include_relations=include_relations,
            ctx=ctx,
        )

        return RetrievalResult(
            query=query,
            blocks=blocks,
            total_count=len(blocks),
            account_id=account_id,
        )

    def assemble_top_k(
        self,
        query: str,
        seed_hits: list[SeedHit],
        account_id: str,
        top_k: int,
        include_overview: bool = False,
        include_relations: bool = False,
        ctx: Optional[RequestContext] = None,
    ) -> RetrievalResult:
        """Assemble result with top-k blocks.

        Args:
            query: Original query text
            seed_hits: L0 retrieval results (may be more than top_k)
            account_id: Account ID for the query
            top_k: Maximum number of blocks to return
            include_overview: Whether to include L1 overview
            include_relations: Whether to include relation edges
            ctx: RequestContext for access control

        Returns:
            RetrievalResult with at most top_k blocks
        """
        # Take top-k by score
        top_hits = sorted(seed_hits, key=lambda h: h.score, reverse=True)[:top_k]

        return self.assemble_result(
            query=query,
            seed_hits=top_hits,
            account_id=account_id,
            include_overview=include_overview,
            include_content=False,  # L2 content fetched on-demand
            include_relations=include_relations,
            ctx=ctx,
        )
