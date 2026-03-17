"""Service API layer - external interface for ContextEngine.

This is the ONLY layer where RequestContext is mandatory and account_id is injected.
All external calls must provide a RequestContext for multi-tenant isolation.

See CLAUDE.md §7 for tool interface spec and §8 for multi-tenant rules.
"""

from typing import Optional, Literal

from core.errors import AccessDeniedError, NodeNotFoundError
from core.interfaces import ContextFS, VectorIndex
from core.models import RequestContext, SeedHit
from retrieval import Planner, L0Retriever, MemoryReadService, AssemblyService, RetrievedBlock, RetrievalResult


Level = Literal["L0", "L1", "L2"]


class MemoryReadAPI:
    """Public API for memory search and read operations.

    All methods enforce multi-tenant isolation via RequestContext.
    account_id is injected from context, cannot be overridden by callers.
    """

    def __init__(self, fs: ContextFS, vector_index: VectorIndex):
        """Initialize the API with required dependencies.

        Args:
            fs: ContextFS implementation for reading nodes
            vector_index: VectorIndex implementation for similarity search
        """
        self._fs = fs
        self._vector_index = vector_index

        # Initialize retrieval components
        self._planner = Planner()
        self._l0_retriever = L0Retriever(vector_index)
        self._read_service = MemoryReadService(fs)
        self._assembly = AssemblyService(fs)

    def search_memory(
        self,
        query: str,
        ctx: RequestContext,
        top_k: int = 10,
        category: Optional[list[str]] = None,
        include_overview: bool = False,
    ) -> list[dict]:
        """Search memories using vector similarity.

        This is the L0 entry point for memory retrieval.
        account_id is MANDATORY and injected from RequestContext.

        Args:
            query: Natural language query text
            ctx: RequestContext containing account_id (injected, not from caller)
            top_k: Maximum number of results to return (default 10)
            category: Optional list of categories to filter by
            include_overview: If True, include L1 overview in results

        Returns:
            List of dicts with keys: uri, score, abstract, overview (optional)

        Raises:
            AccessDeniedError: If caller attempts to override account_id

        Note:
            account_id is ALWAYS taken from ctx.account_id.
            Callers cannot specify a different account_id (security requirement).
        """
        # Step 1: Use Planner to infer query type
        typed_query = self._planner.plan(query, ctx)

        # Step 2: Override account_id from context (MANDATORY - caller cannot override)
        # This is the critical security injection point
        typed_query.account_id = ctx.account_id

        # Apply category filter if specified
        if category is not None:
            typed_query.categories = category

        # Override top_k if specified
        if top_k is not None:
            typed_query.top_k = top_k

        # Step 3: L0 vector retrieval
        seed_hits = self._l0_retriever.retrieve(typed_query, ctx)

        # Step 4: Assemble results
        result = self._assembly.assemble_top_k(
            query=query,
            seed_hits=seed_hits,
            account_id=ctx.account_id,
            top_k=top_k,
            include_overview=include_overview,
            ctx=ctx,
        )

        # Convert to dict format for JSON serialization
        return [
            {
                "uri": block.uri,
                "score": block.score,
                "abstract": block.abstract,
                "overview": block.overview if include_overview else None,
                "has_overview": block.has_overview,
                "has_content": block.has_content,
            }
            for block in result.blocks
        ]

    def read_memory(
        self,
        uri: str,
        level: Level,
        ctx: RequestContext,
    ) -> str:
        """Read a memory node at the specified level.

        Args:
            uri: ContextNode URI (e.g., ctx://{account}/users/{user}/memories/profile)
            level: "L0" for abstract, "L1" for overview, "L2" for full content
            ctx: RequestContext for account validation

        Returns:
            Content at the requested level:
            - L0: abstract (≤100 chars summary)
            - L1: overview (structured summary)
            - L2: content (full original text)

        Raises:
            AccessDeniedError (403): If URI account prefix != ctx.account_id
            NodeNotFoundError (404): If node does not exist or is not ACTIVE
            ValueError: If level is not "L0", "L1", or "L2"

        Note:
            This method validates that the URI's account prefix matches
            the RequestContext.account_id. Cross-account access is ALWAYS denied.
        """
        # Extract and validate account from URI
        uri_account = self._extract_account_from_uri(uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                f"URI account '{uri_account}' does not match context account '{ctx.account_id}'"
            )

        # L0 returns abstract from the node metadata
        if level == "L0":
            node = self._read_service.read_node(uri, ctx)
            return node.abstract

        # L1 and L2 are handled by MemoryReadService
        return self._read_service.read(uri, level, ctx)

    def search_and_read(
        self,
        query: str,
        ctx: RequestContext,
        top_k: int = 3,
        read_level: Level = "L1",
    ) -> list[dict]:
        """Search and read in one call - convenience method.

        Performs search_memory and then read_memory for each result.

        Args:
            query: Natural language query
            ctx: RequestContext
            top_k: Number of results to fetch and read
            read_level: Level to read (L0/L1/L2)

        Returns:
            List of dicts with uri, score, and content at requested level
        """
        # Search for URIs
        search_results = self.search_memory(query, ctx, top_k=top_k)

        # Read each result
        results = []
        for result in search_results:
            try:
                content = self.read_memory(result["uri"], read_level, ctx)
                results.append({
                    "uri": result["uri"],
                    "score": result["score"],
                    "content": content,
                })
            except (AccessDeniedError, NodeNotFoundError):
                # Skip results that can't be read
                continue

        return results

    def _extract_account_from_uri(self, uri: str) -> str:
        """Extract account ID from a ContextNode URI.

        URI format: ctx://{account}/users/{user}/memories/...
                    or ctx://{account}/agents/{agent}/...

        Args:
            uri: ContextNode URI

        Returns:
            Account ID string

        Raises:
            ValueError: If URI format is invalid
        """
        if not uri.startswith("ctx://"):
            raise ValueError(f"Invalid URI format: {uri}")

        path = uri[6:]  # Remove "ctx://"
        parts = path.split("/")

        if len(parts) < 2:
            raise ValueError(f"Invalid URI format: {uri}")

        return parts[0]


# Singleton instance for simple usage
# In production, use dependency injection
_default_api: Optional[MemoryReadAPI] = None


def init_api(fs: ContextFS, vector_index: VectorIndex) -> MemoryReadAPI:
    """Initialize the global API instance.

    Args:
        fs: ContextFS implementation
        vector_index: VectorIndex implementation

    Returns:
        Configured MemoryReadAPI instance
    """
    global _default_api
    _default_api = MemoryReadAPI(fs, vector_index)
    return _default_api


def get_api() -> Optional[MemoryReadAPI]:
    """Get the global API instance.

    Returns:
        MemoryReadAPI if initialized, None otherwise
    """
    return _default_api
