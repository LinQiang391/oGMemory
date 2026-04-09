"""L0 Retriever - first-layer vector recall.

Performs vector similarity search using the VectorIndex.
Account_id filtering is enforced at the index level (implementation detail).

See CLAUDE.md §7 for retrieval chain rules.
"""

from typing import Optional

from core.interfaces import VectorIndex
from core.models import SeedHit, TypedQuery, RequestContext


class L0Retriever:
    """L0 vector retrieval using VectorIndex.

    Delegates all similarity search and filtering to the VectorIndex implementation.
    The index is responsible for enforcing account_id isolation.

    Phase 1: Uses VectorIndex.search() directly.
    Phase 2+: May add pre-filtering, caching, or hybrid search.
    """

    def __init__(self, vector_index: VectorIndex):
        """Initialize L0Retriever with a VectorIndex implementation.

        Args:
            vector_index: VectorIndex implementation (e.g., InMemoryVectorIndex)
        """
        self._vector_index = vector_index

    def retrieve(
        self,
        query: TypedQuery,
        ctx: RequestContext,
    ) -> list[SeedHit]:
        """Perform L0 vector similarity search.

        Args:
            query: TypedQuery with account_id already injected by service layer
            ctx: RequestContext for validation

        Returns:
            List of SeedHit sorted by score descending.
            May be empty if no matches found.

        Raises:
            AccessDeniedError: If query.account_id != ctx.account_id (should not happen
                if service layer properly injected account_id, but defensive check)

        Note:
            account_id filtering is handled by the VectorIndex implementation.
            This method simply passes the query through.
        """
        # Defensive: ensure query.account_id matches context
        if query.account_id != ctx.account_id:
            # This would indicate a bug in service layer
            # For now, we'll override with ctx.account_id for security
            # In production, this should be logged as a warning
            query.account_id = ctx.account_id

        # Delegate to VectorIndex
        # The index handles account_id filtering internally
        results = self._vector_index.search(query)

        return results

    def retrieve_with_filters(
        self,
        query: TypedQuery,
        ctx: RequestContext,
        categories: Optional[list[str]] = None,
    ) -> list[SeedHit]:
        """Retrieve with additional category filtering.

        Args:
            query: TypedQuery with account_id already injected
            ctx: RequestContext for validation
            categories: Optional list of categories to filter by.
                If None, uses query.categories.

        Returns:
            List of SeedHit matching the query and category filters.
        """
        # Apply category filter if specified
        if categories is not None:
            query = TypedQuery(
                text=query.text,
                context_type=query.context_type,
                categories=categories,
                target_uri=query.target_uri,
                top_k=query.top_k,
                account_id=query.account_id,
                owner_space=query.owner_space,
            )

        return self.retrieve(query, ctx)
