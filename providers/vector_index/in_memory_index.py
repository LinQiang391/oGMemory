"""In-Memory Vector Index for testing.

Simple in-memory implementation of VectorIndex protocol.
Supports filter matching for cross-tenant isolation testing.

Used by contract tests to verify IndexRecord.filters behavior.
"""

import math
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, TypedQuery, SeedHit


class InMemoryVectorIndex(VectorIndex):
    """In-memory vector index for testing.

    Stores records in a dict and performs cosine similarity search.
    Filters are applied to enforce tenant isolation.

    This is a TEST implementation using simple cosine similarity.
    Production should use a proper vector database (Qdrant, Milvus, etc.).
    """

    def __init__(self, dimension: int = 384):
        """Initialize an empty in-memory index.

        Args:
            dimension: Embedding dimension for new records
        """
        self._dimension = dimension
        self._records: dict[str, IndexRecord] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Args:
            records: List of IndexRecords to upsert

        Note:
            Idempotent: duplicate IndexRecord.id overwrites existing record.
        """
        for record in records:
            self._records[record.id] = record
            # Generate a mock vector based on text hash (deterministic but pseudo-random)
            self._vectors[record.id] = self._mock_vector(record.text)

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index.

        Args:
            ids: List of record IDs to delete

        Note:
            No-op for non-existent IDs.
        """
        for record_id in ids:
            self._records.pop(record_id, None)
            self._vectors.pop(record_id, None)

    def search(self, query: TypedQuery) -> list[SeedHit]:
        """Vector similarity search with filter support.

        Args:
            query: TypedQuery with account_id for filtering

        Returns:
            List of SeedHit sorted by score descending

        Note:
            Filters are applied to enforce tenant isolation at the index level.
            This is a TEST implementation - production should use proper vector DB.
        """
        # Generate query vector
        query_vector = self._mock_vector(query.text)

        # Calculate similarity for all matching records
        hits = []

        for record_id, record in self._records.items():
            # Apply filters for tenant isolation
            if not self._matches_filters(record, query):
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, self._vectors[record_id])

            # Check category filter
            if query.categories and record.metadata.get("category") not in query.categories:
                continue

            # Check context_type filter
            if query.context_type and record.metadata.get("context_type") != query.context_type:
                continue

            hits.append(
                SeedHit(
                    uri=record.uri,
                    score=similarity,
                    abstract=record.text[:200],  # L0 = abstract snippet
                    has_overview=record.metadata.get("has_overview", False),
                    has_content=record.metadata.get("has_content", False),
                )
            )

        # Sort by score descending and return top-k
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:query.top_k]

    def _matches_filters(self, record: IndexRecord, query: TypedQuery) -> bool:
        """Check if record matches query filters.

        Args:
            record: IndexRecord to check
            query: TypedQuery with filters

        Returns:
            True if record matches all filters
        """
        # CRITICAL: account_id filter is mandatory for tenant isolation
        record_account = record.filters.get("account_id")
        query_account = query.account_id

        if record_account != query_account:
            return False

        # owner_space filter (if specified in query)
        if query.owner_space is not None:
            record_owner = record.filters.get("owner_space")
            if record_owner != query.owner_space:
                return False

        return True

    def _mock_vector(self, text: str) -> list[float]:
        """Generate a mock embedding vector for testing.

        Uses a hash-based approach for deterministic but varied vectors.
        Real implementation should use an actual Embedder.

        Args:
            text: Text to embed

        Returns:
            Mock vector of self._dimension dimensions
        """
        import hashlib

        # Create a deterministic seed from text
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # Generate pseudo-random vector using simple LCG
        vector = []
        for i in range(self._dimension):
            # Linear congruential generator
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            # Map to [-1, 1] range
            value = (seed / 0x7fffffff) * 2 - 1
            vector.append(value)

        return vector

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score in [0, 1] range
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def get_record(self, record_id: str) -> IndexRecord | None:
        """Get a record by ID (for testing).

        Args:
            record_id: Record ID to fetch

        Returns:
            IndexRecord if found, None otherwise
        """
        return self._records.get(record_id)

    def count(self) -> int:
        """Get total record count (for testing).

        Returns:
            Number of records in the index
        """
        return len(self._records)

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._vectors.clear()
