"""In-Memory Vector Index for testing.

Simple in-memory implementation of VectorIndex protocol.
Supports filter matching for cross-tenant isolation testing.

Also implements ``search_by_vector`` / ``search_children``
for the retrieval pipeline.
"""

import math
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit


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

    # -- Retrieval-chain extensions ----------------------------------------

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        hits: list[SeedHit] = []
        for record_id, record in self._records.items():
            if not self._matches_raw_filters(record, filters):
                continue
            similarity = self._cosine_similarity(query_vector, self._vectors[record_id])
            hits.append(self._record_to_vector_hit(record, similarity))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        hits: list[SeedHit] = []
        for record_id, record in self._records.items():
            if not record.uri.startswith(parent_uri + "/"):
                continue
            depth = record.uri[len(parent_uri):].count("/")
            if depth != 1:
                continue
            if not self._matches_raw_filters(record, filters):
                continue
            similarity = self._cosine_similarity(query_vector, self._vectors[record_id])
            hits.append(self._record_to_vector_hit(record, similarity))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _matches_raw_filters(self, record: IndexRecord, filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if key == "level":
                if isinstance(expected, list):
                    if record.level not in expected:
                        return False
                elif record.level != expected:
                    return False
            elif key in ("account_id", "owner_space"):
                actual = record.filters.get(key)
                if actual != expected:
                    return False
            elif key in ("context_type", "category"):
                actual = record.metadata.get(key)
                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
        return True

    @staticmethod
    def _record_to_vector_hit(record: IndexRecord, score: float) -> SeedHit:
        meta = record.metadata or {}
        return SeedHit(
            uri=record.uri,
            score=score,
            level=record.level,
            parent_uri=meta.get("parent_uri"),
            context_type=meta.get("context_type", ""),
            category=meta.get("category", ""),
            owner_space=record.filters.get("owner_space", ""),
            abstract=record.text[:200],
            has_overview=meta.get("has_overview", False),
            has_content=meta.get("has_content", False),
            active_count=meta.get("active_count", 0),
            updated_at=meta.get("updated_at"),
        )

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
