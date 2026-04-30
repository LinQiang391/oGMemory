"""ChromaDB-backed Vector Index with persistence.

Data survives server restarts. Uses ChromaDB's embedding-free mode - vectors
come from the Embedder, not from ChromaDB's built-in models.

VECTORIZATION_PLAN §2 / §3 extensions:
  * ``get_metadata_by_ids`` returns the ``metadata`` dict Chroma stored
    for a batch of IDs so OutboxWorker can skip re-embedding when
    ``text_hash`` is unchanged.
  * The historical md5-based vector fallback in ``upsert`` is removed;
    callers that upsert without a real ``_embedding`` now get a clear
    ``ValueError`` instead of silently indexing against a mock vector
    (CLAUDE.md §工程原则 "禁止 Fallback 路径").
"""

import logging
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit

logger = logging.getLogger(__name__)


class ChromaVectorIndex(VectorIndex):
    """Persistent vector index backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "contextengine",
        persist_directory: str = ".chroma_data",
        dimension: int = 1536,
    ):
        import chromadb

        self._dimension = dimension
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorIndex ready: collection=%s, dir=%s, existing=%d records",
            collection_name, persist_directory, self._collection.count(),
        )

    def upsert(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        documents = [r.text for r in records]
        metadatas = [self._record_to_meta(r) for r in records]
        embeddings: list[list[float]] = []
        for r in records:
            real_emb = (r.metadata or {}).get("_embedding")
            if not (real_emb and isinstance(real_emb, list)):
                # Explicit error instead of silently indexing a mock vector.
                # The caller MUST supply an embedding — the old fallback
                # path would have polluted the index with garbage vectors
                # that happened to pass shape validation but returned
                # meaningless similarity scores at query time.
                raise ValueError(
                    f"ChromaVectorIndex.upsert: record {r.id} ({r.uri}) has no "
                    "'_embedding' in metadata. Real embeddings are required; "
                    "the md5 fallback has been removed (VECTORIZATION_PLAN §2)."
                )
            embeddings.append(real_emb)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        chroma_where = self._build_where(filters)
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            where=chroma_where if chroma_where else None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SeedHit] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""
            dist = results["distances"][0][i] if results["distances"] else 1.0
            score = max(0.0, 1.0 - dist)
            hits.append(SeedHit(
                uri=meta.get("uri", ""),
                score=score,
                level=int(meta.get("level", 2)),
                parent_uri=meta.get("parent_uri"),
                context_type=meta.get("context_type", ""),
                category=meta.get("category", ""),
                owner_space=meta.get("owner_space", ""),
                abstract=doc[:200] if doc else "",
                has_overview=bool(meta.get("has_overview")),
                has_content=bool(meta.get("has_content")),
                active_count=int(meta.get("active_count", 0)),
                updated_at=meta.get("updated_at"),
                metadata={},
            ))
        return hits

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        enhanced = dict(filters)
        enhanced["parent_uri"] = parent_uri
        return self.search_by_vector(query_vector, enhanced, top_k)

    def get_metadata_by_ids(
        self, ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Batched metadata fetch for the fingerprint-skip path.

        Uses a single ``collection.get(ids=...)`` round-trip — Chroma's API
        treats ``ids`` as a primary-key lookup so the cost is O(len(ids))
        in the driver.
        """
        if not ids:
            return {}
        try:
            got = self._collection.get(ids=ids, include=["metadatas"])
        except Exception as exc:
            logger.error(
                "ChromaVectorIndex.get_metadata_by_ids FAILED: %s",
                exc, exc_info=True,
            )
            raise
        result: dict[str, dict[str, Any]] = {}
        returned_ids = got.get("ids") or []
        returned_meta = got.get("metadatas") or []
        for rid, meta in zip(returned_ids, returned_meta):
            if rid is None:
                continue
            result[rid] = dict(meta or {})
        return result

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _record_to_meta(record: IndexRecord) -> dict[str, Any]:
        meta = {
            "uri": record.uri,
            "level": record.level,
            "account_id": record.filters.get("account_id", ""),
            "owner_space": record.filters.get("owner_space", ""),
        }
        # Include the fingerprint-skip + embedding-model metadata so
        # get_metadata_by_ids can return them to OutboxWorker. These keys
        # are reserved by core.models.IndexRecord (see VECTORIZATION_PLAN §1).
        for key in (
            "context_type", "category", "parent_uri",
            "has_overview", "has_content", "active_count", "updated_at",
            "text_hash", "embedding_model", "embedding_dim",
        ):
            val = record.metadata.get(key) if record.metadata else None
            if val is not None:
                meta[key] = val
        sanitized = {}
        for k, v in meta.items():
            if isinstance(v, bool):
                sanitized[k] = v
            elif isinstance(v, (int, float, str)):
                sanitized[k] = v
            elif v is not None:
                sanitized[k] = str(v)
        return sanitized

    @staticmethod
    def _build_where(filters: dict[str, Any]) -> dict | None:
        conditions: list[dict] = []
        for key, expected in filters.items():
            if key == "level":
                if isinstance(expected, list):
                    conditions.append({"level": {"$in": expected}})
                else:
                    conditions.append({"level": expected})
            elif key in ("account_id", "owner_space"):
                if isinstance(expected, list):
                    conditions.append({key: {"$in": expected}})
                else:
                    conditions.append({key: expected})
            elif key in ("context_type", "category"):
                if isinstance(expected, list):
                    conditions.append({key: {"$in": expected}})
                else:
                    conditions.append({key: expected})
            elif key == "parent_uri":
                conditions.append({key: expected})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ``_stable_vector`` (md5-seeded random vector fallback) was removed;
    # see class docstring + ``upsert`` above. Re-introducing it is a
    # regression — the vector index MUST receive real embeddings.

    def count(self) -> int:
        return self._collection.count()