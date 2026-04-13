"""ChromaDB-backed Vector Index with persistence.

Data survives server restarts. Uses ChromaDB's embedding-free mode - vectors
come from the Embedder, not from ChromaDB's built-in models.
"""

import hashlib
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
        embeddings = []
        for r in records:
            real_emb = (r.metadata or {}).get("_embedding")
            if real_emb and isinstance(real_emb, list):
                embeddings.append(real_emb)
            else:
                embeddings.append(self._stable_vector(r.text))
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

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _record_to_meta(record: IndexRecord) -> dict[str, Any]:
        meta = {
            "uri": record.uri,
            "level": record.level,
            "account_id": record.filters.get("account_id", ""),
            "owner_space": record.filters.get("owner_space", ""),
        }
        for key in ("context_type", "category", "parent_uri",
                     "has_overview", "has_content", "active_count", "updated_at"):
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

    def _stable_vector(self, text: str) -> list[float]:
        import numpy as np
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(seed % (2**31))
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def count(self) -> int:
        return self._collection.count()
