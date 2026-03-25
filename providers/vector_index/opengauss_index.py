"""openGauss Vector Index — production VectorIndex backed by openGauss.

Table schema expected:

    CREATE TABLE IF NOT EXISTS vector_index (
        id          VARCHAR(16) PRIMARY KEY,
        uri         VARCHAR(512) NOT NULL,
        level       INTEGER NOT NULL,
        text        TEXT NOT NULL,
        embedding   vector(1536) NOT NULL,
        filters     JSONB NOT NULL,
        metadata    JSONB NOT NULL DEFAULT '{}',
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );

    -- HNSW for ANN
    CREATE INDEX idx_vector_embedding
        ON vector_index USING hnsw (embedding vector_cosine_ops);

    -- Expression indexes on JSONB fields
    CREATE INDEX idx_vector_account     ON vector_index ((filters->>'account_id'));
    CREATE INDEX idx_vector_owner_space ON vector_index ((filters->>'owner_space'));
    CREATE INDEX idx_vector_uri         ON vector_index (uri);
    CREATE INDEX idx_vector_level       ON vector_index (level);

    -- GIN for metadata
    CREATE INDEX idx_vector_metadata    ON vector_index USING GIN (metadata);

Requires:
    pip install psycopg2-binary   (or psycopg2 with libpq)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras

from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit

logger = logging.getLogger(__name__)

_TABLE = "vector_index"


class OpenGaussVectorIndex(VectorIndex):
    """VectorIndex implementation backed by openGauss with pgvector extension."""

    def __init__(
        self,
        dsn: str,
        table: str = _TABLE,
        dimension: int = 1536,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dimension = dimension

    @contextmanager
    def _conn(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        sql = f"""
            INSERT INTO {self._table}
                (id, uri, level, text, embedding, filters, metadata, updated_at)
            VALUES
                (%(id)s, %(uri)s, %(level)s, %(text)s,
                 %(embedding)s::vector, %(filters)s::jsonb, %(metadata)s::jsonb, NOW())
            ON CONFLICT (id) DO UPDATE SET
                uri        = EXCLUDED.uri,
                level      = EXCLUDED.level,
                text       = EXCLUDED.text,
                embedding  = EXCLUDED.embedding,
                filters    = EXCLUDED.filters,
                metadata   = EXCLUDED.metadata,
                updated_at = NOW()
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                for record in records:
                    embedding = record.metadata.get("_embedding")
                    if embedding is None:
                        raise ValueError(
                            f"Missing '_embedding' in metadata for record {record.id}"
                        )

                    embedding_str = _vec_literal(embedding)

                    cur.execute(f"""
                        INSERT INTO {self._table_name}
                            (id, uri, level, text, embedding, filters, metadata)
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            uri = VALUES(uri),
                            level = VALUES(level),
                            text = VALUES(text),
                            embedding = VALUES(embedding),
                            filters = VALUES(filters),
                            metadata = VALUES(metadata),
                            updated_at = NOW()
                    """, (
                        record.id,
                        record.uri,
                        record.level,
                        record.text,
                        embedding_str,
                        Json(record.filters),
                        Json(record.metadata),
                    ))
                conn.commit()
        finally:
            self._return_connection(conn)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE id = ANY(%s)", (ids,),
                )

    # ------------------------------------------------------------------
    # Retrieval-chain extensions
    # ------------------------------------------------------------------

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        where, params = self._where_filters(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k

        sql = f"""
            SELECT id, uri, level, text, filters, metadata,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table}
            WHERE  {where}
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_hit(r) for r in rows]

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        where, params = self._where_filters(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k
        params["parent_uri"] = parent_uri

        sql = f"""
            SELECT id, uri, level, text, filters, metadata,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table}
            WHERE  {where} AND metadata->>'parent_uri' = %(parent_uri)s
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_hit(r) for r in rows]

    # ------------------------------------------------------------------
    # WHERE clause builders
    # ------------------------------------------------------------------

    def _where_filters(self, f: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build WHERE from raw filter dict.

        Routing:
          account_id / owner_space → filters->>  (expression index)
          context_type / category  → metadata @>  (GIN index)
          level                    → level column (B-tree index)
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if "account_id" in f:
            clauses.append("filters->>'account_id' = %(f_account_id)s")
            params["f_account_id"] = f["account_id"]
        if "owner_space" in f:
            clauses.append("filters->>'owner_space' = %(f_owner_space)s")
            params["f_owner_space"] = f["owner_space"]
        if "context_type" in f:
            ct = f["context_type"]
            if isinstance(ct, list):
                clauses.append("metadata->>'context_type' = ANY(%(f_ctx_type)s)")
            else:
                clauses.append("metadata @> %(f_ctx_type)s::jsonb")
                ct = json.dumps({"context_type": ct})
            params["f_ctx_type"] = ct
        if "category" in f:
            cat = f["category"]
            if isinstance(cat, list):
                clauses.append("metadata->>'category' = ANY(%(f_category)s)")
            else:
                clauses.append("metadata @> %(f_category)s::jsonb")
                cat = json.dumps({"category": cat})
            params["f_category"] = cat
        if "level" in f:
            lvl = f["level"]
            if isinstance(lvl, list):
                clauses.append("level = ANY(%(f_level)s)")
            else:
                clauses.append("level = %(f_level)s")
            params["f_level"] = lvl

        return (" AND ".join(clauses) or "TRUE"), params

    # ------------------------------------------------------------------
    # Row → SeedHit
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_hit(row: dict) -> SeedHit:
        meta = _ensure_dict(row.get("metadata"))
        filt = _ensure_dict(row.get("filters"))
        level = int(row.get("level", 2))
        return SeedHit(
            uri=row["uri"],
            score=float(row.get("score", 0)),
            level=level,
            parent_uri=meta.get("parent_uri"),
            context_type=meta.get("context_type", ""),
            category=meta.get("category", ""),
            owner_space=filt.get("owner_space", ""),
            abstract=row.get("text", "")[:200],
            has_overview=meta.get("has_overview", False),
            has_content=meta.get("has_content", False),
            active_count=meta.get("active_count", 0),
            updated_at=meta.get("updated_at"),
        )

    def close(self) -> None:
        """Close all connections in the pool."""
        for conn in self._pool:
            if conn.closed == 0:
                conn.close()
        self._pool.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Backward compatibility convenience method
    # ------------------------------------------------------------------

    def search(self, query, embedder=None, top_k: int | None = None) -> list[SeedHit]:
        """Convenience method for backward compatibility with TypedQuery API.

        Converts TypedQuery to search_by_vector() call.
        This maintains compatibility with older tests while using the new API.

        Args:
            query: TypedQuery object with text, account_id, owner_space, etc.
            embedder: Optional Embedder to convert query.text to vector.
                     If not provided, uses a simple hash-based mock vector.
            top_k: Override for top_k (defaults to query.top_k)

        Returns:
            List of SeedHit results
        """
        from core.models import TypedQuery

        if not isinstance(query, TypedQuery):
            raise TypeError(f"Expected TypedQuery, got {type(query)}")

        # Build filters from TypedQuery
        filters: dict[str, Any] = {}
        if query.account_id:
            filters["account_id"] = query.account_id
        if query.owner_space:
            filters["owner_space"] = query.owner_space
        if query.context_type:
            filters["context_type"] = query.context_type
        if query.categories:
            filters["category"] = query.categories

        # Generate query vector
        if embedder:
            vectors = embedder.embed_texts([query.text])
            query_vector = vectors[0]
        else:
            # Use simple hash-based mock vector for testing
            import hashlib
            seed = int(hashlib.md5(query.text.encode()).hexdigest(), 16)
            query_vector = []
            for i in range(self._dimension):
                seed = (1103515245 * seed + 12345) & 0x7fffffff
                value = (seed / 0x7fffffff) * 2 - 1
                query_vector.append(value)

        return self.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=top_k or query.top_k,
        )

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _ensure_dict(val: Any) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
