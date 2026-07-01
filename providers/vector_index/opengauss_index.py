"""OpenGauss implementation of VectorIndex.

Supports efficient similarity search with HNSW indexing.
"""
import json
import logging
import re
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, SeedHit, TypedQuery

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    OPENGAUSS_AVAILABLE = True
except ImportError:
    OPENGAUSS_AVAILABLE = False
    psycopg2 = None
    Json = None
    RealDictCursor = None

logger = logging.getLogger(__name__)


# Valid PostgreSQL identifier: letters, digits, underscore, must start with letter/underscore
_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_table_name(name: str) -> str:
    """Validate table name to prevent SQL injection.

    Args:
        name: Table name to validate

    Returns:
        The validated table name

    Raises:
        ValueError: If table name contains invalid characters
    """
    if not _TABLE_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid table name '{name}'. "
            "Table names must be valid PostgreSQL identifiers: "
            "start with letter or underscore, contain only letters, digits, and underscores."
        )
    # Also protect against SQL keywords
    upper_name = name.upper()
    sql_keywords = {
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'UNION', 'OR', 'AND', 'WHERE', 'FROM'
    }
    if upper_name in sql_keywords:
        raise ValueError(f"Table name '{name}' is a reserved SQL keyword")
    return name


def _vec_literal(vec: list[float]) -> str:
    """Convert vector to pgvector literal string."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _ensure_dict(val: Any) -> dict:
    """Ensure value is a dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


class OpenGaussVectorIndex(VectorIndex):
    """OpenGauss implementation for production use.

    Features:
        - HNSW indexing for fast approximate nearest neighbor search
        - Multi-tenant isolation via account_id filtering
        - Idempotent upsert with ON CONFLICT
        - Cosine similarity search
    """

    def __init__(
        self,
        connection_string: str,
        dimension: int = 1536,
        table_name: str = "vector_index",
        pool_size: int = 5,
    ):
        """Initialize OpenGauss index.

        Args:
            connection_string: PostgreSQL connection string
            dimension: Embedding vector dimension (default 1536)
            table_name: Table name for vector storage (must be valid identifier)
            pool_size: Connection pool size

        Raises:
            ImportError: If psycopg2 is not installed
            ValueError: If table_name is invalid
        """
        if not OPENGAUSS_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for OpenGaussVectorIndex. "
                "Install with: pip install psycopg2-binary"
            )

        self._connection_string = connection_string
        self._dimension = dimension
        self._table_name = _validate_table_name(table_name)
        self._pool_size = pool_size
        self._pool: list = []
        self._pool_index = 0

        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the vector_index table and indexes if they don't exist."""
        tbl = self._table_name
        dim = self._dimension
        conn = psycopg2.connect(self._connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        id          VARCHAR(16) PRIMARY KEY,
                        uri         VARCHAR(512) NOT NULL,
                        level       INTEGER NOT NULL,
                        text        TEXT NOT NULL,
                        embedding   vector({dim}) NOT NULL,
                        filters     JSONB NOT NULL,
                        metadata    JSONB NOT NULL DEFAULT '{{}}',
                        created_at  TIMESTAMPTZ DEFAULT NOW(),
                        updated_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_account
                        ON {tbl} ((filters->>'account_id'))
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_level
                        ON {tbl} (level)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_filters_gin
                        ON {tbl} USING GIN (filters)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{tbl}_embedding_hnsw
                        ON {tbl} USING hnsw (embedding vector_cosine_ops)
                """)
                conn.commit()
            logger.info("Table '%s' ensured (dim=%d)", tbl, dim)
        except Exception as e:
            conn.rollback()
            logger.warning("Failed to ensure table '%s': %s", tbl, e)
        finally:
            conn.close()

    def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn

        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn):
        """Return a connection to the pool, rolling back dirty transactions."""
        if conn.closed != 0:
            return
        try:
            if conn.info.transaction_status != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                conn.rollback()
        except Exception:
            conn.close()
            return
        if len(self._pool) < self._pool_size:
            self._pool.append(conn)
        else:
            conn.close()

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Uses MERGE INTO for openGauss compatibility (ON CONFLICT not supported).
        Per-record errors are logged and skipped to avoid poisoning the connection.
        """
        if not records:
            return

        conn = self._get_connection()
        try:
            for record in records:
                embedding = record.metadata.get("_embedding")
                if embedding is None:
                    logger.warning("Skipping record %s: missing '_embedding'", record.id)
                    continue

                embedding_str = _vec_literal(embedding)
                filters_json = Json(record.filters)
                metadata_json = Json(record.metadata)

                try:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            MERGE INTO {self._table_name} t
                            USING (SELECT %s AS id) s
                            ON t.id = s.id
                            WHEN MATCHED THEN UPDATE SET
                                uri = %s,
                                level = %s,
                                text = %s,
                                embedding = %s::vector,
                                filters = %s,
                                metadata = %s,
                                updated_at = NOW()
                            WHEN NOT MATCHED THEN INSERT
                                (id, uri, level, text, embedding, filters, metadata)
                            VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                        """, (
                            record.id,
                            record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                            record.id, record.uri, record.level, record.text,
                            embedding_str, filters_json, metadata_json,
                        ))
                    conn.commit()
                except Exception as exc:
                    logger.warning(
                        "Upsert failed for record %s (uri=%s), rolling back: %s",
                        record.id, record.uri, exc,
                    )
                    conn.rollback()
        finally:
            self._return_connection(conn)

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index."""
        if not ids:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                conn.commit()
        finally:
            self._return_connection(conn)

    def search_by_vector(
        self,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Low-level vector search returning SeedHit."""
        where, params = self._build_where(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k

        sql = f"""
            SELECT id, uri, level, text, filters, metadata,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table_name}
            WHERE  {where}
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_hit(r) for r in rows]

    def search_children(
        self,
        parent_uri: str,
        query_vector: list[float],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[SeedHit]:
        """Search immediate children of parent_uri."""
        where, params = self._build_where(filters)
        params["qvec"] = _vec_literal(query_vector)
        params["topk"] = top_k
        params["parent_uri"] = parent_uri

        sql = f"""
            SELECT id, uri, level, text, filters, metadata,
                   1 - (embedding <=> %(qvec)s::vector) AS score
            FROM   {self._table_name}
            WHERE  {where} AND metadata->>'parent_uri' = %(parent_uri)s
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            self._return_connection(conn)

        return [self._row_to_hit(r) for r in rows]

    def _build_where(self, f: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build WHERE clause from filters."""
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if "account_id" in f:
            clauses.append("filters->>'account_id' = %(f_account_id)s")
            params["f_account_id"] = f["account_id"]
        if "owner_space" in f:
            owner = f["owner_space"]
            if isinstance(owner, list):
                clauses.append("filters->>'owner_space' = ANY(%(f_owner_space)s)")
            else:
                clauses.append("filters->>'owner_space' = %(f_owner_space)s")
            params["f_owner_space"] = owner
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

    @staticmethod
    def _row_to_hit(row: dict) -> SeedHit:
        """Convert database row to SeedHit."""
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

    def delete_account_data(self, account_id: str) -> int:
        """Delete all index records for an account.

        Args:
            account_id: Account ID to delete

        Returns:
            Count of deleted records
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE filters->>'account_id' = %(account_id)s",
                    {"account_id": account_id}
                )
                conn.commit()
                return cur.rowcount
        finally:
            self._return_connection(conn)

    def delete_by_owner_space(
        self, account_id: str, owner_space: str
    ) -> int:
        """Delete all records matching account_id + owner_space.

        Args:
            account_id: Account ID to filter
            owner_space: Owner space in colon format "user:{id}" or "agent:{id}"

        Returns:
            Count of deleted records
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} "
                    "WHERE filters->>'account_id' = %(account_id)s "
                    "  AND filters->>'owner_space' = %(owner_space)s",
                    {"account_id": account_id, "owner_space": owner_space}
                )
                conn.commit()
                return cur.rowcount
        finally:
            self._return_connection(conn)

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

    # Backward compatibility convenience method

    def search(self, query, embedder=None, top_k: int | None = None) -> list[SeedHit]:
        """Convenience method for backward compatibility with TypedQuery API."""
        if not isinstance(query, TypedQuery):
            raise TypeError(f"Expected TypedQuery, got {type(query)}")

        filters: dict[str, Any] = {}
        if query.account_id:
            filters["account_id"] = query.account_id
        if query.owner_space:
            filters["owner_space"] = query.owner_space
        if query.context_type:
            filters["context_type"] = query.context_type
        if query.categories:
            filters["category"] = query.categories

        if embedder:
            vectors = embedder.embed_texts([query.text])
            query_vector = vectors[0]
        else:
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
