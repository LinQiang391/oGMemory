"""OpenGauss implementation of VectorIndex.

Supports efficient similarity search with HNSW indexing.
"""
import json
import logging
from typing import Any

from core.interfaces import VectorIndex
from core.models import IndexRecord, TypedQuery, SeedHit

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
    OPENGUASS_AVAILABLE = True
except ImportError:
    OPENGUASS_AVAILABLE = False
    psycopg2 = None
    Json = None
    execute_values = None

logger = logging.getLogger(__name__)


class OpenGaussIndex(VectorIndex):
    """ opengauss implementation for production use.

    Features:
        - HNSW indexing for fast approximate nearest neighbor search
        - Multi-tenant isolation via account_id filtering
        - Idempotent upsert with ON CONFLICT
        - Cosine similarity search
        - Auto-initialization of table schema (skips if already exists)

    Table schema (auto-created):
        CREATE TABLE vector_index (
            id VARCHAR(16) PRIMARY KEY,
            uri VARCHAR(512) NOT NULL,
            level INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding vector(DIMENSION) NOT NULL,
            filters JSONB NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """

    def __init__(
        self,
        connection_string: str,
        dimension: int = 1536,
        table_name: str = "vector_index",
        pool_size: int = 5,
    ):
        """Initialize opengauss index.

        Args:
            connection_string: opengauss connection string
                e.g., "postgres://user:pass@host:5432/dbname"
            dimension: Embedding vector dimension (default 1536 for ada-002)
            table_name: Table name for vector storage
            pool_size: Connection pool size

        Raises:
            ImportError: If psycopg2 is not installed
        """
        if not OPENGUASS_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for OpenGaussIndex. "
                "Install with: pip install psycopg2-binary"
            )

        self._connection_string = connection_string
        self._dimension = dimension
        self._table_name = table_name
        self._pool_size = pool_size
        self._pool: list = []
        self._pool_index = 0
        self._initialized = False

        self._ensure_table_exists()

    def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn

        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if len(self._pool) < self._pool_size and conn.closed == 0:
            self._pool.append(conn)
        else:
            conn.close()

    def _ensure_table_exists(self) -> None:
        """Ensure the vector table and indexes exist.

        Creates:
            - pgvector extension if not exists
            - vector_index table with proper schema (skips if exists)
            - HNSW index for vector similarity search
            - Filter indexes for account_id, owner_space, uri, level
            - GIN index for JSONB metadata queries
            - Auto-update trigger for updated_at column
        """
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (self._table_name,))
                table_exists = cur.fetchone()[0]

                if table_exists:
                    logger.info(f"Table {self._table_name} already exists, skipping creation")
                    self._initialized = True
                    conn.commit()
                    return

                cur.execute(f"""
                    CREATE TABLE {self._table_name} (
                        id VARCHAR(16) PRIMARY KEY,
                        uri VARCHAR(512) NOT NULL,
                        level INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self._dimension}) NOT NULL,
                        filters JSONB NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                logger.info(f"Created table {self._table_name} with dimension {self._dimension}")

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding 
                        ON {self._table_name} USING hnsw (embedding vector_cosine_ops);
                """)
                logger.info(f"Created HNSW index on {self._table_name}.embedding")

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_account 
                        ON {self._table_name} ((filters->>'account_id'));
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_owner_space 
                        ON {self._table_name} ((filters->>'owner_space'));
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_uri 
                        ON {self._table_name} (uri);
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_level 
                        ON {self._table_name} (level);
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_metadata 
                        ON {self._table_name} USING GIN (metadata);
                """)

                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_parent_uri 
                        ON {self._table_name} ((metadata->>'parent_uri'));
                """)
                logger.info(f"Created filter indexes on {self._table_name}")

                cur.execute(f"""
                    CREATE OR REPLACE FUNCTION update_{self._table_name}_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = NOW();
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """)

                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_trigger 
                            WHERE tgname = 'update_{self._table_name}_updated_at'
                        ) THEN
                            CREATE TRIGGER update_{self._table_name}_updated_at 
                                BEFORE UPDATE ON {self._table_name}
                                FOR EACH ROW
                                EXECUTE FUNCTION update_{self._table_name}_updated_at();
                        END IF;
                    END;
                    $$;
                """)
                logger.info(f"Created auto-update trigger for {self._table_name}.updated_at")

                conn.commit()
                self._initialized = True
                logger.info(f"OpenGaussIndex initialized successfully with table {self._table_name}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize table {self._table_name}: {e}")
            raise
        finally:
            self._return_connection(conn)

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Args:
            records: List of IndexRecords to upsert
                Each record must have '_embedding' in metadata

        Raises:
            ValueError: If embedding is missing from record metadata
        """
        if not records:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for record in records:
                    embedding = record.metadata.get("_embedding")
                    if embedding is None:
                        raise ValueError(
                            f"Missing '_embedding' in metadata for record {record.id}"
                        )

                    embedding_str = self._vec_literal(embedding)

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
        """Remove records from the index.

        Args:
            ids: List of record IDs to delete
        """
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
            FROM   {self._table_name}
            WHERE  {where}
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            return [self._row_to_hit(r) for r in rows]
        finally:
            self._return_connection(conn)

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
            FROM   {self._table_name}
            WHERE  {where} AND metadata->>'parent_uri' = %(parent_uri)s
            ORDER  BY embedding <=> %(qvec)s::vector
            LIMIT  %(topk)s
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            return [self._row_to_hit(r) for r in rows]
        finally:
            self._return_connection(conn)

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
