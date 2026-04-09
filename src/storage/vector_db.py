"""openGauss 向量存储后端"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from psycopg.errors import UniqueViolation
except ImportError:
    UniqueViolation = None  # type: ignore[misc, assignment]


@dataclass
class VectorRecord:
    """向量记录数据结构"""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    text: str


class VectorDatabase:
    """openGauss 向量数据库接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connection = None

    def connect(self):
        """建立数据库连接"""
        raise NotImplementedError

    def insert(self, records: List[VectorRecord]) -> int:
        """插入向量记录"""
        raise NotImplementedError

    def search(
        self,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        raise NotImplementedError

    def delete(self, record_ids: List[str]) -> int:
        """删除记录"""
        raise NotImplementedError

    def close(self):
        """关闭数据库连接"""
        raise NotImplementedError


class OpenGaussVectorDB(VectorDatabase):
    """openGauss 向量数据库实现"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "memory_db",
        user: str = "postgres",
        password: str = "",
        table_name: str = "vectors",
        dimension: int = 1536,
        index_type: str = "ivfflat",
        lists: int = 100,
        m: int = 16,
        ef_construction: int = 64,
        sslmode: Optional[str] = None,
        gssencmode: Optional[str] = None,
        bm25_parallel_workers: Optional[int] = None,
    ):
        self.config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "sslmode": sslmode,
            "gssencmode": gssencmode,
        }
        self.table_name = table_name
        self.dimension = dimension
        self.index_type = index_type
        self.lists = lists
        self.m = m
        self.ef_construction = ef_construction
        # BM25 并行构建线程数 1~32，不设置则单线程。见 openGauss BM25 使用指南
        self.bm25_parallel_workers = (
            max(1, min(32, bm25_parallel_workers))
            if bm25_parallel_workers is not None
            else None
        )
        self._initialize_schema()

    def connect(self):
        """建立 openGauss 连接（支持 GaussDB/openGauss 常用参数 sslmode、gssencmode）"""
        try:
            import psycopg

            # 使用 conninfo 字符串确保 gssencmode 等 openGauss 参数正确传给 libpq
            parts = [
                f"host={self.config['host']}",
                f"port={self.config['port']}",
                f"dbname={self.config['database']}",
                f"user={self.config['user']}",
                f"password={self.config['password']}",
            ]
            if self.config.get("sslmode") is not None:
                parts.append(f"sslmode={self.config['sslmode']}")
            if self.config.get("gssencmode") is not None:
                parts.append(f"gssencmode={self.config['gssencmode']}")
            conninfo = " ".join(parts)
            self._connection = psycopg.connect(conninfo=conninfo)
            logger.info("Connected to openGauss")
            return self._connection
        except ImportError:
            raise ImportError("Please install: pip install psycopg")

    def _initialize_schema(self):
        """初始化数据库表结构；若表已存在但向量维度不一致则 DROP 后重建"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT EXISTS (
                SELECT 1 FROM pg_tables
                WHERE tablename = '{self.table_name}'
            )
        """
        )
        exists = cursor.fetchone()[0]

        if exists:
            # 检查向量列维度是否与当前一致
            cursor.execute(
                """
                SELECT a.atttypmod FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = %s AND a.attname = 'vector' AND NOT a.attisdropped
                """,
                (self.table_name,),
            )
            row = cursor.fetchone()
            if row and row[0] is not None and row[0] != self.dimension:
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                conn.commit()
                exists = False
            else:
                cursor.close()
                return

        if not exists:
            self._create_table(cursor)
            conn.commit()
            logger.info(f"Created table: {self.table_name}")

        cursor.close()

    def _create_table(self, cursor):
        """创建向量表和索引（openGauss 7+ 使用内置 vector 类型，无需单独 extension）"""
        cursor.execute(
            f"""
            CREATE TABLE {self.table_name} (
                id VARCHAR(64) PRIMARY KEY,
                vector vector({self.dimension}),
                text_content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        if self.index_type == "ivfflat":
            cursor.execute(
                f"""
                CREATE INDEX idx_{self.table_name}_vector 
                ON {self.table_name} USING ivfflat (vector vector_cosine_ops)
                WITH (lists = {self.lists})
            """
            )
        elif self.index_type == "hnsw":
            cursor.execute(
                f"""
                CREATE INDEX idx_{self.table_name}_vector 
                ON {self.table_name} USING hnsw (vector vector_cosine_ops)
                WITH (m = {self.m}, ef_construction = {self.ef_construction})
            """
            )
        elif self.index_type == "diskann":
            cursor.execute(
                f"""
                CREATE INDEX idx_{self.table_name}_vector 
                ON {self.table_name} USING diskann (vector vector_cosine_ops)
                WITH (index_size = 50)
            """
            )

        # BM25 索引：可选并行构建（1~32），见 https://docs.opengauss.org/zh/docs/latest/datavec/bm25_usage_guide.html
        if self.bm25_parallel_workers is not None:
            cursor.execute(
                f"ALTER TABLE {self.table_name} SET(parallel_workers={self.bm25_parallel_workers});"
            )
        cursor.execute(
            f"""
            CREATE INDEX idx_{self.table_name}_bm25
            ON {self.table_name} USING bm25(text_content)
        """
        )

    def _get_conn(self):
        """获取或创建连接"""
        if self._is_connection_closed():
            self.connect()
        return self._connection

    def _is_connection_closed(self) -> bool:
        """检查连接是否关闭"""
        try:
            return self._connection is None or getattr(
                self._connection, "closed", False
            )
        except (AttributeError, TypeError):
            return True

    def insert(self, records: List[VectorRecord]) -> int:
        """批量插入向量记录"""
        if not records:
            return 0

        conn = self._get_conn()
        cursor = conn.cursor()

        count = 0
        for record in records:
            meta_json = json.dumps(record.metadata)
            try:
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, vector, text_content, metadata)
                    VALUES (%s, %s::vector, %s, %s)
                    """,
                    (record.id, record.vector, record.text, meta_json),
                )
            except Exception as e:
                err_msg = str(e).lower()
                if "unique" in err_msg or "duplicate" in err_msg or "26000" in err_msg:
                    conn.rollback()
                    cursor.execute(
                        f"""
                        UPDATE {self.table_name}
                        SET vector = %s::vector, text_content = %s, metadata = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (record.vector, record.text, meta_json, record.id),
                    )
                else:
                    raise
            count += 1

        conn.commit()
        cursor.close()
        return count

    def search(
        self,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """混合搜索：向量相似度 + BM25全文检索"""
        conn = self._get_conn()
        cursor = conn.cursor()

        if use_hybrid and query_text:
            results = self._hybrid_search(
                cursor, query_vector, query_text, limit, filters
            )
        else:
            results = self._vector_search(cursor, query_vector, limit, filters)

        cursor.close()
        return results

    def _vector_search(
        self,
        cursor,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """纯向量相似度搜索"""
        query = f"""
            SELECT id, text_content, metadata,
                   1 - (vector <=> %s::vector) as similarity
            FROM {self.table_name}
        """

        params: List[Any] = [query_vector]

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY vector <=> %s::vector LIMIT %s"
        params.extend([query_vector, limit])

        cursor.execute(query, params)

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {"id": row[0], "text": row[1], "metadata": row[2], "score": row[3]}
            )

        return results

    def _hybrid_search(
        self,
        cursor,
        query_vector: List[float],
        query_text: str,
        limit: int,
        filters: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """混合搜索：结合向量和BM25"""

        vector_limit = limit * 2
        bm25_limit = limit * 2

        vector_results = self._vector_search(
            cursor, query_vector, vector_limit, filters
        )

        bm25_results = self._bm25_search(cursor, query_text, bm25_limit, filters)

        combined_results = self._rrf_rerank(vector_results, bm25_results, limit)

        return combined_results

    def _bm25_search(
        self,
        cursor,
        query_text: str,
        limit: int,
        filters: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """BM25全文检索（参照 openGauss BM25 使用指南：indexscan 提示 + <&> 操作符 + ORDER BY DESC）"""
        # 使用 indexscan 提示确保走 BM25 索引，否则引用 score 会报 WrongObjectType
        # 见 https://docs.opengauss.org/zh/docs/latest/datavec/bm25_usage_guide.html
        bm25_index_name = f"idx_{self.table_name}_bm25"
        query = f"""
            SELECT /*+ indexscan ({self.table_name} {bm25_index_name}) */
                   id, text_content, metadata,
                   text_content <&> %s AS score
            FROM {self.table_name}
        """

        params: List[Any] = [query_text]

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY text_content <&> %s DESC LIMIT %s"
        params.extend([query_text, limit])

        cursor.execute(query, params)

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {"id": row[0], "text": row[1], "metadata": row[2], "score": row[3]}
            )

        return results

    def _rrf_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 重排序"""

        k = 60
        scores: Dict[str, float] = {}

        for i, result in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k + i)
            doc_id = result["id"]
            if doc_id in scores:
                scores[doc_id] += rrf_score
            else:
                scores[doc_id] = rrf_score

        for i, result in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k + i)
            doc_id = result["id"]
            if doc_id in scores:
                scores[doc_id] += rrf_score
            else:
                scores[doc_id] = rrf_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        seen_ids = set()

        # 合并两份结果以便按 doc_id 取 text/metadata（优先 vector，否则 bm25）
        by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in vector_results}
        for r in bm25_results:
            if r["id"] not in by_id:
                by_id[r["id"]] = r

        for doc_id, rrf_score in sorted_results:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            info = by_id.get(doc_id)
            if info is None:
                continue
            final_results.append(
                {
                    "id": doc_id,
                    "text": info.get("text", ""),
                    "metadata": info.get("metadata", {}),
                    "score": rrf_score,
                }
            )
            if len(final_results) >= limit:
                break

        return final_results

    def delete(self, record_ids: List[str]) -> int:
        """删除记录"""
        if not record_ids:
            return 0

        conn = self._get_conn()
        cursor = conn.cursor()

        placeholders = ",".join(["%s"] * len(record_ids))
        cursor.execute(
            f"""
            DELETE FROM {self.table_name} 
            WHERE id IN ({placeholders})
        """,
            record_ids,
        )

        count = cursor.rowcount
        conn.commit()
        cursor.close()
        return count

    def get_by_id(self, record_id: str) -> Optional[Dict]:
        """根据ID获取记录"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT id, vector, text_content, metadata 
            FROM {self.table_name} 
            WHERE id = %s
        """,
            (record_id,),
        )

        row = cursor.fetchone()
        cursor.close()

        if row:
            return {"id": row[0], "vector": row[1], "text": row[2], "metadata": row[3]}
        return None

    def count(self) -> int:
        """获取记录总数"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]

        cursor.close()
        return count

    def clear(self):
        """清空所有数据（TRUNCATE，不 DROP，避免容器压力过大）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"TRUNCATE TABLE {self.table_name}")
        conn.commit()
        cursor.close()

    def close(self):
        """关闭数据库连接"""
        if self._connection and not self._is_connection_closed():
            self._connection.close()
            logger.info("Database connection closed")
