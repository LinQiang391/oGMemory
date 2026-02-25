"""内存引擎 - 核心 API"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..storage.vector_db import OpenGaussVectorDB, VectorRecord
from .document_processor import DocumentProcessor, TextChunk
from ..embeddings.base import EmbeddingBase, EmbeddingFactory

# Ensure built-in embedding providers are registered before factory is used
from ..embeddings import openai  # noqa: F401


@dataclass
class SearchResult:
    """搜索结果"""

    text: str
    score: float
    metadata: Dict[str, Any]
    source: str
    id: str = ""  # chunk_id，用于 expand 命令


class MemoryEngine:
    """内存引擎 - 主要API接口

    提供文档索引、语义搜索等功能
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "memory_db",
        db_user: str = "postgres",
        db_password: str = "",
        db_sslmode: Optional[str] = None,
        db_gssencmode: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_index_type: str = "ivfflat",
        index_lists: int = 100,
        index_m: int = 16,
        index_ef_construction: int = 64,
    ):
        # 环境变量优先覆盖数据库与嵌入配置（保持 CLI / hooks 行为一致）
        db_host = os.getenv("OG_DB_HOST", db_host)
        db_port = int(os.environ.get("OG_DB_PORT", str(db_port)))
        db_name = os.getenv("OG_DB_NAME", db_name)
        db_user = os.getenv("OG_DB_USER", db_user)
        db_password = os.getenv("OG_DB_PASSWORD", db_password)

        embedding_provider = os.getenv("OG_EMBEDDING_PROVIDER", embedding_provider)
        embedding_model = (
            os.getenv("OG_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or embedding_model
        )

        self.db = OpenGaussVectorDB(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            sslmode=db_sslmode,
            gssencmode=db_gssencmode,
            dimension=self._get_dimension(embedding_provider, embedding_model),
            index_type=vector_index_type,
            lists=index_lists,
            m=index_m,
            ef_construction=index_ef_construction,
        )

        self.embedder = EmbeddingFactory.create(
            embedding_provider, model=embedding_model
        )

        self.processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        self.logger = logging.getLogger(__name__)

    def _get_dimension(self, provider: str, model: str) -> int:
        """获取嵌入维度"""
        embedder = EmbeddingFactory.create(provider, model=model)
        return embedder.get_dimension()

    async def index_file(self, file_path: str) -> int:
        """索引单个文件

        Args:
            file_path: 文件路径

        Returns:
            索引的块数量
        """
        self.logger.info(f"Indexing file: {file_path}")

        text = self.processor.load_document(file_path)
        chunks = self.processor.process(text, source=file_path)

        if not chunks:
            self.logger.warning(f"No chunks generated from: {file_path}")
            return 0

        vectors = await self.embedder.encode([c.content for c in chunks])

        records = []
        for chunk, vector in zip(chunks, vectors):
            metadata = {"source": chunk.source, **chunk.metadata}
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=vector,
                    metadata=metadata,
                    text=chunk.content,
                )
            )

        count = self.db.insert(records)
        self.logger.info(f"Indexed {count} chunks from {file_path}")

        return count

    async def index_directory(self, directory: str) -> int:
        """索引目录中的所有Markdown文件

        Args:
            directory: 目录路径

        Returns:
            索引的块总数
        """
        self.logger.info(f"Indexing directory: {directory}")

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        total_count = 0
        md_files = list(path.glob("**/*.md")) + list(path.glob("**/*.markdown"))

        for md_file in md_files:
            try:
                count = await self.index_file(str(md_file))
                total_count += count
            except Exception as e:
                self.logger.error(f"Failed to index {md_file}: {e}")

        self.logger.info(f"Total indexed chunks: {total_count}")
        return total_count

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[SearchResult]:
        """语义搜索

        Args:
            query: 搜索查询文本
            limit: 返回结果数量限制
            filters: 元数据过滤条件
            use_hybrid: 是否使用混合搜索（向量+BM25）

        Returns:
            搜索结果列表
        """
        self.logger.info(f"Searching: {query}")

        query_vector = await self.embedder.encode_single(query)
        results = self.db.search(
            query_vector,
            query_text=query,
            limit=limit,
            filters=filters,
            use_hybrid=use_hybrid,
        )

        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    text=result["text"],
                    score=result["score"],
                    metadata=result["metadata"],
                    source=result["metadata"].get("source", ""),
                    id=result.get("id", ""),
                )
            )

        return search_results

    async def add_memory(self, text: str, metadata: Optional[Dict] = None) -> str:
        """添加单条记忆

        Args:
                       text: 记忆文本
            metadata: 元数据

        Returns:
            记录ID
        """
        if metadata is None:
            metadata = {}

        vector = await self.embedder.encode_single(text)
        record_id = str(uuid.uuid4())

        record = VectorRecord(
            id=record_id,
            vector=vector,
            metadata=metadata,
            text=text,
        )

        self.db.insert([record])
        return record_id

    def delete_by_source(self, source: str) -> int:
        """删除指定源的所有记录

        Args:
            source: 源文件路径

        Returns:
            删除的记录数
        """
        self.logger.info(f"Deleting records from source: {source}")

        conn = self.db._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT id FROM {self.db.table_name} 
            WHERE metadata->>'source' = %s
        """,
            (source,),
        )

        ids = [row[0] for row in cursor.fetchall()]
        cursor.close()

        if ids:
            return self.db.delete(ids)
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_records": self.db.count(),
            "embedding_model": self.embedder.get_model_name(),
            "embedding_dimension": self.embedder.get_dimension(),
        }

    def clear_all(self):
        """清空所有数据"""
        self.logger.warning("Clearing all data")
        self.db.clear()

    def close(self):
        """关闭连接"""
        self.db.close()
        self.logger.info("Memory engine closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
