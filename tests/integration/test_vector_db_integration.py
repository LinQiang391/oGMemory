"""向量存储集成测试"""

import os
import pytest
from storage.vector_db import OpenGaussVectorDB, VectorRecord


@pytest.mark.skipif(
    not os.environ.get("TEST_DB_PASSWORD"), reason="Test database not configured"
)
class TestOpenGaussVectorDBIntegration:
    """openGauss 向量存储集成测试"""

    @pytest.fixture
    def vector_db(self, test_config):
        """向量数据库 fixture"""
        db = OpenGaussVectorDB(
            host=test_config["db_host"],
            port=test_config["db_port"],
            database=test_config["db_name"],
            user=test_config["db_user"],
            password=test_config["db_password"],
            sslmode=test_config.get("db_sslmode"),
            gssencmode=test_config.get("db_gssencmode"),
            table_name="test_integration_vectors",
            dimension=1536,
        )
        yield db
        db.clear()

    def test_insert_and_search(self, vector_db):
        """测试插入和搜索完整流程"""
        records = [
            VectorRecord(
                id=f"test-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"index": i, "tag": f"tag-{i % 3}"},
                text=f"content {i}",
            )
            for i in range(20)
        ]

        vector_db.insert(records)

        query_vector = [0.5] * 1536
        results = vector_db.search(query_vector=query_vector, limit=5, use_hybrid=False)

        assert len(results) <= 5
        assert all("score" in r for r in results)

    def test_hybrid_search_workflow(self, vector_db, sample_markdown):
        """测试混合搜索工作流"""
        records = [
            VectorRecord(
                id=f"hybrid-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"section": f"section-{i % 2}"},
                text=f"This is content about section-{i % 2} with keyword Python.",
            )
            for i in range(15)
        ]

        vector_db.insert(records)

        results = vector_db.search(
            query_vector=[0.001] * 1536, query_text="Python", limit=5, use_hybrid=True
        )

        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_bm25_search(self, vector_db):
        """测试 BM25 全文搜索"""
        records = [
            VectorRecord(
                id=f"bm25-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"source": "test"},
                text=f"This document mentions Python programming language and related concepts.",
            )
            for i in range(10)
        ]

        vector_db.insert(records)

        results = vector_db._bm25_search(
            vector_db._get_conn().cursor(), "Python programming", limit=5, filters=None
        )

        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_rrf_reranking(self, vector_db):
        """测试 RRF 重排序"""
        vector_results = [
            {"id": f"v-{i}", "score": 0.9 - i * 0.05, "text": f"vector {i}"}
            for i in range(10)
        ]

        bm25_results = [
            {"id": f"b-{i}", "score": 0.8 - i * 0.05, "text": f"bm25 {i}"}
            for i in range(10)
        ]

        fused = vector_db._rrf_rerank(vector_results, bm25_results, limit=5)

        assert len(fused) <= 5
        assert all("score" in r for r in fused)

    def test_full_workflow(self, vector_db, sample_markdown):
        """测试完整工作流：插入→搜索→删除"""
        records = [
            VectorRecord(
                id=f"workflow-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"step": f"step-{i}"},
                text=f"Workflow step {i} content",
            )
            for i in range(5)
        ]

        vector_db.insert(records)

        results = vector_db.search(query_vector=[0.5] * 1536, limit=3, use_hybrid=False)

        assert len(results) > 0

        deleted_count = vector_db.delete([r["id"] for r in results])

        assert deleted_count == len(results)
        assert vector_db.count() == 2
