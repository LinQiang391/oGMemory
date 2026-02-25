"""向量存储单元测试"""

import os
import pytest
from storage.vector_db import OpenGaussVectorDB, VectorRecord


class TestVectorRecord:
    """向量记录数据类测试"""

    def test_create_record(self):
        """测试创建记录"""
        record = VectorRecord(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
            text="test content",
        )

        assert record.id == "test-1"
        assert len(record.vector) == 3
        assert record.metadata["source"] == "test"
        assert record.text == "test content"

    def test_create_record_empty_metadata(self):
        """测试空 metadata"""
        record = VectorRecord(
            id="id-1",
            vector=[0.0, 1.0],
            metadata={},
            text="text",
        )
        assert record.metadata == {}
        assert record.id == "id-1"

    def test_create_record_metadata_types(self):
        """测试 metadata 支持多种值类型"""
        record = VectorRecord(
            id="id-2",
            vector=[0.1] * 4,
            metadata={
                "str": "a",
                "int": 1,
                "float": 1.5,
                "bool": True,
                "list": [1, 2],
            },
            text="content",
        )
        assert record.metadata["str"] == "a"
        assert record.metadata["int"] == 1
        assert record.metadata["float"] == 1.5
        assert record.metadata["bool"] is True
        assert record.metadata["list"] == [1, 2]


@pytest.mark.skipif(
    not os.environ.get("TEST_DB_PASSWORD"), reason="Test database not configured"
)
class TestOpenGaussVectorDBUnit:
    """openGauss 向量存储单元测试"""

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
            table_name="test_unit_vectors",
            dimension=1536,
        )
        yield db
        db.clear()

    def test_connection(self, vector_db):
        """测试数据库连接"""
        assert vector_db._connection is not None
        assert not vector_db._is_connection_closed()

    def test_insert_single_record(self, vector_db):
        """测试插入单个记录"""
        record = VectorRecord(
            id="test-1",
            vector=[0.1] * 1536,
            metadata={"test": True},
            text="test content",
        )

        count = vector_db.insert([record])

        assert count == 1
        assert vector_db.count() == 1

    def test_insert_batch_records(self, vector_db):
        """测试批量插入"""
        records = [
            VectorRecord(
                id=f"test-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"index": i},
                text=f"content {i}",
            )
            for i in range(10)
        ]

        count = vector_db.insert(records)

        assert count == 10
        assert vector_db.count() == 10

    def test_insert_duplicate(self, vector_db):
        """测试插入重复记录（应该更新）"""
        record = VectorRecord(
            id="duplicate-1",
            vector=[0.1] * 1536,
            metadata={"version": 1},
            text="original content",
        )

        vector_db.insert([record])

        updated_record = VectorRecord(
            id="duplicate-1",
            vector=[0.2] * 1536,
            metadata={"version": 2},
            text="updated content",
        )

        vector_db.insert([updated_record])

        retrieved = vector_db.get_by_id("duplicate-1")
        assert retrieved["metadata"]["version"] == 2
        assert retrieved["text"] == "updated content"

    def test_get_by_id(self, vector_db):
        """测试根据ID获取记录"""
        record = VectorRecord(
            id="get-test-1", vector=[0.1] * 1536, metadata={}, text="test content"
        )

        vector_db.insert([record])
        retrieved = vector_db.get_by_id("get-test-1")

        assert retrieved is not None
        assert retrieved["id"] == "get-test-1"
        assert retrieved["text"] == "test content"

    def test_get_by_id_not_found(self, vector_db):
        """测试获取不存在的ID"""
        result = vector_db.get_by_id("non-existent-id")

        assert result is None

    def test_delete_records(self, vector_db):
        """测试删除记录"""
        records = [
            VectorRecord(
                id=f"delete-{i}",
                vector=[i * 0.001] * 1536,
                metadata={},
                text=f"content {i}",
            )
            for i in range(5)
        ]

        vector_db.insert(records)

        deleted_count = vector_db.delete(["delete-1", "delete-2"])

        assert deleted_count == 2
        assert vector_db.count() == 3

    def test_delete_empty_list(self, vector_db):
        """测试删除空列表"""
        deleted_count = vector_db.delete([])

        assert deleted_count == 0

    def test_clear_all(self, vector_db):
        """测试清空所有数据"""
        records = [
            VectorRecord(
                id=f"clear-{i}",
                vector=[i * 0.001] * 1536,
                metadata={},
                text=f"content {i}",
            )
            for i in range(5)
        ]

        vector_db.insert(records)
        assert vector_db.count() == 5

        vector_db.clear()

        assert vector_db.count() == 0

    def test_count_empty(self, vector_db):
        """测试空数据库计数"""
        assert vector_db.count() == 0

    def test_count_with_data(self, vector_db):
        """测试有数据时的计数"""
        records = [
            VectorRecord(
                id=f"count-{i}",
                vector=[i * 0.001] * 1536,
                metadata={},
                text=f"content {i}",
            )
            for i in range(3)
        ]

        vector_db.insert(records)

        assert vector_db.count() == 3

    def test_vector_search(self, vector_db):
        """测试向量搜索"""
        records = [
            VectorRecord(
                id=f"search-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"tag": f"tag-{i % 3}"},
                text=f"content about tag-{i % 3}",
            )
            for i in range(10)
        ]

        vector_db.insert(records)

        query_vector = [0.5] * 1536
        results = vector_db.search(query_vector=query_vector, limit=5, use_hybrid=False)

        assert len(results) <= 5
        assert all("score" in r for r in results)

    def test_search_with_filters(self, vector_db):
        """测试带过滤的搜索"""
        records = [
            VectorRecord(
                id=f"filter-{i}",
                vector=[i * 0.001] * 1536,
                metadata={"category": f"cat-{i % 2}"},
                text=f"content {i}",
            )
            for i in range(10)
        ]

        vector_db.insert(records)

        query_vector = [0.5] * 1536
        results = vector_db.search(
            query_vector=query_vector,
            limit=10,
            filters={"category": "cat-0"},
            use_hybrid=False,
        )

        assert all(r["metadata"]["category"] == "cat-0" for r in results)

    def test_search_empty_database(self, vector_db):
        """测试空数据库搜索"""
        results = vector_db.search(query_vector=[0.1] * 1536, limit=5, use_hybrid=False)

        assert len(results) == 0

    def test_close_connection(self, vector_db):
        """测试关闭连接"""
        vector_db.close()

        assert vector_db._is_connection_closed()
