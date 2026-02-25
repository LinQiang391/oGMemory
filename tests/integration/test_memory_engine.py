"""记忆引擎集成测试"""

import os
import pytest
from core.memory_engine import MemoryEngine, SearchResult


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("TEST_DB_PASSWORD"),
    reason="Required credentials not set",
)
class TestMemoryEngineIntegration:
    """记忆引擎集成测试"""

    @pytest.fixture
    def memory_engine(self, test_config):
        """记忆引擎 fixture"""
        kwargs = {
            "db_host": test_config["db_host"],
            "db_port": test_config["db_port"],
            "db_name": test_config["db_name"],
            "db_user": test_config["db_user"],
            "db_password": test_config["db_password"],
        }
        if test_config.get("db_sslmode") is not None:
            kwargs["db_sslmode"] = test_config["db_sslmode"]
        if test_config.get("db_gssencmode") is not None:
            kwargs["db_gssencmode"] = test_config["db_gssencmode"]
        if test_config.get("embedding_model"):
            kwargs["embedding_model"] = test_config["embedding_model"]
        engine = MemoryEngine(**kwargs)
        yield engine
        engine.clear_all()

    @pytest.mark.asyncio
    async def test_index_and_search_workflow(self, memory_engine, sample_markdown):
        """测试完整索引和搜索工作流"""
        count = await memory_engine.index_file(str(sample_markdown))

        assert count > 0, "应该索引至少一个块"

        results = await memory_engine.search("test document", limit=5)

        assert len(results) > 0, "应该找到相关结果"
        assert all(isinstance(r, SearchResult) for r in results)
        assert any("test" in r.text.lower() for r in results), "应有结果包含与查询相关的内容"

    @pytest.mark.asyncio
    async def test_add_and_retrieve_memory(self, memory_engine):
        """测试添加和检索记忆"""
        record_id = await memory_engine.add_memory(
            "test memory content", metadata={"type": "test", "priority": "high"}
        )

        assert record_id is not None, "应该返回记录ID"

        results = await memory_engine.search("test memory", limit=1)

        assert len(results) > 0, "应该找到添加的记忆"
        assert "test memory content" in results[0].text

    @pytest.mark.asyncio
    async def test_hybrid_search(self, memory_engine):
        """测试混合搜索功能"""
        await memory_engine.add_memory("Python programming language")
        await memory_engine.add_memory("Python best practices guide")
        await memory_engine.add_memory("JavaScript basics")
        await memory_engine.add_memory("Python async programming")

        results = await memory_engine.search("Python", limit=5, use_hybrid=True)

        assert len(results) > 0, "混合搜索应该返回结果"
        assert any("Python" in r.text for r in results), "应该包含Python相关内容"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, memory_engine):
        """测试带过滤的搜索"""
        await memory_engine.add_memory(
            "important note", metadata={"type": "important", "priority": "high"}
        )
        await memory_engine.add_memory(
            "regular note", metadata={"type": "regular", "priority": "low"}
        )

        results = await memory_engine.search("note", filters={"type": "important"})

        assert len(results) == 1, "应该只返回重要的笔记"
        assert results[0].metadata["type"] == "important"

    @pytest.mark.asyncio
    async def test_vector_only_search(self, memory_engine):
        """测试纯向量搜索（不使用混合）"""
        await memory_engine.add_memory("machine learning algorithms")
        await memory_engine.add_memory("deep learning concepts")

        results = await memory_engine.search("learning", use_hybrid=False)

        assert len(results) > 0, "向量搜索应该返回结果"

    @pytest.mark.asyncio
    async def test_delete_by_source(self, memory_engine, sample_markdown):
        """测试根据源删除"""
        count = await memory_engine.index_file(str(sample_markdown))

        assert count > 0, "应该索引文件"

        deleted_count = memory_engine.delete_by_source(str(sample_markdown))

        assert deleted_count > 0, "应该删除记录"

        results = await memory_engine.search("test", limit=10)

        assert len(results) == 0, "删除后应该没有结果"

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_engine):
        """测试获取统计信息"""
        await memory_engine.add_memory("test content")

        stats = memory_engine.get_stats()

        assert "total_records" in stats
        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
        assert stats["total_records"] > 0
        assert stats["embedding_dimension"] > 0

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_engine):
        """测试清空所有数据"""
        await memory_engine.add_memory("test 1")
        await memory_engine.add_memory("test 2")

        stats_before = memory_engine.get_stats()
        assert stats_before["total_records"] >= 2

        memory_engine.clear_all()

        stats_after = memory_engine.get_stats()
        assert stats_after["total_records"] == 0

    @pytest.mark.asyncio
    async def test_index_directory(self, memory_engine, temp_dir):
        """测试索引整个目录"""
        md_file1 = temp_dir / "test1.md"
        md_file2 = temp_dir / "test2.md"

        md_file1.write_text("# Test 1\nContent for test 1.")
        md_file2.write_text("# Test 2\nContent for test 2.")

        count = await memory_engine.index_directory(str(temp_dir))

        assert count >= 2, "应该索引至少两个文件"

        results = await memory_engine.search("test", limit=10)

        assert len(results) > 0, "应该找到相关内容"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, memory_engine):
        """测试空查询"""
        results = await memory_engine.search("", limit=5)

        assert len(results) == 0, "空查询应该返回空结果"

    @pytest.mark.asyncio
    async def test_search_limit(self, memory_engine):
        """测试结果数量限制"""
        for i in range(10):
            await memory_engine.add_memory(f"content {i}")

        results = await memory_engine.search("content", limit=3)

        assert len(results) == 3, "应该返回限制数量的结果"

    @pytest.mark.asyncio
    async def test_search_score_ordering(self, memory_engine):
        """测试搜索结果按分数排序"""
        await memory_engine.add_memory("Python programming")
        await memory_engine.add_memory("Python code")

        results = await memory_engine.search("Python", limit=5)

        assert len(results) > 1, "应该找到多个结果"

        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, "结果应该按分数降序排列"
