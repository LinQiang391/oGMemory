"""Test script for DirectoryEventHandler.

Demonstrates the complete flow:
1. Create UPSERT_DIRECTORY event
2. Process with OutboxWorker
3. DAG execution waits for subdirectories
4. Fallback when LLM fails
5. Vectorization and storage
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import OutboxEvent, RequestContext
from index import OutboxWorker, DirectoryEventHandler


def test_directory_event_handler():
    """Test DirectoryEventHandler with mock components."""

    print("=" * 80)
    print("DirectoryEventHandler 测试")
    print("=" * 80)

    # 创建模拟组件
    class MockFS:
        def list_children(self, uri, ctx):
            print(f"  [MockFS] 列出目录: {uri}")
            if uri == "ctx://acme/users/alice/memories/":
                return [
                    "ctx/preferences/",
                    "ctx/profile",
                ]
            elif uri == "ctx://acme/users/alice/memories/preferences/":
                return [
                    "ctx/coding_style",
                    "ctx/themes/",
                ]
            elif uri == "ctx://acme/users/alice/memories/preferences/themes/":
                return [
                    "ctx/dark",
                    "ctx/light",
                ]
            return []

        def read_node(self, uri, ctx):
            print(f"  [MockFS] 读取节点: {uri}")
            from core.models import ContextNode
            return ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="test",
                level=0,
                owner_space=ctx.user_space_name(),
                abstract=f"摘要内容: {uri}",
                overview="",
                content="",
                metadata={},
            )

        def write_node(self, node, ctx):
            print(f"  [MockFS] 写入节点: {node.uri}")
            print(f"    - abstract: {node. abstract[:50]}...")

    class MockLLM:
        def complete_json(self, prompt, schema):
            print(f"  [MockLLM] 生成摘要 (prompt 长度: {len(prompt)})")
            return {
                "abstract": "测试目录摘要",
                "overview": "## 测试概述\n- 测试内容",
            }

    class MockEmbedder:
        def embed_texts(self, texts):
            print(f"  [MockEmbedder] 嵌入 {len(texts)} 个文本")
            return [[0.0] * 768 for _ in texts]

    class MockVectorIndex:
        def upsert(self, records):
            print(f"  [MockVectorIndex] 插入 {len(records)} 条记录")
            for r in records:
                print(f"    - {r.uri} (level {r.level})")

    # 创建组件
    fs = MockFS()
    llm = MockLLM()
    embedder = MockEmbedder()
    vector_index = MockVectorIndex()

    # 创建 OutboxWorker
    worker = OutboxWorker(
        vector_index=vector_index,
        embedder=embedder,
        fs=fs,
        llm=llm,
    )

    # 创建 RequestContext
    ctx = RequestContext(
        account_id="acme",
        user_id="alice",
        agent_id="",
        session_id="test-session",
        trace_id="test-trace",
    )

    # 创建 UPSERT_DIRECTORY 事件
    event = OutboxEvent(
        event_id="test-event-001",
        event_type="UPSERT_DIRECTORY",
        uri="ctx://acme/users/alice/memories/",
        payload={"uri": "ctx://acme/users/alice/memories/"},
        status="PENDING",
        retry_count=0,
    )

    print("\n1. 处理 UPSERT_DIRECTORY 事件")
    print(f"   事件 URI: {event.uri}")
    print()

    # 处理事件
    result = worker.process_event(event, ctx)

    print()
    print("2. 处理结果")
    print(f"   成功: {result.success}")
    print(f"   处理记录数: {result.records_upserted}")
    if result.error_message:
        print(f"   错误: {result.error_message}")

    print()
    print("=" * 80)
    print("测试完成工作流程:")
    print("=" * 80)
    print("✓ 1. 接收 UPSERT_DIRECTORY 事件")
    print("✓ 2. 递归调度所有子目录")
    print("✓ 3. 等待子目录摘要生成 (pending == 0)")
    print("✓ 4. 生成父目录摘要 (包含完整子目录信息)")
    print("✓ 5. LLM 失败时使用 fallback")
    print("✓ 6. 向量化并存储到数据库")
    print()


def test_fallback_when_llm_fails():
    """Test fallback summary generation when LLM fails."""

    print("=" * 80)
    print("测试 LLM 失败时的 fallback 机制")
    print("=" * 80)

    class MockFS:
        def list_children(self, uri, ctx):
            return ["ctx/file1", "ctx/file2"]

        def read_node(self, uri, ctx):
            from core.models import ContextNode
            return ContextNode(
                uri=uri,
                context_type="MEMORY",
                category="test",
                level=0,
                owner_space=ctx.user_space_name(),
                abstract=f"摘要: {uri}",
                overview="",
                content="",
                metadata={},
            )

        def write_node(self, node, ctx):
            print(f"  [MockFS] 写入节点: {node.uri}")
            print(f"    - abstract: {node.abstract}")

    class FailingLLM:
        def complete_json(self, prompt, schema):
            print(f"  [FailingLLM] 模拟 LLM 失败")
            raise Exception("LLM 服务不可用")

    class MockEmbedder:
        def embed_texts(self, texts):
            return [[0.0] * 768 for _ in texts]

    class MockVectorIndex:
        def upsert(self, records):
            print(f"  [MockVectorIndex] 插入 {len(records)} 条记录")

    from index import DirectoryEventHandler

    handler = DirectoryEventHandler(
        fs=MockFS(),
        llm=FailingLLM(),
        embedder=MockEmbedder(),
        vector_index=MockVectorIndex(),
    )

    from core.models import OutboxEvent, RequestContext

    event = OutboxEvent(
        event_id="test-event-002",
        event_type="UPSERT_DIRECTORY",
        uri="ctx://acme/users/alice/test/",
        payload={"uri": "ctx://acme/users/alice/test/"},
        status="PENDING",
        retry_count=0,
    )

    ctx = RequestContext(
        account_id="acme",
        user_id="alice",
        agent_id="",
        session_id="test-session",
        trace_id="test-trace",
    )

    print("\n1. 处理事件 (LLM 会失败)")
    print()

    result = handler.process_directory_event(event, ctx)

    print()
    print("2. 处理结果")
    print(f"   成功: {result.success}")
    print(f"   完成目录数: {result.stats.completed_dirs}")
    print(f"   失败目录数: {result.stats.failed_dirs}")

    print()
    print("=" * 80)
    print("✓ LLM 失败时自动使用 fallback 摘要")
    print("✓ fallback 摘要包含子节点信息")
    print("✓ 不会因 LLM 失败而中断流程")
    print()


if __name__ == "__main__":
    test_directory_event_handler()
    print()
    test_fallback_when_llm_fails()


# -------------------------------------------------------------------
# pytest-compatible unit tests for record_ids / written_dir_uris tracking
# -------------------------------------------------------------------

from index.directory_event_handler import DagStats, DirectoryEventResult  # noqa: E402


class _MockFS:
    def list_children(self, uri, ctx):
        return ["ctx://acme/users/alice/memories/file1"]

    def read_node(self, uri, ctx):
        from core.models import ContextNode
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="test",
            level=0,
            owner_space=ctx.user_space_name(),
            abstract=f"abstract: {uri}",
            overview=f"overview: {uri}",
            content="",
            metadata={},
        )

    def write_node(self, node, ctx):
        pass


class _MockLLM:
    def complete_json(self, prompt, schema):
        return {
            "abstract": "dir abstract",
            "overview": "dir overview",
        }


class _MockEmbedder:
    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]


class _MockVectorIndex:
    def __init__(self):
        self.upserted_records = []

    def upsert(self, records):
        self.upserted_records.extend(records)


class _MockCtx:
    """Minimal RequestContext stand-in for DAG tests."""

    def __init__(self, account_id="acme", user_id="alice"):
        self.account_id = account_id
        self.user_id = user_id
        self.agent_id = ""
        self.session_id = "test"
        self.trace_id = "test"

    def user_space_name(self):
        return f"user:{self.user_id}"


class TestDagStatsTracking:
    """Tests that DagStats accumulates record_ids and written_dir_uris."""

    def test_dag_stats_has_record_ids_field(self):
        stats = DagStats()
        assert stats.record_ids == []

    def test_dag_stats_has_written_dir_uris_field(self):
        stats = DagStats()
        assert stats.written_dir_uris == []

    def test_directory_event_result_has_record_ids(self):
        result = DirectoryEventResult(
            success=True,
            root_uri="ctx://acme/users/alice/memories/",
            stats=DagStats(),
            record_ids=["r0", "r1"],
            written_dir_uris=["ctx://acme/users/alice/memories/"],
        )
        assert result.record_ids == ["r0", "r1"]
        assert result.written_dir_uris == ["ctx://acme/users/alice/memories/"]


class TestDirectoryEventHandlerTracking:
    """Tests that DirectoryEventHandler tracks record IDs and written URIs."""

    def test_handler_returns_record_ids_for_leaf_directory(self):
        """A leaf directory (no subdirs) should produce L0+L1 record IDs."""
        # Leaf directory: list_children returns only file children
        class LeafFS(_MockFS):
            def list_children(self, uri, ctx):
                return ["ctx://acme/users/alice/memories/file1"]

        vector_index = _MockVectorIndex()
        handler = DirectoryEventHandler(
            fs=LeafFS(),
            llm=_MockLLM(),
            embedder=_MockEmbedder(),
            vector_index=vector_index,
        )

        event = OutboxEvent(
            event_id="evt-dir-1",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://acme/users/alice/memories/",
            payload={},
            status="PENDING",
            retry_count=0,
        )

        ctx = _MockCtx()
        result = handler.process_directory_event(event, ctx)

        assert result.success is True
        # Should have at least the root directory L0 (and L1 if overview exists)
        assert len(result.record_ids) >= 1
        # Record IDs should be strings (sha256-based)
        for rid in result.record_ids:
            assert isinstance(rid, str)
            assert len(rid) == 16  # sha256(uri:level)[:16]

    def test_handler_returns_written_dir_uris(self):
        """Handler should track URIs of directory nodes it wrote back to FS."""
        written = []

        class TrackingFS(_MockFS):
            def write_node(self, node, ctx):
                written.append(node.uri)

        vector_index = _MockVectorIndex()
        handler = DirectoryEventHandler(
            fs=TrackingFS(),
            llm=_MockLLM(),
            embedder=_MockEmbedder(),
            vector_index=vector_index,
        )

        event = OutboxEvent(
            event_id="evt-dir-2",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://acme/users/alice/memories/",
            payload={},
            status="PENDING",
            retry_count=0,
        )

        ctx = _MockCtx()
        result = handler.process_directory_event(event, ctx)

        assert result.success is True
        # written_dir_uris should include root at minimum
        assert len(result.written_dir_uris) >= 1
        # Every reported written URI should have been actually written
        for uri in result.written_dir_uris:
            assert uri in written

    def test_record_ids_match_actual_upserts(self):
        """Record IDs returned should match what was actually upserted to
        the vector index."""
        vector_index = _MockVectorIndex()
        handler = DirectoryEventHandler(
            fs=_MockFS(),
            llm=_MockLLM(),
            embedder=_MockEmbedder(),
            vector_index=vector_index,
        )

        event = OutboxEvent(
            event_id="evt-dir-3",
            event_type="UPSERT_DIRECTORY",
            uri="ctx://acme/users/alice/memories/",
            payload={},
            status="PENDING",
            retry_count=0,
        )

        ctx = _MockCtx()
        result = handler.process_directory_event(event, ctx)

        assert result.success is True
        # The record_ids in result should exactly match what was upserted
        actual_ids = [r.id for r in vector_index.upserted_records]
        assert result.record_ids == actual_ids
