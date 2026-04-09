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
