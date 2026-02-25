"""使用示例"""

import asyncio
from ogmemory import MemoryEngine


async def basic_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")

    with MemoryEngine(
        db_host="localhost",
        db_port=5432,
        db_name="memory_db",
        db_user="postgres",
        db_password="your_password",
    ) as engine:
        await engine.index_directory("./docs")

        results = await engine.search("如何配置数据库连接", limit=5)

        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.score:.4f}] {result.text[:100]}...")
            print(f"   来源: {result.source}\n")


async def agent_memory_example():
    """AI Agent记忆示例"""
    print("=== AI Agent记忆示例 ===\n")

    with MemoryEngine() as engine:
        await engine.add_memory(
            "用户偏好使用深色主题", metadata={"type": "preference", "user": "user123"}
        )

        await engine.add_memory(
            "项目使用Python和PostgreSQL", metadata={"type": "project_info"}
        )

        query = "用户喜欢什么主题？"
        results = await engine.search(query, limit=3)

        print(f"查询: {query}\n")
        for result in results:
            print(f"- {result.text} (相似度: {result.score:.4f})")


async def markdown_knowledge_base():
    """Markdown知识库示例"""
    print("=== Markdown知识库示例 ===\n")

    with MemoryEngine(chunk_size=1500, chunk_overlap=200) as engine:
        await engine.index_directory("./knowledge_base")

        stats = engine.get_stats()
        print(f"索引统计: {stats}\n")

        results = await engine.search(
            "机器学习算法的分类", limit=5, filters={"source": "./knowledge_base/ml.md"}
        )

        for result in results:
            heading = result.metadata.get("heading", "无标题")
            print(f"## {heading}")
            print(f"{result.text[:200]}...\n")


async def main():
    """运行所有示例"""
    try:
        await basic_example()
    except Exception as e:
        print(f"基本示例失败: {e}\n")

    try:
        await agent_memory_example()
    except Exception as e:
        print(f"Agent示例失败: {e}\n")

    try:
        await markdown_knowledge_base()
    except Exception as e:
        print(f"知识库示例失败: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
