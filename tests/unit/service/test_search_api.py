#!/usr/bin/env python3
"""Standalone test script for oG-Memory search_memory API.

This script tests the search_memory and read_memory APIs with real AGFS data.
It will:
1. Check if AGFS has existing data for the test account
2. If no data exists, write test memories first
3. Perform semantic search queries
4. Test read_memory for specific URIs
5. Print detailed results and statistics

Usage:
    python tests/unit/service/test_search_api.py
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
# File is at tests/unit/service/test_search_api.py, so we go up 3 levels
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.models import RequestContext, CandidateMemory, IndexRecord, RetrievalConfig
from core.enums import ContextType
from service.api import MemoryReadAPI, MemoryWriteAPI
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.assembly_service import AssemblyService
from retrieval.memory_read_service import MemoryReadService
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from providers.embedder.mock_embedder import MockEmbedder
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.llm.mock_llm import MockLLM
from pyagfs import AGFSClient


# Configuration
ACCOUNT_ID = "default"
USER_ID = "default"  # Changed from "default-user" to match existing AGFS data
AGENT_ID = "main"
AGFS_API_URL = "http://localhost:1833/api/v1"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_step(step_num: int, description: str):
    """Print a formatted step header."""
    print(f"\n[{step_num}] {description}...")


def print_success(message: str):
    """Print a success message."""
    print(f"  ✓ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"  ✗ {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"  ℹ {message}")


def initialize_components():
    """Initialize all required components for the test.

    Returns:
        Tuple of (fs, write_api, read_api)
    """
    print_step(1, "Initializing components")

    # Initialize AGFS client
    agfs_client = AGFSClient(api_base_url=AGFS_API_URL)
    print_success(f"AGFS client connected to {AGFS_API_URL}")

    # Initialize ContextFS
    fs = AGFSContextFS(agfs_client)
    print_success("ContextFS initialized")

    # Initialize LLM with custom fixtures for our test data
    llm = MockLLM()

    # Configure mock tool calls for extraction
    llm._mock_tool_calls = [
        {
            "tool": "extract_entity",
            "input": {
                "abstract": "数据库迁移项目经验",
                "overview": "从MySQL迁移到PostgreSQL的经验，包括数据类型兼容性和SQL语法差异处理",
                "content": "用户正在进行MySQL到PostgreSQL的数据库迁移项目。遇到的主要问题包括：1) 日期时间类型兼容性，MySQL的DATETIME对应PostgreSQL的TIMESTAMP，但时区处理方式不同；2) SQL语法差异，如字符串拼接，MySQL用CONCAT()，PostgreSQL可以用||操作符；3) PostgreSQL的窗口函数和CTE（WITH子句）功能更强大。",
                "confidence": 0.9,
            }
        },
        {
            "tool": "extract_preference",
            "input": {
                "abstract": "Python编程学习",
                "overview": "学习Python异步编程，特别是asyncio和aiohttp库的使用",
                "content": "用户正在学习Python编程，专注于异步编程。asyncio提供了事件循环和协程支持，aiohttp是异步HTTP客户端。建议先理解async/await语法和事件循环机制，再逐步学习异步I/O操作。",
                "confidence": 0.85,
            }
        },
        {
            "tool": "extract_preference",
            "input": {
                "abstract": "Django框架开发经验",
                "overview": "使用Django框架开发Web应用，关注模型层设计和ORM性能优化",
                "content": "用户使用Django框架开发Web应用。模型层设计关键点：1) 合理使用ForeignKey和ManyToManyField；2) 考虑查询性能，使用select_related和prefetch_related优化查询；3) 利用Meta类配置表选项。建议先设计数据库schema，再定义Django模型。",
                "confidence": 0.88,
            }
        },
    ]
    print_success("MockLLM initialized with custom tool calls")

    # Initialize Write API
    write_api = MemoryWriteAPI(fs=fs, llm=llm, outbox_store=None)
    print_success("MemoryWriteAPI initialized")

    # Initialize retrieval components
    cfg = RetrievalConfig()
    embedder = MockEmbedder()
    vector_index = InMemoryVectorIndex(dimension=384)

    planner = QueryPlanner(cfg)
    seed_retriever = SeedRetriever(vector_index, embedder, cfg)
    hierarchical_searcher = HierarchicalSearcher(vector_index, cfg)
    assembly = AssemblyService(cfg)

    pipeline = RetrievalPipeline(
        planner=planner,
        seed_retriever=seed_retriever,
        hierarchical_searcher=hierarchical_searcher,
        assembly=assembly,
        config=cfg,
    )
    print_success("RetrievalPipeline initialized")

    # Initialize Read API
    read_service = MemoryReadService(fs=fs)
    read_api = MemoryReadAPI(
        pipeline=pipeline,
        read_service=read_service,
        config=cfg,
    )
    print_success("MemoryReadAPI initialized")

    return fs, write_api, read_api, vector_index


def create_request_context() -> RequestContext:
    """Create a RequestContext for the test.

    Returns:
        RequestContext with test account/user/agent IDs
    """
    return RequestContext(
        account_id=ACCOUNT_ID,
        user_id=USER_ID,
        agent_id=AGENT_ID,
        session_id=f"test-session-{int(time.time())}",
        trace_id=f"test-trace-{int(time.time())}",
    )


def check_existing_data(fs: AGFSContextFS, ctx: RequestContext) -> bool:
    """Check if there's existing data in AGFS.

    Args:
        fs: ContextFS instance
        ctx: RequestContext

    Returns:
        True if data exists, False otherwise
    """
    print_step(2, "Checking for existing data")

    profile_uri = f"ctx://{ACCOUNT_ID}/users/{USER_ID}/memories/profile"

    try:
        exists = fs.exists(profile_uri, ctx)
        if exists:
            print_success(f"Found existing data at {profile_uri}")
            return True
        else:
            print_info(f"No existing data at {profile_uri}")
            return False
    except Exception as e:
        print_info(f"Error checking for existing data: {e}")
        return False


def write_test_data(write_api: MemoryWriteAPI, ctx: RequestContext) -> bool:
    """Write test data to AGFS.

    Args:
        write_api: MemoryWriteAPI instance
        ctx: RequestContext

    Returns:
        True if successful, False otherwise
    """
    print_step(3, "Writing test data")

    # Test messages about various topics
    test_messages = [
        {
            "role": "user",
            "content": "我最近在做一个数据库迁移项目，从MySQL迁移到PostgreSQL。过程中遇到了一些数据类型兼容性问题，特别是日期时间类型的处理。"
        },
        {
            "role": "assistant",
            "content": "MySQL到PostgreSQL的迁移确实需要注意数据类型差异。日期时间类型方面，MySQL的DATETIME对应PostgreSQL的TIMESTAMP，但时区处理方式不同。建议使用TIMESTAMP WITH TIME ZONE来保留时区信息。"
        },
        {
            "role": "user",
            "content": "我还发现了一些SQL语法的差异，比如字符串拼接，MySQL用CONCAT()，PostgreSQL可以用||操作符。"
        },
        {
            "role": "assistant",
            "content": "是的，PostgreSQL支持标准的SQL语法。字符串拼接可以用||操作符，也可以用CONCAT()函数。另外，PostgreSQL的窗口函数和CTE（WITH子句）功能更强大，建议在迁移时充分利用这些特性。"
        },
        {
            "role": "user",
            "content": "我最近在学Python编程，特别是异步编程。asyncio和aiohttp库很有用。"
        },
        {
            "role": "assistant",
            "content": "Python的异步编程确实很强大。asyncio提供了事件循环和协程支持，aiohttp是异步HTTP客户端。建议先理解async/await语法和事件循环机制，再逐步学习异步I/O操作。"
        },
        {
            "role": "user",
            "content": "我在用Django框架开发Web应用，模型层的设计很关键。"
        },
        {
            "role": "assistant",
            "content": "Django的ORM非常强大。设计模型时要注意：1) 合理使用ForeignKey和ManyToManyField；2) 考虑查询性能，必要时使用select_related和prefetch_related；3) 利用Meta类配置表选项。建议先设计好数据库schema，再定义Django模型。"
        },
    ]

    try:
        result = write_api.commit_session(
            messages=test_messages,
            ctx=ctx,
            confidence_threshold=0.5,
        )

        print_success(f"Commit session completed")
        print_info(f"  Candidates extracted: {result['candidates_extracted']}")
        print_info(f"  Writes completed: {result['writes_completed']}")
        print_info(f"  Writes skipped: {result['writes_skipped']}")

        # Wait a moment for writes to settle
        time.sleep(0.5)

        return result['writes_completed'] > 0

    except Exception as e:
        print_error(f"Failed to write test data: {e}")
        import traceback
        print_info(f"Traceback: {traceback.format_exc()}")
        return False


def rebuild_vector_index(fs: AGFSContextFS, vector_index: InMemoryVectorIndex, ctx: RequestContext):
    """Rebuild vector index by walking the AGFS filesystem directly.

    Args:
        fs: ContextFS instance
        vector_index: InMemoryVectorIndex instance
        ctx: RequestContext
    """
    print_step(4, "Rebuilding vector index from AGFS")

    import os

    # Walk the AGFS data directory directly
    agfs_base_path = f"/tmp/agfs-data/accounts/{ACCOUNT_ID}/users/{USER_ID}/memories"

    if not os.path.exists(agfs_base_path):
        print_info(f"AGFS directory does not exist: {agfs_base_path}")
        return

    records_created = 0
    nodes_processed = 0

    try:
        # Walk through all categories
        for category in os.listdir(agfs_base_path):
            category_path = os.path.join(agfs_base_path, category)
            if not os.path.isdir(category_path):
                continue

            print_info(f"  Category '{category}'")

            # Walk through all items in this category
            for item_slug in os.listdir(category_path):
                item_path = os.path.join(category_path, item_slug)
                if not os.path.isdir(item_path):
                    continue

                # Check if .meta.json exists (indicates a valid node)
                meta_path = os.path.join(item_path, ".meta.json")
                if not os.path.exists(meta_path):
                    continue

                nodes_processed += 1

                # Build URI from path
                # Path: /tmp/agfs-data/accounts/{account}/users/{user}/memories/{category}/{slug}
                # URI:   ctx://{account}/users/{user}/memories/{category}/{slug}
                item_uri = f"ctx://{ACCOUNT_ID}/users/{USER_ID}/memories/{category}/{item_slug}"

                try:
                    node = fs.read_node(item_uri, ctx)

                    # Create index records for each level
                    abstract_record = IndexRecord(
                        id=f"{hash(item_uri + ':0') % (10**16):016x}",
                        uri=item_uri,
                        level=0,
                        text=node.abstract or "",
                        filters={
                            "account_id": ACCOUNT_ID,
                            "owner_space": ctx.user_space_name(),
                            "category": category,
                            "context_type": node.context_type,
                        },
                        metadata={
                            "category": node.category,
                            "context_type": node.context_type,
                            "has_overview": bool(node.overview),
                            "has_content": bool(node.content),
                        },
                    )
                    vector_index.upsert([abstract_record])
                    records_created += 1

                    if node.overview:
                        overview_record = IndexRecord(
                            id=f"{hash(item_uri + ':1') % (10**16):016x}",
                            uri=item_uri,
                            level=1,
                            text=node.overview or "",
                            filters={
                                "account_id": ACCOUNT_ID,
                                "owner_space": ctx.user_space_name(),
                                "category": category,
                                "context_type": node.context_type,
                            },
                            metadata={
                                "category": node.category,
                                "context_type": node.context_type,
                                "has_overview": True,
                                "has_content": bool(node.content),
                            },
                        )
                        vector_index.upsert([overview_record])
                        records_created += 1

                    if node.content:
                        content_record = IndexRecord(
                            id=f"{hash(item_uri + ':2') % (10**16):016x}",
                            uri=item_uri,
                            level=2,
                            text=node.content or "",
                            filters={
                                "account_id": ACCOUNT_ID,
                                "owner_space": ctx.user_space_name(),
                                "category": category,
                                "context_type": node.context_type,
                            },
                            metadata={
                                "category": node.category,
                                "context_type": node.context_type,
                                "has_overview": bool(node.overview),
                                "has_content": True,
                            },
                        )
                        vector_index.upsert([content_record])
                        records_created += 1

                except Exception as e:
                    print_info(f"    Could not index {item_slug}: {e}")
                    continue

        print_success(f"Processed {nodes_processed} nodes, created {records_created} index records")

    except Exception as e:
        print_error(f"Failed to rebuild index: {e}")
        import traceback
        print_info(f"Traceback: {traceback.format_exc()}")


def _run_search_memory_test(read_api: MemoryReadAPI, ctx: RequestContext):
    """Test the search_memory API with various queries.

    Args:
        read_api: MemoryReadAPI instance
        ctx: RequestContext

    Returns:
        List of search results for summary
    """
    print_step(5, "Testing search_memory API")

    test_queries = [
        "数据库迁移",
        "Python编程",
        "Django框架",
        "异步编程",
    ]

    all_results = []

    for query in test_queries:
        print_info(f"\n  Query: '{query}'")

        try:
            start_time = time.monotonic()
            result = read_api.search_memory(
                query=query,
                ctx=ctx,
                top_k=5,
                mode="QUICK",
            )
            duration = (time.monotonic() - start_time) * 1000  # ms

            print_success(f"Found {len(result.hits)} results in {duration:.1f}ms")

            for i, hit in enumerate(result.hits[:3], 1):  # Show top 3
                print_info(f"    [{i}] Score: {hit.score:.3f}")
                print_info(f"        URI: {hit.uri}")
                print_info(f"        Category: {hit.category}")
                if hit.abstract:
                    abstract_preview = hit.abstract[:80] + "..." if len(hit.abstract) > 80 else hit.abstract
                    print_info(f"        Abstract: {abstract_preview}")

            all_results.append({
                "query": query,
                "count": len(result.hits),
                "duration_ms": duration,
                "top_score": result.hits[0].score if result.hits else 0.0,
            })

        except Exception as e:
            print_error(f"Search failed: {e}")
            import traceback
            print_info(f"Traceback: {traceback.format_exc()}")
            all_results.append({
                "query": query,
                "count": 0,
                "duration_ms": 0,
                "top_score": 0.0,
                "error": str(e),
            })

    return all_results


def _run_read_memory_test(read_api: MemoryReadAPI, ctx: RequestContext, search_results: list):
    """Test the read_memory API with URIs from search results.

    Args:
        read_api: MemoryReadAPI instance
        ctx: RequestContext
        search_results: List of search results from test_search_memory
    """
    print_step(6, "Testing read_memory API")

    # Find the first successful search result with a URI
    test_uri = None
    for result in search_results:
        if result.get("count", 0) > 0:
            # We need to run a small search to get an actual URI
            try:
                search_result = read_api.search_memory(
                    query=result["query"],
                    ctx=ctx,
                    top_k=1,
                )
                if search_result.hits:
                    test_uri = search_result.hits[0].uri
                    break
            except Exception:
                continue

    if not test_uri:
        print_info("No URIs found from search results, skipping read_memory test")
        return

    print_info(f"Reading URI: {test_uri}")

    try:
        start_time = time.monotonic()
        block = read_api.read_memory(uri=test_uri, ctx=ctx)
        duration = (time.monotonic() - start_time) * 1000  # ms

        print_success(f"Read completed in {duration:.1f}ms")
        print_info(f"  Category: {block.category}")
        print_info(f"  Owner space: {block.owner_space}")

        if block.content_excerpt:
            content_preview = block.content_excerpt[:150] + "..." if len(block.content_excerpt) > 150 else block.content_excerpt
            print_info(f"  Content preview: {content_preview}")
        else:
            print_info("  Content: (empty)")

    except Exception as e:
        print_error(f"Read failed: {e}")
        import traceback
        print_info(f"Traceback: {traceback.format_exc()}")


def print_summary(search_results: list, start_time: float):
    """Print test summary.

    Args:
        search_results: List of search results from test_search_memory
        start_time: Test start time
    """
    print_step(7, "Test Summary")

    total_results = sum(r.get("count", 0) for r in search_results)
    avg_results = total_results / len(search_results) if search_results else 0
    total_duration = (time.monotonic() - start_time) * 1000  # ms

    print_success(f"Total test duration: {total_duration:.1f}ms")
    print_info(f"Total search results: {total_results}")
    print_info(f"Average results per query: {avg_results:.1f}")

    print_info("\nQuery breakdown:")
    for result in search_results:
        query = result["query"]
        count = result.get("count", 0)
        duration = result.get("duration_ms", 0)
        top_score = result.get("top_score", 0.0)
        error = result.get("error", "")

        if error:
            print_error(f"  '{query}': FAILED - {error}")
        else:
            print_success(f"  '{query}': {count} results, {duration:.1f}ms, top_score={top_score:.3f}")

    print_success("\n=== TEST COMPLETED ===")


def main():
    """Main test execution."""
    print_section("oG-Memory Search API Test")
    print_info(f"Account: {ACCOUNT_ID}")
    print_info(f"User: {USER_ID}")
    print_info(f"Agent: {AGENT_ID}")
    print_info(f"AGFS: {AGFS_API_URL}")

    start_time = time.monotonic()

    try:
        # Initialize components
        fs, write_api, read_api, vector_index = initialize_components()

        # Create request context
        ctx = create_request_context()

        # Check for existing data
        has_data = check_existing_data(fs, ctx)

        # Write test data if needed
        if not has_data:
            success = write_test_data(write_api, ctx)
            if not success:
                print_error("Failed to write test data, continuing anyway...")

        # Rebuild vector index from AGFS
        rebuild_vector_index(fs, vector_index, ctx)

        # Test search_memory
        search_results = _run_search_memory_test(read_api, ctx)

        # Test read_memory
        _run_read_memory_test(read_api, ctx, search_results)

        # Print summary
        print_summary(search_results, start_time)

        return 0

    except Exception as e:
        print_error(f"Test failed with exception: {e}")
        import traceback
        print_info(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
