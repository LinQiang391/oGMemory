"""Phase 1 End-to-End Integration Tests

These tests verify the complete data flow from write to retrieval:
1. Write: Messages → Extract → Plan → Write to AGFS → Index
2. Index: ContextNode → IndexRecords → VectorIndex
3. Retrieve: Query → Plan → Search → Assemble → Return

Requirements:
    - AGFS server must be running at http://localhost:8080
    - Start with: cd agfs/agfs-server && ./build/agfs-server -c config.yaml
"""

import uuid

import pytest

from core.models import (
    RequestContext,
    ContextNode,
    CandidateMemory,
    TypedQuery,
    WritePlan,
    RelationEdge,
)
from core.interfaces import ContextFS, VectorIndex, Embedder
from core.enums import NodeStatus, ContextType
from core.errors import AccessDeniedError, NodeNotFoundError

from commit.merge_policies import ProfilePolicy, AggregateTopicPolicy, AppendOnlyPolicy
from commit.context_writer import ContextWriter
from commit.candidate_pipeline import CandidatePipeline

from extraction.profile_extractor import ProfileExtractor
from extraction.preference_extractor import PreferenceExtractor
from extraction.entity_event_extractor import EntityEventExtractor

from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.vector_index.in_memory_index import InMemoryVectorIndex

from retrieval import Planner, L0Retriever, MemoryReadService, AssemblyService
from service.api import MemoryReadAPI, init_api


# ============================================================================
# Fixtures - Using conftest.py for AGFS fixtures
# ============================================================================

@pytest.fixture
def sample_messages():
    """Sample user messages for extraction testing."""
    return [
        {"role": "user", "content": "我叫张三，是一名软件工程师，喜欢 Python 编程。"},
        {"role": "assistant", "content": "你好张三！很高兴认识你。"},
        {"role": "user", "content": "我平时习惯用 VS Code 作为编辑器，代码风格喜欢 PEP8。"},
    ]


# ============================================================================
# Write Path Tests
# ============================================================================

class TestWritePathE2E:
    """End-to-end tests for the write path."""

    def test_profile_write_to_agfs(self, agfs, sample_context):
        """Test writing a profile node to AGFS."""
        # Arrange
        profile_candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="profile",
            abstract="张三是一名软件工程师",
            overview="张三，软件工程师，喜欢 Python 编程",
            content="用户姓名：张三，职业：软件工程师，编程语言：Python",
            confidence=0.95,
        )

        policy = ProfilePolicy(fs=agfs)

        # Act - Plan the write
        plan = policy.plan(profile_candidate, sample_context)

        # Assert - WritePlan is correct
        assert plan.action in ["create", "merge"]
        assert sample_context.account_id in plan.target_uri
        assert sample_context.user_id in plan.target_uri
        assert plan.target_uri.endswith("/memories/profile")

        # Act - Write the node to AGFS
        node = ContextNode(
            uri=plan.target_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract=profile_candidate.abstract,
            overview=profile_candidate.overview,
            content=profile_candidate.content,
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(node, sample_context)

        # Assert - Node was written to AGFS
        assert agfs.exists(plan.target_uri, sample_context)

        # Assert - Can read back from AGFS
        read_node = agfs.read_node(plan.target_uri, sample_context)
        assert read_node.abstract == "张三是一名软件工程师"
        assert read_node.category == "profile"
        assert read_node.metadata.get("status") == NodeStatus.ACTIVE.value

    def test_entity_write_and_index(self, agfs, vector_index, sample_context):
        """Test writing an entity to AGFS and indexing it."""
        # Arrange
        entity_candidate = CandidateMemory(
            category="entity",
            owner_scope="user",
            routing_key="python",
            abstract="Python 编程语言",
            overview="张三喜欢使用 Python 进行编程",
            content="Python 是张三最喜欢的编程语言",
            confidence=0.9,
        )

        # Act - Write to AGFS
        uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/entities/python"
        node = ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="entity",
            level=0,
            owner_space="user_space",
            abstract=entity_candidate.abstract,
            overview=entity_candidate.overview,
            content=entity_candidate.content,
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(node, sample_context)

        # Act - Build index records
        index_records = build_index_records(node)

        # Assert - 3 records created (L0, L1, L2)
        assert len(index_records) == 3

        # Assert - All records have account_id filter
        for record in index_records:
            assert "account_id" in record.filters
            assert record.filters["account_id"] == sample_context.account_id
            assert "owner_space" in record.filters

        # Act - Upsert to vector index
        vector_index.upsert(index_records)

        # Assert - Records are searchable
        query = TypedQuery(
            text="Python",
            context_type=ContextType.MEMORY,
            categories=["entity"],
            top_k=10,
            account_id=sample_context.account_id,
            owner_space=None,
        )
        results = vector_index.search(query)

        # Assert - Found results in vector index
        assert len(results) > 0
        assert results[0].uri == uri


# ============================================================================
# Retrieval Path Tests
# ============================================================================

class TestRetrievalPathE2E:
    """End-to-end tests for the retrieval path."""

    def test_planner_categorizes_query(self, sample_context):
        """Test Planner correctly categorizes different query types."""
        planner = Planner()

        test_cases = [
            ("用户有什么偏好？", ContextType.MEMORY),
            ("如何进行代码审查？", ContextType.SKILL),
            ("查找 Python 相关文档", ContextType.RESOURCE),
            ("张三是谁？", ContextType.MEMORY),
        ]

        for query_text, expected_type in test_cases:
            result = planner.plan(query_text, sample_context)
            assert result.context_type == expected_type, \
                f"Query '{query_text}' should be {expected_type}, got {result.context_type}"

    def test_search_and_read_flow(self, agfs, vector_index, sample_context):
        """Test complete search and read flow with AGFS."""
        # Arrange - Write test data to AGFS
        test_uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/profile"
        test_node = ContextNode(
            uri=test_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="张三是软件工程师",
            overview="张三，软件工程师，擅长 Python",
            content="张三是一名软件工程师，主要使用 Python 编程",
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(test_node, sample_context)

        # Index the node
        index_records = build_index_records(test_node)
        vector_index.upsert(index_records)

        # Act - Initialize API with AGFS
        api = MemoryReadAPI(fs=agfs, vector_index=vector_index)

        # Act - Search
        search_results = api.search_memory(
            query="张三是做什么工作的？",
            ctx=sample_context,
            top_k=5,
        )

        # Assert - Search returns results
        assert len(search_results) > 0
        assert search_results[0]["uri"] == test_uri
        assert "张三" in search_results[0]["abstract"]
        assert "软件工程师" in search_results[0]["abstract"]

        # Act - Read L1 (overview) from AGFS
        overview = api.read_memory(test_uri, "L1", sample_context)

        # Assert
        assert "软件工程师" in overview
        assert "Python" in overview

        # Act - Read L2 (content) from AGFS
        content = api.read_memory(test_uri, "L2", sample_context)

        # Assert
        assert "编程" in content


# ============================================================================
# Multi-Tenant Isolation Tests
# ============================================================================

class TestMultiTenantIsolationE2E:
    """End-to-end tests for multi-tenant isolation with AGFS."""

    def test_cross_account_read_denied(self, agfs, vector_index, sample_context):
        """Test that cross-account reads are denied by AGFS."""
        # Arrange - Write data for account-a to AGFS
        account_a_ctx = RequestContext(
            account_id="account-a",
            user_id="user-a",
            agent_id="agent-1",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        account_a_uri = "ctx://account-a/users/user-a/memories/profile"
        account_a_node = ContextNode(
            uri=account_a_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Account A 的数据",
            overview="Account A 概述",
            content="Account A 完整内容",
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(account_a_node, account_a_ctx)

        # Act & Assert - account-b cannot read account-a's data from AGFS
        account_b_ctx = RequestContext(
            account_id="account-b",
            user_id="user-b",
            agent_id="agent-1",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        api = MemoryReadAPI(fs=agfs, vector_index=vector_index)

        with pytest.raises(AccessDeniedError) as exc_info:
            api.read_memory(account_a_uri, "L0", account_b_ctx)

        assert "account-a" in str(exc_info.value) or "account-b" in str(exc_info.value)

    def test_vector_index_isolation(self, vector_index):
        """Test that vector index respects account isolation."""
        # Arrange - Index records from two accounts
        account_a_records = [
            ContextNode(
                uri="ctx://account-a/users/user-a/memories/profile",
                context_type="MEMORY",
                category="profile",
                level=0,
                owner_space="user_space",
                abstract="Account A Profile",
                overview="Overview A",
                content="Content A",
                metadata={},
            )
        ]

        account_b_records = [
            ContextNode(
                uri="ctx://account-b/users/user-b/memories/profile",
                context_type="MEMORY",
                category="profile",
                level=0,
                owner_space="user_space",
                abstract="Account B Profile",
                overview="Overview B",
                content="Content B",
                metadata={},
            )
        ]

        # Index both accounts
        for node in account_a_records:
            vector_index.upsert(build_index_records(node))
        for node in account_b_records:
            vector_index.upsert(build_index_records(node))

        # Act - Search as account-a
        query_a = TypedQuery(
            text="Profile",
            context_type=ContextType.MEMORY,
            categories=None,
            top_k=10,
            account_id="account-a",
            owner_space=None,
        )
        results_a = vector_index.search(query_a)

        # Act - Search as account-b
        query_b = TypedQuery(
            text="Profile",
            context_type=ContextType.MEMORY,
            categories=None,
            top_k=10,
            account_id="account-b",
            owner_space=None,
        )
        results_b = vector_index.search(query_b)

        # Assert - Each account only sees their own data
        assert len(results_a) > 0
        assert len(results_b) > 0

        # Account A results only contain account-a URIs
        for hit in results_a:
            assert "account-a" in hit.uri
            assert "account-b" not in hit.uri

        # Account B results only contain account-b URIs
        for hit in results_b:
            assert "account-b" in hit.uri
            assert "account-a" not in hit.uri


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestFullPipelineE2E:
    """Complete end-to-end pipeline tests with AGFS."""

    def test_write_index_retrieve_pipeline(self, agfs, vector_index, sample_context):
        """Test the complete pipeline: write → index → retrieve with AGFS."""
        # Step 1: Write - Simulate candidate extraction and write to AGFS
        candidate = CandidateMemory(
            category="preference",
            owner_scope="user",
            routing_key="coding_style",
            abstract="用户偏好 PEP8 代码风格",
            overview="用户在编码时遵循 PEP8 规范，使用 VS Code 编辑器",
            content="用户明确表示喜欢 PEP8 代码风格，使用 VS Code 作为主要编辑器",
            confidence=0.9,
        )

        # Step 2: Plan and write to AGFS
        uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/preferences/coding_style"
        node = ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user_space",
            abstract=candidate.abstract,
            overview=candidate.overview,
            content=candidate.content,
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(node, sample_context)

        # Step 3: Index
        index_records = build_index_records(node)
        vector_index.upsert(index_records)

        # Verify indexing
        assert len(index_records) == 3
        for record in index_records:
            assert record.filters["account_id"] == sample_context.account_id

        # Step 4: Retrieve from AGFS
        api = MemoryReadAPI(fs=agfs, vector_index=vector_index)
        results = api.search_memory("用户的代码风格偏好是什么？", ctx=sample_context, top_k=5)

        # Verify retrieval
        assert len(results) > 0
        assert results[0]["uri"] == uri
        assert "PEP8" in results[0]["abstract"]

        # Step 5: Read full content from AGFS
        full_content = api.read_memory(uri, "L2", sample_context)
        assert "VS Code" in full_content

    def test_multiple_memories_isolated(self, agfs, vector_index):
        """Test that multiple users' memories are properly isolated in AGFS."""
        # Create two users in DIFFERENT accounts (true isolation)
        user1_ctx = RequestContext(
            account_id="account-alice",
            user_id="alice",
            agent_id="agent-1",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        user2_ctx = RequestContext(
            account_id="account-bob",
            user_id="bob",
            agent_id="agent-1",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        # Write different profiles to AGFS
        alice_uri = "ctx://account-alice/users/alice/memories/profile"
        bob_uri = "ctx://account-bob/users/bob/memories/profile"

        alice_node = ContextNode(
            uri=alice_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Alice 是数据科学家",
            overview="Alice 专注于机器学习",
            content="Alice 的完整信息",
            metadata={},
        )

        bob_node = ContextNode(
            uri=bob_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space",
            abstract="Bob 是后端工程师",
            overview="Bob 专注分布式系统",
            content="Bob 的完整信息",
            metadata={},
        )

        agfs.write_node(alice_node, user1_ctx)
        agfs.write_node(bob_node, user2_ctx)

        # Index both
        vector_index.upsert(build_index_records(alice_node))
        vector_index.upsert(build_index_records(bob_node))

        # Alice searches - should only see her own data from AGFS
        api = MemoryReadAPI(fs=agfs, vector_index=vector_index)
        alice_results = api.search_memory("用户简介", ctx=user1_ctx, top_k=10)

        assert all("account-alice" in r["uri"] for r in alice_results)
        assert not any("account-bob" in r["uri"] for r in alice_results)

        # Bob searches - should only see his own data from AGFS
        bob_results = api.search_memory("用户简介", ctx=user2_ctx, top_k=10)

        assert all("account-bob" in r["uri"] for r in bob_results)
        assert not any("account-alice" in r["uri"] for r in bob_results)

    def test_agfs_file_structure(self, agfs, sample_context):
        """Test that AGFS creates correct file structure."""
        # Write a node to AGFS
        uri = f"ctx://{sample_context.account_id}/users/{sample_context.user_id}/memories/test/test_node"
        node = ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="test",
            level=0,
            owner_space="user_space",
            abstract="Test abstract",
            overview="Test overview",
            content="Test content",
            metadata={"status": NodeStatus.ACTIVE.value},
        )
        agfs.write_node(node, sample_context)

        # Read back and verify structure
        read_node = agfs.read_node(uri, sample_context)

        assert read_node.abstract == "Test abstract"
        assert read_node.overview == "Test overview"
        assert read_node.content == "Test content"
        assert read_node.metadata.get("status") == NodeStatus.ACTIVE.value
