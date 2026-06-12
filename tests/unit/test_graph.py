"""Unit tests for graph models and memory graph store."""

import pytest

from core.graph_enums import (
    ExtractionStatus,
    GraphEdgeType,
    GraphNodeType,
    NodeStatus,
    RawDataType,
)
from core.graph_models import (
    ExtractedNode,
    GraphEdge,
    GraphStats,
    MemoryNode,
    RawNode,
    SupportingChain,
    SubgraphResult,
    create_extracted_node,
    create_graph_edge,
    create_memory_node,
    create_raw_node,
    now_iso,
)
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import GraphService


@pytest.fixture
def ctx():
    """Create a test request context."""
    return RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )


@pytest.fixture
def store():
    """Create an in-memory graph store."""
    return MemoryGraphStore()


@pytest.fixture
def service(store):
    """Create a graph service."""
    return GraphService(store)


class TestGraphModels:
    """Test graph model creation and properties."""

    def test_create_raw_node(self):
        """Test creating a raw node."""
        node = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Write a story",
            account_id="acc1",
        )

        assert node.node_type == GraphNodeType.RAW
        assert node.raw_type == RawDataType.PROMPT
        assert node.content == "Write a story"
        assert node.account_id == "acc1"
        assert node.node_id is not None
        assert node.uri.startswith("raw:prompt:")

    def test_raw_node_uri(self):
        """Test raw node URI generation."""
        node = create_raw_node(
            raw_type=RawDataType.IMAGE,
            content="base64_image_data",
            account_id="acc1",
        )

        assert "raw:image:" in node.uri

    def test_create_extracted_node(self):
        """Test creating an extracted node."""
        node = create_extracted_node(
            source_raw_id="raw123",
            extractor_type="prompt_analyzer",
            extraction_schema="v1",
            account_id="acc1",
            extracted_content={"intent": "creative_writing"},
        )

        assert node.node_type == GraphNodeType.EXTRACTED
        assert node.source_raw_id == "raw123"
        assert node.extractor_type == "prompt_analyzer"
        assert node.extracted_content["intent"] == "creative_writing"
        assert node.extraction_status == ExtractionStatus.PENDING

    def test_extracted_node_mark_completed(self):
        """Test marking extraction as completed."""
        node = create_extracted_node(
            source_raw_id="raw123",
            extractor_type="test",
            extraction_schema="v1",
            account_id="acc1",
        )

        node.mark_completed(
            extracted_content={"result": "test"},
            confidence=0.9,
        )

        assert node.extraction_status == ExtractionStatus.COMPLETED
        assert node.extracted_content["result"] == "test"
        assert node.confidence == 0.9

    def test_create_memory_node(self):
        """Test creating a memory node."""
        node = create_memory_node(
            summary="User wants to write a story",
            account_id="acc1",
            supporting_node_ids=["extracted123"],
            topic="writing",
            tags=["creative", "fiction"],
        )

        assert node.node_type == GraphNodeType.MEMORY
        assert node.summary == "User wants to write a story"
        assert len(node.supporting_node_ids) == 1
        assert "writing" in node.tags or "writing" == node.topic

    def test_memory_node_add_remove_support(self):
        """Test adding and removing supporting nodes."""
        node = create_memory_node(
            summary="Test memory",
            account_id="acc1",
        )

        node.add_support("node1")
        node.add_support("node2")
        assert len(node.supporting_node_ids) == 2

        node.remove_support("node1")
        assert len(node.supporting_node_ids) == 1
        assert "node1" not in node.supporting_node_ids

    def test_create_graph_edge(self):
        """Test creating a graph edge."""
        edge = create_graph_edge(
            source_id="node1",
            target_id="node2",
            edge_type=GraphEdgeType.SUPPORTS,
            account_id="acc1",
            weight=0.8,
        )

        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.edge_type == GraphEdgeType.SUPPORTS
        assert edge.weight == 0.8
        assert edge.edge_id is not None


class TestMemoryGraphStore:
    """Test in-memory graph store operations."""

    def test_create_and_get_node(self, store, ctx):
        """Test creating and retrieving a node."""
        node = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test content",
            account_id=ctx.account_id,
        )

        created = store.create_node(node, ctx)
        retrieved = store.get_node(created.node_id, ctx)

        assert retrieved is not None
        assert retrieved.node_id == created.node_id
        assert retrieved.content == "Test content"

    def test_update_node(self, store, ctx):
        """Test updating a node."""
        node = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Original",
            account_id=ctx.account_id,
        )
        store.create_node(node, ctx)

        node.content = "Updated"
        updated = store.update_node(node, ctx)

        assert updated.content == "Updated"

    def test_delete_node(self, store, ctx):
        """Test deleting a node."""
        node = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="To delete",
            account_id=ctx.account_id,
        )
        store.create_node(node, ctx)
        node_id = node.node_id

        store.delete_node(node_id, ctx)
        retrieved = store.get_node(node_id, ctx)

        assert retrieved is None

    def test_create_and_get_edge(self, store, ctx):
        """Test creating and retrieving an edge."""
        raw_node = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            account_id=ctx.account_id,
        )
        store.create_node(raw_node, ctx)

        extracted = create_extracted_node(
            source_raw_id=raw_node.node_id,
            extractor_type="test",
            extraction_schema="v1",
            account_id=ctx.account_id,
        )
        store.create_node(extracted, ctx)

        edge = create_graph_edge(
            source_id=extracted.node_id,
            target_id=raw_node.node_id,
            edge_type=GraphEdgeType.EXTRACTED_FROM,
            account_id=ctx.account_id,
        )
        created_edge = store.create_edge(edge, ctx)

        retrieved_edge = store.get_edge(created_edge.edge_id, ctx)
        assert retrieved_edge is not None
        assert retrieved_edge.source_id == extracted.node_id

    def test_get_neighbors(self, store, ctx):
        """Test getting neighboring nodes."""
        raw = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            account_id=ctx.account_id,
        )
        store.create_node(raw, ctx)

        extracted = create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="test",
            extraction_schema="v1",
            account_id=ctx.account_id,
        )
        store.create_node(extracted, ctx)

        # Create edge: extracted -> raw
        edge = create_graph_edge(
            source_id=extracted.node_id,
            target_id=raw.node_id,
            edge_type=GraphEdgeType.EXTRACTED_FROM,
            account_id=ctx.account_id,
        )
        store.create_edge(edge, ctx)

        # Get neighbors of raw node (incoming)
        neighbors = store.get_neighbors(raw.node_id, ctx, direction="incoming")
        assert len(neighbors) == 1
        assert neighbors[0][0].node_id == extracted.node_id
        assert neighbors[0][2] == "incoming"

    def test_traverse(self, store, ctx):
        """Test BFS traversal."""
        # Create: raw -> extracted -> memory
        raw = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            account_id=ctx.account_id,
        )
        store.create_node(raw, ctx)

        extracted = create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="test",
            extraction_schema="v1",
            account_id=ctx.account_id,
        )
        store.create_node(extracted, ctx)

        memory = create_memory_node(
            summary="Test memory",
            account_id=ctx.account_id,
            supporting_node_ids=[extracted.node_id],
        )
        store.create_node(memory, ctx)

        # Create edges: raw <- extracted <- memory (outgoing from extracted)
        store.create_edge(
            create_graph_edge(
                source_id=extracted.node_id,
                target_id=raw.node_id,
                edge_type=GraphEdgeType.EXTRACTED_FROM,
                account_id=ctx.account_id,
            ),
            ctx,
        )
        store.create_edge(
            create_graph_edge(
                source_id=extracted.node_id,
                target_id=memory.node_id,
                edge_type=GraphEdgeType.SUPPORTS,
                account_id=ctx.account_id,
            ),
            ctx,
        )

        # Traverse from extracted (it has outgoing edges to raw and memory)
        result = store.traverse(extracted.node_id, depth=2, ctx=ctx)

        assert len(result.nodes) == 3
        assert len(result.edges) == 2

    def test_pagerank_update(self, store, ctx):
        """Test PageRank updates on edge creation."""
        raw = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            account_id=ctx.account_id,
        )
        store.create_node(raw, ctx)

        extracted = create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="test",
            extraction_schema="v1",
            account_id=ctx.account_id,
        )
        store.create_node(extracted, ctx)

        # Create edge: extracted -> raw
        store.create_edge(
            create_graph_edge(
                source_id=extracted.node_id,
                target_id=raw.node_id,
                edge_type=GraphEdgeType.EXTRACTED_FROM,
                account_id=ctx.account_id,
                weight=1.0,
            ),
            ctx,
        )

        # Check pagerank was updated
        pr = store.get_node_pagerank(raw.node_id, ctx)
        assert pr > 0

    def test_search_memories(self, store, ctx):
        """Test memory search."""
        memory1 = create_memory_node(
            summary="Dragon story",
            account_id=ctx.account_id,
            topic="fantasy",
        )
        store.create_node(memory1, ctx)

        memory2 = create_memory_node(
            summary="Code tutorial",
            account_id=ctx.account_id,
            topic="programming",
        )
        store.create_node(memory2, ctx)

        results = store.search_memories("dragon", ctx)
        assert len(results) == 1
        assert "dragon" in results[0].summary.lower()

    def test_graph_stats(self, store, ctx):
        """Test graph statistics."""
        raw = create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            account_id=ctx.account_id,
        )
        store.create_node(raw, ctx)

        memory = create_memory_node(
            summary="Test memory",
            account_id=ctx.account_id,
        )
        store.create_node(memory, ctx)

        stats = store.get_graph_stats(ctx)

        assert stats.total_nodes == 2
        assert stats.raw_nodes == 1
        assert stats.memory_nodes == 1


class TestGraphService:
    """Test graph service high-level operations."""

    def test_create_raw_node(self, service, ctx):
        """Test creating a raw node via service."""
        node = service.create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test prompt",
            ctx=ctx,
        )

        assert isinstance(node, RawNode)
        assert node.content == "Test prompt"

    def test_create_extracted_node_with_edge(self, service, ctx):
        """Test creating extracted node auto-creates edge."""
        raw = service.create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            ctx=ctx,
        )

        extracted = service.create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="test",
            extraction_schema="v1",
            ctx=ctx,
            extracted_content={"result": "test"},
        )

        assert isinstance(extracted, ExtractedNode)

        # Check edge was created
        neighbors = service.get_neighbors(raw.node_id, ctx, direction="incoming")
        assert len(neighbors) == 1
        assert neighbors[0][0].node_id == extracted.node_id

    def test_create_memory_with_supports(self, service, ctx):
        """Test creating memory with supporting nodes."""
        raw = service.create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            ctx=ctx,
        )

        extracted = service.create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="test",
            extraction_schema="v1",
            ctx=ctx,
        )

        memory = service.create_memory_node(
            summary="Test memory",
            ctx=ctx,
            supporting_node_ids=[extracted.node_id],
        )

        assert isinstance(memory, MemoryNode)
        assert len(memory.supporting_node_ids) == 1

        # Check supports edge was created
        neighbors = service.get_neighbors(extracted.node_id, ctx, direction="outgoing")
        memory_neighbors = [n for n, _, _ in neighbors if n.node_id == memory.node_id]
        assert len(memory_neighbors) == 1

    def test_add_supporting_node(self, service, ctx):
        """Test adding a supporting node to memory."""
        raw = service.create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Test",
            ctx=ctx,
        )

        memory = service.create_memory_node(
            summary="Test memory",
            ctx=ctx,
        )

        # Initially no supports
        chain = service.get_supporting_chain(memory.node_id, ctx)
        assert len(chain.raw_nodes) == 0

        # Add raw as supporting
        service.add_supporting_node(memory.node_id, raw.node_id, ctx)

        # Now should have one supporting node
        chain = service.get_supporting_chain(memory.node_id, ctx)
        assert len(chain.raw_nodes) == 1

    def test_get_supporting_chain(self, service, ctx):
        """Test getting supporting chain for memory."""
        raw = service.create_raw_node(
            raw_type=RawDataType.PROMPT,
            content="Write a dragon story",
            ctx=ctx,
        )

        extracted = service.create_extracted_node(
            source_raw_id=raw.node_id,
            extractor_type="prompt_analyzer",
            extraction_schema="v1",
            ctx=ctx,
            extracted_content={"intent": "creative_writing"},
        )

        memory = service.create_memory_node(
            summary="User wants to write a dragon story",
            ctx=ctx,
            supporting_node_ids=[extracted.node_id],
        )

        chain = service.get_supporting_chain(memory.node_id, ctx)

        assert isinstance(chain.memory, MemoryNode)
        assert len(chain.raw_nodes) == 1
        assert chain.raw_nodes[0].node_id == raw.node_id
        assert len(chain.extracted_nodes) == 1
        assert chain.extracted_nodes[0].extracted_content["intent"] == "creative_writing"

    def test_search_memories(self, service, ctx):
        """Test searching memories."""
        service.create_memory_node(
            summary="How to cook pasta",
            ctx=ctx,
            tags=["cooking", "food"],
        )
        service.create_memory_node(
            summary="Pasta recipe with tomatoes",
            ctx=ctx,
            tags=["cooking", "italian"],
        )

        results = service.search_memories("pasta", ctx)
        assert len(results) == 2

        # Search with tags filter - both have "cooking" tag
        results = service.search_memories("pasta", ctx, tags=["cooking"])
        assert len(results) == 2

    def test_update_memory(self, service, ctx):
        """Test updating memory content."""
        memory = service.create_memory_node(
            summary="Original summary",
            ctx=ctx,
        )

        updated = service.update_memory(
            memory.node_id,
            ctx=ctx,
            summary="Updated summary",
            topic="new_topic",
        )

        assert updated.summary == "Updated summary"
        assert updated.topic == "new_topic"


class TestPromptExtractor:
    """Test prompt extraction functionality."""

    def test_intent_extraction(self):
        """Test intent extraction patterns."""
        from extraction.operators.prompt_extractor import PromptExtractor

        extractor = PromptExtractor()

        assert extractor._extract_intent("write a story about dragons") == "creative_writing"
        assert extractor._extract_intent("write a function to sort") == "code_generation"
        assert extractor._extract_intent("analyze this data") == "analysis"
        assert extractor._extract_intent("what is the meaning of life") == "question"

    def test_subject_extraction(self):
        """Test subject extraction."""
        from extraction.operators.prompt_extractor import PromptExtractor

        extractor = PromptExtractor()

        subject = extractor._extract_subject(
            "write a story about dragons",
            "Write a story about dragons",
        )
        assert "dragon" in subject.lower()

    def test_style_extraction(self):
        """Test style extraction."""
        from extraction.operators.prompt_extractor import PromptExtractor

        extractor = PromptExtractor()

        styles = extractor._extract_style("write a formal letter")
        assert "formal" in styles

        styles = extractor._extract_style("write a creative poem")
        assert "creative" in styles

    def test_format_extraction(self):
        """Test format extraction."""
        from extraction.operators.prompt_extractor import PromptExtractor

        extractor = PromptExtractor()

        assert extractor._extract_format("write a haiku") == "haiku"
        assert extractor._extract_format("make a list of steps") == "list"
        assert extractor._extract_format("write some code") == "code"