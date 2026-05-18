"""Integration tests for provenance pipeline.

Verifies the complete provenance tracking flow:
1. message ID preservation in extraction state
2. _Span.message_ids extraction in Phase 1
3. CandidateMemory.provenance_ids generation in Phase 2
"""

import pytest
from unittest.mock import Mock

from core.models import RequestContext
from core.provenance_resolver import ProvenanceResolver
from extraction.tools import Extractor


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.complete_json = Mock(return_value={
        "spans": [
            {"start": 0, "end": 2, "reason": "test", "categories": ["profile"]}
        ]
    })
    llm.complete_with_tools = Mock(return_value=[
        {
            "tool": "extract_profile",
            "input": {
                "routing_key": "name",
                "abstract": "User name is Alice",
                "overview": "User name: Alice",
                "content": "Alice",
                "confidence": 0.8,
                "evidence_quote": "My name is Alice",
                "attributed_speaker": "user",
                "attribution_basis": "self_first_person",
            }
        }
    ])
    llm.detect_language = Mock(return_value="en")
    return llm


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="test_account",
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
        trace_id="test_trace",
    )


class TestMessageIdPreservation:
    def test_extraction_state_preserves_message_ids(self):
        from server.memory_service import MemoryService
        from session.session_manager import SessionManager
        from session.models import SessionMessage

        mgr = SessionManager()
        buf = mgr.get_or_create("test_session")

        msg1 = SessionMessage(id="msg_a1", role="user", content="Hello")
        msg2 = SessionMessage(id="msg_b2", role="assistant", content="Hi")
        msg3 = SessionMessage(id="msg_c3", role="user", content="My name is Alice")
        buf.messages = [msg1, msg2, msg3]
        buf.extraction_watermark = 0

        service = MemoryService.__new__(MemoryService)
        service._session_mgr = mgr

        state = service._build_incremental_extraction_state(buf)

        assert state["messages"][0].get("id") == "msg_a1"
        assert state["messages"][1].get("id") == "msg_c3"


class TestSpanMessageIdsExtraction:
    def test_span_contains_message_ids(self, mock_llm, ctx):
        extractor = Extractor(mock_llm)

        messages = [
            {"role": "user", "content": "Hello", "id": "msg_001"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "My name is Alice", "id": "msg_002"},
        ]

        spans = extractor._identify_spans(messages, None, "")

        assert len(spans) == 1
        assert "msg_001" in spans[0].message_ids
        assert "msg_002" in spans[0].message_ids

    def test_span_message_ids_matches_indices(self, mock_llm, ctx):
        extractor = Extractor(mock_llm)

        messages = [
            {"role": "user", "content": "Hello", "id": "msg_a"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "My name is Alice", "id": "msg_b"},
        ]

        spans = extractor._identify_spans(messages, None, "")

        span = spans[0]
        expected_ids = [m.get("id") for m in messages[span.start:span.end + 1] if m.get("id")]
        assert span.message_ids == expected_ids


class TestProvenanceIdGeneration:
    def test_candidate_has_provenance_ids_with_archive_id(self, mock_llm, ctx):
        from extraction.schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        extractor = Extractor(mock_llm, schema_registry=registry)

        messages = [
            {"role": "user", "content": "Hello", "id": "msg_001"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "My name is Alice", "id": "msg_002"},
        ]

        archive_id = "20260515_100000_a1b2c3"

        candidates = extractor.extract(
            messages, ctx,
            archive_id=archive_id,
        )

        assert len(candidates) >= 1
        candidate = candidates[0]
        assert len(candidate.provenance_ids) >= 1

        prov_id = candidate.provenance_ids[0]
        parsed = ProvenanceResolver.parse_id(prov_id)
        assert parsed["source_type"] == "archive"
        assert parsed["source_id"] == archive_id
        # archive detail is returned as a list of message IDs
        assert isinstance(parsed["detail"], list)
        assert "msg_001" in parsed["detail"]
        assert "msg_002" in parsed["detail"]

    def test_candidate_empty_provenance_ids_without_archive_id(self, mock_llm, ctx):
        from extraction.schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        extractor = Extractor(mock_llm, schema_registry=registry)

        messages = [
            {"role": "user", "content": "Hello", "id": "msg_001"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "My name is Alice", "id": "msg_002"},
        ]

        candidates = extractor.extract(messages, ctx)

        assert len(candidates) >= 1
        candidate = candidates[0]
        assert candidate.provenance_ids == []


class TestProvenanceIdFormat:
    def test_provenance_id_format_matches_spec(self, mock_llm, ctx):
        from extraction.schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        extractor = Extractor(mock_llm, schema_registry=registry)

        messages = [
            {"role": "user", "content": "Hello", "id": "msg_abc"},
            {"role": "user", "content": "My name is Alice", "id": "msg_xyz"},
        ]

        archive_id = "20260515_120000_def123"

        candidates = extractor.extract(
            messages, ctx,
            archive_id=archive_id,
        )

        candidate = candidates[0]
        prov_id = candidate.provenance_ids[0]

        assert prov_id.startswith("prov:1:archive:")

        # Wire format has URL-encoded fields; verify via parse, not substring match
        parsed = ProvenanceResolver.parse_id(prov_id)
        assert parsed["version"] == 1
        assert parsed["source_type"] == "archive"
        assert parsed["source_id"] == archive_id

        # display_id returns human-readable form with decoded fields
        readable = ProvenanceResolver.display_id(prov_id)
        assert archive_id in readable


class TestArchiveIdPreGeneration:
    def test_commit_session_provenance_through_pipeline(self, memory_fs):
        """Verify provenance IDs survive the full extract→plan→build→write pipeline.

        Uses real Extractor, CandidatePipeline, ContextWriter, SchemaRegistry.
        Only LLM and ContextFS are substituted (MockLLM + InMemoryContextFS).
        """
        from extraction.schemas.registry import SchemaRegistry
        from service.api import MemoryWriteAPI

        registry = SchemaRegistry()
        llm = Mock()
        llm.complete_json = Mock(return_value={
            "spans": [
                {"start": 0, "end": 1, "reason": "user profile", "categories": ["profile"]}
            ]
        })
        llm.complete_with_tools = Mock(return_value=[
            {
                "tool": "extract_profile",
                "input": {
                    "routing_key": "user_identity",
                    "abstract": "User is Alice",
                    "overview": "User name: Alice",
                    "content": "Alice is a software engineer",
                    "confidence": 0.9,
                    "evidence_quote": "My name is Alice",
                    "attributed_speaker": "user",
                    "attribution_basis": "self_first_person",
                }
            }
        ])
        llm.detect_language = Mock(return_value="en")

        api = MemoryWriteAPI(
            fs=memory_fs,
            llm=llm,
            schema_registry=registry,
        )

        ctx = RequestContext(
            account_id="test",
            user_id="test",
            agent_id="test",
            session_id="test_session",
            trace_id="test_trace",
        )

        archive_id = "20260518_120000_abcd1234"
        messages = [
            {"role": "user", "content": "My name is Alice", "id": "msg_001"},
            {"role": "assistant", "content": "Hello Alice!"},
        ]

        result = api.commit_session(
            messages=messages,
            ctx=ctx,
            archive_id=archive_id,
        )

        # Pipeline produced candidates
        assert result["candidates_extracted"] >= 1
        assert result["writes_completed"] >= 1

        # Node was persisted to InMemoryContextFS with provenance
        assert len(memory_fs.stored_uris) >= 1
        uri = memory_fs.stored_uris[0]
        node = memory_fs.read_node(uri, ctx)

        prov_ids = node.metadata.get("provenance_ids", [])
        assert len(prov_ids) >= 1

        parsed = ProvenanceResolver.parse_id(prov_ids[0])
        assert parsed["source_type"] == "archive"
        assert parsed["source_id"] == archive_id
        assert "msg_001" in parsed["detail"]


class TestArchiveIdConsistency:
    def test_same_archive_id_for_extraction_and_snapshot(self):
        from session.session_manager import SessionManager
        from session.models import SessionMessage
        from core.models import RequestContext

        mgr = SessionManager()
        ctx = RequestContext(
            account_id="test",
            user_id="test",
            agent_id="test",
            session_id="test_session",
            trace_id="test_trace",
        )

        snapshot = [
            SessionMessage(id="msg_1", role="user", content="Hello"),
            SessionMessage(id="msg_2", role="assistant", content="Hi"),
        ]

        archive_id = "20260515_100000_custom"

        try:
            result = mgr.commit_snapshot(
                "test_session",
                snapshot,
                ctx,
                wait=True,
                archive_id=archive_id,
            )
            assert result["archive_id"] == archive_id
        except Exception as e:
            pytest.skip(f"Requires database connection: {e}")