"""Unit tests for ProvenanceResolver."""

import pytest
from core.provenance_resolver import ProvenanceResolver, VALID_SOURCE_TYPES


class TestBuildId:
    def test_build_archive_with_detail(self):
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", "msg_a3f8,msg_b7c1"
        )
        assert result == "prov:1:archive:20260513_100000_a1b2:msg_a3f8,msg_b7c1"

    def test_build_archive_without_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_archive_empty_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2", "")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_memory_with_uri(self):
        result = ProvenanceResolver.build_id(
            "memory", "ctx://acme/users/alice/memories/entities/rust"
        )
        assert result == "prov:1:memory:ctx://acme/users/alice/memories/entities/rust:"

    def test_build_dream(self):
        result = ProvenanceResolver.build_id("dream", "20260513_dream_001")
        assert result == "prov:1:dream:20260513_dream_001:"

    def test_build_graph(self):
        result = ProvenanceResolver.build_id("graph", "node_123")
        assert result == "prov:1:graph:node_123:"

    def test_build_invalid_source_type(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.build_id("invalid", "some_id")
        assert "Invalid source_type" in str(exc_info.value)


class TestParseId:
    def test_parse_archive_with_detail(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:msg_a3f8"
        )
        assert result == {
            "version": 1,
            "source_type": "archive",
            "source_id": "20260513_100000_a1b2",
            "detail": "msg_a3f8",
        }

    def test_parse_archive_without_detail(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:"
        )
        assert result == {
            "version": 1,
            "source_type": "archive",
            "source_id": "20260513_100000_a1b2",
            "detail": "",
        }

    def test_parse_memory_with_uri(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:memory:ctx://acme/users/alice/memories/entities/rust:"
        )
        assert result == {
            "version": 1,
            "source_type": "memory",
            "source_id": "ctx://acme/users/alice/memories/entities/rust",
            "detail": "",
        }

    def test_parse_memory_with_uri_and_detail(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:memory:ctx://acme/users/alice/memories/entities/rust:field_x"
        )
        assert result == {
            "version": 1,
            "source_type": "memory",
            "source_id": "ctx://acme/users/alice/memories/entities/rust",
            "detail": "field_x",
        }

    def test_parse_dream(self):
        result = ProvenanceResolver.parse_id("prov:1:dream:20260513_dream_001:")
        assert result == {
            "version": 1,
            "source_type": "dream",
            "source_id": "20260513_dream_001",
            "detail": "",
        }

    def test_parse_invalid_provenance_prefix(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.parse_id("invalid:1:archive:id:")
        assert "Invalid provenance ID" in str(exc_info.value)

    def test_parse_invalid_source_type(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.parse_id("prov:1:invalid:id:")
        assert "Invalid source_type" in str(exc_info.value)

    def test_parse_invalid_version(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.parse_id("prov:abc:archive:id:")
        assert "Invalid version" in str(exc_info.value)

    def test_parse_too_few_parts(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.parse_id("prov:1:archive")
        assert "Invalid provenance ID format" in str(exc_info.value)


class TestRoundtrip:
    def test_roundtrip_archive_with_detail(self):
        original = ("archive", "20260513_100000_a1b2", "msg_a3f8,msg_b7c1")
        built = ProvenanceResolver.build_id(*original)
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_type"] == original[0]
        assert parsed["source_id"] == original[1]
        assert parsed["detail"] == original[2]

    def test_roundtrip_archive_no_detail(self):
        original = ("archive", "20260513_100000_a1b2", "")
        built = ProvenanceResolver.build_id(*original)
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_type"] == original[0]
        assert parsed["source_id"] == original[1]
        assert parsed["detail"] == original[2]

    def test_roundtrip_memory_uri(self):
        original = ("memory", "ctx://acme/users/alice/memories/entities/rust", "")
        built = ProvenanceResolver.build_id(*original)
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_type"] == original[0]
        assert parsed["source_id"] == original[1]
        assert parsed["detail"] == original[2]


class TestValidateSourceType:
    def test_valid_source_types(self):
        for source_type in VALID_SOURCE_TYPES:
            ProvenanceResolver.validate_source_type(source_type)

    def test_invalid_source_type(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.validate_source_type("unknown")
        assert "Invalid source_type" in str(exc_info.value)