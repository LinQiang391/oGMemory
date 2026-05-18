"""Unit tests for ProvenanceResolver."""

import pytest
from urllib.parse import quote
from core.provenance_resolver import ProvenanceResolver, VALID_SOURCE_TYPES


class TestBuildId:
    def test_build_archive_with_detail(self):
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", "msg_a3f8,msg_b7c1"
        )
        # source_id has no special chars (unchanged by encoding);
        # detail comma is encoded to %2C
        assert result == "prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1"

    def test_build_archive_without_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_archive_empty_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2", "")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_memory_with_uri(self):
        # URI colons and slashes are URL-encoded
        result = ProvenanceResolver.build_id(
            "memory", "ctx://acme/users/alice/memories/entities/rust"
        )
        enc = quote("ctx://acme/users/alice/memories/entities/rust", safe="")
        assert result == f"prov:1:memory:{enc}:"

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

    def test_build_detail_with_colons(self):
        # detail containing colons is encoded safely
        result = ProvenanceResolver.build_id(
            "memory", "ctx://acme/path", "field:subfield"
        )
        enc_id = quote("ctx://acme/path", safe="")
        enc_detail = quote("field:subfield", safe="")
        assert result == f"prov:1:memory:{enc_id}:{enc_detail}"


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

    def test_parse_archive_with_encoded_comma(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1"
        )
        assert result["source_id"] == "20260513_100000_a1b2"
        assert result["detail"] == "msg_a3f8,msg_b7c1"

    def test_parse_memory_with_encoded_uri(self):
        enc = quote("ctx://acme/users/alice/memories/entities/rust", safe="")
        result = ProvenanceResolver.parse_id(f"prov:1:memory:{enc}:")
        assert result == {
            "version": 1,
            "source_type": "memory",
            "source_id": "ctx://acme/users/alice/memories/entities/rust",
            "detail": "",
        }

    def test_parse_memory_with_encoded_uri_and_detail(self):
        enc_id = quote("ctx://acme/users/alice/memories/entities/rust", safe="")
        enc_detail = quote("field_x", safe="")
        result = ProvenanceResolver.parse_id(f"prov:1:memory:{enc_id}:{enc_detail}")
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

    def test_roundtrip_memory_uri_with_detail(self):
        original = ("memory", "ctx://acme/path", "field:with:colons")
        built = ProvenanceResolver.build_id(*original)
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_id"] == "ctx://acme/path"
        assert parsed["detail"] == "field:with:colons"


class TestDisplayId:
    def test_display_decodes_uri(self):
        pid = ProvenanceResolver.build_id(
            "memory", "ctx://acme/users/memories/entities/rust"
        )
        readable = ProvenanceResolver.display_id(pid)
        assert readable == "prov:1:memory:ctx://acme/users/memories/entities/rust:"

    def test_display_decodes_comma_detail(self):
        pid = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", "msg_a3f8,msg_b7c1"
        )
        readable = ProvenanceResolver.display_id(pid)
        assert readable == "prov:1:archive:20260513_100000_a1b2:msg_a3f8,msg_b7c1"

    def test_display_decodes_colon_in_detail(self):
        pid = ProvenanceResolver.build_id(
            "memory", "ctx://acme/path", "field:subfield"
        )
        readable = ProvenanceResolver.display_id(pid)
        assert readable == "prov:1:memory:ctx://acme/path:field:subfield"

    def test_display_plain_values(self):
        pid = ProvenanceResolver.build_id("dream", "20260513_dream_001")
        readable = ProvenanceResolver.display_id(pid)
        assert readable == "prov:1:dream:20260513_dream_001:"

    def test_display_invalid_raises(self):
        with pytest.raises(ValueError):
            ProvenanceResolver.display_id("not_a_provenance_id")


class TestValidateSourceType:
    def test_valid_source_types(self):
        for source_type in VALID_SOURCE_TYPES:
            ProvenanceResolver.validate_source_type(source_type)

    def test_invalid_source_type(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.validate_source_type("unknown")
        assert "Invalid source_type" in str(exc_info.value)
