"""Unit tests for ProvenanceResolver."""

import pytest
from urllib.parse import quote
from core.provenance_resolver import ProvenanceResolver, VALID_SOURCE_TYPES


class TestBuildId:
    def test_build_archive_with_list_detail(self):
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", ["msg_a3f8", "msg_b7c1"]
        )
        # list is joined with comma, then comma is URL-encoded to %2C
        assert result == "prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1"

    def test_build_archive_with_single_msg_id(self):
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", ["msg_a3f8"]
        )
        assert result == "prov:1:archive:20260513_100000_a1b2:msg_a3f8"

    def test_build_archive_with_empty_list(self):
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", []
        )
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_archive_without_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_archive_empty_detail(self):
        result = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2", "")
        assert result == "prov:1:archive:20260513_100000_a1b2:"

    def test_build_archive_with_string_detail_rejected(self):
        # archive with comma-string is ambiguous; use list instead
        result = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", "msg_a3f8,msg_b7c1"
        )
        # Still works (string detail is accepted for backwards compat),
        # but parse_id won't split it into a list
        assert result == "prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1"

    def test_build_non_archive_rejects_list_detail(self):
        with pytest.raises(ValueError, match="List detail is only supported for archive"):
            ProvenanceResolver.build_id("memory", "ctx://path", ["a", "b"])

    def test_build_memory_with_uri(self):
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
        result = ProvenanceResolver.build_id(
            "memory", "ctx://acme/path", "field:subfield"
        )
        enc_id = quote("ctx://acme/path", safe="")
        enc_detail = quote("field:subfield", safe="")
        assert result == f"prov:1:memory:{enc_id}:{enc_detail}"


class TestParseId:
    def test_parse_archive_returns_list(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:msg_a3f8"
        )
        assert result == {
            "version": 1,
            "source_type": "archive",
            "source_id": "20260513_100000_a1b2",
            "detail": ["msg_a3f8"],
        }

    def test_parse_archive_without_detail_returns_empty_list(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:"
        )
        assert result == {
            "version": 1,
            "source_type": "archive",
            "source_id": "20260513_100000_a1b2",
            "detail": [],
        }

    def test_parse_archive_with_encoded_comma_returns_list(self):
        result = ProvenanceResolver.parse_id(
            "prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1"
        )
        assert result["source_id"] == "20260513_100000_a1b2"
        assert result["detail"] == ["msg_a3f8", "msg_b7c1"]

    def test_parse_memory_returns_string_detail(self):
        enc = quote("ctx://acme/users/alice/memories/entities/rust", safe="")
        result = ProvenanceResolver.parse_id(f"prov:1:memory:{enc}:")
        assert result == {
            "version": 1,
            "source_type": "memory",
            "source_id": "ctx://acme/users/alice/memories/entities/rust",
            "detail": "",
        }

    def test_parse_memory_with_detail_returns_string(self):
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
    def test_roundtrip_archive_with_list_detail(self):
        built = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", ["msg_a3f8", "msg_b7c1"]
        )
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_type"] == "archive"
        assert parsed["source_id"] == "20260513_100000_a1b2"
        assert parsed["detail"] == ["msg_a3f8", "msg_b7c1"]

    def test_roundtrip_archive_no_detail(self):
        built = ProvenanceResolver.build_id("archive", "20260513_100000_a1b2")
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["detail"] == []

    def test_roundtrip_memory_uri(self):
        built = ProvenanceResolver.build_id(
            "memory", "ctx://acme/users/alice/memories/entities/rust"
        )
        parsed = ProvenanceResolver.parse_id(built)
        assert parsed["source_id"] == "ctx://acme/users/alice/memories/entities/rust"
        assert parsed["detail"] == ""

    def test_roundtrip_memory_uri_with_detail(self):
        built = ProvenanceResolver.build_id(
            "memory", "ctx://acme/path", "field:with:colons"
        )
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

    def test_display_shows_list_as_comma_separated(self):
        pid = ProvenanceResolver.build_id(
            "archive", "20260513_100000_a1b2", ["msg_a3f8", "msg_b7c1"]
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


class TestValidateInput:
    def test_valid_source_types(self):
        for source_type in VALID_SOURCE_TYPES:
            ProvenanceResolver.validate_input(source_type)

    def test_invalid_source_type(self):
        with pytest.raises(ValueError) as exc_info:
            ProvenanceResolver.validate_input("unknown")
        assert "Invalid source_type" in str(exc_info.value)

    def test_non_archive_rejects_list_detail(self):
        with pytest.raises(ValueError, match="List detail is only supported for archive"):
            ProvenanceResolver.validate_input("memory", ["a", "b"])
