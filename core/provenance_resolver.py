"""Provenance ID construction and parsing.

Provenance IDs trace any memory or relationship back to its original source.

Format:
    prov:{version}:{source_type}:{urlencoded(source_id)}:{urlencoded(detail)}

source_id and detail are URL-encoded so that `:` can serve as a safe delimiter.
This allows source_id to contain colons (e.g., memory URIs like ctx://...)
and detail to contain colons (e.g., field:subfield) without ambiguity.

Wire format examples:
    prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1
    prov:1:archive:20260513_100000_a1b2:
    prov:1:memory:ctx%3A%2F%2Facme%2Fusers%2Fmemories%2Fentities%2Frust:
    prov:1:dream:20260513_dream_001:

Human-readable form (for logs/debug only, never stored or parsed):
    prov:1:archive:20260513_100000_a1b2:msg_a3f8,msg_b7c1
    prov:1:memory:ctx://acme/users/memories/entities/rust:
"""

from __future__ import annotations

from urllib.parse import quote, unquote

VALID_SOURCE_TYPES = {"archive", "memory", "dream", "graph"}


class ProvenanceResolver:
    """Build and parse Provenance IDs with URL-encoded fields."""

    @staticmethod
    def validate_source_type(source_type: str) -> None:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {source_type}. Must be one of {VALID_SOURCE_TYPES}"
            )

    @staticmethod
    def build_id(source_type: str, source_id: str, detail: str = "") -> str:
        """Build a provenance ID with URL-encoded source_id and detail.

        Encoding ensures `:` is a safe delimiter even when source_id
        contains colons (memory URIs) or detail contains special chars.
        """
        ProvenanceResolver.validate_source_type(source_type)
        # Encode so `:` never appears inside source_id or detail
        enc_id = quote(source_id, safe="")
        enc_detail = quote(detail, safe="")
        return f"prov:1:{source_type}:{enc_id}:{enc_detail}"

    @staticmethod
    def parse_id(pid: str) -> dict:
        """Parse a provenance ID back into its components.

        Decodes the URL-encoded source_id and detail fields.
        """
        if not pid.startswith("prov:"):
            raise ValueError(f"Invalid provenance ID: {pid}")

        parts = pid.split(":")
        if len(parts) < 5:
            raise ValueError(f"Invalid provenance ID format: {pid}")

        version_str = parts[1]
        source_type = parts[2]
        # With encoding, `:` never appears inside source_id or detail,
        # so split(":") always yields exactly 5 parts.
        # parts[3] = encoded source_id, parts[4] = encoded detail
        enc_source_id = parts[3]
        enc_detail = parts[4]

        ProvenanceResolver.validate_source_type(source_type)

        try:
            version = int(version_str)
        except ValueError:
            raise ValueError(f"Invalid version in provenance ID: {version_str}")

        return {
            "version": version,
            "source_type": source_type,
            "source_id": unquote(enc_source_id),
            "detail": unquote(enc_detail),
        }

    @staticmethod
    def display_id(pid: str) -> str:
        """Return a human-readable provenance ID for logs and debugging.

        Decodes URL-encoded fields back to their original form so colons
        and other special characters are visible.  NOT suitable for storage
        or parsing — use the raw `pid` for that.
        """
        parsed = ProvenanceResolver.parse_id(pid)
        return (
            f"prov:{parsed['version']}:{parsed['source_type']}"
            f":{parsed['source_id']}:{parsed['detail']}"
        )
