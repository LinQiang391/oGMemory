"""Provenance ID construction and parsing.

Provenance IDs trace any memory or relationship back to its original source.

Format:
    prov:{version}:{source_type}:{source_id}:{detail}

Example:
    prov:1:archive:20260513_100000_a1b2:msg_a3f8,msg_b7c1
    prov:1:archive:20260513_100000_a1b2:
    prov:1:memory:ctx://acme/users/alice/memories/entities/rust:
    prov:1:dream:20260513_dream_001:
"""

from __future__ import annotations

VALID_SOURCE_TYPES = {"archive", "memory", "dream", "graph"}
KNOWN_DETAIL_PREFIXES = {"msgs", "v"}


class ProvenanceResolver:
    """Build and parse Provenance IDs with intelligent URI parsing."""

    @staticmethod
    def validate_source_type(source_type: str) -> None:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {source_type}. Must be one of {VALID_SOURCE_TYPES}"
            )

    @staticmethod
    def build_id(source_type: str, source_id: str, detail: str = "") -> str:
        ProvenanceResolver.validate_source_type(source_type)
        return f"prov:1:{source_type}:{source_id}:{detail}"

    @staticmethod
    def parse_id(pid: str) -> dict:
        if not pid.startswith("prov:"):
            raise ValueError(f"Invalid provenance ID: {pid}")

        parts = pid.split(":")
        if len(parts) < 4:
            raise ValueError(f"Invalid provenance ID format: {pid}")

        version_str = parts[1]
        source_type = parts[2]

        ProvenanceResolver.validate_source_type(source_type)

        try:
            version = int(version_str)
        except ValueError:
            raise ValueError(f"Invalid version in provenance ID: {version_str}")

        if source_type == "memory":
            # Memory source_id is a URI (e.g., ctx://acme/users/alice/memories/entities/rust)
            # URI itself contains colons, so simple split(":") would break incorrectly.
            # Rejoin parts[3:] and find the last colon to separate source_id from detail.
            remaining = ":".join(parts[3:])
            if remaining.endswith(":"):
                source_id = remaining[:-1]
                detail = ""
            else:
                last_colon = remaining.rfind(":")
                if last_colon == -1:
                    source_id = remaining
                    detail = ""
                else:
                    source_id = remaining[:last_colon]
                    detail = remaining[last_colon + 1:]
        else:
            if len(parts) == 4:
                source_id = parts[3]
                detail = ""
            else:
                source_id = parts[3]
                detail = parts[4] if len(parts) > 4 else ""

        return {
            "version": version,
            "source_type": source_type,
            "source_id": source_id,
            "detail": detail,
        }