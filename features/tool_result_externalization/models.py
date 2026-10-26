"""Request-local and persisted models for tool-result externalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ToolResultPart:
    """One provider-neutral tool result found in a message."""

    part_path: str
    payload_path: tuple[Any, ...]
    tool_call_id: str | None
    tool_name: str | None
    output_text: str
    mime_type: str | None
    serialization: Literal["raw_string", "canonical_json"]
    source_ref: str | None
    original_chars: int
    original_bytes: int
    source_collection: bool = False
    source_refs: tuple[str, ...] = ()
    occurrence_index: int = 0


@dataclass(frozen=True)
class NormalizedMessage:
    """A copied input message plus the result parts recognized within it."""

    source_message_id: str | None
    turn_id: str | None
    role: str
    ordinary_text_parts: tuple[str, ...]
    tool_results: tuple[ToolResultPart, ...]
    original: dict
    message_index: int
    identity_is_stable: bool = False


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata committed alongside the complete logical tool output."""

    schema_version: int
    artifact_id: str
    account_id: str
    session_id: str
    owner_user_id: str
    owner_agent_id: str
    turn_id: str | None
    source_message_id: str | None
    tool_call_id: str | None
    tool_name: str | None
    part_path: str
    occurrence_index: int
    mime_type: str | None
    detected_type: str
    serialization: str
    original_chars: int
    original_bytes: int
    sha256: str
    synopsis: str
    externalization_reason: str
    identity_strength: str
    created_at: str
    fact_candidates: tuple[dict[str, Any], ...] = ()
    evidence_candidates: tuple[dict[str, Any], ...] = ()
    record_candidates: tuple[dict[str, Any], ...] = ()
    record_layout: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None

    @property
    def ref(self) -> str:
        return f"ogmemory://tool-results/{self.artifact_id}"

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.schema_version < 2:
            result.pop("fact_candidates", None)
            result.pop("evidence_candidates", None)
            result.pop("record_candidates", None)
            result.pop("record_layout", None)
            result.pop("coverage", None)
            return result
        result["fact_candidates"] = list(self.fact_candidates)
        result["evidence_candidates"] = list(self.evidence_candidates)
        result["record_candidates"] = list(self.record_candidates)
        return result

    def public_dict(self) -> dict:
        """Return bounded metadata safe for list/MCP responses."""
        result = self.to_dict()
        for key in (
            "fact_candidates",
            "evidence_candidates",
            "record_candidates",
            "record_layout",
        ):
            result.pop(key, None)
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> ArtifactMetadata:
        values = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name in raw
        }
        if "fact_candidates" in values:
            values["fact_candidates"] = tuple(values["fact_candidates"] or ())
        if "evidence_candidates" in values:
            values["evidence_candidates"] = tuple(values["evidence_candidates"] or ())
        if "record_candidates" in values:
            values["record_candidates"] = tuple(values["record_candidates"] or ())
        return cls(**values)


@dataclass(frozen=True)
class ExternalizationResult:
    """A successfully verified artifact and its deterministic safe stub."""

    metadata: ArtifactMetadata
    stub: str
    reused: bool = False


@dataclass(frozen=True)
class SourceCollectionReuse:
    """An atomically verified collection projected as canonical source stubs."""

    stub: str
    source_count: int
    metadata: tuple[ArtifactMetadata, ...] = ()


@dataclass(frozen=True)
class SourceFragmentPassthrough:
    """A verified bounded read fragment that must remain model-visible verbatim."""

    metadata: ArtifactMetadata


@dataclass(frozen=True)
class GovernedMessages:
    """Result of governing one request's message projection."""

    messages: list[dict]
    source_message_ids: list[str | None]
    fallback_original_contents: list[str]
    removed_message_ids: list[str]
    removed_fallback_messages: list[tuple[str, str]]
    preserved_message_ids: list[str]
    externalized_count: int = 0
    failed_count: int = 0
    fallback_original: bool = False
