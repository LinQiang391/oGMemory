"""Core protocol interfaces for ContextEngine.

Phase 0 frozen — these protocols define the contracts between layers.
All protocols use typing.Protocol for structural subtyping.
"""

from typing import Protocol

from core.models import (
    RequestContext,
    ContextNode,
    RelationEdge,
    CandidateMemory,
    WritePlan,
    IndexRecord,
    OutboxEvent,
    TypedQuery,
    SeedHit,
)


class ContextFS(Protocol):
    """File system abstraction for context persistence.

    All methods enforce RequestContext-based access control via _is_accessible.
    Implementations MUST validate account_id prefix in URI matches RequestContext.

    AGFS implementation writes nodes to:
    /accounts/{account}/users/{user}/memories/{category}/{slug}/
    With files: content.md, .relations.json, .abstract.md, .overview.md, .meta.json

    See CLAUDE.md §2 AGFS Directory Spec for write order semantics.
    """

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node to AGFS with atomic 4-step write order.

        Write order (strictly enforced, Repair Job depends on this):
        ① content.md (largest file, write first)
        ② .relations.json
        ③ .abstract.md, .overview.md (parallel)
        ④ .meta.json (commit point: status=ACTIVE)

        Step ④ completion triggers OutboxEvent registration.
        """
        ...

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from AGFS.

        Raises AccessDeniedError if URI account prefix != ctx.account_id.
        Raises NodeNotFoundError if node does not exist or status != ACTIVE.
        """
        ...

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node from AGFS.

        Raises AccessDeniedError if URI account prefix != ctx.account_id.
        """
        ...

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        """Move/rename a node in AGFS.

        Raises AccessDeniedError if either URI account prefix != ctx.account_id.
        """
        ...

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List child URIs under a given URI.

        Returns only URIs where the caller has access.
        """
        ...

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if a node exists and is ACTIVE.

        Returns True only if .meta.json exists AND status == ACTIVE.
        PENDING nodes are not visible to upper layers.
        """
        ...


class RelationStore(Protocol):
    """Graph relation storage.

    v1 implementation: Read/write .relations.json in AGFS node directories.
    Phase 2+ implementation: May migrate to graph database, interface unchanged.

    Edges are directed and weighted.
    """

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        """Get all outgoing edges from a node.

        Returns empty list if node has no relations or does not exist.
        """
        ...

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        """Add or update relation edges.

        Idempotent: same edge (from_uri, to_uri, relation_type) is updated.
        """
        ...


class CandidateExtractor(Protocol):
    """Extract candidate memories from message history.

    Four implementations (one per memory category):
    - ProfileExtractor: User profile and state
    - PreferenceExtractor: User preferences and opinions
    - EntityEventExtractor: Entities and events
    - ProcedureExtractor: Skills and procedures (agent scope)
    """

    def extract(self, messages: list[dict], ctx: RequestContext) -> list[CandidateMemory]:
        """Extract candidate memories from a conversation.

        Messages format: {"role": "user|assistant", "content": "..."}
        Returns list of candidates with confidence scores.
        """
        ...


class MergePolicy(Protocol):
    """Determine write action for a candidate memory.

    Four implementations (one per memory type):
    - ProfilePolicy: Always merge, fixed URI
    - AggregateTopicPolicy: Merge by slug (preference, entity, pattern)
    - AppendOnlyPolicy: Always create, unique event_id (event, case)
    - SkillToolPolicy: Fixed URI, accumulate best practices (skill)
    """

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for a candidate.

        Returns WritePlan with action: create|merge|append|skip
        """
        ...


class VectorIndex(Protocol):
    """Vector similarity search interface.

    Implementation layer handles account_id filtering transparently.
    Upper layers (retrieval) are unaware of tenant isolation.

    Upsert is idempotent: same IndexRecord.id overwrites safely.
    """

    def upsert(self, records: list[IndexRecord]) -> None:
        """Add or update records in the index.

        Idempotent: duplicate IndexRecord.id overwrites.
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove records from the index.

        No-op for non-existent ids.
        """
        ...

    def search(self, query: TypedQuery) -> list[SeedHit]:
        """Vector similarity search with filter support.

        Returns top-k SeedHit sorted by score descending.
        Automatically filters by query.account_id (implementation detail).
        """
        ...


class Embedder(Protocol):
    """Text embedding interface.

    Converts text to fixed-length vectors for vector indexing.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns list of vectors, same length as input.
        All vectors have same dimension (implementation-specific).
        """
        ...


class LLM(Protocol):
    """Large language model interface for JSON completion.

    Used by Extractor and Planner for structured output.
    """

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete a prompt and return JSON matching schema.

        Schema format: {"type": "object", "properties": {...}}
        Returns parsed JSON dict.
        """
        ...
