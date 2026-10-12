"""Database-neutral contracts for scene projection and retrieval."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.models import ContextNode
from scene.models import PersistedScene, SceneArchive, SceneCluster, SceneSnapshot

SceneSourceState = tuple[int, object]
ResolvedArchiveSource = tuple[str, float]


@dataclass(frozen=True)
class SceneDecisionInput:
    """Detached scenes, complete member evidence and indexed vectors at one revision.

    Nodes include every existing scene member and the offered new evidence.
    Embeddings use node URIs as keys. Adapters return caller-owned node and vector
    copies so a decision can operate independently of subsequent backend changes.
    """

    snapshot: SceneSnapshot
    nodes: tuple[ContextNode, ...]
    embeddings: dict[str, list[float]]


@runtime_checkable
class SceneDecisionStore(Protocol):
    """Consistent snapshot reads and atomic publication scoped by account and owner.

    Scene changes and source-memory changes advance the same monotonic scope
    revision, including manual edits and deletions. Adapters own transactions
    and synchronization across readers and writers.
    """

    def load_input(self, account_id: str, owner_space: str) -> SceneDecisionInput:
        """Read scene state, member nodes and indexed vectors from one revision."""
        ...

    def compare_and_swap(self, expected: SceneSnapshot, replacement: SceneSnapshot) -> bool:
        """Publish the full scene replacement when the scope revision still matches.

        Both snapshots identify the same account and owner. The replacement
        revision equals expected.revision + 1. Return True after atomic commit;
        return False on revision conflict, preserving the current state.
        Raise ValueError for a scope mismatch or invalid replacement revision.
        """
        ...


@runtime_checkable
class SceneNodeSource(Protocol):
    """Source of canonical memory nodes used to build scene projections."""

    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: Sequence[str],
        page_size: int,
    ) -> Iterator[ContextNode]: ...


@runtime_checkable
class SceneStore(Protocol):
    """Read-side scene store contract."""

    def load_active(
        self,
        account_id: str,
        owner_space: str,
    ) -> list[PersistedScene]: ...

    def contributions(
        self,
        account_id: str,
        owner_spaces: Sequence[str],
        candidate_uris: Sequence[str],
    ) -> list[tuple[PersistedScene, str, float]]: ...

    def archive_evidence(
        self,
        account_id: str,
        scene_ids: Sequence[str],
    ) -> list[SceneArchive]: ...


@runtime_checkable
class SceneProjectionStore(SceneStore, Protocol):
    """Write-side scene projection contract.

    ``publish`` must atomically install one active version for a scope, retire
    projections absent from ``clusters``, and advance source state only after
    the complete projection succeeds. Locking is backend-owned.
    """

    def list_dirty_scopes(
        self,
        account_id: str,
        limit: int = 20,
    ) -> list[str]: ...

    def scope_is_dirty(self, account_id: str, owner_space: str) -> bool: ...

    def projection_is_quiescent(self, account_id: str) -> bool: ...

    def source_state(
        self,
        account_id: str,
        owner_space: str,
    ) -> SceneSourceState: ...

    def publish(
        self,
        account_id: str,
        owner_space: str,
        clusters: Sequence[SceneCluster],
        node_count: int,
        max_updated_at: object,
        identity_overlap: float,
    ) -> None: ...


class SceneArchiveResolver(Protocol):
    def resolve_archive_sources(
        self,
        account_id: str,
        archive_nodes: Sequence[ContextNode],
    ) -> Mapping[str, ResolvedArchiveSource]: ...


class SceneSourceExpansionStore(Protocol):
    def source_related_contexts(self, account_id: str, scene_id: str) -> set[str]: ...

    def source_related_nodes(self, account_id: str, scene_id: str) -> list[dict]: ...


class SceneEntityExpansionStore(Protocol):
    def entity_anchor_candidates(
        self,
        account_id: str,
        owner_spaces: Sequence[str],
    ) -> list[str]: ...

    def entity_anchor_nodes(
        self,
        account_id: str,
        anchor_uris: Sequence[str],
    ) -> list[dict]: ...
