"""Abstract scene-memory models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneMember:
    context_uri: str
    weight: float
    role: str = "member"


@dataclass(frozen=True)
class SceneCandidate:
    """Evidence group and lexical hints supplied to the scene decision stage."""

    account_id: str
    owner_space: str
    members: tuple[SceneMember, ...]
    centroid: tuple[float, ...]
    quality: float
    label_hint: str
    summary_hint: str
    source_fingerprint: str
    basis: str = "semantic"
    source_archive_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneCluster:
    owner_space: str
    members: tuple[SceneMember, ...]
    centroid: tuple[float, ...]
    quality: float
    label: str
    summary: str
    source_fingerprint: str
    basis: str = "semantic"
    source_archive_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PersistedScene:
    scene_id: str
    version: int
    owner_space: str
    centroid: tuple[float, ...]
    quality: float
    member_count: int
    members: frozenset[str] = field(default_factory=frozenset)
    basis: str = "semantic"


@dataclass(frozen=True)
class SceneContribution:
    context_uri: str
    scene_id: str
    scene_score: float
    membership_weight: float
    support: float
    evidence: float


@dataclass(frozen=True)
class SceneArchive:
    scene_id: str
    archive_id: str
    messages: tuple[dict, ...]
    member_support: int
    member_abstracts: tuple[str, ...] = ()
    member_contexts: tuple[tuple[str, str], ...] = ()
    archive_alias_ids: tuple[str, ...] = ()
