"""Versioned memory evidence for background consolidation plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from core.errors import AccessDeniedError, NodeNotFoundError, ValidationError
from core.interfaces import ContextFS
from core.models import RequestContext

MAX_EVIDENCE_NODES = 128


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryEvidence:
    """Detached, canonical JSON of one persisted ContextNode and its revision."""

    uri: str
    version: int
    node_json: str
    fingerprint: str


@dataclass(frozen=True)
class MemoryEvidenceSnapshot:
    """Ordered evidence checkpoints for one account and owner space."""

    account_id: str
    owner_space: str
    nodes: tuple[MemoryEvidence, ...]
    fingerprint: str

    def to_dict(self) -> dict:
        """Return a JSON-ready copy for a plan's evidence manifest."""
        return {
            "schema_version": 1,
            "account_id": self.account_id,
            "owner_space": self.owner_space,
            "fingerprint": self.fingerprint,
            "nodes": [
                {
                    "uri": node.uri,
                    "version": node.version,
                    "fingerprint": node.fingerprint,
                    "node": json.loads(node.node_json),
                }
                for node in self.nodes
            ],
        }


@dataclass(frozen=True)
class MemoryEvidenceCheck:
    """Freshness observed by rereading a plan's source nodes."""

    changed_uris: tuple[str, ...]
    unavailable_uris: tuple[str, ...]

    @property
    def is_current(self) -> bool:
        return not self.changed_uris and not self.unavailable_uris


class MemoryEvidenceService:
    """Capture and recheck evidence through ContextFS access-controlled reads.

    Each source has its own checkpoint. Writers use the captured versions with
    storage-level optimistic locking when applying a consolidation plan.
    """

    def __init__(self, fs: ContextFS) -> None:
        self._fs = fs

    def capture(
        self, uris: Sequence[str], ctx: RequestContext, *, owner_space: str
    ) -> MemoryEvidenceSnapshot:
        """Read up to 128 unique canonical memory URIs in deterministic order."""
        if isinstance(uris, str) or not uris or len(uris) > MAX_EVIDENCE_NODES:
            raise ValidationError("uris", f"provide 1 to {MAX_EVIDENCE_NODES} memory URIs")
        if any(not isinstance(uri, str) or not uri for uri in uris):
            raise ValidationError("uris", "memory URIs must be nonempty strings")
        nodes = tuple(self._read(uri, ctx, owner_space) for uri in sorted(set(uris)))
        manifest = {
            "schema_version": 1,
            "account_id": ctx.account_id,
            "owner_space": owner_space,
            "sources": [(node.uri, node.version, node.fingerprint) for node in nodes],
        }
        return MemoryEvidenceSnapshot(
            account_id=ctx.account_id,
            owner_space=owner_space,
            nodes=nodes,
            fingerprint=_digest(_canonical_json(manifest)),
        )

    def check(self, snapshot: MemoryEvidenceSnapshot, ctx: RequestContext) -> MemoryEvidenceCheck:
        """Report source revisions and availability at the time of rechecking."""
        if snapshot.account_id != ctx.account_id:
            raise AccessDeniedError(
                snapshot.account_id, ctx.account_id, "evidence account mismatch"
            )
        changed = []
        unavailable = []
        for source in snapshot.nodes:
            try:
                current = self._read(source.uri, ctx, snapshot.owner_space)
            except NodeNotFoundError:
                unavailable.append(source.uri)
                continue
            if current.fingerprint != source.fingerprint:
                changed.append(source.uri)
        return MemoryEvidenceCheck(tuple(changed), tuple(unavailable))

    def _read(self, uri: str, ctx: RequestContext, owner_space: str) -> MemoryEvidence:
        kind, _, owner_id = owner_space.partition(":")
        if kind not in {"user", "agent"} or not owner_id or "/" in owner_id:
            raise ValidationError("owner_space", "provide a user:<id> or agent:<id> space")
        prefix = f"ctx://{ctx.account_id}/{kind}s/{owner_id}/memories/"
        if not uri.startswith(prefix):
            raise AccessDeniedError(uri, ctx.account_id, "evidence scope mismatch")
        node = self._fs.read_node(uri, ctx)
        if node.uri != uri or node.owner_space != owner_space:
            raise AccessDeniedError(uri, ctx.account_id, "evidence node scope mismatch")
        if node.context_type != "MEMORY":
            raise ValidationError("uri", "evidence requires a MEMORY node")
        version = node.metadata.get("version")
        if type(version) is not int or version < 1:
            raise ValidationError("version", "evidence requires a positive persisted version")
        node_json = _canonical_json(asdict(node))
        return MemoryEvidence(uri, version, node_json, _digest(node_json))
