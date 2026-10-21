"""Scene-card projection through the shared lexical-index contract."""

from collections.abc import Iterable, Sequence
from typing import Protocol

from core.lexical import LexicalIndex, LexicalRecord
from scene.models import SceneSnapshot


class SceneCandidateIndex(Protocol):
    def search(self, query: str, snapshot: SceneSnapshot, limit: int) -> Sequence[str]: ...

    def close(self, timeout: float = 5.0) -> bool: ...


class SceneProjectionQueue(Protocol):
    def pending_projections(self, limit: int = 20) -> Iterable[SceneSnapshot]:
        """Yield current snapshots under an exclusive projection lease."""
        ...

    def acknowledge_projection(self, snapshot: SceneSnapshot) -> None: ...


class SceneIndexRefresh:
    def __init__(self, queue: SceneProjectionQueue, index: "LexicalSceneIndex"):
        self.queue, self.index = queue, index

    def __call__(self):
        for snapshot in self.queue.pending_projections():
            self.index.publish(snapshot)
            self.queue.acknowledge_projection(snapshot)


class LexicalSceneIndex:
    """Publish scene cards and validate search hits against canonical revisions."""

    def __init__(self, index: LexicalIndex):
        self.index = index

    def close(self, timeout: float = 5.0) -> bool:
        return self.index.close(timeout=timeout)

    def publish(self, snapshot: SceneSnapshot) -> None:
        records = []
        for scene in snapshot.scenes:
            if scene.status != "ACTIVE":
                continue
            cluster = scene.cluster
            identity = f"{snapshot.account_id}/{snapshot.owner_space}/{scene.scene_id}"
            records.append(LexicalRecord(
                identity, identity, 0,
                "\n".join((cluster.label, cluster.summary,
                           " ".join(value for _, value in cluster.boundary))),
                snapshot.account_id, snapshot.owner_space,
                {"scene_id": scene.scene_id, "scene_version": scene.version,
                 "context_type": "SCENE"},
            ))
        rebuild = self.index.begin_rebuild()
        try:
            if not self.index.replace_scopes(
                rebuild_id=rebuild, account_id=snapshot.account_id,
                owner_spaces=[snapshot.owner_space], records=records,
            ):
                raise RuntimeError("scene lexical projection changed during publication")
        finally:
            self.index.abort_rebuild(rebuild)

    def search(self, query: str, snapshot: SceneSnapshot, limit: int) -> list[str]:
        versions = {s.scene_id: s.version for s in snapshot.scenes if s.status == "ACTIVE"}
        hits = self.index.search(query, limit, {
            "account_id": snapshot.account_id, "owner_space": [snapshot.owner_space],
            "context_type": "SCENE",
        })
        return list(dict.fromkeys(
            h.metadata["scene_id"] for h in hits
            if h.score > 0 and h.metadata.get("scene_id") in versions
            and h.metadata.get("scene_version") == versions[h.metadata["scene_id"]]
        ))
