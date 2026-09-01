from __future__ import annotations

from collections.abc import Iterator

from core.models import ContextNode
from scene.store import SceneNodeSource, SceneProjectionStore


class InMemorySceneAdapter:
    def list_dirty_scopes(self, account_id, limit=20):
        return []

    def scope_is_dirty(self, account_id, owner_space):
        return False

    def projection_is_quiescent(self, account_id):
        return True

    def source_state(self, account_id, owner_space):
        return 0, None

    def publish(
        self,
        account_id,
        owner_space,
        clusters,
        node_count,
        max_updated_at,
        identity_overlap,
    ):
        return None

    def load_active(self, account_id, owner_space):
        return []

    def contributions(self, account_id, owner_spaces, candidate_uris):
        return []

    def archive_evidence(self, account_id, scene_ids):
        return []


class InMemoryNodeSource:
    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: list[str],
        page_size: int,
    ) -> Iterator[ContextNode]:
        return iter(())


def test_non_sql_scene_adapters_satisfy_runtime_contracts():
    assert isinstance(InMemorySceneAdapter(), SceneProjectionStore)
    assert isinstance(InMemoryNodeSource(), SceneNodeSource)
