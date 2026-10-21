from scene.config import InMemorySceneConfigProvider, SceneConfig, SceneConfigCache


def test_config_defaults_and_scoped_overrides():
    defaults = SceneConfig()
    defaults.validate()
    configured = defaults.merged({"lexical_enabled": True, "neighbor_k": 12})
    configured.validate()
    assert configured.lexical_enabled is True
    assert configured.neighbor_k == 12
    assert configured.max_llm_calls_per_batch == 3
    assert defaults.neighbor_k == 8


def test_config_publish_advances_revision_and_refreshes_cache():
    provider = InMemorySceneConfigProvider()
    cache = SceneConfigCache(provider)
    original = cache.get("acct", "user:alice")
    published = provider.publish(
        "acct", "user:alice", original.revision,
        {"max_candidate_scenes": 4}, source="ui",
    )
    assert published.revision == 1
    assert published.source == "ui"
    assert cache.get("acct", "user:alice") == published
    assert published.config.max_candidate_scenes == 4
    assert original.config.max_candidate_scenes == 8
    assert original.revision == 0
    exported = provider.export("acct", "user:alice")
    assert exported["values"]["max_candidate_scenes"] == 4
    assert exported["revision"] == 1


def test_config_owner_and_account_scopes_advance_independently():
    provider = InMemorySceneConfigProvider()
    for account, owner, neighbors in (
        ("acct", "user:alice", 4),
        ("acct", "user:bob", 6),
        ("other", "user:alice", 12),
    ):
        provider.publish(account, owner, 0, {"neighbor_k": neighbors})
    assert provider.get_effective("acct", "user:alice").config.neighbor_k == 4
    assert provider.get_effective("acct", "user:bob").config.neighbor_k == 6
    assert provider.get_effective("other", "user:alice").config.neighbor_k == 12


def test_config_rollback_publishes_new_revision_to_cache_and_listeners():
    provider = InMemorySceneConfigProvider()
    cache = SceneConfigCache(provider)
    notifications = []
    provider.subscribe(lambda *event: notifications.append(event))
    first = provider.publish("acct", "user:alice", 0, {"scene_boost": 0.3})
    second = provider.publish("acct", "user:alice", first.revision, {"scene_boost": 0.6})
    restored = provider.rollback("acct", "user:alice", first.revision)
    assert second.revision == 2
    assert restored.revision == 3
    assert restored.source == "rollback"
    assert restored.config.scene_boost == 0.3
    assert cache.get("acct", "user:alice") == restored
    assert notifications[-1] == ("acct", "user:alice", restored)
