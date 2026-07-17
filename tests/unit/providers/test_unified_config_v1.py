"""Tests for V1 auth and sharing config in OgMemConfig."""

from __future__ import annotations

import pytest

from providers.unified_config import OgMemConfig


def test_load_reads_v1_auth_and_sharing_from_env(monkeypatch):
    monkeypatch.setenv("OG_ROLE_CONTROL_ENABLED", "true")
    monkeypatch.setenv("OG_ROOT_API_KEY", "root-key")
    monkeypatch.setenv("OG_ADMIN_API_KEYS", "admin-a,admin-b")
    monkeypatch.setenv("OG_AGENT_SHARED_MODE", "user")
    monkeypatch.setenv("OG_AGENT_SHARED_LIST", "agent-1,agent-2")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST", "127.0.0.1,10.0.0.0/8")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST_TRUST_PROXY", "true")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is True
    assert cfg.root_api_key == "root-key"
    assert cfg.admin_api_keys == ["admin-a", "admin-b"]
    assert cfg.agent_shared_mode == "user"
    assert cfg.agent_shared_list == ["agent-1", "agent-2"]
    assert cfg.http_ip_allowlist == ["127.0.0.1", "10.0.0.0/8"]
    assert cfg.http_ip_allowlist_trust_proxy is True


def test_load_uses_safe_defaults_for_v1_auth_and_sharing():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is False
    assert cfg.root_api_key is None
    assert cfg.admin_api_keys == []
    assert cfg.agent_shared_mode == "off"
    assert cfg.agent_shared_list == []
    assert cfg.http_ip_allowlist == []
    assert cfg.http_ip_allowlist_trust_proxy is False


def test_trace_turn_limit_reads_storage_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setenv("OGMEM_TRACE_MAX_TURN_EVENTS", "64")
    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {"storage": {"trace_max_turn_events": 256}},
    )
    assert OgMemConfig.load("ignored.yaml").trace_max_turn_events == 256

    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"storage": {}})
    assert OgMemConfig.load("ignored.yaml").trace_max_turn_events == 64


@pytest.mark.parametrize("invalid", [0, 1, 1025])
def test_trace_turn_limit_rejects_invalid_bounds(invalid):
    with pytest.raises(ValueError, match="between 2 and 1024"):
        OgMemConfig(trace_max_turn_events=invalid)


def test_skill_sedimentation_defaults_off_and_loads_nested_config(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "storage": {"backend": "sql", "connection_string": "postgresql://unused"},
            "skills": {"sedimentation": {
                "enabled": True, "min_evidence_turns": 7,
                "min_distinct_sessions": 4, "semantic_update_threshold": .94,
                "semantic_hold_threshold": .81, "candidate_limit": 25,
                "catalog_scan_limit": 2500,
            }},
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.skill_sedimentation_enabled is True
    assert cfg.skill_min_evidence_turns == 7
    assert cfg.skill_min_distinct_sessions == 4
    assert cfg.skill_semantic_update_threshold == .94
    assert cfg.skill_candidate_limit == 25
    assert cfg.skill_catalog_scan_limit == 2500
    default_cfg = OgMemConfig()
    assert default_cfg.skill_sedimentation_enabled is False
    assert default_cfg.skill_min_evidence_turns == 1
    assert default_cfg.skill_min_distinct_sessions == 1


def test_skill_catalog_scan_limit_cannot_be_below_page_size():
    with pytest.raises(ValueError, match="cannot be below"):
        OgMemConfig(skill_candidate_limit=200, skill_catalog_scan_limit=100)


def test_enabled_skill_sedimentation_requires_sql_backend():
    with pytest.raises(ValueError, match="requires storage.backend=sql"):
        OgMemConfig(skill_sedimentation_enabled=True, storage_backend="agfs")


def test_enabled_skill_sedimentation_requires_sql_connection_string():
    with pytest.raises(ValueError, match="storage.sql.connection_string"):
        OgMemConfig(skill_sedimentation_enabled=True, storage_backend="sql")


def test_enabled_skill_sedimentation_rejects_lease_shorter_than_external_call_budget():
    with pytest.raises(ValueError, match="at least 180"):
        OgMemConfig(
            skill_sedimentation_enabled=True,
            storage_backend="sql",
            sql_connection_string="postgresql://unused",
            skill_lease_seconds=179,
        )


def test_compact_prepare_token_ttl_defaults_to_300():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.compact_prepare_token_ttl == 300


def test_compact_prepare_token_ttl_reads_yaml_and_env(monkeypatch):
    from providers import unified_config

    monkeypatch.setenv("OGMEM_COMPACT_PREPARE_TOKEN_TTL", "123")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"compact_prepare_token_ttl": 456}},
    )

    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.compact_prepare_token_ttl == 456

    monkeypatch.setattr(unified_config, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("unused.yaml")
    assert cfg.compact_prepare_token_ttl == 123


def test_rolling_compress_enabled_defaults_to_true():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.rolling_compress_enabled is True


def test_rolling_compress_enabled_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_ENABLED", "false")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.rolling_compress_enabled is False


def test_rolling_compress_enabled_yaml_overrides_env(monkeypatch):
    from providers import unified_config

    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_ENABLED", "false")
    monkeypatch.setattr(
        unified_config,
        "_load_yaml",
        lambda _path: {"memory": {"rolling_compress_enabled": True}},
    )

    cfg = OgMemConfig.load("unused.yaml")

    assert cfg.rolling_compress_enabled is True


def test_archive_merge_prd_defaults():
    cfg = OgMemConfig()

    assert cfg.archive_max_count == 10
    assert cfg.archive_merge_threshold == 10
    assert not hasattr(cfg, "archive_auto_merge_enabled")
    assert not hasattr(cfg, "archive_merge_max_messages")


def test_archive_merge_reads_prd_yaml_and_env(monkeypatch):
    monkeypatch.setenv("OGMEM_ARCHIVE_MAX_COUNT", "7")
    monkeypatch.setenv("OGMEM_ARCHIVE_MERGE_THRESHOLD", "6")

    cfg = OgMemConfig.load(config_path="missing.yaml")

    assert cfg.archive_max_count == 7
    assert cfg.archive_merge_threshold == 6


def test_rolling_fallback_defaults_disabled_and_env_override(monkeypatch):
    assert OgMemConfig().rolling_compress_fallback_enabled is False

    monkeypatch.setenv("OGMEM_ROLLING_COMPRESS_FALLBACK_ENABLED", "true")
    cfg = OgMemConfig.load(config_path="missing.yaml")

    assert cfg.rolling_compress_fallback_enabled is True


def test_compact_gap_completion_defaults():
    cfg = OgMemConfig()

    assert cfg.prefetch_enabled is False
    assert cfg.prefetch_top_k == 5
    assert cfg.session_state_bridge_enabled is True
    assert cfg.session_state_sync_interval_turns == 1
    assert cfg.topic_detection_enabled is False
    assert cfg.compression_quality_enabled is False
    assert cfg.compression_quality_persist_metadata is False
    assert cfg.retrieval_candidate_pool_enabled is False
    assert cfg.retrieval_result_ranker_strategy == "semantic"
    assert cfg.retrieval_global_result_rerank_enabled is False
    assert cfg.retrieval_owner_scope_filter_enabled is False


def test_compact_gap_completion_reads_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "memory": {
                "prefetch_enabled": True,
                "prefetch_top_k": 9,
                "session_state_bridge_enabled": False,
                "session_state_sync_interval_turns": 4,
                "topic_detection_enabled": True,
                "compression_quality_enabled": True,
                "compression_quality_persist_metadata": True,
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.prefetch_enabled is True
    assert cfg.prefetch_top_k == 9
    assert cfg.session_state_bridge_enabled is False
    assert cfg.session_state_sync_interval_turns == 4
    assert cfg.topic_detection_enabled is True
    assert cfg.compression_quality_enabled is True
    assert cfg.compression_quality_persist_metadata is True

    monkeypatch.setenv("OGMEM_PREFETCH_ENABLED", "false")
    monkeypatch.setenv("OGMEM_PREFETCH_TOP_K", "3")
    monkeypatch.setenv("OGMEM_SESSION_STATE_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS", "2")
    monkeypatch.setenv("OGMEM_TOPIC_DETECTION_ENABLED", "false")
    monkeypatch.setenv("OGMEM_COMPRESSION_QUALITY_ENABLED", "false")
    monkeypatch.setenv("OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA", "false")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"memory": {}})
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.prefetch_enabled is False
    assert cfg.prefetch_top_k == 3
    assert cfg.session_state_bridge_enabled is True
    assert cfg.session_state_sync_interval_turns == 2
    assert cfg.topic_detection_enabled is False
    assert cfg.compression_quality_enabled is False
    assert cfg.compression_quality_persist_metadata is False


def test_retrieval_config_reads_yaml_and_env(monkeypatch):
    from providers import unified_config as uc

    monkeypatch.setattr(
        uc,
        "_load_yaml",
        lambda _path: {
            "retrieval": {
                "max_top_k": 60,
                "default_top_k": 15,
                "global_search_topk": 45,
                "candidate_pool_enabled": True,
                "candidate_pool_multiplier": 4,
                "min_candidate_pool_size": 20,
                "result_ranker_strategy": "metadata_aware",
                "global_result_rerank_enabled": True,
                "owner_scope_filter_enabled": True,
            }
        },
    )
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.retrieval_max_top_k == 60
    assert cfg.retrieval_default_top_k == 15
    assert cfg.retrieval_global_search_topk == 45
    assert cfg.retrieval_candidate_pool_enabled is True
    assert cfg.retrieval_candidate_pool_multiplier == 4
    assert cfg.retrieval_min_candidate_pool_size == 20
    assert cfg.retrieval_result_ranker_strategy == "metadata_aware"
    assert cfg.retrieval_global_result_rerank_enabled is True
    assert cfg.retrieval_owner_scope_filter_enabled is True

    monkeypatch.setenv("OGMEM_RETRIEVAL_CANDIDATE_POOL_ENABLED", "false")
    monkeypatch.setenv("OGMEM_RETRIEVAL_RESULT_RANKER_STRATEGY", "semantic")
    monkeypatch.setenv("OGMEM_RETRIEVAL_GLOBAL_RESULT_RERANK_ENABLED", "false")
    monkeypatch.setenv("OGMEM_RETRIEVAL_OWNER_SCOPE_FILTER_ENABLED", "false")
    monkeypatch.setattr(uc, "_load_yaml", lambda _path: {"retrieval": {}})
    cfg = OgMemConfig.load("ignored.yaml")
    assert cfg.retrieval_candidate_pool_enabled is False
    assert cfg.retrieval_result_ranker_strategy == "semantic"
    assert cfg.retrieval_global_result_rerank_enabled is False
    assert cfg.retrieval_owner_scope_filter_enabled is False
