from __future__ import annotations

import pytest

from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import (
    AGFSContextFS,
    build_uri,
    uri_to_path,
)
from fs.agfs_adapter.agfs_context_fs import (
    parse_uri as parse_agfs_uri,
)
from fs.sql_adapter.sql_context_fs import SQLContextFS
from fs.sql_adapter.sql_context_fs import parse_uri as parse_sql_uri
from index.index_record_builder import build_index_records
from providers.unified_config import OgMemConfig
from providers.vector_index.chroma_index import ChromaVectorIndex


def _ctx(account_id: str = "acct") -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="unrelated-user",
        agent_id="unrelated-agent",
        session_id="session",
        trace_id="trace",
    )


def _code_node(**metadata) -> ContextNode:
    return ContextNode(
        uri="ctx://acct/resources/a",
        context_type="RESOURCE",
        category="code",
        level=0,
        owner_space="resource:acct",
        abstract="parse config",
        overview="parse_config(path)",
        content="def parse_config(path):\n    return path\n",
        metadata=metadata,
    )


def _clear_feature_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OGMEM_CODE_RESOURCES_ENABLED",
        "OGMEM_CODE_RETRIEVAL_ENABLED",
        "OGMEM_CODE_RETRIEVAL_SEMANTIC_ENABLED",
        "OGMEM_CODE_RETRIEVAL_BM25_ENABLED",
        "OGMEM_CODE_RETRIEVAL_SYMBOLIC_ENABLED",
        "OGMEM_CODE_RETRIEVAL_GRAPH_ENABLED",
        "OGMEM_TOOL_OUTPUT_FILTER_ENABLED",
        "OGMEM_TOOL_OUTPUT_FILTER_READ_ENABLED",
        "OGMEM_TOOL_OUTPUT_FILTER_BASH_ENABLED",
        "OGMEM_TOOL_OUTPUT_FILTER_RECOVERY_MAX_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_feature_defaults_and_effective_master_switches():
    cfg = OgMemConfig()

    assert cfg.code_resources_enabled is False
    assert cfg.code_retrieval_enabled is False
    assert cfg.tool_output_filter_enabled is False
    assert cfg.code_retrieval_semantic_enabled is True
    assert cfg.code_retrieval_bm25_enabled is True
    assert cfg.code_retrieval_symbolic_enabled is True
    assert cfg.code_retrieval_graph_enabled is True
    assert cfg.tool_output_filter_read_enabled is True
    assert cfg.tool_output_filter_bash_enabled is True
    assert cfg.tool_output_recovery_max_chars == 50000
    assert cfg.code_retrieval_route_enabled("semantic") is False
    assert cfg.tool_output_filter_kind_enabled("Read") is False

    enabled = OgMemConfig(code_retrieval_enabled=True, tool_output_filter_enabled=True)
    assert enabled.code_retrieval_route_enabled("semantic") is True
    assert enabled.tool_output_filter_kind_enabled("Read") is True
    assert enabled.tool_output_filter_kind_enabled("Bash") is True


def test_feature_yaml_accepts_flat_and_nested_forms(monkeypatch):
    _clear_feature_env(monkeypatch)
    monkeypatch.setattr(
        "providers.unified_config._load_yaml",
        lambda _path: {
            "features": {
                "code_resources_enabled": True,
                "code_retrieval": {
                    "enabled": True,
                    "semantic_enabled": False,
                    "bm25_enabled": True,
                },
                "code_retrieval_symbolic_enabled": False,
                "tool_output_filter": {
                    "enabled": True,
                    "read_enabled": False,
                    "bash_enabled": True,
                },
                "tool_output": {"recovery_max_chars": 1234},
            }
        },
    )

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.code_resources_enabled is True
    assert cfg.code_retrieval_enabled is True
    assert cfg.effective_code_retrieval_semantic_enabled is False
    assert cfg.effective_code_retrieval_bm25_enabled is True
    assert cfg.effective_code_retrieval_symbolic_enabled is False
    assert cfg.effective_tool_output_filter_read_enabled is False
    assert cfg.effective_tool_output_filter_bash_enabled is True
    assert cfg.tool_output_recovery_max_chars == 1234


def test_feature_env_round_trip_and_recovery_cap_validation(monkeypatch):
    _clear_feature_env(monkeypatch)
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda _path: {})
    monkeypatch.setenv("OGMEM_CODE_RESOURCES_ENABLED", "true")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("OGMEM_CODE_RETRIEVAL_GRAPH_ENABLED", "false")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_ENABLED", "true")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_BASH_ENABLED", "false")
    monkeypatch.setenv("OGMEM_TOOL_OUTPUT_FILTER_RECOVERY_MAX_CHARS", "777")

    cfg = OgMemConfig.load("ignored.yaml")

    assert cfg.code_resources_enabled is True
    assert cfg.code_retrieval_enabled is True
    assert cfg.effective_code_retrieval_graph_enabled is False
    assert cfg.effective_tool_output_filter_read_enabled is True
    assert cfg.effective_tool_output_filter_bash_enabled is False
    assert cfg.tool_output_recovery_max_chars == 777

    with pytest.raises(ValueError, match="must be positive"):
        OgMemConfig(tool_output_recovery_max_chars=0)


def test_resource_uri_parsers_and_access_are_account_scoped():
    node_uri = "ctx://acct/resources/chunk-1"
    root_uri = "ctx://acct/resources/"
    expected_node = {
        "account": "acct",
        "owner_type": "resources",
        "owner_id": "",
        "category": "resource",
        "slug": "chunk-1",
    }
    expected_root = {**expected_node, "slug": ""}

    assert parse_agfs_uri(node_uri) == expected_node
    assert parse_sql_uri(node_uri) == expected_node
    assert parse_agfs_uri(root_uri) == expected_root
    assert parse_sql_uri(root_uri) == expected_root
    assert uri_to_path(node_uri) == "/accounts/acct/resources/chunk-1/"
    assert build_uri("acct", "resources", "", "resource", "chunk-1") == node_uri

    agfs = AGFSContextFS(None)
    sql = SQLContextFS.__new__(SQLContextFS)
    assert agfs._is_accessible(node_uri, _ctx()) is True
    assert sql._is_accessible(node_uri, _ctx()) is True
    assert agfs._is_accessible(node_uri, _ctx("other")) is False
    assert sql._is_accessible(node_uri, _ctx("other")) is False


def test_index_record_scalar_allowlist_resource_filter_and_no_index_marker():
    metadata = {
        "resource_type": "code",
        "resource_root_slug": "project",
        "root_fingerprint": "f" * 64,
        "file_key": "a" * 64,
        "file_slug": "a" * 32,
        "relative_path": "src/service.py",
        "source_extension": ".py",
        "language": "python",
        "start_line": 1,
        "end_line": 2,
        "symbol": "parse_config",
        "symbol_kind": "function",
        "signature": "parse_config(path)",
        "chunk_hash": "b" * 64,
        "chunk_key": "c" * 64,
        "graph": {"calls": ["open"]},
    }
    records = build_index_records(_code_node(**metadata))

    assert len(records) == 2
    for record in records:
        assert record.filters["resource_type"] == "code"
        assert record.metadata["relative_path"] == "src/service.py"
        assert record.metadata["start_line"] == 1
        assert "graph" not in record.metadata
        chroma_meta = ChromaVectorIndex._record_to_meta(record)
        assert chroma_meta["resource_type"] == "code"
        assert chroma_meta["chunk_key"] == "c" * 64
        assert "graph" not in chroma_meta

    disabled = _code_node(**metadata, indexing_disabled=True)
    assert build_index_records(disabled) == []
