"""Unit tests for ``m3.config.load_m3_runtime_config`` graph-store DSN resolution.

The graph store (graph_nodes / graph_edges) is a separate role from the
M3 vector index. The resolution order is:

1. ``m3.graph_store.connection_string``  (explicit M3 override)
2. ``storage.connection_string``         (default; same DB as the rest
                                          of the SQL state, usually the
                                          pgvector Postgres)
3. ``None``                              (CLI drivers' --connection-string
                                          still wins at runtime via
                                          PipelineRunner.build_context)

The legacy fallback to ``vector_db.connection_string`` was removed:
that was what glued the M3 graph store to the openGauss container even
when STORAGE_DB pointed elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from m3.config import load_m3_runtime_config


@pytest.fixture()
def fake_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a YAML fixture and point load_m3_runtime_config at it.

    Each test patches the YAML body to control which blocks are set.
    The path is exposed via the ``OGMEM_CONFIG`` env var so
    ``load_m3_runtime_config`` picks it up.
    """
    cfg = tmp_path / "ogmemory.yaml"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OGMEM_CONFIG", str(cfg))
    yield cfg


def _write(cfg_path: Path, body: str) -> None:
    cfg_path.write_text(body, encoding="utf-8")


def test_default_is_no_graph_dsn(fake_config_path):
    """Empty YAML → ``graph_connection_string`` is None.

    Callers who want SQL graph storage must pass ``--connection-string``
    or set ``m3.graph_store.connection_string`` / ``storage.connection_string``.
    """
    runtime = load_m3_runtime_config(str(fake_config_path))
    assert runtime.graph_connection_string is None
    # Vector index also has no DSN — the legacy coupling is gone.
    assert runtime.vector_index_connection_string is None


def test_storage_connection_string_used_when_m3_block_absent(fake_config_path):
    """Default: graph store follows storage.

    With STORAGE_DB_* wired in by the container entrypoint, ``storage``
    resolves to a Postgres DSN and the graph store rides along.
    """
    _write(
        fake_config_path,
        """
storage:
  backend: sql
  connection_string: "host=127.0.0.1 port=31875 dbname=ogmem user=ogmem password=x"
vector_db:
  type: opengauss
  connection_string: "host=opengauss port=5432 dbname=postgres user=gaussdb password=y"
""",
    )
    runtime = load_m3_runtime_config(str(fake_config_path))
    # Graph store → storage DB (the pgvector Postgres).
    assert runtime.graph_connection_string == (
        "host=127.0.0.1 port=31875 dbname=ogmem user=ogmem password=x"
    )
    # Vector index → its own DB (openGauss), independently.
    assert runtime.vector_index_connection_string == (
        "host=opengauss port=5432 dbname=postgres user=gaussdb password=y"
    )


def test_m3_graph_store_overrides_storage(fake_config_path):
    """``m3.graph_store.connection_string`` wins over storage."""
    _write(
        fake_config_path,
        """
storage:
  connection_string: "host=storage dbname=ogmem"
m3:
  graph_store:
    connection_string: "host=dedicated-graph dbname=ogmem_graph"
""",
    )
    runtime = load_m3_runtime_config(str(fake_config_path))
    assert runtime.graph_connection_string == "host=dedicated-graph dbname=ogmem_graph"


def test_legacy_vector_db_fallback_is_removed(fake_config_path):
    """The graph store does NOT fall back to ``vector_db.connection_string``.

    Setting only ``vector_db.connection_string`` (the old single-DB
    topology) yields ``graph_connection_string=None``. Callers that
    actually want the graph store on the same DB as the vector index
    must set ``m3.graph_store.connection_string`` explicitly.
    """
    _write(
        fake_config_path,
        """
vector_db:
  connection_string: "host=opengauss port=5432 dbname=postgres"
""",
    )
    runtime = load_m3_runtime_config(str(fake_config_path))
    assert runtime.graph_connection_string is None
    # Vector index still reads it.
    assert runtime.vector_index_connection_string == (
        "host=opengauss port=5432 dbname=postgres"
    )


def test_explicit_m3_graph_store_can_match_vector_db(fake_config_path):
    """Opt-in to the legacy single-DB topology by setting the DSN explicitly."""
    shared_dsn = "host=opengauss port=5432 dbname=postgres"
    _write(
        fake_config_path,
        f"""
storage:
  connection_string: "host=storage port=5432 dbname=ogmem"
vector_db:
  connection_string: "{shared_dsn}"
m3:
  graph_store:
    connection_string: "{shared_dsn}"
""",
    )
    runtime = load_m3_runtime_config(str(fake_config_path))
    # Both end up on openGauss because the user asked for it explicitly.
    assert runtime.graph_connection_string == shared_dsn
    assert runtime.vector_index_connection_string == shared_dsn


def test_m3_graph_store_empty_string_falls_through(fake_config_path):
    """An empty ``m3.graph_store.connection_string`` does NOT shadow storage.

    ``_pick`` treats empty strings as unset (see m3/config.py), so a
    placeholder block with no value behaves as if the block were absent.
    """
    _write(
        fake_config_path,
        """
storage:
  connection_string: "host=storage dbname=ogmem"
m3:
  graph_store:
    connection_string: ""
""",
    )
    runtime = load_m3_runtime_config(str(fake_config_path))
    assert runtime.graph_connection_string == "host=storage dbname=ogmem"
