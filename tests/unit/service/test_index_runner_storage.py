from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.run_index_service import _create_storage


def test_sql_runner_uses_sql_outbox_and_shared_pool():
    cfg = SimpleNamespace(
        storage_backend="sql", sql_connection_string="test-dsn",
        sql_pool_size=3, sql_dialect="postgresql",
    )
    with (
        patch("fs.sql_adapter.pool.SharedConnectionPool") as pool,
        patch("fs.sql_adapter.SQLContextFS") as fs,
        patch("commit.sql_outbox_store.SQLOutboxStore") as outbox,
    ):
        result = _create_storage(cfg)
    pool.assert_called_once_with("test-dsn", pool_size=3, dialect="postgresql")
    fs.assert_called_once_with(pool=pool.return_value)
    outbox.assert_called_once_with(pool=pool.return_value, fs=fs.return_value)
    assert result == (fs.return_value, outbox.return_value, pool.return_value)


def test_sql_runner_rejects_missing_dsn_instead_of_using_agfs():
    with pytest.raises(ValueError, match="storage.connection_string"):
        _create_storage(SimpleNamespace(storage_backend="sql", sql_connection_string=""))


def test_agfs_runner_keeps_configured_mount_and_outbox():
    cfg = SimpleNamespace(
        storage_backend="agfs", agfs_base_url="http://agfs:8080",
        agfs_mount_prefix="/memory",
    )
    with (
        patch("pyagfs.AGFSClient") as client,
        patch("fs.agfs_adapter.agfs_context_fs.AGFSContextFS") as fs,
        patch("commit.outbox_store.OutboxStore") as outbox,
    ):
        result = _create_storage(cfg)
    client.assert_called_once_with(api_base_url="http://agfs:8080")
    fs.assert_called_once_with(client=client.return_value, mount_prefix="/memory")
    outbox.assert_called_once_with(
        client=client.return_value, fs=fs.return_value, mount_prefix="/memory",
    )
    assert result == (fs.return_value, outbox.return_value, None)
