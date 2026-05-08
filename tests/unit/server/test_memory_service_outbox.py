"""Unit tests for MemoryService outbox worker wiring."""

from unittest.mock import Mock, patch

from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def test_start_outbox_worker_uses_sql_notify_listener(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    fake_fs = Mock()
    fake_outbox = Mock()
    fake_outbox.supports_batch_claim = True

    monkeypatch.setattr(service, "_get_context_fs", lambda: fake_fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs: fake_outbox)

    fake_listener = Mock()
    fake_listener_cls = Mock(return_value=fake_listener)
    monkeypatch.setattr(
        "index.sql_notify_listener.SQLNotifyListener",
        fake_listener_cls,
    )

    thread_state = {}

    class FakeThread:
        def __init__(self, target, daemon, name):
            thread_state["target"] = target
            thread_state["daemon"] = daemon
            thread_state["name"] = name

        def start(self):
            thread_state["started"] = True

    monkeypatch.setattr("server.memory_service.threading.Thread", FakeThread)

    service._start_outbox_worker()

    fake_listener_cls.assert_called_once()
    assert service._outbox_listener is fake_listener
    assert thread_state["target"] == fake_listener.run_forever
    assert thread_state["daemon"] is True
    assert thread_state["name"] == "outbox-listener"
    assert thread_state["started"] is True


def test_shared_providers_use_instance_config(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    fake_index = Mock()
    fake_embedder = Mock()
    create_vector_index = Mock(return_value=fake_index)
    create_embedder = Mock(return_value=fake_embedder)
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", create_vector_index)
    monkeypatch.setattr(service._provider_cfg, "create_embedder", create_embedder)

    with patch(
        "server.memory_service.ProviderConfig.from_env",
        side_effect=AssertionError("should not consult global config"),
    ):
        assert service._get_shared_vector_index() is fake_index
        assert service._get_shared_embedder() is fake_embedder

    create_vector_index.assert_called_once_with()
    create_embedder.assert_called_once_with()


def test_shutdown_stops_listener_and_joins_thread():
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    listener = Mock()
    thread = Mock()
    thread.is_alive.return_value = True

    service._outbox_listener = listener
    service._outbox_thread = thread

    service.shutdown(join_timeout=0.25)

    listener.stop.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=0.25)
    assert service._outbox_listener is None
    assert service._outbox_thread is None


def test_health_skips_agfs_probe_in_sql_mode(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    agfs_client = Mock(side_effect=AssertionError("AGFS probe should be skipped"))
    monkeypatch.setattr("server.memory_service.AGFSClient", agfs_client)

    # Mock the SQL pool to avoid actual connection attempts
    mock_pool = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
    mock_conn.rollback = Mock()
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=mock_pool))

    health = service.health()

    assert health["status"] == "ok"
    assert health["storage_backend"] == "sql"
    assert health["agfs"] is False
    assert health["sql"] == "connected"
    agfs_client.assert_not_called()


def test_health_probes_agfs_in_agfs_mode(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="agfs",
        agfs_base_url="http://localhost:1833",
    )
    service = MemoryService(config=cfg)

    client = Mock()
    client.ls.return_value = []
    agfs_client = Mock(return_value=client)
    monkeypatch.setattr("server.memory_service.AGFSClient", agfs_client)

    health = service.health()

    assert health["status"] == "ok"
    assert health["storage_backend"] == "agfs"
    assert health["agfs"] is True
    agfs_client.assert_called_once_with(api_base_url="http://localhost:1833")
    client.ls.assert_called_once_with("/")
