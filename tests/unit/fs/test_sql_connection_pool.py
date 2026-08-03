from fs.sql_adapter import pool as module
from fs.sql_adapter.pool import SharedConnectionPool


class _Connection:
    closed = 0

    def close(self):
        self.closed = 1


class _StaleConnection(_Connection):
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _sql):
            raise RuntimeError("stale socket")

    def cursor(self):
        return self._Cursor()


def test_new_connection_has_bounded_connect_timeout(monkeypatch):
    calls = []
    connection = _Connection()

    def connect(dsn, **kwargs):
        calls.append((dsn, kwargs))
        return connection

    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    monkeypatch.setattr(module.psycopg2, "connect", connect)
    pool = SharedConnectionPool("dbname=test", connect_timeout_seconds=7)

    borrowed = pool.get_connection()

    assert borrowed is connection
    assert calls == [("dbname=test", {"connect_timeout": 7})]
    pool.return_connection(borrowed)
    pool.close_all()


def test_connect_timeout_must_be_positive(monkeypatch):
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)

    try:
        SharedConnectionPool("dbname=test", connect_timeout_seconds=0)
    except ValueError as exc:
        assert str(exc) == "connect_timeout_seconds must be positive"
    else:
        raise AssertionError("zero timeout should be rejected")


def test_default_borrow_discards_stale_idle_connection(monkeypatch):
    stale = _StaleConnection()
    fresh = _Connection()
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    monkeypatch.setattr(module.psycopg2, "connect", lambda *_args, **_kwargs: fresh)
    pool = SharedConnectionPool("dbname=test")
    pool.return_connection(stale)

    assert pool.get_connection() is fresh
    assert stale.closed == 1


def test_bounded_retrying_caller_can_skip_idle_preflight(monkeypatch):
    stale = _StaleConnection()
    monkeypatch.setattr(module, "_HAS_PSYCOPG2", True)
    pool = SharedConnectionPool("dbname=test")
    pool.return_connection(stale)

    assert pool.get_connection(validate=False) is stale
