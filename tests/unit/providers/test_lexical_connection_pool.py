from providers.lexical_index.connection_pool import LexicalConnectionPool


class _Cursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.last = (sql, params)

    def fetchone(self):
        return (1,)


class _Connection:
    def __init__(self):
        self.closed = 0
        self.commits = 0

    def cursor(self):
        return _Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        self.closed = 1


def test_lexical_pool_is_bounded_reusable_and_provider_local():
    calls = []

    def connect(dsn, **kwargs):
        calls.append((dsn, kwargs))
        return _Connection()

    pool = LexicalConnectionPool(
        "host=gauss",
        1,
        connect,
        connect_timeout_seconds=7,
        session_statement_timeout_ms=5000,
    )
    first = pool.get_connection()
    pool.return_connection(first)
    second = pool.get_connection()

    assert second is first
    assert calls == [("host=gauss", {"connect_timeout": 7})]
    pool.return_connection(second)
    assert pool.close_all(timeout=1.0) is True
    assert first.closed == 1


def test_failed_acquisition_does_not_decrement_another_borrower():
    calls = 0

    def connect(_dsn, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("connect failed")
        return _Connection()

    pool = LexicalConnectionPool("host=gauss", 2, connect)
    borrowed = pool.get_connection()

    try:
        pool.get_connection()
    except RuntimeError as exc:
        assert str(exc) == "connect failed"
    else:
        raise AssertionError("second acquisition should fail")

    assert pool.close_all(timeout=0) is False
    pool.return_connection(borrowed)
    assert pool.close_all(timeout=0) is True
