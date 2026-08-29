from fs.sql_adapter import pool as pool_module
from fs.sql_adapter import schema as schema_module
from fs.sql_adapter.dialect import resolve_sql_dialect
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.postgresql_dialect import PostgreSQLDialect


class _Cursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchone(self):
        return ("PG",)


class _Connection:
    def __init__(self):
        self.autocommit = False
        self.cursor_instance = _Cursor()

    def cursor(self):
        return self.cursor_instance


def test_postgresql_schema_is_applied_through_dialect():
    conn = _Connection()
    schema_module._schema_ensured.clear()

    schema_module.ensure_schema(conn, "postgresql")

    schema_sql = conn.cursor_instance.statements[0][0]
    assert "current_setting('app.account_id', true)" in schema_sql
    assert conn.autocommit is False
    schema_module._schema_ensured.clear()


def test_shared_pool_exposes_postgresql_dialect(monkeypatch):
    monkeypatch.setattr(pool_module, "_HAS_PSYCOPG2", True)

    pool = SharedConnectionPool("dbname=test")

    assert pool.dialect == "postgresql"
    assert isinstance(pool.sql_dialect, PostgreSQLDialect)


def test_dialect_factory_returns_postgresql_implementation():
    dialect = resolve_sql_dialect("postgres")

    assert isinstance(dialect, PostgreSQLDialect)


def test_workspace_root_sql_is_owned_by_postgresql_dialect():
    cursor = _Cursor()

    PostgreSQLDialect().ensure_workspace_root(cursor, "acct", "root")

    assert "ON CONFLICT" in cursor.statements[0][0]


def test_postgresql_declares_notify_capability():
    cursor = _Cursor()
    dialect = PostgreSQLDialect()

    dialect.notify_outbox(cursor, "ogmem_outbox", "{}")

    assert dialect.supports_batch_claim is True
    assert "pg_notify" in cursor.statements[0][0]
