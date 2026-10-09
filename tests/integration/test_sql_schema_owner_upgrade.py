"""Canonical-schema upgrades on explicitly configured disposable SQL databases."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql

from commit.sql_outbox_store import SQLOutboxStore
from commit.write_receipts import RECEIPTS_KEY, operation_digest
from core.errors import AccessDeniedError, ConcurrentModificationError
from core.models import CandidateMemory, ContextNode, RequestContext
from fs.sql_adapter import schema as schema_module
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from session.session_manager import SessionManager
from session.session_state import SessionState
from session.sql_archive_store import SQLSessionArchiveStore


@pytest.fixture(params=["postgresql", "opengauss"])
def schema_database(request):
    dialect = request.param
    dsn = os.environ.get(f"OGMEM_SCHEMA_{dialect.upper()}_TEST_DSN")
    if not dsn:
        pytest.skip(f"configure OGMEM_SCHEMA_{dialect.upper()}_TEST_DSN for a disposable database")
    name = "schema_upgrade_" + uuid4().hex
    control = psycopg2.connect(dsn)
    control.autocommit = True
    with control.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    isolated_dsn = psycopg2.extensions.make_dsn(dsn, options=f"-c search_path={name}")
    prior_ensured = set(schema_module._schema_ensured)
    schema_module._schema_ensured.clear()
    pool = SharedConnectionPool(isolated_dsn, pool_size=1, dialect=dialect)
    try:
        yield dialect, name, pool
    finally:
        pool.close_all()
        schema_module._schema_ensured.clear()
        schema_module._schema_ensured.update(prior_ensured)
        with control.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
        control.close()


def reapply_schema(pool, dialect):
    schema_module._schema_ensured.clear()
    SQLContextFS(pool=pool)


def owner_columns(pool, schema_name):
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name, data_type, character_maximum_length, is_nullable, column_default "
                "FROM information_schema.columns WHERE table_schema=%s AND table_name='session_archives' "
                "AND column_name IN ('owner_user_id', 'owner_agent_id') ORDER BY column_name",
                (schema_name,),
            )
            return cursor.fetchall()
    finally:
        conn.rollback()
        pool.return_connection(conn)


def assert_owner_columns(columns):
    assert [row[0] for row in columns] == ["owner_agent_id", "owner_user_id"]
    assert all(row[1:4] == ("character varying", 128, "NO") and "''" in row[4] for row in columns)


def test_new_schema_and_reapplication_keep_owner_column_contract(schema_database):
    dialect, name, pool = schema_database
    SQLContextFS(pool=pool)
    first = owner_columns(pool, name)
    assert_owner_columns(first)
    reapply_schema(pool, dialect)
    assert owner_columns(pool, name) == first


def test_legacy_archive_upgrade_preserves_rows_and_adds_both_owners(schema_database):
    dialect, name, pool = schema_database
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE session_archives (archive_id VARCHAR(64) NOT NULL, "
                "session_id VARCHAR(128) NOT NULL, account_id VARCHAR(128) NOT NULL, "
                "abstract TEXT NOT NULL DEFAULT '', overview TEXT NOT NULL DEFAULT '', "
                "messages JSONB NOT NULL DEFAULT '[]', metadata JSONB NOT NULL DEFAULT '{}', "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "PRIMARY KEY (account_id, session_id, archive_id))"
            )
            cursor.execute(
                "INSERT INTO session_archives (archive_id,session_id,account_id,abstract,messages) "
                "VALUES ('legacy-archive','legacy-session','legacy-account','keep-me','[]')"
            )
        conn.commit()
    finally:
        pool.return_connection(conn)
    SQLSessionArchiveStore(pool=pool)
    assert_owner_columns(owner_columns(pool, name))
    reapply_schema(pool, dialect)
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, "legacy-account")
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT abstract, owner_user_id, owner_agent_id FROM session_archives "
                "WHERE archive_id='legacy-archive'"
            )
            assert cursor.fetchone() == ("keep-me", "", "")
    finally:
        conn.rollback()
        pool.return_connection(conn)


def test_unrelated_ddl_failure_is_not_acknowledged_as_schema_success(schema_database):
    dialect, _, pool = schema_database
    conn = pool.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TABLE session_archives (archive_id VARCHAR(64))")
        conn.commit()
    finally:
        pool.return_connection(conn)
    with pytest.raises(RuntimeError) as error:
        SQLContextFS(pool=pool)
    assert isinstance(error.value.__cause__, psycopg2.Error)
    assert error.value.__cause__.pgcode == "42703"
    assert not schema_module._schema_ensured


class NoModel:
    def __getattr__(self, name):
        raise AssertionError(f"SQL schema/component validation must not call a model: {name}")
