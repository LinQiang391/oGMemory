from __future__ import annotations

from unittest.mock import MagicMock

from core.lexical import LexicalRecord
from providers.lexical_index import sql_index as module
from providers.lexical_index.sql_index import (
    PreparedSQLLexicalRecord,
    SQLNativeProjectionStore,
)


class DuplicateKeyError(Exception):
    pass


def _index() -> SQLNativeProjectionStore:
    index = SQLNativeProjectionStore.__new__(SQLNativeProjectionStore)
    index._table = "lex"
    return index


def _record() -> LexicalRecord:
    return LexicalRecord(
        id="mem-1",
        uri="ctx://acct/users/u1/memories/mem-1",
        level=0,
        text="coffee",
        account_id="acct",
        owner_space="user:u1",
        metadata={"context_type": "memory", "category": "profile"},
    )


def test_owned_pool_is_closed_when_schema_initialization_fails(monkeypatch):
    class FakePool:
        def __init__(self, *args, **kwargs):
            self.close_calls = 0

        def close_all(self):
            self.close_calls += 1
            return True

    pool = FakePool()
    monkeypatch.setattr(module, "LexicalConnectionPool", lambda *args, **kwargs: pool)
    monkeypatch.setattr(
        SQLNativeProjectionStore,
        "_ensure_schema",
        lambda self: (_ for _ in ()).throw(RuntimeError("incompatible schema")),
    )

    try:
        SQLNativeProjectionStore(connection_string="dbname=test")
    except RuntimeError as exc:
        assert str(exc) == "incompatible schema"
    else:
        raise AssertionError("schema initialization should fail")

    assert pool.close_calls == 1


def test_schema_error_is_preserved_when_owned_pool_cleanup_also_fails(monkeypatch):
    class FakePool:
        def close_all(self):
            raise RuntimeError("cleanup-secondary")

    monkeypatch.setattr(module, "LexicalConnectionPool", lambda *args, **kwargs: FakePool())
    monkeypatch.setattr(
        SQLNativeProjectionStore,
        "_ensure_schema",
        lambda self: (_ for _ in ()).throw(RuntimeError("schema-original")),
    )

    try:
        SQLNativeProjectionStore(connection_string="dbname=test")
    except RuntimeError as exc:
        assert str(exc) == "schema-original"
    else:
        raise AssertionError("schema failure should be propagated")


def test_generation_uses_sequence_without_locking_metadata_singleton():
    index = _index()
    cursor = MagicMock()
    cursor.fetchone.return_value = (42,)

    assert index._next_generation(cursor) == 42

    cursor.execute.assert_called_once_with(
        "SELECT nextval(%s)", (index._generation_sequence,)
    )


def test_mark_scopes_dirty_locks_accounts_before_allocating_generation():
    index = _index()
    connection = MagicMock()
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    index._ensure_compatible = lambda _cursor: None
    calls = []
    index._acquire_mutation_locks = lambda _cursor, **kwargs: calls.append(
        ("lock", tuple(kwargs["account_ids"]))
    )
    index._next_generation = lambda _cursor: calls.append(("generation",)) or 7
    index._upsert_scope_epoch = lambda *_args, **_kwargs: calls.append(("upsert",))

    index.mark_scopes_dirty((("acct-b", "user:u2"), ("acct-a", "user:u1")))

    assert calls == [
        ("lock", ("acct-b", "acct-a")),
        ("generation",),
        ("upsert",),
        ("upsert",),
    ]
    connection.commit.assert_called_once_with()


def test_mark_all_scopes_dirty_uses_monotonic_database_updates():
    index = _index()
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    index._ensure_compatible = lambda _cursor: None
    index._next_generation = lambda _cursor: 7

    index.mark_all_scopes_dirty()

    statements = [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]
    assert any(
        "GREATEST(all_scopes_dirty_generation,%s)" in statement
        for statement in statements
    )
    assert any(
        "dirty_generation=GREATEST(dirty_generation,%s)" in statement
        for statement in statements
    )


def test_schema_initialization_acquires_transaction_lock_before_ddl():
    index = _index()
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = RuntimeError("stop-after-first-statement")
    index._borrow = lambda: connection
    index._return = lambda _connection: None

    try:
        index._ensure_schema()
    except RuntimeError as exc:
        assert str(exc) == "stop-after-first-statement"
    else:
        raise AssertionError("schema initialization should stop")

    cursor.execute.assert_called_once_with(
        "SELECT pg_advisory_xact_lock(%s)",
        (module._schema_lock_key("lex"),),
    )
    connection.rollback.assert_called_once_with()


def test_doc_upsert_duplicate_insert_race_retries_update(monkeypatch):
    index = _index()
    index._record_write_is_superseded = lambda *_args: False
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    monkeypatch.setattr(module.psycopg2, "IntegrityError", DuplicateKeyError)
    update_count = 0

    def execute(sql, params=None):
        nonlocal update_count
        if f"UPDATE {index._docs_table}" in sql:
            update_count += 1
            cursor.rowcount = 0 if update_count == 1 else 1
        elif f"INSERT INTO {index._docs_table}" in sql:
            raise DuplicateKeyError("duplicate")

    cursor.execute.side_effect = execute

    inserted = index._upsert_doc(
        cursor,
        record=_record(),
        metadata_json="{}",
        generation=2,
    )

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT lexical_doc_insert" in executed
    assert "RELEASE SAVEPOINT lexical_doc_insert" in executed
    assert update_count == 2
    assert inserted is False


def test_doc_upsert_reports_successful_insert_for_record_count():
    index = _index()
    index._record_write_is_superseded = lambda *_args: False
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.rowcount = 0

    assert index._upsert_doc(
        cursor,
        record=_record(),
        metadata_json="{}",
        generation=2,
    ) is True


def test_doc_upsert_ignores_write_older_than_live_projection():
    index = _index()
    index._record_write_is_superseded = lambda *_args: False
    cursor = MagicMock()
    cursor.rowcount = 0
    cursor.fetchone.return_value = (9,)

    assert index._upsert_doc(
        cursor,
        record=_record(),
        metadata_json="{}",
        generation=2,
    ) is False

    statements = [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]
    assert any("projection_generation<=%s" in statement for statement in statements)
    assert not any(statement.startswith(f"INSERT INTO {index._docs_table}") for statement in statements)


def test_record_write_is_blocked_by_newer_scope_or_record_tombstone():
    index = _index()
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)

    assert index._record_write_is_superseded(cursor, _record(), 2) is True

    sql, params = cursor.execute.call_args.args
    assert index._tombstones_table in sql
    assert index._scope_tombstones_table in sql
    assert params == ("mem-1", 2, "acct", "user:u1", 2)


def test_mutation_locks_accounts_before_records_in_stable_order():
    index = _index()
    cursor = MagicMock()

    index._acquire_mutation_locks(
        cursor,
        account_ids=("other", "acct", "acct"),
        record_ids=("z", "a", "a"),
    )

    assert [call.args[1] for call in cursor.execute.call_args_list] == [
        (module._schema_lock_key("lex:mutation:account:acct"),),
        (module._schema_lock_key("lex:mutation:account:other"),),
        (module._schema_lock_key("lex:mutation:record:a"),),
        (module._schema_lock_key("lex:mutation:record:z"),),
    ]


def test_record_account_lookup_includes_live_and_staged_rebuild_documents():
    index = _index()
    cursor = MagicMock()
    cursor.fetchall.return_value = [("acct",), ("other",)]

    assert index._record_accounts(cursor, ("a", "b")) == ("acct", "other")

    sql, params = cursor.execute.call_args.args
    assert index._docs_table in sql
    assert index._rebuild_docs_table in sql
    assert params == ("a", "b", "a", "b")


def test_scope_tombstone_duplicate_insert_race_retries_update(monkeypatch):
    index = _index()
    cursor = MagicMock()
    monkeypatch.setattr(module.psycopg2, "IntegrityError", DuplicateKeyError)
    update_count = 0

    def execute(sql, params=None):
        nonlocal update_count
        if f"UPDATE {index._scope_tombstones_table}" in sql:
            update_count += 1
            cursor.rowcount = 0 if update_count == 1 else 1
        elif f"INSERT INTO {index._scope_tombstones_table}" in sql:
            raise DuplicateKeyError("duplicate")

    cursor.execute.side_effect = execute

    index._upsert_scope_tombstone(cursor, "acct", "user:u1", 3)

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT lexical_scope_tombstone_insert" in executed
    assert "RELEASE SAVEPOINT lexical_scope_tombstone_insert" in executed
    assert update_count == 2


def test_record_tombstone_generation_never_moves_backwards():
    index = _index()
    cursor = MagicMock()
    cursor.rowcount = 1

    index._upsert_record_tombstone(cursor, "mem-1", 3)

    sql, params = cursor.execute.call_args.args
    assert "GREATEST(projection_generation,%s)" in sql
    assert params == (3, "mem-1")


def test_record_delete_does_not_remove_newer_projection():
    index = _index()
    cursor = MagicMock()
    cursor.rowcount = 0

    assert index._delete_record(cursor, "mem-1", 3) is False

    sql, params = cursor.execute.call_args.args
    assert "projection_generation<=%s" in sql
    assert params == ("mem-1", 3)


def test_scope_epoch_duplicate_insert_race_retries_update(monkeypatch):
    index = _index()
    cursor = MagicMock()
    monkeypatch.setattr(module.psycopg2, "IntegrityError", DuplicateKeyError)
    update_count = 0

    def execute(sql, params=None):
        nonlocal update_count
        if f"UPDATE {index._scope_epochs_table}" in sql:
            update_count += 1
            cursor.rowcount = 0 if update_count == 1 else 1
        elif f"INSERT INTO {index._scope_epochs_table}" in sql:
            raise DuplicateKeyError("duplicate")

    cursor.execute.side_effect = execute

    index._upsert_scope_epoch(
        cursor,
        "acct",
        "user:u1",
        rebuild_generation=4,
        dirty_generation=0,
    )

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT lexical_scope_epoch_insert" in executed
    assert "RELEASE SAVEPOINT lexical_scope_epoch_insert" in executed
    assert update_count == 2


def test_ensure_scope_epoch_duplicate_insert_race_is_idempotent(monkeypatch):
    index = _index()
    cursor = MagicMock()
    monkeypatch.setattr(module.psycopg2, "IntegrityError", DuplicateKeyError)
    fetch_count = 0

    def fetchone():
        nonlocal fetch_count
        fetch_count += 1
        return None

    def execute(sql, params=None):
        if f"INSERT INTO {index._scope_epochs_table}" in sql:
            raise DuplicateKeyError("duplicate")

    cursor.fetchone.side_effect = fetchone
    cursor.execute.side_effect = execute

    index._ensure_scope_epoch_exists(cursor, "acct", "user:u1")

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert "ROLLBACK TO SAVEPOINT lexical_scope_epoch_ensure" in executed
    assert "RELEASE SAVEPOINT lexical_scope_epoch_ensure" in executed


def test_rebuild_publish_uses_set_based_copy_with_newer_write_guards():
    index = _index()
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    index._ensure_compatible = lambda _cursor: None
    index._ensure_scope_epoch_exists = lambda *_args: None
    index._upsert_scope_epoch = lambda *_args, **_kwargs: None
    index._discard_rebuild = lambda *_args: None
    index._adjust_record_count = lambda *_args: None
    index._prune_tombstones = lambda *_args: None
    cursor.fetchone.side_effect = [(7, 8), (0,), None]
    cursor.fetchall.return_value = [("user:u1",)]

    def execute(sql, _params=None):
        normalized = " ".join(sql.split())
        if normalized.startswith(f"DELETE FROM {index._docs_table}"):
            cursor.rowcount = 3
        elif normalized.startswith(f"INSERT INTO {index._docs_table}"):
            cursor.rowcount = 2

    cursor.execute.side_effect = execute

    assert index._publish_rebuild("rebuild", "acct", ("user:u1",)) is True

    statements = [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]
    publish = next(
        statement
        for statement in statements
        if statement.startswith(f"INSERT INTO {index._docs_table}")
    )
    assert f"FROM {index._rebuild_docs_table}" in publish
    assert f"FROM {index._tombstones_table}" in publish
    assert f"FROM {index._scope_tombstones_table}" in publish
    assert "NOT EXISTS" in publish


def test_rebuild_staging_uses_bounded_driver_batch(monkeypatch):
    index = _index()
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    index._ensure_compatible = lambda _cursor: None
    cursor.fetchone.return_value = (1,)
    calls = []
    monkeypatch.setattr(
        module,
        "execute_batch",
        lambda cur, statement, params, page_size: calls.append(
            (cur, statement, params, page_size)
        ),
    )
    prepared = PreparedSQLLexicalRecord(_record(), "{}", "coffee")

    index._stage_rebuild_batch("rebuild", [prepared])

    assert len(calls) == 1
    assert calls[0][0] is cursor
    assert f"DELETE FROM {index._rebuild_docs_table}" in calls[0][1]
    assert f"INSERT INTO {index._rebuild_docs_table}" in calls[0][1]
    assert calls[0][2][0][:4] == ("rebuild", "mem-1", "rebuild", "mem-1")
    assert calls[0][3] == 100
    connection.commit.assert_called_once_with()


def test_stale_rebuild_expiry_uses_the_database_clock():
    index = _index()
    index._rebuild_lease_seconds = 3600
    cursor = MagicMock()

    index._expire_stale_rebuilds(cursor)

    assert cursor.execute.call_count == 2
    for sql, params in (call.args for call in cursor.execute.call_args_list):
        assert "CURRENT_TIMESTAMP" in sql
        assert "interval '1 second'" in sql
        assert params == (3600,)


def test_delete_scope_is_set_based_and_records_scope_tombstone():
    index = _index()
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    index._borrow = lambda: connection
    index._return = lambda _connection: None
    index._ensure_compatible = lambda _cursor: None
    index._next_generation = lambda _cursor: 9
    index._scope_where = lambda *_args: (
        "account_id=%s AND owner_space=%s",
        ("acct", "user:u1"),
    )
    index._upsert_scope_tombstone = MagicMock()
    index._adjust_record_count = MagicMock()
    index._prune_tombstones = MagicMock()

    def execute(sql, _params=None):
        if f"DELETE FROM {index._docs_table}" in sql:
            cursor.rowcount = 4

    cursor.execute.side_effect = execute

    assert index.delete_scope(account_id="acct", owner_space="user:u1") == 4

    statements = [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]
    assert sum(statement.startswith(f"DELETE FROM {index._docs_table}") for statement in statements) == 1
    delete_statement = next(
        statement
        for statement in statements
        if statement.startswith(f"DELETE FROM {index._docs_table}")
    )
    assert "projection_generation<=%s" in delete_statement
    assert not any("SELECT record_id" in statement for statement in statements)
    index._upsert_scope_tombstone.assert_called_once_with(
        cursor, "acct", "user:u1", 9
    )
    index._adjust_record_count.assert_called_once_with(cursor, -4)
    connection.commit.assert_called_once_with()


def test_replace_scopes_preserves_publish_error_when_abort_also_fails():
    index = _index()
    index._max_write_batch_size = 10
    index._publish_rebuild = MagicMock(side_effect=RuntimeError("publish-original"))
    index.abort_rebuild = MagicMock(side_effect=RuntimeError("abort-secondary"))

    try:
        index.replace_scopes(
            rebuild_id="rebuild",
            account_id="acct",
            owner_spaces=("user:u1",),
            records=(),
        )
    except RuntimeError as exc:
        assert str(exc) == "publish-original"
    else:
        raise AssertionError("publish failure should be propagated")

    index.abort_rebuild.assert_called_once_with("rebuild")
