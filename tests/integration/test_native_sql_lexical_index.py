"""Opt-in real-database coverage for openGauss native BM25 retrieval.

Set TEST_OPENGAUSS_CONNECTION_STRING to enable it.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

from core.lexical import LexicalRecord
from providers.lexical_index.native_sql_index import OpenGaussLexicalIndex

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.integration


def _record(record_id: str, text: str, *, account_id: str = "acct") -> LexicalRecord:
    return LexicalRecord(
        id=record_id,
        uri=f"ctx://{account_id}/{record_id}",
        level=0,
        text=text,
        account_id=account_id,
        owner_space="user:u1",
        metadata={"context_type": "memory", "category": "profile"},
    )


def _drop_projection(dsn: str, prefix: str) -> None:
    suffixes = (
        "terms",
        "docs",
        "tombstones",
        "rebuild_docs",
        "rebuilds",
        "scope_epochs",
        "scope_tombstones",
        "metadata",
    )
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        for suffix in suffixes:
            cur.execute(f"DROP TABLE IF EXISTS {prefix}_{suffix} CASCADE")
        cur.execute(f"DROP SEQUENCE IF EXISTS {prefix}_generation_seq CASCADE")


def _explain(dsn: str, sql: str, params=()) -> str:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET enable_seqscan=off")
        cur.execute(f"EXPLAIN {sql}", params)
        return "\n".join(str(row[0]) for row in cur.fetchall())


def test_opengauss_native_bm25_path():
    dsn = os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING")
    if not dsn:
        pytest.skip("TEST_OPENGAUSS_CONNECTION_STRING is not configured")

    native_prefix = f"native_it_og_{uuid.uuid4().hex[:8]}"
    native = None
    try:
        native = OpenGaussLexicalIndex(
            connection_string=dsn,
            table_name=native_prefix,
            provider_name="opengauss",
        )
        native.upsert_many(
            [_record("m1", "coffee database"), _record("m2", "coffee", account_id="other")]
        )
        assert [hit.id for hit in native.search("coffee", 5, {"account_id": "acct"})] == ["m1"]
        native.upsert_many([_record("m1", "tea only")])
        assert native.search("coffee", 5, {"account_id": "acct"}) == []
        rebuild_id = native.begin_rebuild()
        assert native.replace_scopes(
            rebuild_id=rebuild_id,
            account_id="acct",
            owner_spaces=["user:u1"],
            records=[_record("m3", "native coffee retrieval")],
        )
        assert [hit.id for hit in native.search("coffee", 5, {"account_id": "acct"})] == ["m3"]
        native.delete(["m3"])
        assert native.search("coffee", 5, {"account_id": "acct"}) == []
        health = native.health()
        assert health["available"] and health["schema_compatible"] and health["index_ready"]
        assert health["query_timeout_ms"] == 5000
        assert health["engine"] == "opengauss_bm25"
        plan = _explain(
            dsn,
            f"SELECT record_id FROM {native_prefix}_docs "
            "ORDER BY content <&> %s DESC LIMIT 5",
            ("coffee",),
        )
        assert "Index Scan" in plan and "native_bm25" in plan
    finally:
        if native is not None:
            native.close()
        _drop_projection(dsn, native_prefix)


def test_older_inflight_write_cannot_overwrite_published_rebuild():
    dsn = os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING")
    if not dsn:
        pytest.skip("TEST_OPENGAUSS_CONNECTION_STRING is not configured")

    native_prefix = f"native_race_{uuid.uuid4().hex[:8]}"
    older = newer = None
    release = threading.Event()
    worker = rebuild_worker = None
    errors: list[Exception] = []
    try:
        older = OpenGaussLexicalIndex(
            connection_string=dsn,
            table_name=native_prefix,
            provider_name="opengauss",
        )
        newer = OpenGaussLexicalIndex(
            connection_string=dsn,
            table_name=native_prefix,
            provider_name="opengauss",
        )
        entered = threading.Event()
        original_upsert = older._upsert_doc

        def delayed_upsert(cur, **kwargs):
            entered.set()
            if not release.wait(5.0):
                raise RuntimeError("test release timed out")
            return original_upsert(cur, **kwargs)

        older._upsert_doc = delayed_upsert

        def write_older_generation():
            try:
                older.upsert_many([_record("same-id", "older payload")])
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=write_older_generation)
        worker.start()
        assert entered.wait(5.0)

        rebuilt = threading.Event()

        def publish_newer_rebuild():
            try:
                rebuild_id = newer.begin_rebuild()
                assert newer.replace_scopes(
                    rebuild_id=rebuild_id,
                    account_id="acct",
                    owner_spaces=["user:u1"],
                    records=[_record("same-id", "newer snapshot")],
                )
                rebuilt.set()
            except Exception as exc:
                errors.append(exc)

        rebuild_worker = threading.Thread(target=publish_newer_rebuild)
        rebuild_worker.start()
        assert not rebuilt.wait(0.1)
        release.set()
        worker.join(5.0)
        rebuild_worker.join(5.0)

        assert not worker.is_alive()
        assert not rebuild_worker.is_alive()
        assert rebuilt.is_set()
        assert errors == []
        assert older.search("older", 5, {"account_id": "acct"}) == []
        assert [
            hit.id for hit in newer.search("newer", 5, {"account_id": "acct"})
        ] == ["same-id"]
        assert newer.ready_scopes("acct", ["user:u1"]) == frozenset({"user:u1"})
    finally:
        release.set()
        if worker is not None:
            worker.join(5.0)
        if rebuild_worker is not None:
            rebuild_worker.join(5.0)
        for index in (older, newer):
            if index is not None:
                index.close()
        _drop_projection(dsn, native_prefix)


def test_newer_delete_cannot_be_overwritten_by_older_inflight_upsert():
    dsn = os.environ.get("TEST_OPENGAUSS_CONNECTION_STRING")
    if not dsn:
        pytest.skip("TEST_OPENGAUSS_CONNECTION_STRING is not configured")

    native_prefix = f"native_delete_race_{uuid.uuid4().hex[:8]}"
    older = newer = None
    release = threading.Event()
    worker = delete_worker = None
    errors: list[Exception] = []
    try:
        older = OpenGaussLexicalIndex(
            connection_string=dsn,
            table_name=native_prefix,
            provider_name="opengauss",
        )
        newer = OpenGaussLexicalIndex(
            connection_string=dsn,
            table_name=native_prefix,
            provider_name="opengauss",
        )
        entered = threading.Event()
        original_upsert = older._upsert_doc

        def delayed_upsert(cur, **kwargs):
            entered.set()
            if not release.wait(5.0):
                raise RuntimeError("test release timed out")
            return original_upsert(cur, **kwargs)

        older._upsert_doc = delayed_upsert

        def write_older_generation():
            try:
                older.upsert_many([_record("same-id", "stale older payload")])
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=write_older_generation)
        worker.start()
        assert entered.wait(5.0)

        deleted = threading.Event()

        def delete_newer_generation():
            try:
                newer.delete(["same-id"])
                deleted.set()
            except Exception as exc:
                errors.append(exc)

        delete_worker = threading.Thread(target=delete_newer_generation)
        delete_worker.start()
        assert not deleted.wait(0.1)
        release.set()
        worker.join(5.0)
        delete_worker.join(5.0)

        assert not worker.is_alive()
        assert not delete_worker.is_alive()
        assert deleted.is_set()
        assert errors == []
        assert newer.search("stale", 5, {"account_id": "acct"}) == []
    finally:
        release.set()
        for thread in (worker, delete_worker):
            if thread is not None:
                thread.join(5.0)
        for index in (older, newer):
            if index is not None:
                index.close()
        _drop_projection(dsn, native_prefix)
