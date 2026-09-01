import os
import uuid

import psycopg2
import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.errors import ConcurrentModificationError
from core.models import ContextNode, RelationEdge, RequestContext
from fs.sql_adapter import SharedConnectionPool, SQLContextFS
from fs.sql_adapter.sql_trace_store import SQLTraceEventStore
from providers.relation_store.sql_relation_store import SQLRelationStore
from session.sql_archive_store import SQLSessionArchiveStore


@pytest.fixture
def opengauss_sql_dsn():
    dsn = os.environ.get("OGMEM_OPENGAUSS_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_OPENGAUSS_SQL_DSN is not configured")
    return dsn


def test_opengauss_sql_context_storage_round_trip(opengauss_sql_dsn):
    account_id = f"ogmem-it-{uuid.uuid4().hex[:12]}"
    ctx = RequestContext(
        account_id=account_id,
        user_id="alice",
        agent_id="main",
        session_id="session-1",
        trace_id="trace-1",
    )
    old_uri = f"ctx://{account_id}/users/alice/memories/profile"
    new_uri = old_uri + "-moved"
    related_uri = f"ctx://{account_id}/users/alice/memories/other"
    pool = SharedConnectionPool(
        opengauss_sql_dsn,
        pool_size=5,
        dialect="opengauss",
    )
    fs = SQLContextFS(pool=pool)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    relations = SQLRelationStore(pool=pool)
    archives = SQLSessionArchiveStore(pool=pool)

    try:
        node = ContextNode(
            uri=old_uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="",
            overview="",
            content="initial",
            metadata={
                "_expected_version": 0,
                "_relations": [
                    RelationEdge(
                        from_uri=old_uri,
                        to_uri=related_uri,
                        relation_type="related_to",
                        weight=0.5,
                        reason="integration",
                    )
                ]
            },
        )
        event = outbox.build_write_event(node)
        fs.write_node_with_outbox(node, ctx, event)
        assert fs.read_node(old_uri, ctx).content == "initial"
        pending = outbox.list_pending(account_id)
        assert len(pending) == 1
        pending_uri, pending_event = pending[0]
        assert outbox.try_acquire(pending_event, pending_uri, "opengauss-it") is True
        outbox.release(pending_event, pending_uri)

        with pytest.raises(ConcurrentModificationError):
            fs.write_node(node, ctx)

        node.metadata.pop("_expected_version")
        node.content = "updated"
        fs.write_node(node, ctx)
        assert fs.read_node(old_uri, ctx).content == "updated"

        relations.upsert_edges(
            [
                RelationEdge(
                    from_uri=old_uri,
                    to_uri=related_uri,
                    relation_type="related_to",
                    weight=0.5,
                    reason="integration",
                )
            ],
            ctx,
        )
        assert relations.get_edges(old_uri, ctx)[0].reason == "integration"

        fs.move_node(old_uri, new_uri, ctx)
        moved = fs.read_node(new_uri, ctx)
        assert moved.content == "updated"
        assert moved.metadata["_relations"][0].from_uri == new_uri

        result = archives.write_archive(
            "session-1",
            "overview",
            "abstract",
            [{"role": "user", "content": "hello"}],
            ctx,
            archive_id="archive-1",
        )
        assert result.success is True
        assert archives.read_archive("session-1", "archive-1", ctx).abstract == "abstract"
    finally:
        conn = pool.get_connection()
        try:
            SQLContextFS.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                for table in (
                    "relation_edges",
                    "outbox_events",
                    "context_nodes",
                    "session_archives",
                ):
                    cur.execute(
                        f"DELETE FROM {table} WHERE account_id = %s",
                        (account_id,),
                    )
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()


def test_opengauss_tenant_isolation_blocks_cross_account_access(opengauss_sql_dsn):
    account_a = f"ogmem-rls-a-{uuid.uuid4().hex[:12]}"
    account_b = f"ogmem-rls-b-{uuid.uuid4().hex[:12]}"
    uri = f"ctx://{account_a}/users/alice/memories/private"
    pool = SharedConnectionPool(opengauss_sql_dsn, dialect="opengauss")
    fs = SQLContextFS(pool=pool)
    fs.write_node(
        ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="private",
            overview="private",
            content="private",
        ),
        RequestContext(
            account_id=account_a,
            user_id="alice",
            agent_id="main",
            session_id="session-rls",
            trace_id="trace-rls",
        ),
    )
    conn = pool.get_connection()
    try:
        SQLContextFS.bind_tenant(conn, account_b)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM context_nodes WHERE uri = %s", (uri,))
            assert cur.fetchone()[0] == 0
            with pytest.raises(psycopg2.Error, match="tenant_write_guard"):
                cur.execute(
                    """
                    INSERT INTO context_nodes
                        (uri, account_id, owner_space, category, context_type,
                         level, status, relations, metadata, version)
                    VALUES (%s, %s, %s, %s, %s, 0, 'ACTIVE', '[]', '{}', 1)
                    """,
                    (
                        "ctx://acct-b/users/bob/memories/blocked",
                        "acct-b",
                        "user:bob",
                        "profile",
                        "MEMORY",
                    ),
                )
        conn.rollback()
    finally:
        SQLContextFS.bind_tenant(conn, account_a)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM context_nodes WHERE uri = %s", (uri,))
        conn.commit()
        pool.return_connection(conn)
        pool.close_all()


def test_opengauss_sql_trace_storage_round_trip(opengauss_sql_dsn):
    account_id = f"ogmem-trace-it-{uuid.uuid4().hex[:12]}"
    owner_space = "user:alice"
    agent_id = "main"
    session_id = "session-1"
    pool = SharedConnectionPool(
        opengauss_sql_dsn,
        pool_size=3,
        dialect="opengauss",
    )
    traces = SQLTraceEventStore(pool=pool)

    try:
        event = traces.insert(
            event_id="turn-end-1",
            account_id=account_id,
            owner_space=owner_space,
            agent_id=agent_id,
            session_id=session_id,
            hook_type="turn_end",
            content={"event_type": "turn_end"},
        )
        finalized = traces.finalize_session(
            account_id=account_id,
            owner_space=owner_space,
            agent_id=agent_id,
            session_id=session_id,
            stop_event_index=event["event_index"],
        )

        assert event["inserted"] is True
        assert finalized["status"] == "finalized"
    finally:
        conn = pool.get_connection()
        try:
            SQLContextFS.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                for table in (
                    "trace_session_events",
                    "trace_sessions",
                    "trace_events",
                ):
                    cur.execute(
                        f"DELETE FROM {table} WHERE account_id = %s",
                        (account_id,),
                    )
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()
