"""SQL archive and session ownership integration checks."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import psycopg2
import pytest

from core.models import Role
from fs.sql_adapter.session_mutation import SessionMutationLostError, active_session_guard
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.session_manager import SessionManager
from session.session_state import SessionState
from tests.integration.test_session_mutation_sql import ctx as ctx
from tests.integration.test_session_mutation_sql import database as database


@pytest.mark.parametrize("method", ["commit", "commit_snapshot"])
@pytest.mark.parametrize("required", [True, False])
def test_archive_only_real_sql_manager_rejects_unowned_commit(database, ctx, method, required):
    _, make = database
    fs, _, archive = make()
    held, release = threading.Event(), threading.Event()

    def owner():
        with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            held.set()
            assert release.wait(10)

    manager = SessionManager(get_archive_store=lambda: archive, archive_store_required=required,
                             session_state_bridge_enabled=False)
    manager._compress = Mock(side_effect=AssertionError("unguarded archive reached compression"))
    manager.add_message(ctx.session_id, "user", "Original archive source.", ctx, message_id="source")
    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        args = dict(session_id=ctx.session_id, ctx=ctx, wait=False, archive_id="stable-archive")
        if method == "commit_snapshot":
            args["snapshot"] = list(manager.get_or_create(ctx.session_id).messages)
        try:
            with pytest.raises(SessionMutationLostError, match="requires its guard"):
                getattr(manager, method)(**args)
            assert not holding.done()
            assert archive.list_archives(ctx.session_id, ctx) == []
            assert len(manager.get_or_create(ctx.session_id).messages) == 1
            manager._compress.assert_not_called()
        finally:
            release.set()
            holding.result(timeout=5)


@pytest.mark.parametrize("method", ["commit", "commit_snapshot"])
def test_guarded_sql_manager_keeps_archive_and_acknowledged_state(database, ctx, method):
    _, make = database
    fs, _, archive = make()
    manager = SessionManager(get_archive_store=lambda: archive, archive_store_required=True,
                             get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager._compress = Mock(return_value=("Original source", "Original source"))
    manager.add_message(ctx.session_id, "user", "Original archive source.", ctx, message_id="source")
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        args = dict(session_id=ctx.session_id, ctx=ctx, wait=False, archive_id="stable-archive")
        if method == "commit_snapshot":
            args["snapshot"] = list(manager.get_or_create(ctx.session_id).messages)
        response = getattr(manager, method)(**args)
        assert response["status"] == "completed" and response["archived"]
    assert len(archive.list_archives(ctx.session_id, ctx)) == 1
    assert fs.exists(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx)












def _seed_archives(fs, archive, ctx):
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        for index in range(2):
            assert archive.write_archive(session_id=ctx.session_id, ctx=ctx, archive_id=f"seed-{index}",
                overview="Original overview", abstract="Original abstract",
                messages=[{"role": "user", "content": f"Original source {index}"}]).success


def test_optional_auto_merge_late_sql_getter_rejects_before_first_list(database, ctx):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    generic, current = Mock(), []
    generic.list_archives.return_value = []
    current.append(generic)

    def generic_write(**kwargs):
        current[0] = archive
        return Mock(success=True, uri="generic://archive", archive_id=kwargs["archive_id"])

    generic.write_archive.side_effect = generic_write
    manager = SessionManager(get_archive_store=lambda: current[0], archive_merge_threshold=2, archive_max_count=1)
    manager._compress = Mock(return_value=("New original source", "New original source"))
    manager.add_message(ctx.session_id, "user", "New original source.", ctx, message_id="new-source")
    attempted, held, release = threading.Event(), threading.Event(), threading.Event()
    original_merge = manager._merge_archives_once
    original_list = archive.list_archives
    archive.list_archives = Mock(wraps=original_list)
    archive.write_archive = Mock(wraps=archive.write_archive)

    def merge_once(*args):
        try:
            return original_merge(*args)
        finally:
            attempted.set()

    manager._merge_archives_once = merge_once

    def owner():
        with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
            held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        try:
            result = manager.commit(ctx.session_id, ctx, wait=True, archive_id="generic-new")
            assert result["status"] == "completed"
            assert attempted.wait(5) and not holding.done()
            archive.list_archives.assert_not_called()
            archive.write_archive.assert_not_called()
            assert len(original_list(ctx.session_id, ctx)) == 2
        finally:
            release.set()
            holding.result(timeout=5)


@pytest.mark.parametrize("method", ["delete_session_archives", "delete_persisted_session_state"])
def test_direct_sql_cleanup_rejects_while_another_session_owner_is_active(database, ctx, method):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    manager = SessionManager(get_archive_store=lambda: archive, get_context_fs=lambda: fs)
    manager.add_message(ctx.session_id, "user", "Original state source.", ctx, message_id="state-source")
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    with fs.session_mutation_lock(uri, ctx):
        assert manager.save_session_state(ctx.session_id, ctx)
    before = fs.read_node(uri, ctx).content
    if method == "delete_session_archives":
        # The reported bypass is the public fixed SQL getter with no ContextFS.
        manager = SessionManager(get_archive_store=lambda: archive)
    held, release = threading.Event(), threading.Event()

    def owner():
        with fs.session_mutation_lock(uri, ctx):
            held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as workers:
        holding = workers.submit(owner)
        assert held.wait(5)
        try:
            with pytest.raises(SessionMutationLostError):
                getattr(manager, method)(ctx.session_id, ctx)
            assert not holding.done()
            assert len(archive.list_archives(ctx.session_id, ctx)) == 2
            assert fs.read_node(uri, ctx).content == before
        finally:
            release.set()
            holding.result(timeout=5)
    with fs.session_mutation_lock(uri, ctx):
        assert getattr(manager, method)(ctx.session_id, ctx) == (2 if method == "delete_session_archives" else 1)


@pytest.mark.parametrize("fail_mark", [False, True])
def test_actual_sql_merge_all_mutations_keep_one_guarded_store_and_connection(database, ctx, fail_mark):
    _, make = database
    fs, _, archive = make()
    _seed_archives(fs, archive, ctx)
    manager = SessionManager(get_archive_store=lambda: archive, archive_merge_threshold=2, archive_max_count=1)
    uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    seen = []
    with fs.session_mutation_lock(uri, ctx) as guard:
        original_borrow = archive._get_connection

        def borrow():
            assert active_session_guard() is guard and threading.get_ident() == guard.thread_id
            connection = original_borrow()
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                seen.append(cursor.fetchone()[0])
            return connection

        archive._get_connection = borrow
        if fail_mark:
            original_mark = archive.mark_archive_merged
            calls = []

            def mark(session, archive_id, request, merged_id):
                calls.append(archive_id)
                return original_mark(session, archive_id, request, merged_id) if len(calls) == 1 else False

            archive.mark_archive_merged = mark
        assert manager._merge_archives_once(ctx.session_id, ctx) is not fail_mark
        guard.ensure_owned()
        active = archive.list_archives(ctx.session_id, ctx)
        if fail_mark:
            assert {entry.archive_id for entry in active} == {"seed-0", "seed-1"}
        else:
            assert len(active) == 1 and active[0].archive_id.startswith("merged_")
        assert len(seen) >= 6 and len(set(seen)) == 1


@pytest.mark.parametrize("field", ["account_id", "user_id", "agent_id", "session_id"])
@pytest.mark.parametrize("method", ["_merge_archives_once", "delete_session_archives"])
def test_archive_mutation_helper_rejects_full_context_mismatch_before_access(database, ctx, field, method):
    _, make = database
    fs, _, archive = make()
    manager = SessionManager(get_archive_store=lambda: archive)
    request = replace(ctx, **{field: "foreign-identity"})
    archive.list_archives = Mock(side_effect=AssertionError("mismatched mutation reached first list"))
    archive.delete_session_archives = Mock(side_effect=AssertionError("mismatched mutation reached delete"))
    with pytest.raises(SessionMutationLostError), fs.session_mutation_lock(
            SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        getattr(manager, method)(request.session_id, request)
    archive.list_archives.assert_not_called()
    archive.delete_session_archives.assert_not_called()
