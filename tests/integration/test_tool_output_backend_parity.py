from __future__ import annotations

import os
import uuid

import pytest

from commit.outbox_store import OutboxStore
from commit.sql_outbox_store import SQLOutboxStore
from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import RequestContext
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from index.repair_job import AGFSSnapshot, OutboxEventSink, RepairJob
from service.tool_output import ToolOutputRepository


def _ctx(account_id: str, agent_id: str) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="tool-output-user",
        agent_id=agent_id,
        session_id=f"session-{uuid.uuid4().hex[:12]}",
        trace_id=f"trace-{uuid.uuid4().hex[:12]}",
    )


def _store_original(repository: ToolOutputRepository, ctx: RequestContext) -> str:
    digest = "a" * 64
    return repository.store_original(
        ctx,
        slug=f"tool_output_bash_{uuid.uuid4().hex}",
        content_digest=digest,
        original="complete tool output\n" * 80,
        filtered="complete tool output\n",
        tool_name="Bash",
        strategy="bash_passthrough_below_min_size",
        metadata={
            "bash_category": "test_runner",
            "exit_code": 0,
            "original_lines": 80,
            "filtered_lines": 1,
            "was_shortened": True,
        },
    )


class _SingleNodeAGFSSnapshot(AGFSSnapshot):
    """Expose one production AGFS node through the RepairJob snapshot contract."""

    def __init__(self, fs, uri: str) -> None:
        self._fs = fs
        self._client = fs._client
        self._uri = uri

    def list_nodes_under_account(self, account_id: str) -> list[str]:
        expected_prefix = f"ctx://{account_id}/"
        return [self._uri] if self._uri.startswith(expected_prefix) else []

    def read_file(self, uri: str, filename: str) -> str | None:
        if uri != self._uri:
            return None
        try:
            raw = self._client.read(self._path(filename))
        except Exception:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def write_file(self, uri: str, filename: str, content: str) -> None:
        if uri != self._uri:
            raise ValueError("unexpected repair URI")
        self._client.write(self._path(filename), content.encode("utf-8"))

    def list_files(self, uri: str) -> list[str]:
        if uri != self._uri:
            return []
        entries = self._client.ls(self._fs._uri_to_agfs_path(uri))
        return sorted(
            str(entry.get("name") or "")
            for entry in entries
            if not entry.get("is_dir", entry.get("isDir", False))
        )

    def _path(self, filename: str) -> str:
        return self._fs._uri_to_agfs_path(self._uri) + filename


class _ForbiddenRepairEventSink(OutboxEventSink):
    def create_event(self, event) -> None:
        raise AssertionError(f"RepairJob emitted an event for tool output: {event.event_id}")


@pytest.mark.integration
def test_agfs_tool_output_round_trip_has_no_outbox_or_repair_event(agfs, agfs_client):
    account_id = f"ogmemory-tool-output-agfs-{uuid.uuid4().hex[:12]}"
    ctx = _ctx(account_id, "agent-a")
    repository = ToolOutputRepository(agfs)
    uri = _store_original(repository, ctx)
    outbox = OutboxStore(
        client=agfs_client,
        fs=agfs,
        mount_prefix=agfs._mount_prefix,
    )

    assert repository.read_original(ctx, uri).startswith("complete tool output")
    assert outbox.list_pending(account_id) == []

    summary = RepairJob(
        _SingleNodeAGFSSnapshot(agfs, uri),
        _ForbiddenRepairEventSink(),
    ).scan_account(account_id)
    assert summary.total_nodes_scanned == 1
    assert summary.nodes_skipped == 1
    assert summary.outbox_events_created == 0
    assert outbox.list_pending(account_id) == []

    agfs.delete_node(uri, ctx)
    with pytest.raises(NodeNotFoundError):
        repository.read_original(ctx, uri)


@pytest.mark.integration
def test_sql_tool_output_round_trip_is_current_agent_only_and_has_no_outbox():
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN") or os.environ.get("TEST_SQL_CONNECTION_STRING")
    if not dsn:
        pytest.skip("OGMEM_TEST_SQL_DSN is required for real SQL tool-output parity")

    account_id = f"ogmemory-tool-output-sql-{uuid.uuid4().hex[:12]}"
    owner_ctx = _ctx(account_id, "agent-a")
    other_agent_ctx = _ctx(account_id, "agent-b")
    pool = SharedConnectionPool(dsn, pool_size=2)
    fs = SQLContextFS(pool=pool)
    repository = ToolOutputRepository(fs)
    outbox = SQLOutboxStore(pool=pool, fs=fs)
    uri = ""
    try:
        uri = _store_original(repository, owner_ctx)

        assert repository.read_original(owner_ctx, uri).startswith("complete tool output")
        assert outbox.list_pending(account_id) == []
        with pytest.raises(AccessDeniedError):
            repository.read_original(other_agent_ctx, uri)

        fs.delete_node(uri, owner_ctx)
        with pytest.raises(NodeNotFoundError):
            repository.read_original(owner_ctx, uri)
        assert outbox.list_pending(account_id) == []
    finally:
        if uri and fs.exists(uri, owner_ctx):
            fs.delete_node(uri, owner_ctx)
        pool.close_all()
