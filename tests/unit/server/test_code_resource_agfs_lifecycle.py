"""AGFS-specific unit coverage for durable code RESOURCE deletion.

The filesystem and Outbox objects are protocol doubles.  These tests verify
lifecycle ordering only; real AGFS cooperation is covered by integration tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from commit.code_resource_events import build_code_resource_delete_event
from commit.outbox_store import OutboxPublishOutcomeUnknownError
from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext
from service.code_resource_lifecycle import CodeResourceLifecycleService


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _node() -> ContextNode:
    return ContextNode(
        uri="ctx://acct/resources/" + "a" * 32,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="code",
        overview="path: module.py",
        content="value = 1\n",
        metadata={
            "resource_type": "code",
            "resource_root_slug": "repo",
            "root_fingerprint": "f" * 64,
            "file_key": "b" * 64,
            "file_slug": "b" * 32,
            "relative_path": "module.py",
            "chunk_key": "c" * 64,
        },
    )


class _Lease:
    def ensure_owned(self) -> None:
        return None


class _AGFSDeleteFS:
    def __init__(self, node: ContextNode):
        self.nodes = {node.uri: node}
        self.tombstones: set[str] = set()
        self.payload_deletes: list[str] = []

    def list_children(self, uri, ctx):
        return sorted(self.nodes)

    def read_node(self, uri, ctx):
        try:
            return self.nodes[uri]
        except KeyError as exc:
            raise NodeNotFoundError(uri) from exc

    def exists(self, uri, ctx):
        return uri in self.nodes

    def is_tombstoned(self, uri, ctx):
        return uri in self.tombstones

    @contextmanager
    def resource_mutation_lock(self, uri, ctx):
        yield _Lease()

    def write_delete_tombstone(self, uri, ctx, payload, *, mutation_lease):
        mutation_lease.ensure_owned()
        assert payload["delete_event_id"]
        assert len(payload["record_ids"]) == 3
        self.tombstones.add(uri)

    def clear_delete_tombstone(self, uri, ctx, *, mutation_lease):
        mutation_lease.ensure_owned()
        self.tombstones.discard(uri)

    def delete_node_payload(self, uri, ctx, *, mutation_lease):
        mutation_lease.ensure_owned()
        self.payload_deletes.append(uri)
        self.nodes.pop(uri, None)


class _FailingOutbox:
    def __init__(self, error: Exception):
        self.error = error
        self.cancelled: list[str] = []
        self.calls: list[str] = []

    @staticmethod
    def build_delete_event(uri):
        return build_code_resource_delete_event(uri)

    def cancel_pending_upserts(self, uri):
        self.calls.append("cancel")
        self.cancelled.append(uri)
        return []

    def persist_event(self, event, uri):
        self.calls.append("persist")
        raise self.error


@pytest.mark.parametrize(
    ("error", "tombstone_retained"),
    [
        (RuntimeError("delete outbox unavailable"), False),
        (
            OutboxPublishOutcomeUnknownError("publish outcome unknown"),
            True,
        ),
    ],
)
def test_agfs_delete_publish_failure_never_removes_payload(
    error,
    tombstone_retained,
):
    node = _node()
    fs = _AGFSDeleteFS(node)
    outbox = _FailingOutbox(error)
    service = CodeResourceLifecycleService(
        context_fs=fs,
        outbox_store=outbox,
        config=SimpleNamespace(code_resources_enabled=True),
    )

    with pytest.raises(type(error), match=str(error)):
        service.delete(slug="repo", ctx=_ctx())

    assert node.uri in fs.nodes
    assert (node.uri in fs.tombstones) is tombstone_retained
    assert fs.payload_deletes == []
    assert outbox.cancelled == []
    assert outbox.calls == ["persist"]
