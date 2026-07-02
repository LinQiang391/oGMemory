from __future__ import annotations

import pytest

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext, Role
from providers.unified_config import OgMemConfig
from server.auth import AuthorizationError
from server.memory_service import MemoryService


class _FakeContextFS:
    def __init__(self, nodes):
        self.nodes = {node.uri: node for node in nodes}
        self.archived = []

    def read_node(self, uri, ctx):
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return self.nodes[uri]

    def archive_node(self, uri, ctx):
        self.read_node(uri, ctx)
        self.archived.append(uri)
        self.nodes[uri].metadata["status"] = "ARCHIVED"

    def write_node(self, node, ctx):
        self.nodes[node.uri] = node
        self.nodes[node.uri].metadata["status"] = "ACTIVE"


class _FakeEvent:
    event_id = "evt-1"


class _FakeOutbox:
    def __init__(self):
        self.archived = []

    def register_archive(self, uri, ctx):
        self.archived.append((uri, ctx))
        return _FakeEvent()


class _FailingOutbox(_FakeOutbox):
    def register_archive(self, uri, ctx):
        super().register_archive(uri, ctx)
        raise RuntimeError("outbox unavailable")


class _FakeAudit:
    def __init__(self):
        self.records = []

    def record(self, account_id, **kwargs):
        self.records.append({"account_id": account_id, **kwargs})


def _ctx(*, role=Role.MEMBER, agent_id="bot") -> RequestContext:
    return RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id=agent_id,
        session_id="s1",
        trace_id="t1",
        role=role,
    )


def _node(uri, *, category="preference", owner_space="user:alice"):
    return ContextNode(
        uri=uri,
        context_type="MEMORY",
        category=category,
        level=0,
        owner_space=owner_space,
        abstract="abstract",
        overview="overview",
        content="content",
        metadata={},
    )


def _service(fs, outbox=None, audit=None):
    service = MemoryService(config=OgMemConfig())
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda _fs: outbox
    service.get_audit_service = lambda: audit or _FakeAudit()
    return service


def test_forget_memory_archives_registers_outbox_and_audit():
    uri = "ctx://acct-1/users/alice/memories/preferences/coding_style"
    fs = _FakeContextFS([_node(uri)])
    outbox = _FakeOutbox()
    audit = _FakeAudit()
    service = _service(fs, outbox=outbox, audit=audit)

    payload = service.forget_memory(
        uri,
        _ctx(),
        external_uri="ogmemory://memories/preference/coding_style",
        reason="stale",
        client_request_id="req-1",
    )

    assert payload["ok"] is True
    assert payload["mode"] == "archive"
    assert fs.archived == [uri]
    assert outbox.archived[0][0] == uri
    assert audit.records[0]["action"] == "mcp.delete_memory"
    assert audit.records[0]["details"]["reason"] == "stale"
    assert audit.records[0]["details"]["client_request_id"] == "req-1"


def test_forget_memory_restores_node_when_outbox_registration_fails():
    uri = "ctx://acct-1/users/alice/memories/preferences/coding_style"
    node = _node(uri)
    fs = _FakeContextFS([node])
    service = _service(fs, outbox=_FailingOutbox())

    with pytest.raises(RuntimeError, match="outbox unavailable"):
        service.forget_memory(uri, _ctx())

    assert fs.archived == [uri]
    assert fs.nodes[uri].metadata["status"] == "ACTIVE"


def test_forget_memory_rejects_profile():
    uri = "ctx://acct-1/users/alice/memories/profile"
    service = _service(_FakeContextFS([_node(uri, category="profile")]))

    with pytest.raises(AuthorizationError, match="profile memory"):
        service.forget_memory(uri, _ctx())


def test_forget_memory_requires_admin_for_agent_scoped_memory():
    uri = "ctx://acct-1/agents/bot/memories/tools/grep"
    service = _service(_FakeContextFS([_node(uri, category="tool", owner_space="agent:bot")]))

    with pytest.raises(AuthorizationError, match="agent-scoped"):
        service.forget_memory(uri, _ctx(role=Role.MEMBER))


def test_forget_memory_allows_admin_for_agent_scoped_memory():
    uri = "ctx://acct-1/agents/bot/memories/tools/grep"
    fs = _FakeContextFS([_node(uri, category="tool", owner_space="agent:bot")])
    service = _service(fs)

    payload = service.forget_memory(uri, _ctx(role=Role.ADMIN))

    assert payload["deleted"] is True
    assert fs.archived == [uri]
