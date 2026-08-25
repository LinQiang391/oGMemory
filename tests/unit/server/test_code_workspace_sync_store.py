"""Unit tests for protected workspace-sync control-plane persistence.

The local-store test uses the real JSON-backed ``ControlPlaneStore``.  SQL
tests use a mocked connection/cursor in place of PostgreSQL, so they verify
SQL construction and serialization only, not a database integration path. A
small fake AGFS client exercises only the atomic mutex primitive contract, not
an AGFS backend integration path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from server.control_plane_store import ControlPlaneStore
from server.sql_control_plane_store import SQLControlPlaneStore


def test_local_workspace_sync_state_is_copied_and_enumerated_by_registration(tmp_path):
    store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    first = {
        "root_id": "root-a",
        "workspace_path": "/private/a",
        "dirty": True,
        "snapshot": {"a.py": {"mtime_ns": 1, "size": 1}},
    }
    second = {
        "root_id": "root-b",
        "workspace_path": "/private/b",
        "dirty": False,
        "snapshot": {},
    }

    store.write_code_workspace_sync_state("acct-a", "root-a", first)
    store.write_code_workspace_sync_state("acct-b", "root-b", second)
    returned = store.read_code_workspace_sync_state("acct-a", "root-a")
    assert returned is not None
    returned["snapshot"]["a.py"]["size"] = 99

    stored = store.read_code_workspace_sync_state("acct-a", "root-a")
    assert stored is not None
    assert stored["snapshot"]["a.py"]["size"] == 1
    assert store.list_code_workspace_sync_states("acct-a") == [("root-a", first)]
    assert store.list_code_workspace_sync_registrations() == [
        ("acct-a", "root-a", first),
        ("acct-b", "root-b", second),
    ]


class FakeAtomicAGFSClient:
    """Minimal AGFS double for the short control-plane mutex protocol."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.created: list[str] = []
        self.writes: list[tuple[str, bytes]] = []
        self.removed: list[str] = []
        self.directories: list[str] = []
        self.read_overrides: dict[str, bytes] = {}
        self.read_sequences: dict[str, list[bytes]] = {}
        self.stats: dict[str, dict[str, object]] = {}
        self.on_write = None

    def create(self, path: str) -> None:
        if path in self.files:
            raise RuntimeError("already exists")
        self.files[path] = b""
        self.stats[path] = {"modTime": datetime.now(UTC).isoformat()}
        self.created.append(path)

    def read(self, path: str) -> bytes:
        sequence = self.read_sequences.get(path)
        if sequence:
            return sequence.pop(0)
        if path in self.read_overrides:
            return self.read_overrides[path]
        try:
            return self.files[path]
        except KeyError as exc:
            raise RuntimeError("not found") from exc

    def write(self, path: str, data: bytes) -> None:
        self.files[path] = data
        self.stats.setdefault(path, {"modTime": datetime.now(UTC).isoformat()})
        self.writes.append((path, data))
        if self.on_write is not None:
            self.on_write(path, data)

    def rm(self, path: str) -> None:
        self.files.pop(path, None)
        self.stats.pop(path, None)
        self.removed.append(path)

    def mkdir(self, path: str) -> None:
        self.directories.append(path)

    def stat(self, path: str) -> dict[str, object]:
        try:
            return self.stats[path]
        except KeyError as exc:
            raise RuntimeError("not found") from exc


def test_agfs_workspace_sync_mutex_uses_atomic_create_and_matching_token():
    """Unit-only: fake AGFS verifies mutex ownership, not backend parity."""

    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: {"root_id": "root-a", "version": 1, "dirty": True},
    )

    state_path = store.code_workspace_sync_path("acct", "root-a")
    lock_path = state_path + ".lock"
    lock_writes = [(path, payload) for path, payload in client.writes if path == lock_path]
    assert result["dirty"] is True
    assert client.created == [lock_path]
    assert len(lock_writes) == 1
    lock_marker = json.loads(lock_writes[0][1].decode("utf-8"))
    assert lock_marker["kind"] == "code_workspace_sync_mutex"
    assert lock_marker["token"]
    assert lock_path in client.removed
    assert lock_path not in client.files
    assert json.loads(client.files[state_path].decode("utf-8"))["root_id"] == "root-a"


def test_agfs_workspace_sync_mutation_fails_closed_without_atomic_create():
    store = ControlPlaneStore(mount_prefix="/control", client=object())

    with pytest.raises(RuntimeError, match="atomic workspace sync state locking"):
        store.mutate_code_workspace_sync_state(
            "acct",
            "root-a",
            lambda current: {"root_id": "root-a"},
        )


def test_agfs_workspace_sync_mutex_never_releases_a_replaced_token():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    lock_path = store.code_workspace_sync_path("acct", "root-a") + ".lock"
    state_path = store.code_workspace_sync_path("acct", "root-a")

    def replace_after_state_write(path: str, _data: bytes) -> None:
        if path == state_path:
            client.read_overrides[lock_path] = json.dumps(
                {
                    "kind": "code_workspace_sync_mutex",
                    "token": "other-worker",
                    "expires_at": 9999999999,
                }
            ).encode("utf-8")

    client.on_write = replace_after_state_write

    store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: {"root_id": "root-a"},
    )

    assert lock_path not in client.removed


def test_agfs_workspace_sync_mutex_reclaims_stale_uninitialized_lock():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    state_path = store.code_workspace_sync_path("acct", "root-a")
    lock_path = state_path + ".lock"
    client.files[lock_path] = b""
    client.stats[lock_path] = {
        "modTime": (datetime.now(UTC) - timedelta(seconds=6)).isoformat()
    }

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda _current: {"root_id": "root-a", "dirty": True},
    )

    assert result["dirty"] is True
    assert lock_path in client.removed
    assert json.loads(client.files[state_path].decode("utf-8"))["root_id"] == "root-a"


def test_agfs_workspace_sync_mutex_recognizes_its_token_after_transient_read():
    client = FakeAtomicAGFSClient()
    store = ControlPlaneStore(mount_prefix="/control", client=client)
    lock_path = store.code_workspace_sync_path("acct", "root-a") + ".lock"
    client.read_sequences[lock_path] = [b""]

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda _current: {"root_id": "root-a", "dirty": True},
    )

    assert result["dirty"] is True
    assert lock_path in client.removed


def _sql_store_with_cursor(cursor: Mock, connection: Mock) -> SQLControlPlaneStore:
    store = SQLControlPlaneStore.__new__(SQLControlPlaneStore)
    store._get_connection = lambda: connection
    store._return_connection = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=cursor)
    context.__exit__ = Mock(return_value=False)
    connection.cursor.return_value = context
    return store


def test_sql_workspace_sync_transition_locks_row_and_serializes_lease():
    """Unit-only: a mocked cursor checks the SQL transition contract."""

    cursor = Mock()
    connection = Mock()
    store = _sql_store_with_cursor(cursor, connection)
    expires_at = datetime.fromtimestamp(150.0, UTC)
    cursor.fetchone.return_value = ({"root_id": "root-a", "dirty": True}, "old", expires_at)
    observed_current = []

    result = store.mutate_code_workspace_sync_state(
        "acct",
        "root-a",
        lambda current: (
            observed_current.append(current)
            or {
                **current,
                "dirty": False,
                "lease_owner": "worker-new",
                "lease_expires_at": 300.0,
            }
        ),
    )

    queries = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("ON CONFLICT (account_id, root_id) DO NOTHING" in query for query in queries)
    assert any("FOR UPDATE" in query for query in queries)
    update_call = cursor.execute.call_args_list[-1]
    assert "to_timestamp(%s)" in update_call.args[0]
    assert update_call.args[1][1:3] == ("worker-new", 300.0)
    assert observed_current == [
        {
            "root_id": "root-a",
            "dirty": True,
            "lease_owner": "old",
            "lease_expires_at": 150.0,
        }
    ]
    assert result["lease_owner"] == "worker-new"
    assert result["lease_expires_at"] == 300.0
    connection.commit.assert_called_once_with()
    store._return_connection.assert_called_once_with(connection)


def test_sql_workspace_sync_registration_enumeration_restores_lease_fields():
    """Unit-only: mocked SQL rows prove restart-recovery serialization."""

    cursor = Mock()
    connection = Mock()
    store = _sql_store_with_cursor(cursor, connection)
    cursor.fetchall.return_value = [
        (
            "acct-a",
            "root-a",
            {"workspace_path": "/private/a"},
            "worker-a",
            datetime.fromtimestamp(250.0, UTC),
        ),
        (
            "acct-b",
            "root-b",
            {"workspace_path": "/private/b"},
            None,
            None,
        ),
    ]

    registrations = store.list_code_workspace_sync_registrations()

    assert registrations == [
        (
            "acct-a",
            "root-a",
            {
                "workspace_path": "/private/a",
                "lease_owner": "worker-a",
                "lease_expires_at": 250.0,
            },
        ),
        (
            "acct-b",
            "root-b",
            {
                "workspace_path": "/private/b",
                "lease_owner": None,
                "lease_expires_at": None,
            },
        ),
    ]
    query = cursor.execute.call_args.args[0]
    assert "ORDER BY account_id, root_id" in query
    connection.rollback.assert_called_once_with()
