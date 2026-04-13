"""Tests for V1 audit service."""

from __future__ import annotations

from pathlib import Path

from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore


def test_audit_service_records_and_reads_events(tmp_path: Path):
    store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    audit = AuditService(store)

    event = audit.record(
        "acct-1",
        actor="alice",
        target="user:bob",
        action="role_changed",
        result="success",
        trace_id="trace-1",
        details={"role": "admin"},
    )

    rows = audit.list_logs("acct-1")
    fetched = audit.get_log("acct-1", event.log_id)

    assert len(rows) == 1
    assert rows[0]["action"] == "role_changed"
    assert fetched is not None
    assert fetched["details"]["role"] == "admin"
