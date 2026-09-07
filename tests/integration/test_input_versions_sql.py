"""Plan revision checks using SQL memories and existing session archives."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from commit.input_versions import collect_input_versions, find_changed_inputs
from core.models import ContextNode, RequestContext, Role
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore
from server.tenant_admin import TenantAdminService
from session.sql_archive_store import SQLSessionArchiveStore


@pytest.mark.integration
def test_sql_input_versions_and_archive_evidence(tmp_path):
    dsn = os.environ.get("OGMEMORY_EVIDENCE_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEMORY_EVIDENCE_TEST_DSN to an isolated test database")
    run_id = uuid4().hex[:12]
    run_root = Path(os.environ.get("OGMEMORY_EVIDENCE_TEST_OUTPUT", str(tmp_path))) / run_id
    store = ControlPlaneStore("/revisions", local_root=str(run_root / "control-plane"))
    keys = APIKeyManager(store)
    account_id, user_id, agent_id = (f"{kind}-{run_id}" for kind in ("dream01", "user", "agent"))
    admin_key = keys.create_account(account_id, user_id)
    ctx = RequestContext(account_id, user_id, agent_id, run_id, run_id, Role.ADMIN)
    admin = TenantAdminService(keys, store, AuditService(store))
    admin.create_agent(ctx, account_id, agent_id, owner_user_id=user_id)
    assert keys.resolve_user_key(admin_key, account_id)["user_id"] == user_id
    assert admin.get_agent(ctx, account_id, agent_id)["owner_user_id"] == user_id

    pool = SharedConnectionPool(dsn, pool_size=2)
    try:
        fs = SQLContextFS(pool=pool)
        archives = SQLSessionArchiveStore(pool=pool)
        message = {"id": "message-1", "role": "user", "content": "Project A releases on Friday."}
        archive = archives.write_archive(run_id, "Release schedule", "Project A", [message], ctx)
        assert archive.success
        nodes = []
        for kind, owner_id in (("user", user_id), ("agent", agent_id)):
            node = ContextNode(
                uri=f"ctx://{account_id}/{kind}s/{owner_id}/memories/events/release",
                context_type="MEMORY",
                category="events",
                level=4,
                owner_space=f"{kind}:{owner_id}",
                abstract="Project A release",
                overview="Release schedule",
                content=message["content"],
                metadata={"_expected_version": 0, "provenance_ids": [archive.uri]},
            )
            fs.write_node(node, ctx)
            nodes.append(fs.read_node(node.uri, ctx))
        original = collect_input_versions(nodes)
        assert original == {node.uri: 1 for node in nodes}
        assert find_changed_inputs(fs, original, ctx) == {}

        updated = nodes[0]
        updated.content = "Project A releases on Saturday."
        correction = archives.write_archive(
            run_id,
            "Updated release schedule",
            "Project A",
            [{**message, "content": updated.content}],
            ctx,
        )
        assert correction.success
        updated.metadata["provenance_ids"].append(correction.uri)
        updated.metadata["_expected_version"] = original[updated.uri]
        fs.write_node(updated, ctx)
        changed = find_changed_inputs(fs, original, ctx)
        assert changed == {updated.uri: 2}
        refreshed_nodes = [fs.read_node(node.uri, ctx) for node in nodes]
        refreshed = collect_input_versions(refreshed_nodes)
        assert refreshed == {nodes[0].uri: 2, nodes[1].uri: 1}
        assert find_changed_inputs(fs, refreshed, ctx) == {}
        source = archives.read_archive(run_id, archive.archive_id, ctx)
        assert source.messages == [message]
        assert refreshed_nodes[0].metadata["provenance_ids"] == [archive.uri, correction.uri]
    finally:
        pool.close_all()

    reopened_pool = SharedConnectionPool(dsn, pool_size=2)
    try:
        fs = SQLContextFS(pool=reopened_pool)
        archives = SQLSessionArchiveStore(pool=reopened_pool)
        persisted = json.loads(json.dumps(original))
        assert find_changed_inputs(fs, persisted, ctx) == changed
        assert find_changed_inputs(fs, refreshed, ctx) == {}
        assert archives.read_archive(run_id, archive.archive_id, ctx).messages == [message]
        report = {
            "account_id": account_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "input_versions": original,
            "changed_inputs": changed,
            "refreshed_versions": refreshed,
            "source_archives": [archive.uri, correction.uri],
            "original_archive_read": "passed",
            "persistent_version_check": "passed",
        }
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "input-versions-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Revision verification: {run_root / 'input-versions-report.json'}")
    finally:
        reopened_pool.close_all()
