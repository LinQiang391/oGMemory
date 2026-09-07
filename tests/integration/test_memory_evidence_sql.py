"""SQL evidence lifecycle with fresh control-plane identities."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from core.models import ContextNode, RequestContext, Role
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS
from retrieval.context_reader import ContextReader
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore
from server.tenant_admin import TenantAdminService
from service.memory_evidence import MemoryEvidenceService


@pytest.mark.integration
def test_sql_evidence_lifecycle(tmp_path):
    dsn = os.environ.get("OGMEMORY_EVIDENCE_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEMORY_EVIDENCE_TEST_DSN to an isolated test database")
    run_id = uuid4().hex[:12]
    run_root = Path(os.environ.get("OGMEMORY_EVIDENCE_TEST_OUTPUT", str(tmp_path))) / run_id
    store = ControlPlaneStore("/evidence", local_root=str(run_root / "control-plane"))
    keys = APIKeyManager(store)
    account_id = f"dream01-{run_id}"
    user_id = f"user-{run_id}"
    agent_id = f"agent-{run_id}"
    admin_key = keys.create_account(account_id, user_id)
    ctx = RequestContext(account_id, user_id, agent_id, run_id, run_id, Role.ADMIN)
    admin = TenantAdminService(keys, store, AuditService(store))
    admin.create_agent(ctx, account_id, agent_id, owner_user_id=user_id)
    assert keys.resolve_user_key(admin_key, account_id)["user_id"] == user_id
    assert admin.get_agent(ctx, account_id, agent_id)["owner_user_id"] == user_id

    pool = SharedConnectionPool(dsn, pool_size=2)
    try:
        fs = SQLContextFS(pool=pool)
        evidence = MemoryEvidenceService(fs)
        nodes = []
        for kind, owner_id in (("user", user_id), ("agent", agent_id)):
            node = ContextNode(
                uri=f"ctx://{account_id}/{kind}s/{owner_id}/memories/events/release",
                context_type="MEMORY",
                category="events",
                level=4,
                owner_space=f"{kind}:{owner_id}",
                abstract="Project A release",
                overview="The project release schedule",
                content="Project A releases on Friday.",
                metadata={"_expected_version": 0, "provenance_ids": [f"{run_id}:message:1"]},
            )
            fs.write_node(node, ctx)
            nodes.append(node)

        original = evidence.capture([nodes[0].uri], ctx, owner_space=ctx.user_space_name())
        agent_snapshot = evidence.capture([nodes[1].uri], ctx, owner_space=ctx.agent_space_name())
        assert original.nodes[0].version == 1
        assert agent_snapshot.nodes[0].version == 1
        assert evidence.check(original, ctx).is_current
        assert evidence.check(agent_snapshot, ctx).is_current

        updated = fs.read_node(nodes[0].uri, ctx)
        updated.content = "Project A releases on Saturday."
        updated.metadata["_expected_version"] = original.nodes[0].version
        fs.write_node(updated, ctx)
        content_check = evidence.check(original, ctx)
        assert content_check.changed_uris == (nodes[0].uri,)
        content_snapshot = evidence.capture([nodes[0].uri], ctx, owner_space=ctx.user_space_name())
        assert content_snapshot.nodes[0].version == 2

        updated = fs.read_node(nodes[0].uri, ctx)
        updated.metadata["provenance_ids"].append(f"{run_id}:message:2")
        updated.metadata["_expected_version"] = content_snapshot.nodes[0].version
        fs.write_node(updated, ctx)
        provenance_check = evidence.check(content_snapshot, ctx)
        assert provenance_check.changed_uris == (nodes[0].uri,)
        latest = evidence.capture([nodes[0].uri], ctx, owner_space=ctx.user_space_name())
        assert latest.nodes[0].version == 3
        assert latest.to_dict()["nodes"][0]["node"]["content"] == updated.content
        assert original.to_dict()["nodes"][0]["node"]["content"] == "Project A releases on Friday."
        assert evidence.check(agent_snapshot, ctx).is_current
    finally:
        pool.close_all()

    reopened_pool = SharedConnectionPool(dsn, pool_size=2)
    try:
        reopened_fs = SQLContextFS(pool=reopened_pool)
        reopened_service = MemoryEvidenceService(reopened_fs)
        reread = reopened_service.capture([nodes[0].uri], ctx, owner_space=ctx.user_space_name())
        assert reread == latest
        assert reopened_service.check(original, ctx).changed_uris == (nodes[0].uri,)
        block = ContextReader(reopened_fs, storage_backend="sql").read(nodes[0].uri, ctx)
        assert block.content_excerpt == "Project A releases on Saturday."
        report = {
            "account_id": account_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "original": original.to_dict(),
            "content_change": asdict(content_check),
            "provenance_change": asdict(provenance_check),
            "latest": reread.to_dict(),
            "agent": agent_snapshot.to_dict(),
            "persistent_reread": "passed",
            "read_memory_content": block.content_excerpt,
        }
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "evidence-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Evidence verification: {run_root / 'evidence-report.json'}")
    finally:
        reopened_pool.close_all()
