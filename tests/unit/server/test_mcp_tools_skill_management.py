from __future__ import annotations

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.mcp_tools import delete_skill, install_skill, update_skill


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeService:
    def __init__(self):
        self._cfg = OgMemConfig()
        self.calls = []

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="root",
            agent_id=params.get("agentId") or "bot",
            session_id="skill-management",
            trace_id="skill-management",
            role=Role.ROOT,
        )

    def add_managed_skill(self, ctx, data):
        self.calls.append(("install", ctx.agent_id, data))
        return {"status": "success", "action": "create"}

    def update_managed_skill(self, ctx, name, data):
        self.calls.append(("update", ctx.agent_id, name, data))
        return {"status": "success", "action": "update"}

    def delete_published_skill(self, ctx, name, source):
        self.calls.append(("delete", ctx.agent_id, name, source))
        return {"action": "delete", "name": name, "source": source}


SKILL_MD = """---
name: review-code
description: Review code changes.
---

# Review code
"""


def test_install_update_and_delete_skill_use_selected_agent_scope():
    service = _FakeService()

    created = install_skill(service, data=SKILL_MD, agent_id="bot")
    updated = update_skill(
        service,
        name="review-code",
        data=SKILL_MD,
        agent_id="bot",
    )
    deleted = delete_skill(
        service,
        name="review-code",
        source="imported",
        agent_id="bot",
    )

    assert created["action"] == "create"
    assert updated["action"] == "update"
    assert deleted == {
        "action": "delete",
        "name": "review-code",
        "source": "imported",
    }
    assert service.calls == [
        ("install", "bot", SKILL_MD),
        ("update", "bot", "review-code", SKILL_MD),
        ("delete", "bot", "review-code", "imported"),
    ]
