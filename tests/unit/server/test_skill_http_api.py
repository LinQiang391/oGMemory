from __future__ import annotations

import base64
import hashlib

from core.models import ContextNode, RequestContext, Role
from providers.unified_config import OgMemConfig
from server import app as app_module
from server.auth import ResolvedIdentity
from skills.artifact import internal_skill_uri, is_governed_skill


class _Auth:
    def role_control_active(self):
        return True

    def resolve_identity(self, headers):
        normalized = {str(key).lower(): value for key, value in headers.items()}
        if normalized.get("x-api-key") != "root-key":
            raise AssertionError("missing test API key")
        return ResolvedIdentity(role=Role.ROOT, account_id="acct-1", user_id="root")


class _FS:
    def list_children(self, uri, ctx):
        assert uri == "ctx://acct-1/agents/bot/skills"
        return ["ctx://acct-1/agents/bot/skills/generated/review-code"]

    def read_node(self, uri, ctx):
        return ContextNode(
            uri="ctx://acct-1/agents/bot/skills/generated/review-code",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:bot",
            abstract="Review code before release.",
            overview="Agent skill: review-code",
            content="---\nname: review-code\ndescription: Review code.\n---\n# Review code\n",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": 0.95},
        )


class _Service:
    _cfg = OgMemConfig(storage_backend="sql", sql_connection_string="postgresql://unused")

    def __init__(self):
        self.fs = _FS()
        self.wake_count = 0
        self.last_skill_source = None

    def get_auth_service(self):
        return _Auth()

    @property
    def default_agent_id(self):
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self):
        return self._cfg.skill_search_scan_limit

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id="http",
            trace_id="http",
        )

    def _get_context_fs(self):
        return self.fs

    def list_published_skills(self, ctx, *, limit=50, cursor=None):
        node = self.fs.read_node(internal_skill_uri(ctx.account_id, ctx.agent_id, "review-code"), ctx)
        rows = []
        if is_governed_skill(node):
            rows.append({"name": "review-code", "description": node.abstract, "updated_at": ""})
        return {"skills": rows[:limit], "truncated": False}

    def read_published_skill(self, ctx, name, source=None):
        self.last_skill_source = source
        if source == "imported":
            data = b"print('ok')\n"
            return ContextNode(
                uri="ctx://acct-1/agents/bot/skills/imported/managed-skill",
                context_type="SKILL",
                category="skill",
                level=0,
                owner_space="agent:bot",
                abstract="Managed skill.",
                overview="Agent skill: managed-skill",
                content="# Managed\n",
                metadata={
                    "ogmemory_managed": True,
                    "publication_status": "published",
                    "package_files": [{
                        "path": "scripts/run.py",
                        "data": base64.b64encode(data).decode("ascii"),
                        "encoding": "base64",
                        "media_type": "text/x-python",
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }],
                },
            )
        node = self.fs.read_node(internal_skill_uri(ctx.account_id, ctx.agent_id, name), ctx)
        if not is_governed_skill(node):
            raise FileNotFoundError(name)
        return node

    def list_skill_candidates(self, ctx, state=None):
        return [
            {
                "capability_key": "review-code",
                "state": state or "accumulating",
                "evidence_count": 4,
                "distinct_session_count": 2,
                "summary": "Review changes.",
                "reason": "Awaiting one more session.",
            }
        ]

    def add_session_trace(self, params):
        return {"ok": True, "event_index": 9}

    def wake_skill_sedimentation(self):
        self.wake_count += 1

    def validate_managed_skill(self, data):
        return {"valid": True, "name": "managed-skill"}

    def add_managed_skill(self, ctx, data):
        return {"name": "managed-skill", "action": "create"}

    def read_managed_skill_file(self, ctx, name, path):
        return {
            "name": "run.py",
            "path": path,
            "uri": f"ogmemory://skills/imported/{name}/{path}",
            "source": "imported",
            "content": "print('ok')\n",
            "encoding": "utf-8",
        }

    def delete_published_skill(self, ctx, name, source):
        self.last_skill_source = source
        if source not in {"generated", "imported"}:
            raise ValueError("source is required and must be generated or imported")
        return {
            "name": name,
            "uri": f"ogmemory://skills/{source}/{name}",
            "source": source,
            "action": "delete",
        }

def test_skill_http_list_read_and_candidates(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    client = app_module.app.test_client()
    headers = {"X-Agent-ID": "bot", "X-API-Key": "root-key"}

    listing = client.get("/api/v1/skills", headers=headers)
    detail = client.get("/api/v1/skills/review-code", headers=headers)
    candidates = client.get(
        "/api/v1/skill-candidates?state=accumulating", headers=headers
    )

    assert listing.status_code == 200
    assert listing.get_json()["skills"][0]["name"] == "review-code"
    assert detail.status_code == 200
    assert detail.get_json()["content"].startswith("---\nname: review-code")
    assert candidates.status_code == 200
    assert candidates.get_json()["candidates"][0]["evidence_count"] == 4
    assert "session_id" not in candidates.get_data(as_text=True)


def test_skill_http_forwards_source_selector(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().get(
        "/api/v1/skills/review-code?source=generated",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["source"] == "generated"
    assert service.last_skill_source == "generated"


def test_skill_http_rejects_invalid_name_before_storage(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    response = app_module.app.test_client().get(
        "/api/v1/skills/INVALID_name",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
    )
    assert response.status_code == 400


def test_skill_http_add_and_validate_routes(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    client = app_module.app.test_client()
    headers = {"X-Agent-ID": "bot", "X-API-Key": "root-key"}
    data = {"name": "managed-skill", "description": "Managed", "content": "# Managed"}

    created = client.post("/api/v1/skills", headers=headers, json={"data": data})
    validated = client.post("/api/v1/skills/validate", headers=headers, json={"data": data})
    assert created.status_code == 201
    assert created.get_json()["action"] == "create"
    assert validated.get_json()["valid"] is True


def test_skill_http_returns_and_reads_managed_package_files(monkeypatch):
    monkeypatch.setattr(app_module, "_service", _Service())
    client = app_module.app.test_client()
    headers = {"X-Agent-ID": "bot", "X-API-Key": "root-key"}

    detail = client.get(
        "/api/v1/skills/managed-skill?source=imported", headers=headers
    )
    file_response = client.get(
        "/api/v1/skills/managed-skill/scripts/run.py", headers=headers
    )

    assert detail.status_code == 200
    assert detail.get_json()["files"][1]["path"] == "scripts/run.py"
    assert file_response.status_code == 200
    assert file_response.get_json()["content"] == "print('ok')\n"


def test_skill_http_delete_requires_and_forwards_source(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()
    headers = {"X-Agent-ID": "bot", "X-API-Key": "root-key"}

    missing_source = client.delete("/api/v1/skills/review-code", headers=headers)
    deleted = client.delete(
        "/api/v1/skills/review-code?source=generated", headers=headers
    )

    assert missing_source.status_code == 400
    assert deleted.status_code == 200
    assert deleted.get_json() == {
        "name": "review-code",
        "uri": "ogmemory://skills/generated/review-code",
        "source": "generated",
        "action": "delete",
    }
    assert service.last_skill_source == "generated"


def test_skill_http_duplicate_import_returns_conflict(monkeypatch):
    service = _Service()

    def conflict(ctx, data):
        raise FileExistsError("skill already exists: managed-skill")

    service.add_managed_skill = conflict
    monkeypatch.setattr(app_module, "_service", service)
    response = app_module.app.test_client().post(
        "/api/v1/skills",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        json={"data": {"name": "managed-skill"}},
    )

    assert response.status_code == 409


def test_session_end_trace_events_wake_worker_after_persistence(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)

    for index, event in enumerate(
        (
            {"hook_type": "SessionEnd"},
            {"event_type": "session_end"},
        )
    ):
        response = app_module.app.test_client().post(
            "/api/v1/trace/events",
            headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
            json={
                "request_id": f"evt-session-end-{index}",
                "session_id": "session-1",
                "agent_id": "bot",
                "event": event,
            },
        )

        assert response.status_code == 200

    assert service.wake_count == 2
