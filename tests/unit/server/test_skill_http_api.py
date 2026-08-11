from __future__ import annotations

import base64
import hashlib
import io
import zipfile

from werkzeug.test import EnvironBuilder

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

    def validate_managed_skill(self, data, **options):
        self.last_validation = (data, options)
        return {
            "valid": True,
            "strict": options.get("strict", False),
            "name": "managed-skill",
            "errors": [],
            "warnings": [],
        }

    def add_managed_skill(self, ctx, data):
        self.last_skill_data = data
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


def test_skill_http_validation_forwards_format_options(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)
    response = app_module.app.test_client().post(
        "/api/v1/skills/validate",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        json={
            "data": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            "strict": True,
            "skill_dir_name": "demo-dir",
            "source_path": "demo-dir/SKILL.md",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["strict"] is True
    assert service.last_validation[1] == {
        "strict": True,
        "skill_dir_name": "demo-dir",
        "source_path": "demo-dir/SKILL.md",
    }


def test_skill_http_add_accepts_zip_upload(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            "managed-skill/SKILL.md",
            "---\nname: managed-skill\ndescription: Managed.\n---\n\n# Managed\n",
        )
    payload = archive.getvalue()

    response = app_module.app.test_client().post(
        "/api/v1/skills",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        data={"file": (io.BytesIO(payload), "managed-skill.zip")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert service.last_skill_data == payload


def test_skill_http_rejects_zip_over_20_mb_with_json_error(monkeypatch):
    from skills import management

    monkeypatch.setattr(app_module, "_service", _Service())
    monkeypatch.setattr(management, "MAX_SKILL_ARCHIVE_BYTES", 10)
    response = app_module.app.test_client().post(
        "/api/v1/skills",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        data=b"x" * 11,
        content_type="application/zip",
    )

    assert response.status_code == 413
    assert response.get_json() == {"error": "skill archive exceeds 10 bytes"}


def test_skill_http_rejects_multipart_over_upload_limit(monkeypatch):
    monkeypatch.setattr(app_module, "_MAX_SKILL_UPLOAD_BODY_BYTES", 10)

    response = app_module.app.test_client().post(
        "/api/v1/skills",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        data=b"x" * 11,
        content_type="multipart/form-data; boundary=test",
    )

    assert response.status_code == 413
    assert response.is_json


def test_skill_http_accepts_near_limit_json_package(monkeypatch):
    service = _Service()
    monkeypatch.setattr(app_module, "_service", service)
    file_size = 5 * 1024 * 1024 - 1024
    encoded = base64.b64encode(b"x" * file_size).decode("ascii")
    files = [{"path": "SKILL.md", "content": "skill"}]
    files.extend(
        {"path": f"assets/data-{index}.bin", "content_base64": encoded}
        for index in range(4)
    )

    response = app_module.app.test_client().post(
        "/api/v1/skills/validate",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        json={"data": {"files": files}},
    )

    assert response.status_code == 200
    assert service.last_validation[0]["files"] == files


def test_skill_http_bounds_chunked_zip_without_content_length(monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 10)
    monkeypatch.setattr(app_module, "_service", _Service())
    builder = EnvironBuilder(
        method="POST",
        path="/api/v1/skills",
        headers={"X-Agent-ID": "bot", "X-API-Key": "root-key"},
        input_stream=io.BytesIO(b"x" * 11),
        content_type="application/zip",
    )
    environ = builder.get_environ()
    environ.pop("CONTENT_LENGTH", None)
    environ["wsgi.input_terminated"] = True

    response = app_module.app.test_client().open(environ)

    assert response.status_code == 413
    assert response.is_json


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
