from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from cli.commands.skill import _SkillClient, run, sync_skills
from cli.main import build_parser


class _Client:
    def __init__(self, skills: dict[str, str]):
        self.skills = skills

    def list_skills(self, *, query=None, limit=100, cursor=None):
        return {"skills": [{"name": name} for name in self.skills]}

    def read_skill(self, name, source=None):
        return {"name": name, "content": self.skills[name]}


def _args(action: str, **overrides):
    values = {
        "skill_action": action,
        "base_url": None,
        "api_key": None,
        "account_id": None,
        "user_id": None,
        "agent_id": None,
        "config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _RunClient:
    def __init__(self):
        self.calls = []

    def list_skills(self, *, query=None, limit=50, cursor=None):
        self.calls.append(("list", query, limit, cursor))
        return {"skills": [{"name": "review-code"}]}

    def read_skill(self, name, source=None):
        self.calls.append(("show", name, source))
        return {"name": name, "content": "# Review code"}

    def list_candidates(self, *, state=None):
        self.calls.append(("candidates", state))
        return {"candidates": [{"state": state}]}


def test_skill_parser_supports_all_actions():
    parser = build_parser()
    assert parser.parse_args(["skill", "list"]).skill_action == "list"
    assert parser.parse_args(["skill", "show", "demo"]).skill_action == "show"
    assert parser.parse_args(
        ["skill", "show", "Package_One", "--source", "imported"]
    ).source == "imported"
    assert parser.parse_args(["skill", "candidates"]).skill_action == "candidates"
    assert parser.parse_args(["skill", "sync", "--target-dir", "x"]).skill_action == "sync"


def test_run_list_prints_server_payload(monkeypatch, capsys):
    client = _RunClient()
    monkeypatch.setattr(_SkillClient, "from_args", lambda args: client)

    assert run(_args("list", query="review", limit=7)) == 0

    assert json.loads(capsys.readouterr().out)["skills"][0]["name"] == "review-code"
    assert client.calls == [("list", "review", 7, None)]


def test_run_show_supports_json_and_raw_output(monkeypatch, capsys):
    client = _RunClient()
    monkeypatch.setattr(_SkillClient, "from_args", lambda args: client)

    assert run(_args("show", name="review-code", source="generated", raw=False)) == 0
    assert json.loads(capsys.readouterr().out)["content"] == "# Review code"
    assert run(_args("show", name="review-code", source="generated", raw=True)) == 0
    assert capsys.readouterr().out == "# Review code\n"
    assert client.calls == [
        ("show", "review-code", "generated"),
        ("show", "review-code", "generated"),
    ]


def test_skill_client_reads_managed_mixed_case_name(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"name": "Package_One", "source": "imported"}

    def fake_get(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("cli.commands.skill.httpx.get", fake_get)

    payload = _SkillClient("https://memory.example", {}).read_skill(
        "Package_One", source="imported"
    )

    assert payload["source"] == "imported"
    assert request["url"].endswith("/api/v1/skills/Package_One")
    assert request["params"] == {"source": "imported"}


def test_run_candidates_forwards_state(monkeypatch, capsys):
    client = _RunClient()
    monkeypatch.setattr(_SkillClient, "from_args", lambda args: client)

    assert run(_args("candidates", state="held")) == 0

    assert json.loads(capsys.readouterr().out) == {"candidates": [{"state": "held"}]}
    assert client.calls == [("candidates", "held")]


def test_skill_client_sends_authenticated_http_request(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            request["checked"] = True

        def json(self):
            return {"skills": []}

    def fake_get(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("cli.commands.skill.httpx.get", fake_get)
    client = _SkillClient("https://memory.example/", {"X-API-Key": "secret"})

    assert client.list_skills(query="review", limit=3, cursor="next") == {"skills": []}
    assert request == {
        "url": "https://memory.example/api/v1/skills",
        "headers": {"X-API-Key": "secret"},
        "params": {"limit": 3, "q": "review", "cursor": "next"},
        "timeout": 30.0,
        "checked": True,
    }


def test_skill_client_rejects_malformed_http_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    monkeypatch.setattr("cli.commands.skill.httpx.get", lambda *args, **kwargs: Response())

    try:
        _SkillClient("http://localhost", {}).list_candidates()
    except ValueError as exc:
        assert "non-object" in str(exc)
    else:
        raise AssertionError("expected malformed response to fail")


def test_run_reports_http_error(monkeypatch, capsys):
    class FailingClient:
        def list_skills(self, **kwargs):
            raise httpx.ConnectError("server unavailable")

    monkeypatch.setattr(_SkillClient, "from_args", lambda args: FailingClient())

    assert run(_args("list", query=None, limit=50)) == 1
    assert "server unavailable" in capsys.readouterr().err


def test_sync_requires_a_target_before_contacting_server(monkeypatch, capsys):
    monkeypatch.delenv("OGMEM_SKILL_SYNC_TARGET", raising=False)
    args = SimpleNamespace(
        skill_action="sync", target_dir=None, dry_run=False, prune=False,
        base_url=None, api_key=None, account_id=None, user_id=None, agent_id=None, config=None,
    )
    assert run(args) == 1
    assert "no files were written" in capsys.readouterr().err


def test_sync_writes_manifest_and_skill_atomically(tmp_path: Path):
    content = "---\nname: demo\ndescription: Demo\n---\n# Demo\n"
    result = sync_skills(_Client({"demo": content}), tmp_path, dry_run=False, prune=False)
    assert result["written"] == ["demo"]
    assert (tmp_path / "demo" / "SKILL.md").read_text(encoding="utf-8") == content
    assert json.loads((tmp_path / ".ogmemory-managed-skills.json").read_text())["skills"] == ["demo"]


def test_sync_dry_run_performs_zero_writes(tmp_path: Path):
    result = sync_skills(_Client({"demo": "content"}), tmp_path, dry_run=True, prune=False)
    assert result["written"] == ["demo"]
    assert list(tmp_path.iterdir()) == []


def test_sync_ignores_managed_package_catalog_entries(tmp_path: Path):
    class MixedClient:
        def list_skills(self, *, query=None, limit=100, cursor=None):
            return {
                "skills": [
                    {"name": "generated-skill", "source": "generated"},
                    {"name": "Package_One", "source": "imported"},
                ],
                "truncated": False,
            }

        def read_skill(self, name, source=None):
            assert (name, source) == ("generated-skill", "generated")
            return {"content": "# Generated\n", "truncated": False}

    result = sync_skills(
        MixedClient(), tmp_path, dry_run=True, prune=False
    )

    assert result["written"] == ["generated-skill"]


def test_sync_refuses_unmanaged_collision(tmp_path: Path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("human", encoding="utf-8")
    try:
        sync_skills(_Client({"demo": "generated"}), tmp_path, dry_run=False, prune=False)
    except ValueError as exc:
        assert "unmanaged" in str(exc)
    else:
        raise AssertionError("expected unmanaged collision")


def test_sync_without_prune_preserves_absent_skill_ownership(tmp_path: Path):
    old = tmp_path / "old"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("old", encoding="utf-8")
    manifest = tmp_path / ".ogmemory-managed-skills.json"
    manifest.write_text(
        json.dumps({"version": 1, "skills": ["old"]}), encoding="utf-8",
    )

    sync_skills(_Client({}), tmp_path, dry_run=False, prune=False)

    assert (old / "SKILL.md").read_text(encoding="utf-8") == "old"
    assert json.loads(manifest.read_text(encoding="utf-8"))["skills"] == ["old"]

    sync_skills(_Client({"old": "updated"}), tmp_path, dry_run=False, prune=False)
    assert (old / "SKILL.md").read_text(encoding="utf-8") == "updated"


def test_prune_only_removes_manifest_managed_skills(tmp_path: Path):
    old = tmp_path / "old"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("old", encoding="utf-8")
    human = tmp_path / "human"
    human.mkdir()
    (human / "SKILL.md").write_text("human", encoding="utf-8")
    (tmp_path / ".ogmemory-managed-skills.json").write_text(
        json.dumps({"version": 1, "skills": ["old"]}), encoding="utf-8"
    )
    sync_skills(_Client({}), tmp_path, dry_run=False, prune=True)
    assert not old.exists()
    assert (human / "SKILL.md").read_text(encoding="utf-8") == "human"


def test_sync_fetches_every_page_before_pruning(tmp_path: Path):
    skills = {f"skill-{index}": f"content {index}" for index in range(125)}

    class PaginatedClient(_Client):
        def list_skills(self, *, query=None, limit=100, cursor=None):
            start = int(cursor or 0)
            names = list(self.skills)[start : start + limit]
            end = start + len(names)
            truncated = end < len(self.skills)
            return {
                "skills": [{"name": name} for name in names],
                "truncated": truncated,
                "next_cursor": str(end) if truncated else None,
            }

    sync_skills(PaginatedClient(skills), tmp_path, dry_run=False, prune=True)
    manifest = json.loads((tmp_path / ".ogmemory-managed-skills.json").read_text())
    assert len(manifest["skills"]) == 125
    assert (tmp_path / "skill-124" / "SKILL.md").read_text() == "content 124"


def test_sync_rejects_truncated_page_without_cursor(tmp_path: Path):
    class BrokenClient(_Client):
        def list_skills(self, *, query=None, limit=100, cursor=None):
            return {"skills": [], "truncated": True}

    try:
        sync_skills(BrokenClient({}), tmp_path, dry_run=False, prune=True)
    except ValueError as exc:
        assert "no next cursor" in str(exc)
    else:
        raise AssertionError("expected inconsistent pagination to fail")
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_sync_rejects_traversal_and_absolute_manifest_names(tmp_path: Path):
    manifest = tmp_path / ".ogmemory-managed-skills.json"
    for invalid in ("../victim", str(tmp_path / "victim")):
        manifest.write_text(
            json.dumps({"version": 1, "skills": [invalid]}), encoding="utf-8"
        )
        try:
            sync_skills(_Client({}), tmp_path, dry_run=False, prune=True)
        except ValueError as exc:
            assert "invalid skill name" in str(exc)
        else:
            raise AssertionError("expected unsafe manifest name to fail")


def test_skill_client_does_not_use_llm_base_url_environment(monkeypatch):
    monkeypatch.delenv("OGMEM_HTTP_BASE_URL", raising=False)
    monkeypatch.delenv("OGMEM_SERVER_URL", raising=False)
    monkeypatch.setenv("OGMEM_BASE_URL", "https://llm.example.invalid")
    args = SimpleNamespace(
        base_url=None, api_key=None, account_id=None, user_id=None, agent_id=None
    )
    assert _SkillClient.from_args(args).base_url == "http://127.0.0.1:8090"
    monkeypatch.setenv("OGMEM_SERVER_URL", "https://server.example.invalid")
    assert _SkillClient.from_args(args).base_url == "https://server.example.invalid"
    monkeypatch.setenv("OGMEM_HTTP_BASE_URL", "https://http.example.invalid")
    assert _SkillClient.from_args(args).base_url == "https://http.example.invalid"


def test_sync_rolls_back_all_files_if_manifest_commit_fails(tmp_path: Path, monkeypatch):
    from cli.commands import skill as command

    old_skill = tmp_path / "old" / "SKILL.md"
    old_skill.parent.mkdir(parents=True)
    old_skill.write_text("old before", encoding="utf-8")
    manifest = tmp_path / ".ogmemory-managed-skills.json"
    old_manifest = json.dumps({"version": 1, "skills": ["old"]})
    manifest.write_text(old_manifest, encoding="utf-8")
    original = command._atomic_write_text

    def fail_manifest(path, content):
        if path.name == ".ogmemory-managed-skills.json":
            raise OSError("simulated manifest failure")
        original(path, content)

    monkeypatch.setattr(command, "_atomic_write_text", fail_manifest)
    try:
        sync_skills(
            _Client({"old": "old changed", "new": "new content"}),
            tmp_path,
            dry_run=False,
            prune=False,
        )
    except OSError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("expected simulated write failure")
    assert old_skill.read_text(encoding="utf-8") == "old before"
    assert not (tmp_path / "new").exists()
    assert manifest.read_text(encoding="utf-8") == old_manifest


def test_sync_rolls_back_if_a_skill_write_fails_midway(tmp_path: Path, monkeypatch):
    from cli.commands import skill as command

    original = command._atomic_write_text
    calls = 0

    def fail_second_skill(path, content):
        nonlocal calls
        if path.name == "SKILL.md":
            calls += 1
            if calls == 2:
                raise OSError("simulated skill failure")
        original(path, content)

    monkeypatch.setattr(command, "_atomic_write_text", fail_second_skill)
    try:
        sync_skills(
            _Client({"first": "one", "second": "two"}),
            tmp_path,
            dry_run=False,
            prune=False,
        )
    except OSError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("expected simulated write failure")
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
