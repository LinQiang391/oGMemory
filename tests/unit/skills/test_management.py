from __future__ import annotations

import base64
import codecs
from types import MethodType, SimpleNamespace
from urllib.parse import urlparse

import pytest

from core.errors import ConcurrentModificationError
from core.models import ContextNode, RequestContext, Role
from server.auth import AuthorizationError
from server.mcp_tools import search_skills
from server.memory_service import MemoryService
from skills import management as managed_skill
from skills.management import (
    MAX_MEDIA_TYPE_LENGTH,
    MAX_PACKAGE_PATH_BYTES,
    MAX_SKILL_BYTES,
    package_file_manifest,
    parse_managed_skill,
)

CONTENT = """---
name: review-code
description: Review code safely.
---

# Review code

Inspect the diff and run focused tests.
"""


class _FS:
    def __init__(self):
        self.nodes: dict[str, ContextNode] = {}
        self.read_calls = 0

    def exists(self, uri, ctx):
        return uri in self.nodes

    def write_node(self, node, ctx):
        if node.metadata.get("_expected_version") == 0 and node.uri in self.nodes:
            raise ConcurrentModificationError(node.uri, 0, 1)
        node.metadata["version"] = 1
        self.nodes[node.uri] = node

    def read_node(self, uri, ctx):
        self.read_calls += 1
        if uri not in self.nodes:
            raise FileNotFoundError(uri)
        return self.nodes[uri]

    def list_children(self, uri, ctx):
        prefix = f"{uri}/"
        return [
            key
            for key in self.nodes
            if key.startswith(prefix) and "/" not in key[len(prefix):]
        ]

    def delete_node(self, uri, ctx):
        if uri not in self.nodes:
            raise FileNotFoundError(uri)
        del self.nodes[uri]


class _Auth:
    def role_control_active(self):
        return False

    def require_role(self, ctx, *roles):
        if ctx.role not in roles:
            raise AuthorizationError("forbidden")


def _service(fs: _FS) -> MemoryService:
    service = MemoryService.__new__(MemoryService)
    service._get_context_fs = MethodType(lambda self: fs, service)
    service.get_auth_service = MethodType(lambda self: _Auth(), service)
    service._cfg = SimpleNamespace(skill_search_scan_limit=2500, agent_id="agent")
    service.build_context = MethodType(lambda self, params, identity=None: _ctx(), service)
    return service


def _ctx(role=Role.ROOT):
    return RequestContext("acct", "user", "agent", "session", "trace", role=role)


def test_parse_structured_and_mcp_skill_inputs():
    structured = parse_managed_skill({
        "name": "review-code",
        "description": "Review code safely.",
        "content": "# Review code\n\nInspect the diff.",
    })
    mcp = parse_managed_skill({
        "name": "search_web",
        "description": "Search the web.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    })

    assert structured.name == "review-code"
    assert structured.content.startswith("---\nname: review-code")
    assert mcp.name == "search-web"
    assert "**query** (string, required)" in mcp.content


def test_parse_accepts_frontmatter_delimiters_with_trailing_spaces():
    parsed = parse_managed_skill(
        "---   \nname: spaced-delimiter\ndescription: Demo.\n---   \n\n# Demo\n"
    )

    assert parsed.name == "spaced-delimiter"


def test_parse_accepts_utf8_bom_package():
    parsed = parse_managed_skill(
        {
            "files": [
                {
                    "path": "pkg/SKILL.md",
                    "content": "\ufeff---\nname: bom-demo\ndescription: Demo.\n---\n\n# Demo\n",
                }
            ]
        }
    )

    assert parsed.name == "bom-demo"


def test_parse_redacts_frontmatter_secret_without_changing_yaml_type():
    parsed = parse_managed_skill(
        "---\nname: secret-description\n"
        "description: token=abc123456789\n---\n\n# Demo\n"
    )

    assert parsed.description == "[REDACTED_SECRET]"
    assert "abc123456789" not in parsed.content


def test_structured_frontmatter_compares_sanitized_description():
    parsed = parse_managed_skill(
        {
            "name": "secret-description",
            "description": "token=abc123456789",
            "content": (
                "---\nname: secret-description\n"
                "description: token=abc123456789\n---\n\n# Demo\n"
            ),
        }
    )

    assert parsed.description == "[REDACTED_SECRET]"


def test_parse_rejects_oversized_skill_before_redaction():
    content = (
        "---\nname: huge-secret\ndescription: Demo.\n---\n\n# Demo\npassword="
        + "a" * MAX_SKILL_BYTES
    )

    with pytest.raises(ValueError, match="SKILL.md exceeds"):
        parse_managed_skill(content)


@pytest.mark.parametrize(
    "data",
    [
        " " * (MAX_SKILL_BYTES + 1)
        + "---\nname: huge-space\ndescription: Demo.\n---\n\n# Demo\n",
        {
            "name": "huge-space",
            "description": "Demo.",
            "content": " " * (MAX_SKILL_BYTES + 1) + "# Demo",
        },
    ],
    ids=["markdown", "structured"],
)
def test_parse_rejects_oversized_raw_whitespace(data):
    with pytest.raises(ValueError, match="SKILL.md exceeds"):
        parse_managed_skill(data)


@pytest.mark.parametrize(
    "data",
    [
        {"name": 123, "description": "Demo.", "content": "# Demo"},
        "---\nname: true\ndescription: Demo.\n---\n\n# Demo\n",
        {"name": 123, "description": "Demo.", "inputSchema": {}},
    ],
)
def test_parse_rejects_non_string_names(data):
    with pytest.raises(ValueError, match="invalid skill name|MCP tool name"):
        parse_managed_skill(data)


@pytest.mark.parametrize(
    "path",
    ["/", ".", "./file.txt", "/absolute.txt", "C:\\secret.txt", "dir//file.txt"],
)
def test_package_rejects_noncanonical_or_absolute_file_paths(path):
    package = {
        "files": [
            {
                "path": "SKILL.md",
                "content": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            {"path": path, "content": "data"},
        ]
    }

    with pytest.raises(ValueError, match="unsafe package path"):
        parse_managed_skill(package)


def test_package_rejects_case_insensitive_duplicate_paths():
    package = {
        "files": [
            {
                "path": "SKILL.md",
                "content": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            {"path": "skill.md", "content": "# Conflicting entry\n"},
        ]
    }

    with pytest.raises(ValueError, match="duplicate package path"):
        parse_managed_skill(package)


def test_parse_rejects_recursive_yaml_alias():
    content = (
        "---\nname: cyclic\ndescription: Demo.\n"
        "tags: &tags [*tags]\n---\n\n# Demo\n"
    )

    with pytest.raises(ValueError, match="recursive YAML alias"):
        parse_managed_skill(content)


@pytest.mark.parametrize(
    "file_record, message",
    [
        ({"path": "a" * (MAX_PACKAGE_PATH_BYTES + 1), "content": "x"}, "path exceeds"),
        (
            {
                "path": "file.txt",
                "content": "x",
                "media_type": "x" * (MAX_MEDIA_TYPE_LENGTH + 1),
            },
            "media type exceeds",
        ),
        ({"path": "file.txt", "content": "x", "media_type": {"type": "text"}}, "media type"),
    ],
)
def test_package_bounds_file_metadata(file_record, message):
    package = {
        "files": [
            {
                "path": "SKILL.md",
                "content": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            file_record,
        ]
    }

    with pytest.raises(ValueError, match=message):
        parse_managed_skill(package)


def test_package_size_limit_uses_raw_bytes_before_redaction(monkeypatch):
    monkeypatch.setattr(managed_skill, "MAX_PACKAGE_BYTES", 100)
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        + "A" * 200
        + "\n-----END PRIVATE KEY-----"
    )
    package = {
        "files": [
            {
                "path": "SKILL.md",
                "content": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            {"path": "secret.txt", "content": private_key},
        ]
    }

    with pytest.raises(ValueError, match="skill package exceeds 100 bytes"):
        parse_managed_skill(package)


@pytest.mark.parametrize(
    "data",
    [
        "# Missing frontmatter",
        "---\nname: INVALID NAME\ndescription: Bad\n---\n\n# Bad\n",
        {"name": "demo", "description": "", "content": "# Demo"},
    ],
)
def test_parse_rejects_invalid_skill_inputs(data):
    with pytest.raises(ValueError):
        parse_managed_skill(data)


def test_add_complete_package_is_sanitized_and_isolated():
    fs = _FS()
    service = _service(fs)
    package = {
        "files": [
            {"path": "demo/SKILL.md", "content": "---\nname: Package_One\ndescription: Package.\n---\n\n# Package\nsecret_token=abc123456789\n"},
            {"path": "demo/scripts/run.py", "content": "print('ok')\n"},
            {"path": "demo/assets/logo.bin", "content_base64": base64.b64encode(b"\x00\xff").decode("ascii")},
        ]
    }

    created = service.add_managed_skill(_ctx(), package)
    node = next(iter(fs.nodes.values()))

    assert created["source"] == "imported"
    assert created["uri"] == "ogmemory://skills/imported/Package_One"
    assert created["auxiliary_files"] == 2
    assert node.uri == "ctx://acct/agents/agent/skills/imported/Package_One"
    assert "abc123456789" not in node.content
    assert len(node.metadata["package_files"]) == 2
    assert all(
        item["uri"].startswith("ogmemory://skills/imported/Package_One/")
        and item["source"] == "imported"
        for item in service.validate_managed_skill(package)["files"]
    )
    manifest = service.read_published_skill(_ctx(), "Package_One", "imported")
    script = service.read_managed_skill_file(
        _ctx(), "Package_One", "scripts/run.py"
    )
    binary = service.read_managed_skill_file(
        _ctx(), "Package_One", "assets/logo.bin"
    )
    assert manifest.metadata["package_files"]
    assert script["content"] == "print('ok')\n"
    assert script["encoding"] == "utf-8"
    assert base64.b64decode(binary["content_base64"]) == b"\x00\xff"
    assert binary["encoding"] == "base64"


def test_package_manifest_percent_encodes_special_path_characters():
    skill = parse_managed_skill({
        "files": [
            {
                "path": "pkg/SKILL.md",
                "content": "---\nname: demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            {"path": "pkg/references/guide #1?.md", "content": "# Guide\n"},
        ]
    })

    uri = package_file_manifest(skill, source="imported")[1]["uri"]

    assert uri.endswith("/references/guide%20%231%3F.md")
    assert urlparse(uri).fragment == ""
    assert urlparse(uri).query == ""


def test_utf16_package_text_is_redacted_without_changing_byte_order():
    fs = _FS()
    service = _service(fs)
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz"
    raw = codecs.BOM_UTF16_BE + f"API_KEY={secret}\n".encode("utf-16-be")
    package = {
        "files": [
            {
                "path": "pkg/SKILL.md",
                "content": "---\nname: utf16-demo\ndescription: Demo.\n---\n\n# Demo\n",
            },
            {
                "path": "pkg/scripts/config.ps1",
                "content_base64": base64.b64encode(raw).decode("ascii"),
            },
        ]
    }

    service.add_managed_skill(_ctx(), package)
    result = service.read_managed_skill_file(
        _ctx(), "utf16-demo", "scripts/config.ps1"
    )
    stored = base64.b64decode(result["content_base64"])

    assert stored.startswith(codecs.BOM_UTF16_BE)
    decoded = stored[len(codecs.BOM_UTF16_BE):].decode("utf-16-be")
    assert secret not in decoded
    assert "[REDACTED_SECRET]" in decoded


def test_catalog_limit_does_not_read_every_full_node():
    fs = _FS()
    service = _service(fs)
    root = "ctx://acct/agents/agent/skills/generated"
    for index in range(200):
        uri = f"{root}/skill-{index:03d}"
        fs.nodes[uri] = ContextNode(
            uri=uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:agent",
            abstract=f"Skill {index}",
            overview="",
            content="# Skill\n",
            metadata={"ogmemory_generated": True, "publication_status": "published"},
        )

    result = service.list_published_skills(_ctx(), limit=1)

    assert len(result["skills"]) == 1
    assert result["truncated"] is True
    assert fs.read_calls == 2


def test_search_scan_limit_handles_unpublished_nodes_without_cursor_error():
    fs = _FS()
    service = _service(fs)
    service._cfg.skill_search_scan_limit = 2
    root = "ctx://acct/agents/agent/skills/generated"
    for name, published in (("a", True), ("b", False), ("c", True)):
        uri = f"{root}/{name}"
        fs.nodes[uri] = ContextNode(
            uri=uri,
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:agent",
            abstract="Review skill",
            overview="",
            content="# Skill\n",
            metadata={
                "ogmemory_generated": True,
                "publication_status": "published" if published else "draft",
            },
        )

    result = search_skills(service, query="review", top_k=2, agent_id="agent")

    assert [row["name"] for row in result["skills"]] == ["a", "c"]
    assert result["truncated"] is False


def test_catalog_lists_and_reads_same_name_by_source():
    fs = _FS()
    service = _service(fs)
    service.add_managed_skill(_ctx(), CONTENT)
    generated_uri = "ctx://acct/agents/agent/skills/generated/review-code"
    fs.nodes[generated_uri] = ContextNode(
        uri=generated_uri,
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:agent",
        abstract="Generated review skill.",
        overview="Agent skill: review-code",
        content="# Generated\n",
        metadata={"ogmemory_generated": True, "publication_status": "published"},
    )

    listing = service.list_published_skills(_ctx(), limit=50)
    first = service.list_published_skills(_ctx(), limit=1)
    second = service.list_published_skills(_ctx(), limit=1, cursor=first["next_cursor"])

    assert [(row["name"], row["source"]) for row in listing["skills"]] == [
        ("review-code", "generated"),
        ("review-code", "imported"),
    ]
    assert service.read_published_skill(_ctx(), "review-code", "generated").content == "# Generated\n"
    assert service.read_published_skill(_ctx(), "review-code", "imported").content.startswith("---")
    assert first["skills"][0]["source"] == "generated"
    assert second["skills"][0]["source"] == "imported"
    with pytest.raises(ValueError, match="source is required"):
        service.read_published_skill(_ctx(), "review-code")


def test_catalog_supports_managed_package_name():
    fs = _FS()
    service = _service(fs)
    package = {
        "files": [{
            "path": "Package_One/SKILL.md",
            "content": "---\nname: Package_One\ndescription: Package.\n---\n\n# Package\n",
        }]
    }
    service.add_managed_skill(_ctx(), package)

    listing = service.list_published_skills(_ctx(), limit=1)

    assert listing["skills"][0]["name"] == "Package_One"
    assert listing["skills"][0]["source"] == "imported"
    assert service.read_published_skill(_ctx(), "Package_One").content.startswith("---")


def test_catalog_reads_legacy_generated_skill_after_directory_migration():
    fs = _FS()
    service = _service(fs)
    legacy_uri = "ctx://acct/agents/agent/skills/review-code"
    fs.nodes[legacy_uri] = ContextNode(
        uri=legacy_uri,
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:agent",
        abstract="Legacy generated skill.",
        overview="Agent skill: review-code",
        content="# Legacy generated\n",
        metadata={"ogmemory_generated": True, "publication_status": "published"},
    )

    listing = service.list_published_skills(_ctx(), limit=50)
    detail = service.read_published_skill(_ctx(), "review-code", "generated")

    assert listing["skills"] == [
        {
            "name": "review-code",
            "description": "Legacy generated skill.",
            "source": "generated",
            "updated_at": "",
        }
    ]
    assert detail.content == "# Legacy generated\n"


def test_add_rejects_duplicate_managed_name():
    service = _service(_FS())
    service.add_managed_skill(_ctx(), CONTENT)
    with pytest.raises(FileExistsError):
        service.add_managed_skill(_ctx(), CONTENT)


def test_delete_requires_source_and_only_deletes_selected_source():
    fs = _FS()
    service = _service(fs)
    service.add_managed_skill(_ctx(), CONTENT)
    generated_uri = "ctx://acct/agents/agent/skills/generated/review-code"
    fs.nodes[generated_uri] = ContextNode(
        uri=generated_uri,
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:agent",
        abstract="Generated review skill.",
        overview="Agent skill: review-code",
        content="# Generated\n",
        metadata={"ogmemory_generated": True, "publication_status": "published"},
    )
    legacy_generated_uri = "ctx://acct/agents/agent/skills/review-code"
    fs.nodes[legacy_generated_uri] = ContextNode(
        uri=legacy_generated_uri,
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:agent",
        abstract="Legacy generated review skill.",
        overview="Agent skill: review-code",
        content="# Legacy generated\n",
        metadata={"ogmemory_generated": True, "publication_status": "published"},
    )

    with pytest.raises(ValueError, match="source is required"):
        service.delete_published_skill(_ctx(), "review-code", None)

    result = service.delete_published_skill(_ctx(), "review-code", "imported")

    assert result == {
        "name": "review-code",
        "uri": "ogmemory://skills/imported/review-code",
        "source": "imported",
        "action": "delete",
    }
    assert generated_uri in fs.nodes
    assert "ctx://acct/agents/agent/skills/imported/review-code" not in fs.nodes

    generated_result = service.delete_published_skill(
        _ctx(), "review-code", "generated"
    )

    assert generated_result["uri"] == "ogmemory://skills/generated/review-code"
    assert generated_uri not in fs.nodes
    assert legacy_generated_uri not in fs.nodes


def test_delete_requires_privileged_role():
    service = _service(_FS())
    with pytest.raises(AuthorizationError):
        service.delete_published_skill(_ctx(Role.MEMBER), "review-code", "generated")


def test_add_maps_create_race_to_duplicate_conflict():
    class _RaceFS(_FS):
        def exists(self, uri, ctx):
            return False

        def write_node(self, node, ctx):
            raise ConcurrentModificationError(node.uri, 0, 1)

    with pytest.raises(FileExistsError, match="already exists"):
        _service(_RaceFS()).add_managed_skill(_ctx(), CONTENT)


def test_add_requires_privileged_role():
    with pytest.raises(AuthorizationError):
        _service(_FS()).add_managed_skill(_ctx(Role.MEMBER), CONTENT)
