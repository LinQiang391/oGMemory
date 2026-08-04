from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_example_deploy_keeps_auth_disabled_until_user_sets_real_key():
    cfg = yaml.safe_load((ROOT / "deploy" / "ogmemory.example.yaml").read_text(encoding="utf-8"))

    assert cfg["mcp"]["enabled"] is True
    assert cfg["auth"]["role_control_enabled"] is False
    assert "root_api_key" not in cfg["auth"]


def test_standalone_dockerfile_remains_jenkins_sed_compatible():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11-slim-bookworm")
    assert "GO_BASE_IMAGE" not in dockerfile
    assert "FROM ${" not in dockerfile
    assert "go build" not in dockerfile
    assert "COPY agfs/agfs-server" in dockerfile
    assert "COPY docker/entrypoint-standalone.sh /opt/entrypoint.sh" in dockerfile
    assert "entrypoint-standalone-mcp.sh" not in dockerfile
    assert "ENV OGMEM_MCP_DEFAULT_TOOLSET=read-only" in dockerfile

    patched = re.sub(
        r"^.*$",
        "FROM docker.xuanyuan.run/python:3.11-slim-bookworm",
        dockerfile,
        count=1,
    )
    assert "FROM ${" not in patched


def test_standalone_dockerfile_defines_mcp_env_used_by_default_config():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")
    config = (ROOT / "deploy" / "ogmemory.example.yaml").read_text(encoding="utf-8")

    mcp_envs = set(re.findall(r"\$\{(OGMEM_MCP_[A-Z0-9_]+)\}", config))
    docker_envs = set(re.findall(r"^ENV (OGMEM_MCP_[A-Z0-9_]+)=", dockerfile, flags=re.M))

    assert mcp_envs <= docker_envs


def test_standalone_dockerfile_keeps_mcp_dependency_delta_minimal():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")

    assert '"mcp[cli]>=2.0.0,<3"' in dockerfile
    assert "libsqlite3-0" not in dockerfile
    assert "rank-bm25" not in dockerfile
    assert "nltk" not in dockerfile
    assert "python-dotenv" not in dockerfile


def test_standalone_entrypoint_uses_one_process_with_threads():
    entrypoint = (ROOT / "docker" / "entrypoint-standalone.sh").read_text(encoding="utf-8")

    assert '--workers 1' in entrypoint
    assert '--worker-class gthread' in entrypoint
    assert '--threads "${OGMEM_THREADS}"' in entrypoint
    assert 'OGMEM_THREADS=${OGMEM_THREADS:-${OGMEM_WORKERS:-2}}' in entrypoint
