from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_SCRIPTS = (
    "examples/ogmemory_mcp_sse_demo/stdlib/ogmemory_mcp_sse_demo.py",
    "examples/ogmemory_mcp_sse_demo/sdk/ogmemory_mcp_sse_demo.py",
)


@pytest.mark.parametrize("script", DEMO_SCRIPTS)
def test_default_url_matches_documented_sse_port(
    monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    for name in (
        "OGMEM_MCP_API_KEY",
        "OGMEM_MCP_ACCOUNT_ID",
        "OGMEM_MCP_USER_ID",
        "OGMEM_MCP_AGENT_ID",
    ):
        monkeypatch.setenv(name, "test-value")
    monkeypatch.delenv("OGMEM_MCP_URL", raising=False)

    path = REPO_ROOT / script
    spec = importlib.util.spec_from_file_location(f"demo_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    url, _ = module.config()

    assert url == "http://127.0.0.1:8191/mcp"


def test_readme_declares_runnable_sse_debug_config() -> None:
    readme = (REPO_ROOT / "examples/ogmemory_mcp_sse_demo/README.md").read_text(
        encoding="utf-8"
    )

    assert "export OGMEM_MCP_PORT=8191" in readme
    assert "export OG_ROLE_CONTROL_ENABLED=true" in readme
    assert "export OG_ROOT_API_KEY='local-debug-root-key'" in readme
    assert "export OGMEM_MCP_DEFAULT_TOOLSET=all" in readme
    assert "root key 仅限本机隔离调试" in readme
