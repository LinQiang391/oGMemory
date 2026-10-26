"""Cross-profile contract for the shared code search and edit tools.

The retrieval facade, workspace resolver, and lifecycle callback are test
doubles. This is contract-level equality evidence for shared tool owners; it
is not HTTP/MCP, durable backend, or Claude E2E evidence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.models import RequestContext
from core.resource_identity import compute_root_fingerprint
from server.mcp_tools import search_code
from service.code_file_edit import CodeFileEditService
from service.code_workspace_sync import ResolvedCodeWorkspace


class _NoAuth:
    def role_control_active(self):
        return False


class _SearchFacade:
    def __init__(self, profile: bool) -> None:
        self._cfg = SimpleNamespace(
            code_memory_only=profile,
            mcp_max_request_bytes=1_048_576,
            mcp_max_response_chars=50_000,
        )
        self.params = None

    def get_auth_service(self):
        return _NoAuth()

    def build_context(self, params, identity=None):
        del identity
        return _ctx(session_id=params.get("sessionId") or "session")

    def search_code_resources(self, params):
        self.params = dict(params)
        return {
            "status": "ok",
            "evidence": [
                {
                    "locator": "repo/src/service.py:1-2",
                    "relative_path": "src/service.py",
                    "root_slug": "repo",
                    "start_line": 1,
                    "end_line": 2,
                    "symbol": "service",
                    "excerpt": "def service():\n    return 1",
                }
            ],
            "diagnostics": {"private": "omitted from MCP"},
        }


class _Resolver:
    def __init__(self, root: Path) -> None:
        self.workspace = ResolvedCodeWorkspace(
            root_id="root-1",
            root_slug="repo",
            root_fingerprint=compute_root_fingerprint(root),
            physical_root=root.resolve(),
            max_files=100,
            language_hints={},
        )
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def resolve(self, *, ctx, root_slugs=()):
        del ctx
        self.calls.append(("manual", tuple(root_slugs)))
        return self.workspace

    def resolve_registered(self, *, ctx, root_slugs=()):
        del ctx
        self.calls.append(("registered", tuple(root_slugs)))
        return self.workspace


def _ctx(*, session_id: str = "session") -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id=session_id,
        trace_id="trace",
    )


def _exercise_profile(profile: bool, root: Path) -> dict[str, Any]:
    root.mkdir()
    target = root / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("def service():\n    return 1\n", encoding="utf-8")
    ctx = _ctx()

    search_service = _SearchFacade(profile)
    search_result = search_code(
        search_service,
        query="function service",
        scope={"root_slugs": ["repo"]},
        agent_id="agent",
        session_id="session",
    )

    resolver = _Resolver(root)
    refresh_calls: list[str] = []
    invalidations: list[str | None] = []

    def refresh_file(*, workspace, relative_path, ctx):
        del workspace, ctx
        refresh_calls.append(relative_path)
        return {"status": "PENDING"}

    editor = CodeFileEditService(
        workspace_resolver=resolver,  # type: ignore[arg-type]
        code_retrieval_mode="full",
        refresh_code_file=refresh_file,
        invalidate_cache=invalidations.append,
    )
    edit_result = editor.edit(
        file_path="src/service.py",
        old_string="return 1",
        new_string="return 2",
        replace_all=False,
        root_slug="repo",
        ctx=ctx,
    )

    search_params = dict(search_service.params)
    search_ctx = search_params.pop("_ctx")
    return {
        "search": search_result,
        "search_params": search_params,
        "search_context": {
            "account_id": search_ctx.account_id,
            "agent_id": search_ctx.agent_id,
            "session_id": search_ctx.session_id,
        },
        "edit": edit_result,
        "edited_content": target.read_text(encoding="utf-8"),
        "resolver_calls": resolver.calls,
        "refresh_calls": refresh_calls,
        "invalidations": invalidations,
    }


def test_shared_code_tool_outputs_and_side_effects_are_profile_invariant(tmp_path):
    normal = _exercise_profile(False, tmp_path / "normal")
    code_only = _exercise_profile(True, tmp_path / "code-only")

    assert normal == code_only
    assert "# code_search status: ok" in normal["search"]
    assert normal["edit"]["refresh_state"] == "accepted"
    assert normal["edited_content"] == "def service():\n    return 2\n"
    assert normal["resolver_calls"] == [("registered", ("repo",))]
    assert normal["refresh_calls"] == ["src/service.py"]
    assert normal["invalidations"] == ["acct"]
