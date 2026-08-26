"""Unit coverage for atomic registered-workspace source editing.

``RecordingResolver`` and callback functions replace the protected control
store and durable lifecycle. These tests prove filesystem safety, exact-match
semantics, and local refresh delegation only; they are not backend integration
or MCP transport evidence.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from core.models import RequestContext
from core.resource_identity import compute_root_fingerprint
from service import code_file_edit as edit_module
from service.code_file_edit import CodeFileEditService
from service.code_workspace_sync import ResolvedCodeWorkspace


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _workspace(root: Path) -> ResolvedCodeWorkspace:
    return ResolvedCodeWorkspace(
        root_id="root-1",
        root_slug="repo",
        root_fingerprint=compute_root_fingerprint(root),
        physical_root=root.resolve(),
        max_files=100,
        language_hints={"src/service.py": "python"},
    )


class RecordingResolver:
    def __init__(self, workspace: ResolvedCodeWorkspace) -> None:
        self.workspace = workspace
        self.calls: list[tuple[str, tuple[str, ...], RequestContext]] = []

    def resolve(self, *, ctx, root_slugs=()):
        self.calls.append(("manual", tuple(root_slugs), ctx))
        return self.workspace

    def resolve_registered(self, *, ctx, root_slugs=()):
        self.calls.append(("registered", tuple(root_slugs), ctx))
        return self.workspace


def _service(
    root: Path,
    *,
    mode: str = "full",
    refresh=None,
    invalidations: list[str | None] | None = None,
) -> tuple[CodeFileEditService, RecordingResolver, list[dict[str, object]]]:
    resolver = RecordingResolver(_workspace(root))
    refresh_calls: list[dict[str, object]] = []

    def default_refresh(**kwargs):
        refresh_calls.append(dict(kwargs))
        return {"status": "PENDING"}

    callback = refresh or default_refresh
    service = CodeFileEditService(
        workspace_resolver=resolver,  # type: ignore[arg-type]
        code_retrieval_mode=mode,
        refresh_code_file=callback,
        invalidate_cache=(invalidations.append if invalidations is not None else None),
    )
    return service, resolver, refresh_calls


def test_exact_edit_is_atomic_preserves_mode_and_refreshes_full_mode(tmp_path):
    root = tmp_path / "repo"
    target = root / "src" / "service.py"
    target.parent.mkdir(parents=True)
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    target.chmod(0o640)
    invalidations: list[str | None] = []
    service, resolver, refresh_calls = _service(root, invalidations=invalidations)

    result = service.edit(
        file_path="src/service.py",
        old_string="return 1",
        new_string="return 2",
        replace_all=False,
        root_slug="repo",
        ctx=_ctx(),
    )

    assert result == {
        "ok": True,
        "relative_path": "src/service.py",
        "occurrences": 1,
        "edit_state": "modified",
        "refresh_state": "accepted",
    }
    assert target.read_text(encoding="utf-8") == "def value():\n    return 2\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert resolver.calls == [("registered", ("repo",), resolver.calls[0][2])]
    assert resolver.calls[0][2].account_id == "acct"
    assert refresh_calls[0]["relative_path"] == "src/service.py"
    assert refresh_calls[0]["workspace"] == resolver.workspace
    assert invalidations == ["acct"]
    assert not list(target.parent.glob(".service.py.ogm-edit-*"))


def test_replace_all_and_contained_absolute_path_use_request_local_refresh(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "values.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    invalidations: list[str | None] = []
    service, resolver, refresh_calls = _service(
        root,
        mode="candidate_local",
        invalidations=invalidations,
    )

    result = service.edit(
        file_path=str(target),
        old_string="value = 1",
        new_string="value = 2",
        replace_all=True,
        ctx=_ctx(),
    )

    assert result["occurrences"] == 2
    assert result["refresh_state"] == "request_local"
    assert target.read_text(encoding="utf-8") == "value = 2\nvalue = 2\n"
    assert resolver.calls[0][0] == "manual"
    assert refresh_calls == []
    assert invalidations == ["acct"]


@pytest.mark.parametrize(
    ("old_string", "replace_all", "message"),
    [
        ("missing", False, "old_string not found"),
        ("value = 1", False, "matched 2 times"),
    ],
)
def test_match_validation_leaves_target_unchanged(
    tmp_path,
    old_string,
    replace_all,
    message,
):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "values.py"
    original = "value = 1\nvalue = 1\n"
    target.write_text(original, encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)

    with pytest.raises(ValueError, match=message):
        service.edit(
            file_path="values.py",
            old_string=old_string,
            new_string="value = 2",
            replace_all=replace_all,
            ctx=_ctx(),
        )

    assert target.read_text(encoding="utf-8") == original
    assert refresh_calls == []
    assert not list(root.glob(".values.py.ogm-edit-*"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("file_path", "", "file_path is required"),
        ("old_string", "", "old_string must be a non-empty string"),
        ("new_string", None, "new_string must be a string"),
        ("replace_all", 1, "replace_all must be a boolean"),
        ("root_slug", "", "root_slug must be a non-empty string"),
    ],
)
def test_edit_request_validation_fails_before_workspace_resolution(
    tmp_path,
    field,
    value,
    message,
):
    root = tmp_path / "repo"
    root.mkdir()
    service, resolver, _refresh_calls = _service(root)
    params = {
        "file_path": "service.py",
        "old_string": "old",
        "new_string": "new",
        "replace_all": False,
        "root_slug": None,
        "ctx": _ctx(),
    }
    params[field] = value

    with pytest.raises(ValueError, match=message):
        service.edit(**params)

    assert resolver.calls == []


def test_workspace_escape_and_symlink_escape_are_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)

    for unsafe_path in ("../outside.py", str(outside)):
        with pytest.raises(ValueError, match="contained file"):
            service.edit(
                file_path=unsafe_path,
                old_string="value = 1",
                new_string="value = 2",
                replace_all=False,
                ctx=_ctx(),
            )

    link = root / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="contained file"):
        service.edit(
            file_path="linked.py",
            old_string="value = 1",
            new_string="value = 2",
            replace_all=False,
            ctx=_ctx(),
        )

    assert outside.read_text(encoding="utf-8") == "value = 1\n"
    assert refresh_calls == []


def test_non_regular_and_non_utf8_targets_are_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    directory = root / "folder"
    directory.mkdir()
    binary = root / "binary.py"
    binary.write_bytes(b"\xff\xfe")
    service, _resolver, refresh_calls = _service(root)

    with pytest.raises(ValueError, match="regular file"):
        service.edit(
            file_path="folder",
            old_string="old",
            new_string="new",
            replace_all=False,
            ctx=_ctx(),
        )
    with pytest.raises(ValueError, match="UTF-8"):
        service.edit(
            file_path="binary.py",
            old_string="old",
            new_string="new",
            replace_all=False,
            ctx=_ctx(),
        )

    assert refresh_calls == []


def test_oversized_target_is_rejected_before_reading_contents(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "large.py"
    with target.open("wb") as stream:
        stream.truncate(edit_module.MAX_CODE_FILE_SIZE + 1)
    service, _resolver, refresh_calls = _service(root)
    monkeypatch.setattr(
        edit_module.os,
        "read",
        lambda *_args: pytest.fail("oversized target must not be read"),
    )

    with pytest.raises(ValueError, match="exceeds maximum editable size"):
        service.edit(
            file_path="large.py",
            old_string="value",
            new_string="updated",
            replace_all=False,
            ctx=_ctx(),
        )

    assert refresh_calls == []


def test_oversized_replacement_is_rejected_before_atomic_publication(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    target.write_text("x\n", encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)
    monkeypatch.setattr(edit_module, "MAX_CODE_FILE_SIZE", 8)

    with pytest.raises(ValueError, match="exceeds maximum editable size"):
        service.edit(
            file_path="service.py",
            old_string="x",
            new_string="123456789",
            replace_all=False,
            ctx=_ctx(),
        )

    assert target.read_text(encoding="utf-8") == "x\n"
    assert refresh_calls == []
    assert not list(root.glob(".service.py.ogm-edit-*"))


def test_utf8_replacement_budget_uses_bytes_not_character_count(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "unicode.py"
    target.write_text("é\n", encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)
    monkeypatch.setattr(edit_module, "MAX_CODE_FILE_SIZE", 3)

    with pytest.raises(ValueError, match="exceeds maximum editable size"):
        service.edit(
            file_path="unicode.py",
            old_string="é",
            new_string="你",
            replace_all=False,
            ctx=_ctx(),
        )

    assert target.read_text(encoding="utf-8") == "é\n"
    assert refresh_calls == []


def test_write_failure_preserves_original_and_cleans_temporary_file(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    original = "value = 1\n"
    target.write_text(original, encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(edit_module.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="write failed"):
        service.edit(
            file_path="service.py",
            old_string="value = 1",
            new_string="value = 2",
            replace_all=False,
            ctx=_ctx(),
        )

    assert target.read_text(encoding="utf-8") == original
    assert refresh_calls == []
    assert not list(root.glob(".service.py.ogm-edit-*"))


def test_concurrent_target_change_aborts_without_overwriting_other_writer(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    service, _resolver, refresh_calls = _service(root)
    original_revalidate = edit_module._revalidate_target

    def change_then_revalidate(**kwargs):
        target.write_text("value = concurrent\n", encoding="utf-8")
        original_revalidate(**kwargs)

    monkeypatch.setattr(edit_module, "_revalidate_target", change_then_revalidate)

    with pytest.raises(RuntimeError, match="target changed during edit"):
        service.edit(
            file_path="service.py",
            old_string="value = 1",
            new_string="value = 2",
            replace_all=False,
            ctx=_ctx(),
        )

    assert target.read_text(encoding="utf-8") == "value = concurrent\n"
    assert refresh_calls == []
    assert not list(root.glob(".service.py.ogm-edit-*"))


def test_refresh_failure_is_reported_after_successful_edit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    invalidations: list[str | None] = []

    def fail_refresh(**_kwargs):
        raise RuntimeError("simulated refresh failure")

    service, _resolver, _refresh_calls = _service(
        root,
        refresh=fail_refresh,
        invalidations=invalidations,
    )

    result = service.edit(
        file_path="service.py",
        old_string="value = 1",
        new_string="value = 2",
        replace_all=False,
        ctx=_ctx(),
    )

    assert result["ok"] is True
    assert result["edit_state"] == "modified"
    assert result["refresh_state"] == "failed"
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert invalidations == ["acct"]


def test_visible_refresh_state_is_projected_without_raw_lifecycle_payload(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")

    def visible_refresh(**_kwargs):
        return {"status": "PENDING", "index_visible": True, "private": "/srv/repo"}

    service, _resolver, _refresh_calls = _service(root, refresh=visible_refresh)
    result = service.edit(
        file_path="service.py",
        old_string="value = 1",
        new_string="value = 2",
        replace_all=False,
        ctx=_ctx(),
    )

    assert result["refresh_state"] == "visible"
    assert "/srv/repo" not in str(result)
    assert set(result) == {
        "ok",
        "relative_path",
        "occurrences",
        "edit_state",
        "refresh_state",
    }


def test_directory_fsync_failure_reports_modified_unconfirmed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    service, _resolver, _refresh_calls = _service(root)
    real_fsync = edit_module.os.fsync

    def fail_directory_fsync(descriptor):
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            raise OSError("simulated directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(edit_module.os, "fsync", fail_directory_fsync)

    result = service.edit(
        file_path="service.py",
        old_string="value = 1",
        new_string="value = 2",
        replace_all=False,
        ctx=_ctx(),
    )

    assert result["edit_state"] == "modified_unconfirmed"
    assert target.read_text(encoding="utf-8") == "value = 2\n"
