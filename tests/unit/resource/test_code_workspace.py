from __future__ import annotations

import subprocess
import time

import pytest

from core.resource_identity import ResourceIdentityError
from extraction.code_workspace import (
    BoundedCodeWorkspaceWalkStats,
    iter_bounded_code_files,
    iter_code_files,
    validate_code_candidate_path,
)


def test_workspace_scan_is_sorted_and_skips_generated_directories(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "b.py").write_text("print('b')\n", encoding="utf-8")
    (root / "a.ts").write_text("export const a = 1\n", encoding="utf-8")
    (root / "README.md").write_text("docs\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.ts").write_text("ignored\n", encoding="utf-8")

    paths = [path.relative_to(root).as_posix() for path in iter_code_files(root)]

    assert paths == ["a.ts", "pkg/b.py"]


def test_workspace_scan_respects_max_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a\n", encoding="utf-8")
    (root / "b.py").write_text("b\n", encoding="utf-8")

    assert [path.name for path in iter_code_files(root, max_files=1)] == ["a.py"]


def test_workspace_scan_stops_before_later_escape_after_file_cap(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a\n", encoding="utf-8")
    later = root / "z"
    later.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    (later / "escape.py").symlink_to(outside)

    paths = [path.relative_to(root).as_posix() for path in iter_code_files(root, max_files=1)]

    assert paths == ["a.py"]


def test_workspace_scan_rejects_symlink_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)

    with pytest.raises(ResourceIdentityError, match="escapes"):
        list(iter_code_files(root))


def test_workspace_scan_accepts_internal_file_symlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "target.py"
    target.write_text("ok\n", encoding="utf-8")
    (root / "alias.py").symlink_to(target)

    paths = [path.name for path in iter_code_files(root)]

    assert paths == ["alias.py", "target.py"]


def test_workspace_scan_uses_git_ignore_rules_when_available(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("ignored.py\ngenerated/\n", encoding="utf-8")
    (root / "tracked.py").write_text("tracked = True\n", encoding="utf-8")
    (root / "untracked.py").write_text("untracked = True\n", encoding="utf-8")
    (root / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    (root / "generated").mkdir()
    (root / "generated" / "output.py").write_text("generated = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "tracked.py", ".gitignore"],
        check=True,
        capture_output=True,
    )

    paths = [path.relative_to(root).as_posix() for path in iter_code_files(root)]

    assert paths == ["tracked.py", "untracked.py"]


def test_bounded_candidate_scan_uses_git_visible_ignore_rules_when_available(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("ignored.py\ngenerated/\n", encoding="utf-8")
    (root / "tracked.py").write_text("tracked = True\n", encoding="utf-8")
    (root / "untracked.py").write_text("untracked = True\n", encoding="utf-8")
    (root / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    (root / "generated").mkdir()
    (root / "generated" / "output.py").write_text("generated = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "tracked.py", ".gitignore"],
        check=True,
        capture_output=True,
    )

    files = list(iter_bounded_code_files(root, entry_limit=100))

    assert [item.relative_path for item in files] == ["tracked.py", "untracked.py"]


def test_bounded_git_candidate_scan_yields_validated_paths_collected_before_entry_cap(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")
    stats = BoundedCodeWorkspaceWalkStats()

    files = list(iter_bounded_code_files(root, entry_limit=1, stats=stats))

    assert [item.relative_path for item in files] == ["a.py"]
    assert stats.visited_entries == 1
    assert stats.entry_limit_reached is True


def test_workspace_scan_falls_back_when_git_is_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "visible.py").write_text("visible = True\n", encoding="utf-8")
    (root / "vendor").mkdir()
    (root / "vendor" / "hidden.py").write_text("hidden = True\n", encoding="utf-8")
    monkeypatch.setattr(
        "extraction.code_workspace.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )

    paths = [path.relative_to(root).as_posix() for path in iter_code_files(root)]

    assert paths == ["visible.py"]


def test_bounded_candidate_scan_falls_back_when_git_is_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "visible.py").write_text("visible = True\n", encoding="utf-8")
    (root / "vendor").mkdir()
    (root / "vendor" / "hidden.py").write_text("hidden = True\n", encoding="utf-8")
    monkeypatch.setattr(
        "extraction.code_workspace.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )

    files = list(iter_bounded_code_files(root, entry_limit=100))

    assert [item.relative_path for item in files] == ["visible.py"]


def test_workspace_scan_allows_explicit_hints_for_regular_nonstandard_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (root / "template.txt").write_text("def render():\n    return 1\n", encoding="utf-8")

    paths = [
        path.relative_to(root).as_posix()
        for path in iter_code_files(
            root,
            language_hints={"Dockerfile": "shell", "template.txt": "python"},
        )
    ]

    assert paths == ["Dockerfile", "template.txt"]


def test_workspace_scan_rejects_unsafe_or_unknown_language_hints(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")

    with pytest.raises(ResourceIdentityError, match="relative_path"):
        list(iter_code_files(root, language_hints={"../Dockerfile": "shell"}))
    with pytest.raises(ValueError, match="unsupported code language hint"):
        list(iter_code_files(root, language_hints={"Dockerfile": "unknown"}))


def test_bounded_candidate_scan_stops_at_entry_cap_and_reports_deadline(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")

    capped = BoundedCodeWorkspaceWalkStats()
    files = list(
        iter_bounded_code_files(
            root,
            entry_limit=1,
            stats=capped,
        )
    )

    assert [item.relative_path for item in files] == ["a.py"]
    assert capped.visited_entries == 1
    assert capped.entry_limit_reached is True

    expired = BoundedCodeWorkspaceWalkStats()
    assert list(
        iter_bounded_code_files(
            root,
            deadline=time.monotonic() - 1,
            stats=expired,
        )
    ) == []
    assert expired.deadline_reached is True


def test_candidate_path_validator_preserves_existing_containment_and_ignore_rules(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "visible.py"
    source.write_text("visible = True\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)

    validated = validate_code_candidate_path(root, "visible.py")
    assert validated is not None
    assert validated.relative_path == "visible.py"
    assert validated.path == source
    assert validate_code_candidate_path(root, "node_modules/ignored.py") is None

    with pytest.raises(ResourceIdentityError, match="escapes"):
        validate_code_candidate_path(root, "escape.py")
