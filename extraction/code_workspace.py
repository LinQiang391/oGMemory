"""Deterministic code workspace discovery for Python 3.11."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.resource_identity import (
    ResourceIdentityError,
    normalize_posix_relative_path,
)
from extraction.code_processor import (
    SUPPORTED_CODE_EXTENSIONS,
    validate_language_hint,
)

DEFAULT_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "target",
        ".next",
        ".turbo",
        "coverage",
    }
)

_GIT_BOUNDED_PATH_BYTES_LIMIT = 64 * 1024


@dataclass(frozen=True)
class ValidatedCodeWorkspaceFile:
    """One internal, root-contained code candidate.

    This descriptor is intentionally for extraction/retrieval internals only.
    Public callers must project its logical ``relative_path`` and must never
    serialize ``path``.
    """

    relative_path: str
    path: Path


@dataclass
class BoundedCodeWorkspaceWalkStats:
    """Mutable counters for a cancellation-aware safe workspace walk."""

    visited_entries: int = 0
    entry_limit_reached: bool = False
    deadline_reached: bool = False
    validation_failures: int = 0


def iter_bounded_code_files(
    workspace_root: str | Path,
    *,
    skip_dirs: set[str] | frozenset[str] | None = None,
    language_hints: Mapping[str, str] | None = None,
    start_relative_path: str | Path | None = None,
    entry_limit: int = 100_000,
    deadline: float | None = None,
    stats: BoundedCodeWorkspaceWalkStats | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Iterator[ValidatedCodeWorkspaceFile]:
    """Yield safe code candidates with deterministic traversal safety bounds.

    Candidate-local discovery follows the same Git-visible/ignore semantics as
    :func:`iter_code_files` when Git is available.  It additionally bounds the
    Git result stream and its safe-walker fallback before enough eligible files
    have been accepted.  Both paths share the exact containment, symlink,
    exclusion, regular-file and language checks.
    """

    if isinstance(entry_limit, bool) or not isinstance(entry_limit, int) or entry_limit < 1:
        raise ValueError("entry_limit must be a positive integer")
    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace root is not a directory")
    ignored = set(skip_dirs or DEFAULT_SKIP_DIRS)
    hints = _normalize_language_hints(language_hints)
    state = stats if stats is not None else BoundedCodeWorkspaceWalkStats()
    start = root
    if start_relative_path is not None:
        candidate = Path(start_relative_path)
        if candidate.is_absolute():
            raise ValueError("start_relative_path must be contained")
        try:
            start = (root / candidate).resolve(strict=True)
            _ensure_contained(root, start)
        except (OSError, ResourceIdentityError):
            raise ValueError("start_relative_path must be contained") from None
        if not start.is_dir():
            return

    start_relative = None if start == root else start.relative_to(root).as_posix()

    def expired() -> bool:
        if deadline is not None and monotonic_clock() >= deadline:
            state.deadline_reached = True
            return True
        return False

    def walk(current: Path) -> Iterator[ValidatedCodeWorkspaceFile]:
        if expired() or state.entry_limit_reached:
            return
        try:
            _ensure_contained(root, current)
            with os.scandir(current) as scanner:
                entries = sorted(
                    scanner,
                    key=lambda entry: entry.name
                    + ("/" if entry.is_dir(follow_symlinks=False) else ""),
                )
        except (OSError, ResourceIdentityError):
            state.validation_failures += 1
            return
        for entry in entries:
            if expired():
                return
            if state.visited_entries >= entry_limit:
                state.entry_limit_reached = True
                return
            state.visited_entries += 1
            path = current / entry.name
            try:
                is_symlink = entry.is_symlink()
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                state.validation_failures += 1
                continue
            if is_symlink:
                # ``_validate_candidate`` resolves then proves containment;
                # escaped links are dropped rather than becoming a path leak.
                try:
                    item = _validate_candidate(root, path, ignored, hints)
                except ResourceIdentityError:
                    state.validation_failures += 1
                    continue
                if item is not None:
                    yield ValidatedCodeWorkspaceFile(relative_path=item[0], path=item[1])
                continue
            if is_directory:
                if entry.name in ignored or entry.name.startswith(".git"):
                    continue
                yield from walk(path)
                continue
            try:
                item = _validate_candidate(root, path, ignored, hints)
            except ResourceIdentityError:
                state.validation_failures += 1
                continue
            if item is not None:
                yield ValidatedCodeWorkspaceFile(relative_path=item[0], path=item[1])

    git_candidates = _bounded_git_visible_candidates(
        root,
        start_relative_path=start_relative,
        entry_limit=entry_limit,
        deadline=deadline,
        stats=state,
        monotonic_clock=monotonic_clock,
    )
    if git_candidates is None:
        yield from walk(start)
        return
    for candidate in git_candidates:
        if expired():
            return
        try:
            item = _validate_candidate(root, candidate, ignored, hints)
        except ResourceIdentityError:
            state.validation_failures += 1
            continue
        if item is not None:
            yield ValidatedCodeWorkspaceFile(relative_path=item[0], path=item[1])


def _bounded_git_visible_candidates(
    root: Path,
    *,
    start_relative_path: str | None,
    entry_limit: int,
    deadline: float | None,
    stats: BoundedCodeWorkspaceWalkStats,
    monotonic_clock: Callable[[], float],
) -> list[Path] | None:
    """Return a bounded Git-visible path list, or ``None`` for safe fallback.

    ``git ls-files`` is streamed rather than captured wholesale: an unusually
    large worktree cannot allocate an unbounded subprocess result before the
    candidate walker has a chance to enforce its entry/deadline budget.
    """

    if deadline is not None and monotonic_clock() >= deadline:
        stats.deadline_reached = True
        return []
    argv = [
        "git",
        "-C",
        str(root),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    if start_relative_path is not None:
        argv.extend(("--", start_relative_path))
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if process.stdout is None:
        _terminate_bounded_process(process)
        return None

    paths: list[Path] = []
    pending = bytearray()

    def consume(data: bytes) -> bool:
        """Return false after an intentionally reached bounded stream limit."""

        pending.extend(data)
        if len(pending) > _GIT_BOUNDED_PATH_BYTES_LIMIT:
            stats.validation_failures += 1
            return False
        while True:
            delimiter = pending.find(b"\0")
            if delimiter < 0:
                return True
            raw_path = bytes(pending[:delimiter])
            del pending[: delimiter + 1]
            if not raw_path:
                continue
            if stats.visited_entries >= entry_limit:
                stats.entry_limit_reached = True
                return False
            stats.visited_entries += 1
            paths.append(root / Path(os.fsdecode(raw_path)))

    selector: selectors.BaseSelector | None = None
    try:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            if deadline is not None:
                remaining = deadline - monotonic_clock()
                if remaining <= 0:
                    stats.deadline_reached = True
                    _terminate_bounded_process(process)
                    return []
                wait_seconds = min(remaining, 0.1)
            else:
                wait_seconds = 0.1
            events = selector.select(wait_seconds)
            if events:
                data = os.read(process.stdout.fileno(), 4096)
                if data and not consume(data):
                    _terminate_bounded_process(process)
                    return _sorted_bounded_git_paths(root, paths) if stats.entry_limit_reached else None
            if process.poll() is None:
                continue
            while True:
                data = os.read(process.stdout.fileno(), 4096)
                if not data:
                    break
                if not consume(data):
                    _terminate_bounded_process(process)
                    return _sorted_bounded_git_paths(root, paths) if stats.entry_limit_reached else None
            return _sorted_bounded_git_paths(root, paths) if process.wait() == 0 else None
    except (OSError, ValueError):
        _terminate_bounded_process(process)
        return None
    finally:
        if selector is not None:
            selector.close()
        process.stdout.close()


def _sorted_bounded_git_paths(root: Path, paths: list[Path]) -> list[Path]:
    """Keep the bounded Git source deterministic before candidate validation."""

    def relative_sort_key(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            # Preserve deterministic handling of malformed external output;
            # the normal candidate validator rejects it before it can escape.
            return path.as_posix()

    return sorted(paths, key=relative_sort_key)


def _terminate_bounded_process(process: subprocess.Popen[bytes]) -> None:
    """Stop a bounded Git probe without allowing it to outlive this request."""

    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            pass


def validate_code_candidate_path(
    workspace_root: str | Path,
    candidate: str | Path,
    *,
    skip_dirs: set[str] | frozenset[str] | None = None,
    language_hints: Mapping[str, str] | None = None,
) -> ValidatedCodeWorkspaceFile | None:
    """Revalidate an externally enumerated relative candidate path.

    Ripgrep output and any future accelerator must call this before a file is
    opened.  Absolute paths and traversal attempts cannot bypass the existing
    candidate validator.
    """

    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace root is not a directory")
    raw = Path(candidate)
    if raw.is_absolute():
        return None
    item = _validate_candidate(
        root,
        root / raw,
        set(skip_dirs or DEFAULT_SKIP_DIRS),
        _normalize_language_hints(language_hints),
    )
    if item is None:
        return None
    return ValidatedCodeWorkspaceFile(relative_path=item[0], path=item[1])


def iter_code_files(
    workspace_root: str | Path,
    *,
    skip_dirs: set[str] | frozenset[str] | None = None,
    max_files: int | None = None,
    language_hints: Mapping[str, str] | None = None,
) -> Iterator[Path]:
    """Yield eligible files in deterministic relative-path order.

    Git worktrees use Git's standard ignore semantics as the candidate source.
    A non-Git root, a Git invocation failure, or an unavailable Git executable
    falls back to the existing safe walker.  Every candidate is revalidated so
    Git output never weakens containment, symlink, regular-file, skip-dir, or
    file-cap protections.

    ``language_hints`` is an internal lifecycle override keyed by normalized
    relative path.  A valid hint can make a regular extensionless or
    misclassified file eligible; callers are still responsible for ensuring it
    is only supplied for an authorized registered root.
    """

    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace root is not a directory")
    if max_files is not None and max_files < 1:
        return
    ignored = set(skip_dirs or DEFAULT_SKIP_DIRS)
    normalized_hints = _normalize_language_hints(language_hints)

    yielded = 0
    seen: set[str] = set()
    git_candidates = _git_visible_candidates(root)
    candidates: Iterable[Path] = (
        _walk_candidates(root, ignored) if git_candidates is None else git_candidates
    )
    for candidate in candidates:
        item = _validate_candidate(root, candidate, ignored, normalized_hints)
        if item is None:
            continue
        relative_path, path = item
        if relative_path in seen:
            continue
        seen.add(relative_path)
        yield path
        yielded += 1
        if max_files is not None and yielded >= max_files:
            return


def normalize_language_hints(language_hints: Mapping[str, str] | None) -> dict[str, str]:
    """Return validated language hints keyed by safe POSIX relative path."""

    return _normalize_language_hints(language_hints)


def _normalize_language_hints(language_hints: Mapping[str, str] | None) -> dict[str, str]:
    if language_hints is None:
        return {}
    if not isinstance(language_hints, Mapping):
        raise ValueError("language_hints must be a mapping of relative paths to languages")
    normalized: dict[str, str] = {}
    for relative_path, language in language_hints.items():
        if not isinstance(relative_path, str):
            raise ValueError("language hint paths must be strings")
        path = normalize_posix_relative_path(relative_path)
        normalized_language = validate_language_hint(language)
        if normalized_language is None:
            raise ValueError("language hint is required")
        normalized[path] = normalized_language
    return normalized


def _git_visible_candidates(root: Path) -> list[Path] | None:
    """Return Git-visible paths, or ``None`` when the safe walker is required."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    return sorted(
        (
            root / entry
            for entry in result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
            if entry
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _walk_candidates(root: Path, ignored: set[str]) -> Iterator[Path]:
    """Yield deterministic candidates without traversing past the file cap.

    ``iter_code_files`` applies the cap after validation.  Keeping this walk
    lazy preserves the legacy bound: once the caller has accepted enough
    eligible files, directories that follow in lexical order are never read.
    """

    def walk(current: Path) -> Iterator[Path]:
        _ensure_contained(root, current)
        try:
            with os.scandir(current) as scanner:
                entries = sorted(
                    scanner,
                    key=lambda entry: entry.name
                    + ("/" if entry.is_dir(follow_symlinks=False) else ""),
                )
        except OSError:
            return

        for entry in entries:
            path = current / entry.name
            try:
                is_symlink = entry.is_symlink()
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue

            if is_symlink:
                try:
                    canonical = path.resolve(strict=True)
                except FileNotFoundError:
                    continue
                _ensure_contained(root, canonical)
                if canonical.is_dir():
                    continue
                yield path
                continue

            if is_directory:
                if entry.name in ignored or entry.name.startswith(".git"):
                    continue
                yield from walk(path)
                continue

            yield path

    yield from walk(root)


def _validate_candidate(
    root: Path,
    candidate: Path,
    ignored: set[str],
    language_hints: Mapping[str, str],
) -> tuple[str, Path] | None:
    """Revalidate one Git or walker candidate without exposing its real path."""

    try:
        lexical_relative = candidate.relative_to(root)
        relative_path = normalize_posix_relative_path(lexical_relative)
    except (ValueError, ResourceIdentityError):
        raise ResourceIdentityError("workspace entry escapes declared root") from None

    relative_parts = Path(relative_path).parts
    if any(part in ignored or part.startswith(".git") for part in relative_parts[:-1]):
        return None
    if candidate.suffix.lower() not in SUPPORTED_CODE_EXTENSIONS and relative_path not in language_hints:
        return None

    try:
        canonical = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    _ensure_contained(root, canonical)
    if not canonical.is_file():
        return None
    return relative_path, candidate


def _ensure_contained(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ResourceIdentityError("workspace entry escapes declared root") from exc


__all__ = [
    "BoundedCodeWorkspaceWalkStats",
    "DEFAULT_SKIP_DIRS",
    "ValidatedCodeWorkspaceFile",
    "iter_bounded_code_files",
    "iter_code_files",
    "normalize_language_hints",
    "validate_code_candidate_path",
]
