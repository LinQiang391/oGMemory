"""Atomic exact-string edits inside authenticated registered workspaces."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.models import RequestContext
from core.resource_identity import (
    ResourceIdentityError,
    normalize_posix_relative_path,
    normalize_resource_relative_path,
)
from extraction.code_processor import MAX_CODE_FILE_SIZE
from service.code_workspace_sync import (
    CodeWorkspaceRegistryResolver,
    ResolvedCodeWorkspace,
)

logger = logging.getLogger(__name__)

_EDIT_READ_CHUNK_BYTES = 1024 * 1024


class RefreshCodeFile(Protocol):
    """Canonical one-path refresh callback supplied by ``MemoryService``."""

    def __call__(
        self,
        *,
        workspace: ResolvedCodeWorkspace,
        relative_path: str,
        ctx: RequestContext,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    owner: int
    group: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileSnapshot:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            owner=int(getattr(value, "st_uid", 0)),
            group=int(getattr(value, "st_gid", 0)),
            size=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
        )


class CodeFileEditService:
    """Own exact replacement, atomic publication, and refresh state."""

    def __init__(
        self,
        *,
        workspace_resolver: CodeWorkspaceRegistryResolver,
        code_retrieval_mode: str,
        refresh_code_file: RefreshCodeFile | None = None,
        invalidate_cache: Callable[[str | None], None] | None = None,
    ) -> None:
        if workspace_resolver is None:
            raise RuntimeError("workspace resolver is required for code editing")
        mode = str(code_retrieval_mode or "full").strip().lower()
        if mode not in {"full", "candidate_local"}:
            raise ValueError("code_retrieval_mode is invalid")
        if mode == "full" and refresh_code_file is None:
            raise RuntimeError("full code editing requires the canonical refresh callback")
        self._workspace_resolver = workspace_resolver
        self._code_retrieval_mode = mode
        self._refresh_code_file = refresh_code_file
        self._invalidate_cache = invalidate_cache

    def edit(
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        ctx: RequestContext,
        root_slug: str | None = None,
    ) -> dict[str, Any]:
        """Replace exact text once or everywhere without exposing physical paths."""

        _validate_edit_request(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            root_slug=root_slug,
        )
        requested_roots = (root_slug,) if root_slug is not None else ()
        if self._code_retrieval_mode == "candidate_local":
            workspace = self._workspace_resolver.resolve(
                ctx=ctx,
                root_slugs=requested_roots,
            )
        else:
            workspace = self._workspace_resolver.resolve_registered(
                ctx=ctx,
                root_slugs=requested_roots,
            )

        target, relative_path = _resolve_target(workspace, file_path)
        original, snapshot = _read_utf8_regular_file(target)
        occurrences = original.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string not found in target file")
        if not replace_all and occurrences != 1:
            raise ValueError(
                f"old_string matched {occurrences} times; pass replace_all=true "
                "or provide a more specific old_string"
            )
        result_size = _replacement_size_bytes(
            original_size=snapshot.size,
            old_string=old_string,
            new_string=new_string,
            occurrences=occurrences,
        )
        if result_size > MAX_CODE_FILE_SIZE:
            raise ValueError("edited file exceeds maximum editable size")
        updated = (
            original.replace(old_string, new_string)
            if replace_all
            else original.replace(old_string, new_string, 1)
        )
        edit_state = _atomic_replace_utf8(
            target=target,
            workspace_root=workspace.physical_root,
            snapshot=snapshot,
            content=updated,
        )
        refresh_state = self._refresh_after_edit(
            workspace=workspace,
            relative_path=relative_path,
            ctx=ctx,
        )
        return {
            "ok": True,
            "relative_path": relative_path,
            "occurrences": occurrences,
            "edit_state": edit_state,
            "refresh_state": refresh_state,
        }

    def _refresh_after_edit(
        self,
        *,
        workspace: ResolvedCodeWorkspace,
        relative_path: str,
        ctx: RequestContext,
    ) -> str:
        if self._code_retrieval_mode == "candidate_local":
            self._invalidate_safely(ctx.account_id, relative_path=relative_path)
            return "request_local"

        refresh_state = "failed"
        try:
            refresh = self._refresh_code_file
            if refresh is None:
                raise RuntimeError("canonical refresh callback is unavailable")
            result = refresh(
                workspace=workspace,
                relative_path=relative_path,
                ctx=ctx,
            )
            refresh_state = _safe_refresh_state(result)
        except Exception as exc:
            logger.warning(
                "code file edit refresh failed account=%s relative_path=%s error=%s",
                ctx.account_id,
                relative_path,
                type(exc).__name__,
            )
        if not self._invalidate_safely(ctx.account_id, relative_path=relative_path):
            refresh_state = "failed"
        return refresh_state

    def _invalidate_safely(self, account_id: str, *, relative_path: str) -> bool:
        if self._invalidate_cache is None:
            return True
        try:
            self._invalidate_cache(account_id)
            return True
        except Exception as exc:
            logger.warning(
                "code file edit cache invalidation failed account=%s relative_path=%s error=%s",
                account_id,
                relative_path,
                type(exc).__name__,
            )
            return False


def _validate_edit_request(
    *,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    root_slug: str | None,
) -> None:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path is required")
    if not isinstance(old_string, str) or not old_string:
        raise ValueError("old_string must be a non-empty string")
    if not isinstance(new_string, str):
        raise ValueError("new_string must be a string")
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all must be a boolean")
    if root_slug is not None and (
        not isinstance(root_slug, str) or not root_slug.strip()
    ):
        raise ValueError("root_slug must be a non-empty string when provided")


def _resolve_target(
    workspace: ResolvedCodeWorkspace,
    file_path: str,
) -> tuple[Path, str]:
    root = workspace.physical_root.resolve(strict=True)
    try:
        requested = Path(file_path).expanduser()
        if requested.is_absolute():
            candidate = requested
        else:
            candidate = root / normalize_posix_relative_path(file_path)
        target = candidate.resolve(strict=True)
        target.relative_to(root)
        relative_path = normalize_resource_relative_path(root, target)
    except FileNotFoundError as exc:
        raise FileNotFoundError("target file was not found") from exc
    except (OSError, RuntimeError, ResourceIdentityError, ValueError) as exc:
        raise ValueError(
            "file_path must identify a contained file in the registered workspace"
        ) from exc
    return target, relative_path


def _read_utf8_regular_file(target: Path) -> tuple[str, _FileSnapshot]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError("target file was not found") from exc
    except OSError as exc:
        raise ValueError("target file could not be opened safely") from exc
    try:
        before = _FileSnapshot.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode):
            raise ValueError("target must be a regular file")
        if before.size > MAX_CODE_FILE_SIZE:
            raise ValueError("target file exceeds maximum editable size")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_EDIT_READ_CHUNK_BYTES, MAX_CODE_FILE_SIZE + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_CODE_FILE_SIZE:
                raise ValueError("target file exceeds maximum editable size")
        after = _FileSnapshot.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if after != before:
        raise RuntimeError("target changed during edit; retry with fresh code search")
    try:
        return b"".join(chunks).decode("utf-8"), before
    except UnicodeDecodeError as exc:
        raise ValueError("target file must be UTF-8 text") from exc


def _atomic_replace_utf8(
    *,
    target: Path,
    workspace_root: Path,
    snapshot: _FileSnapshot,
    content: str,
) -> str:
    parent = target.parent
    descriptor = -1
    temp_path: Path | None = None
    replaced = False
    try:
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{target.name}.ogm-edit-",
            dir=str(parent),
        )
        temp_path = Path(raw_temp_path)
        os.fchmod(descriptor, stat.S_IMODE(snapshot.mode))
        payload = content.encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            if len(payload) > MAX_CODE_FILE_SIZE:
                raise ValueError("edited file exceeds maximum editable size")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _revalidate_target(
            target=target,
            workspace_root=workspace_root,
            snapshot=snapshot,
        )
        os.replace(temp_path, target)
        replaced = True
        try:
            _fsync_directory(parent)
        except OSError as exc:
            logger.warning(
                "code file edit directory fsync failed error=%s",
                type(exc).__name__,
            )
            return "modified_unconfirmed"
        return "modified"
    except (ValueError, RuntimeError):
        raise
    except OSError as exc:
        if replaced:
            raise RuntimeError("code file edit publication could not be confirmed") from exc
        raise RuntimeError("code file write failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temp_path is not None and not replaced:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("code file edit temporary cleanup failed")


def _replacement_size_bytes(
    *,
    original_size: int,
    old_string: str,
    new_string: str,
    occurrences: int,
) -> int:
    """Calculate the UTF-8 size of a replacement without building its result."""

    try:
        old_size = len(old_string.encode("utf-8"))
        new_size = len(new_string.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("edit strings must be valid UTF-8") from exc
    return original_size - occurrences * old_size + occurrences * new_size


def _revalidate_target(
    *,
    target: Path,
    workspace_root: Path,
    snapshot: _FileSnapshot,
) -> None:
    try:
        root = workspace_root.resolve(strict=True)
        current_target = target.resolve(strict=True)
        current_target.relative_to(root)
        if current_target != target:
            raise RuntimeError("target changed during edit; retry with fresh code search")
        current = _FileSnapshot.from_stat(os.stat(target, follow_symlinks=False))
    except FileNotFoundError as exc:
        raise RuntimeError("target changed during edit; retry with fresh code search") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("target changed"):
            raise
        raise RuntimeError("target changed during edit; retry with fresh code search") from exc
    if current != snapshot or not stat.S_ISREG(current.mode):
        raise RuntimeError("target changed during edit; retry with fresh code search")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_refresh_state(result: Mapping[str, Any]) -> str:
    if not isinstance(result, Mapping):
        raise RuntimeError("canonical code refresh returned an invalid result")
    if result.get("index_visible") is True:
        return "visible"
    status = str(result.get("status") or "").strip().lower()
    if status in {"visible", "index_visible", "complete", "completed"}:
        return "visible"
    return "accepted"


__all__ = ["CodeFileEditService"]
