"""Filesystem artifact store with session-scoped plain-text paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

from core.models import RequestContext

from .models import ArtifactMetadata
from .store_base import (
    ArtifactNotVisibleError,
    ToolResultArtifactStore,
    artifact_reuse_matches,
    decode_utf8_range,
    metadata_visible,
    require_safe_path_component,
    require_session_owner,
)

_ARTIFACT_ID = re.compile(r"^tr_[a-f0-9]{16,64}(?:_[a-f0-9]{8,16})?$")


class LocalToolResultArtifactStore(ToolResultArtifactStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _base(self, session_id: str, ctx: RequestContext) -> Path:
        require_session_owner(session_id, ctx)
        account_id = require_safe_path_component(ctx.account_id, field="account_id")
        user_id = require_safe_path_component(ctx.user_id, field="user_id")
        agent_id = require_safe_path_component(ctx.agent_id, field="agent_id")
        safe_session_id = require_safe_path_component(session_id, field="session_id")
        return (
            self.root
            / "accounts"
            / account_id
            / "user"
            / user_id
            / "agents"
            / agent_id
            / "sessions"
            / safe_session_id
            / "tool-results"
        )

    def _dir(self, artifact_id: str, session_id: str, ctx: RequestContext) -> Path:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return self._base(session_id, ctx) / artifact_id

    @staticmethod
    def _load(path: Path) -> ArtifactMetadata:
        try:
            return ArtifactMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc

    def write(self, metadata: ArtifactMetadata, output_text: str, ctx: RequestContext) -> bool:
        directory = self._dir(metadata.artifact_id, metadata.session_id, ctx)
        if not metadata_visible(metadata, metadata.session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        encoded = output_text.encode("utf-8")
        if len(encoded) != metadata.original_bytes or hashlib.sha256(encoded).hexdigest() != metadata.sha256:
            raise ValueError("artifact output does not match metadata")

        with self._lock:
            if (directory / "metadata.json").is_file():
                existing, existing_output = self.read_all(
                    metadata.artifact_id, metadata.session_id, ctx,
                )
                if artifact_reuse_matches(
                    existing, metadata, existing_output, output_text,
                ):
                    return True
                raise FileExistsError("artifact identity conflict")

            directory.parent.mkdir(parents=True, exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix=f".{metadata.artifact_id}.", dir=directory.parent))
            try:
                (temp / "_writing.json").write_text(
                    json.dumps({"artifact_id": metadata.artifact_id, "started_at": metadata.created_at}),
                    encoding="utf-8",
                )
                output_path = temp / "output.txt"
                output_path.write_bytes(encoded)
                with output_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                verified = output_path.read_bytes()
                if hashlib.sha256(verified).hexdigest() != metadata.sha256:
                    raise IOError("artifact verification failed")
                (temp / "metadata.json").write_text(
                    json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                try:
                    os.replace(temp, directory)
                except OSError as publish_error:
                    # Another process may have won the immutable directory
                    # publication race. Never overwrite it; verify whether it
                    # is the same logical retry before surfacing the error.
                    try:
                        existing, existing_output = self.read_all(
                            metadata.artifact_id, metadata.session_id, ctx,
                        )
                    except ArtifactNotVisibleError:
                        raise publish_error
                    if artifact_reuse_matches(
                        existing, metadata, existing_output, output_text,
                    ):
                        return True
                    raise FileExistsError(
                        "artifact identity conflict"
                    ) from publish_error
                try:
                    (directory / "_writing.json").unlink()
                except OSError:
                    pass
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        return False

    def read_all(self, artifact_id: str, session_id: str, ctx: RequestContext) -> tuple[ArtifactMetadata, str]:
        directory = self._dir(artifact_id, session_id, ctx)
        metadata = self.read_metadata(artifact_id, session_id, ctx)
        try:
            encoded = (directory / "output.txt").read_bytes()
            output = encoded.decode("utf-8")
        except Exception as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        if len(encoded) != metadata.original_bytes or hashlib.sha256(encoded).hexdigest() != metadata.sha256:
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata, output

    def read_metadata(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
    ) -> ArtifactMetadata:
        directory = self._dir(artifact_id, session_id, ctx)
        metadata = self._load(directory / "metadata.json")
        if not metadata_visible(metadata, session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata

    def read_range(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
        offset_bytes: int,
        length_bytes: int,
    ) -> dict:
        if offset_bytes < 0 or length_bytes < 1:
            raise ValueError(
                "offset_bytes must be non-negative and length_bytes must be positive"
            )
        directory = self._dir(artifact_id, session_id, ctx)
        metadata = self.read_metadata(artifact_id, session_id, ctx)
        base_offset = min(offset_bytes, metadata.original_bytes)
        try:
            with (directory / "output.txt").open("rb") as handle:
                handle.seek(base_offset)
                raw = handle.read(length_bytes + 6)
            decoded = decode_utf8_range(raw, 0, length_bytes)
        except (OSError, UnicodeDecodeError) as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        decoded["actual_offset_bytes"] += base_offset
        decoded["next_offset_bytes"] += base_offset
        decoded["eof"] = decoded["next_offset_bytes"] >= metadata.original_bytes
        return {"metadata": metadata.to_dict(), "ref": metadata.ref, **decoded}

    def list(self, session_id: str, ctx: RequestContext, limit: int = 50, cursor: str | None = None) -> dict:
        require_session_owner(session_id, ctx)
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        base = self._base(session_id, ctx)
        names = sorted(
            (
                entry.name
                for entry in base.iterdir()
                if entry.is_dir() and _ARTIFACT_ID.fullmatch(entry.name)
            ),
            reverse=True,
        ) if base.is_dir() else []
        start = 0
        if cursor:
            try:
                start = names.index(cursor) + 1
            except ValueError:
                raise ValueError("invalid cursor") from None
        artifacts = []
        scanned = start
        last_scanned = None
        for name in names[start:]:
            scanned += 1
            last_scanned = name
            try:
                metadata = self.read_metadata(name, session_id, ctx)
                artifacts.append(metadata.to_dict() | {"ref": metadata.ref})
            except ArtifactNotVisibleError:
                continue
            if len(artifacts) >= limit:
                break
        truncated = scanned < len(names)
        return {
            "artifacts": artifacts,
            "next_cursor": last_scanned if truncated else None,
            "truncated": truncated,
        }

    def delete_session_artifacts(self, session_id: str, ctx: RequestContext) -> int:
        require_session_owner(session_id, ctx)
        base = self._base(session_id, ctx)
        if not base.is_dir():
            return 0
        with self._lock:
            artifact_ids = [
                entry.name
                for entry in base.iterdir()
                if entry.is_dir() and _ARTIFACT_ID.fullmatch(entry.name)
            ]
            shutil.rmtree(base)
            return len(artifact_ids)
