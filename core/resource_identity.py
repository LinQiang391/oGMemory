"""Shared identity and metadata helpers for account-scoped resources."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from core.models import ContextNode

MIN_DIGEST_SLUG_HEX = 32

# Only scalar fields needed by code retrieval are allowed into IndexRecord
# metadata.  In particular, ``graph`` remains in ContextFS for graph-route
# hydration and is deliberately absent here.
CODE_INDEX_METADATA_FIELDS: tuple[str, ...] = (
    "resource_type",
    "resource_root_slug",
    "root_fingerprint",
    "file_key",
    "file_slug",
    "relative_path",
    "source_extension",
    "language",
    "start_line",
    "end_line",
    "symbol",
    "symbol_kind",
    "signature",
    "chunk_hash",
    "chunk_key",
)


class ResourceIdentityError(ValueError):
    """Base error for deterministic resource identity failures."""


class ResourceCollisionError(ResourceIdentityError):
    """A shortened URI identity resolved to a different full key."""


class RootBindingError(ResourceIdentityError):
    """A root slug/fingerprint pair violates the account-wide bijection."""


class AmbiguousRootBindingError(RootBindingError):
    """Historical metadata maps one fingerprint to multiple root slugs."""


_T = TypeVar("_T")


def _sha256_parts(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resource_owner_space(account_id: str) -> str:
    """Return the account-scoped owner marker used by RESOURCE nodes."""
    if not account_id:
        raise ResourceIdentityError("account_id must not be empty")
    return f"resource:{account_id}"


def compute_root_fingerprint(root: str | os.PathLike[str]) -> str:
    """Hash the canonical real path of a resource root."""
    canonical = os.path.realpath(os.fspath(root))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_resource_relative_path(
    root: str | os.PathLike[str],
    file_path: str | os.PathLike[str],
) -> str:
    """Return a canonical POSIX path under ``root``, preserving its suffix.

    ``Path.resolve`` follows existing symlinks, so a symlink that escapes the
    declared root is rejected.  ``strict=False`` also supports refresh of a
    file that has just been deleted.
    """
    canonical_root = Path(root).expanduser().resolve(strict=False)
    candidate = Path(file_path).expanduser()
    if not candidate.is_absolute():
        candidate = canonical_root / candidate
    canonical_path = candidate.resolve(strict=False)
    try:
        relative = canonical_path.relative_to(canonical_root)
    except ValueError as exc:
        raise ResourceIdentityError("path escapes declared resource root") from exc
    if relative == Path("."):
        raise ResourceIdentityError("resource file path must not be the root directory")
    return relative.as_posix()


def normalize_posix_relative_path(relative_path: str | os.PathLike[str]) -> str:
    """Validate and normalize an already-relative path without dropping suffixes."""
    raw = os.fspath(relative_path).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or raw == "." or ".." in path.parts:
        raise ResourceIdentityError("relative_path must be a non-empty contained path")
    return path.as_posix()


def compute_file_key(
    root_fingerprint: str,
    root_slug: str,
    relative_path: str | os.PathLike[str],
) -> str:
    """Build the full file identity required by the migration contract."""
    if not root_fingerprint or not root_slug:
        raise ResourceIdentityError("root_fingerprint and root_slug must not be empty")
    normalized = normalize_posix_relative_path(relative_path)
    return _sha256_parts(root_fingerprint, root_slug, normalized)


def compute_chunk_key(
    file_key: str,
    *,
    start_line: int,
    end_line: int,
    symbol: str,
    chunk_hash: str,
) -> str:
    """Build a full chunk identity from the full file key and chunk locators."""
    if not file_key or not chunk_hash:
        raise ResourceIdentityError("file_key and chunk_hash must not be empty")
    if start_line < 1 or end_line < start_line:
        raise ResourceIdentityError("invalid chunk line range")
    return _sha256_parts(file_key, start_line, end_line, symbol, chunk_hash)


def digest_slug(full_key: str, *, length: int = MIN_DIGEST_SLUG_HEX) -> str:
    """Return a shortened hexadecimal slug with at least 128 bits of digest."""
    if length < MIN_DIGEST_SLUG_HEX or length > 64:
        raise ResourceIdentityError("digest slug length must be between 32 and 64 hex chars")
    normalized = full_key.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        normalized = hashlib.sha256(full_key.encode("utf-8")).hexdigest()
    return normalized[:length]


def guard_full_key_collision(
    existing_metadata: Mapping[str, Any],
    *,
    key_field: str,
    expected_key: str,
    uri: str = "",
) -> None:
    """Fail closed unless an existing shortened URI carries the same full key."""
    actual = existing_metadata.get(key_field)
    if actual == expected_key:
        return
    uri_ref = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12] if uri else "unknown"
    if actual is None:
        reason = f"existing resource lacks {key_field}"
    else:
        reason = f"existing {key_field} differs from requested identity"
    raise ResourceCollisionError(f"resource identity collision ({reason}; uri_ref={uri_ref})")


def _metadata(item: ContextNode | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(item, ContextNode):
        return item.metadata
    nested = item.get("metadata")
    if isinstance(nested, Mapping):
        return nested
    return item


def _binding_pairs(
    items: Iterable[ContextNode | Mapping[str, Any]],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in items:
        metadata = _metadata(item)
        slug = metadata.get("resource_root_slug")
        fingerprint = metadata.get("root_fingerprint")
        if isinstance(slug, str) and slug and isinstance(fingerprint, str) and fingerprint:
            pairs.add((slug, fingerprint))
    return pairs


def validate_root_binding(
    items: Iterable[ContextNode | Mapping[str, Any]],
    *,
    root_slug: str,
    root_fingerprint: str,
) -> None:
    """Enforce the account-wide slug/fingerprint bijection for a new binding."""
    pairs = _binding_pairs(items)
    fingerprints_for_slug = {fingerprint for slug, fingerprint in pairs if slug == root_slug}
    slugs_for_fingerprint = {slug for slug, fingerprint in pairs if fingerprint == root_fingerprint}
    if len(slugs_for_fingerprint) > 1:
        raise AmbiguousRootBindingError("root fingerprint is bound to multiple resource root slugs")
    if fingerprints_for_slug and fingerprints_for_slug != {root_fingerprint}:
        raise RootBindingError("resource root slug is already bound to another fingerprint")
    if slugs_for_fingerprint and slugs_for_fingerprint != {root_slug}:
        raise RootBindingError("root fingerprint is already bound to another resource root slug")


def resolve_root_slug(
    items: Iterable[ContextNode | Mapping[str, Any]],
    *,
    root_fingerprint: str,
) -> str | None:
    """Recover the unique root slug for refresh when the caller omitted a name."""
    slugs = {slug for slug, fingerprint in _binding_pairs(items) if fingerprint == root_fingerprint}
    if len(slugs) > 1:
        raise AmbiguousRootBindingError("root fingerprint is bound to multiple resource root slugs")
    return next(iter(slugs), None)


def metadata_matches_exact(metadata: Mapping[str, Any], **criteria: Any) -> bool:
    """Return true only for type-preserving, exact metadata equality."""
    for key, expected in criteria.items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if type(actual) is not type(expected) or actual != expected:
            return False
    return True


def select_nodes_by_metadata(
    items: Iterable[_T],
    **criteria: Any,
) -> list[_T]:
    """Select ContextNodes or metadata-bearing mappings with exact equality."""
    selected: list[_T] = []
    for item in items:
        if isinstance(item, (ContextNode, Mapping)) and metadata_matches_exact(
            _metadata(item), **criteria
        ):
            selected.append(item)
    return selected


def canonical_node_uri(uri: str, *, parent_uri: str | None = None) -> str:
    """Normalize an L2 ``/content.md`` hit to its owning node URI.

    A parent URI is trusted only when it is the exact structural parent of the
    supplied L2 URI.  Without that evidence, only the precise ``/content.md``
    suffix is removed; similarly named suffixes are left untouched.
    """
    normalized = uri.rstrip("/")
    if parent_uri:
        parent = parent_uri.rstrip("/")
        if normalized == f"{parent}/content.md":
            return parent
    suffix = "/content.md"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized


def redact_path(
    path: str | os.PathLike[str],
    *,
    declared_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return a feature-local safe path reference for public errors and logs."""
    canonical = Path(path).expanduser().resolve(strict=False)
    if declared_root is not None:
        root = Path(declared_root).expanduser().resolve(strict=False)
        try:
            return canonical.relative_to(root).as_posix()
        except ValueError:
            pass
    digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:12]
    return f"<path:{digest}>"


def code_index_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Copy the explicit scalar code metadata allowlist for vector indexing."""
    result: dict[str, str | int | float | bool] = {}
    for key in CODE_INDEX_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result
