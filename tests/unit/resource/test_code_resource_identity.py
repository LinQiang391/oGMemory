from __future__ import annotations

import hashlib
import os

import pytest

from core.models import ContextNode
from core.resource_identity import (
    AmbiguousRootBindingError,
    ResourceCollisionError,
    ResourceIdentityError,
    RootBindingError,
    canonical_node_uri,
    compute_chunk_key,
    compute_file_key,
    compute_root_fingerprint,
    compute_workspace_root_id,
    digest_slug,
    guard_full_key_collision,
    normalize_resource_relative_path,
    redact_path,
    reserve_workspace_root_binding_state,
    resolve_root_slug,
    resolve_workspace_root_binding_state,
    resource_owner_space,
    select_nodes_by_metadata,
    validate_resource_uri_slug,
    validate_root_binding,
)


def _node(**metadata) -> ContextNode:
    return ContextNode(
        uri="ctx://acct/resources/abc",
        context_type="RESOURCE",
        category="code",
        level=0,
        owner_space="resource:acct",
        abstract="a",
        overview="o",
        content="c",
        metadata=metadata,
    )


def test_root_fingerprint_hashes_realpath(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    expected = hashlib.sha256(os.path.realpath(root).encode("utf-8")).hexdigest()

    assert compute_root_fingerprint(root) == expected
    assert len(expected) == 64


def test_relative_path_preserves_extension_and_rejects_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    source = root / "pkg" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n", encoding="utf-8")

    assert normalize_resource_relative_path(root, source) == "pkg/service.py"

    outside = tmp_path / "outside.ts"
    outside.write_text("export {}\n", encoding="utf-8")
    escaped = root / "escaped.ts"
    escaped.symlink_to(outside)
    with pytest.raises(ResourceIdentityError, match="escapes"):
        normalize_resource_relative_path(root, escaped)


def test_full_file_and_chunk_keys_keep_collision_samples_distinct(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    fingerprint = compute_root_fingerprint(root)

    py_key = compute_file_key(fingerprint, "project", "pkg/foo.py")
    ts_key = compute_file_key(fingerprint, "project", "pkg/foo.ts")
    other_dir_key = compute_file_key(fingerprint, "project", "other/foo.py")

    assert len({py_key, ts_key, other_dir_key}) == 3
    assert digest_slug(py_key) == py_key[:32]
    assert len(digest_slug(py_key)) == 32

    first = compute_chunk_key(
        py_key,
        start_line=1,
        end_line=3,
        symbol="parse",
        chunk_hash="a" * 64,
    )
    duplicate_symbol = compute_chunk_key(
        py_key,
        start_line=10,
        end_line=12,
        symbol="parse",
        chunk_hash="b" * 64,
    )
    assert first != duplicate_symbol


def test_digest_slug_rejects_less_than_128_bits():
    with pytest.raises(ResourceIdentityError, match="between 32 and 64"):
        digest_slug("a" * 64, length=31)


@pytest.mark.parametrize("slug", ["", ".", "..", "%2F", r"foo\bar", "foo+bar"])
def test_resource_uri_slug_rejects_empty_encoded_and_unsafe_segments(slug):
    with pytest.raises(ResourceIdentityError, match="one safe path segment"):
        validate_resource_uri_slug(slug)


def test_collision_guard_requires_matching_full_key():
    guard_full_key_collision(
        {"chunk_key": "full-key"},
        key_field="chunk_key",
        expected_key="full-key",
        uri="ctx://acct/resources/abc",
    )

    with pytest.raises(ResourceCollisionError, match="differs"):
        guard_full_key_collision(
            {"chunk_key": "other-key"},
            key_field="chunk_key",
            expected_key="full-key",
            uri="ctx://acct/resources/abc",
        )
    with pytest.raises(ResourceCollisionError, match="lacks"):
        guard_full_key_collision(
            {},
            key_field="chunk_key",
            expected_key="full-key",
            uri="ctx://acct/resources/abc",
        )


def test_root_binding_is_bijective_and_refresh_resolution_is_unambiguous():
    existing = [
        _node(resource_root_slug="alpha", root_fingerprint="fp-alpha"),
        _node(resource_root_slug="alpha", root_fingerprint="fp-alpha"),
    ]
    validate_root_binding(
        existing,
        root_slug="alpha",
        root_fingerprint="fp-alpha",
    )
    assert resolve_root_slug(existing, root_fingerprint="fp-alpha") == "alpha"
    assert resolve_root_slug(existing, root_fingerprint="missing") is None

    with pytest.raises(RootBindingError, match="slug is already bound"):
        validate_root_binding(
            existing,
            root_slug="alpha",
            root_fingerprint="fp-other",
        )
    with pytest.raises(RootBindingError, match="fingerprint is already bound"):
        validate_root_binding(
            existing,
            root_slug="beta",
            root_fingerprint="fp-alpha",
        )

    ambiguous = [
        _node(resource_root_slug="alpha", root_fingerprint="fp-alpha"),
        _node(resource_root_slug="beta", root_fingerprint="fp-alpha"),
    ]
    with pytest.raises(AmbiguousRootBindingError, match="multiple"):
        resolve_root_slug(ambiguous, root_fingerprint="fp-alpha")


def test_protected_workspace_binding_state_matches_control_plane_shape():
    fingerprint = "a" * 64
    root_id = compute_workspace_root_id("acct", fingerprint)

    state = reserve_workspace_root_binding_state(
        None,
        account_id="acct",
        root_slug="custom-root",
        root_fingerprint=fingerprint,
    )

    assert state == {
        "version": 1,
        "by_fingerprint": {
            fingerprint: {
                "root_id": root_id,
                "resource_root_slug": "custom-root",
            }
        },
        "by_slug": {
            "custom-root": {
                "root_id": root_id,
                "root_fingerprint": fingerprint,
            }
        },
    }
    assert resolve_workspace_root_binding_state(
        state,
        account_id="acct",
        root_slug="custom-root",
    ) == (root_id, "custom-root", fingerprint)

    with pytest.raises(RootBindingError, match="different root slug"):
        reserve_workspace_root_binding_state(
            state,
            account_id="acct",
            root_slug="other-root",
            root_fingerprint=fingerprint,
        )


def test_exact_metadata_selection_does_not_coerce_or_prefix_match():
    nodes = [
        _node(file_key="abc", start_line=1),
        _node(file_key="abcd", start_line=1),
        _node(file_key="abc", start_line=True),
    ]

    assert select_nodes_by_metadata(nodes, file_key="abc", start_line=1) == [nodes[0]]


@pytest.mark.parametrize(
    ("uri", "parent_uri", "expected"),
    [
        (
            "ctx://acct/resources/chunk/content.md",
            "ctx://acct/resources/chunk",
            "ctx://acct/resources/chunk",
        ),
        (
            "ctx://acct/resources/chunk/content.md",
            "ctx://acct/resources/other",
            "ctx://acct/resources/chunk",
        ),
        (
            "ctx://acct/resources/chunk/content.md.bak",
            None,
            "ctx://acct/resources/chunk/content.md.bak",
        ),
        (
            "ctx://acct/resources/chunk",
            "ctx://acct/resources",
            "ctx://acct/resources/chunk",
        ),
    ],
)
def test_canonical_node_uri_only_removes_exact_l2_suffix(uri, parent_uri, expected):
    assert canonical_node_uri(uri, parent_uri=parent_uri) == expected


def test_path_redaction_returns_relative_path_or_digest(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    inside = root / "src" / "service.py"
    outside = tmp_path / "secret" / "token.txt"

    assert redact_path(inside, declared_root=root) == "src/service.py"
    redacted = redact_path(outside, declared_root=root)
    assert redacted.startswith("<path:")
    assert str(tmp_path) not in redacted
    assert "secret" not in redacted
    assert resource_owner_space("acct") == "resource:acct"
