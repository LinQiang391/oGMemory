"""Shared identity and governance rules for generated Agent Skills."""

from __future__ import annotations

import hashlib
import re
from typing import Any

MAX_SKILL_NAME_LENGTH = 64
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def normalize_skill_name(value: Any) -> str:
    """Return the canonical, bounded kebab-case artifact name."""

    normalized = (
        re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(value).lower()))
        .strip("-")[:MAX_SKILL_NAME_LENGTH]
        .rstrip("-")
    )
    # Truncation can otherwise leave a trailing separator, producing a value
    # that normalize_skill_name itself cannot pass to validate_skill_name.
    if normalized:
        validate_skill_name(normalized)
    return normalized


def validate_skill_name(name: str) -> str:
    """Validate a public skill identifier and return it unchanged."""

    if len(name) > MAX_SKILL_NAME_LENGTH or not _SKILL_NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    return name


def internal_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    return (
        f"ctx://{account_id}/agents/{agent_id}/skills/generated/"
        f"{validate_skill_name(name)}"
    )


def legacy_internal_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    return f"ctx://{account_id}/agents/{agent_id}/skills/{validate_skill_name(name)}"


def internal_managed_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    from skills.management import validate_managed_skill_name

    return (
        f"ctx://{account_id}/agents/{agent_id}/skills/imported/"
        f"{validate_managed_skill_name(name)}"
    )


def canonical_skill_uri(name: str, source: str | None = None) -> str:
    if source not in {None, "generated", "imported"}:
        raise ValueError("source must be generated or imported")
    if source == "imported":
        from skills.management import validate_managed_skill_name

        validate_managed_skill_name(name)
        directory = source
    else:
        validate_skill_name(name)
        directory = "generated"
    return f"ogmemory://skills/{directory}/{name}"


def is_governed_skill(node: Any) -> bool:
    """Return whether a node belongs to the reviewed generated catalog."""

    metadata = getattr(node, "metadata", {}) or {}
    # Publication is an immutable event. A later configuration change must not
    # make an already-reviewed artifact disappear from MCP/CLI discovery.
    return (
        metadata.get("ogmemory_generated") is True
        and metadata.get("publication_status") == "published"
    )


def is_managed_skill(node: Any) -> bool:
    """Return whether a skill was explicitly installed through management APIs."""

    metadata = getattr(node, "metadata", {}) or {}
    return (
        metadata.get("ogmemory_managed") is True
        and metadata.get("publication_status") == "published"
    )


def is_published_skill(node: Any) -> bool:
    return is_governed_skill(node) or is_managed_skill(node)


def artifact_body_hash(content: str) -> str:
    """Hash the complete portable artifact; governance lives outside SKILL.md."""

    normalized = "\n".join(line.rstrip() for line in content.strip().splitlines()) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
