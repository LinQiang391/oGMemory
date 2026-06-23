"""oGMemory MCP resource URI parsing and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlparse

from core.models import ContextNode, RequestContext
from server.mcp_security import UNTRUSTED_DATA_NOTICE, raise_not_found_or_not_visible


@dataclass(frozen=True)
class MemoryTypeSpec:
    public_type: str
    internal_category: str
    directory: str
    owner: str


MEMORY_TYPES: dict[str, MemoryTypeSpec] = {
    "profile": MemoryTypeSpec("profile", "profile", "profile", "user"),
    "preference": MemoryTypeSpec("preference", "preference", "preferences", "user"),
    "entity": MemoryTypeSpec("entity", "entity", "entities", "user"),
    "event": MemoryTypeSpec("event", "event", "events", "user"),
    "case": MemoryTypeSpec("case", "case", "cases", "user"),
    "pattern": MemoryTypeSpec("pattern", "pattern", "patterns", "user"),
    "skill": MemoryTypeSpec("skill", "skill", "skills", "agent"),
    "tool_memory": MemoryTypeSpec("tool_memory", "tool", "tools", "agent"),
}

_INTERNAL_TO_PUBLIC = {
    spec.internal_category: spec.public_type for spec in MEMORY_TYPES.values()
}
_DIRECTORY_TO_PUBLIC = {
    spec.directory: spec.public_type for spec in MEMORY_TYPES.values()
}


@dataclass(frozen=True)
class ParsedOgMemoryUri:
    kind: str
    memory_type: str | None = None
    memory_id: str | None = None


def parse_ogmemory_uri(uri: str) -> ParsedOgMemoryUri:
    _validate_public_uri_text(uri)
    parsed = urlparse(uri)
    if parsed.scheme != "ogmemory":
        raise ValueError("Only ogmemory:// URIs are accepted")

    parts = [parsed.netloc, *[p for p in parsed.path.split("/") if p]]
    if parts == ["status"]:
        return ParsedOgMemoryUri(kind="status")
    if parts == ["types"]:
        return ParsedOgMemoryUri(kind="types")
    if len(parts) in (2, 3) and parts[0] == "memories":
        memory_type = parts[1]
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory type: {memory_type}")
        if len(parts) == 2:
            return ParsedOgMemoryUri(kind="collection", memory_type=memory_type)
        memory_id = unquote(parts[2])
        if not memory_id:
            raise ValueError("Memory id must not be empty")
        return ParsedOgMemoryUri(kind="item", memory_type=memory_type, memory_id=memory_id)
    raise ValueError(f"Unsupported ogmemory URI: {uri}")


def external_to_internal_uri(uri: str, ctx: RequestContext) -> str:
    parsed = parse_ogmemory_uri(uri)
    if parsed.kind != "item" or not parsed.memory_type or not parsed.memory_id:
        raise ValueError(f"URI does not reference a single memory: {uri}")
    spec = MEMORY_TYPES[parsed.memory_type]
    memory_id = quote(parsed.memory_id, safe="")

    if parsed.memory_type == "profile":
        if parsed.memory_id != "self":
            raise ValueError("profile memory id must be 'self'")
        return f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"

    if parsed.memory_type == "skill":
        if not ctx.agent_id:
            raise ValueError("agent_id is required for skill memories")
        return f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/{memory_id}"

    if spec.owner == "agent":
        if not ctx.agent_id:
            raise ValueError(f"agent_id is required for {parsed.memory_type} memories")
        return f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/{spec.directory}/{memory_id}"

    return f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/{spec.directory}/{memory_id}"


def internal_to_external_uri(uri: str, *, category: str = "") -> str:
    clean_uri = uri.removesuffix("/content.md").rstrip("/")
    if not clean_uri.startswith("ctx://"):
        raise ValueError("Internal URI must use ctx://")

    parts = clean_uri.split("/")
    if len(parts) < 6:
        raise ValueError(f"Unsupported internal URI: {uri}")

    if "/skills/" in clean_uri:
        memory_id = unquote(parts[-1])
        return f"ogmemory://memories/skill/{quote(memory_id, safe='')}"

    if "/memories/profile" in clean_uri:
        return "ogmemory://memories/profile/self"

    if "/memories/" not in clean_uri:
        raise ValueError(f"Unsupported internal URI: {uri}")

    memory_index = parts.index("memories")
    if len(parts) <= memory_index + 2:
        raise ValueError(f"Internal URI does not reference a memory item: {uri}")
    directory = parts[memory_index + 1]
    memory_id = unquote(parts[memory_index + 2])
    public_type = _DIRECTORY_TO_PUBLIC.get(directory)
    if not public_type and category:
        public_type = _INTERNAL_TO_PUBLIC.get(category, category)
    if not public_type:
        raise ValueError(f"Unknown internal memory directory: {directory}")
    return f"ogmemory://memories/{public_type}/{quote(memory_id, safe='')}"


def external_id_from_uri(uri: str) -> str:
    parsed = parse_ogmemory_uri(uri)
    return parsed.memory_id or ""


def public_type_from_internal(category: str, uri: str = "") -> str:
    if category in _INTERNAL_TO_PUBLIC:
        return _INTERNAL_TO_PUBLIC[category]
    if uri:
        try:
            parsed = parse_ogmemory_uri(internal_to_external_uri(uri, category=category))
            return parsed.memory_type or category
        except Exception:
            pass
    return category


def render_status_resource(service: Any) -> dict[str, Any]:
    cfg = service._cfg
    write_enabled = bool(getattr(cfg, "mcp_enable_write_tools", False))
    delete_enabled = bool(getattr(cfg, "mcp_enable_delete_tools", False))
    return {
        "status": "ok",
        "backend": "og-memory",
        "transport": "mcp",
        "storage_backend": cfg.storage_backend,
        "vector_db_type": cfg.vector_db_type,
        "toolsets": {
            "read_only": True,
            "write": write_enabled,
            "delete": delete_enabled,
            "runtime": False,
            "diagnostic": False,
        },
    }


def render_types_resource(service: Any) -> dict[str, Any]:
    cfg = service._cfg
    write_enabled = bool(getattr(cfg, "mcp_enable_write_tools", False))
    delete_enabled = bool(getattr(cfg, "mcp_enable_delete_tools", False))
    return {
        "types": [
            {
                "type": spec.public_type,
                "owner": spec.owner,
                "writable_type": spec.public_type not in {"profile"},
                "write_tool_enabled": write_enabled,
                "deletable_type": spec.public_type not in {"profile"},
                "delete_tool_enabled": delete_enabled,
            }
            for spec in MEMORY_TYPES.values()
        ]
    }


def render_resource(service: Any, uri: str, ctx: RequestContext) -> dict[str, Any]:
    parsed = parse_ogmemory_uri(uri)
    if parsed.kind == "status":
        return render_status_resource(service)
    if parsed.kind == "types":
        return render_types_resource(service)
    if parsed.kind == "collection" and parsed.memory_type:
        return render_collection_resource(service, parsed.memory_type, ctx)
    if parsed.kind == "item":
        return render_item_resource(service, uri, ctx)
    raise ValueError(f"Unsupported resource URI: {uri}")


def render_collection_resource(service: Any, memory_type: str, ctx: RequestContext) -> dict[str, Any]:
    spec = MEMORY_TYPES[memory_type]
    limit = int(getattr(service._cfg, "mcp_collection_limit", 50))
    memories: list[dict[str, Any]] = []

    try:
        context_fs = service._get_context_fs()
        if memory_type == "profile":
            internal_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            if context_fs.exists(internal_uri, ctx):
                memories.append(_node_summary(context_fs.read_node(internal_uri, ctx)))
        elif memory_type == "skill":
            if ctx.agent_id:
                base_uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills"
                for child_uri in context_fs.list_children(base_uri, ctx)[:limit]:
                    memories.append(_node_summary(context_fs.read_node(child_uri, ctx)))
        else:
            owner_path = "agents" if spec.owner == "agent" else "users"
            owner_id = ctx.agent_id if spec.owner == "agent" else ctx.user_id
            if owner_id:
                base_uri = f"ctx://{ctx.account_id}/{owner_path}/{owner_id}/memories/{spec.directory}"
                for child_uri in context_fs.list_children(base_uri, ctx)[:limit]:
                    memories.append(_node_summary(context_fs.read_node(child_uri, ctx)))
    except Exception as exc:
        raise_not_found_or_not_visible(exc)

    return {
        "uri": f"ogmemory://memories/{memory_type}",
        "type": memory_type,
        "memories": memories,
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }


def render_item_resource(service: Any, uri: str, ctx: RequestContext) -> dict[str, Any]:
    try:
        internal_uri = external_to_internal_uri(uri, ctx)
        node = service._get_context_fs().read_node(internal_uri, ctx)
        return _node_detail(node)
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise


def _node_summary(node: ContextNode) -> dict[str, Any]:
    external_uri = internal_to_external_uri(node.uri, category=node.category)
    return {
        "uri": external_uri,
        "type": public_type_from_internal(node.category, node.uri),
        "id": external_id_from_uri(external_uri),
        "abstract": node.abstract,
        "updated_at": node.metadata.get("updated_at", ""),
        "has_content": bool(node.content),
    }


def _node_detail(node: ContextNode) -> dict[str, Any]:
    external_uri = internal_to_external_uri(node.uri, category=node.category)
    return {
        "uri": external_uri,
        "type": public_type_from_internal(node.category, node.uri),
        "id": external_id_from_uri(external_uri),
        "abstract": node.abstract,
        "overview": node.overview,
        "content": node.content,
        "metadata": {
            "created_at": node.metadata.get("created_at", ""),
            "updated_at": node.metadata.get("updated_at", ""),
            "version": node.metadata.get("version", 1),
        },
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }


def _validate_public_uri_text(uri: str) -> None:
    if not uri or not isinstance(uri, str):
        raise ValueError("URI must be a non-empty string")
    if uri.startswith("ctx://"):
        raise ValueError("Internal ctx:// URIs are not accepted")
    if "\\" in uri or ".." in uri or ".." in unquote(uri):
        raise ValueError("Invalid URI")
    lowered = uri.lower()
    if "%2f" in lowered or "%5c" in lowered:
        raise ValueError("Encoded path separators are not accepted")
