"""Tool handlers for the oGMemory MCP adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from core.models import RetrievedBlock
from providers.token_tracker import request_usage_scope
from retrieval.result_ranker import is_temporal_query
from server.mcp_resources import (
    MEMORY_TYPES,
    external_id_from_uri,
    external_to_internal_uri_candidates,
    internal_to_external_uri,
    parse_ogmemory_uri,
    public_type_from_internal,
    render_resource,
)
from server.mcp_security import (
    UNTRUSTED_DATA_NOTICE,
    build_mcp_context,
    is_not_found_or_not_visible_error,
    raise_not_found_or_not_visible,
    with_untrusted_data_notice,
)
from skills.artifact import canonical_skill_uri

_FORBIDDEN_WRITE_FIELDS = {
    "type",
    "memory_type",
    "memory_id",
    "uri",
    "abstract",
    "overview",
}
_REVIEWED_SKILL_NOTICE = (
    "This is a published Agent Skill from the oGMemory skill catalog. "
    "Activate it only when its description matches the current task."
)


class SkillCatalogService(Protocol):
    @property
    def default_agent_id(self) -> str: ...

    @property
    def skill_search_scan_limit(self) -> int: ...

    def get_auth_service(self): ...

    def build_context(self, params, identity=None): ...

    def list_published_skills(self, ctx, *, limit: int, cursor: str | None): ...

    def read_published_skill(self, ctx, name: str, source: str | None = None): ...

    def list_skill_candidates(self, ctx, state: str | None = None): ...


def list_memories(
    service: Any,
    *,
    limit: int = 100,
    cursor: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List a page of memory nodes visible to the current MCP identity."""

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    if cursor:
        parsed_cursor = parse_ogmemory_uri(cursor)
        if parsed_cursor.kind != "item" or parsed_cursor.memory_type == "skill":
            raise ValueError("cursor must reference an ogmemory:// memory item")

    _validate_request_size(
        service,
        {
            "limit": limit,
            "cursor": cursor,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    try:
        memories, has_more = _list_memory_nodes_page(
            service,
            ctx,
            limit=limit,
            cursor=cursor,
        )
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise
    payload = {
        "memories": memories,
        "count": len(memories),
        "truncated": has_more,
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }
    if has_more and memories:
        payload["next_cursor"] = memories[-1]["uri"]
    return _truncate_memory_list_payload(
        payload, _configured_max_response_chars(service)
    )


def _list_memory_nodes_page(
    service: Any,
    ctx: Any,
    *,
    limit: int,
    cursor: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    context_fs = service._get_context_fs()
    candidates: list[dict[str, Any]] = []
    fetch_limit = limit + 1

    def append_nodes(base_uri: str, external_prefix: str) -> None:
        after_uri = None
        if cursor:
            if cursor.startswith(external_prefix):
                after_uri = f"{base_uri.rstrip('/')}/{cursor.removeprefix(external_prefix)}"
            elif cursor > external_prefix:
                return
        if hasattr(context_fs, "list_node_summaries_page"):
            nodes = context_fs.list_node_summaries_page(
                base_uri,
                ctx,
                limit=fetch_limit,
                after_uri=after_uri,
            )
        else:
            child_uris = sorted(context_fs.list_children(base_uri, ctx))
            if after_uri:
                child_uris = [uri for uri in child_uris if uri > after_uri]
            nodes = [
                context_fs.read_node(child_uri, ctx)
                for child_uri in child_uris[:fetch_limit]
            ]
        summaries = (_memory_node_summary(node) for node in nodes)
        candidates.extend(
            summary for summary in summaries if cursor is None or summary["uri"] > cursor
        )

    for memory_type, spec in MEMORY_TYPES.items():
        if memory_type == "skill":
            continue
        if memory_type == "profile":
            internal_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/profile"
            external_uri = "ogmemory://memories/profile/self"
            if (cursor is None or external_uri > cursor) and context_fs.exists(
                internal_uri, ctx
            ):
                candidates.append(
                    _memory_node_summary(context_fs.read_node(internal_uri, ctx))
                )
            continue
        if memory_type == "pattern":
            append_nodes(
                f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/patterns",
                "ogmemory://memories/pattern/",
            )
            if ctx.agent_id:
                append_nodes(
                    f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/patterns",
                    "ogmemory://memories/pattern/agent/",
                )
            continue

        owner_id = ctx.agent_id if spec.owner == "agent" else ctx.user_id
        if not owner_id:
            continue
        owner_path = "agents" if spec.owner == "agent" else "users"
        append_nodes(
            f"ctx://{ctx.account_id}/{owner_path}/{owner_id}/memories/{spec.directory}",
            f"ogmemory://memories/{spec.public_type}/",
        )

    candidates.sort(key=lambda memory: memory["uri"])
    return candidates[:limit], len(candidates) > limit


def _memory_node_summary(node: Any) -> dict[str, Any]:
    external_uri = internal_to_external_uri(node.uri, category=node.category)
    return {
        "uri": external_uri,
        "type": public_type_from_internal(node.category, node.uri),
        "id": external_id_from_uri(external_uri),
        "abstract": node.abstract,
        "updated_at": node.metadata.get("updated_at", ""),
        "has_content": bool(node.content) or bool(node.metadata.get("has_content")),
    }


def search_skills(
    service: SkillCatalogService,
    *,
    query: str,
    top_k: int = 5,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Search only the published Agent Skill catalog."""

    if not (query or "").strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    rows, catalog_truncated = _all_published_skill_summaries(
        service,
        agent_id=agent_id,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    query_tokens = set(_search_tokens(query))
    for row in rows:
        haystack = set(_search_tokens(f"{row['name']} {row.get('description', '')}"))
        row["score"] = len(query_tokens & haystack) / max(1, len(query_tokens | haystack))
    skills = sorted(rows, key=lambda row: (-float(row["score"]), row["name"]))[:top_k]
    return {
        "query": query.strip(),
        "skills": skills,
        "truncated": catalog_truncated,
        "artifact_notice": _REVIEWED_SKILL_NOTICE,
    }


def list_skills(
    service: SkillCatalogService,
    *,
    agent_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List published Agent Skills without loading full SKILL.md bodies."""

    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to list skills")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    payload = service.list_published_skills(ctx, limit=limit, cursor=cursor)
    for row in payload["skills"]:
        row["uri"] = canonical_skill_uri(row["name"], row.get("source"))
    return payload


def read_skill(
    service: SkillCatalogService,
    *,
    name: str,
    source: str | None = None,
    include_files: bool = False,
    agent_id: str | None = None,
    max_chars: int = 50000,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Read one reviewed Agent Skill as a SKILL.md artifact."""

    if max_chars < 1 or max_chars > 50000:
        raise ValueError("max_chars must be between 1 and 50000")
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to read skills")
    node = (
        service.read_published_skill(ctx, name)
        if source is None
        else service.read_published_skill(ctx, name, source=source)
    )
    resolved_source = "imported" if node.metadata.get("ogmemory_managed") else "generated"
    uri = canonical_skill_uri(name, resolved_source)
    content = node.content[:max_chars]
    result = {
        "name": name,
        "description": node.abstract,
        "uri": uri,
        "source": resolved_source,
        "content": content,
        "truncated": len(node.content) > len(content),
        "artifact_type": "agent_skill",
        "review_status": "published",
        "artifact_notice": _REVIEWED_SKILL_NOTICE,
    }
    if include_files and resolved_source == "imported":
        from skills.management import (
            ManagedSkill,
            deserialize_package_file,
            package_file_manifest,
        )

        package = ManagedSkill(
            name=name,
            description=node.abstract,
            content=node.content,
            files=tuple(
                deserialize_package_file(record)
                for record in node.metadata.get("package_files") or []
            ),
        )
        result["files"] = package_file_manifest(package, source="imported")
    return result


def list_skill_candidates(
    service: SkillCatalogService,
    *,
    state: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List candidate summaries; candidate trace bodies are never returned."""

    if state and state not in {"accumulating", "held"}:
        raise ValueError("state must be one of: accumulating, held")
    ctx = build_mcp_context(
        service,
        {"agent_id": agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
        use_stdio_agent_default=True,
    )
    if not ctx.agent_id:
        raise ValueError("agent_id is required to list skill candidates")
    candidates = service.list_skill_candidates(ctx, state=state)
    return {"candidates": candidates}


def _all_published_skill_summaries(
    service: SkillCatalogService, **kwargs: Any
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    cursor = None
    scan_limit = service.skill_search_scan_limit
    while True:
        page = list_skills(
            service, limit=min(100, scan_limit - len(rows)), cursor=cursor, **kwargs
        )
        rows.extend(page["skills"])
        if len(rows) >= scan_limit:
            return rows[:scan_limit], bool(page.get("truncated"))
        if not page.get("truncated"):
            return rows, False
        next_cursor = page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            raise RuntimeError("skill catalog returned an invalid pagination cursor")
        cursor = next_cursor


def _search_tokens(value: str) -> list[str]:
    import re

    lowered = value.casefold()
    words = re.findall(r"[^\W_]{2,}", lowered, flags=re.UNICODE)
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered)
    cjk_bigrams = [
        run[index:index + 2]
        for run in cjk_runs
        for index in range(max(1, len(run) - 1))
    ]
    return words + cjk_bigrams


def _serialized_chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True))


def _resize_original_output_recovery(
    payload: dict[str, Any],
    keep_chars: int,
) -> dict[str, Any]:
    result = dict(payload)
    original = result.get("original")
    if not isinstance(original, str):
        raise RuntimeError("original-output recovery returned an invalid result")
    bounded_original = original[:max(0, keep_chars)]
    truncated = bool(result.get("truncated")) or len(bounded_original) < len(original)
    result["original"] = bounded_original
    result["returned_chars"] = len(bounded_original)
    result["truncated"] = truncated

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        bounded_diagnostics = dict(diagnostics)
        bounded_diagnostics["output_chars"] = len(bounded_original)
        bounded_diagnostics["truncated"] = truncated
        result["diagnostics"] = bounded_diagnostics
    return result


def _bound_original_output_recovery_response(
    payload: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    result = with_untrusted_data_notice(payload)
    if _serialized_chars(result) <= max_chars:
        return result

    original = result.get("original")
    if not isinstance(original, str):
        raise ValueError(
            "original-output recovery response limit is too small for response envelope"
        )
    empty = _resize_original_output_recovery(result, 0)
    if _serialized_chars(empty) > max_chars:
        raise ValueError(
            "original-output recovery response limit is too small for response envelope"
        )

    low, high = 0, len(original)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _resize_original_output_recovery(result, middle)
        if _serialized_chars(candidate) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return _resize_original_output_recovery(result, low)


def _truncate_content_record(row: dict[str, Any], keep_chars: int) -> dict[str, Any]:
    import hashlib

    result = dict(row)
    original = str(result.get("content", ""))
    keep_chars = max(0, keep_chars)
    original_offset = int(
        result.get("actual_offset_bytes", result.get("offset_bytes", 0)) or 0
    )
    start_char = 0
    if keep_chars < len(original) and "match_offset_bytes" in result:
        encoded_original = original.encode("utf-8")
        relative_match = int(result.get("match_offset_bytes", original_offset)) - original_offset
        relative_match = max(0, min(len(encoded_original), relative_match))
        while (
            relative_match > 0
            and relative_match < len(encoded_original)
            and encoded_original[relative_match] & 0xC0 == 0x80
        ):
            relative_match -= 1
        match_start = len(encoded_original[:relative_match].decode("utf-8"))
        match_length_bytes = max(1, int(result.get("match_length_bytes", 1) or 1))
        match_end_bytes = min(len(encoded_original), relative_match + match_length_bytes)
        while (
            match_end_bytes > 0
            and match_end_bytes < len(encoded_original)
            and encoded_original[match_end_bytes] & 0xC0 == 0x80
        ):
            match_end_bytes -= 1
        match_end = max(
            match_start + 1,
            len(encoded_original[:match_end_bytes].decode("utf-8")),
        )
        match_chars = max(1, match_end - match_start)
        if keep_chars >= match_chars:
            available = keep_chars - match_chars
            start_char = max(0, match_start - available // 2)
            start_char = min(start_char, max(0, len(original) - keep_chars))
        else:
            start_char = match_start
    content = original[start_char:start_char + keep_chars]
    encoded = content.encode("utf-8")
    offset = original_offset + len(original[:start_char].encode("utf-8"))
    result["content"] = content
    if "offset_bytes" in result:
        result["offset_bytes"] = offset
    if "actual_offset_bytes" in result:
        result["actual_offset_bytes"] = offset
    if "actual_length_bytes" in result:
        result["actual_length_bytes"] = len(encoded)
    if "next_offset_bytes" in result:
        result["next_offset_bytes"] = offset + len(encoded)
    if "decoded_chars" in result:
        result["decoded_chars"] = len(content)
    if "content_sha256" in result:
        result["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    if "chunk_sha256" in result:
        result["chunk_sha256"] = hashlib.sha256(encoded).hexdigest()
    if "eof" in result and keep_chars < len(original):
        result["eof"] = False
    if "fragment_complete" in result and keep_chars < len(original):
        result["fragment_complete"] = False
        empty_sha = hashlib.sha256(b"").hexdigest()
        if "context_before" in result:
            result["context_before"] = ""
            result["context_before_offset_bytes"] = offset
            result["context_before_sha256"] = empty_sha
        if "context_after" in result:
            result["context_after"] = ""
            result["context_after_offset_bytes"] = offset + len(encoded)
            result["context_after_sha256"] = empty_sha
    if "returned_chars" in result:
        result["returned_chars"] = (
            len(content)
            + len(str(result.get("context_before", "")))
            + len(str(result.get("context_after", "")))
        )
    total_bytes = result.get("total_bytes")
    next_offset = result.get("next_offset_bytes")
    if isinstance(total_bytes, int) and isinstance(next_offset, int):
        remaining_bytes = max(0, total_bytes - next_offset)
        continuation_required = remaining_bytes > 0 and not bool(result.get("eof"))
        result["continuation_required"] = continuation_required
        result["remaining_bytes"] = remaining_bytes
        result["next_request"] = (
            {
                "offset_bytes": next_offset,
                "length_bytes": min(
                    remaining_bytes,
                    max(1, int(result.get("actual_length_bytes", 0) or 1)),
                ),
            }
            if continuation_required else None
        )
    return result


def bound_tool_result_read_response(
    payload: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    if _serialized_chars(payload) <= max_chars:
        return payload
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("tool-result response limit is too small for artifact metadata")
    empty = _truncate_content_record(payload, 0)
    if _serialized_chars(empty) > max_chars:
        raise ValueError("tool-result response limit is too small for artifact envelope")
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if _serialized_chars(_truncate_content_record(payload, middle)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return _truncate_content_record(payload, low)


def bound_tool_result_collection_response(
    payload: dict[str, Any],
    key: str,
    max_chars: int,
) -> dict[str, Any]:
    if _serialized_chars(payload) <= max_chars:
        return payload
    items = list(payload.get(key, []) or [])
    result = dict(payload)
    result[key] = []
    result["truncated"] = True
    if _serialized_chars(result) > max_chars:
        raise ValueError("tool-result response limit is too small for response metadata")

    for item in items:
        row = dict(item)
        candidate = dict(result)
        candidate[key] = [*result[key], row]
        if _serialized_chars(candidate) <= max_chars:
            result = candidate
            continue
        content = row.get("content")
        if isinstance(content, str):
            low, high = 0, len(content)
            fitted = None
            while low <= high:
                middle = (low + high) // 2
                bounded_row = _truncate_content_record(row, middle)
                candidate = dict(result)
                candidate[key] = [*result[key], bounded_row]
                if _serialized_chars(candidate) <= max_chars:
                    fitted = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            if fitted is not None and fitted[key][-1].get("content"):
                result = fitted
        break

    if key == "artifacts" and items:
        if not result[key]:
            raise ValueError("tool-result response limit is too small for one artifact summary")
        result["next_cursor"] = result[key][-1].get("artifact_id")
    return result


def list_tool_result_artifacts(
    service: Any,
    *,
    session_id: str,
    agent_id: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """List metadata for verified artifacts owned by one complete session tuple."""
    if not session_id:
        raise ValueError("session_id is required")
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.list_tool_result_artifacts({
        "sessionId": session_id,
        "limit": limit,
        "cursor": cursor,
        "_ctx": ctx,
    })
    payload = {
        "_ogmemory_artifact_list_source": True,
        "schema_version": 1,
        "source_tool": "list_tool_result_artifacts",
        "truncated": False,
        "next_cursor": None,
        **payload,
    }
    max_chars = _configured_max_response_chars(service)
    effective_limit = max_chars
    while effective_limit > 0:
        bounded = bound_tool_result_collection_response(
            payload,
            "artifacts",
            effective_limit,
        )
        attested = service.attest_tool_result_source(bounded, ctx)
        overflow = _serialized_chars(attested) - max_chars
        if overflow <= 0:
            return attested
        effective_limit -= overflow
    raise ValueError("tool-result response limit is too small for list attestation")


def read_tool_result_artifact(
    service: Any,
    *,
    session_id: str,
    ref: str,
    offset_bytes: int = 0,
    length_bytes: int | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Read one bounded UTF-8 byte range.

    A transport response can be shorter than ``length_bytes``.  Callers must
    inspect ``continuation_required``/``eof`` and, while continuation is
    required, repeat the read using ``next_request.offset_bytes`` and
    ``next_request.length_bytes`` before answering from the complete Artifact.
    """
    if not session_id:
        raise ValueError("session_id is required")
    max_chars = _configured_max_response_chars(service)
    resolved_length = (
        int(service._cfg.tool_result_read_default_bytes)
        if length_bytes is None
        else length_bytes
    )
    if resolved_length < 1:
        raise ValueError("length_bytes must be positive")
    # A byte limit also bounds decoded Unicode characters and response size.
    resolved_length = min(resolved_length, max_chars)
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.read_tool_result_artifact({
        "sessionId": session_id,
        "ref": ref,
        "offset_bytes": offset_bytes,
        "length_bytes": resolved_length,
        "_ctx": ctx,
    })
    bounded = bound_tool_result_read_response(payload, max_chars)
    return service.attest_tool_result_source(bounded, ctx)


def search_tool_result_artifacts(
    service: Any,
    *,
    session_id: str,
    query: str,
    top_k: int = 5,
    max_chars: int = 20_000,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
) -> dict[str, Any]:
    """Return bounded exact lines, objects, rows, functions, or short paragraphs.

    Call this only when the stub and existing assistant analysis do not contain
    the required exact fact.  Use ``read_tool_result_artifact`` only if these
    precise fragments are still insufficient; never guess missing identifiers,
    numbers, status codes, errors, counts, or ordering.
    """
    if not session_id:
        raise ValueError("session_id is required")
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    response_limit = _configured_max_response_chars(service)
    ctx = build_mcp_context(
        service,
        {"session_id": session_id, "agent_id": agent_id or service.default_agent_id},
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    payload = service.search_tool_result_artifacts({
        "sessionId": session_id,
        "query": query,
        "top_k": top_k,
        "max_chars": min(max(1, max_chars), response_limit),
        "_ctx": ctx,
    })
    bounded = bound_tool_result_collection_response(payload, "results", response_limit)
    bounded["results"] = [
        service.attest_tool_result_source(row, ctx)
        for row in bounded.get("results", [])
    ]
    return bounded


def query_memory(
    service: Any,
    *,
    query: str,
    types: list[str] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    if not (query or "").strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    requested_types = types or []
    unknown = sorted(t for t in requested_types if t not in MEMORY_TYPES)
    if unknown:
        raise ValueError(f"Unknown memory types: {', '.join(unknown)}")
    _validate_request_size(
        service,
        {
            "query": query,
            "types": requested_types,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    categories = [MEMORY_TYPES[t].internal_category for t in requested_types] or None
    search_top_k = _internal_search_top_k(service, query, top_k)

    try:
        with request_usage_scope(ctx):
            result = service.get_read_api().search_memory(
                query.strip(),
                ctx,
                top_k=search_top_k,
                categories=categories,
                score_threshold=score_threshold,
                include_debug=False,
                fill_content_for_top_k=min(top_k, 3),
            )
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise

    hits = []
    for hit in result.hits[:top_k]:
        try:
            hits.append(_hit_to_public(hit))
        except ValueError:
            continue

    payload = {
        "request_id": result.request_id,
        "query": result.query or query,
        "hits": hits,
        "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
    }
    return _truncate_query_payload(payload, _configured_max_response_chars(service))


def read_memory(
    service: Any,
    *,
    uri: str,
    level: str = "all",
    max_chars: int = 12000,
    recover_original: bool = False,
    session_id: str | None = None,
    agent_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    if level not in {"abstract", "overview", "content", "all"}:
        raise ValueError("level must be one of: abstract, overview, content, all")
    if max_chars < 1 or max_chars > 50000:
        raise ValueError("max_chars must be between 1 and 50000")
    if not isinstance(recover_original, bool):
        raise ValueError("recover_original must be a boolean")

    if recover_original:
        if level != "all":
            raise ValueError("level is not supported with recover_original")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("uri must be a filtered-output memory URI")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise ValueError("max_chars must be an integer with recover_original")
        _validate_request_size(
            service,
            {
                "uri": uri,
                "max_chars": max_chars,
                "recover_original": recover_original,
                "session_id": session_id,
                "agent_id": agent_id,
            },
        )
        max_chars = min(max_chars, _configured_max_response_chars(service))
        # Preserve a context supplied by the authenticated HTTP adapter.  When
        # the wrapper builds the context itself, use that same trusted context
        # for the recovery facade; do not reintroduce the optional raw
        # ``agent_id`` argument after transport/header resolution.  This keeps
        # the documented URI-only recovery call working for authenticated MCP
        # clients that bind the agent through the transport.
        ctx = request_context or build_mcp_context(
            service,
            {"session_id": session_id, "agent_id": agent_id},
            transport_headers=transport_headers,
            allow_stdio_env_auth=allow_stdio_env_auth,
            resolved_identity=resolved_identity,
            use_stdio_agent_default=True,
        )
        trusted_agent_id = getattr(ctx, "agent_id", "")
        try:
            result = _public_result_mapping(
                service.read_tool_output_original(
                    {
                        "memory_uri": uri,
                        "max_chars": max_chars,
                        "sessionId": session_id,
                        "agentId": trusted_agent_id,
                        "_ctx": ctx,
                    }
                ),
                operation="original-output recovery",
            )
            return _bound_original_output_recovery_response(result, max_chars)
        except Exception as exc:
            raise_not_found_or_not_visible(exc)
            raise

    _validate_request_size(
        service,
        {
            "uri": uri,
            "level": level,
            "max_chars": max_chars,
            "session_id": session_id,
            "agent_id": agent_id,
        },
    )
    max_chars = min(max_chars, _configured_max_response_chars(service))

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    parsed = parse_ogmemory_uri(uri)

    if parsed.kind != "item":
        payload = render_resource(service, uri, ctx)
        return _truncate_payload(payload, max_chars)

    try:
        if parsed.memory_type == "skill" and parsed.memory_id:
            source = parsed.skill_source or "generated"
            node = service.read_published_skill(ctx, parsed.memory_id, source=source)
            canonical_uri = internal_to_external_uri(node.uri, category=node.category)
            payload = {
                "uri": canonical_uri,
                "type": "skill",
                "id": parsed.memory_id,
                "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
            }
            if level in {"abstract", "all"}:
                payload["abstract"] = node.abstract
            if level in {"overview", "all"}:
                payload["overview"] = node.overview
            if level in {"content", "all"}:
                payload["content"] = node.content
            return _truncate_payload(payload, max_chars)

        context_fs = service._get_context_fs()
        last_exc: Exception | None = None
        node = None
        for internal_uri in external_to_internal_uri_candidates(uri, ctx):
            try:
                if hasattr(context_fs, "read_node_fields"):
                    node = context_fs.read_node_fields(internal_uri, ctx, level)
                else:
                    node = context_fs.read_node(internal_uri, ctx)
                break
            except Exception as exc:
                if not is_not_found_or_not_visible_error(exc):
                    raise
                last_exc = exc
        if node is None:
            if last_exc is not None:
                raise last_exc
            raise ValueError(f"URI does not reference a single memory: {uri}")
        canonical_uri = internal_to_external_uri(node.uri, category=node.category)
        canonical_parsed = parse_ogmemory_uri(canonical_uri)
        payload = {
            "uri": canonical_uri,
            "type": canonical_parsed.memory_type,
            "id": canonical_parsed.memory_id,
            "untrusted_data_notice": UNTRUSTED_DATA_NOTICE,
        }
        if level in {"abstract", "all"}:
            payload["abstract"] = node.abstract
        if level in {"overview", "all"}:
            payload["overview"] = node.overview
        if level in {"content", "all"}:
            payload["content"] = node.content
        return _truncate_payload(payload, max_chars)
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise


def delete_memory(
    service: Any,
    *,
    uri: str,
    reason: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    client_request_id: str | None = None,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    parsed = parse_ogmemory_uri(uri)
    if parsed.kind != "item":
        raise ValueError("delete_memory requires a single ogmemory://memories/{type}/{id} URI")
    if parsed.memory_type == "skill":
        raise ValueError("generated skills cannot be deleted through ogm_delete_memory")
    _validate_request_size(
        service,
        {
            "uri": uri,
            "reason": reason,
            "session_id": session_id,
            "agent_id": agent_id,
            "client_request_id": client_request_id,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )

    try:
        last_exc: Exception | None = None
        result = None
        for internal_uri in external_to_internal_uri_candidates(uri, ctx):
            try:
                result = service.forget_memory(
                    internal_uri,
                    ctx,
                    external_uri=uri,
                    reason=reason,
                    client_request_id=client_request_id,
                )
                break
            except Exception as exc:
                if not is_not_found_or_not_visible_error(exc):
                    raise
                last_exc = exc
        if result is None:
            if last_exc is not None:
                raise last_exc
            raise ValueError(f"URI does not reference a single memory: {uri}")
    except Exception as exc:
        raise_not_found_or_not_visible(exc)
        raise

    return {
        "ok": bool(result.get("ok", True)),
        "deleted": bool(result.get("deleted", True)),
        "mode": result.get("mode", "archive"),
        "uri": uri,
        "type": parsed.memory_type,
        "id": parsed.memory_id,
        "client_request_id": client_request_id,
    }


def _hit_to_public(hit: RetrievedBlock) -> dict[str, Any]:
    external_uri = internal_to_external_uri(hit.uri, category=hit.category)
    parsed = parse_ogmemory_uri(external_uri)
    row = {
        "uri": external_uri,
        "type": parsed.memory_type or public_type_from_internal(hit.category, hit.uri),
        "id": external_id_from_uri(external_uri),
        "score": hit.score,
        "abstract": hit.abstract,
        "has_content": hit.has_content or bool(hit.content_excerpt),
    }
    if hit.overview:
        row["overview"] = hit.overview
    if hit.content_excerpt:
        row["content_excerpt"] = hit.content_excerpt
    for field in ("when", "who", "where"):
        value = (hit.metadata or {}).get(field)
        if value:
            row[field] = value
    return row


def _internal_search_top_k(service: Any, query: str, requested_top_k: int) -> int:
    if not is_temporal_query(query):
        return requested_top_k
    max_top_k = getattr(getattr(service.get_read_api(), "_cfg", None), "max_top_k", 50)
    return min(max_top_k, 50, max(requested_top_k * 3, 15))


def _truncate_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    remaining = max_chars
    truncated = bool(payload.get("truncated", False))
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str) and key in {"abstract", "overview", "content"}:
            if len(value) > remaining:
                result[key] = value[: max(0, remaining)]
                truncated = True
                remaining = 0
            else:
                result[key] = value
                remaining -= len(value)
        else:
            result[key] = value
    result["truncated"] = truncated
    return result


def _truncate_query_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    remaining = max_chars
    truncated = False
    result = dict(payload)
    truncated_hits: list[dict[str, Any]] = []
    for hit in payload.get("hits", []):
        if remaining <= 0:
            truncated = True
            break
        row = dict(hit)
        for field in ("abstract", "overview", "content_excerpt"):
            value = row.get(field)
            if not isinstance(value, str):
                continue
            if len(value) > remaining:
                row[field] = value[: max(0, remaining)]
                truncated = True
                remaining = 0
                break
            remaining -= len(value)
        truncated_hits.append(row)
    result["hits"] = truncated_hits
    result["truncated"] = truncated
    return result


def _truncate_memory_list_payload(
    payload: dict[str, Any], max_chars: int
) -> dict[str, Any]:
    result = dict(payload)
    result.setdefault("truncated", False)
    if len(json.dumps(result, ensure_ascii=False, default=str)) <= max_chars:
        return result

    memories = list(payload.get("memories", []))
    result["memories"] = []
    result["count"] = 0
    result["truncated"] = True
    result.pop("next_cursor", None)
    if len(json.dumps(result, ensure_ascii=False, default=str)) > max_chars:
        raise ValueError(
            "mcp_max_response_chars is too small for ogm_list_memories metadata"
        )
    for memory in memories:
        candidate = dict(result)
        candidate["memories"] = [*result["memories"], memory]
        candidate["count"] = len(candidate["memories"])
        candidate["next_cursor"] = memory["uri"]
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) > max_chars:
            break
        result = candidate
    if not result["memories"]:
        raise ValueError(
            "mcp_max_response_chars is too small for one ogm_list_memories item"
        )
    return result


def _configured_max_request_bytes(service: Any) -> int:
    value = int(getattr(service._cfg, "mcp_max_request_bytes", 1048576) or 1048576)
    return max(1, value)


def _configured_max_response_chars(service: Any) -> int:
    value = int(getattr(service._cfg, "mcp_max_response_chars", 50000) or 50000)
    return max(1, min(value, 50000))


def _public_result_mapping(value: object, *, operation: str) -> dict[str, Any]:
    """Validate a service facade result before exposing it through MCP."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{operation} returned an invalid result")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeError(f"{operation} returned an invalid result")
        result[key] = item
    return result


def _validate_request_size(service: Any, payload: dict[str, Any]) -> None:
    max_bytes = _configured_max_request_bytes(service)
    encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"MCP request exceeds mcp_max_request_bytes ({max_bytes})")


def write_memory(
    service: Any,
    *,
    text: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    client_request_id: str | None = None,
    force_extract: bool = False,
    wait: bool = False,
    transport_headers: dict[str, Any] | None = None,
    allow_stdio_env_auth: bool = True,
    resolved_identity: Any | None = None,
    request_context: Any | None = None,
) -> dict[str, Any]:
    if session_id is None or not session_id.strip():
        raise ValueError("session_id is required for write_memory")
    normalized_messages = _normalize_write_messages(text=text, messages=messages)
    _validate_request_size(
        service,
        {
            "messages": normalized_messages,
            "session_id": session_id,
            "agent_id": agent_id,
            "client_request_id": client_request_id,
            "force_extract": force_extract,
            "wait": wait,
        },
    )

    arguments = {"session_id": session_id, "agent_id": agent_id}
    ctx = request_context or build_mcp_context(
        service,
        arguments,
        transport_headers=transport_headers,
        allow_stdio_env_auth=allow_stdio_env_auth,
        resolved_identity=resolved_identity,
    )
    params: dict[str, Any] = {
        "sessionId": session_id,
        "messages": normalized_messages,
        "_ctx": ctx,
    }
    if agent_id:
        params["agentId"] = agent_id
    if client_request_id:
        params["clientRequestId"] = client_request_id
        params["client_request_id"] = client_request_id
    if force_extract:
        params["forceExtract"] = True
    if wait:
        params["wait"] = True

    try:
        result = service.after_turn(params)
    except Exception as exc:
        _record_mcp_write_audit(
            service,
            ctx,
            session_id=session_id,
            agent_id=agent_id,
            client_request_id=client_request_id,
            result="failed",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    raw_status = result.get("status")
    accepted = not (
        result.get("ok") is False
        or result.get("retryable") is True
        or (isinstance(raw_status, int) and 400 <= raw_status <= 599)
        or raw_status in {"failed", "error", "rejected"}
    )
    _record_mcp_write_audit(
        service,
        ctx,
        session_id=session_id,
        agent_id=agent_id,
        client_request_id=client_request_id,
        result="success" if accepted else "failed",
        details={
            "status": result.get("status", "accepted"),
            "accepted_messages": len(normalized_messages) if accepted else 0,
            "force_extract_requested": force_extract,
            "wait_requested": wait,
        },
    )

    accepted_messages = len(normalized_messages) if accepted else 0
    return {
        "ok": accepted,
        "status": result.get("status", "accepted"),
        "retryable": bool(result.get("retryable", False)),
        "session_id": session_id,
        "agent_id": agent_id or ctx.agent_id,
        "client_request_id": client_request_id,
        "submitted_messages": len(normalized_messages),
        "accepted_messages": accepted_messages,
        "stats": _public_write_stats(result),
    }


def _normalize_write_messages(
    *,
    text: str | None,
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if text and messages:
        raise ValueError("Provide either text or messages, not both")
    if text is not None:
        if not text.strip():
            raise ValueError("text must not be empty")
        return [{"role": "user", "content": text}]
    if not messages:
        raise ValueError("messages or text is required")

    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        forbidden = sorted(k for k in message if k in _FORBIDDEN_WRITE_FIELDS)
        if forbidden:
            raise ValueError(
                f"messages[{index}] cannot set atomic memory fields: {', '.join(forbidden)}"
            )
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{index}].role must be system, user, assistant, or tool")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be a non-empty string")
        normalized_message: dict[str, Any] = {"role": role, "content": content}
        if message.get("id"):
            normalized_message["id"] = str(message["id"])
        if message.get("created_at"):
            normalized_message["created_at"] = str(message["created_at"])
        normalized.append(normalized_message)
    return normalized


def _record_mcp_write_audit(
    service: Any,
    ctx: Any,
    *,
    session_id: str,
    agent_id: str | None,
    client_request_id: str | None,
    result: str,
    details: dict[str, Any],
) -> None:
    get_audit_service = getattr(service, "get_audit_service", None)
    if not callable(get_audit_service):
        return
    try:
        get_audit_service().record(
            ctx.account_id,
            actor=ctx.user_id,
            target=session_id,
            action="mcp.write_memory",
            result=result,
            trace_id=ctx.trace_id,
            details={
                "role": str(ctx.role),
                "agent_id": agent_id or ctx.agent_id,
                "session_id": session_id,
                "client_request_id": client_request_id,
                **details,
            },
        )
    except Exception:
        return


def _public_write_stats(result: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "pending_tokens",
        "buffered_tokens",
        "candidates_extracted",
        "writes_completed",
        "writes_failed",
        "task_id",
        "reason",
    }
    stats: dict[str, Any] = {}
    for field in allowed_fields:
        value = result.get(field)
        if isinstance(value, (str, int, float, bool)) or value is None:
            stats[field] = value
    return stats
