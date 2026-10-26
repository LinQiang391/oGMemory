"""Provider-neutral recognition and rewriting of structured tool results."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .models import NormalizedMessage, ToolResultPart

_RESULT_BLOCK_TYPES = {"tool_result", "toolresult"}
_CALL_BLOCK_TYPES = {"tool_call", "toolcall", "tool_use", "tooluse"}
_STUB_REF = re.compile(r"^ogmemory://tool-results/tr_[a-f0-9_]+$")
_STUB_START = re.compile(r"^\[ogmemory-tool-result v=2\]$")


def canonical_tool_call_id(value: Any) -> str:
    """Normalize provider punctuation rewrites without inventing an identity."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(value or "")).casefold()


def _complete_stub_ref(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lines = value.splitlines()
    if len(lines) < 3 or _STUB_START.fullmatch(lines[0]) is None:
        return None
    if lines[-1] != "[/ogmemory-tool-result]":
        return None
    fields = {}
    for line in lines[1:-1]:
        if ": " not in line:
            return None
        key, field_value = line.split(": ", 1)
        if key in fields:
            return None
        fields[key] = field_value
    ref = fields.get("ref")
    sha256 = fields.get("sha256", "")
    if not isinstance(ref, str) or not _STUB_REF.fullmatch(ref):
        return None
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        return None
    return ref


def _decoded_source_value(value: Any) -> Any | None:
    """Decode canonical JSON or OpenClaw's exact structuredContent projection."""
    if not isinstance(value, str):
        return value
    candidate = value
    first_line, separator, remainder = value.partition("\n")
    if separator and first_line.rstrip("\r") == "structuredContent:":
        candidate = remainder
    try:
        return json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _canonical_text(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, "raw_string"
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "canonical_json",
    )


def _first_text(mapping: dict, *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _source_identity(message: dict) -> tuple[str | None, str | None, bool]:
    source = message.get("_ogmem_source")
    if isinstance(source, dict):
        source_id = _first_text(source, "source_message_id", "sourceMessageId")
        turn_id = _first_text(source, "turn_id", "turnId")
        explicitly_fallback = (
            source.get("identity_strength") == "fallback"
            or source.get("identity_stable") is False
        )
        return source_id, turn_id, bool(source_id or turn_id) and not explicitly_fallback
    # Only the adapter-owned private envelope attests runtime stability. An
    # arbitrary provider-level ``id`` must not silently become a strong key.
    return None, None, False


class ToolResultNormalizer:
    """Recognize known tool-result schemas without mutating caller objects."""

    def normalize(self, messages: list[Any]) -> list[NormalizedMessage]:
        copied = [deepcopy(message) if isinstance(message, dict) else {
            "role": getattr(message, "role", ""),
            "content": deepcopy(getattr(message, "content", "")),
        } for message in messages]
        pending_calls: list[tuple[str | None, str | None]] = []
        normalized: list[NormalizedMessage] = []
        occurrence_by_call: dict[str, int] = {}
        active_turn_id: str | None = None

        for message_index, message in enumerate(copied):
            source_message_id, turn_id, stable = _source_identity(message)
            if turn_id and active_turn_id and turn_id != active_turn_id:
                pending_calls.clear()
                occurrence_by_call.clear()
            if turn_id:
                active_turn_id = turn_id
            role = str(message.get("role", ""))
            parts: list[ToolResultPart] = []
            ordinary: list[str] = []
            associated_call: tuple[str | None, str | None] | None = None
            role_key = role.casefold()

            content = message.get("content", "")
            if role_key in {"toolresult", "tool_result", "tool"}:
                call_id = _first_text(
                    message,
                    "tool_call_id",
                    "toolCallId",
                    "tool_use_id",
                    "toolUseId",
                )
                tool_name = _first_text(message, "tool_name", "toolName", "name")
                associated_call = self._take_pending_call(pending_calls, call_id)
                if associated_call:
                    call_id = call_id or associated_call[0]
                    tool_name = tool_name or associated_call[1]
                parts.extend(self._message_level_parts(content, call_id, tool_name))
            else:
                parts.extend(self._block_parts(content))
                if parts:
                    associations: dict[str, tuple[str | None, str | None] | None] = {}
                    enriched_parts = []
                    for part in parts:
                        association = None
                        if part.tool_call_id:
                            if part.tool_call_id not in associations:
                                associations[part.tool_call_id] = self._take_pending_call(
                                    pending_calls,
                                    part.tool_call_id,
                                )
                            association = associations[part.tool_call_id]
                        else:
                            association = self._take_pending_call(pending_calls, None)
                        if association:
                            part = ToolResultPart(**{
                                **part.__dict__,
                                "tool_call_id": part.tool_call_id or association[0],
                                "tool_name": part.tool_name or association[1],
                            })
                        enriched_parts.append(part)
                    parts = enriched_parts

            # Each assistant tool-call message starts a new provider call batch.
            # Results cannot legitimately arrive for an older unresolved batch
            # after a newer assistant message, so stale anonymous calls must not
            # leak across turns. A normal user message is likewise a boundary;
            # Anthropic user-role tool_result blocks are not.
            next_calls: list[tuple[str | None, str | None]] = []
            self._collect_calls(message, next_calls)
            if role_key == "assistant":
                pending_calls[:] = next_calls
                for call_id, _tool_name in next_calls:
                    if call_id:
                        occurrence_by_call.pop(call_id, None)
            elif role_key == "user" and not parts:
                pending_calls.clear()
            elif next_calls:
                pending_calls.extend(next_calls)

            self._collect_ordinary_text(content, role, ordinary)
            with_occurrences: list[ToolResultPart] = []
            for part in parts:
                occurrence_key = part.tool_call_id or f"{message_index}:{part.part_path}"
                occurrence = occurrence_by_call.get(occurrence_key, 0)
                occurrence_by_call[occurrence_key] = occurrence + 1
                with_occurrences.append(
                    ToolResultPart(**{**part.__dict__, "occurrence_index": occurrence})
                )
            normalized.append(NormalizedMessage(
                source_message_id=source_message_id,
                turn_id=turn_id,
                role=role,
                ordinary_text_parts=tuple(ordinary),
                tool_results=tuple(with_occurrences),
                original=message,
                message_index=message_index,
                identity_is_stable=stable,
            ))
        return normalized

    @staticmethod
    def _take_pending_call(
        pending: list[tuple[str | None, str | None]],
        call_id: str | None,
    ) -> tuple[str | None, str | None] | None:
        """Match explicit ids exactly; use FIFO only for anonymous results."""
        if call_id:
            for index, candidate in enumerate(pending):
                if candidate[0] == call_id:
                    return pending.pop(index)
            return None
        return pending.pop(0) if pending else None

    @staticmethod
    def _collect_calls(message: dict, pending: list[tuple[str | None, str | None]]) -> None:
        seen_ids = {call_id for call_id, _tool_name in pending if call_id}

        def append_call(call_id: str | None, tool_name: str | None) -> None:
            if call_id:
                if call_id in seen_ids:
                    return
                seen_ids.add(call_id)
            pending.append((call_id, tool_name))

        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).casefold()
            if block_type in _CALL_BLOCK_TYPES and not any(
                key in block for key in ("tool_output", "toolOutput")
            ):
                append_call(
                    _first_text(
                        block,
                        "id",
                        "tool_call_id",
                        "toolCallId",
                        "tool_use_id",
                        "toolUseId",
                    ),
                    _first_text(block, "name", "tool_name", "toolName"),
                )
        calls = message.get("tool_calls") or message.get("toolCalls") or []
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    function = call.get("function") if isinstance(call.get("function"), dict) else {}
                    append_call(
                        _first_text(
                            call,
                            "id",
                            "tool_call_id",
                            "toolCallId",
                            "tool_use_id",
                            "toolUseId",
                        ),
                        _first_text(call, "name", "tool_name", "toolName")
                        or _first_text(function, "name"),
                    )

    @staticmethod
    def _source_ref(value: Any) -> str | None:
        envelope = _decoded_source_value(value)
        if envelope is None:
            return _complete_stub_ref(value)
        if isinstance(envelope, dict) and envelope.get("_ogmemory_artifact_source") is True:
            ref = envelope.get("artifact_ref")
            return str(ref) if isinstance(ref, str) else None
        return None

    @classmethod
    def _source_collection_claim(
        cls,
        value: Any,
        *,
        tool_name: str | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        envelope = _decoded_source_value(value)
        if not isinstance(envelope, dict):
            return False, ()
        results = envelope.get("results")
        metadata_listing = False
        canonical_tool_name = str(tool_name or "").rsplit("__", 1)[-1].casefold()
        if (
            not isinstance(results, list)
            and canonical_tool_name == "list_tool_result_artifacts"
        ):
            results = envelope.get("artifacts")
            metadata_listing = True
        if not isinstance(results, list):
            return False, ()
        if not results:
            return (
                bool(
                    metadata_listing
                    and envelope.get("_ogmemory_artifact_list_source") is True
                    and envelope.get("schema_version") == 1
                    and envelope.get("source_tool")
                    == "list_tool_result_artifacts"
                ),
                (),
            )
        claimed = metadata_listing
        refs: list[str] = []
        for item in results:
            if metadata_listing:
                ref = item.get("ref") if isinstance(item, dict) else None
                if isinstance(ref, str) and _STUB_REF.fullmatch(ref):
                    claimed = True
                else:
                    ref = None
            else:
                if isinstance(item, dict) and (
                    item.get("_ogmemory_artifact_source") is True
                    or "artifact_ref" in item
                    or "source_attestation" in item
                ):
                    claimed = True
                ref = cls._source_ref(item)
            if ref:
                refs.append(ref)
        return claimed, tuple(refs)

    @staticmethod
    def _part(
        value: Any,
        path: tuple[Any, ...],
        call_id: str | None,
        tool_name: str | None,
        mime_type: str | None = None,
        source_ref: str | None = None,
        source_collection: bool = False,
        source_refs: tuple[str, ...] = (),
    ) -> ToolResultPart:
        text, serialization = _canonical_text(value)
        return ToolResultPart(
            part_path="/" + "/".join(str(item) for item in path),
            payload_path=path,
            tool_call_id=call_id,
            tool_name=tool_name,
            output_text=text,
            mime_type=mime_type,
            serialization=serialization,
            source_ref=source_ref,
            original_chars=len(text),
            original_bytes=len(text.encode("utf-8")),
            source_collection=source_collection,
            source_refs=source_refs,
        )

    def _message_level_parts(
        self,
        content: Any,
        call_id: str | None,
        tool_name: str | None,
    ) -> list[ToolResultPart]:
        if isinstance(content, list):
            result: list[ToolResultPart] = []
            for index, block in enumerate(content):
                if isinstance(block, dict):
                    for payload_key in self._payload_keys(block, allow_text=True):
                        payload = block[payload_key]
                        nested = self._source_aware_parts(
                            payload,
                            ("content", index, payload_key),
                            call_id,
                            tool_name,
                            _first_text(block, "mime_type", "mimeType"),
                        )
                        result.extend(nested or [self._part(
                            payload, ("content", index, payload_key),
                            call_id, tool_name,
                            _first_text(block, "mime_type", "mimeType"),
                            self._source_ref(payload),
                        )])
                elif isinstance(block, str):
                    result.append(self._part(
                        block,
                        ("content", index),
                        call_id,
                        tool_name,
                        source_ref=self._source_ref(block),
                    ))
            return result
        nested = self._source_aware_parts(
            content, ("content",), call_id, tool_name, None,
        )
        return nested or [self._part(
            content, ("content",), call_id, tool_name,
            source_ref=self._source_ref(content),
        )]

    def _block_parts(self, content: Any) -> list[ToolResultPart]:
        if not isinstance(content, list):
            return []
        result: list[ToolResultPart] = []
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).casefold()
            if block_type in _RESULT_BLOCK_TYPES:
                payload_keys = self._payload_keys(block, allow_text=True)
            elif block_type in _CALL_BLOCK_TYPES:
                payload_keys = self._payload_keys(block, allow_text=False)
            else:
                payload_keys = []
            if not payload_keys:
                continue
            for payload_key in payload_keys:
                payload = block[payload_key]
                path = ("content", index, payload_key)
                call_id = _first_text(
                    block,
                    "tool_call_id",
                    "toolCallId",
                    "tool_use_id",
                    "toolUseId",
                    "id",
                )
                tool_name = _first_text(block, "tool_name", "toolName", "name")
                mime_type = _first_text(block, "mime_type", "mimeType")
                nested = self._source_aware_parts(
                    payload, path, call_id, tool_name, mime_type,
                )
                result.extend(nested or [self._part(
                    payload, path, call_id, tool_name, mime_type,
                    self._source_ref(payload),
                )])
        return result

    def _source_aware_parts(
        self,
        value: Any,
        path: tuple[Any, ...],
        call_id: str | None,
        tool_name: str | None,
        mime_type: str | None,
    ) -> list[ToolResultPart]:
        decoded = _decoded_source_value(value)
        source_value = decoded if isinstance(decoded, dict) else value
        direct_ref = self._source_ref(source_value)
        if direct_ref:
            return [self._part(
                source_value, path, call_id, tool_name, mime_type, direct_ref,
            )]
        collection_claimed, collection_refs = self._source_collection_claim(
            source_value,
            tool_name=tool_name,
        )
        if collection_claimed:
            return [self._part(
                source_value,
                path,
                call_id,
                tool_name,
                mime_type,
                source_collection=True,
                source_refs=collection_refs,
            )]
        if not isinstance(source_value, dict) or not isinstance(source_value.get("results"), list):
            return []
        parts: list[ToolResultPart] = []
        for index, item in enumerate(source_value["results"]):
            ref = self._source_ref(item)
            if ref:
                parts.append(self._part(
                    item,
                    (*path, "results", index),
                    call_id,
                    tool_name,
                    mime_type,
                    ref,
                ))
        return parts

    @staticmethod
    def _payload_keys(block: dict, *, allow_text: bool) -> list[str]:
        keys = ["tool_output", "toolOutput", "output", "result", "content"]
        if allow_text:
            keys.append("text")
        return [key for key in keys if key in block]

    @staticmethod
    def _collect_ordinary_text(content: Any, role: str, target: list[str]) -> None:
        if role.casefold() in {"toolresult", "tool_result", "tool"}:
            return
        if isinstance(content, str):
            target.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    target.append(block)
                elif isinstance(block, dict):
                    block_type = str(block.get("type", "")).casefold()
                    if block_type not in _RESULT_BLOCK_TYPES and block_type not in _CALL_BLOCK_TYPES:
                        text = block.get("text")
                        if isinstance(text, str):
                            target.append(text)

    @staticmethod
    def set_payload(message: dict, path: tuple[Any, ...], value: Any) -> None:
        parent: Any = message
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value

    @staticmethod
    def remove_payload(message: dict, path: tuple[Any, ...]) -> None:
        parent: Any = message
        for key in path[:-1]:
            parent = parent[key]
        payload_key = path[-1]
        if isinstance(parent, list) and isinstance(payload_key, int):
            parent[payload_key] = ""
        elif isinstance(parent, dict):
            if len(path) == 1:
                parent[payload_key] = ""
            else:
                parent.pop(payload_key, None)

    @staticmethod
    def strip_private_source(message: dict) -> dict:
        message.pop("_ogmem_source", None)
        return message
