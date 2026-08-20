"""Persistent server-issued identities for complete user-message turns.

OpenClaw 2026.6.10 does not expose stable message or turn ids to the context
engine.  This module derives an ordered, owner-bound projection for a complete
user -> tools -> terminal-assistant turn, allocates server identities once, and
later restores those identities only when the complete projection matches
uniquely.  Content digests are evidence inside the ordered projection; they are
never used as a global content -> Artifact identity map.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from typing import Any, Mapping

from core.models import RequestContext

from .normalizer import ToolResultNormalizer, canonical_tool_call_id


_HEX_64 = re.compile(r"^[a-f0-9]{64}$")
_TURN_ID = re.compile(r"^ogt_[a-f0-9]{32}$")
_MESSAGE_ID = re.compile(r"^ogm_[a-f0-9]{32}$")
_TOOL_CALL_TYPES = {"tool_call", "toolcall", "tool_use", "tooluse"}
_TOOL_RESULT_TYPES = {
    "tool_result", "toolresult", "tool_output", "tooloutput",
    "function_result", "functionresult", "function_output", "functionoutput",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _role(message: Any) -> str:
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("role", "")).strip().casefold().replace("-", "_")


def _type(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _tool_block_projection(block: Mapping[str, Any]) -> dict[str, Any]:
    block_type = _type(block.get("type"))
    result = {"type": block_type}
    call_id = _first_text(
        block,
        "id", "tool_call_id", "toolCallId", "tool_use_id", "toolUseId",
    )
    if call_id:
        result["call_key"] = canonical_tool_call_id(call_id)
    name = _first_text(block, "name", "tool_name", "toolName")
    if name:
        result["name"] = name
    if block_type in _TOOL_CALL_TYPES:
        for key in ("arguments", "input"):
            if key in block:
                result[key] = block[key]
    if block_type in _TOOL_RESULT_TYPES or block_type in _TOOL_CALL_TYPES:
        for key in (
            "tool_output", "toolOutput", "output", "result", "content", "text",
        ):
            if key in block:
                result[key] = block[key]
    return result


def _content_projection(content: Any) -> Any:
    if isinstance(content, list):
        projected = []
        for block in content:
            if isinstance(block, Mapping):
                block_type = _type(block.get("type"))
                if block_type in _TOOL_CALL_TYPES or block_type in _TOOL_RESULT_TYPES:
                    projected.append(_tool_block_projection(block))
                else:
                    projected.append(deepcopy(dict(block)))
            else:
                projected.append(deepcopy(block))
        return projected
    if isinstance(content, Mapping):
        block_type = _type(content.get("type"))
        if block_type in _TOOL_CALL_TYPES or block_type in _TOOL_RESULT_TYPES:
            return _tool_block_projection(content)
        return deepcopy(dict(content))
    return deepcopy(content)


def _message_projection(message: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": _role(message),
        "content": _content_projection(message.get("content", "")),
    }
    call_id = _first_text(
        message,
        "tool_call_id", "toolCallId", "tool_use_id", "toolUseId",
    )
    if call_id:
        result["call_key"] = canonical_tool_call_id(call_id)
    tool_name = _first_text(message, "tool_name", "toolName", "name")
    if tool_name:
        result["tool_name"] = tool_name
    calls = message.get("tool_calls") or message.get("toolCalls")
    if isinstance(calls, list):
        projected_calls = []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
            projected_calls.append({
                "call_key": canonical_tool_call_id(_first_text(
                    call,
                    "id", "tool_call_id", "toolCallId", "tool_use_id", "toolUseId",
                )),
                "name": _first_text(call, "name", "tool_name", "toolName")
                or _first_text(function, "name"),
                "arguments": function.get("arguments", call.get("arguments")),
            })
        result["tool_calls"] = projected_calls
    for key in ("isError", "stopReason", "stop_reason"):
        if key in message:
            result[key] = message[key]
    if message.get("error") or message.get("errorMessage"):
        result["error"] = True
    return result


def _assistant_has_tool_call(message: Mapping[str, Any]) -> bool:
    if _role(message) != "assistant":
        return False
    calls = message.get("tool_calls") or message.get("toolCalls")
    if isinstance(calls, list) and calls:
        return True
    content = message.get("content")
    blocks = content if isinstance(content, list) else [content]
    return any(
        isinstance(block, Mapping) and _type(block.get("type")) in _TOOL_CALL_TYPES
        for block in blocks
    )


def _assistant_tool_calls(message: Mapping[str, Any]) -> list[tuple[str, str]]:
    if _role(message) != "assistant":
        return []
    result: list[tuple[str, str]] = []
    content = message.get("content")
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        if _type(block.get("type")) not in _TOOL_CALL_TYPES:
            continue
        result.append((
            canonical_tool_call_id(_first_text(
                block,
                "id", "tool_call_id", "toolCallId", "tool_use_id", "toolUseId",
            )),
            _first_text(block, "name", "tool_name", "toolName"),
        ))
    calls = message.get("tool_calls") or message.get("toolCalls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = (
                call.get("function")
                if isinstance(call.get("function"), Mapping)
                else {}
            )
            result.append((
                canonical_tool_call_id(_first_text(
                    call,
                    "id", "tool_call_id", "toolCallId", "tool_use_id", "toolUseId",
                )),
                _first_text(call, "name", "tool_name", "toolName")
                or _first_text(function, "name"),
            ))
    return result


def _terminal_assistant(message: Mapping[str, Any]) -> bool:
    if _role(message) != "assistant" or _assistant_has_tool_call(message):
        return False
    if message.get("isError") is True or message.get("error") or message.get("errorMessage"):
        return False
    stop_reason = _first_text(message, "stopReason", "stop_reason").casefold()
    if stop_reason and re.search(r"error|abort|cancel|tool|fail", stop_reason):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            (isinstance(block, str) and bool(block.strip()))
            or (
                isinstance(block, Mapping)
                and _type(block.get("type"))
                not in (_TOOL_CALL_TYPES | _TOOL_RESULT_TYPES)
                and isinstance(block.get("text"), str)
                and bool(block["text"].strip())
            )
            for block in content
        )
    return False


@dataclass(frozen=True)
class TurnLedgerRegistration:
    state: dict[str, Any]
    entry: dict[str, Any]
    messages: list[dict]
    source_ids: frozenset[str]
    changed: bool


@dataclass(frozen=True)
class TurnLedgerRestore:
    messages: list[dict]
    source_groups: tuple[frozenset[str], ...] = ()
    matched_turn_keys: tuple[str, ...] = ()


class ToolResultTurnLedger:
    """Build and restore server-issued strong identities for complete turns."""

    STATE_KEY = "tool_result_turn_ledger"
    SCHEMA_VERSION = 1
    MAX_COMPLETED_TURNS = 1_024

    def __init__(
        self,
        identity_key: bytes,
        *,
        normalizer: ToolResultNormalizer | None = None,
    ) -> None:
        if len(identity_key) < 32:
            raise ValueError("turn-ledger identity key must be at least 32 bytes")
        self.identity_key = bytes(identity_key)
        self.normalizer = normalizer or ToolResultNormalizer()

    @staticmethod
    def owner(ctx: RequestContext, session_id: str) -> dict[str, str]:
        return {
            "account_id": str(ctx.account_id),
            "user_id": str(ctx.user_id),
            "agent_id": str(ctx.agent_id),
            "session_id": str(session_id),
        }

    def _hmac_id(self, prefix: str, fields: list[Any]) -> str:
        digest = hmac.new(
            self.identity_key,
            _canonical(fields).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        return prefix + digest

    @staticmethod
    def _clean_source(message: Any) -> dict:
        copied = deepcopy(message) if isinstance(message, Mapping) else {
            "role": getattr(message, "role", ""),
            "content": deepcopy(getattr(message, "content", "")),
        }
        copied.pop("_ogmem_source", None)
        copied.pop("_ogmem_semantic_source_validated", None)
        copied.pop("_ogmem_source_type", None)
        copied.pop("_ogmem_source_attestation", None)
        return copied

    def _candidate(
        self,
        messages: list[Any],
        *,
        start_boundary: int,
        ctx: RequestContext,
        session_id: str,
    ) -> dict[str, Any] | None:
        if not messages:
            return None
        copied = [self._clean_source(message) for message in messages]
        if _role(copied[0]) != "user" or not _terminal_assistant(copied[-1]):
            return None
        if not any(_assistant_has_tool_call(message) for message in copied):
            return None
        normalized = self.normalizer.normalize(copied)
        if normalized[0].tool_results or normalized[-1].tool_results:
            return None
        if any(
            _role(copied[index]) == "user" and not normalized[index].tool_results
            for index in range(1, len(copied) - 1)
        ):
            # A second ordinary user prompt is a new outer Turn. User-role
            # Anthropic tool_result blocks remain allowed because the
            # normalizer classifies them as results rather than user text.
            return None

        declared_calls: dict[str, str] = {}
        result_calls: set[str] = set()
        source_result_calls: set[str] = set()
        results = []
        for message in normalized:
            for call_key, tool_name in _assistant_tool_calls(message.original):
                if not call_key or call_key in declared_calls:
                    return None
                declared_calls[call_key] = tool_name
            for part in message.tool_results:
                call_key = canonical_tool_call_id(part.tool_call_id)
                if not call_key or call_key not in declared_calls:
                    return None
                declared_name = declared_calls[call_key]
                if declared_name and part.tool_name and declared_name != part.tool_name:
                    return None
                if part.source_ref or part.source_collection:
                    source_result_calls.add(call_key)
                    continue
                result_calls.add(call_key)
                results.append({
                    "message_ordinal": message.message_index,
                    "tool_call_id": part.tool_call_id or "",
                    "tool_call_key": canonical_tool_call_id(part.tool_call_id),
                    "tool_name": part.tool_name or "",
                    "part_path": part.part_path,
                    "occurrence_index": part.occurrence_index,
                    "content_sha256": hashlib.sha256(
                        part.output_text.encode("utf-8")
                    ).hexdigest(),
                    "original_chars": part.original_chars,
                    "artifact_id": "",
                    "hot_seen_at": "",
                })
        if not results:
            return None
        if result_calls | source_result_calls != set(declared_calls):
            return None
        projections = [_digest(_message_projection(message)) for message in copied]
        owner = self.owner(ctx, session_id)
        key_payload = {
            "owner": owner,
            "start_boundary": max(0, int(start_boundary)),
            "message_projections": projections,
            "results": [
                {
                    key: row[key]
                    for key in (
                        "message_ordinal", "tool_call_key", "tool_name", "part_path",
                        "occurrence_index", "content_sha256", "original_chars",
                    )
                }
                for row in results
            ],
        }
        return {
            "turn_key": "ogk_" + _digest(key_payload),
            "start_boundary": max(0, int(start_boundary)),
            "message_projections": projections,
            "results": results,
            "messages": copied,
        }

    @staticmethod
    def _valid_result(row: Any, message_count: int) -> dict[str, Any] | None:
        if not isinstance(row, Mapping):
            return None
        try:
            ordinal = int(row.get("message_ordinal"))
            occurrence = int(row.get("occurrence_index", 0))
            chars = int(row.get("original_chars"))
        except (TypeError, ValueError):
            return None
        digest = str(row.get("content_sha256") or "")
        source_id = str(row.get("source_message_id") or "")
        if (
            ordinal < 0 or ordinal >= message_count or occurrence < 0 or chars < 0
            or not _HEX_64.fullmatch(digest)
            or not _MESSAGE_ID.fullmatch(source_id)
        ):
            return None
        artifact_id = str(row.get("artifact_id") or "")
        if artifact_id and not re.fullmatch(r"tr_[a-f0-9_]+", artifact_id):
            artifact_id = ""
        return {
            "message_ordinal": ordinal,
            "source_message_id": source_id,
            "tool_call_id": str(row.get("tool_call_id") or ""),
            "tool_call_key": str(row.get("tool_call_key") or ""),
            "tool_name": str(row.get("tool_name") or ""),
            "part_path": str(row.get("part_path") or ""),
            "occurrence_index": occurrence,
            "content_sha256": digest,
            "original_chars": chars,
            "artifact_id": artifact_id,
            "hot_seen_at": str(row.get("hot_seen_at") or ""),
        }

    def _valid_entry(
        self,
        row: Any,
        *,
        owner: Mapping[str, str],
    ) -> dict[str, Any] | None:
        if not isinstance(row, Mapping):
            return None
        try:
            sequence = int(row.get("turn_sequence"))
            start = int(row.get("start_boundary", 0))
        except (TypeError, ValueError):
            return None
        semantic_id = str(row.get("semantic_turn_id") or "")
        turn_key = str(row.get("turn_key") or "")
        projections = row.get("message_projections")
        message_sources = row.get("message_source_ids")
        raw_results = row.get("results")
        if (
            sequence < 1 or start < 0 or not _TURN_ID.fullmatch(semantic_id)
            or not re.fullmatch(r"ogk_[a-f0-9]{64}", turn_key)
            or not isinstance(projections, list) or not projections
            or not all(isinstance(item, str) and _HEX_64.fullmatch(item) for item in projections)
            or not isinstance(message_sources, list)
            or len(message_sources) != len(projections)
            or not all(isinstance(item, str) and _MESSAGE_ID.fullmatch(item) for item in message_sources)
            or len(set(message_sources)) != len(message_sources)
            or row.get("complete") is not True
            or not isinstance(raw_results, list) or not raw_results
        ):
            return None
        results = []
        result_positions: set[tuple[int, str, int]] = set()
        for item in raw_results:
            parsed = self._valid_result(item, len(projections))
            if parsed is None:
                # A partially damaged result list cannot safely attest the
                # complete ordered Turn. Reject the whole entry, not just the
                # malformed row, so no message in it gains strong identity.
                return None
            position = (
                parsed["message_ordinal"],
                parsed["part_path"],
                parsed["occurrence_index"],
            )
            if (
                parsed["source_message_id"]
                != message_sources[parsed["message_ordinal"]]
                or parsed["tool_call_key"]
                != canonical_tool_call_id(parsed["tool_call_id"])
                or position in result_positions
            ):
                return None
            result_positions.add(position)
            results.append(parsed)
        expected_turn_key = "ogk_" + _digest({
            "owner": dict(owner),
            "start_boundary": start,
            "message_projections": list(projections),
            "results": [
                {
                    key: result[key]
                    for key in (
                        "message_ordinal", "tool_call_key", "tool_name",
                        "part_path", "occurrence_index", "content_sha256",
                        "original_chars",
                    )
                }
                for result in results
            ],
        })
        if turn_key != expected_turn_key:
            return None
        return {
            "turn_sequence": sequence,
            "semantic_turn_id": semantic_id,
            "turn_key": turn_key,
            "start_boundary": start,
            "complete": True,
            "mode": "user_message",
            "governance_complete": row.get("governance_complete") is True,
            "message_projections": list(projections),
            "message_source_ids": list(message_sources),
            "results": results,
        }

    def normalized_state(
        self,
        raw_state: Any,
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> dict[str, Any]:
        expected_owner = self.owner(ctx, session_id)
        state_is_empty = raw_state is None or (
            isinstance(raw_state, Mapping) and not raw_state
        )
        if not state_is_empty and (
            not isinstance(raw_state, Mapping)
            or raw_state.get("schema_version") != self.SCHEMA_VERSION
            or raw_state.get("owner") != expected_owner
        ):
            return {
                "schema_version": self.SCHEMA_VERSION,
                "owner": expected_owner,
                "next_turn_sequence": 1,
                "active_turns": [],
                "completed_turns": [],
                "owner_mismatch": True,
            }
        rows = raw_state.get("completed_turns") if isinstance(raw_state, Mapping) else []
        parsed_entries = []
        raw_sequences: list[int] = []
        raw_sequence_corrupt = False
        if isinstance(rows, list):
            for row in rows:
                try:
                    raw_sequence = int(row.get("turn_sequence"))
                    if raw_sequence < 1:
                        raise ValueError("turn_sequence must be positive")
                    raw_sequences.append(raw_sequence)
                except (AttributeError, TypeError, ValueError):
                    raw_sequence_corrupt = True
                entry = self._valid_entry(row, owner=expected_owner)
                if entry is not None:
                    parsed_entries.append(entry)
        elif not state_is_empty:
            raw_sequence_corrupt = True
        key_counts: dict[str, int] = {}
        sequence_counts: dict[int, int] = {}
        semantic_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for entry in parsed_entries:
            key_counts[entry["turn_key"]] = key_counts.get(entry["turn_key"], 0) + 1
            sequence_counts[entry["turn_sequence"]] = (
                sequence_counts.get(entry["turn_sequence"], 0) + 1
            )
            semantic_counts[entry["semantic_turn_id"]] = (
                semantic_counts.get(entry["semantic_turn_id"], 0) + 1
            )
            for source_id in entry["message_source_ids"]:
                source_counts[source_id] = source_counts.get(source_id, 0) + 1
        valid = [
            entry
            for entry in parsed_entries
            if key_counts[entry["turn_key"]] == 1
            and sequence_counts[entry["turn_sequence"]] == 1
            and semantic_counts[entry["semantic_turn_id"]] == 1
            and all(source_counts[source_id] == 1 for source_id in entry["message_source_ids"])
        ]
        valid.sort(key=lambda item: item["turn_sequence"])
        valid = valid[-self.MAX_COMPLETED_TURNS:]
        allocation_unsafe = raw_sequence_corrupt
        try:
            requested_next = (
                int(raw_state.get("next_turn_sequence", 1))
                if isinstance(raw_state, Mapping)
                else 1
            )
            if requested_next < 1:
                raise ValueError("next_turn_sequence must be positive")
        except (TypeError, ValueError):
            requested_next = 1
            allocation_unsafe = not state_is_empty
        next_sequence = max(
            requested_next,
            max(raw_sequences, default=0) + 1,
            max((entry["turn_sequence"] for entry in valid), default=0) + 1,
            1,
        )
        normalized = {
            "schema_version": self.SCHEMA_VERSION,
            "owner": expected_owner,
            "next_turn_sequence": next_sequence,
            "active_turns": [],
            "completed_turns": valid,
        }
        if allocation_unsafe:
            normalized["allocation_unsafe"] = True
        return normalized

    @staticmethod
    def _attach_entry_sources(messages: list[dict], entry: Mapping[str, Any]) -> tuple[list[dict], frozenset[str]]:
        copied = [deepcopy(message) for message in messages]
        source_ids = frozenset(str(item) for item in entry["message_source_ids"])
        for ordinal, message in enumerate(copied):
            message["_ogmem_source"] = {
                "source_message_id": entry["message_source_ids"][ordinal],
                "turn_id": entry["semantic_turn_id"],
                "identity_strength": "strong",
                "server_turn_ledger_attested": True,
            }
            message["_ogmem_semantic_source_validated"] = True
        return copied, source_ids

    def register_user_message_turn(
        self,
        raw_state: Any,
        messages: list[Any],
        *,
        start_boundary: int,
        ctx: RequestContext,
        session_id: str,
    ) -> TurnLedgerRegistration | None:
        candidate = self._candidate(
            messages,
            start_boundary=start_boundary,
            ctx=ctx,
            session_id=session_id,
        )
        if candidate is None:
            return None
        state = self.normalized_state(raw_state, ctx=ctx, session_id=session_id)
        if (
            state.pop("owner_mismatch", False)
            or state.pop("allocation_unsafe", False)
        ):
            return None
        existing = next(
            (
                entry for entry in state["completed_turns"]
                if entry["turn_key"] == candidate["turn_key"]
            ),
            None,
        )
        changed = existing is None
        if existing is None:
            sequence = state["next_turn_sequence"]
            semantic_id = self._hmac_id("ogt_", [
                state["owner"], sequence, candidate["turn_key"],
            ])
            message_source_ids = [
                self._hmac_id("ogm_", [semantic_id, ordinal, candidate["turn_key"]])
                for ordinal in range(len(candidate["message_projections"]))
            ]
            results = []
            for result in candidate["results"]:
                results.append({
                    **result,
                    "source_message_id": message_source_ids[result["message_ordinal"]],
                })
            existing = {
                "turn_sequence": sequence,
                "semantic_turn_id": semantic_id,
                "turn_key": candidate["turn_key"],
                "start_boundary": candidate["start_boundary"],
                "complete": True,
                "mode": "user_message",
                "governance_complete": False,
                "message_projections": candidate["message_projections"],
                "message_source_ids": message_source_ids,
                "results": results,
            }
            state["completed_turns"].append(existing)
            state["completed_turns"] = state["completed_turns"][-self.MAX_COMPLETED_TURNS:]
            state["next_turn_sequence"] = sequence + 1
        attached, source_ids = self._attach_entry_sources(candidate["messages"], existing)
        return TurnLedgerRegistration(state, deepcopy(existing), attached, source_ids, changed)

    def restore_completed_turns(
        self,
        raw_state: Any,
        messages: list[Any],
        *,
        ctx: RequestContext,
        session_id: str,
    ) -> TurnLedgerRestore:
        copied = [
            deepcopy(message) if isinstance(message, Mapping) else {
                "role": getattr(message, "role", ""),
                "content": deepcopy(getattr(message, "content", "")),
            }
            for message in messages
        ]
        state = self.normalized_state(raw_state, ctx=ctx, session_id=session_id)
        if state.pop("owner_mismatch", False) or not state["completed_turns"]:
            return TurnLedgerRestore(copied)
        projections = [
            _digest(_message_projection(self._clean_source(message)))
            for message in copied
        ]
        matches: dict[str, list[int]] = {}
        for entry in state["completed_turns"]:
            needle = entry["message_projections"]
            if len(needle) > len(projections):
                continue
            starts = [
                start
                for start in range(len(projections) - len(needle) + 1)
                if projections[start:start + len(needle)] == needle
            ]
            if starts:
                matches[entry["turn_key"]] = starts

        candidates: list[tuple[Mapping[str, Any], int, int]] = []
        for entry in state["completed_turns"]:
            starts = matches.get(entry["turn_key"], [])
            if len(starts) != 1:
                continue
            start = starts[0]
            end = start + len(entry["message_projections"])
            # If another Ledger entry matches the same complete slice, identity
            # is ambiguous even when tool content hashes happen to be equal.
            competitors = sum(
                1
                for other in state["completed_turns"]
                if start in matches.get(other["turn_key"], [])
                and len(other["message_projections"]) == end - start
            )
            if competitors == 1:
                candidates.append((entry, start, end))

        # Reject every overlapping candidate symmetrically. Choosing the first
        # Ledger row would make identity restoration depend on persisted order
        # when a damaged/incorrect state contains nested complete projections.
        safe_candidates = [
            candidate
            for candidate in candidates
            if not any(
                candidate is not other
                and not (candidate[2] <= other[1] or candidate[1] >= other[2])
                for other in candidates
            )
        ]
        groups: list[frozenset[str]] = []
        matched_keys: list[str] = []
        for entry, start, end in sorted(safe_candidates, key=lambda item: item[1]):
            turn_messages = copied[start:end]
            attached, source_ids = self._attach_entry_sources(turn_messages, entry)
            copied[start:end] = attached
            groups.append(source_ids)
            matched_keys.append(entry["turn_key"])
        return TurnLedgerRestore(
            copied,
            tuple(groups),
            tuple(matched_keys),
        )

    def update_entry_results(
        self,
        raw_state: Any,
        *,
        turn_key: str,
        artifacts: Mapping[tuple[int, str, int], tuple[str, str]],
        ctx: RequestContext,
        session_id: str,
    ) -> dict[str, Any]:
        state = self.normalized_state(raw_state, ctx=ctx, session_id=session_id)
        if state.pop("owner_mismatch", False):
            return state
        for entry in state["completed_turns"]:
            if entry["turn_key"] != turn_key:
                continue
            for result in entry["results"]:
                value = artifacts.get((
                    result["message_ordinal"],
                    result["part_path"],
                    result["occurrence_index"],
                ))
                if value is None:
                    continue
                result["artifact_id"], result["hot_seen_at"] = value
            break
        return state

    def mark_governance_complete(
        self,
        raw_state: Any,
        *,
        turn_key: str,
        ctx: RequestContext,
        session_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Persist the successful ingest boundary for exact afterTurn retries."""
        state = self.normalized_state(raw_state, ctx=ctx, session_id=session_id)
        if state.pop("owner_mismatch", False):
            return state, False
        for entry in state["completed_turns"]:
            if entry["turn_key"] != turn_key:
                continue
            changed = entry.get("governance_complete") is not True
            entry["governance_complete"] = True
            return state, changed
        return state, False
