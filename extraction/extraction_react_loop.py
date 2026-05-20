"""ReAct loop for memory extraction with tool use.

Implements iterative tool-assisted extraction with safety checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any
from uuid import uuid4

from core.interfaces import ContextFS, LLM
from core.models import CandidateMemory, RequestContext
from core.react_loop import (
    ReactIterationTrace,
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
    ReactTrace,
)
from core.uri_resolver import URIResolver
from extraction.prefetch import MemoryPrefetcher, PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, parse_tool_call

logger = getLogger(__name__)


# Built-in tool definitions for reading memory nodes
_READ_TOOL = {
    "name": "read",
    "description": "Read a memory node by URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the memory node to read.",
            }
        },
        "required": ["uri"],
    },
}

_LIST_TOOL = {
    "name": "list",
    "description": "List child nodes under a directory URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the directory to list.",
            }
        },
        "required": ["uri"],
    },
}

_RELATIONS_TOOL = {
    "name": "get_relations",
    "description": "Get related nodes for a URI. Returns one-hop neighbors.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}

_ACCESS_STATS_TOOL = {
    "name": "get_access_stats",
    "description": "Get access stats for a node: last access time, 30d hit count.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}


@dataclass
class ExtractionIterationTrace(ReactIterationTrace):
    """Extraction-specific iteration trace with safety check fields."""
    safety_check_triggered: bool = False
    refetch_uris: list[str] = field(default_factory=list)


@dataclass
class ExtractionTaskContext(ReactTaskContext):
    """Task context for extraction ReAct loop."""
    conversation_text: str
    prefetch_result: PrefetchResult | None = None


@dataclass
class ExtractionTrace(ReactTrace):
    """Extraction-specific trace with candidate count."""
    final_candidate_count: int = 0


@dataclass
class ExtractionReActResult:
    """Result of a ReAct extraction loop."""
    candidates: list[CandidateMemory] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    iterations: int = 0
    read_uris: set[str] = field(default_factory=set)
    trace: ExtractionTrace | None = None


class ExtractionReActLoop(ReactLoop):
    """ReAct loop for memory extraction with tool use.

    Workflow:
    1. Start with system prompt + prefetch context + conversation
    2. While iteration < max_iterations:
       a. Call LLM with tools (read, list, extract_*)
       b. If tool calls: execute and continue
       c. If content: parse operations → safety check → return
       d. If neither: disable tools and continue
    3. Return extracted candidates with metadata

    Safety features:
    - Refetch check: re-read existing files before write operations
    - URI validation: ensure all URIs are within allowed namespace
    - Single safety re-read: prevent infinite loops
    """

    def __init__(
        self,
        llm: LLM,
        fs: ContextFS,
        registry: SchemaRegistry,
        uri_resolver: URIResolver,
        prefetcher: MemoryPrefetcher | None = None,
        max_iterations: int = 3,
        timeout_seconds: float = 30.0,
        internal_tool_usage_tracker: Any | None = None,
    ):
        super().__init__(llm, max_iterations, timeout_seconds)
        self._fs = fs
        self._registry = registry
        self._uri_resolver = uri_resolver
        self._prefetcher = prefetcher
        self._internal_tool_usage_tracker = internal_tool_usage_tracker

        self._extraction_tools = build_extraction_tools(registry)

        # All available tools: read, list, relations, access_stats, and extraction tools
        self._all_tools = [_READ_TOOL, _LIST_TOOL, _RELATIONS_TOOL, _ACCESS_STATS_TOOL] + self._extraction_tools

        self._did_safety_reread: bool = False

    def run(
        self,
        conversation_text: str,
        ctx: RequestContext,
        prefetch_result: PrefetchResult | None = None,
    ) -> ExtractionReActResult:
        # Reset runtime state
        self._read_uris = set()
        self._did_safety_reread = False

        # Track prefetch URIs if provided
        if prefetch_result:
            self._read_uris.update(prefetch_result.read_uris)

        task_ctx = ExtractionTaskContext(
            conversation_text=conversation_text,
            prefetch_result=prefetch_result,
        )

        loop_result = super().run(ctx, task_ctx)
        return self._to_react_result(loop_result, ctx)

    def _get_tools(self) -> list[dict]:
        return self._all_tools

    def _create_iter_trace(
        self,
        iteration: int,
        latency_seconds: float,
        tool_calls_count: int = 0,
        tool_call_names: list[str] | None = None,
        content_length: int = 0,
    ) -> ExtractionIterationTrace:
        return ExtractionIterationTrace(
            iteration=iteration,
            latency_seconds=latency_seconds,
            tool_calls_count=tool_calls_count,
            tool_call_names=tool_call_names or [],
            content_length=content_length,
        )

    def _build_messages(
        self,
        ctx: RequestContext,
        task_ctx: ReactTaskContext,
    ) -> list[dict]:
        conversation_text = task_ctx.conversation_text
        prefetch_result = task_ctx.prefetch_result

        messages = []

        # System prompt with instruction
        system_prompt = self._build_system_prompt(ctx)
        messages.append({"role": "system", "content": system_prompt})

        # Add prefetch context if available
        if prefetch_result and prefetch_result.messages:
            for msg in prefetch_result.messages:
                messages.append({"role": "system", "content": msg})

        # Add conversation text
        messages.append({
            "role": "user",
            "content": f"## Conversation\n\n{conversation_text}",
        })

        return messages

    def _build_system_prompt(self, ctx: RequestContext) -> str:
        tool_descriptions = self._build_tool_descriptions()

        prompt = f"""# Memory Extraction Task

You are a memory extraction assistant. Your task is to extract structured memories from the conversation.

## Available Tools

{chr(10).join(tool_descriptions)}

## Instructions

1. Use `read` and `list` tools to explore existing memories before making decisions.
2. Use `extract_*` tools to create new memories or update existing ones.
3. Always read existing content before deciding to merge or create new entries.
4. **If no existing memories found (read/list returns empty or error), proceed to create new memories directly using extract_* tools. Do NOT wait or loop endlessly.**
5. Return your final result using the extraction tool calls in your response.

## Context

Account: {ctx.account_id}
User: {ctx.user_id}
Agent: {ctx.agent_id}
"""
        return prompt

    def _execute_tool(
        self,
        name: str,
        input: dict,
        ctx: RequestContext,
    ) -> dict | str:
        try:
            if name == "read":
                uri = input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                node = self._fs.read_node(uri, ctx)
                return {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

            elif name == "list":
                uri = input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                children = self._fs.list_children(uri, ctx)
                return {
                    "uri": uri,
                    "children": children,
                }

            elif name == "get_relations":
                uri = input.get("uri", "")
                return self._execute_get_relations(uri, ctx)

            elif name == "get_access_stats":
                uri = input.get("uri", "")
                return self._execute_get_stats(uri, ctx)

            # Extraction tools are parsed later, just record status
            elif name.startswith("extract_"):
                return {"status": "recorded"}

            else:
                return {"error": f"Unknown tool: {name}"}

        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"error": str(e)}

    def _to_candidate(self, tool_name: str, tool_input: dict) -> CandidateMemory | None:
        """Convert extract_* tool call to CandidateMemory.

        Unified parsing used by both _parse_output and _to_react_result.
        """
        if not tool_name.startswith("extract_"):
            return None
        if "raw" in tool_input and len(tool_input) == 1:
            logger.warning(
                "Skipping tool call with unparsed raw arguments: tool=%s raw=%.200s",
                tool_name, tool_input["raw"],
            )
            return None
        parsed = parse_tool_call(tool_name, tool_input, self._registry)
        if parsed:
            memory_type, owner_scope, candidate = parsed
            logger.info(f"Parsed candidate from {tool_name}: {memory_type}/{candidate.routing_key}")
            return candidate
        return None

    def _parse_output(self, content: str) -> list[Any] | None:
        try:
            data = json.loads(content)

            candidates = []

            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Check for tool_calls key
                if "tool_calls" in data:
                    items = data["tool_calls"]
                # Check for operations key
                elif "operations" in data:
                    items = data["operations"]
                else:
                    # Treat dict itself as single operation
                    items = [data]
            else:
                logger.warning(f"Unexpected JSON structure: {type(data)}")
                return None

            for item in items:
                if not isinstance(item, dict):
                    continue

                tool_name = item.get("name", "")
                tool_input = item.get("input", item)

                if not tool_name.startswith("extract_"):
                    # Maybe input has the tool name
                    for key in item.keys():
                        if key.startswith("extract_"):
                            tool_name = key
                            tool_input = item[key]
                            break

                # Use unified _to_candidate method
                candidate = self._to_candidate(tool_name, tool_input)
                if candidate:
                    candidates.append(candidate)

            return candidates if candidates else None

        except json.JSONDecodeError:
            logger.warning("Failed to parse content as JSON")
            return None
        except Exception as e:
            logger.error(f"Error parsing operations: {e}")
            return None

    def _should_continue_after_output(
        self,
        parsed: list[Any],
        messages: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        if self._did_safety_reread:
            return False

        candidates = parsed
        if not candidates:
            return False

        # Safety check: refetch unread existing files
        refetch_uris = self._check_unread_existing_files(candidates, ctx)
        if refetch_uris:
            logger.info(f"Found unread existing files: {refetch_uris}, refetching...")
            self._add_refetch_results(messages, refetch_uris, ctx)
            if iter_trace and isinstance(iter_trace, ExtractionIterationTrace):
                iter_trace.safety_check_triggered = True
                iter_trace.refetch_uris = refetch_uris
            return True

        return False

    def _track_tool_usage(
        self,
        tool_name: str,
        tool_input: dict,
        result: dict | str,
        tools_used: list[dict],
        read_uris: set[str],
    ) -> None:
        super()._track_tool_usage(tool_name, tool_input, result, tools_used, read_uris)

        # Track read operations
        if tool_name == "read" and tool_input.get("uri"):
            uri = tool_input["uri"]
            self._read_uris.add(uri)
            read_uris.add(uri)

    def _check_unread_existing_files(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        if self._did_safety_reread:
            return []

        refetch_uris = []

        for candidate in candidates:
            schema = self._registry.get(candidate.category)
            if schema is None:
                continue

            # Skip add-only schemas (no conflict risk)
            if schema.is_add_only:
                continue

            fields = self._candidate_to_fields(candidate)
            try:
                uri = self._uri_resolver.resolve(candidate.category, fields, ctx)
            except Exception as e:
                logger.warning(f"Failed to resolve URI for {candidate.category}: {e}")
                continue

            # Check if URI was already read
            if uri in self._read_uris:
                continue

            # Check if file exists
            if self._fs.exists(uri, ctx):
                refetch_uris.append(uri)

        return refetch_uris

    def _add_refetch_results(
        self, messages: list[dict], refetch_uris: list[str], ctx: RequestContext
    ) -> None:
        self._did_safety_reread = True

        for uri in refetch_uris:
            try:
                node = self._fs.read_node(uri, ctx)
                self._read_uris.add(uri)

                result = {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

                # Add as read tool result
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"refetch_{uri}",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"uri": uri}),
                        },
                    }],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"refetch_{uri}",
                    "content": json.dumps(result),
                })

            except Exception as e:
                logger.warning(f"Failed to refetch {uri}: {e}")

        # Add safety reminder
        messages.append({
            "role": "user",
            "content": (
                "SAFETY REMINDER: The files above were automatically read because they "
                "exist and you didn't read them before deciding to write. Please consider "
                "the existing content when making write decisions. You can now output "
                "updated operations."
            ),
        })

    def _validate_operations_uris(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        errors = []

        for candidate in candidates:
            fields = self._candidate_to_fields(candidate)
            try:
                uri = self._uri_resolver.resolve(candidate.category, fields, ctx)
                if not self._uri_resolver.validate_uri(uri, ctx):
                    errors.append(
                        f"Invalid URI for {candidate.category}/{candidate.routing_key}: {uri}"
                    )
            except Exception as e:
                errors.append(
                    f"Failed to validate URI for {candidate.category}/{candidate.routing_key}: {e}"
                )

        return errors

    def _candidate_to_fields(self, candidate: CandidateMemory) -> dict:
        # Get the routing key field name for this category
        fields = {
            "routing_key": candidate.routing_key,
            "abstract": candidate.abstract,
            "overview": candidate.overview,
            "content": candidate.content,
            "confidence": candidate.confidence,
        }

        if candidate.when:
            fields["when"] = candidate.when
        if candidate.who:
            fields["who"] = candidate.who
        if candidate.where:
            fields["where"] = candidate.where
        # Add optional fields if present
        if candidate.tool_stats:
            fields.update(candidate.tool_stats)

        return fields

    def _execute_get_relations(self, uri: str, ctx: RequestContext) -> dict:
        try:
            if hasattr(self._fs, 'get_relations'):
                edges = self._fs.get_relations(uri, ctx)
                return {"uri": uri, "relations": [
                    {"target": e.target_uri, "weight": e.weight, "reason": e.reason}
                    for e in edges
                ]}
            return {"uri": uri, "relations": [], "note": "Not available"}
        except Exception as e:
            return {"error": str(e)}

    def _execute_get_stats(self, uri: str, ctx: RequestContext) -> dict:
        try:
            node = self._fs.read_node(uri, ctx)
            metadata = node.metadata or {}
            return {
                "uri": uri,
                "last_accessed_at": metadata.get("last_accessed_at"),
                "hit_count_30d": metadata.get("hit_count_30d", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _to_react_result(self, loop_result: ReactLoopResult, ctx: RequestContext) -> ExtractionReActResult:
        trace = None
        if loop_result.trace:
            trace = ExtractionTrace(
                total_iterations=loop_result.trace.total_iterations,
                total_latency_seconds=loop_result.trace.total_latency_seconds,
                iterations=loop_result.trace.iterations,
                total_read_uris=len(self._read_uris),
                final_candidate_count=len(loop_result.output),
            )

        # Collect candidates from parsed output (content JSON)
        candidates = [item for item in loop_result.output if isinstance(item, CandidateMemory)]

        # Also extract candidates from tools_used (extract_* tool calls)
        # This handles cases where LLM used tool calling instead of outputting JSON
        for tool_record in loop_result.tools_used:
            tool_name = tool_record.get("tool_name", "")
            tool_params = tool_record.get("params", {})
            tool_result = tool_record.get("result", {})
            # Use params (original input) with unified _to_candidate method
            if tool_result.get("status") == "recorded":
                candidate = self._to_candidate(tool_name, tool_params)
                if candidate:
                    candidates.append(candidate)

        # Validate URIs before returning
        errors = self._validate_operations_uris(candidates, ctx)
        if errors:
            logger.error(f"URI validation errors: {errors}")

        return ExtractionReActResult(
            candidates=candidates,
            tools_used=loop_result.tools_used,
            iterations=loop_result.iterations,
            read_uris=self._read_uris.copy(),
            trace=trace,
        )

    # --- Internal tool usage tracking (from upstream) ---
    # These hook into the base class _execute_and_append_tool_calls flow
    # via _record_round / _record_tool_call overrides.

    def _record_round(self, tool_calls: list[dict], ctx: RequestContext) -> str:
        round_id = _round_id_from_tool_calls(tool_calls)
        if not self._internal_tool_usage_tracker:
            return round_id
        usage = _usage_from_tool_calls(tool_calls)
        self._internal_tool_usage_tracker.record_round(
            account_id=ctx.account_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            pipeline="extraction.lazy",
            round_id=round_id,
            tool_names=[
                call.get("name") or call.get("tool", "")
                for call in tool_calls
                if call.get("name") or call.get("tool")
            ],
            model=usage.get("model", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read=usage.get("cache_read", 0),
            cache_write=usage.get("cache_write", 0),
        )
        return round_id

    def _record_tool_call(
        self,
        *,
        ctx: RequestContext,
        round_id: str,
        tool_name: str,
        status: str,
        duration_ms: float,
        error_type: str = "",
    ) -> None:
        if not self._internal_tool_usage_tracker or not tool_name:
            return
        self._internal_tool_usage_tracker.record_tool_call(
            account_id=ctx.account_id,
            session_id=ctx.session_id,
            pipeline="extraction.lazy",
            round_id=round_id,
            tool_name=tool_name,
            status=status,
            duration_ms=duration_ms,
            error_type=error_type,
        )


def _usage_from_tool_calls(tool_calls: list[dict]) -> dict:
    for call in tool_calls or []:
        usage = call.get("_llm_usage")
        if isinstance(usage, dict):
            return usage
    return {}


def _round_id_from_tool_calls(tool_calls: list[dict]) -> str:
    usage = _usage_from_tool_calls(tool_calls)
    if usage.get("round_id"):
        return str(usage["round_id"])
    for call in tool_calls or []:
        if call.get("id"):
            return str(call["id"])
    return f"llm_{uuid4().hex[:12]}"