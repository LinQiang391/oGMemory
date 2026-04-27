"""Two-phase candidate extraction.

Phase 1 — Span Identification: Cheap JSON-mode LLM call identifies which
user message ranges contain extractable information.

Phase 2 — Span Structuring: For each span, focused tool-call extraction
with +/- 3 turns context.  Parallelized across spans (max_workers=4).

Falls back to single-span mode when Phase 1 fails or returns no spans,
preserving backward compatibility with mock / test scenarios.
"""

from __future__ import annotations

import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.interfaces import LLM
from core.models import RequestContext, CandidateMemory

from extraction.tool_schemas import (
    get_tool_definitions,
    get_tool_to_category,
    parse_tool_call as _parse_tool_call_legacy,
)
from extraction.tool_builder import (
    build_extraction_tools,
    build_tool_to_category,
    parse_tool_call,
)
from extraction.prefetch import MemoryPrefetcher, PrefetchResult

logger = logging.getLogger(__name__)

# Tool definitions for LLM function calling (generated from Pydantic models)
EXTRACTION_TOOLS = get_tool_definitions()

# Map tool name to (category, owner_scope) (generated from Pydantic models)
TOOL_TO_CATEGORY = get_tool_to_category()


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """A contiguous message range identified as containing extractable info."""
    start: int
    end: int
    reason: str = ""
    categories: list[str] = field(default_factory=list)


@dataclass
class _PreparedSpan:
    """A span with its final Phase 2 prompt prepared for tool-calling."""
    span: _Span
    prompt: str
    prompt_sha: str


# JSON schema for Phase 1 span identification
_SPAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "reason": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end"],
            },
        },
    },
    "required": ["spans"],
}


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()


def _short_sha(sha: str) -> str:
    return sha[:12]


def _merge_unique_categories(primary: list[str], extra: list[str]) -> list[str]:
    """Merge categories in order so deduped prompts keep all prefetch scopes."""
    merged: list[str] = []
    seen: set[str] = set()
    for category in [*primary, *extra]:
        if not category or category in seen:
            continue
        merged.append(category)
        seen.add(category)
    return merged


# Phase 1 prompt — lightweight, no tool definitions
_SPAN_PROMPT = """\
Analyze this conversation and identify contiguous message ranges (spans) \
that contain personal information worth remembering.

Rules:
- ONLY identify spans from non-assistant messages. Ignore lines starting \
with "[N] assistant:".
- Each span should contain a coherent unit of extractable information.
- Merge adjacent user messages about the same topic into one span.
- Skip pure greetings, acknowledgments, or general knowledge.
- If a "Previously Extracted Context" section is present, DO NOT re-identify \
content already listed there.

For each span, specify:
- start: message index (the [N] prefix number)
- end: message index (inclusive)
- reason: what extractable information this span contains
- categories: likely categories from: profile, preference, entity, event, \
case, pattern, skill, tool

{session_summary_block}\
Conversation:
{conversation}

Return JSON: {{"spans": [{{"start": int, "end": int, "reason": str, \
"categories": [str]}}]}}
If nothing is extractable: {{"spans": []}}"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor:
    """Two-phase candidate memory extractor.

    Phase 1 uses a cheap JSON-mode LLM call to identify which user message
    ranges contain extractable information.  Phase 2 runs focused tool-call
    extraction per span with surrounding context, parallelized across spans.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_manager: Any | None = None,
        schema_registry: Any | None = None,
        fs: Any | None = None,
        vector_index: Any | None = None,
        embedder: Any | None = None,
        uri_resolver: Any | None = None,
        mode: str = "eager",
        react_timeout_seconds: float = 30.0,
    ):
        """Initialize the Extractor.

        Args:
            llm: LLM instance with complete_with_tools() and complete_json().
            prompt_manager: Optional PromptManager for template-based prompts.
                          If None, auto-creates one from default templates dir.
            schema_registry: Optional SchemaRegistry for dynamic tool generation.
                           If provided, uses build_extraction_tools() and
                           build_tool_to_category() instead of hardcoded tool_schemas.py.
            fs: Optional ContextFS for prefetching existing memories.
            vector_index: Optional VectorIndex for prefetching existing memories.
            embedder: Optional Embedder for prefetching existing memories.
            uri_resolver: Optional URIResolver for prefetching existing memories.
            mode: Extraction mode — "eager" (prefetch + single-shot) or
                  "lazy" (ReAct loop with read tools). Defaults to "eager".
            react_timeout_seconds: Maximum seconds for ReAct loop before timeout. Defaults to 30.0.
        """
        self._llm = llm
        if prompt_manager is None:
            from extraction.prompts import PromptManager as _PM
            prompt_manager = _PM()
        self._prompt_manager = prompt_manager
        self._schema_registry = schema_registry
        self._fs = fs
        self._vector_index = vector_index
        self._embedder = embedder
        self._uri_resolver = uri_resolver
        self._mode = mode
        self._react_timeout_seconds = react_timeout_seconds
        self._prefetcher: Optional[MemoryPrefetcher] = None
        if fs and vector_index and embedder and uri_resolver and schema_registry:
            self._prefetcher = MemoryPrefetcher(fs, vector_index, embedder, schema_registry, uri_resolver)
            logger.info(f"Extractor: prefetcher enabled ({mode} mode)")

        # Use schema-driven tools if registry provided, otherwise fallback to legacy
        if schema_registry is not None:
            self._extraction_tools = build_extraction_tools(schema_registry)
            self._tool_to_category = build_tool_to_category(schema_registry)
            self._parse_tool_call_fn = lambda name, inp: parse_tool_call(name, inp, schema_registry)
            logger.info(f"Extractor using SchemaRegistry with {len(self._extraction_tools)} tools")
        else:
            self._extraction_tools = EXTRACTION_TOOLS
            self._tool_to_category = TOOL_TO_CATEGORY
            self._parse_tool_call_fn = _parse_tool_call_legacy
            logger.info("Extractor using legacy hardcoded tool_schemas.py")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> list[CandidateMemory]:
        """Extract candidate memories from conversation messages.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # No user content to extract from
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") != "assistant"
        ]
        if not user_indices:
            return []

        # Detect language once
        language = self._detect_language(messages)

        # Phase 1: identify spans worth extracting
        spans = self._identify_spans(messages, session_time, session_summary)
        if not spans:
            user_content_len = sum(
                len(m.get("content", "") or "")
                for m in messages if m.get("role") != "assistant"
            )
            _MIN_FALLBACK_LEN = 200
            if user_content_len >= _MIN_FALLBACK_LEN:
                spans = [_Span(
                    start=0,
                    end=len(messages) - 1,
                    reason="fallback: Phase 1 returned no spans",
                    categories=[],
                )]
                logger.info(
                    "Phase 1 empty → fallback full-conversation span [0, %d] "
                    "(user_content_len=%d)",
                    len(messages) - 1, user_content_len,
                )
            else:
                logger.info(
                    "Phase 1 empty, user content too short (%d chars < %d) → skip Phase 2",
                    user_content_len, _MIN_FALLBACK_LEN,
                )
                return []

        # Phase 2: structure each span
        candidates = self._structure_spans(
            spans, messages, language, session_time, session_summary, tool_stats_text, ctx,
        )

        filtered = [c for c in candidates if c.confidence >= 0.5]
        dropped = len(candidates) - len(filtered)
        if dropped > 0:
            for c in candidates:
                if c.confidence < 0.5:
                    logger.info(
                        "Dropped candidate (confidence=%.2f < 0.5): category=%s routing_key=%s abstract=%.80s",
                        c.confidence, c.category, c.routing_key, c.abstract,
                    )
            logger.info(
                "Phase 2 confidence filter: %d kept, %d dropped (< 0.5)",
                len(filtered), dropped,
            )
        return filtered

    # ------------------------------------------------------------------
    # Phase 1 — Span Identification
    # ------------------------------------------------------------------

    def _identify_spans(
        self,
        messages: list[dict],
        session_time: Any | None,
        session_summary: str,
    ) -> list[_Span]:
        """Use a cheap JSON-mode call to find extractable message ranges."""
        conversation = self._format_indexed_conversation(messages, session_time)

        summary_block = ""
        if session_summary:
            summary_block = (
                "Previously Extracted Context "
                "(DO NOT re-identify these — they are already saved):\n"
                f"{session_summary}\n\n"
            )

        prompt = _SPAN_PROMPT.format(
            conversation=conversation,
            session_summary_block=summary_block,
        )

        try:
            result = self._llm.complete_json(prompt, schema=_SPAN_SCHEMA)
            spans: list[_Span] = []
            n = len(messages)
            for s in result.get("spans", []):
                if n <= 1:
                    # Single message: clamp everything to [0, 0]
                    span = _Span(
                        start=0, end=0,
                        reason=s.get("reason", ""),
                        categories=s.get("categories", []),
                    )
                else:
                    span = _Span(
                        start=max(0, int(s.get("start", 0))),
                        end=min(n - 1, int(s.get("end", 0))),
                        reason=s.get("reason", ""),
                        categories=s.get("categories", []),
                    )
                if span.start <= span.end:
                    spans.append(span)
            if not spans:
                logger.warning("Phase 1 returned 0 valid spans for %d messages", len(messages))
            return spans
        except Exception as exc:
            logger.error("Phase 1 span identification FAILED: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Phase 2 — Span Structuring
    # ------------------------------------------------------------------

    def _structure_spans(
        self,
        spans: list[_Span],
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Run focused tool-call extraction for each span."""
        prepared_spans = self._prepare_unique_spans(
            spans, messages, language,
            session_time, session_summary, tool_stats_text, ctx,
        )
        if not prepared_spans:
            return []

        # Single span → skip thread pool overhead
        if len(prepared_spans) == 1:
            return self._structure_prepared_span(prepared_spans[0], ctx)

        all_candidates: list[CandidateMemory] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    self._structure_prepared_span, prepared, ctx,
                ): prepared.span
                for prepared in prepared_spans
            }
            for future in as_completed(futures):
                span = futures[future]
                try:
                    all_candidates.extend(future.result())
                except Exception as exc:
                    logger.error(
                        "Span %d-%d structuring FAILED: %s",
                        span.start, span.end, exc, exc_info=True,
                    )
                    raise
        return all_candidates

    def _prepare_unique_spans(
        self,
        spans: list[_Span],
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> list[_PreparedSpan]:
        """Build final Phase 2 prompts and drop duplicate prompt_sha work."""
        prepared_spans: list[_PreparedSpan] = []
        seen: dict[str, _PreparedSpan] = {}
        skipped = 0

        for span in spans:
            prepared = self._prepare_span(
                span, messages, language,
                session_time, session_summary, tool_stats_text, ctx,
            )
            existing = seen.get(prepared.prompt_sha)
            if existing is not None:
                if existing.prompt == prepared.prompt:
                    original_categories = list(existing.span.categories)
                    existing.span.categories = _merge_unique_categories(
                        original_categories,
                        prepared.span.categories,
                    )
                    skipped += 1
                    logger.info(
                        "Phase 2 duplicate prompt skipped: session=%s trace=%s "
                        "prompt_sha=%s duplicate_span=[%d-%d] original_span=[%d-%d] "
                        "duplicate_categories=%s original_categories=%s merged_categories=%s",
                        getattr(ctx, "session_id", ""),
                        getattr(ctx, "trace_id", ""),
                        _short_sha(prepared.prompt_sha),
                        prepared.span.start,
                        prepared.span.end,
                        existing.span.start,
                        existing.span.end,
                        prepared.span.categories,
                        original_categories,
                        existing.span.categories,
                    )
                    continue
                logger.warning(
                    "Phase 2 prompt_sha collision detected; keeping both spans: "
                    "session=%s trace=%s prompt_sha=%s span=[%d-%d] existing_span=[%d-%d]",
                    getattr(ctx, "session_id", ""),
                    getattr(ctx, "trace_id", ""),
                    _short_sha(prepared.prompt_sha),
                    prepared.span.start,
                    prepared.span.end,
                    existing.span.start,
                    existing.span.end,
                )

            seen[prepared.prompt_sha] = prepared
            prepared_spans.append(prepared)

        if skipped:
            logger.info(
                "Phase 2 prompt dedupe: session=%s trace=%s spans=%d unique=%d skipped=%d",
                getattr(ctx, "session_id", ""),
                getattr(ctx, "trace_id", ""),
                len(spans),
                len(prepared_spans),
                skipped,
            )
        return prepared_spans

    def _prepare_span(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> _PreparedSpan:
        """Prepare the final prompt used for one Phase 2 extraction request."""
        prompt = self._build_span_prompt(
            span, messages, language,
            session_time, session_summary, tool_stats_text, ctx,
        )
        if not self._use_lazy_mode():
            prompt = self._with_prefetch_context(span, prompt, ctx)
        prompt_sha = _prompt_sha(prompt)
        return _PreparedSpan(span=span, prompt=prompt, prompt_sha=prompt_sha)

    def _use_lazy_mode(self) -> bool:
        """Return whether this extractor can run the lazy ReAct path."""
        return bool(self._mode == "lazy" and self._fs and self._schema_registry and self._uri_resolver)

    def _structure_prepared_span(
        self,
        prepared: _PreparedSpan,
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Extract structured candidates from an already prepared span prompt."""
        span = prepared.span
        if self._use_lazy_mode():
            return self._structure_span_lazy(span, prepared.prompt, ctx)
        return self._structure_span_eager(span, prepared.prompt, ctx, prepared.prompt_sha)

    def _structure_span(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Extract structured candidates from a single span."""
        prepared = self._prepare_span(
            span, messages, language,
            session_time, session_summary, tool_stats_text, ctx,
        )
        return self._structure_prepared_span(prepared, ctx)

    def _build_span_prompt(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> str:
        """Build the base Phase 2 prompt for a span before eager prefetch."""
        # Focused context: +/- 3 turns around the span
        ctx_start = max(0, span.start - 3)
        ctx_end = min(len(messages) - 1, span.end + 3)
        focused = messages[ctx_start: ctx_end + 1]

        conversation = self._format_conversation(
            focused, session_time,
            session_summary="",
            tool_stats_text=(
                tool_stats_text if span.end >= len(messages) - 3 else ""
            ),
        )
        return self._build_prompt(language, conversation, session_summary, ctx=ctx)

    def _with_prefetch_context(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
    ) -> str:
        """Inject eager prefetch context into a Phase 2 prompt when available."""
        if self._prefetcher is None:
            return prompt
        prefetch_result = self._collect_prefetch_context(span.categories, ctx)
        if not prefetch_result or not prefetch_result.messages:
            return prompt
        prefetch_block = (
            "== Existing Memories (DO NOT re-extract these — update or extend them) ==\n"
            + "\n".join(prefetch_result.messages)
            + "\n==\n"
        )
        return prefetch_block + "\n" + prompt

    def _structure_span_eager(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
        prompt_sha: str,
    ) -> list[CandidateMemory]:
        """Eager mode: prefetch + single-shot extraction."""
        tool_calls = self._llm.complete_with_tools(
            prompt=prompt,
            tools=self._extraction_tools,
            tool_choice="required",
        )

        if not tool_calls:
            logger.warning(
                "Phase 2 zero tool_calls: session=%s trace=%s span=[%d-%d] "
                "categories=%s prompt_len=%d prompt_sha=%s",
                getattr(ctx, "session_id", ""),
                getattr(ctx, "trace_id", ""),
                span.start,
                span.end,
                span.categories,
                len(prompt),
                _short_sha(prompt_sha),
            )

        candidates: list[CandidateMemory] = []
        for call in tool_calls:
            c = self._to_candidate(call)
            if c is not None:
                candidates.append(c)
        return candidates

    def _structure_span_lazy(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
    ) -> list[CandidateMemory]:
        """Lazy mode: ReAct loop with read tools, fallback to eager on failure."""
        try:
            from extraction.react_loop import ExtractionReActLoop

            # Collect prefetch result (URI list only, no full reads in lazy mode)
            prefetch_result = None
            if self._prefetcher is not None:
                prefetch_result = self._collect_prefetch_context(span.categories, ctx)

            loop = ExtractionReActLoop(
                llm=self._llm,
                fs=self._fs,
                registry=self._schema_registry,
                uri_resolver=self._uri_resolver,
                prefetcher=self._prefetcher,
                timeout_seconds=self._react_timeout_seconds,
            )
            result = loop.run(prompt, ctx, prefetch_result)
            logger.info(
                f"ReAct loop completed: {len(result.candidates)} candidates, "
                f"{result.iterations} iterations, {len(result.read_uris)} reads"
            )
            return result.candidates
        except Exception as e:
            logger.error(f"ReAct loop failed, falling back to eager mode: {e}")
            prompt = self._with_prefetch_context(span, prompt, ctx)
            prompt_sha = _prompt_sha(prompt)
            return self._structure_span_eager(span, prompt, ctx, prompt_sha)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _collect_prefetch_context(
        self,
        categories: list[str],
        ctx: RequestContext,
    ) -> PrefetchResult | None:
        """Collect prefetch context for given categories."""
        if self._prefetcher is None:
            return None
        try:
            return self._prefetcher.prefetch_for_span(categories, ctx)
        except Exception as e:
            logger.error(f"Prefetch failed, continuing without: {e}")
            return None

    def _build_prompt(
        self,
        language: str,
        conversation: str,
        session_summary: str = "",
        ctx: RequestContext | None = None,
    ) -> str:
        """Build extraction prompt from YAML template."""
        system_prompt = self._prompt_manager.render(
            "extraction", "system_prompt",
            session_summary=session_summary,
        )
        examples = self._prompt_manager.render("extraction", "examples")
        output_instruction = self._prompt_manager.render(
            "extraction", "output_instruction",
            output_language=language,
        )

        # Identity anchoring: tell the LLM who the user is
        identity_block = ""
        if ctx is not None:
            identity_block = (
                f"\n\n== IDENTITY ANCHOR ==\n"
                f"The user (the human you serve) is identified as '{ctx.user_id}'.\n"
                f"In the conversation, this user may appear as:\n"
                f"  - A first-person 'I' / 'me' in direct dialogue\n"
                f"  - Their name in a [Name]: prefix if it matches '{ctx.user_id}'\n"
                f"  - Any alias or nickname that resolves to '{ctx.user_id}'\n"
                f"Profile facts MUST be BY this user AND ABOUT this user.\n"
                f"Statements by OTHER named speakers → extract_entity, NOT extract_profile.\n"
                f"If unsure whether a speaker is the user → extract_entity.\n==\n"
            )

        return (
            f"{system_prompt}{identity_block}\n\n{examples}\n\n"
            f"{output_instruction}\n\nConversation:\n{conversation}"
        )

    @staticmethod
    def _content_to_str(content) -> str:
        """Normalize message content to plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def _detect_language(self, messages: list[dict]) -> str:
        """Detect language from user messages."""
        from core.language import detect_language

        user_text = " ".join(
            self._content_to_str(m.get("content", ""))
            for m in messages
            if m.get("role") == "user"
        )
        return detect_language(user_text)

    def _format_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> str:
        """Format messages into conversation text with session time header."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"**Session Time:** {time_str} ({day_of_week})"]
        lines.append(
            "Relative times (e.g., 'last week', 'next month') are based "
            "on Session Time, not today."
        )

        if session_summary:
            lines.append("")
            lines.append(
                "## Previously Extracted Context "
                "(DO NOT re-extract these — they are already saved)"
            )
            lines.append(session_summary)

        if tool_stats_text:
            lines.append("")
            lines.append(tool_stats_text)

        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(
                    f"assistant [do not extract from this]: {content}"
                )
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_indexed_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
    ) -> str:
        """Format messages with [N] index prefixes for span identification."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"Session Time: {time_str} ({day_of_week})"]
        lines.append("")

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(f"[{i}] assistant: {content}")
            else:
                lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)

    def _to_candidate(self, tool_call: dict) -> CandidateMemory | None:
        """Convert tool call result to CandidateMemory."""
        tool_name = tool_call.get("tool", "")
        input_data = tool_call.get("input", {})
        if "raw" in input_data and len(input_data) == 1:
            logger.warning(
                "Skipping tool call with unparsed raw arguments: tool=%s raw=%.200s",
                tool_name, input_data["raw"],
            )
            return None
        result = self._parse_tool_call_fn(tool_name, input_data)
        if result is None:
            return None
        _, _, candidate = result
        return candidate
