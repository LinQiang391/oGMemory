"""M3-Bench ReAct control adapter over the heterogeneous graph.

Wraps the generic :class:`core.react_loop.ReactLoop` template method
with M3-specific [Search] and [Answer] tools. The LLM sees only the
two tool definitions; the adapter handles:

* [Search] → :meth:`GraphService.search_by_embedding` against the
  EpisodicMemoryGenerator nodes (configurable via ``extractor_types``),
  with the query embedded by the configured ``Embedder``. Hits are
  returned with provenance (video_id, clip_id, description, characters,
  events) so the LLM can reason over them.
* [Answer] → marks the loop as complete and captures the answer text
  on the task context. The next iteration runs in content-only mode
  (tools disabled), the adapter surfaces the captured answer as the
  loop's final output, and the run terminates.

The control loop reuses the M3-Agent prompt structure (system
message establishing the role, then user message with the question).
Future work can extend the system prompt with per-video context,
example few-shots, or different extractor types (face_detection,
voice_segmentation) once those operators ship.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.interfaces import Embedder
from core.models import RequestContext
from core.react_loop import (
    ReactIterationTrace,
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
)
from core.tools.operational_tool_spec import operational_tool
from service.graph_service import GraphService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task context
# ---------------------------------------------------------------------------

@dataclass
class M3TaskContext(ReactTaskContext):
    """Per-question state carried across the ReAct iterations.

    Holds the question, the (optional) video scope, and an
    accumulated search history that the LLM (and post-run audits) can
    inspect. ``final_answer`` is populated by the ``Answer`` tool
    once the LLM decides it has enough evidence.
    """

    question: str
    video_id: str | None = None
    search_history: list[dict] = field(default_factory=list)
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@operational_tool(name="Search")
def Search(
    query: str,
    top_k: int = 5,
    ctx: RequestContext = None,                # type: ignore[assignment]
    graph: GraphService = None,                # type: ignore[assignment]
    embedder: Embedder = None,                # type: ignore[assignment]
) -> dict:
    """Search the M3 graph for evidence relevant to a natural-language query.

    Embeds the query with the configured embedder, runs a top-k cosine
    similarity search over the indexed EpisractedNodes, and returns
    the hits with provenance (video_id, clip_id, description,
    characters, events, entities) so the LLM can cite each hit in
    subsequent rounds. The extractor type set and video scope come
    from adapter-level config (``extractor_types`` and
    ``default_video_id`` constructor args); they are NOT exposed to
    the LLM so the tool schema stays minimal.

    :param query: Natural-language search query (English / Chinese
        mixed scripts both work — the embedder handles tokenisation).
    :param top_k: Number of top hits to return (default 5).
    """
    if not query or not query.strip():
        return {"error": "query is required"}
    embedding = embedder.embed_texts([query])[0]
    # Adapter-level config (set per-adapter in __init__) is captured in
    # the closure via _SEARCH_CONFIG — the operational tool is bound
    # to the adapter in __init__ below.
    cfg = _SEARCH_CONFIG
    kwargs: dict = {
        "query": list(embedding),
        "ctx": ctx,
        "extractor_types": cfg["extractor_types"],
        "top_k": int(top_k),
    }
    if cfg.get("scope_video_id"):
        from service.graph_service import ClipScope
        kwargs["scope"] = ClipScope(video_id=cfg["scope_video_id"])
    hits = graph.search_by_embedding(**kwargs)
    return {
        "query": query,
        "hits": [
            {
                "node_id": h.node.node_id,
                "score": h.score,
                "video_id": h.video_id,
                "clip_id": h.clip_id,
                "description": h.node.extracted_content.get("description", ""),
                "characters": h.node.extracted_content.get("characters", []),
                "events": h.node.extracted_content.get("events", []),
                "entities": h.node.extracted_content.get("entities", []),
            }
            for h in hits
        ],
    }


# Module-level mutable config the Search tool reads at call time. Set
# by OGMemoryControlAdapter.__init__ — keeps the @operational_tool
# decorator's introspection contract (only query + top_k + 3 runtime
# types as params) while still letting per-adapter config flow into
# the handler.
_SEARCH_CONFIG: dict = {
    "extractor_types": ("episodic_memory_generator",),
    "scope_video_id": None,
}


@operational_tool(name="Answer")
def Answer(final_answer: str) -> dict:
    """Produce the final answer and end the loop.

    Call this when you have enough evidence to answer the question.
    The adapter captures ``final_answer`` and terminates the ReAct
    loop on the next iteration. Subsequent rounds run with tools
    disabled so the LLM doesn't double-call.

    :param final_answer: The answer text. Keep it concise and direct —
        the evaluator scores verbatim against the ground truth.
    """
    return {"status": "answered", "answer": final_answer}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class OGMemoryControlAdapter(ReactLoop):
    """M3 ReAct adapter — drives the [Search]/[Answer] loop.

    Usage::

        adapter = OGMemoryControlAdapter(
            llm=llm, graph=graph, embedder=embedder,
            max_iterations=5, extractor_types=("episodic_memory_generator",),
        )
        result = adapter.run(ctx, M3TaskContext(question="..."))
        print(result.output[0])  # the final answer

    The adapter exposes ``last_task_ctx`` on the instance after
    :meth:`run` so callers can introspect the search history and the
    captured answer without re-parsing ``result.output``.
    """

    def __init__(
        self,
        llm,
        graph: GraphService,
        embedder: Embedder,
        *,
        max_iterations: int = 5,
        timeout_seconds: float = 60.0,
        extractor_types: tuple[str, ...] = ("episodic_memory_generator",),
        default_video_id: str | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            operational_tools=[Search, Answer],
        )
        self._graph = graph
        self._embedder = embedder
        self._extractor_types = extractor_types
        self._default_video_id = default_video_id
        # Push per-adapter config into the module-level config dict
        # the Search tool reads. Done once at construction so the
        # handler is fully bound before the loop starts; safe because
        # each adapter instance overrides this on entry to run().
        _SEARCH_CONFIG["extractor_types"] = tuple(extractor_types)
        _SEARCH_CONFIG["scope_video_id"] = default_video_id
        # Per-run state set inside run() so the subclass can read it
        # in _execute_and_append_tool_calls / _parse_output.
        self._current_task_ctx: M3TaskContext | None = None
        self._final_answer: str | None = None

    # ------------------------------------------------------------------
    # ReactLoop abstract methods
    # ------------------------------------------------------------------

    def _runtime_injection_values(self, ctx: RequestContext) -> dict:
        return {
            "ctx": ctx,
            "graph": self._graph,
            "embedder": self._embedder,
        }

    def _get_custom_tools(self) -> list[dict]:
        # All tools are operational; no custom tools needed.
        return []

    def _execute_custom_tool(self, name, input, ctx):
        return {"error": f"Unknown tool: {name}"}

    def _build_messages(self, ctx, task_ctx: M3TaskContext):
        # Stash the task context and propagate per-question scope into
        # the module-level config the Search tool reads. Done here so
        # any tool call within this run() sees the right scope.
        self._current_task_ctx = task_ctx
        self._final_answer = None
        self._disable_tools_for_iteration = False

        scope = task_ctx.video_id or self._default_video_id
        _SEARCH_CONFIG["scope_video_id"] = scope

        system = (
            "You are an expert video analyst answering questions about a "
            "robot's point-of-view video. Use the [Search] tool to find "
            "relevant episodic evidence in the graph memory, then call "
            "[Answer] once you have enough to respond. Be concise — the "
            "evaluator scores verbatim against the ground truth."
        )
        if scope:
            system += (
                f" Restrict your evidence to video {scope!r}."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {task_ctx.question}"},
        ]

    def _parse_output(self, content, ctx):
        # Prefer the captured Answer text; fall back to the LLM's
        # free-form content if the model decided to answer without
        # calling [Answer] (some LLMs do this on the last round).
        if self._final_answer:
            return [self._final_answer]
        if content and content.strip():
            return [content.strip()]
        return None

    # ------------------------------------------------------------------
    # Hooks for [Answer] handling — capture the answer and end the loop
    # ------------------------------------------------------------------

    def _execute_and_append_tool_calls(
        self,
        messages,
        tool_calls,
        tools_used,
        read_uris,
        ctx,
        round_id="",
    ):
        """Override to detect [Answer] and short-circuit the loop."""
        # Track [Search] hits on the task context for post-run audit.
        for tc in tool_calls:
            name = tc.get("name", "") or tc.get("tool", "")
            if name == "Search" and self._current_task_ctx is not None:
                # The result will land in messages; we don't have it yet.
                # Stash the query for later correlation.
                self._current_task_ctx.search_history.append({
                    "round": len(self._current_task_ctx.search_history) + 1,
                    "query": tc.get("input", {}).get("query", ""),
                    "top_k": tc.get("input", {}).get("top_k", 5),
                    "stage": "requested",
                })

        result = super()._execute_and_append_tool_calls(
            messages, tool_calls, tools_used, read_uris, ctx, round_id=round_id,
        )

        for tc in tool_calls:
            name = tc.get("name", "") or tc.get("tool", "")
            if name == "Answer":
                self._final_answer = tc.get("input", {}).get("final_answer", "")
                if self._current_task_ctx is not None:
                    self._current_task_ctx.final_answer = self._final_answer
                # Tell the LLM to wrap up on the next round so
                # _parse_output fires with the captured answer. We
                # also disable tools so the next call returns plain
                # content rather than another tool call.
                messages.append({
                    "role": "user",
                    "content": (
                        "You called [Answer] — now output your final "
                        "answer text directly so the evaluator can "
                        "score it. Do not call any more tools."
                    ),
                })
                self._disable_tools_for_iteration = True
                logger.info(
                    "OGMemoryControlAdapter: Answer captured (%d chars); "
                    "ending loop on next iteration",
                    len(self._final_answer or ""),
                )
        return result

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def run(self, ctx: RequestContext, task_ctx: M3TaskContext) -> ReactLoopResult:
        """Reset per-run state and delegate to the base loop."""
        self._current_task_ctx = task_ctx
        self._final_answer = None
        return super().run(ctx, task_ctx)