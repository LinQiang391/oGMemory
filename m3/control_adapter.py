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
import re
from copy import deepcopy
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

    ``before_clip`` is the M3-Bench temporal scope: when set, every
    [Search] call restricts its hits to clips with
    ``clip_id <= before_clip + 1`` (the question's own clip IS
    included — the load-bearing utterance usually sits in it, seconds
    before the question timestamp; the annotation's ``before_clip``
    is the clip BEFORE the question's own clip). The hard filter lives
    in :class:`service.graph_service.ClipScope` rather than the
    question string, so the query embedding is not polluted. Previous
    behaviour both used ``clip_id <= before_clip`` (silently excluding
    the question's own clip) AND appended " (information must be from
    clips before clip_id=N)" to the question — visible in the query
    embedding as noise ("drinks for three people before clip_id=0").
    """

    question: str
    video_id: str | None = None
    before_clip: int | None = None
    search_history: list[dict] = field(default_factory=list)
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Name-stripping helpers for [Search] BM25 rerank
# ---------------------------------------------------------------------------

_NAME_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _name_tokens_from_hits(
    hits: list,
) -> set[str]:
    """Collect normalized name tokens from the ``characters`` field of each hit.

    Each ``characters`` entry may be a plain string or a dict with at
    least a ``name`` key (the shape emitted by
    :mod:`m3.episodic_memory_generator`; see
    :func:`m3.index_episodic._stringify_named_items` for the schema). Names
    are lowercased and stripped; multi-word names stay as a single phrase
    so the word-boundary regex in :func:`_strip_name_tokens_from_query`
    can match them as one unit.

    Defensive against malformed values: anything that isn't a string or
    a dict-with-name is skipped silently. A hit with no ``characters``
    field contributes nothing.
    """
    names: set[str] = set()
    for hit in hits:
        ec = getattr(getattr(hit, "node", None), "extracted_content", None) or {}
        chars = ec.get("characters") or []
        if not isinstance(chars, list):
            continue
        for c in chars:
            if isinstance(c, str):
                token = c.strip()
            elif isinstance(c, dict):
                token = str(c.get("name") or "").strip()
            else:
                continue
            if token:
                names.add(token.lower())
    return names


def _strip_name_tokens_from_query(
    query: str,
    name_tokens: set[str],
) -> str:
    """Remove ``name_tokens`` from ``query`` as whole-word, case-insensitive matches.

    Whitespace introduced by the removals is collapsed. Falls back to
    the original ``query`` when:

    * ``query`` is empty or whitespace-only, or
    * ``name_tokens`` is empty (nothing to strip), or
    * stripping would leave no alphanumeric content (an empty string
      would short-circuit the BM25 rerank and silently degrade the
      tool to vector-only ordering — which is exactly what we want to
      avoid in the small-candidate regime where BM25 is decisive).

    The patterns are cached on a module-level dict because the same set
    of names recurs across questions of the same video.
    """
    if not query or not query.strip() or not name_tokens:
        return query

    out = query
    for name in name_tokens:
        if not name:
            continue
        pattern = _NAME_PATTERN_CACHE.get(name)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            _NAME_PATTERN_CACHE[name] = pattern
        out = pattern.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Strip a bare space left in front of sentence-final punctuation so
    # the output reads naturally ("What does?" not "What does ?").
    # Cosmetic for downstream prompts; BM25 tokenization is
    # whitespace/punctuation-agnostic so this only affects display.
    out = re.sub(r"\s+([?.!,;:])", r"\1", out)
    if not out or not any(ch.isalnum() for ch in out):
        return query
    return out


# Match Title-Case word sequences (1+ letters each), optionally joined
# by a single space or hyphen ("Lewis", "Lewis Hamilton", "Anne-Marie").
# Single uppercase "I" is excluded because ``[A-Z][a-z]+`` requires at
# least one lowercase follower. All-caps abbreviations are intentionally
# excluded — M3-Bench names are always Title Case in the question text.
_QUERY_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*\b")

# Capitalized words that are NOT names. Question-leading interrogatives
# and common sentence-initial nouns; these survive the first-word skip
# when they appear mid-sentence ("What does What look like?") and must
# not be stripped from the query. Kept short on purpose — extending it
# is safer than shrinking it (false negatives = we miss a name, false
# positives = we strip a content word).
_NAME_STOPWORDS = frozenset({
    # Question words
    "What", "Where", "When", "Who", "Why", "How",
    # Common verbs at sentence start
    "Is", "Are", "Was", "Were", "Do", "Does", "Did",
    "Can", "Could", "Would", "Should", "Will",
    "Put", "Take", "Get", "Give", "Find", "Show", "Tell",
    "Make", "Let", "Go", "Come", "Bring", "Move",
    # Determiners / pronouns
    "The", "This", "That", "These", "Those",
    "My", "Your", "His", "Her", "Our", "Their", "Its",
    # Prepositions (rare capitalized, but mid-sentence possible after "?")
    "In", "On", "At", "To", "For", "From", "By", "With", "Of",
    # Conjunctions
    "And", "Or", "But", "So", "Yet",
})

# First-word interrogatives — when the query starts with one of these,
# we skip it (it's a question word, not a name) and run the regex over
# the rest of the query. When the query starts with anything else
# (likely a name, e.g. "Lewis took what?"), we include the first word
# as a candidate.
_QUESTION_FIRST_WORDS = frozenset({
    "what", "where", "when", "who", "why", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will",
})


def _query_name_candidates(query: str) -> set[str]:
    """Cheap regex extractor for proper-name candidates in the query.

    Used as a SUPPLEMENT to :func:`_name_tokens_from_hits` so the
    dual-path retrieval in :func:`Search` can strip names from the
    query even when the correct clip — the one whose ``characters``
    field carries the name — never made it into the first vector
    top-k. The headline failure case is: the question says "Lewis" but
    Whisper transcribed the name as "Louis / Louise / Luis", so the
    first embedding pulls in other clips and the correct Lewis clip
    drops out of top-8. Without this regex layer, the characters from
    first hits would never include "Lewis" and we'd never strip it.

    Conservative: requires Title Case (``[A-Z][a-z]+``, one or more
    lowercase letters), filters a small stoplist of common capitalized
    non-names, and skips the first word of the query IF it is a
    question word. When the query starts with anything else (likely a
    name, e.g. "Lewis took what?"), the first word is included.

    Returns lowercased tokens so the result composes with the
    lowercased set from :func:`_name_tokens_from_hits`.
    """
    if not query or not query.strip():
        return set()

    candidates: set[str] = set()

    # Decide whether to scan the first word. Look only at the first
    # whitespace-separated token; ignore leading punctuation.
    stripped = query.lstrip()
    first_space = stripped.find(" ")
    first_word = stripped[:first_space] if first_space >= 0 else stripped

    if first_word and first_word.lower() in _QUESTION_FIRST_WORDS:
        # Skip the question word; scan the rest.
        scan = stripped[first_space + 1:] if first_space >= 0 else ""
    else:
        # First word is not a question word — scan the whole query
        # so leading names ("Lewis took what?") are captured.
        scan = stripped

    for match in _QUERY_NAME_PATTERN.finditer(scan):
        name = match.group(0)
        if name in _NAME_STOPWORDS:
            continue
        candidates.add(name.lower())
    return candidates


def _combined_name_tokens(query: str, hits: list) -> set[str]:
    """Union of characters-from-hits and regex-from-query, lowercased.

    Single entry point for both the BM25 ``lex_query`` stripping and
    the dual-path vector stripping in :func:`Search`. Combining both
    sources makes the stripping robust to the failure mode where the
    correct clip didn't make the first top-k and therefore can't
    contribute its ``characters`` field.
    """
    return _name_tokens_from_hits(hits) | _query_name_candidates(query)


def _merge_hits_by_node(*hit_lists: list) -> list:
    """Union of multiple hit lists, dedup by ``node_id``, keep the hit
    with the higher vector score for each node.

    The dual-path retrieval in :func:`Search` calls
    ``graph.search_by_embedding`` twice — once with the original
    ``query`` and once with the name-stripped ``stripped_query`` — and
    the two top-k sets can overlap on nodes that are relevant to both
    formulations. This merger keeps one row per node and preserves the
    stronger of the two cosine scores (a node that better matches
    *either* query is the better candidate).

    The output list is unordered — downstream BM25 rerank re-sorts by
    fused score.
    """
    merged: dict[str, object] = {}
    for hits in hit_lists:
        if not hits:
            continue
        for hit in hits:
            nid = getattr(getattr(hit, "node", None), "node_id", None)
            if nid is None:
                continue
            existing = merged.get(nid)
            if existing is None or hit.score > existing.score:
                merged[nid] = hit
    return list(merged.values())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@operational_tool(name="Search")
def Search(
    query: str,
    top_k: int = 8,
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
    :param top_k: Number of top hits to return (default 8; was 5).
    """
    if not query or not query.strip():
        return {"error": "query is required"}
    embedding = embedder.embed_texts([query])[0]
    # Adapter-level config (set per-adapter in __init__) is captured in
    # the closure via _SEARCH_CONFIG — the operational tool is bound
    # to the adapter in __init__ below.
    cfg = _SEARCH_CONFIG
    # When the LLM doesn't pass a top_k, fall back to the adapter's
    # default. This is what makes the per-run top_k from the pipeline
    # YAML actually take effect.
    effective_top_k = int(top_k) if top_k else int(cfg.get("default_top_k", 8))

    def _build_search_kwargs(emb: list[float]) -> dict:
        """Build the kwargs for ``graph.search_by_embedding`` with the
        current per-task scope (video_id / before_clip) applied. Used by
        both the primary and the dual-path retrievals so the second
        search stays inside the same scope."""
        kw: dict = {
            "query": list(emb),
            "ctx": ctx,
            "extractor_types": cfg["extractor_types"],
            "top_k": effective_top_k,
        }
        if cfg.get("scope_video_id") or cfg.get("scope_before_clip") is not None:
            from service.graph_service import ClipScope
            kw["scope"] = ClipScope(
                video_id=cfg.get("scope_video_id"),
                before_clip=cfg.get("scope_before_clip"),
            )
        return kw

    # --- Primary retrieval: embed the original query as-is -----------
    hits = graph.search_by_embedding(**_build_search_kwargs(embedding))

    # --- Build the name set for stripping -----------------------------
    # Two sources, unioned:
    #   (a) characters from the in-hand top-k hits — catches names that
    #       the first search DID find.
    #   (b) Title-Case regex on the query itself — catches names that
    #       the first search DID NOT find (the headline failure case:
    #       "Lewis" in the question, but the correct clip — the only
    #       one whose characters field carries "Lewis" — dropped out of
    #       the first top-k because the polluted embedding dragged
    #       cosine to ~0.25). Without source (b), P1 #6 would still
    #       miss this case entirely.
    name_tokens = _combined_name_tokens(query, hits)
    stripped_query = _strip_name_tokens_from_query(query, name_tokens)

    # --- Dual-path retrieval: re-embed without the names --------------
    # Only worth the second embed + store call when stripping actually
    # changed something. Skipping the trivial case keeps latency flat
    # for questions that don't name anyone.
    if stripped_query != query and stripped_query:
        embedding_stripped = embedder.embed_texts([stripped_query])[0]
        hits_stripped = graph.search_by_embedding(
            **_build_search_kwargs(embedding_stripped),
        )
        hits = _merge_hits_by_node(hits, hits_stripped)

    # BM25 rerank over the union. ``lex_query`` is the stripped form
    # because names like "Lewis" are 0-hit tokens against transcripts
    # that Whisper heard as "Louis / Louise / Luis" — keeping them in
    # the BM25 query only dilutes the lexical signal that the rerank is
    # here to surface.
    from m3.transcript_rerank import rerank_with_transcript_bm25
    hits = rerank_with_transcript_bm25(query, hits, lex_query=stripped_query)
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
                # Surface the per-clip ASR transcript so dialogue-driven
                # questions (M3-Bench "what do they plan", "what did they
                # say", "how do they feel") can be answered even when the
                # EpisodicMemoryGenerator's description didn't preserve
                # the load-bearing words verbatim.
                "speech_transcript": h.node.extracted_content.get(
                    "speech_transcript", [],
                ),
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
    "scope_before_clip": None,
    "default_top_k": 8,
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
        default_top_k: int = 8,
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
        self._default_top_k = default_top_k
        # Push per-adapter config into the module-level config dict
        # the Search tool reads. Done once at construction so the
        # handler is fully bound before the loop starts; safe because
        # each adapter instance overrides this on entry to run().
        _SEARCH_CONFIG["extractor_types"] = tuple(extractor_types)
        _SEARCH_CONFIG["scope_video_id"] = default_video_id
        _SEARCH_CONFIG["default_top_k"] = int(default_top_k)
        # Per-run state set inside run() so the subclass can read it
        # in _execute_and_append_tool_calls / _parse_output.
        self._current_task_ctx: M3TaskContext | None = None
        self._final_answer: str | None = None
        self._search_cache: dict[tuple[str, int], dict] = {}

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
        self._search_cache.clear()

        scope = task_ctx.video_id or self._default_video_id
        # Wire ``before_clip`` through to the [Search] ClipScope so the
        # store hard-filters the search window. Previously this was only
        # encoded as text appended to the question, which polluted the
        # query embedding and left enforcement to the LLM.
        #
        # M3-Bench annotations set ``before_clip`` to the clip BEFORE the
        # clip that contains the question timestamp (verified on
        # robot.json: before_clip == clip_index(timestamp) - 1 for every
        # question), and the load-bearing utterance is usually in the
        # question's own clip, seconds before the timestamp. The
        # searchable window must therefore extend through the question's
        # own clip: ``clip_id <= before_clip + 1``. (Time-precise
        # scoping via node source_start_sec is a follow-up.)
        bc = task_ctx.before_clip
        _SEARCH_CONFIG["scope_before_clip"] = (bc + 1) if bc is not None else None
        _SEARCH_CONFIG["scope_video_id"] = scope

        system = (
            "You are an expert video analyst answering questions about a "
            "robot's point-of-view video. Use the [Search] tool to find "
            "relevant episodic evidence in the graph memory, then call "
            "[Answer] once you have enough to respond. Be concise — the "
            "evaluator scores verbatim against the ground truth.\n\n"
            "Tips for searching effectively:\n"
            "- Do not repeat a query with the same top_k. Review existing "
            "hits first; if evidence is missing, search a different action, "
            "location, speaker, or synonym. Repeated queries reuse prior "
            "results and do not provide new evidence.\n"
            "- Each [Search] hit has TWO fields: ``description`` (a visual "
            "summary written by the LLM) and ``speech_transcript`` (a list "
            "of {start_sec, end_sec, text} dicts from the Whisper ASR). "
            "The transcript is the strongest signal for dialogue-driven "
            "questions — what people SAID, what they PLAN, how they FEEL, "
            "or any speech-anchored fact. Even when the description "
            "doesn't mention it, the transcript often does. Read BOTH "
            "fields on every hit before forming a query.\n"
            "- If the description looks thin or doesn't match your query, "
            "skim the transcript text in the same hit — the answer is "
            "often there.\n"
            "- When the description and transcript disagree (description "
            "says 'happy' but transcript says 'I'm exhausted'), the "
            "transcript wins for affect/emotion questions.\n"
            "- Only commit to an answer you can ground in a retrieved "
            "hit. If the evidence does not mention the person, object, "
            "or fact the question asks about, answer from the closest "
            "supported evidence and say so plainly rather than "
            "inventing a match.\n"
            "- If the first [Search] returns no useful hits, broaden the "
            "query with synonyms, the location, or the speaker's action "
            "before giving up."
        )
        if scope:
            system += (
                f" Restrict your evidence to video {scope!r}."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {task_ctx.question}"},
        ]

    def _parse_output(self, content, ctx):  # noqa: ARG002  # type: ignore[reportUnusedArgument]  # part of the abstract contract
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

    def _execute_tool(self, name, input, ctx):
        """Reuse successful searches within this run, including empty hits.

        Scope and request identity are fixed during a run. The cache is
        cleared in _build_messages so evidence cannot leak between tasks.
        Do not merge semantic paraphrases or different retrieval depths.
        """
        if name != "Search":
            return super()._execute_tool(name, input, ctx)

        params = dict(input)
        params.setdefault("top_k", self._default_top_k)
        query = params.get("query", "")
        try:
            top_k = int(params["top_k"] or self._default_top_k)
        except (TypeError, ValueError):
            return super()._execute_tool(name, params, ctx)
        if not isinstance(query, str) or not query.strip():
            return super()._execute_tool(name, params, ctx)
        params["top_k"] = top_k
        key = (" ".join(query.split()).casefold(), top_k)
        cached = self._search_cache.get(key)
        if cached is not None:
            result = deepcopy(cached)
            result["duplicate_query"] = True
            result["guidance"] = (
                "This query and top_k were already searched in this task. "
                "These are reused results, not new evidence. Read the existing "
                "description and speech_transcript; search a different clue "
                "or call Answer and state any missing evidence."
            )
        else:
            result = super()._execute_tool(name, params, ctx)
            # A transient failure must remain retryable.
            if isinstance(result, dict) and not result.get("error"):
                self._search_cache[key] = deepcopy(result)

        if self._current_task_ctx is not None:
            self._current_task_ctx.search_history.append({
                "round": len(self._current_task_ctx.search_history) + 1,
                "query": query,
                "top_k": top_k,
                "stage": "requested",
                "cache_hit": cached is not None,
            })
        return result

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
        result = super()._execute_and_append_tool_calls(
            messages, tool_calls, tools_used, read_uris, ctx, round_id=round_id,
        )

        # Adapter-level budget notice. After 3 [Search] rounds without
        # an [Answer] call the LLM is rephrasing the query and burning
        # the iteration budget. A single neutral notice reminds it the
        # budget is finite — without the old FORCE COMMIT wording that
        # pushed committing to unverified guesses ("use the best fact
        # even if the name does not appear") and drove the fabricated
        # answers the failure taxonomy flagged.
        if self._current_task_ctx is not None and not self._final_answer:
            search_rounds = sum(
                1 for h in self._current_task_ctx.search_history
                if h.get("stage") == "requested"
            )
            if search_rounds >= 3:
                # Only inject once.
                already_injected = any(
                    m.get("role") == "user"
                    and isinstance(m.get("content"), str)
                    and "search budget" in m["content"]
                    for m in messages
                )
                if not already_injected:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Search budget: you have used most of your "
                            "search rounds. Review the [Search] hits you "
                            "already have — if one of them answers the "
                            "question, call [Answer] with that evidence "
                            "now. If none of them does, call [Answer] "
                            "with your best assessment and state what "
                            "the evidence supports."
                        ),
                    })
                    logger.info(
                        "OGMemoryControlAdapter: injected search-budget "
                        "notice after %d search rounds for task %s",
                        search_rounds,
                        getattr(self._current_task_ctx, "question", "?"),
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
