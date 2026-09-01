"""Vision-only M3-Bench QA MVP over episodic-memory retrieval.

This command reads M3-Bench annotations, retrieves relevant episodic evidence
for each question, and asks an LLM to answer directly. It intentionally avoids
full M3-Agent ReAct control for the MVP; the interface is structured so a
multi-round ``[Search]`` / ``[Answer]`` adapter can replace the direct answerer
later without changing result files.

Usage::

    python -m m3.qa \
        --annotations data/annotations/robot.json \
        --index data/episodic_index.jsonl \
        --output-jsonl data/m3_qa_results.jsonl \
        --llm openai

Offline smoke / tests::

    python -m m3.qa --annotations tests.json --index index.jsonl --llm mock
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from m3.config import video_id_from_path
from m3.index_episodic import build_embedder
from m3.search import EpisodicSearcher, Evidence, format_evidence_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

# Matches the literal tool tags the LLM might emit in its content stream
# (``[Search]`` / ``[Answer]``) instead of using the tool_calls channel.
# Some models echo the protocol text into the answer when they fail to
# format the tool call, which lands verbatim in ``predicted_answer`` via
# ``_parse_output``. These aren't real answers — flag them as invalid.
_PROTOCOL_SCAFFOLDING_RE = re.compile(r"\[(Search|Answer)\]", re.IGNORECASE)


def _has_protocol_scaffolding(text: str) -> bool:
    """True if ``text`` contains the ``[Search]`` / ``[Answer]`` protocol tags.

    Used by both :func:`run_react_qa` (which marks the row as failed when
    producing it) and :func:`m3.pipeline.stages.qa_stage` (defense in
    depth — catches rows written by older drivers that didn't run the
    new check).
    """
    return bool(text) and bool(_PROTOCOL_SCAFFOLDING_RE.search(text))


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "A concise final answer to the question, grounded only in the evidence.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of which evidence supports the answer.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence from 0 to 1.",
        },
    },
    "required": ["answer"],
}


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def iter_annotation_questions(annotations: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalized QA records from an M3-Bench annotation object."""
    for key, info in annotations.items():
        if not isinstance(info, dict):
            continue
        video_id = str(key)
        declared_path = info.get("video_path")
        if declared_path:
            video_id = video_id_from_path(str(declared_path))
        qa_list = info.get("qa_list") or info.get("questions") or []
        if not isinstance(qa_list, list):
            continue
        for idx, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or qa.get("query")
            if not question:
                continue
            qid = qa.get("question_id") or qa.get("id") or f"{video_id}_Q{idx:04d}"
            yield {
                "video_id": video_id,
                "question_id": qid,
                "question_index": idx,
                "question": str(question),
                "answer": qa.get("answer") or qa.get("gt_answer") or qa.get("ground_truth") or "",
                "type": qa.get("type") or qa.get("category") or [],
                "timestamp": qa.get("timestamp"),
                "before_clip": qa.get("before_clip"),
                "raw": qa,
            }


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected annotation object at {path}")
    return list(iter_annotation_questions(data))


# ---------------------------------------------------------------------------
# LLM answering
# ---------------------------------------------------------------------------

def build_llm(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    config_path: str | None = None,
):
    """Construct the answer LLM for QA from CLI + ogmemory.yaml.

    Backwards-compatible wrapper around :func:`m3.llm_factory.build_qa_llm`
    — the canonical dispatch lives there so QA / episodic / judge roles
    share the same provider logic. Kept under the old name because
    external callers (notably ``m3.eval``) do ``from m3.qa import build_llm``.
    """
    from m3.llm_factory import build_qa_llm
    return build_qa_llm(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        config_path=config_path,
    )


def answer_question(
    *,
    llm,
    question: str,
    evidence: list[Evidence],
    video_id: str,
) -> dict[str, Any]:
    """Answer one question using direct retrieved evidence."""
    evidence_text = format_evidence_block(evidence)
    if not evidence_text:
        evidence_text = "(No evidence retrieved.)"
    prompt = f"""You answer questions about a long egocentric robot video.
Use ONLY the evidence clips below. If the evidence is insufficient, give the
best supported concise answer and mention uncertainty in the reasoning.

Video: {video_id}
Question: {question}

Evidence clips:
{evidence_text}

Return JSON with keys: answer, reasoning, confidence.
"""
    result = llm.complete_json(prompt, ANSWER_SCHEMA)
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = _fallback_answer_from_text(evidence_text)
    reasoning = result.get("reasoning") if isinstance(result.get("reasoning"), str) else ""
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0 if not evidence else 0.5
    return {
        "answer": answer.strip(),
        "reasoning": reasoning.strip(),
        "confidence": float(confidence),
    }


def _fallback_answer_from_text(text: str) -> str:
    """Tiny guard for schema-incomplete mock / weak LLM outputs."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Description:"):
            return line.split(":", 1)[1].strip()
    return "Insufficient evidence."


def _coerce_before_clip(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_qa(args: argparse.Namespace) -> int:
    annotations_path = Path(args.annotations).resolve()
    index_path = Path(args.index).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1
    if not index_path.exists():
        logger.error("episodic index not found: %s", index_path)
        return 1

    questions = load_questions(annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
    if args.limit is not None:
        questions = questions[:args.limit]

    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    searcher = EpisodicSearcher.from_index_path(
        index_path,
        embedder=embedder,
        hybrid_weight=args.hybrid_weight,
    )
    llm = build_llm(
        args.llm,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.temperature,
        config_path=args.config,
    )

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((str(rec.get("video_id")), str(rec.get("question_id"))))

    mode = "a" if args.resume else "w"
    answered = failed = skipped = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue
            before_clip = _coerce_before_clip(q.get("before_clip")) if args.respect_before_clip else None
            evidence = searcher.search(
                q["question"],
                video_id=q["video_id"],
                top_k=args.top_k,
                before_clip=before_clip,
                include_neighbors=args.include_neighbors,
            )
            # When include_neighbors>0, the searcher expands ±N
            # around the top hits so the answerer can ground temporal
            # questions like "where is she at the start" in adjacent
            # clips. Helps Q1 (before_clip=2) see both clip 0 and 1.
            try:
                answer = answer_question(
                    llm=llm,
                    question=q["question"],
                    evidence=evidence,
                    video_id=q["video_id"],
                )
                error = ""
                answered += 1
            except Exception as exc:
                logger.exception("QA failed for %s/%s", q["video_id"], q["question_id"])
                answer = {"answer": "", "reasoning": "", "confidence": 0.0}
                error = str(exc)
                failed += 1

            out = {
                "video_id": q["video_id"],
                "question_id": q["question_id"],
                "question_index": q["question_index"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "type": q["type"],
                "timestamp": q["timestamp"],
                "before_clip": q["before_clip"],
                "predicted_answer": answer["answer"],
                "reasoning": answer["reasoning"],
                "confidence": answer["confidence"],
                "evidence": [ev.to_dict() for ev in evidence],
                "error": error,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(
        "QA wrote %d answered records to %s (%d failed, %d skipped)",
        answered, output_path, failed, skipped,
    )
    return 2 if failed and answered == 0 else 0


# ---------------------------------------------------------------------------
# Adapter-driven driver (the modern path)
# ---------------------------------------------------------------------------

def run_react_qa(args: argparse.Namespace, graph=None, embedder=None, llm=None) -> int:
    """Adapter-driven QA: drives OGMemoryControlAdapter per question.

    Modern replacement for :func:`run_qa` (single-pass MVP that reads
    ``episodic_index.jsonl`` via the JSONL searcher). The adapter path
    uses ``GraphService.search_by_embedding`` against the indexed
    EpisractedNodes — embeddings land in the graph via
    ``m3.index_episodic`` (see ``test_index_episodic_persist_embeddings.py``
    for the contract).

    Falls back to building the graph / embedder / LLM from CLI args
    if any are None — same pattern as the other graph-using drivers.
    Resumable via ``--resume`` (skip ``(video_id, question_id)`` pairs
    already in the output JSONL).
    """
    from core.models import RequestContext
    from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
    from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter
    from m3.index_episodic import build_embedder as _build_embedder

    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1

    questions = load_questions(annotations_path)
    logger.info("Loaded %d questions from %s", len(questions), annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
        logger.info("After --video filter (%s): %d questions", args.video, len(questions))
    if args.limit is not None:
        questions = questions[:args.limit]

    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)

    if embedder is None:
        embedder = _build_embedder(
            args.embedder,
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
            dimension=args.embedding_dimension,
            config_path=args.config,
        )
    if llm is None:
        llm = build_llm(
            args.llm,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            temperature=args.temperature,
            config_path=args.config,
        )

    adapter = OGMemoryControlAdapter(
        llm=llm,
        graph=graph,
        embedder=embedder,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
        default_video_id=args.default_video_id,
        default_top_k=args.top_k,
    )

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((str(rec.get("video_id")), str(rec.get("question_id"))))
        if done_keys:
            logger.info("Resume: skipping %d already-answered questions", len(done_keys))

    mode = "a" if args.resume else "w"
    answered = failed = skipped = 0
    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-qa",
        trace_id="m3-qa",
    )

    with output_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue

            before_clip = _coerce_before_clip(q.get("before_clip")) if args.respect_before_clip else None
            task_ctx = M3TaskContext(
                question=str(q["question"]),
                video_id=str(q["video_id"]) if not args.default_video_id else args.default_video_id,
                before_clip=before_clip,
            )
            # NOTE: do NOT append "(information must be from clips
            # before clip_id=N)" to the question. That polluted the
            # query embedding ("drinks for three people before clip_id=0")
            # AND left enforcement to the LLM. The temporal filter is
            # now wired into ClipScope(before_clip=…) via the [Search]
            # tool — the store hard-filters by ``clip_id <= before_clip``
            # so only relevant clips appear in the search hits.
            search_history: list = []
            iterations = 0
            finish_reason = ""
            try:
                result = adapter.run(ctx, task_ctx)
                predicted = result.output[0] if result.output else ""
                iterations = result.iterations
                finish_reason = result.finish_reason
                search_history = list(task_ctx.search_history)
                # Three-way failure classification — mirrors
                # m3.eval.answer_with_adapter so the two drivers agree
                # on what "answered" / "failed" mean. The previous
                # code only checked exceptions, so a loop that
                # finished with finish_reason="max_iterations" or
                # content="[Answer] …" (scaffolding leak) was silently
                # recorded as answered.
                if finish_reason == "llm_error":
                    # Surface the upstream's own message + status code
                    # so the JSONL row tells operators whether it was a
                    # 400 (protocol mismatch, retry won't help) or a 500
                    # (transient, retry might). result.llm_error is the
                    # original :class:`LLMCallError` raised by
                    # ``_call_llm`` and threaded through to the result.
                    llm_err = getattr(result, "llm_error", None)
                    if llm_err is not None:
                        error = f"llm_error: {llm_err}"
                    else:
                        error = "llm_error: adapter loop terminated due to LLM transport failure"
                    predicted = ""
                    failed += 1
                elif not result.completed:
                    error = f"loop did not complete ({finish_reason})"
                    failed += 1
                elif _has_protocol_scaffolding(predicted):
                    error = (
                        "predicted_answer contains [Search]/[Answer] "
                        "protocol scaffolding"
                    )
                    predicted = ""
                    failed += 1
                else:
                    error = ""
                    answered += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Adapter failed for %s/%s", q["video_id"], q["question_id"],
                )
                predicted = ""
                error = str(exc)
                failed += 1

            out = {
                "video_id": q["video_id"],
                "question_id": q["question_id"],
                "question_index": q["question_index"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "type": q["type"],
                "timestamp": q["timestamp"],
                "before_clip": q["before_clip"],
                "predicted_answer": str(predicted).strip(),
                "iterations": iterations,
                "finish_reason": finish_reason,
                "search_rounds": len(search_history),
                "search_history": search_history,
                "error": error,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(
        "QA wrote %d answered records to %s (%d failed, %d skipped)",
        answered, output_path, failed, skipped,
    )
    return 2 if failed and answered == 0 else 0


def parse_react_qa_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """CLI for the adapter-driven QA driver.

    Shares most flags with :func:`parse_args` (legacy single-pass
    driver) but drops ``--index`` (we read from the graph now) and
    adds ``--graph-store``, ``--connection-string``,
    ``--max-iterations``, ``--timeout-seconds``, ``--default-video-id``.
    """
    p = argparse.ArgumentParser(
        description="Run M3-Bench QA via OGMemoryControlAdapter (graph retrieval).",
    )
    p.add_argument("--annotations", required=True)
    p.add_argument("--output-jsonl", default="data/m3_qa_results.jsonl")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--config", default=None,
                   help="ogmemory.yaml path (default: deploy/ogmemory.yaml or $OGMEM_CONFIG)")
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--respect-before-clip", action="store_true", default=True)
    p.add_argument("--ignore-before-clip", dest="respect_before_clip", action="store_false")
    # Adapter
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--default-video-id", default=None,
                   help="Override the per-question video scope passed to [Search]")
    p.add_argument("--top-k", type=int, default=8,
                   help="Number of evidence hits returned per [Search] "
                        "(default 8; was 5 — the lower default hurt recall "
                        "for short benchmarks).")
    p.add_argument("--include-neighbors", type=int, default=1,
                   help="Expand ±N neighbor clips around the top hits so the "
                        "answerer can see temporal context (default 1; was 0).")
    # Embedder (for [Search])
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Answer LLM
    p.add_argument("--llm", default=None,
                   help="Answer LLM label (mock/minimax/any OpenAI-compatible); "
                        "defaults to m3.qa_llm.provider from ogmemory.yaml")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main_react_qa(argv: Iterable[str] | None = None) -> int:
    args = parse_react_qa_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_react_qa(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vision-only M3 QA over episodic retrieval.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--index", required=True, help="episodic_index.jsonl from m3.index_episodic")
    p.add_argument("--output-jsonl", default="data/m3_qa_results.jsonl")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--resume", action="store_true")
    # Retrieval
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--include-neighbors", type=int, default=1)
    p.add_argument("--respect-before-clip", action="store_true", default=True)
    p.add_argument("--ignore-before-clip", dest="respect_before_clip", action="store_false")
    p.add_argument("--hybrid-weight", type=float, default=0.15)
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Answer LLM
    p.add_argument("--llm", default=None,
                   help="Answer LLM label (mock/minimax/any OpenAI-compatible); "
                        "defaults to m3.qa_llm.provider from ogmemory.yaml")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_qa(args)


if __name__ == "__main__":
    sys.exit(main())
