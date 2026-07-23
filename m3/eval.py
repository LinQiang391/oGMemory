"""M3-Bench evaluation harness.

Drives :class:`OGMemoryControlAdapter` over every QA in an M3-Bench
annotations JSON, captures the adapter's answer and per-round
search history, then asks a judge LLM (default: GPT-4o) to score the
predicted answer against the ground truth. Results land in two
files:

* ``m3_eval_results.jsonl`` — one row per question with predicted /
  expected answers, judge score, search history, latency.
* ``m3_eval_summary.json`` — aggregate counts and per-question-type
  buckets (mean score, p50 iterations, count).

The harness is resumable: re-running with ``--resume`` skips any
``(video_id, question_id)`` already present in the results file. This
matters for the 1276-QA × 5-rounds evaluation budget — partial runs
can be picked up without recomputing.

Usage::

    python -m m3.eval \
        --annotations data/annotations/robot.json \
        --graph-store sql --connection-string "$GRAPH_DSN" \
        --embedder openai --llm minimax --judge-llm openai \
        --output-dir data/m3_eval/2026-07-22

Offline smoke / tests::

    python -m m3.eval --annotations tests.json \\
        --graph-store memory --embedder mock --llm mock --judge-llm mock
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter
from m3.index_episodic import build_embedder
from m3.qa import (
    ANSWER_SCHEMA,
    _coerce_before_clip,
    build_llm,
    iter_annotation_questions,
    load_questions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": (
                "1.0 if predicted answer is correct, 0.5 if partially "
                "correct, 0.0 if wrong or missing."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Brief explanation of why the predicted answer received "
                "the given score."
            ),
        },
    },
    "required": ["score", "reasoning"],
}


JUDGE_PROMPT_TEMPLATE = """You are an exacting evaluator for M3-Bench long-video QA.

The predicted answer and the expected (ground truth) answer are below.
Score the predicted answer:

* 1.0 — predicted answer is essentially correct (matches expected or
  conveys the same fact, possibly with minor wording differences)
* 0.5 — predicted answer is partially correct (right idea but wrong
  details, or missing a key part)
* 0.0 — predicted answer is wrong, contradicts the expected answer,
  or says "insufficient evidence"

Return JSON with keys: score (0/0.5/1) and reasoning.

Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}
"""


def build_judge_llm(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config_path: str | None = None,
):
    """Construct the judge LLM. Defaults to OpenAI / GPT-4o via env.

    Falls back to ``MockLLM`` when ``--judge-llm mock``. The judge is
    structurally identical to the answer LLM at the Protocol level;
    the only difference is the prompt and the JSON-mode schema we
    feed it.
    """
    from m3.config import load_m3_runtime_config
    runtime = load_m3_runtime_config(config_path)
    # Reuse qa_llm config unless the user overrides.
    configured = runtime.qa_llm
    provider = provider or configured.provider
    model = model or configured.model
    base_url = base_url or configured.base_url
    api_key = api_key or configured.api_key

    if provider == "mock":
        from providers.llm import MockLLM
        return MockLLM()

    if provider == "openai":
        from providers.llm import get_openai_llm
        OpenAILLM, _ = get_openai_llm()
        return OpenAILLM(
            api_key=api_key,
            base_url=base_url,
            model=model or "gpt-4o-mini",
            temperature=0.0,
            max_tokens=configured.max_tokens,
            json_mode=True,
        )

    if provider == "minimax":
        from providers.llm import get_minimax_llm
        MiniMaxLLM = get_minimax_llm()
        return MiniMaxLLM(
            api_key=api_key,
            base_url=base_url,
            model=model or "MiniMax-M3",
            temperature=0.0,
            max_tokens=configured.max_tokens,
            json_mode=True,
        )

    raise ValueError(f"unknown judge llm provider: {provider}")


def judge_answer(
    *,
    judge_llm,
    question: str,
    expected: str,
    predicted: str,
) -> tuple[float, str]:
    """Ask the judge LLM to score the predicted answer.

    Returns ``(score, reasoning)``. Falls back to ``(0.0, "<reason>")``
    when the judge LLM fails or returns an unusable response — never
    raises, so a flaky judge can't abort the whole eval run.
    """
    if not predicted or not predicted.strip():
        return 0.0, "predicted answer is empty"
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected=expected or "(no expected answer)",
        predicted=predicted,
    )
    try:
        result = judge_llm.complete_json(prompt, JUDGE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 — judge is best-effort
        logger.warning("Judge LLM failed: %s", exc)
        return 0.0, f"judge failed: {exc}"
    raw_score = result.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    # Normalise to the {0.0, 0.5, 1.0} set the docstring promises.
    if score >= 0.75:
        score = 1.0
    elif score >= 0.25:
        score = 0.5
    else:
        score = 0.0
    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    return score, reasoning.strip()


# ---------------------------------------------------------------------------
# Per-question result schema
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    video_id: str
    question_id: str
    question_index: int
    question: str
    expected_answer: str
    predicted_answer: str
    type: str = ""
    timestamp: Any = None
    before_clip: Any = None
    iterations: int = 0
    finish_reason: str = ""
    search_rounds: int = 0
    search_history: list[dict] = field(default_factory=list)
    judge_score: float = 0.0
    judge_reasoning: str = ""
    latency_seconds: float = 0.0
    error: str = ""

    def to_jsonl(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-type aggregation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[min(99, max(0, int(p) - 1))]


def aggregate(
    results: Iterable[QuestionResult],
    *,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Compute per-type and overall metrics from a result stream."""
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    overall: list[QuestionResult] = []
    for r in results:
        overall.append(r)
        # type may be a string or list; coerce to a stable string.
        t = r.type
        if isinstance(t, list):
            t = ",".join(str(x) for x in t)
        elif not isinstance(t, str):
            t = str(t) if t else ""
        t = t.strip() or "(unspecified)"
        by_type[t].append(r)

    def _bucket(rs: list[QuestionResult]) -> dict:
        if not rs:
            return {"count": 0}
        scores = [r.judge_score for r in rs]
        iters = [float(r.iterations) for r in rs]
        lats = [r.latency_seconds for r in rs]
        return {
            "count": len(rs),
            "answered": sum(1 for r in rs if not r.error),
            "failed": sum(1 for r in rs if r.error),
            "mean_score": statistics.fmean(scores) if scores else 0.0,
            "p50_iterations": _percentile(iters, 50),
            "p95_iterations": _percentile(iters, 95),
            "mean_latency_seconds": statistics.fmean(lats) if lats else 0.0,
            "p95_latency_seconds": _percentile(lats, 95),
        }

    return {
        "total_questions": len(overall),
        "answered": sum(1 for r in overall if not r.error),
        "failed": sum(1 for r in overall if r.error),
        "overall_mean_score": _bucket(overall).get("mean_score", 0.0),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "by_type": {t: _bucket(rs) for t, rs in sorted(by_type.items())},
    }


# ---------------------------------------------------------------------------
# Per-question driver
# ---------------------------------------------------------------------------

def answer_with_adapter(
    *,
    adapter: OGMemoryControlAdapter,
    question: str,
    video_id: str,
    question_id: str,
) -> QuestionResult:
    """Run one question through the adapter. Captures the answer and
    search history into a QuestionResult; never raises."""
    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-eval",
        trace_id=f"m3-eval-{question_id}",
    )
    task_ctx = M3TaskContext(question=question, video_id=video_id)
    started = time.monotonic()
    try:
        result = adapter.run(ctx, task_ctx)
        predicted = (
            result.output[0]
            if result.output else ""
        )
        return QuestionResult(
            video_id=video_id,
            question_id=question_id,
            question_index=0,  # overwritten by caller
            question=question,
            expected_answer="",  # overwritten by caller
            predicted_answer=str(predicted).strip(),
            finish_reason=result.finish_reason,
            iterations=result.iterations,
            search_history=list(task_ctx.search_history),
            search_rounds=len(task_ctx.search_history),
            latency_seconds=time.monotonic() - started,
            error="" if result.completed else f"loop did not complete ({result.finish_reason})",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adapter failed for %s/%s", video_id, question_id)
        return QuestionResult(
            video_id=video_id,
            question_id=question_id,
            question_index=0,
            question=question,
            expected_answer="",
            predicted_answer="",
            latency_seconds=time.monotonic() - started,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "m3_eval_results.jsonl"
    summary_path = output_dir / "m3_eval_summary.json"

    questions = load_questions(annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
    if args.type_filter:
        wanted = set(args.type_filter.split(","))
        questions = [
            q for q in questions
            if (q.get("type") if isinstance(q.get("type"), str)
                else ",".join(q.get("type") or []))
            in wanted
        ]
    if args.limit is not None:
        questions = questions[:args.limit]
    logger.info("Loaded %d questions", len(questions))

    # Build the graph service the adapter will query.
    from m3.load_clips import build_graph_service
    graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    logger.info("Graph store: %s", store_kind)

    # Build the adapter (embedder + answer LLM).
    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    answer_llm = build_llm(
        args.llm,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.temperature,
        config_path=args.config,
    )
    adapter = OGMemoryControlAdapter(
        llm=answer_llm,
        graph=graph,
        embedder=embedder,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
        default_video_id=args.default_video_id,
    )

    # Build the judge LLM (separate from the answer LLM so we can
    # use GPT-4o for scoring while minimax serves the answers).
    judge_llm = build_judge_llm(
        args.judge_llm,
        model=args.judge_llm_model,
        base_url=args.judge_llm_base_url,
        api_key=args.judge_llm_api_key,
        config_path=args.config,
    )
    judge_provider = args.judge_llm or "openai"
    judge_model = args.judge_llm_model or "gpt-4o-mini"

    # Resume support: skip (video_id, question_id) pairs already on disk.
    done_keys: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((str(rec.get("video_id")), str(rec.get("question_id"))))
        logger.info("Resume: skipping %d already-answered questions", len(done_keys))

    mode = "a" if args.resume else "w"
    all_results: list[QuestionResult] = []
    answered = failed = skipped = 0
    with results_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue
            # Run the adapter on the question.
            qr = answer_with_adapter(
                adapter=adapter,
                question=str(q["question"]),
                video_id=str(q["video_id"]),
                question_id=str(q["question_id"]),
            )
            qr.question_index = q.get("question_index", 0)
            qr.expected_answer = str(q.get("answer") or "")
            qr.type = (
                q.get("type") if isinstance(q.get("type"), str)
                else ",".join(q.get("type") or [])
            )
            qr.timestamp = q.get("timestamp")
            qr.before_clip = q.get("before_clip")

            # Judge predicted vs expected.
            score, reasoning = judge_answer(
                judge_llm=judge_llm,
                question=str(q["question"]),
                expected=str(q.get("answer") or ""),
                predicted=qr.predicted_answer,
            )
            qr.judge_score = score
            qr.judge_reasoning = reasoning
            if not qr.error:
                answered += 1
            else:
                failed += 1

            f.write(json.dumps(qr.to_jsonl(), ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(qr)

    # Write the summary.
    summary = aggregate(all_results, judge_provider=judge_provider, judge_model=judge_model)
    summary["answered"] = answered
    summary["failed"] = failed
    summary["skipped"] = skipped
    summary["annotations_path"] = str(annotations_path)
    summary["results_path"] = str(results_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        "Eval wrote %d results (%d answered, %d failed, %d skipped) → %s",
        len(all_results), answered, failed, skipped, results_path,
    )
    logger.info(
        "Overall mean score: %.3f — summary at %s",
        summary["overall_mean_score"], summary_path,
    )
    return 2 if failed and answered == 0 else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run M3-Bench evaluation: drives OGMemoryControlAdapter, "
                    "judges via GPT-4o, aggregates per question type.",
    )
    p.add_argument("--annotations", required=True,
                   help="Path to M3-Bench annotation JSON")
    p.add_argument("--output-dir", default="data/m3_eval/run",
                   help="Directory for results.jsonl + summary.json")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--config", default=None,
                   help="ogmemory.yaml path (default: deploy/ogmemory.yaml or $OGMEM_CONFIG)")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    # Filtering
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--type-filter", default=None,
                   help="Comma-separated question-type values to include")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip (video_id, question_id) pairs already in results.jsonl")
    # Adapter config
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--default-video-id", default=None,
                   help="Default video scope for [Search] when question.video_id is missing")
    # Embedder (for [Search])
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Answer LLM
    p.add_argument("--llm", choices=("mock", "openai", "minimax"), default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    # Judge LLM (separate from answer LLM)
    p.add_argument("--judge-llm", choices=("mock", "openai", "minimax"), default="openai")
    p.add_argument("--judge-llm-model", default="gpt-4o-mini")
    p.add_argument("--judge-llm-base-url", default=None)
    p.add_argument("--judge-llm-api-key", default=None)
    # Verbosity
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())