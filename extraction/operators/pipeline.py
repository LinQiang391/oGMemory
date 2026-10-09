"""Operator Pipeline — DAG orchestration for multimodal extractors.

A Pipeline is an ordered collection of steps. Each step runs one
operator (or a parallel group / conditional) against the same raw
node. Step results are collected into a ``PipelineResult`` that the
caller (typically ``OperatorService``) writes into the graph.

Pipeline spec (JSON-serialisable)::

    {
      "steps": [
        {"operator": "image_caption", "alias": "cap"},
        {"operator": "object_detection", "params": {"threshold": 0.6}},
        {
          "parallel": [
            {"operator": "face_detection"},
            {"operator": "pose_detection"}
          ]
        },
        {
          "operator": "image_caption",
          "when": "results.cap.confidence < 0.5"
        }
      ]
    }

Usage::

    pipe = OperatorPipeline.from_spec(spec_dict)
    result = pipe.execute(raw_node, ctx, registry=registry)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    MultimodalExtractor,
    is_multimodal_extractor,
)
from extraction.operators.registry import OperatorRegistry

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.models import RequestContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec model
# ---------------------------------------------------------------------------

@dataclass
class StepSpec:
    """One step in a pipeline, possibly wrapping a parallel group.

    A step is *atomic* (single operator) unless ``sub_steps`` is non-empty,
    in which case those sub_steps run concurrently against the same input.
    """
    operator: str | None = None      # operator_type id (single-op mode)
    params: dict[str, Any] = field(default_factory=dict)
    alias: str | None = None
    when: str | None = None          # expression evaluated on prior results
    sub_steps: list["StepSpec"] = field(default_factory=list)

    @property
    def is_parallel(self) -> bool:
        return bool(self.sub_steps)


@dataclass
class PipelineSpec:
    """Top-level pipeline definition."""
    steps: list[StepSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {"steps": [_step_to_dict(s) for s in self.steps]}


def _step_to_dict(s: StepSpec) -> dict:
    if s.is_parallel:
        return {"parallel": [_step_to_dict(sub) for sub in s.sub_steps]}
    d: dict[str, Any] = {"operator": s.operator}
    if s.params:
        d["params"] = dict(s.params)
    if s.alias:
        d["alias"] = s.alias
    if s.when:
        d["when"] = s.when
    return d


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Aggregated outcome for one pipeline step."""
    operator: str               # operator_type
    alias: str | None           # alias if provided
    extraction: ExtractionResult
    sub_results: list["StepResult"] = field(default_factory=list)
    skipped: bool = False       # True when `when` evaluated false
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.extraction.error is None


@dataclass
class PipelineResult:
    """Aggregated outcome for an entire pipeline execution."""
    raw_node_id: str
    steps: list[StepResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_alias(self) -> dict[str, StepResult]:
        """Index steps by alias. Steps without an alias are omitted."""
        return {s.alias: s for s in self.steps if s.alias}

    def to_dict(self) -> dict:
        return {
            "raw_node_id": self.raw_node_id,
            "ok": self.ok,
            "errors": list(self.errors),
            "steps": [_step_result_to_dict(s) for s in self.steps],
        }


def _step_result_to_dict(s: StepResult) -> dict:
    if s.sub_results:
        return {
            "parallel": [_step_result_to_dict(sub) for sub in s.sub_results],
            "skipped": s.skipped,
        }
    return {
        "operator": s.operator,
        "alias": s.alias,
        "skipped": s.skipped,
        "duration_ms": round(s.duration_ms, 3),
        "extracted_content": s.extraction.extracted_content,
        "confidence": s.extraction.confidence,
        "schema_used": s.extraction.schema_used,
        "error": s.extraction.error,
    }


# ---------------------------------------------------------------------------
# Conditional expression evaluator
# ---------------------------------------------------------------------------

# A tiny safe expression evaluator for `when` clauses. Supports:
#   results.<alias>.confidence <op> <literal>
#   results.<alias>.ok
# where <op> is one of < <= > >= == !=.
#
# This is deliberately minimal — full Python eval would be a security
# hazard given that pipeline specs come from HTTP requests.

_TOKEN_RE = re.compile(
    r"""
    \s*(?P<op><=|>=|==|!=|<|>|\bnot\b|\band\b|\bor\b|\bTrue\b|\bFalse\b|\bNone\b)
    |\s*(?P<num>-?\d+\.?\d*)
    |\s*(?P<ident>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
    """,
    re.VERBOSE,
)


def _safe_eval_when(expr: str, results_by_alias: dict[str, StepResult], *, validate_only: bool = False) -> bool:
    """Evaluate a restricted boolean expression against step results.

    The grammar is intentionally tiny:

        <expr>     ::= <term> ( ("and"|"or") <term> )*
        <term>     ::= "not" <term>
                     | "True" | "False"
                     | <ident> ( <op> <literal> )?
        <ident>    ::= "results" "." <alias> ( "." <attr> )?
        <attr>     ::= "ok" | "confidence" | "skipped"
        <op>       ::= "<" | "<=" | ">" | ">=" | "==" | "!="
        <literal>  ::= number | "True" | "False" | "None"

    Anything not matching this grammar raises PipelineSpecError.
    """
    if not expr or not expr.strip():
        return True

    tokens = _tokenize(expr)

    # Recursive descent
    pos = [0]

    def peek() -> str | None:
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take() -> str:
        if peek() is None:
            raise PipelineSpecError(f"Unexpected end of expression: {expr!r}")
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_or() -> bool:
        left = parse_and()
        while peek() == "or":
            take()
            right = parse_and()
            left = left or right
        return left

    def parse_and() -> bool:
        left = parse_not()
        while peek() == "and":
            take()
            right = parse_not()
            left = left and right
        return left

    def parse_not() -> bool:
        if peek() == "not":
            take()
            return not parse_not()
        return parse_atom()

    def parse_atom() -> bool:
        tok = peek()
        if tok is None:
            raise PipelineSpecError(f"Unexpected end of expression: {expr!r}")

        if tok == "True":
            take(); return True
        if tok == "False":
            take(); return False

        # Identifier: results.<alias>[.<attr>] <op> <literal>?
        if re.match(r"^[A-Za-z_]", tok or ""):
            ident = take()
            if not re.fullmatch(r"results\.[A-Za-z_][A-Za-z_0-9]*(?:\.(?:ok|confidence|skipped))?", ident):
                raise PipelineSpecError(f"Invalid identifier: {ident!r}")
            value = True if validate_only else _resolve_ident(ident, results_by_alias)
            op = peek()
            if op in ("<", "<=", ">", ">=", "==", "!="):
                take()
                rhs_tok = take()
                rhs = _parse_literal(rhs_tok)
                return True if validate_only else _apply_op(op, value, rhs)
            # bare boolean (must coerce to bool)
            return bool(value)

        raise PipelineSpecError(f"Unexpected token {tok!r} in expression: {expr!r}")

    result = parse_or()
    if pos[0] != len(tokens):
        raise PipelineSpecError(
            f"Unconsumed tokens after {expr!r}: {tokens[pos[0]:]!r}"
        )
    return result


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        # Consume an operator as a complete token, including at end of input.
        op = next((op for op in ("<=", ">=", "==", "!=", "<", ">")
                   if expr.startswith(op, i)), None)
        if op is not None:
            tokens.append(op)
            i += len(op)
            continue
        # Identifiers: keyword OR dotted path like results.<alias>.<attr>
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", expr[i:])
        if m:
            tok = m.group(0)
            if tok in ("and", "or", "not", "True", "False", "None"):
                tokens.append(tok)
            else:
                tokens.append(tok)
            i += len(tok); continue
        # Numbers
        m = re.match(r"-?\d+\.?\d*", expr[i:])
        if m:
            tokens.append(m.group(0))
            i += len(m.group(0)); continue
        raise PipelineSpecError(f"Cannot tokenize char {c!r} in {expr!r}")
    return tokens


def _resolve_ident(ident: str, results: dict[str, StepResult]) -> Any:
    parts = ident.split(".")
    if len(parts) < 2 or parts[0] != "results":
        raise PipelineSpecError(
            f"Identifiers must start with 'results.': got {ident!r}"
        )
    alias = parts[1]
    step = results.get(alias)
    if step is None:
        raise PipelineSpecError(f"Unknown alias in when-clause: {alias!r}")
    if len(parts) == 2:
        # Bare results.<alias> — treat as ok-ness
        return step.ok
    attr = parts[2]
    if attr == "ok":
        return step.ok
    if attr == "confidence":
        return step.extraction.confidence
    if attr == "skipped":
        return step.skipped
    raise PipelineSpecError(f"Unknown attribute {attr!r} in {ident!r}")


def _parse_literal(tok: str) -> Any:
    if tok == "True":
        return True
    if tok == "False":
        return False
    if tok == "None":
        return None
    try:
        if "." in tok:
            return float(tok)
        return int(tok)
    except ValueError:
        raise PipelineSpecError(f"Invalid literal: {tok!r}")


def _apply_op(op: str, left: Any, right: Any) -> bool:
    try:
        if op == "<":  return left < right
        if op == "<=": return left <= right
        if op == ">":  return left > right
        if op == ">=": return left >= right
        if op == "==": return left == right
        if op == "!=": return left != right
    except TypeError as exc:
        raise PipelineSpecError(
            f"Cannot compare {left!r} {op} {right!r}: {exc}"
        )
    raise PipelineSpecError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# Pipeline executor
# ---------------------------------------------------------------------------

class PipelineSpecError(ValueError):
    """Raised for malformed pipeline specs or `when` expressions."""


class OperatorPipeline:
    """Executable pipeline of extraction steps.

    Construct via ``from_spec(dict)`` for JSON-driven pipelines, or
    programmatically via the ``add`` / ``parallel`` chainable methods.
    """

    def __init__(
        self,
        spec: PipelineSpec | None = None,
        *,
        max_parallel_workers: int = 4,
    ) -> None:
        self._spec = spec or PipelineSpec()
        self._max_parallel_workers = max_parallel_workers

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: dict | PipelineSpec) -> "OperatorPipeline":
        if isinstance(spec, PipelineSpec):
            return cls(spec)
        if not isinstance(spec, dict):
            raise PipelineSpecError(f"Pipeline spec must be dict, got {type(spec).__name__}")
        steps_raw = spec.get("steps") or []
        if not isinstance(steps_raw, list):
            raise PipelineSpecError("steps must be a list")
        steps = [_parse_step(s) for s in steps_raw]
        return cls(PipelineSpec(steps=steps))

    def validate(self, registry: OperatorRegistry, raw_type: Any) -> None:
        """Preflight asynchronous work without executing operators or predicates."""
        if not self._spec.steps:
            raise PipelineSpecError("Pipeline must contain at least one step")

        def visit(steps):
            for step in steps:
                if step.when:
                    _safe_eval_when(step.when, {}, validate_only=True)
                if step.is_parallel:
                    visit(step.sub_steps)
                else:
                    if not registry.has(step.operator):
                        raise PipelineSpecError(f"Unknown operator: {step.operator!r}")
                    if raw_type not in registry.get(step.operator).supported_types:
                        raise PipelineSpecError(f"Operator {step.operator!r} does not support {raw_type}")
        visit(self._spec.steps)

    def add(
        self,
        operator: str | MultimodalExtractor,
        *,
        alias: str | None = None,
        **params: Any,
    ) -> "OperatorPipeline":
        op_type = operator.extractor_type if is_multimodal_extractor(operator) else operator
        if not (isinstance(op_type, str) and op_type):
            raise PipelineSpecError(
                f"add() requires an operator_type string or MultimodalExtractor instance, got {operator!r}"
            )
        self._spec.steps.append(StepSpec(operator=op_type, params=params, alias=alias))
        return self

    def parallel(self, *operators: str | MultimodalExtractor) -> "OperatorPipeline":
        if not operators:
            raise PipelineSpecError("parallel() requires at least one operator")
        sub = [
            StepSpec(
                operator=o.extractor_type if is_multimodal_extractor(o) else o,
            )
            for o in operators
        ]
        self._spec.steps.append(StepSpec(sub_steps=sub))
        return self

    def when(self, expr: str) -> "OperatorPipeline":
        """Attach a `when` predicate to the most recently added step."""
        if not self._spec.steps:
            raise PipelineSpecError("when() called with no preceding step")
        self._spec.steps[-1].when = expr
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        raw_node: "RawNode",
        ctx: "RequestContext",
        *,
        registry: OperatorRegistry,
    ) -> PipelineResult:
        """Run all steps against ``raw_node`` and aggregate results.

        Step failures are captured into ``StepResult.extraction.error``
        rather than raised, so that one bad operator does not abort the
        rest of the pipeline.
        """
        result = PipelineResult(raw_node_id=raw_node.node_id)
        results_by_alias: dict[str, StepResult] = {}

        for step in self._spec.steps:
            # Conditional short-circuit
            if step.when:
                try:
                    keep = _safe_eval_when(step.when, results_by_alias)
                except PipelineSpecError as exc:
                    result.errors.append(f"Invalid when-clause: {exc}")
                    keep = False
                if not keep:
                    result.steps.append(StepResult(
                        operator=step.operator or "<parallel>",
                        alias=step.alias,
                        extraction=_skipped_result(raw_node.node_id),
                        skipped=True,
                    ))
                    continue

            if step.is_parallel:
                sub_results = self._run_parallel(
                    step.sub_steps, raw_node, ctx, registry, result, results_by_alias,
                )
                result.steps.append(StepResult(
                    operator="<parallel>",
                    alias=step.alias,
                    extraction=_aggregate_result(raw_node.node_id, sub_results),
                    sub_results=sub_results,
                ))
                # Register sub-step aliases individually too
                for sub in sub_results:
                    if sub.alias:
                        results_by_alias[sub.alias] = sub
            else:
                sr = self._run_step(step, raw_node, ctx, registry, result)
                result.steps.append(sr)
                if sr.alias:
                    results_by_alias[sr.alias] = sr

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step: StepSpec,
        raw_node: "RawNode",
        ctx: "RequestContext",
        registry: OperatorRegistry,
        result: PipelineResult,
    ) -> StepResult:
        assert step.operator is not None
        try:
            operator = registry.get(step.operator)
        except Exception as exc:
            err = f"Operator '{step.operator}' lookup failed: {exc}"
            result.errors.append(err)
            logger.warning(err)
            return StepResult(
                operator=step.operator,
                alias=step.alias,
                extraction=ExtractionResult(
                    node_id=raw_node.node_id,
                    extracted_content={},
                    confidence=0.0,
                    schema_used="",
                    error=str(exc),
                ),
            )

        import time
        start = time.perf_counter()
        try:
            extraction = operator.extract(raw_node, ctx)
            # Per-step params: store on metadata so consumers can inspect
            if step.params and extraction.error is None:
                extraction.extracted_content.setdefault("_params", {}).update(step.params)
        except Exception as exc:
            err = f"Operator '{step.operator}' raised: {exc}"
            logger.exception(err)
            extraction = ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={},
                confidence=0.0,
                schema_used=getattr(operator, "schema_name", ""),
                error=str(exc),
            )
            result.errors.append(err)
        duration_ms = (time.perf_counter() - start) * 1000.0

        return StepResult(
            operator=step.operator,
            alias=step.alias,
            extraction=extraction,
            duration_ms=duration_ms,
        )

    def _run_parallel(
        self,
        sub_steps: list[StepSpec],
        raw_node: "RawNode",
        ctx: "RequestContext",
        registry: OperatorRegistry,
        result: PipelineResult,
        results_by_alias: dict[str, StepResult] | None = None,
    ) -> list[StepResult]:
        # Each sub-step's own when-clause is evaluated against the alias
        # results already collected from earlier TOP-LEVEL steps. Sub-
        # steps inside one parallel group intentionally cannot see each
        # other's results (peer sub-step aliases), because they run
        # concurrently and there is no defined completion order. They
        # DO see results from prior top-level steps (the alias dict
        # is plumbed through, not a fresh empty dict), so an `alias`
        # collected at the top level can still gate a parallel sub-step.
        if results_by_alias is None:
            results_by_alias = {}
        outs: list[StepResult | None] = [None] * len(sub_steps)
        to_run: list[tuple[int, StepSpec]] = []
        for idx, sub in enumerate(sub_steps):
            if sub.when:
                try:
                    keep = _safe_eval_when(sub.when, results_by_alias)
                except PipelineSpecError as exc:
                    result.errors.append(f"Invalid when-clause in parallel sub-step: {exc}")
                    keep = False
                if not keep:
                    outs[idx] = StepResult(
                        operator=sub.operator or "<parallel>",
                        alias=sub.alias,
                        extraction=_skipped_result(raw_node.node_id),
                        skipped=True,
                    )
                    continue
            to_run.append((idx, sub))

        if to_run:
            with ThreadPoolExecutor(max_workers=self._max_parallel_workers) as ex:
                futures = {
                    ex.submit(self._run_step, sub, raw_node, ctx, registry, result): idx
                    for idx, sub in to_run
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        outs[idx] = fut.result()
                    except Exception as exc:
                        outs[idx] = StepResult(
                            operator=to_run[0][1].operator or "<parallel>",
                            alias=None,
                            extraction=ExtractionResult(
                                node_id=raw_node.node_id,
                                extracted_content={},
                                confidence=0.0,
                                schema_used="",
                                error=str(exc),
                            ),
                        )
        return [o for o in outs if o is not None]

    @property
    def spec(self) -> PipelineSpec:
        return self._spec


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

def _parse_step(raw: Any) -> StepSpec:
    if not isinstance(raw, dict):
        raise PipelineSpecError(f"Step must be a dict, got {type(raw).__name__}")
    for key in ("when", "alias"):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            raise PipelineSpecError(f"{key} must be a string")
    if raw.get("params") is not None and not isinstance(raw["params"], dict):
        raise PipelineSpecError("params must be a dict")
    if "parallel" in raw:
        subs = raw["parallel"]
        if not isinstance(subs, list) or not subs:
            raise PipelineSpecError("'parallel' must be a non-empty list")
        return StepSpec(
            sub_steps=[_parse_step(s) for s in subs],
            alias=raw.get("alias"),
            when=raw.get("when"),
        )
    op = raw.get("operator")
    if not op or not isinstance(op, str):
        raise PipelineSpecError(f"Step missing 'operator': {raw!r}")
    return StepSpec(
        operator=op,
        params=dict(raw.get("params") or {}),
        alias=raw.get("alias"),
        when=raw.get("when"),
    )


def _skipped_result(node_id: str) -> ExtractionResult:
    return ExtractionResult(
        node_id=node_id,
        extracted_content={"skipped": True},
        confidence=0.0,
        schema_used="",
    )


def _aggregate_result(node_id: str, subs: list[StepResult]) -> ExtractionResult:
    """Build a synthetic ExtractionResult that aggregates parallel sub-results.

    The synthetic result has the union of extracted_content and the
    average confidence. It is mostly for convenience — consumers
    typically iterate ``step.sub_results`` directly.
    """
    merged: dict[str, Any] = {"_parallel": True}
    confidences: list[float] = []
    errors: list[str] = []
    for s in subs:
        if s.skipped:
            continue
        merged.setdefault(s.operator, s.extraction.extracted_content)
        confidences.append(s.extraction.confidence)
        if s.extraction.error:
            errors.append(f"{s.operator}: {s.extraction.error}")
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return ExtractionResult(
        node_id=node_id,
        extracted_content=merged,
        confidence=avg_conf,
        schema_used="parallel_aggregate_v1",
        error="; ".join(errors) if errors else None,
    )
