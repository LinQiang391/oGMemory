"""Unit tests for OperatorPipeline — DAG orchestration."""

from __future__ import annotations

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators import (
    ExtractionResult,
    MultimodalExtractor,
    OperatorPipeline,
    OperatorRegistry,
    PipelineSpecError,
)


class _RecordingExtractor:
    """Captures extract() calls and returns deterministic results."""

    def __init__(self, op_type, content=None, confidence=0.9, schema="v1", error=None):
        self._op_type = op_type
        self._content = content or {"from": op_type}
        self._confidence = confidence
        self._schema = schema
        self._error = error
        self.calls = 0

    @property
    def supported_types(self):
        return [RawDataType.PROMPT]

    @property
    def extractor_type(self):
        return self._op_type

    @property
    def schema_name(self):
        return self._schema

    @property
    def info(self):
        from extraction.operators import OperatorInfo, default_info
        return default_info(self)

    def extract(self, raw_node, ctx):
        self.calls += 1
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=dict(self._content),
            confidence=self._confidence,
            schema_used=self._schema,
            error=self._error,
        )


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


@pytest.fixture
def raw_node():
    return create_raw_node(
        raw_type=RawDataType.PROMPT,
        content="hello world",
        account_id="acct",
    )


@pytest.fixture
def registry():
    reg = OperatorRegistry()
    reg.register(_RecordingExtractor("cap", content={"text": "a picture"}))
    reg.register(_RecordingExtractor("obj", content={"items": [{"label": "cat"}]}, confidence=0.7))
    reg.register(_RecordingExtractor("face", content={"faces": 1}, confidence=0.95))
    reg.register(_RecordingExtractor("pose", content={"keypoints": 17}, confidence=0.6))
    reg.register(_RecordingExtractor("skippable", content={"x": 1}, confidence=0.0))
    return reg


class TestPipelineSpecParsing:
    def test_from_spec_simple(self, registry):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"operator": "cap"}]
        })
        assert len(pipe.spec.steps) == 1
        assert pipe.spec.steps[0].operator == "cap"

    def test_from_spec_parallel(self, registry):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"parallel": [
                {"operator": "face"},
                {"operator": "pose"},
            ]}]
        })
        step = pipe.spec.steps[0]
        assert step.is_parallel
        assert len(step.sub_steps) == 2

    def test_from_spec_alias_and_params(self, registry):
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "obj", "alias": "objects", "params": {"threshold": 0.5}},
            ]
        })
        step = pipe.spec.steps[0]
        assert step.alias == "objects"
        assert step.params == {"threshold": 0.5}

    def test_from_spec_when(self, registry):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"operator": "cap", "when": "results.x.ok"}]
        })
        assert pipe.spec.steps[0].when == "results.x.ok"

    def test_spec_round_trip(self):
        spec = {
            "steps": [
                {"operator": "cap", "alias": "a"},
                {"operator": "obj", "params": {"t": 0.5}},
                {"parallel": [
                    {"operator": "face"},
                    {"operator": "pose"},
                ]},
            ],
        }
        pipe = OperatorPipeline.from_spec(spec)
        assert pipe.spec.to_dict() == spec

    def test_spec_missing_operator_raises(self):
        with pytest.raises(PipelineSpecError, match="missing 'operator'"):
            OperatorPipeline.from_spec({"steps": [{"alias": "x"}]})

    def test_spec_parallel_must_be_list(self):
        with pytest.raises(PipelineSpecError, match="non-empty list"):
            OperatorPipeline.from_spec({"steps": [{"parallel": []}]})

    def test_spec_step_must_be_dict(self):
        with pytest.raises(PipelineSpecError, match="must be a dict"):
            OperatorPipeline.from_spec({"steps": ["bad"]})


class TestPipelineExecution:
    def test_single_step(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({"steps": [{"operator": "cap"}]})
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.ok
        assert len(result.steps) == 1
        sr = result.steps[0]
        assert sr.operator == "cap"
        assert sr.ok
        assert sr.extraction.extracted_content == {"text": "a picture"}

    def test_multiple_sequential(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "cap", "alias": "c"},
                {"operator": "obj"},
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.ok
        assert len(result.steps) == 2
        assert result.by_alias()["c"].operator == "cap"

    def test_parallel_step(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"parallel": [
                {"operator": "face"},
                {"operator": "pose"},
            ]}]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.ok
        step = result.steps[0]
        # StepResult doesn't have is_parallel; the parallel step is signalled
        # by the presence of sub_results and operator == "<parallel>"
        assert step.operator == "<parallel>"
        assert len(step.sub_results) == 2
        op_set = {sub.operator for sub in step.sub_results}
        assert op_set == {"face", "pose"}

    def test_when_skip(self, registry, raw_node, ctx):
        # The `when` clause references an alias that does not exist
        # in the results dict — the second step should be skipped
        # and an error appended.
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "cap", "alias": "first"},
                {"operator": "obj", "when": "results.ghost_alias.ok", "alias": "c"},
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.steps[1].skipped is True
        assert any("ghost_alias" in e for e in result.errors)

    def test_when_run(self, registry, raw_node, ctx):
        # First step ok=True → second runs
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "cap", "alias": "good"},
                {"operator": "obj", "when": "results.good.ok", "alias": "obj_run"},
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.steps[1].skipped is False
        assert result.steps[1].extraction.extracted_content == {"items": [{"label": "cat"}]}

    def test_when_with_confidence(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "obj", "alias": "low"},   # confidence 0.7
                {"operator": "cap", "when": "results.low.confidence < 0.5", "alias": "nope"},
                {"operator": "cap", "when": "results.low.confidence > 0.5", "alias": "yes"},
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        skipped = {s.alias for s in result.steps if s.skipped}
        ran = {s.alias for s in result.steps if not s.skipped and s.alias}
        assert "nope" in skipped
        assert "yes" in ran

    def test_parallel_substep_when_sees_top_level_alias(
        self, registry, raw_node, ctx,
    ):
        """Parallel sub-step's `when` must see top-level results.

        Regression for: sub-step `when` evaluation was passed an empty
        dict, so any sub-step referencing ``results.<alias>`` was always
        skipped (or always raised "unknown alias"). Sub-steps should
        remain blind to *peer* sub-step results (concurrent), but they
        must see results already collected from earlier top-level steps.
        """
        # The fixture's "cap" extractor returns confidence 0.9 → ok=True.
        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "cap", "alias": "first"},
                {"parallel": [
                    # This sub-step gates on a top-level alias that
                    # DOES exist. With the bug it would be skipped
                    # (or raise "Unknown alias") because evaluation
                    # received {} instead of results_by_alias.
                    {"operator": "obj", "when": "results.first.ok", "alias": "gated"},
                    # A sub-step that intentionally does NOT have a
                    # when — should always run regardless.
                    {"operator": "face", "alias": "ungated"},
                ]},
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.ok
        parallel = result.steps[1]
        assert parallel.operator == "<parallel>"
        by_alias = {sub.alias: sub for sub in parallel.sub_results}
        assert "gated" in by_alias, "sub-step should run when top-level alias ok"
        assert by_alias["gated"].skipped is False
        assert by_alias["gated"].extraction.error is None
        assert "ungated" in by_alias
        assert by_alias["ungated"].skipped is False

    def test_unknown_operator_records_error(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({"steps": [{"operator": "ghost"}]})
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert not result.ok
        assert "ghost" in result.errors[0]
        sr = result.steps[0]
        assert sr.extraction.error is not None

    def test_operator_exception_does_not_abort(self, registry, raw_node, ctx):
        # Inject a bad operator that raises
        class _Boom:
            @property
            def supported_types(self): return [RawDataType.PROMPT]
            @property
            def extractor_type(self): return "boom"
            @property
            def schema_name(self): return "x"
            @property
            def info(self):
                from extraction.operators import default_info
                return default_info(self)
            def extract(self, raw_node, ctx):
                raise RuntimeError("kaboom")
        registry.register(_Boom())

        pipe = OperatorPipeline.from_spec({
            "steps": [
                {"operator": "boom"},
                {"operator": "cap"},  # must still run
            ]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert not result.ok
        assert len(result.steps) == 2
        assert result.steps[0].extraction.error is not None
        assert result.steps[1].ok  # second step is unaffected

    def test_duration_recorded(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({"steps": [{"operator": "cap"}]})
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.steps[0].duration_ms >= 0.0

    def test_params_stored_in_extracted_content(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"operator": "cap", "params": {"threshold": 0.7}}]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.steps[0].extraction.extracted_content["_params"] == {"threshold": 0.7}

    def test_result_to_dict(self, registry, raw_node, ctx):
        pipe = OperatorPipeline.from_spec({
            "steps": [{"operator": "cap", "alias": "a"}]
        })
        result = pipe.execute(raw_node, ctx, registry=registry)
        d = result.to_dict()
        assert d["raw_node_id"] == raw_node.node_id
        assert d["ok"] is True
        assert d["steps"][0]["operator"] == "cap"


class TestPipelineBuilder:
    def test_add_chainable(self, registry, raw_node, ctx):
        pipe = OperatorPipeline().add("cap", alias="c").add("obj")
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert len(result.steps) == 2

    def test_parallel_chainable(self, registry, raw_node, ctx):
        pipe = OperatorPipeline().parallel("face", "pose")
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert len(result.steps[0].sub_results) == 2

    def test_when_chainable(self, registry, raw_node, ctx):
        pipe = (
            OperatorPipeline()
            .add("cap", alias="c")
            .when("results.c.ok")
            .add("obj")
        )
        result = pipe.execute(raw_node, ctx, registry=registry)
        assert result.steps[1].skipped is False

    def test_when_without_step_raises(self):
        with pytest.raises(PipelineSpecError, match="no preceding step"):
            OperatorPipeline().when("results.x.ok")
