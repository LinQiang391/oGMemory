"""Pipeline execution engine."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from m3.config import load_m3_runtime_config
from m3.pipeline.context import PipelineContext, StageResult
from m3.pipeline.spec import PipelineSpec, StageSpec
from m3.pipeline.stages import (
    DEFAULT_STAGE_PROBES,
    DEFAULT_STAGE_REGISTRY,
    ProbeFn,
    ProbeResult,
    StageFn,
    _safe_probe,
)

logger = logging.getLogger(__name__)


def _build_vector_index(ctx: PipelineContext):
    """Resolve a :class:`VectorIndex` for the pipeline from the YAML config.

    Reads ``m3.vector_index.*`` (backend / table_name / pool_size) plus
    the shared ``vector_db.connection_string`` via
    :func:`m3.config.load_m3_runtime_config`, then delegates to
    :func:`m3.vector_index.build_m3_vector_index` for backend selection.

    Returns ``None`` (not an exception) when no connection string
    is configured. The runner treats ``None`` as "skip the vector
    index; let ``GraphService.search_by_embedding`` fall back to
    the legacy JSONB path." This matches what ``--graph-store
    memory`` runs / dev smoke runs / unit tests actually need:
    they have no DB to point the index at, and forcing one would
    either crash loudly on a missing env var or trip a psycopg2
    error on a malformed placeholder DSN. The previous design
    required every pipeline run to set up a DB, which is not
    what the user asked for when they typed ``--graph-store
    memory``.

    The genuine config bugs (wrong backend name, missing
    ``embedding_dimension``) still surface as ``ValueError``
    from ``build_m3_vector_index`` because those are user-fixable
    config issues that shouldn't be silently swallowed.
    """
    from m3.config import load_m3_runtime_config
    from m3.vector_index import build_m3_vector_index

    runtime = load_m3_runtime_config(str(ctx.config_path))

    # Graph storage and vector storage have independent connections.
    # ctx.connection_string belongs exclusively to the graph store.
    if not runtime.vector_index_connection_string:
        logger.info(
            "Pipeline vector index skipped: configure vector_db.connection_string. "
            "GraphService.search_by_embedding falls back to the legacy JSONB path."
        )
        return None

    return build_m3_vector_index(runtime)

# Read-only view over the global default registry. Tests and plugin
# loaders can still mutate :data:`DEFAULT_STAGE_REGISTRY` directly, but
# runner instances that don't pass an explicit ``registry=`` get this
# frozen view so one test cannot accidentally poison the next.
DEFAULT_STAGE_REGISTRY_VIEW: Mapping[str, StageFn] = MappingProxyType(
    DEFAULT_STAGE_REGISTRY
)

# Parallel read-only view for the probe registry. Same semantics as
# :data:`DEFAULT_STAGE_REGISTRY_VIEW`: plugin loaders can extend
# :data:`DEFAULT_STAGE_PROBES`, but runner instances default to this
# frozen view.
DEFAULT_STAGE_PROBES_VIEW: Mapping[str, ProbeFn] = MappingProxyType(
    DEFAULT_STAGE_PROBES
)


class _FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _coerce_value(value: str) -> Any:
    """Best-effort string-to-typed coercion for ``--set`` overrides.

    Recognizes common CLI bool spellings (``true/false/yes/no/on/off``),
    the JSON/YAML null forms, ints, and floats. Anything more complex
    (lists, dicts) is parsed as a YAML scalar so callers can express
    structured values like ``--set foo=[1,2,3]`` without bespoke syntax.
    """
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # YAML-style scalars (lists, dicts, etc.).
    import yaml

    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    if isinstance(parsed, (list, dict)):
        return parsed
    return value


def _format_value(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_FormatDict({k: str(v) for k, v in variables.items()}))
    if isinstance(value, dict):
        return {k: _format_value(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_value(v, variables) for v in value]
    return value


def set_nested_param(stage: StageSpec, dotted_path: str, value: Any) -> None:
    """Set a nested stage field used by --set overrides.

    Supports paths like ``params.max_frames`` or ``enabled``.
    """
    if dotted_path == "enabled":
        stage.enabled = bool(value)
        return
    if dotted_path.startswith("params."):
        key_path = dotted_path[len("params."):].split(".")
        target = stage.params
        for key in key_path[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot set nested param {dotted_path!r}; {key!r} is not a mapping")
            target = child
        target[key_path[-1]] = value
        return
    raise ValueError(f"unsupported stage override path: {dotted_path}")


class PipelineRunner:
    """Execute a declarative M3 pipeline spec."""

    def __init__(
        self,
        registry: Mapping[str, StageFn] | None = None,
        probes: Mapping[str, ProbeFn] | None = None,
        smart_resume: bool = True,
    ) -> None:
        # Default to a frozen view over the module-level registry so
        # accidental mutation from one runner cannot leak into another.
        # Callers needing to add plugins for the lifetime of one run
        # should pass ``dict(DEFAULT_STAGE_REGISTRY, **overrides)``.
        self._registry: Mapping[str, StageFn] = registry or DEFAULT_STAGE_REGISTRY_VIEW
        self._probes: Mapping[str, ProbeFn] = probes or DEFAULT_STAGE_PROBES_VIEW
        # When False the runner skips the probe layer entirely and
        # always invokes the stage function. CLI plumbing flips this
        # off when ``--no-smart-resume`` is passed. ``--force`` keeps
        # the existing ``--force`` semantics for individual drivers
        # (e.g. voice_asr) and is orthogonal: smart-resume is a runner
        # shortcut; a stage's params.force bypasses both its probe and its
        # driver's dedup. The outer benchmark --force is a separate control.
        self._smart_resume = bool(smart_resume)

    def build_context(
        self,
        spec: PipelineSpec,
        *,
        video: Path,
        video_id: str,
        seconds: int,
        work_dir: Path,
        config_path: Path,
        graph_store: str,
        connection_string: str | None,
        account_id: str,
        variables: dict[str, Any] | None = None,
    ) -> PipelineContext:
        if graph_store == "sql" and not connection_string:
            connection_string = load_m3_runtime_config(str(config_path)).graph_connection_string
            if not connection_string:
                raise RuntimeError(
                    "--graph-store sql requires --connection-string, or "
                    "m3.graph_store.connection_string / storage.connection_string "
                    "set in ogmemory.yaml (graph store defaults to STORAGE_DB; "
                    "see m3/config.py for the resolution order)"
                )

        base_vars: dict[str, Any] = {
            "video": str(video),
            "video_id": video_id,
            "seconds": seconds,
            "work_dir": str(work_dir),
            "config": str(config_path),
            "graph_store": graph_store,
            "connection_string": connection_string or "",
            "account_id": account_id,
        }
        if variables:
            base_vars.update(variables)

        artifacts = {
            name: Path(_format_value(template, base_vars)).resolve()
            for name, template in spec.artifacts.items()
        }
        base_vars.update({name: str(path) for name, path in artifacts.items()})
        return PipelineContext(
            video=video,
            video_id=video_id,
            seconds=seconds,
            work_dir=work_dir,
            config_path=config_path,
            graph_store=graph_store,
            connection_string=connection_string,
            account_id=account_id,
            artifacts=artifacts,
            variables=base_vars,
        )

    def apply_overrides(
        self,
        spec: PipelineSpec,
        *,
        enable_stages: list[str] | None = None,
        disable_stages: list[str] | None = None,
        set_values: list[str] | None = None,
    ) -> None:
        stages = {s.id: s for s in spec.stages}
        available = sorted(stages)
        for stage_id in enable_stages or []:
            if stage_id not in stages:
                raise ValueError(
                    f"unknown stage in --enable-stage: {stage_id!r}; "
                    f"available stages: {available}"
                )
            stages[stage_id].enabled = True
        for stage_id in disable_stages or []:
            if stage_id not in stages:
                raise ValueError(
                    f"unknown stage in --disable-stage: {stage_id!r}; "
                    f"available stages: {available}"
                )
            stages[stage_id].enabled = False
        for item in set_values or []:
            if "=" not in item:
                raise ValueError(
                    f"--set expects stage.path=value, got {item!r}"
                )
            lhs, raw_value = item.split("=", 1)
            # Reject accidental double '=' (e.g. --set foo=bar=baz) by
            # flagging when the LHS itself contains '=' after the split
            # boundary. argparse hands us the full post-`=` payload, so
            # an extra '=' is part of ``raw_value`` and silently typed
            # as a string. We surface it explicitly.
            if "=" in raw_value:
                raise ValueError(
                    f"--set value for {lhs!r} contains stray '='; "
                    f"got {item!r}"
                )
            parts = lhs.split(".", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"--set expects stage.path=value, got {item!r}; "
                    f"available stages: {available}"
                )
            stage_id, path = parts
            if stage_id not in stages:
                raise ValueError(
                    f"unknown stage in --set: {stage_id!r}; "
                    f"available stages: {available}"
                )
            set_nested_param(stages[stage_id], path, _coerce_value(raw_value))

    def validate(self, spec: PipelineSpec) -> None:
        """Validate ``spec`` against this runner's stage registry.

        Thin wrapper that forwards to :meth:`PipelineSpec.validate`
        so the same structural checks run whether callers validate
        ahead of time or rely on :meth:`run` to do it.
        """
        spec.validate(registry=self._registry)

    def run(self, spec: PipelineSpec, ctx: PipelineContext) -> dict[str, Any]:
        self.validate(spec)
        if ctx.video.exists():
            from m3.pipeline.provenance import validate_input
            validate_input(ctx, spec)
        # Build the graph service + OperatorService ONCE here so every
        # stage sees the same instance. Without this each driver's
        # parse_args/run call builds its own MemoryGraphStore() and the
        # graph state from load_clips is invisible to extract_frames /
        # extract_audio / episodic_memory / etc. SQL connections also
        # collapse to one pool instead of one per stage.
        if ctx.graph is None:
            from m3.load_clips import build_graph_service
            # Build the VectorIndex BEFORE the graph service so we can
            # inject it via build_graph_service(..., vector_index=...).
            # The qa stage needs the same instance as the
            # index_episodic stage so embeddings written there are
            # visible to GraphService.search_by_embedding later.
            #
            # Guard: only attempt VectorIndex construction when the
            # graph store actually hosts a database that can hold one.
            # For ``graph_store="memory"`` (the dev / smoke default)
            # there's no DB to point at — building a VectorIndex
            # would force the user to set a connection string they
            # don't otherwise need, and would fail loudly with a
            # psycopg2 error the moment the YAML points at a
            # placeholder DSN. Pre-this guard, ``m3 run --pipeline
            # …`` without OGMEM_GRAPH_CONNECTION_STRING crashed here
            # for memory-store runs that previously worked.
            if ctx.vector_index is None and ctx.graph_store == "sql":
                ctx.vector_index = _build_vector_index(ctx)
                if ctx.vector_index is not None:
                    logger.info(
                        "Pipeline vector index: %s",
                        type(ctx.vector_index).__name__,
                    )
            graph, store_kind = build_graph_service(
                ctx.graph_store, ctx.connection_string,
                vector_index=ctx.vector_index,
            )
            ctx.graph = graph
            logger.info("Pipeline graph service: %s", store_kind)
        if ctx.op_service is None:
            from extraction.operators import default_registry
            from extraction.operators.builtin import register_all_defaults
            from service.operator_service import OperatorService
            registry = default_registry()
            register_all_defaults(registry)
            ctx.op_service = OperatorService(ctx.graph, registry=registry)
        for stage in spec.stages:
            if not stage.enabled:
                ctx.stage_results[stage.id] = StageResult(stage.id, stage.type, ok=True, counts={"skipped": 1})
                logger.info("Skipping disabled stage %s (%s)", stage.id, stage.type)
                continue
            for need in stage.needs:
                needed = ctx.stage_results.get(need)
                if needed is None:
                    raise RuntimeError(f"stage {stage.id} cannot run before dependency {need}")
                if not needed.ok:
                    raise RuntimeError(f"stage {stage.id} dependency {need} failed")
            params = _format_value(stage.params, ctx.variables)
            effective = StageSpec(
                id=stage.id,
                type=stage.type,
                enabled=stage.enabled,
                needs=list(stage.needs),
                params=params,
                outputs=dict(stage.outputs),
            )

            # Smart-resume probe (Layer 2): the per-stage probe inspects
            # the graph + artefacts and decides whether the stage is
            # already complete. When ``complete=True`` we short-circuit
            # the expensive stage function and log a skip. Probes run
            # only when smart_resume is enabled (CLI: --no-smart-resume
            # disables); probes are wrapped to swallow exceptions so a
            # buggy probe never aborts the pipeline.
            if self._smart_resume and not ctx.variables.get("force") and not effective.params.get("force"):
                probe = self._probes.get(effective.type)
                if probe is not None:
                    wrapped = _safe_probe(probe)
                    try:
                        probe_result = wrapped(ctx, effective, ctx.graph)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "probe for stage %s raised %s; running stage anyway",
                            effective.id, exc,
                        )
                        probe_result = ProbeResult(complete=False, reason=f"probe error: {exc}")
                    if probe_result.complete:
                        logger.info(
                            "Skipping stage %s (%s): %s",
                            effective.id, effective.type, probe_result.reason,
                        )
                        ctx.stage_results[stage.id] = StageResult(
                            stage_id=effective.id,
                            stage_type=effective.type,
                            ok=True,
                            counts={
                                "skipped": 1,
                                "probe_reason_count": probe_result.count,
                            },
                            warnings=[f"smart-resume skip: {probe_result.reason}"],
                        )
                        continue

            logger.info("Running stage %s (%s)", effective.id, effective.type)
            try:
                result = self._registry[effective.type](ctx, effective)
            except Exception as exc:
                result = StageResult(
                    stage_id=effective.id,
                    stage_type=effective.type,
                    ok=False,
                    errors=[{"error": str(exc)}],
                )
                ctx.stage_results[stage.id] = result
                raise
            ctx.stage_results[stage.id] = result
            if not result.ok:
                # A returned failure is as terminal as an exception, but keep
                # its structured diagnostics and persist the summary below.
                logger.error("Stage %s failed; stopping pipeline", stage.id)
                break

        summary = self.summary(spec, ctx)
        summary_path = ctx.artifacts.get("summary_json") or (ctx.work_dir / "pipeline_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    @staticmethod
    def summary(spec: PipelineSpec, ctx: PipelineContext) -> dict[str, Any]:
        face_result = ctx.stage_results.get("detect_faces")
        index_result = ctx.stage_results.get("build_index")
        index_summary = None
        if index_result is not None and not index_result.counts.get("skipped"):
            index_summary = {"index_jsonl": str(ctx.artifacts.get("index_jsonl", ""))}
        return {
            "pipeline": {"name": spec.name, "version": spec.version, "description": spec.description},
            "video_id": ctx.video_id,
            "selected_video": str(ctx.artifacts.get("selected_video", "")),
            "annotations": str(ctx.artifacts.get("annotations", "")),
            "data_jsonl": str(ctx.artifacts.get("data_jsonl", "")),
            "frames_jsonl": str(ctx.artifacts.get("frames_jsonl", "")),
            "faces": face_result.to_dict() if face_result is not None else None,
            "episodic_jsonl": str(ctx.artifacts.get("episodic_jsonl", "")),
            "memory": ctx.stage_results.get("create_memory_nodes", StageResult("create_memory_nodes", "m3.memory.from_episodic", True)).counts if "create_memory_nodes" in ctx.stage_results else None,
            "index": index_summary,
            "artifacts": {k: str(v) for k, v in ctx.artifacts.items()},
            "stages": {k: v.to_dict() for k, v in ctx.stage_results.items()},
        }
