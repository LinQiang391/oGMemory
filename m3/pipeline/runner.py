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
from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY, StageFn

logger = logging.getLogger(__name__)

# Read-only view over the global default registry. Tests and plugin
# loaders can still mutate :data:`DEFAULT_STAGE_REGISTRY` directly, but
# runner instances that don't pass an explicit ``registry=`` get this
# frozen view so one test cannot accidentally poison the next.
DEFAULT_STAGE_REGISTRY_VIEW: Mapping[str, StageFn] = MappingProxyType(
    DEFAULT_STAGE_REGISTRY
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
    ) -> None:
        # Default to a frozen view over the module-level registry so
        # accidental mutation from one runner cannot leak into another.
        # Callers needing to add plugins for the lifetime of one run
        # should pass ``dict(DEFAULT_STAGE_REGISTRY, **overrides)``.
        self._registry: Mapping[str, StageFn] = registry or DEFAULT_STAGE_REGISTRY_VIEW

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
                    "--graph-store sql requires --connection-string or vector_db.connection_string in ogmemory.yaml"
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
