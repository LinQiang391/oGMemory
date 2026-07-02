"""Pipeline spec models and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageSpec:
    """Declarative description of one pipeline stage."""

    id: str
    type: str
    enabled: bool = True
    needs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageSpec":
        if not isinstance(data, dict):
            raise ValueError(f"stage must be a mapping, got {type(data).__name__}")
        stage_id = data.get("id")
        stage_type = data.get("type")
        if not stage_id or not isinstance(stage_id, str):
            raise ValueError("stage missing string id")
        if not stage_type or not isinstance(stage_type, str):
            raise ValueError(f"stage {stage_id!r} missing string type")
        needs = data.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        if not isinstance(needs, list) or not all(isinstance(x, str) for x in needs):
            raise ValueError(f"stage {stage_id!r} needs must be a list of strings")
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"stage {stage_id!r} params must be a mapping")
        outputs = data.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise ValueError(f"stage {stage_id!r} outputs must be a mapping")
        return cls(
            id=stage_id,
            type=stage_type,
            enabled=bool(data.get("enabled", True)),
            needs=list(needs),
            params=dict(params),
            outputs={str(k): str(v) for k, v in outputs.items()},
        )


@dataclass
class PipelineSpec:
    """Declarative M3 pipeline specification."""

    name: str
    version: int
    description: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    stages: list[StageSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineSpec":
        if not isinstance(data, dict):
            raise ValueError(f"pipeline spec must be a mapping, got {type(data).__name__}")
        name = data.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("pipeline spec missing string name")
        stages_data = data.get("stages") or []
        if not isinstance(stages_data, list):
            raise ValueError("pipeline spec stages must be a list")
        artifacts = data.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            raise ValueError("pipeline spec artifacts must be a mapping")
        return cls(
            name=name,
            version=int(data.get("version", 1)),
            description=str(data.get("description", "")),
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            stages=[StageSpec.from_dict(s) for s in stages_data],
        )


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load a pipeline spec from YAML."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineSpec.from_dict(data or {})
