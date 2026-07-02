"""Runtime context and result models for M3 pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    """Structured outcome of one pipeline stage."""

    stage_id: str
    stage_type: str
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "ok": self.ok,
            "outputs": self.outputs,
            "counts": self.counts,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class PipelineContext:
    """Shared state for a pipeline run."""

    video: Path
    video_id: str
    seconds: int
    work_dir: Path
    config_path: Path
    graph_store: str
    connection_string: str | None
    account_id: str
    artifacts: dict[str, Path]
    variables: dict[str, Any]
    stage_results: dict[str, StageResult] = field(default_factory=dict)

    def artifact(self, name: str) -> Path:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"unknown pipeline artifact: {name}") from exc
