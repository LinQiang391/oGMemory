"""Runtime context and result models for M3 pipelines."""

from __future__ import annotations

from collections.abc import Mapping
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

    def validate_against_registry(
        self,
        registry: Mapping[str, Any] | None,
    ) -> None:
        """Raise if ``stage_type`` is not present in ``registry``.

        Production code passes the runner's registry so a typo in a
        stage's ``type`` field surfaces here instead of silently
        producing an empty result. Tests can pass ``None`` to skip the
        check.
        """
        if registry is None:
            return
        if self.stage_type not in registry:
            raise ValueError(
                f"StageResult for {self.stage_id!r} references unknown "
                f"stage_type {self.stage_type!r}; expected one of "
                f"{sorted(registry)}"
            )

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
    # Runner-managed singletons. ``None`` by default so direct test
    # construction of PipelineContext still works (those tests build
    # their own MemoryGraphStore per fixture); the runner populates
    # these fields once at the start of PipelineRunner.run() so every
    # stage sees the same graph service. Without this, each stage's
    # parse_args/run call builds a fresh MemoryGraphStore and the
    # graph state from load_clips is invisible to extract_frames /
    # extract_audio / episodic_memory / etc.
    graph: Any = None
    op_service: Any = None
    # Vector index instance shared across the index_episodic and qa
    # stages. The runner constructs it once from the YAML config (so
    # it can pick up ``m3.vector_index.*`` overrides) and the qa stage
    # wires it into the GraphService so ``search_by_embedding`` routes
    # through HNSW + scope columns instead of JSONB. Tests that
    # construct PipelineContext directly can leave this ``None`` —
    # the affected stages (build_index, qa) raise a clear error
    # rather than silently degrading to the legacy store path.
    vector_index: Any = None

    def artifact(self, name: str, *, optional: bool = False) -> Path | None:
        try:
            return self.artifacts[name]
        except KeyError:
            if optional:
                return None
            raise KeyError(f"unknown pipeline artifact: {name}") from None
