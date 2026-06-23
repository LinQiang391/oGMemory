"""OperatorService — high-level facade for operator execution.

Responsibilities:

1. Resolve raw nodes by ID (via the existing ``GraphService``)
2. Run a single operator OR a pipeline of operators
3. Persist each step's result as an ``ExtractedNode`` in the
   heterogeneous graph, automatically creating the ``extracted_from``
   edge (so the existing retrieval / PageRank machinery works).
4. Track async tasks (via ``TaskStore``) and expose status queries.

The service is the only place that ties together the ``OperatorRegistry``,
``OperatorPipeline``, and ``GraphService``. Operators themselves do not
know about the graph; the graph does not know about operators.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from core.graph_enums import RawDataType
from core.graph_models import ExtractedNode, RawNode
from extraction.operators import (
    OperatorPipeline,
    OperatorRegistry,
    PipelineResult,
)
from extraction.operators.base import ExtractionResult
from extraction.operators.task_store import (
    InMemoryTaskStore,
    TaskRecord,
    TaskStore,
)

if TYPE_CHECKING:
    from core.models import RequestContext
    from service.graph_service import GraphService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OperatorServiceError(Exception):
    """Base class for OperatorService errors."""


class RawNodeNotFoundError(OperatorServiceError):
    """Raised when a referenced raw node does not exist."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OperatorService:
    """Coordinates operator execution, graph persistence, and async tracking.

    The service is stateless w.r.t. user data — it is safe to share
    one instance across all request threads.

    Args:
        graph_service: Used to read raw nodes and write extracted nodes.
        registry: Operator registry; defaults to ``default_registry()``.
        task_store: Backend for async task tracking; defaults to an
            in-memory store suitable for single-process deployments.
    """

    def __init__(
        self,
        graph_service: "GraphService",
        *,
        registry: OperatorRegistry | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        from extraction.operators import default_registry
        self._graph = graph_service
        self._registry = registry or default_registry()
        self._tasks = task_store or InMemoryTaskStore()
        self._async_lock = threading.Lock()
        self._async_threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def list_operators(
        self, raw_type: RawDataType | None = None,
    ) -> list[dict]:
        """Return public metadata for every registered operator."""
        return [
            {
                "operator_type": info.operator_type,
                "display_name": info.display_name,
                "description": info.description,
                "supported_raw_types": [t.value for t in info.supported_raw_types],
                "schema_name": info.schema_name,
                "version": info.version,
                "config_schema": info.config_schema,
            }
            for info in self._registry.list(raw_type=raw_type)
        ]

    def describe_operator(self, operator_type: str) -> dict:
        info = self._registry.get_info(operator_type)
        return {
            "operator_type": info.operator_type,
            "display_name": info.display_name,
            "description": info.description,
            "supported_raw_types": [t.value for t in info.supported_raw_types],
            "schema_name": info.schema_name,
            "version": info.version,
            "config_schema": info.config_schema,
        }

    def run_operator(
        self,
        raw_node_id: str,
        operator_type: str,
        ctx: "RequestContext",
        *,
        params: dict[str, Any] | None = None,
        write_to_graph: bool = True,
    ) -> dict:
        """Run a single operator against an existing raw node.

        Args:
            raw_node_id: ID of the raw node to extract from.
            operator_type: Registered operator id.
            ctx: RequestContext for access control.
            params: Optional per-call parameters (forwarded as ``**kwargs``
                to the extractor's ``extract`` method's `params` field).
            write_to_graph: If True (default), persist the result as
                an ExtractedNode and create the ``extracted_from`` edge.

        Returns:
            Dict with extraction details and (if written) the new
            extracted_node_id.
        """
        raw = self._get_raw_node(raw_node_id, ctx)
        operator = self._registry.get(operator_type)

        extraction = operator.extract(raw, ctx)
        # Apply per-call params to the extracted_content metadata
        if params and extraction.error is None:
            extraction.extracted_content.setdefault("_params", {}).update(params)

        result: dict[str, Any] = {
            "operator": operator_type,
            "raw_node_id": raw_node_id,
            "ok": extraction.error is None,
            "confidence": extraction.confidence,
            "extracted_content": extraction.extracted_content,
            "schema_used": extraction.schema_used,
            "error": extraction.error,
        }

        if write_to_graph and extraction.error is None:
            extracted = self._graph.create_extracted_node(
                source_raw_id=raw_node_id,
                extractor_type=operator_type,
                extraction_schema=extraction.schema_used,
                ctx=ctx,
                extracted_content=extraction.extracted_content,
            )
            result["extracted_node_id"] = extracted.node_id

        return result

    def run_pipeline(
        self,
        raw_node_id: str,
        pipeline_spec: dict,
        ctx: "RequestContext",
        *,
        write_to_graph: bool = True,
    ) -> dict:
        """Run a full pipeline of operators against a raw node.

        Args:
            raw_node_id: ID of the raw node to extract from.
            pipeline_spec: A JSON-serialisable pipeline spec (see
                :class:`OperatorPipeline` for the format).
            ctx: RequestContext.
            write_to_graph: Persist each step's result as an ExtractedNode.

        Returns:
            Dict with the full pipeline result (steps, errors, node IDs).
        """
        raw = self._get_raw_node(raw_node_id, ctx)
        pipe = OperatorPipeline.from_spec(pipeline_spec)
        result: PipelineResult = pipe.execute(raw, ctx, registry=self._registry)

        out = result.to_dict()
        if write_to_graph:
            out["created_nodes"] = self._persist_pipeline(result, ctx)
        return out

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    def run_pipeline_async(
        self,
        raw_node_id: str,
        pipeline_spec: dict,
        ctx: "RequestContext",
        *,
        write_to_graph: bool = True,
    ) -> str:
        """Run a pipeline on a background thread; return task_id.

        The caller can poll ``get_task_status(task_id)`` for the result.
        """
        rec = self._tasks.submit(
            raw_node_id=raw_node_id,
            operator_service="OperatorService.run_pipeline_async",
            trace_id=ctx.trace_id,
        )

        def _worker():
            self._tasks.mark_running(rec.task_id)
            try:
                result = self.run_pipeline(
                    raw_node_id, pipeline_spec, ctx,
                    write_to_graph=write_to_graph,
                )
                self._tasks.mark_completed(rec.task_id, result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Async pipeline failed for task %s", rec.task_id)
                self._tasks.mark_failed(rec.task_id, str(exc))
            finally:
                # Drop the thread reference — without this, _async_threads
                # grows unboundedly with each run_pipeline_async call.
                with self._async_lock:
                    self._async_threads.pop(rec.task_id, None)
                # Reap terminal records older than the cutoff so the
                # task store doesn't accumulate stale entries forever.
                # Swallow prune errors — a failure here must not crash
                # the worker thread, which is the only owner of the task.
                try:
                    self._tasks.prune()
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "prune() failed in worker cleanup for %s", rec.task_id,
                    )

        t = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"op-pipeline-{rec.task_id[:8]}",
        )
        with self._async_lock:
            self._async_threads[rec.task_id] = t
        t.start()
        return rec.task_id

    def get_task_status(self, task_id: str) -> dict | None:
        rec = self._tasks.get(task_id)
        return rec.to_dict() if rec else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_raw_node(
        self, raw_node_id: str, ctx: "RequestContext"
    ) -> RawNode:
        node = self._graph.get_raw_node(raw_node_id, ctx)
        if node is None:
            raise RawNodeNotFoundError(f"Raw node not found: {raw_node_id}")
        return node

    def _persist_pipeline(
        self, result: PipelineResult, ctx: "RequestContext",
    ) -> list[dict]:
        """Write each step's extraction to the graph; return node IDs.

        Skips skipped steps and steps whose extraction raised an error
        (since writing a failed extraction would be misleading).
        """
        created: list[dict] = []
        for step in result.steps:
            if step.skipped:
                continue
            if step.extraction.error is not None:
                continue
            # For a parallel step, write each sub-step
            if step.sub_results:
                for sub in step.sub_results:
                    if sub.skipped or sub.extraction.error is not None:
                        continue
                    self._write_extracted(
                        raw_id=result.raw_node_id,
                        operator_type=sub.operator,
                        schema_used=sub.extraction.schema_used,
                        extracted_content=sub.extraction.extracted_content,
                        ctx=ctx,
                        out=created,
                    )
            else:
                self._write_extracted(
                    raw_id=result.raw_node_id,
                    operator_type=step.operator,
                    schema_used=step.extraction.schema_used,
                    extracted_content=step.extraction.extracted_content,
                    ctx=ctx,
                    out=created,
                )
        return created

    def _write_extracted(
        self,
        *,
        raw_id: str,
        operator_type: str,
        schema_used: str,
        extracted_content: dict,
        ctx: "RequestContext",
        out: list[dict],
    ) -> None:
        # Skip the synthetic "<parallel>" wrapper label
        if operator_type == "<parallel>":
            return
        extracted = self._graph.create_extracted_node(
            source_raw_id=raw_id,
            extractor_type=operator_type,
            extraction_schema=schema_used,
            ctx=ctx,
            extracted_content=extracted_content,
        )
        out.append({
            "operator": operator_type,
            "extracted_node_id": extracted.node_id,
            "schema_used": schema_used,
        })
