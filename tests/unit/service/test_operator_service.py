"""Unit tests for OperatorService."""

from __future__ import annotations

import hashlib
import time

import pytest

from core.graph_enums import ExtractionStatus, GraphEdgeType, RawDataType
from core.models import RequestContext
from extraction.operators import (
    ExtractionResult,
    OperatorRegistry,
)
from extraction.operators.base import OperatorInfo, default_info
from extraction.operators.builtin import (
    register_all_defaults,
)
from extraction.operators.materialize import MaterializationPolicy
from extraction.operators.task_store import InMemoryTaskStore
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import GraphService
from service.operator_service import (
    OperatorService,
    RawNodeNotFoundError,
)


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


@pytest.fixture
def store():
    return MemoryGraphStore()


@pytest.fixture
def graph(store):
    return GraphService(store)


@pytest.fixture
def registry():
    reg = OperatorRegistry()
    register_all_defaults(reg)
    return reg


@pytest.fixture
def image_raw_node(graph, ctx):
    return graph.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"fake-image-bytes",
        ctx=ctx,
    )


@pytest.fixture
def document_raw_node(graph, ctx):
    return graph.create_raw_node(
        raw_type=RawDataType.DOCUMENT,
        content=(
            "# Intro\nThis is the introduction with enough text.\n\n"
            "Body:\nThis is the body section with more text.\n"
        ),
        ctx=ctx,
    )


@pytest.fixture
def service(graph, registry):
    return OperatorService(
        graph_service=graph,
        registry=registry,
        task_store=InMemoryTaskStore(),
    )


# ---------------------------------------------------------------------------
# Listing / describing
# ---------------------------------------------------------------------------

class TestListAndDescribe:
    def test_list_operators(self, service):
        ops = service.list_operators()
        assert len(ops) >= 5
        types = {op["operator_type"] for op in ops}
        assert "face_detection" in types
        assert "object_detection" in types
        assert "image_caption" in types

    def test_list_operators_filter_by_raw_type(self, service):
        img_ops = service.list_operators(raw_type=RawDataType.IMAGE)
        for op in img_ops:
            assert "image" in op["supported_raw_types"]
        doc_ops = service.list_operators(raw_type=RawDataType.DOCUMENT)
        for op in doc_ops:
            assert "document" in op["supported_raw_types"]

    def test_describe_operator(self, service):
        info = service.describe_operator("face_detection")
        assert info["operator_type"] == "face_detection"
        assert info["config_schema"] is not None
        assert "min_confidence" in info["config_schema"]["properties"]

    def test_describe_unknown_raises(self, service):
        from extraction.operators import OperatorRegistryError
        with pytest.raises(OperatorRegistryError):
            service.describe_operator("ghost")


# ---------------------------------------------------------------------------
# Single operator
# ---------------------------------------------------------------------------

class TestRunOperator:
    def test_run_face_detection(self, service, image_raw_node, ctx):
        result = service.run_operator(
            image_raw_node.node_id, "face_detection", ctx,
        )
        assert result["operator"] == "face_detection"
        assert result["ok"] is True
        assert "face_count" in result["extracted_content"]
        # Graph write should produce an ExtractedNode
        assert "extracted_node_id" in result

    def test_run_with_params(self, service, image_raw_node, ctx):
        result = service.run_operator(
            image_raw_node.node_id, "face_detection", ctx,
            params={"min_confidence": 0.5, "max_faces": 1},
        )
        assert result["ok"] is True
        # params are stored under _params metadata
        assert result["extracted_content"]["_params"]["max_faces"] == 1

    def test_run_dry_run_no_write(self, service, image_raw_node, ctx):
        result = service.run_operator(
            image_raw_node.node_id, "face_detection", ctx,
            write_to_graph=False,
        )
        assert "extracted_node_id" not in result

    def test_run_creates_extracted_node_with_edge(
        self, service, graph, image_raw_node, ctx,
    ):
        result = service.run_operator(
            image_raw_node.node_id, "object_detection", ctx,
        )
        extracted = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert extracted is not None
        assert extracted.extractor_type == "object_detection"
        assert extracted.source_raw_id == image_raw_node.node_id

    def test_run_unknown_raw_node(self, service, ctx):
        with pytest.raises(RawNodeNotFoundError):
            service.run_operator("missing", "face_detection", ctx)

    def test_run_unknown_operator(self, service, image_raw_node, ctx):
        from extraction.operators import OperatorRegistryError
        with pytest.raises(OperatorRegistryError):
            service.run_operator(
                image_raw_node.node_id, "ghost", ctx,
            )

    def test_run_on_document_raw_node(self, service, document_raw_node, ctx):
        result = service.run_operator(
            document_raw_node.node_id, "document_structure", ctx,
        )
        assert result["ok"] is True
        assert result["extracted_content"]["section_count"] >= 1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @pytest.mark.parametrize("parallel", [False, True])
    def test_successful_extractions_are_completed(self, service, graph, image_raw_node, ctx, parallel):
        steps = [{"operator": "image_caption"}, {"operator": "face_detection"}]
        result = service.run_pipeline(
            image_raw_node.node_id,
            {"steps": [{"parallel": steps}] if parallel else steps},
            ctx,
        )
        actual_steps = result["steps"][0]["parallel"] if parallel else result["steps"]
        for created, step in zip(result["created_nodes"], actual_steps):
            node = graph.get_extracted_node(created["extracted_node_id"], ctx)
            assert node.extraction_status == ExtractionStatus.COMPLETED
            assert node.confidence == step["confidence"]
            assert node.extracted_content == step["extracted_content"]

    def test_sequential_pipeline(
        self, service, image_raw_node, ctx,
    ):
        spec = {
            "steps": [
                {"operator": "image_caption", "alias": "cap"},
                {"operator": "face_detection", "alias": "faces"},
            ]
        }
        result = service.run_pipeline(
            image_raw_node.node_id, spec, ctx,
        )
        assert result["ok"] is True
        # Both steps created ExtractedNodes
        assert len(result["created_nodes"]) == 2
        ops = {n["operator"] for n in result["created_nodes"]}
        assert ops == {"image_caption", "face_detection"}

    def test_parallel_pipeline(
        self, service, image_raw_node, ctx,
    ):
        spec = {
            "steps": [{"parallel": [
                {"operator": "face_detection"},
                {"operator": "pose_detection"},
            ]}]
        }
        result = service.run_pipeline(
            image_raw_node.node_id, spec, ctx,
        )
        # Two ExtractedNodes — one per parallel sub-step
        assert len(result["created_nodes"]) == 2
        ops = {n["operator"] for n in result["created_nodes"]}
        assert ops == {"face_detection", "pose_detection"}

    def test_conditional_skip(
        self, service, image_raw_node, ctx,
    ):
        # Reference an unknown alias → second step skipped, no node created
        spec = {
            "steps": [
                {"operator": "image_caption", "alias": "c"},
                {"operator": "face_detection", "when": "results.missing.ok"},
            ]
        }
        result = service.run_pipeline(
            image_raw_node.node_id, spec, ctx,
        )
        # Only the caption step should have created a node
        assert len(result["created_nodes"]) == 1
        assert result["created_nodes"][0]["operator"] == "image_caption"
        # The skipped step shows up in the result with skipped=True
        assert result["steps"][1]["skipped"] is True

    def test_pipeline_dry_run(self, service, image_raw_node, ctx):
        spec = {"steps": [{"operator": "face_detection"}]}
        result = service.run_pipeline(
            image_raw_node.node_id, spec, ctx, write_to_graph=False,
        )
        assert "created_nodes" not in result or not result.get("created_nodes")

    def test_pipeline_error_step(self, service, registry, image_raw_node, ctx):
        # Add a faulty operator
        class _Boom:
            @property
            def supported_types(self): return [RawDataType.IMAGE]
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
        registry.register(_Boom(), override=True)

        spec = {
            "steps": [
                {"operator": "boom"},
                {"operator": "face_detection"},
            ]
        }
        result = service.run_pipeline(
            image_raw_node.node_id, spec, ctx,
        )
        # First step errored; second still ran
        assert not result["ok"]
        # Only the second step created a graph node (error step is skipped)
        ops = {n["operator"] for n in result["created_nodes"]}
        assert "face_detection" in ops
        assert "boom" not in ops


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

class TestAsyncPipeline:
    def test_run_async_returns_task_id(self, service, image_raw_node, ctx):
        spec = {"steps": [{"operator": "face_detection", "alias": "f"}]}
        task_id = service.run_pipeline_async(
            image_raw_node.node_id, spec, ctx,
        )
        assert task_id

        # Poll until done (or timeout)
        for _ in range(50):
            rec = service.get_task_status(task_id, ctx)
            if rec and rec["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Async task did not complete in time")

        assert rec["status"] == "completed"
        assert rec["account_id"] == ctx.account_id
        assert rec["result"]["ok"] is True
        assert len(rec["result"]["created_nodes"]) == 1

    def test_unknown_task_returns_none(self, service, ctx):
        assert service.get_task_status("ghost", ctx) is None

    @pytest.mark.parametrize("owner", ["other-account", ""])
    @pytest.mark.parametrize("status", ["processing", "completed", "failed"])
    def test_task_status_denies_foreign_and_unowned_tasks(self, service, ctx, owner, status):
        rec = service._tasks.submit(
            raw_node_id="private", operator_service="test", account_id=owner,
        )
        rec.status = status
        rec.result = {"secret": "private result"}
        rec.error = "private error"
        assert service.get_task_status(rec.task_id, ctx) is None

    def test_async_error_captured(self, service, registry, image_raw_node, ctx):
        # Make a registry that throws when asked
        class _Boom:
            @property
            def supported_types(self): return [RawDataType.IMAGE]
            @property
            def extractor_type(self): return "boom"
            @property
            def schema_name(self): return "x"
            @property
            def info(self):
                from extraction.operators import default_info
                return default_info(self)
            def extract(self, raw_node, ctx):
                raise RuntimeError("always fails")
        registry.register(_Boom(), override=True)
        spec = {"steps": [{"operator": "boom"}]}
        task_id = service.run_pipeline_async(
            image_raw_node.node_id, spec, ctx,
        )
        for _ in range(50):
            rec = service.get_task_status(task_id, ctx)
            if rec and rec["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Async task did not finish")
        assert rec["status"] == "completed"
        # The error step itself does not bubble up as a service failure
        # — it's captured in the pipeline result
        assert rec["result"]["ok"] is False

    def test_async_threads_cleared_after_completion(
        self, service, image_raw_node, ctx,
    ):
        """_async_threads must be drained after the worker finishes.

        Regression for the leak where every run_pipeline_async call
        appended a Thread reference and never removed it, so long-running
        services accumulated Thread objects indefinitely.
        """
        spec = {"steps": [{"operator": "face_detection"}]}
        for _ in range(3):
            task_id = service.run_pipeline_async(
                image_raw_node.node_id, spec, ctx,
            )
            for _ in range(50):
                rec = service.get_task_status(task_id, ctx)
                if rec and rec["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            else:
                pytest.fail("Async task did not complete in time")
            assert rec["status"] == "completed"
            # Give the finally-block a moment to release the lock and
            # pop the entry. The thread has already exited, so this is
            # just about the bookkeeping happening in the parent.
            for _ in range(20):
                if task_id not in service._async_threads:
                    break
                time.sleep(0.01)
            assert task_id not in service._async_threads
        # And the dict is fully empty after a small batch.
        assert service._async_threads == {}

    def test_completed_tasks_get_pruned(
        self, service, image_raw_node, ctx,
    ):
        """The worker's finally-block must call prune() on the task store.

        Regression for: InMemoryTaskStore.prune() existed but was never
        called, so terminal task records (completed/failed) accumulated
        indefinitely in the in-memory store.
        """
        spec = {"steps": [{"operator": "face_detection"}]}

        # Plant a stale record with an ancient finished_at — this is
        # the kind of entry prune() is supposed to reclaim.
        stale = service._tasks.submit(
            raw_node_id="stale", operator_service="test",
        )
        stale.finished_at = "2000-01-01T00:00:00+00:00"
        stale.status = "completed"
        assert service._tasks.get(stale.task_id) is stale

        # Run a fresh async task. Its worker's finally-block invokes
        # prune(); with the default 24h cutoff, the stale record (and
        # only the stale record) is reaped.
        task_id = service.run_pipeline_async(
            image_raw_node.node_id, spec, ctx,
        )
        for _ in range(50):
            rec = service.get_task_status(task_id, ctx)
            if rec and rec["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Async task did not complete in time")

        # The stale record has been pruned; the fresh one remains.
        assert service._tasks.get(stale.task_id) is None
        assert service._tasks.get(task_id) is not None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------
#

def _item_id(item, source_raw_id, source_metadata):
    seed = f"item/{source_raw_id}/{item['i']}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _item_payload(item, source_raw_id, source_metadata):
    md = {"i": item["i"], "label": item.get("label", "")}
    if source_metadata is not None:
        md["source_video_id"] = source_metadata.get("video_id")
    return md


def _item_edge_props(item, source_raw_id, source_metadata):
    return {"relation": "has_item", "i": item["i"]}


class _ItemsExtractor:
    """Synthetic operator that returns ``extracted_content["items"]`` so
    the materialization path runs. Test fixture only; not registered
    globally."""

    extractor_type = "items_extractor"
    schema_name = "items_extractor_v1"
    supported_types = [RawDataType.VIDEO]

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.materialization = MaterializationPolicy(
            child_node_kind="raw",
            child_raw_type=RawDataType.IMAGE,
            collection_field="items",
            child_id_fn=_item_id,
            child_payload_fn=_item_payload,
            edge_type=GraphEdgeType.REFERENCES,
            edge_properties_fn=_item_edge_props,
        )
        return info

    def extract(self, raw_node, ctx):
        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content={
                "summary": "synthetic items",
                "items": [
                    {"i": 0, "label": "first"},
                    {"i": 1, "label": "second"},
                    {"i": 2, "label": "third"},
                ],
            },
            confidence=1.0,
            schema_used=self.schema_name,
        )


class TestRunOperatorMaterialization:
    """End-to-end check that OperatorService.run_operator honors an
    operator's ``MaterializationPolicy``: spawns child ``RawNode``s,
    creates the ``REFERENCES`` edge, and writes a lightweight summary
    (no per-item collection) on the parent ``ExtractedNode``.
    """

    @pytest.fixture
    def items_service(self, graph, registry):
        registry.register(_ItemsExtractor(), override=True)
        return OperatorService(
            graph_service=graph,
            registry=registry,
            task_store=InMemoryTaskStore(),
        )

    @pytest.fixture
    def video_node(self, graph, ctx):
        return graph.create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="",
            ctx=ctx,
            metadata={"video_id": "v1", "clip_id": 0, "source_path": "/x.mp4"},
            node_id="items-src-1",
        )

    def test_run_operator_creates_child_nodes(
        self, items_service, graph, ctx, video_node,
    ):
        result = items_service.run_operator(
            video_node.node_id, "items_extractor", ctx,
        )
        assert result["ok"] is True
        materialized = result["materialized_children"]
        assert len(materialized) == 3
        # All resumed=False because we just created them.
        assert all(not m["resumed"] for m in materialized)
        # All edge ids are non-empty.
        assert all(m["edge_id"] for m in materialized)
        # Every child ID corresponds to a real IMAGE RawNode in the graph.
        for m in materialized:
            node = graph.get_raw_node(m["child_node_id"], ctx)
            assert node is not None
            assert node.raw_type == RawDataType.IMAGE
            # source_video_id came from source_metadata via the policy fn
            assert node.metadata["source_video_id"] == "v1"

    def test_parent_extracted_node_has_lightweight_summary(
        self, items_service, graph, ctx, video_node,
    ):
        result = items_service.run_operator(
            video_node.node_id, "items_extractor", ctx,
        )
        extracted = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert extracted is not None
        content = extracted.extracted_content
        # Heavy per-item collection is stripped from the persisted
        # ExtractedNode — only child IDs are kept.
        assert "items" not in content
        assert content["child_node_ids"] == [
            m["child_node_id"] for m in result["materialized_children"]
        ]
        assert content["child_count"] == 3
        # Other top-level keys are preserved.
        assert content["summary"] == "synthetic items"

    def test_creates_references_edges_to_each_child(
        self, items_service, graph, ctx, video_node,
    ):
        result = items_service.run_operator(
            video_node.node_id, "items_extractor", ctx,
        )
        neighbors = graph.get_neighbors(
            video_node.node_id, ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        assert len(neighbors) == 3
        # All neighbors should be the children we just created.
        neighbor_ids = {n[0].node_id for n in neighbors}
        expected = {m["child_node_id"] for m in result["materialized_children"]}
        assert neighbor_ids == expected

    def test_idempotent_rerun_resumes_existing_children(
        self, items_service, graph, ctx, video_node,
    ):
        first = items_service.run_operator(
            video_node.node_id, "items_extractor", ctx,
        )
        first_extracted_id = first["extracted_node_id"]

        second = items_service.run_operator(
            video_node.node_id, "items_extractor", ctx,
        )
        # Materialization is per-child, so the operator may have created
        # a second ExtractedNode on top of the first. The child RawNodes
        # are reused — that's what materialization's idempotency buys.
        for m in second["materialized_children"]:
            assert m["resumed"] is True
            assert m["edge_id"] is None

        # The graph still has only one IMAGE child per logical item —
        # no duplicates from the re-run.
        all_image_nodes = graph.list_raw_nodes(
            raw_type=RawDataType.IMAGE, ctx=ctx,
        )
        assert len(all_image_nodes) == 3

        # The original ExtractedNode still exists (no overwrite happened
        # to the children underneath).
        assert graph.get_extracted_node(first_extracted_id, ctx) is not None

    def test_operators_without_policy_are_unaffected(
        self, items_service, image_raw_node, ctx,
    ):
        """Sanity: an operator without a materialization policy (image_caption
        returns a single description, no items list) doesn't get the
        materialization treatment — materialized_children is empty and
        the parent ExtractedNode holds the full extracted_content as
        before the refactor."""
        result = items_service.run_operator(
            image_raw_node.node_id, "image_caption", ctx,
        )
        assert result["ok"] is True
        assert result["materialized_children"] == []
        extracted = items_service._graph.get_extracted_node(
            result["extracted_node_id"], ctx,
        )
        # Heavy content preserved — no policy means no strip.
        assert "caption" in extracted.extracted_content or "description" in extracted.extracted_content
